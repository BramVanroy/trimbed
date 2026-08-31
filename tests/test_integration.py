"""End-to-end checks against real Hub models.

These are the only tests that need network access, and the ones that load weights are
slow on top of that, so the two halves are asked for separately::

    uv run --all-extras pytest -m "network and not slow"   # the tokenizer trims
    uv run --all-extras pytest -m "network and slow"       # the model trims as well

The cases cover one checkpoint per shape the trim has to survive: a WordPiece encoder
whose masked-LM head carries a bias even though its weights are tied, a SentencePiece
encoder-decoder with a padded embedding matrix, a byte-level BPE causal LM with a chat
template, a large-vocabulary causal LM whose embedding matrix is padded well past the
end of its vocabulary, and a hybrid mamba/attention checkpoint whose chat template ships
as its own file. Each one is trimmed as a bare tokenizer and then run through the whole
pipeline, so the embedding surgery is measured against the original model's outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast

from trimbed.config import SelectionConfig, TrimConfig
from trimbed.counting import CorpusCounts
from trimbed.loading import load_tokenizer
from trimbed.pipeline import TrimPipeline
from trimbed.remap import IdRemap
from trimbed.report import CONFIG_FILENAME, REPORT_FILENAME, TrimReport
from trimbed.selection import select_tokens
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import trim_tokenizer
from trimbed.verify import verify_tokenizer


pytestmark = pytest.mark.network

TEXTS = [
    "De kat zat op de mat en keek naar buiten.",
    "Vocabulary trimming reduces the size of the embedding table.",
    "Het weer in Amsterdam is vandaag bewolkt met kans op regen.",
    "Machine learning models often have very large vocabularies.",
    "Zij loopt elke ochtend een half uur door het park.",
]

MESSAGES = [
    {"role": "user", "content": "Wat is de hoofdstad van Nederland?"},
    {"role": "assistant", "content": "Amsterdam is de hoofdstad van Nederland."},
]

# Token ids a checkpoint stores by name; every one of them has to follow the remap.
SPECIAL_TOKEN_IDS = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "cls_token_id", "sep_token_id")


@dataclass(frozen=True)
class ModelCase:
    """One Hub checkpoint and the shape it contributes to the matrix.

    Attributes:
        model_id: Hub id of the checkpoint.
        model_type: Tokenizer backend the spec is expected to report.
        uses_byte_level: Whether the tokenizer encodes over the byte alphabet.
        model_class: Class `resolve_model_class` is expected to load the weights as.
        has_chat_template: Whether the checkpoint ships a chat template.
        tolerance: Largest output difference the model comparison accepts.
    """

    model_id: str
    model_type: str
    uses_byte_level: bool
    model_class: str
    has_chat_template: bool
    tolerance: float = 1e-5


CASES = [
    # A masked-LM head whose decoder bias is its own parameter even though the weights
    # are tied to the input embedding, so tying does not carry the bias along.
    ModelCase("google-bert/bert-base-cased", "WordPiece", False, "BertForMaskedLM", False),
    # Encoder-decoder, and 32,128 embedding rows for 32,100 tokens: the padded matrix.
    # T5's activations run large enough that float32 noise exceeds the default tolerance.
    ModelCase("google-t5/t5-small", "Unigram", False, "T5ForConditionalGeneration", False, tolerance=1e-4),
    # Byte-level BPE with a chat template in tokenizer_config.json and a padding_idx.
    ModelCase("HuggingFaceTB/SmolLM2-135M-Instruct", "BPE", True, "LlamaForCausalLM", True),
    # 151,936 embedding rows for 151,669 tokens, so reading the row count off the config
    # picks up 267 rows that are not in the vocabulary at all.
    ModelCase("Qwen/Qwen3-0.6B", "BPE", True, "Qwen3ForCausalLM", True),
    # Mamba layers interleaved with attention, and a chat template that ships as its own
    # chat_template.jinja file. Its unk and pad tokens are the ones skeletoken re-registers
    # on every removal, so this is where their matching flags would get lost.
    ModelCase("ibm-granite/granite-4.0-h-350m", "BPE", True, "GraniteMoeHybridForCausalLM", True),
]

# Composite configs, whose token ids live on a `text_config` sub-config rather than at
# the top level, are not represented here: the checkpoints that have one are multimodal
# and far too large to run a CPU forward pass over. That shape is covered offline by
# `tests/test_model_trim.py`.

CASE_IDS = [case.model_id for case in CASES]
CHAT_CASES = [case for case in CASES if case.has_chat_template]
CHAT_CASE_IDS = [case.model_id for case in CHAT_CASES]


def _corpus_texts(tokenizer: PreTrainedTokenizerFast) -> list[str]:
    """Return the documents a case is trimmed against.

    A chat-capable tokenizer also gets its own rendered conversation, so the control
    tokens the template emits are part of the corpus and survive the selection.

    Args:
        tokenizer: The tokenizer the corpus is counted with.

    Returns:
        The corpus documents.
    """
    if not tokenizer.chat_template:
        return list(TEXTS)
    return [*TEXTS, tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)]


def _counts_for(tokenizer: PreTrainedTokenizerFast, texts: list[str]) -> CorpusCounts:
    """Count token occurrences over a handful of texts.

    Args:
        tokenizer: The tokenizer to encode with.
        texts: The documents to count.

    Returns:
        Counts carrying the same texts as verification samples.
    """
    counts = CorpusCounts()
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        counts.counts.update(ids)
        counts.total_num_tokens += len(ids)
    counts.num_documents = len(texts)
    counts.samples = list(texts)
    return counts


@dataclass
class TrimmedTokenizerCase:
    """What one case's tokenizer-only trim produced.

    Attributes:
        original: The tokenizer before trimming.
        tokenizer: The tokenizer after trimming.
        spec: A spec over the original tokenizer.
        trimmed_spec: A spec over the trimmed tokenizer.
        remap: The old-id to new-id mapping that was applied.
        texts: The corpus the selection was made from.
    """

    original: PreTrainedTokenizerFast
    tokenizer: PreTrainedTokenizerFast
    spec: TokenizerSpec
    trimmed_spec: TokenizerSpec
    remap: IdRemap
    texts: list[str]


@dataclass
class TrimmedCheckpoint:
    """What one case's full pipeline run produced.

    Attributes:
        output_dir: Directory the trimmed artefacts were written to.
        report: The run's report.
        original: The tokenizer before trimming.
        tokenizer: The trimmed tokenizer, reloaded from `output_dir`.
        remap: The mapping between the two vocabularies.
    """

    output_dir: Path
    report: TrimReport
    original: PreTrainedTokenizerFast
    tokenizer: PreTrainedTokenizerFast
    remap: IdRemap


@pytest.fixture(scope="module")
def trimmed_tokenizer():
    """Trim each case's tokenizer at most once, and share the result across the tests."""
    cache: dict[str, TrimmedTokenizerCase] = {}

    def get(case: ModelCase) -> TrimmedTokenizerCase:
        if case.model_id not in cache:
            original = load_tokenizer(case.model_id, trust_remote_code=True)
            spec = TokenizerSpec.from_tokenizer(original, source=case.model_id)
            texts = _corpus_texts(original)
            selection = select_tokens(spec, _counts_for(original, texts), SelectionConfig(min_count=1))
            result = trim_tokenizer(original, spec, selection.kept_ids)
            cache[case.model_id] = TrimmedTokenizerCase(
                original, result.tokenizer, spec, result.spec, result.remap, texts
            )
        return cache[case.model_id]

    return get


@pytest.fixture(scope="module")
def trimmed_checkpoint(tmp_path_factory):
    """Run the whole pipeline once per case, weights included, and share the artefacts."""
    cache: dict[str, TrimmedCheckpoint] = {}

    def get(case: ModelCase) -> TrimmedCheckpoint:
        if case.model_id not in cache:
            root = tmp_path_factory.mktemp(case.model_id.replace("/", "-"))
            original = load_tokenizer(case.model_id, trust_remote_code=True)
            corpus = root / "corpus"
            corpus.mkdir()
            texts = _corpus_texts(original)
            (corpus / "train.jsonl").write_text(
                "\n".join(json.dumps({"text": text}) for text in texts), encoding="utf-8"
            )
            config = TrimConfig.model_validate(
                {
                    "model": case.model_id,
                    "output_dir": str(root / "trimmed"),
                    "corpus": {"datasets": [{"path": str(corpus), "split": "train", "streaming": False}]},
                    "selection": {"min_count": 1},
                    "verify_model_samples": len(texts),
                    "verify_tolerance": case.tolerance,
                }
            )
            report = TrimPipeline(config).run()
            tokenizer = load_tokenizer(str(config.output_dir), trust_remote_code=True)
            cache[case.model_id] = TrimmedCheckpoint(
                config.output_dir,
                report,
                original,
                tokenizer,
                IdRemap.from_vocabularies(original.get_vocab(), tokenizer.get_vocab()),
            )
        return cache[case.model_id]

    return get


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_tokenizer_is_the_family_the_case_claims(case, trimmed_tokenizer):
    result = trimmed_tokenizer(case)

    assert result.spec.model_type == case.model_type
    assert result.spec.uses_byte_level is case.uses_byte_level
    assert result.spec.vocab_size == len(result.original)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_trimming_sheds_most_of_the_vocabulary_without_changing_tokenization(case, trimmed_tokenizer):
    result = trimmed_tokenizer(case)

    # A handful of sentences uses a vanishing fraction of any of these vocabularies.
    assert result.trimmed_spec.vocab_size < result.spec.vocab_size / 50
    verification = verify_tokenizer(result.original, result.tokenizer, result.remap, result.texts)
    assert verification.checked == len(result.texts)
    assert verification.identical == verification.checked


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_tokenizer_config_survives_the_trim(case, trimmed_tokenizer):
    result = trimmed_tokenizer(case)

    # `to_transformers()` would build a fresh tokenizer and lose all of this; the
    # save/reload staging in `trim_tokenizer` is what carries it over.
    assert result.tokenizer.chat_template == result.original.chat_template
    assert result.tokenizer.model_max_length == result.original.model_max_length
    assert result.tokenizer.special_tokens_map == result.original.special_tokens_map


@pytest.mark.parametrize("case", CHAT_CASES, ids=CHAT_CASE_IDS)
def test_the_chat_template_renders_the_same_conversation(case, trimmed_tokenizer):
    result = trimmed_tokenizer(case)

    rendered = result.original.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    assert result.tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True) == rendered

    # return_dict=False asks for the bare id list rather than a BatchEncoding.
    old_ids = result.original.apply_chat_template(MESSAGES, add_generation_prompt=True, return_dict=False)
    new_ids = result.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, return_dict=False)
    assert result.remap.map_sequence(old_ids) == new_ids


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_checkpoint_is_loaded_as_the_class_it_names(case, trimmed_checkpoint):
    checkpoint = trimmed_checkpoint(case)

    # `AutoModel` would return the base model and drop the head this class carries.
    assert checkpoint.report.model_trim is not None
    assert checkpoint.report.model_trim.model_class == case.model_class


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_embedding_table_shrinks_to_the_kept_vocabulary(case, trimmed_checkpoint):
    checkpoint = trimmed_checkpoint(case)
    model_trim = checkpoint.report.model_trim

    assert model_trim.new_embedding_rows == checkpoint.report.vocabulary.trimmed_size
    assert model_trim.new_embedding_rows < model_trim.old_embedding_rows
    assert model_trim.parameters_removed > 0
    assert len(checkpoint.tokenizer) == model_trim.new_embedding_rows


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_trimmed_model_reproduces_the_original_outputs(case, trimmed_checkpoint):
    checkpoint = trimmed_checkpoint(case)
    verification = checkpoint.report.model_verification

    # Only a forward pass proves the embedding rows followed the renumbering.
    assert verification is not None
    assert verification.skipped == 0
    assert verification.checked == checkpoint.report.corpus.documents
    assert verification.ok, f"max diff {verification.max_hidden_diff:g}/{verification.max_logit_diff}"


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_saved_checkpoint_keeps_its_special_token_ids(case, trimmed_checkpoint):
    checkpoint = trimmed_checkpoint(case)

    for attribute in SPECIAL_TOKEN_IDS:
        old_id = getattr(checkpoint.original, attribute, None)
        expected = None if old_id is None else checkpoint.remap.old_to_new[old_id]
        assert getattr(checkpoint.tokenizer, attribute, None) == expected


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_run_leaves_a_report_and_the_config_behind(case, trimmed_checkpoint):
    checkpoint = trimmed_checkpoint(case)

    assert checkpoint.report.verification.identical == checkpoint.report.verification.checked
    saved = TrimReport.model_validate_json((checkpoint.output_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert saved == checkpoint.report
    assert (checkpoint.output_dir / CONFIG_FILENAME).exists()


@pytest.mark.parametrize(
    ("model_id", "expected_type"),
    [
        ("FacebookAI/xlm-roberta-base", "Unigram"),
        # mT5 ships spiece.model only, with no tokenizer.json to read.
        ("google/mt5-small", "Unigram"),
    ],
)
def test_other_families_trim_too(model_id, expected_type):
    tokenizer = load_tokenizer(model_id)
    spec = TokenizerSpec.from_tokenizer(tokenizer, source=model_id)
    assert spec.model_type == expected_type

    counts = _counts_for(tokenizer, list(TEXTS))
    selection = select_tokens(spec, counts, SelectionConfig(min_count=1))
    trimmed = trim_tokenizer(tokenizer, spec, selection.kept_ids)

    assert trimmed.spec.vocab_size < spec.vocab_size
    for text in TEXTS:
        ids = trimmed.tokenizer(text)["input_ids"]
        assert max(ids) < len(trimmed.tokenizer)
        decoded = trimmed.tokenizer.decode(trimmed.tokenizer(text, add_special_tokens=False)["input_ids"])
        assert decoded == tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"])
