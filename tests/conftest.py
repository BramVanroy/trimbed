"""Shared fixtures: tiny tokenizers and models built offline.

Everything here is constructed in-process rather than downloaded, so the whole suite
runs without network access. The fixtures deliberately reproduce the awkward parts of
real tokenizers (added tokens above the base vocab, a post-processor with hard-coded
ids, a Unigram model with a non-zero `unk_id`) because those are where trimming
goes wrong.
"""

from __future__ import annotations

import logging

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from trimbed.bytelevel import bytes_to_unicode


BYTE_LEVEL_MERGES = [
    ("Ġ", "d"),
    ("Ġd", "e"),
    ("h", "e"),
    ("t", "he"),
    ("Ġ", "t"),
    ("Ġt", "he"),
    ("a", "t"),
    ("Ġ", "a"),
    ("Ġa", "t"),
]


def _wrap(tokenizer: Tokenizer, **kwargs: object) -> PreTrainedTokenizerFast:
    """Wrap a raw `tokenizers.Tokenizer` in the transformers fast-tokenizer API."""
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, **kwargs)


@pytest.fixture
def byte_level_bpe() -> PreTrainedTokenizerFast:
    """A byte-level BPE tokenizer with special tokens above the base vocab.

    Mirrors the shape of Qwen-style tokenizers: the 256 byte-alphabet characters plus a
    few merged tokens, `added_tokens` numbered after the base vocab, and a
    TemplateProcessing post-processor that hard-codes an id.
    """
    vocab = {char: index for index, char in enumerate(sorted(bytes_to_unicode().values()))}
    for left, right in BYTE_LEVEL_MERGES:
        merged = left + right
        vocab.setdefault(merged, len(vocab))

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=BYTE_LEVEL_MERGES))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    base_size = len(vocab)
    tokenizer.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|im_end|>",
        pair="$A $B",
        special_tokens=[("<|im_end|>", base_size + 2)],
    )
    return _wrap(tokenizer, eos_token="<|im_end|>", pad_token="<|endoftext|>")


CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\nyou are a cat<|im_end|>\n{% endif %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _add_word(word: str, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
    """Make `word` a single token, reachable through a left-to-right chain of merges.

    A merge rule only fires if both its halves are themselves reachable, so declaring
    `("a", "ssistant")` and stopping there would leave the token in the vocabulary and
    unreachable, which is the very failure the dependency closure exists to prevent.
    """
    prefix = word[0]
    vocab.setdefault(prefix, len(vocab))
    for char in word[1:]:
        vocab.setdefault(char, len(vocab))
        merges.append((prefix, char))
        prefix += char
        vocab.setdefault(prefix, len(vocab))


@pytest.fixture
def chat_bpe() -> PreTrainedTokenizerFast:
    """A byte-level BPE tokenizer that ships a chat template.

    The words the template works with (`assistant` as literal text, `system` through a
    comparison) are single tokens here and appear in no fixture corpus, so a corpus-driven
    trim drops them unless `keep_chat_template` pulls them back in. The template renders
    the same either way; only the id-level check notices.
    """
    vocab = {char: index for index, char in enumerate(sorted(bytes_to_unicode().values()))}
    merges = list(BYTE_LEVEL_MERGES)
    for left, right in BYTE_LEVEL_MERGES:
        vocab.setdefault(left + right, len(vocab))
    for word in ("user", "assistant", "system"):
        _add_word(word, vocab, merges)

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    return _wrap(tokenizer, eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=CHAT_TEMPLATE)


@pytest.fixture
def chat_messages() -> list[dict[str, str]]:
    """A short conversation the chat fixture can render."""
    return [{"role": "user", "content": "the cat"}, {"role": "assistant", "content": "the dog"}]


@pytest.fixture
def wordpiece() -> PreTrainedTokenizerFast:
    """A WordPiece tokenizer with the usual BERT special tokens inside the base vocab."""
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokens += list("abcdefghijklmnopqrstuvwxyz")
    tokens += ["the", "cat", "sat", "##ing", "##ed", "run", "dog", "##s"]
    vocab = {token: index for index, token in enumerate(tokens)}

    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=64))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]", vocab["[SEP]"]), cls=("[CLS]", vocab["[CLS]"]))
    return _wrap(tokenizer, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]")


@pytest.fixture
def unigram() -> PreTrainedTokenizerFast:
    """A Unigram tokenizer whose `unk_id` is deliberately not zero.

    Unigram addresses its unknown token by position, so any trim that removes an earlier
    entry has to renumber it. Putting it at index 2 makes that failure visible.
    """
    entries: list[tuple[str, float]] = [("▁the", -1.0), ("▁cat", -2.0), ("<unk>", 0.0)]
    entries += [(f"▁{word}", -3.0) for word in ("dog", "sat", "ran", "hat")]
    entries += [(char, -6.0) for char in "abcdefghijklmnopqrstuvwxyz▁"]

    tokenizer = Tokenizer(models.Unigram(vocab=entries, unk_id=2, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
    tokenizer.decoder = decoders.Metaspace(replacement="▁")
    return _wrap(tokenizer, unk_token="<unk>")


@pytest.fixture
def wordlevel() -> PreTrainedTokenizerFast:
    """A flat word-level tokenizer."""
    tokens = ["[UNK]", "the", "cat", "sat", "on", "mat", "dog", "ran", "fast"]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return _wrap(tokenizer, unk_token="[UNK]")


@pytest.fixture
def wordlevel_undeclared_post_processor() -> PreTrainedTokenizerFast:
    """A word-level tokenizer whose post-processor names a plain vocabulary entry.

    `<sep>` is referenced by the template but never declared as a special token, so
    nothing promotes it to an added token and it has to be recognised as structural on
    the strength of the post-processor alone.
    """
    tokens = ["[UNK]", "<sep>", "the", "cat", "sat", "on", "mat", "dog", "ran", "fast"]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <sep>",
        pair="$A <sep> $B",
        special_tokens=[("<sep>", vocab["<sep>"])],
    )
    return _wrap(tokenizer, unk_token="[UNK]")


class _StrictHandler(logging.Handler):
    """A handler that turns a log record it cannot format into a test failure.

    `logging` swallows a formatting error: it prints "--- Logging error ---" to stderr and
    carries on, so a call passing `%`-style arguments alongside an already-formatted
    f-string produces a traceback in every real run while every test still passes. Nothing
    else in the suite would notice, because `caplog` records `msg` without formatting it.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            record.getMessage()
        except TypeError as exc:
            raise AssertionError(f"log call cannot be formatted: {record.msg!r} % {record.args!r}") from exc


@pytest.fixture(autouse=True)
def strict_logging():
    """Format every record the package logs, so a broken log call fails the test.

    Autouse and at DEBUG, because the calls most likely to be wrong are the INFO ones a
    default configuration never even formats.
    """
    logger = logging.getLogger("trimbed")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    logger.handlers = [_StrictHandler()]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield
    logger.handlers, logger.propagate = handlers, propagate
    logger.setLevel(level)


@pytest.fixture
def package_logs(caplog):
    """Capture the package's own log records.

    `configure_logging` turns propagation off, and a script test earlier in the session
    may already have called it, so propagation is forced back on for the duration.
    """
    logger = logging.getLogger("trimbed")
    propagate = logger.propagate
    logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="trimbed"):
        yield caplog
    logger.propagate = propagate


@pytest.fixture
def sample_texts() -> list[str]:
    """Short English texts the tiny fixtures can encode."""
    return ["the cat sat", "the dog", "a hat", "the the the", "at the"]


@pytest.fixture
def corpus_dataset(sample_texts: list[str]):
    """An in-memory dataset standing in for a Hub dataset."""
    from datasets import Dataset

    return Dataset.from_dict({"text": sample_texts * 4})


@pytest.fixture
def tiny_model_factory():
    """Return a factory building tiny transformer models with configurable heads.

    Three shapes matter for embedding surgery and each is reachable from here: tied
    embeddings, untied embeddings with a separate `lm_head`, and an encoder with no
    output head at all.
    """
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertForMaskedLM, BertModel

    def build(vocab_size: int, *, tied: bool = True, with_head: bool = True):
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
            tie_word_embeddings=tied,
        )
        torch.manual_seed(0)
        model = BertForMaskedLM(config) if with_head else BertModel(config)
        model.eval()
        return model

    return build


@pytest.fixture
def tiny_seq2seq_factory():
    """Return a factory building a tiny encoder-decoder model.

    An encoder-decoder is the one shape that refuses to run on `input_ids` alone, and
    whose hidden states come back split per stack rather than as a single tensor.
    """
    torch = pytest.importorskip("torch")
    from transformers import T5Config, T5ForConditionalGeneration

    def build(vocab_size: int):
        config = T5Config(
            vocab_size=vocab_size,
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
        )
        torch.manual_seed(0)
        model = T5ForConditionalGeneration(config)
        model.eval()
        return model

    return build
