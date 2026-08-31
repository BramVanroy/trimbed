import pytest

from trimbed.config import SelectionConfig
from trimbed.counting import CorpusCounts
from trimbed.model_trim import trim_model
from trimbed.report import VerificationReport
from trimbed.selection import select_tokens
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import trim_tokenizer
from trimbed.verify import verify_model, verify_tokenizer


def _trim_for(tokenizer, texts, config=None):
    spec = TokenizerSpec.from_tokenizer(tokenizer)
    counts = CorpusCounts()
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        counts.counts.update(ids)
        counts.total_num_tokens += len(ids)
    selection = select_tokens(spec, counts, config or SelectionConfig(min_count=1))
    return trim_tokenizer(tokenizer, spec, selection.kept_ids)


def test_a_lossless_trim_verifies_exactly(byte_level_bpe, sample_texts):
    trimmed = _trim_for(byte_level_bpe, sample_texts)
    result = verify_tokenizer(byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts)

    assert result.checked == len(sample_texts)
    assert result.identical == result.checked
    assert result.exact_rate == 1.0
    assert result.ok


def test_an_aggressive_trim_still_decodes_to_the_same_text(byte_level_bpe, sample_texts):
    # Capping hard forces merges away, so the ids differ but the text must not.
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    structural = len(select_tokens(spec, None, SelectionConfig(keep_tokens=[])).structural_ids)
    trimmed = _trim_for(byte_level_bpe, sample_texts, SelectionConfig(min_count=1, max_vocab_size=structural + 3))

    result = verify_tokenizer(byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts)

    assert result.identical < result.checked
    assert result.equivalent_text == result.checked
    assert result.text_rate == 1.0
    assert result.ok


def test_failures_are_captured_and_bounded(byte_level_bpe, sample_texts):
    trimmed = _trim_for(byte_level_bpe, sample_texts)
    # Text the trimmed vocabulary was never meant to cover is not a fair comparison,
    # but it must be reported rather than crash.
    result = verify_tokenizer(byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts)
    assert isinstance(result, VerificationReport)
    assert len(result.failures) <= 5


def test_an_empty_sample_set_is_vacuously_fine(byte_level_bpe, sample_texts):
    trimmed = _trim_for(byte_level_bpe, sample_texts)
    result = verify_tokenizer(byte_level_bpe, trimmed.tokenizer, trimmed.remap, [])

    assert result.checked == 0
    assert result.exact_rate == 1.0
    assert result.text_rate == 1.0
    assert result.length_ratio == 1.0
    assert result.ok


def test_dropped_merges_show_up_as_a_longer_encoding(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    structural = len(select_tokens(spec, None, SelectionConfig(keep_tokens=[])).structural_ids)
    trimmed = _trim_for(byte_level_bpe, sample_texts, SelectionConfig(min_count=1, max_vocab_size=structural + 3))

    result = verify_tokenizer(byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts)

    # Losing merges costs sequence length even though the text still round-trips.
    assert result.length_ratio > 1.0
    assert result.trimmed_tokens > result.original_tokens


class TestModelVerification:
    """Running both models over the same texts, which is what proves the gather index."""

    @pytest.fixture
    def trimmed_pair(self, byte_level_bpe, sample_texts, tiny_model_factory):
        """An original and a correctly trimmed model, plus their tokenizers."""
        pytest.importorskip("torch")
        trimmed = _trim_for(byte_level_bpe, sample_texts)
        original_model = tiny_model_factory(len(byte_level_bpe), tied=False)

        import copy

        trimmed_model = copy.deepcopy(original_model)
        trim_model(trimmed_model, trimmed.remap)
        return original_model, trimmed_model, trimmed

    @pytest.mark.torch
    def test_a_faithful_trim_reproduces_the_original(self, byte_level_bpe, sample_texts, trimmed_pair):
        original_model, trimmed_model, trimmed = trimmed_pair

        result = verify_model(
            original_model, trimmed_model, byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts
        )

        assert result.checked == len(sample_texts)
        assert result.skipped == 0
        assert result.max_logit_diff is not None
        assert result.ok

    @pytest.mark.torch
    def test_a_wrong_gather_index_is_caught(self, byte_level_bpe, sample_texts, trimmed_pair):
        import torch

        original_model, trimmed_model, trimmed = trimmed_pair
        # Exactly the bug the tokenizer check cannot see: the vocabulary is right, the
        # rows behind it are not.
        with torch.no_grad():
            embeddings = trimmed_model.get_input_embeddings().weight
            embeddings.data = embeddings.data.flip(0)

        result = verify_model(
            original_model, trimmed_model, byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts
        )

        assert not result.ok
        assert result.max_hidden_diff > 1e-5

    @pytest.mark.torch
    def test_an_encoder_without_a_head_compares_hidden_states_only(
        self, byte_level_bpe, sample_texts, tiny_model_factory
    ):
        import copy

        trimmed = _trim_for(byte_level_bpe, sample_texts)
        original_model = tiny_model_factory(len(byte_level_bpe), with_head=False)
        trimmed_model = copy.deepcopy(original_model)
        trim_model(trimmed_model, trimmed.remap)

        result = verify_model(
            original_model, trimmed_model, byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts
        )

        assert result.max_logit_diff is None
        assert result.ok

    @pytest.mark.torch
    def test_texts_that_no_longer_map_one_to_one_are_skipped(
        self, byte_level_bpe, sample_texts, tiny_model_factory, trimmed_pair
    ):
        original_model, _, _ = trimmed_pair
        spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
        structural = len(select_tokens(spec, None, SelectionConfig(keep_tokens=[])).structural_ids)
        capped = _trim_for(byte_level_bpe, sample_texts, SelectionConfig(min_count=1, max_vocab_size=structural + 3))
        capped_model = tiny_model_factory(len(byte_level_bpe), tied=False)
        trim_model(capped_model, capped.remap)

        result = verify_model(
            original_model, capped_model, byte_level_bpe, capped.tokenizer, capped.remap, sample_texts
        )

        assert result.skipped > 0

    @pytest.fixture
    def trimmed_seq2seq_pair(self, byte_level_bpe, sample_texts, tiny_seq2seq_factory):
        """An original and a correctly trimmed encoder-decoder, plus their tokenizers."""
        pytest.importorskip("torch")
        import copy

        trimmed = _trim_for(byte_level_bpe, sample_texts)
        original_model = tiny_seq2seq_factory(len(byte_level_bpe))
        trimmed_model = copy.deepcopy(original_model)
        trim_model(trimmed_model, trimmed.remap)
        return original_model, trimmed_model, trimmed

    @pytest.mark.torch
    def test_an_encoder_decoder_runs_one_decoder_step(self, byte_level_bpe, sample_texts, trimmed_seq2seq_pair):
        original_model, trimmed_model, trimmed = trimmed_seq2seq_pair

        result = verify_model(
            original_model, trimmed_model, byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts
        )

        # Without that step a seq2seq refuses to run at all, and its output head (the
        # half the trim actually resizes) would never be compared.
        assert result.checked == len(sample_texts)
        assert result.max_logit_diff is not None
        assert result.ok

    @pytest.mark.torch
    def test_an_encoder_decoder_without_a_start_token_is_refused(
        self, byte_level_bpe, sample_texts, trimmed_seq2seq_pair
    ):
        original_model, trimmed_model, trimmed = trimmed_seq2seq_pair
        original_model.config.decoder_start_token_id = None

        with pytest.raises(ValueError, match="decoder_start_token_id"):
            verify_model(original_model, trimmed_model, byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts)

    @pytest.mark.torch
    def test_a_model_with_nothing_to_compare_is_refused(self, byte_level_bpe, sample_texts, trimmed_pair):
        import torch

        _, trimmed_model, trimmed = trimmed_pair

        class Opaque(torch.nn.Module):
            """A model whose output exposes neither hidden states nor logits."""

            def forward(self, **kwargs):
                return torch.nn.utils.rnn.PackedSequence  # any object without the attributes

        opaque = Opaque()
        opaque.weight = torch.nn.Parameter(torch.zeros(1))

        with pytest.raises(ValueError, match="nothing to compare"):
            verify_model(opaque, trimmed_model, byte_level_bpe, trimmed.tokenizer, trimmed.remap, sample_texts)
