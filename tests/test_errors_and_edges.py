"""Error paths and edge cases that the happy-path tests never reach."""

import json

import pytest

from trimbed.config import SelectionConfig
from trimbed.counting import CorpusCounts
from trimbed.exceptions import MissingDependencyError
from trimbed.remap import IdRemap
from trimbed.selection import select_tokens
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import _rewrite_added_tokens_decoder, trim_tokenizer


class TestMissingDependency:
    def test_the_message_names_the_extra_to_install(self):
        error = MissingDependencyError("torch", "model", "Trimming embeddings")

        assert isinstance(error, ImportError)
        assert "pip install 'trimbed[model]'" in str(error)
        assert "Trimming embeddings" in str(error)
        assert error.package == "torch"
        assert error.extra == "model"

    def test_require_torch_explains_itself_when_torch_is_absent(self, monkeypatch):
        import builtins

        from trimbed.loading import require_torch

        real_import = builtins.__import__

        def fail_on_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_torch)
        with pytest.raises(MissingDependencyError, match=r"trimbed\[model\]"):
            require_torch()


class TestLoading:
    @pytest.mark.parametrize(
        ("message", "expected_extra_hint"),
        [("No module named 'sentencepiece'", "sentencepiece"), ("cannot import protobuf", "protobuf")],
    )
    def test_a_conversion_without_its_dependencies_names_the_extra(self, monkeypatch, message, expected_extra_hint):
        from trimbed.loading import load_tokenizer

        def raise_import_error(cls, *args, **kwargs):
            raise ImportError(message)

        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", classmethod(raise_import_error))
        with pytest.raises(MissingDependencyError, match=r"trimbed\[convert\]") as caught:
            load_tokenizer("some/model")

        assert caught.value.package == expected_extra_hint

    def test_a_slow_tokenizer_is_refused_with_a_clear_reason(self, monkeypatch):
        from trimbed.loading import load_tokenizer

        class SlowTokenizer:
            is_fast = False

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", classmethod(lambda cls, *a, **k: SlowTokenizer())
        )
        with pytest.raises(ValueError, match="slow tokenizer"):
            load_tokenizer("some/model")

    @pytest.mark.torch
    def test_a_dtype_can_be_requested(self, tiny_model_factory, tmp_path, monkeypatch):
        import torch

        from trimbed.config import EmbeddingTrimConfig
        from trimbed.loading import load_model

        tiny_model_factory(16, with_head=False).save_pretrained(tmp_path)
        loaded = load_model(str(tmp_path), config=EmbeddingTrimConfig(dtype="float16"))
        assert loaded.get_input_embeddings().weight.dtype is torch.float16

    @pytest.mark.torch
    def test_without_a_config_the_model_keeps_its_own_dtype(self, tiny_model_factory, tmp_path):
        import torch

        from trimbed.loading import load_model

        tiny_model_factory(16, with_head=False).save_pretrained(tmp_path)
        loaded = load_model(str(tmp_path))

        assert loaded.get_input_embeddings().weight.dtype is torch.float32
        assert loaded.training is False


class TestSelectionEdges:
    def test_structural_tokens_absent_from_the_vocabulary_are_skipped(self, wordpiece):
        from trimbed.backends.base import VocabBackend

        class Absentminded(VocabBackend):
            model_type = "Absentminded"

            def structural_tokens(self, spec):
                return {"[UNK]", "<not-in-this-vocabulary>"}

        spec = TokenizerSpec.from_tokenizer(wordpiece)
        # A third-party backend may name a token this particular checkpoint lacks; that
        # is not fatal, the token simply cannot be protected.
        spec.__dict__["backend"] = Absentminded()

        selection = select_tokens(spec, None, SelectionConfig(keep_tokens=["cat"]))

        assert spec.vocabulary["[UNK]"] in selection.structural_ids
        # The added tokens are structural regardless; the invented one contributes no id.
        assert selection.structural_ids == set(spec.added_token_ids)

    def test_explicit_ids_are_recorded_as_such(self, wordpiece):
        spec = TokenizerSpec.from_tokenizer(wordpiece)
        target = spec.vocabulary["cat"]

        selection = select_tokens(spec, None, SelectionConfig(keep_token_ids=[target]))

        assert "explicit_id" in selection.provenance[target]

    def test_a_cap_at_or_above_the_selection_changes_nothing(self, wordpiece):
        spec = TokenizerSpec.from_tokenizer(wordpiece)
        uncapped = select_tokens(spec, None, SelectionConfig(keep_tokens=["cat"]))
        capped = select_tokens(spec, None, SelectionConfig(keep_tokens=["cat"], max_vocab_size=len(uncapped) + 10))

        assert capped.kept_ids == uncapped.kept_ids

    def test_a_corpus_with_no_counts_contributes_nothing(self, wordpiece):
        spec = TokenizerSpec.from_tokenizer(wordpiece)
        selection = select_tokens(spec, CorpusCounts(), SelectionConfig(keep_tokens=["cat"]))

        assert selection.kept_ids == selection.structural_ids | {spec.vocabulary["cat"]}

    def test_capping_a_bpe_selection_frees_parents_as_children_go(self, byte_level_bpe):
        spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
        counts = CorpusCounts()
        ids = byte_level_bpe("the cat sat at the dog", add_special_tokens=False)["input_ids"]
        counts.counts.update(ids)
        counts.total_num_tokens = len(ids)

        uncapped = select_tokens(spec, counts, SelectionConfig(min_count=1))
        # Cap tightly enough that "the" must go, which then frees "he" to be dropped too.
        capped = select_tokens(spec, counts, SelectionConfig(min_count=1, max_vocab_size=len(uncapped) - 4))

        assert len(capped) == len(uncapped) - 4
        dependencies = spec.backend.dependencies(spec)
        for token_id in capped.kept_ids:
            for parent in dependencies.get(token_id, ()):
                assert parent in capped.kept_ids


class TestPresetEdges:
    def test_ascii_printable_covers_more_than_letters(self, wordpiece):
        from trimbed.presets import resolve_preset

        spec = TokenizerSpec.from_tokenizer(wordpiece)
        printable = resolve_preset("ascii_printable", spec)

        assert "cat" in printable
        assert "[UNK]" in printable

    def test_script_skips_tokens_with_no_letters_at_all(self, byte_level_bpe):
        from trimbed.presets import resolve_preset

        spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
        latin = resolve_preset("script:Latin", spec)

        assert "the" in latin
        # Punctuation belongs to no script, so it must not be swept in.
        assert "!" not in latin


class TestTokenizerTrimEdges:
    def test_rewriting_a_missing_config_is_a_no_op(self, tmp_path):
        _rewrite_added_tokens_decoder(tmp_path / "absent.json", IdRemap.from_kept([0, 1]))

    def test_a_config_without_the_decoder_map_is_left_alone(self, tmp_path):
        path = tmp_path / "tokenizer_config.json"
        path.write_text(json.dumps({"model_max_length": 512}), encoding="utf-8")

        _rewrite_added_tokens_decoder(path, IdRemap.from_kept([0, 1]))

        assert json.loads(path.read_text(encoding="utf-8")) == {"model_max_length": 512}

    def test_the_decoder_map_is_remapped_and_pruned(self, tmp_path):
        path = tmp_path / "tokenizer_config.json"
        path.write_text(
            json.dumps({"added_tokens_decoder": {"5": {"content": "<a>"}, "9": {"content": "<gone>"}}}),
            encoding="utf-8",
        )

        _rewrite_added_tokens_decoder(path, IdRemap.from_kept([1, 5]))

        decoder = json.loads(path.read_text(encoding="utf-8"))["added_tokens_decoder"]
        assert decoder == {"1": {"content": "<a>"}}

    def test_a_reload_mismatch_is_reported(self, wordpiece, monkeypatch, sample_texts):
        spec = TokenizerSpec.from_tokenizer(wordpiece)
        kept = select_tokens(spec, None, SelectionConfig(keep_tokens=["cat"])).kept_ids
        # Pretend the reload silently re-registered tokens at their old ids.
        monkeypatch.setattr(TokenizerSpec, "vocab_size", property(lambda self: 99999))
        with pytest.raises(RuntimeError, match="remap expected"):
            trim_tokenizer(wordpiece, spec, kept)


class TestVerifyFailures:
    def test_mismatched_text_is_recorded_as_a_failure(self, wordpiece, byte_level_bpe):
        from trimbed.verify import verify_tokenizer

        # The WordPiece fixture cannot represent these words and emits [UNK], while the
        # byte-level tokenizer reproduces them exactly, which is a guaranteed decode mismatch.
        remap = IdRemap.from_kept(range(len(wordpiece)))
        result = verify_tokenizer(wordpiece, byte_level_bpe, remap, ["quantum entanglement"] * 8)

        assert not result.ok
        assert 0 < len(result.failures) <= 5


@pytest.mark.torch
class TestModelTrimEdges:
    def test_list_valued_token_ids_are_remapped(self, tiny_model_factory):
        from trimbed.model_trim import trim_model

        model = tiny_model_factory(20)
        remap = IdRemap.from_kept([0, 2, 4, 6, 8])
        model.config.eos_token_id = [2, 6, 19]  # 19 does not survive

        trim_model(model, remap)

        assert model.config.eos_token_id == [remap.to_new(2), remap.to_new(6)]

    def test_the_generation_config_is_remapped_too(self, tiny_model_factory):
        from transformers import GenerationConfig

        from trimbed.model_trim import trim_model

        model = tiny_model_factory(20)
        remap = IdRemap.from_kept([0, 2, 4, 6, 8])
        # BertForMaskedLM carries none by default; generative checkpoints do, and a stale
        # id there would silently pad with the wrong token.
        model.generation_config = GenerationConfig(pad_token_id=4, eos_token_id=19)

        trim_model(model, remap)

        assert model.generation_config.pad_token_id == remap.to_new(4)
        assert model.generation_config.eos_token_id is None
