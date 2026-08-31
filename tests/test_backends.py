import pytest

from trimbed.backends import get_backend, register_backend, supported_model_types
from trimbed.backends.base import VocabBackend
from trimbed.bytelevel import byte_level_alphabet
from trimbed.spec import TokenizerSpec


def test_every_builtin_family_is_registered():
    assert set(supported_model_types()) == {"BPE", "WordPiece", "Unigram", "WordLevel"}


def test_unknown_model_types_name_the_supported_ones():
    with pytest.raises(KeyError, match="BPE"):
        get_backend("SomeFutureModel")


def test_registering_without_a_model_type_is_rejected():
    with pytest.raises(ValueError, match="model_type"):

        @register_backend
        class Nameless(VocabBackend):
            pass


def test_registering_a_duplicate_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register_backend
        class Duplicate(VocabBackend):
            model_type = "BPE"


def test_a_new_family_needs_only_a_decorated_class():
    @register_backend
    class Fictional(VocabBackend):
        model_type = "FictionalModel"

    assert isinstance(get_backend("FictionalModel"), Fictional)
    assert "FictionalModel" in supported_model_types()


def test_bpe_protects_the_whole_byte_alphabet(byte_level_bpe):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    structural = spec.backend.structural_tokens(spec)
    # Without every byte-level character, some byte sequences become unencodable.
    assert byte_level_alphabet() <= structural


def test_bpe_dependencies_follow_the_merge_chain(byte_level_bpe):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    dependencies = spec.backend.dependencies(spec)
    vocabulary = spec.vocabulary

    # "the" is built by merging "t" with "he", which is itself built from "h" and "e".
    assert dependencies[vocabulary["the"]] == (vocabulary["t"], vocabulary["he"])
    assert dependencies[vocabulary["he"]] == (vocabulary["h"], vocabulary["e"])
    # Single characters are atomic.
    assert vocabulary["t"] not in dependencies


def test_wordpiece_has_no_dependencies_but_protects_unk(wordpiece):
    spec = TokenizerSpec.from_tokenizer(wordpiece)
    assert spec.backend.dependencies(spec) == {}
    assert spec.backend.structural_tokens(spec) == {"[UNK]"}


def test_unigram_has_no_dependencies_but_protects_unk(unigram):
    spec = TokenizerSpec.from_tokenizer(unigram)
    assert spec.backend.dependencies(spec) == {}
    assert spec.backend.structural_tokens(spec) == {"<unk>"}


def test_a_bpe_without_byte_level_protects_only_the_unk_token():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # Not every BPE tokenizer is byte-level; a character-level one has no byte alphabet
    # to protect, but its merge dependencies matter just as much.
    vocab = {"<unk>": 0, "t": 1, "h": 2, "e": 3, "he": 4, "the": 5}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[("h", "e"), ("t", "he")], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    spec = TokenizerSpec.from_tokenizer(PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>"))

    assert spec.uses_byte_level is False
    assert spec.backend.structural_tokens(spec) == {"<unk>"}
    assert spec.backend.dependencies(spec)[vocab["the"]] == (vocab["t"], vocab["he"])


def test_a_merge_referring_outside_the_vocabulary_is_ignored():
    import json

    from tokenizers import Tokenizer, models

    vocab = {"<unk>": 0, "t": 1, "h": 2, "e": 3, "he": 4, "the": 5}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[("h", "e"), ("t", "he")], unk_token="<unk>"))
    document = json.loads(tokenizer.to_str())
    # `tokenizers` refuses to load this, but a hand-edited or partially trimmed
    # tokenizer.json can still carry it, and reading one must not blow up.
    document["model"]["merges"].append(["x", "y"])
    spec = TokenizerSpec.from_json_str(json.dumps(document))

    assert spec.backend.dependencies(spec) == {4: (2, 3), 5: (1, 4)}


def test_byte_fallback_tokens_are_structural():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    # With byte fallback on, the <0xNN> tokens are the escape hatch for anything the
    # vocabulary cannot spell; dropping one makes those inputs unencodable.
    entries = [("<unk>", 0.0), ("▁cat", -1.0)] + [(f"<0x{byte:02X}>", -10.0) for byte in range(256)]
    tokenizer = Tokenizer(models.Unigram(vocab=entries, unk_id=0, byte_fallback=True))
    spec = TokenizerSpec.from_tokenizer(PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>"))

    structural = spec.backend.structural_tokens(spec)
    assert {f"<0x{byte:02X}>" for byte in range(256)} <= structural
    assert "▁cat" not in structural


def test_wordlevel_protects_unk(wordlevel):
    spec = TokenizerSpec.from_tokenizer(wordlevel)
    assert spec.model_type == "WordLevel"
    assert spec.backend.structural_tokens(spec) == {"[UNK]"}


def test_backend_repr_names_the_model_type(wordpiece):
    spec = TokenizerSpec.from_tokenizer(wordpiece)
    assert "WordPiece" in repr(spec.backend)
