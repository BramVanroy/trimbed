import json

import pytest

from trimbed.bytelevel import byte_level_alphabet, bytes_to_unicode, decode_byte_level, unicode_to_bytes
from trimbed.spec import TokenizerSpec


def test_bytes_to_unicode_is_a_bijection_over_all_256_bytes():
    mapping = bytes_to_unicode()
    assert len(mapping) == 256
    assert len(set(mapping.values())) == 256
    assert unicode_to_bytes() == {char: byte for byte, char in mapping.items()}
    assert byte_level_alphabet() == frozenset(mapping.values())


def test_decode_byte_level_undoes_the_mapping():
    assert decode_byte_level("Ġde") == " de"
    assert decode_byte_level("the") == "the"


def test_decode_byte_level_rejects_partial_and_foreign_sequences():
    # A lone continuation byte is not valid UTF-8 on its own.
    partial = bytes_to_unicode()[0xA9]
    assert decode_byte_level(partial) is None
    assert decode_byte_level("→ not in the alphabet") is None


def test_byte_level_spec_reports_its_shape(byte_level_bpe):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe, source="fixture")
    summary = spec.describe()
    assert summary["model_type"] == "BPE"
    assert summary["uses_byte_level"] is True
    assert summary["special_tokens"] == 3
    assert summary["source"] == "fixture"


def test_added_tokens_are_part_of_one_id_space(byte_level_bpe):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    # This is the Qwen-style trap: specials sit above the base vocab, but the spec must
    # present a single contiguous id space covering both.
    assert spec.vocab_size == len(byte_level_bpe)
    assert spec.added_token_ids == spec.special_token_ids
    assert all(token_id in spec.id_to_token for token_id in spec.added_token_ids)
    assert sorted(spec.vocabulary.values()) == list(range(spec.vocab_size))


def test_surface_forms_decode_byte_level_tokens(byte_level_bpe):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    assert spec.surface_forms["Ġthe"] == " the"
    assert spec.surface_forms["the"] == "the"


def test_surface_forms_strip_wordpiece_prefixes(wordpiece):
    spec = TokenizerSpec.from_tokenizer(wordpiece)
    assert spec.uses_byte_level is False
    assert spec.surface_forms["##ing"] == "ing"
    assert spec.surface_forms["cat"] == "cat"


def test_surface_forms_undo_metaspace(unigram):
    spec = TokenizerSpec.from_tokenizer(unigram)
    assert spec.surface_forms["▁cat"] == " cat"


def test_metaspace_is_undone_from_inside_a_pre_tokenizer_sequence():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "▁cat": 1}, unk_token="[UNK]"))
    # Converted SentencePiece tokenizers nest Metaspace inside a Sequence, which
    # skeletoken's `initial_subword_prefix` has to descend into to find the replacement.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace(replacement="▁")]
    )
    spec = TokenizerSpec.from_tokenizer(PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]"))

    assert spec.surface_forms["▁cat"] == " cat"


def test_unk_token_is_exposed(wordpiece, unigram, byte_level_bpe):
    assert TokenizerSpec.from_tokenizer(wordpiece).unk_token == "[UNK]"
    assert TokenizerSpec.from_tokenizer(unigram).unk_token == "<unk>"
    assert TokenizerSpec.from_tokenizer(byte_level_bpe).unk_token is None


def test_a_non_fast_tokenizer_is_refused():
    with pytest.raises(ValueError, match="fast"):
        TokenizerSpec.from_tokenizer(object())


def test_from_json_str_round_trips(byte_level_bpe):
    payload = byte_level_bpe.backend_tokenizer.to_str()
    spec = TokenizerSpec.from_json_str(payload, source="inline")
    assert spec.model_type == "BPE"
    assert spec.source == "inline"


def test_from_path_reads_a_tokenizer_json(byte_level_bpe, tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(byte_level_bpe.backend_tokenizer.to_str(), encoding="utf-8")
    spec = TokenizerSpec.from_path(path)
    assert spec.model_type == "BPE"
    assert spec.max_id == spec.vocab_size - 1


def test_post_processor_tokens_are_found_even_when_nothing_declared_them(wordlevel_undeclared_post_processor):
    spec = TokenizerSpec.from_json_str(wordlevel_undeclared_post_processor.backend_tokenizer.to_str())
    assert spec.post_processor_token_ids == frozenset({spec.vocabulary["<sep>"]})
    assert spec.vocabulary["<sep>"] not in spec.added_token_ids


def test_a_tokenizer_without_a_post_processor_has_no_post_processor_tokens(wordlevel):
    spec = TokenizerSpec.from_tokenizer(wordlevel)
    assert spec.post_processor_token_ids == frozenset()


def test_a_post_processor_naming_a_missing_token_is_ignored_not_fatal(wordlevel_undeclared_post_processor):
    document = json.loads(wordlevel_undeclared_post_processor.backend_tokenizer.to_str())
    document["post_processor"]["special_tokens"]["<sep>"]["tokens"] = ["<ghost>"]

    spec = TokenizerSpec.from_json_str(json.dumps(document))

    # The document is already broken; reading it should not add a KeyError of our own.
    assert spec.post_processor_token_ids == frozenset()
