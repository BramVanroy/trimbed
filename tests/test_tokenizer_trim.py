import json

import pytest

from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import build_trimmed_model, trim_tokenizer


def keep_everything_used(tokenizer, spec, texts):
    """Ids needed to encode `texts`, plus everything structurally required."""
    from trimbed.config import SelectionConfig
    from trimbed.counting import CorpusCounts
    from trimbed.selection import select_tokens

    counts = CorpusCounts()
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        counts.counts.update(ids)
        counts.total_num_tokens += len(ids)
    return select_tokens(spec, counts, SelectionConfig(min_count=1)).kept_ids


def _added_tokens_by_content(tokenizer):
    """The tokenizer's added-token entries, keyed by their surface string."""
    document = json.loads(tokenizer.backend_tokenizer.to_str())
    return {entry["content"]: entry for entry in document["added_tokens"]}


@pytest.mark.parametrize("fixture_name", ["byte_level_bpe", "wordpiece", "unigram", "wordlevel"])
def test_every_family_trims_and_reloads(fixture_name, request, sample_texts):
    tokenizer = request.getfixturevalue(fixture_name)
    spec = TokenizerSpec.from_tokenizer(tokenizer)
    kept = keep_everything_used(tokenizer, spec, sample_texts)

    trimmed = trim_tokenizer(tokenizer, spec, kept)

    assert trimmed.spec.vocab_size == len(kept)
    assert trimmed.spec.vocab_size < spec.vocab_size
    assert len(trimmed.remap) == len(kept)


@pytest.mark.parametrize("fixture_name", ["byte_level_bpe", "wordpiece", "unigram", "wordlevel"])
def test_trimmed_ids_never_exceed_the_new_vocabulary(fixture_name, request, sample_texts):
    tokenizer = request.getfixturevalue(fixture_name)
    spec = TokenizerSpec.from_tokenizer(tokenizer)
    trimmed = trim_tokenizer(tokenizer, spec, keep_everything_used(tokenizer, spec, sample_texts))

    for text in sample_texts:
        # add_special_tokens exercises the post-processor, which hard-codes ids.
        ids = trimmed.tokenizer(text)["input_ids"]
        assert max(ids) < len(trimmed.tokenizer), (text, ids, len(trimmed.tokenizer))


@pytest.mark.parametrize("fixture_name", ["byte_level_bpe", "wordpiece", "unigram", "wordlevel"])
def test_text_still_decodes_to_itself(fixture_name, request, sample_texts):
    tokenizer = request.getfixturevalue(fixture_name)
    spec = TokenizerSpec.from_tokenizer(tokenizer)
    trimmed = trim_tokenizer(tokenizer, spec, keep_everything_used(tokenizer, spec, sample_texts))

    for text in sample_texts:
        original = tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"])
        after = trimmed.tokenizer.decode(trimmed.tokenizer(text, add_special_tokens=False)["input_ids"])
        assert after == original


def test_the_post_processor_id_is_remapped_not_left_stale(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed = trim_tokenizer(byte_level_bpe, spec, keep_everything_used(byte_level_bpe, spec, sample_texts))

    document = json.loads(trimmed.tokenizer.backend_tokenizer.to_str())
    embedded = document["post_processor"]["special_tokens"]["<|im_end|>"]["ids"]
    assert embedded == [trimmed.tokenizer.convert_tokens_to_ids("<|im_end|>")]
    assert embedded[0] < trimmed.spec.vocab_size


def test_special_tokens_keep_working_after_the_trim(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed = trim_tokenizer(byte_level_bpe, spec, keep_everything_used(byte_level_bpe, spec, sample_texts))

    assert trimmed.tokenizer.eos_token == "<|im_end|>"
    assert trimmed.tokenizer("hello")["input_ids"][-1] == trimmed.tokenizer.eos_token_id


def test_the_remap_carries_special_tokens_embeddings(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed = trim_tokenizer(byte_level_bpe, spec, keep_everything_used(byte_level_bpe, spec, sample_texts))

    # Added tokens must map back to their original rows, not be treated as new tokens
    # that would get random embeddings.
    old_id = byte_level_bpe.convert_tokens_to_ids("<|im_end|>")
    new_id = trimmed.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert trimmed.remap.to_old(new_id) == old_id


def test_added_token_matching_flags_survive_the_trim(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed = trim_tokenizer(byte_level_bpe, spec, keep_everything_used(byte_level_bpe, spec, sample_texts))

    # skeletoken adopts the pad token this fixture declares and re-registers it on every
    # removal. Doing that with the flags reset would make `<|endoftext|>` refuse to match
    # after a letter and eat the whitespace behind it, and a chat template puts both of
    # those in every prompt. Fixed upstream in 0.6.1; this holds it there.
    before = _added_tokens_by_content(byte_level_bpe)
    after = _added_tokens_by_content(trimmed.tokenizer)

    assert set(after) == set(before)
    for content, entry in after.items():
        for flag in ("single_word", "lstrip", "rstrip", "normalized", "special"):
            assert entry[flag] == before[content][flag], f"{content}.{flag} changed"


def test_a_special_token_still_matches_flush_against_a_word(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed = trim_tokenizer(byte_level_bpe, spec, keep_everything_used(byte_level_bpe, spec, sample_texts))

    text = "the<|endoftext|> at"
    original_ids = byte_level_bpe(text, add_special_tokens=False)["input_ids"]

    assert trimmed.remap.map_sequence(original_ids) == trimmed.tokenizer(text, add_special_tokens=False)["input_ids"]


def test_bpe_merges_referencing_removed_tokens_are_dropped(byte_level_bpe, sample_texts):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed_model = build_trimmed_model(spec, keep_everything_used(byte_level_bpe, spec, sample_texts))

    vocabulary = trimmed_model.vocabulary
    for left, right in trimmed_model.model.merges.root:
        assert left in vocabulary
        assert right in vocabulary
        assert left + right in vocabulary


def test_unigram_unk_is_renumbered_not_left_dangling(unigram, sample_texts):
    spec = TokenizerSpec.from_tokenizer(unigram)
    trimmed_model = build_trimmed_model(spec, keep_everything_used(unigram, spec, sample_texts))

    assert trimmed_model.model.unk_id < trimmed_model.vocabulary_size
    assert trimmed_model.unk_token == "<unk>"


def test_added_tokens_decoder_is_rewritten(byte_level_bpe, sample_texts, tmp_path):
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    trimmed = trim_tokenizer(byte_level_bpe, spec, keep_everything_used(byte_level_bpe, spec, sample_texts))
    trimmed.tokenizer.save_pretrained(tmp_path)

    config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    for raw_id in config.get("added_tokens_decoder", {}):
        assert int(raw_id) < trimmed.spec.vocab_size


def test_a_post_processor_token_survives_a_trim_that_never_asked_for_it(wordlevel_undeclared_post_processor):
    from trimbed.config import SelectionConfig
    from trimbed.selection import select_tokens

    tokenizer = wordlevel_undeclared_post_processor
    spec = TokenizerSpec.from_json_str(tokenizer.backend_tokenizer.to_str())
    kept = select_tokens(spec, None, SelectionConfig(keep_tokens=["the", "cat"])).kept_ids

    trimmed = trim_tokenizer(tokenizer, spec, kept)

    assert "<sep>" in trimmed.spec.vocabulary
    assert trimmed.tokenizer("the cat")["input_ids"][-1] == trimmed.spec.vocabulary["<sep>"]
