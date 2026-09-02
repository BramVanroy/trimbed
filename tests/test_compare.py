"""Diffing two tokenizers.

The interesting cases are the ones a user reaches for the command with: a real trim (a
subset, renumbered in place), a tokenizer that only looks like one (reordered, or with
tokens the original never had), and a trim that quietly dropped something structural.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from trimbed.compare import (
    MIXED,
    NON_LETTER,
    UNDECODABLE,
    EncodingDiff,
    GroupDiff,
    VocabularyDiff,
    compare_tokenizers,
)
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import trim_tokenizer


MULTILINGUAL_TOKENS = ["[UNK]", "the", "cat", "кот", "кошка", "γάτα", "catж", "123", "!!", "@#"]


def word_level(tokens: list[str], post_processor: bool = False) -> PreTrainedTokenizerFast:
    """Build a flat word-level tokenizer over exactly these tokens, in this order."""
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if post_processor:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A [SEP]", pair="$A $B", special_tokens=[("[SEP]", vocab["[SEP]"])]
        )
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def spec_of(tokenizer: PreTrainedTokenizerFast, source: str) -> TokenizerSpec:
    """A spec over a tokenizer, labelled so the report has something to print."""
    return TokenizerSpec.from_tokenizer(tokenizer, source=source)


@pytest.fixture
def trimmed_bpe(byte_level_bpe):
    """The byte-level BPE fixture trimmed to its structural floor.

    That drops every merged token, which is what makes the trim visible in the encoding
    comparison: the survivors are the byte alphabet and the added tokens, so text still
    encodes but one character at a time.
    """
    spec = TokenizerSpec.from_tokenizer(byte_level_bpe, source="base")
    return spec, trim_tokenizer(byte_level_bpe, spec, spec.structural_ids).spec


def test_a_trim_is_a_subset_renumbered_in_place(trimmed_bpe):
    base, trimmed = trimmed_bpe

    report = compare_tokenizers(base, trimmed)

    assert report.vocabulary.is_subset
    assert report.vocabulary.ids_contiguous
    assert report.vocabulary.order_preserved
    assert report.vocabulary.introduced == 0
    assert report.vocabulary.removed == base.vocab_size - trimmed.vocab_size
    assert report.vocabulary.removed_fraction > 0
    assert not report.components.structural_break


def test_comparing_a_tokenizer_with_itself_reports_no_difference(wordpiece):
    spec = spec_of(wordpiece, "self")

    report = compare_tokenizers(spec, spec, texts=["the cat sat"])

    assert report.vocabulary.removed == 0
    assert report.vocabulary.shared == spec.vocab_size
    assert all(preset.removed == 0 for preset in report.profile.presets)
    assert report.profile.removed_by_decile == [0] * 10
    assert report.encoding.identical == report.encoding.checked == 1
    assert report.encoding.length_ratio == 1.0
    assert "presets cut" not in report.render()


def test_a_reordered_vocabulary_is_not_a_trim():
    base = spec_of(word_level(MULTILINGUAL_TOKENS), "base")
    shuffled = spec_of(word_level(list(reversed(MULTILINGUAL_TOKENS))), "shuffled")

    report = compare_tokenizers(base, shuffled)

    assert report.vocabulary.is_subset
    assert report.vocabulary.ids_contiguous
    assert not report.vocabulary.order_preserved
    assert "reordered" in report.render()


def test_tokens_the_original_never_had_are_called_out():
    base = spec_of(word_level(MULTILINGUAL_TOKENS), "base")
    other = spec_of(word_level(["[UNK]", "the", "hond", "kat"]), "other")

    report = compare_tokenizers(base, other, examples=2)

    assert not report.vocabulary.is_subset
    assert report.vocabulary.introduced == 2
    assert report.vocabulary.introduced_examples == ["hond", "kat"]
    assert "not a subset" in report.render()


def test_a_dropped_post_processor_token_is_a_structural_break(package_logs):
    base = spec_of(word_level(["[UNK]", "[SEP]", "the", "cat"], post_processor=True), "base")
    other = spec_of(word_level(["[UNK]", "the"]), "other")

    report = compare_tokenizers(base, other)

    assert report.components.structural_break
    assert report.components.removed_post_processor_tokens == ["[SEP]"]
    assert "structural loss" in report.render()
    assert any("structural tokens" in record.message for record in package_logs.records)


def test_dropped_added_and_special_tokens_are_named(byte_level_bpe):
    base = spec_of(byte_level_bpe, "base")
    other = spec_of(word_level(["[UNK]", "the"]), "other")

    report = compare_tokenizers(base, other)

    assert report.components.removed_added_tokens == ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    assert report.components.removed_special_tokens == report.components.removed_added_tokens
    assert report.components.base_uses_byte_level and not report.components.other_uses_byte_level
    assert "byte-level yes -> no" in report.render()


@pytest.mark.parametrize(
    ("base_template", "other_template", "expected"),
    [
        ("{{ x }}", "{{ x }}", "identical"),
        ("{{ x }}", "{{ y }}", "changed"),
        ("{{ x }}", None, "only in base"),
        (None, "{{ y }}", "only in other"),
        (None, None, "absent"),
    ],
)
def test_chat_templates_are_compared(wordpiece, base_template, other_template, expected):
    spec = spec_of(wordpiece, "self")
    base = dataclasses.replace(spec, chat_template=base_template)
    other = dataclasses.replace(spec, chat_template=other_template)

    report = compare_tokenizers(base, other)

    assert report.components.chat_template == expected


def test_presets_are_attributed_and_empty_ones_left_out(trimmed_bpe):
    base, trimmed = trimmed_bpe

    report = compare_tokenizers(base, trimmed, presets=["script:Latin"])

    presets = {preset.name: preset for preset in report.profile.presets}
    # The byte-level fixture declares no unknown token, so that preset matches nothing.
    assert "unk" not in presets
    assert presets["byte_alphabet"].always_kept
    assert presets["byte_alphabet"].removed == 0
    assert presets["script:Latin"].always_kept is False
    assert 0 < presets["ascii_letters"].kept < presets["ascii_letters"].base_tokens
    assert "presets cut" in report.render()


def test_a_preset_that_is_gone_entirely_is_reported():
    base = spec_of(word_level(MULTILINGUAL_TOKENS), "base")
    other = spec_of(word_level(["[UNK]", "the", "cat"]), "other")

    report = compare_tokenizers(base, other, presets=["script:Cyrillic"])

    cyrillic = next(preset for preset in report.profile.presets if preset.name == "script:Cyrillic")
    # "catж" is not in there: the preset asks for tokens whose letters are *all* Cyrillic.
    assert cyrillic.base_tokens == 2
    assert cyrillic.kept == 0
    assert "presets lost" in report.render()


def test_scripts_and_categories_bucket_the_whole_vocabulary():
    base = spec_of(word_level(MULTILINGUAL_TOKENS), "base")
    other = spec_of(word_level(["[UNK]", "the", "cat", "123"]), "other")

    report = compare_tokenizers(base, other)

    scripts = {group.name: group for group in report.profile.scripts}
    assert sum(group.base_tokens for group in report.profile.scripts) == base.vocab_size
    assert scripts["CYRILLIC"].kept == 0
    assert scripts["GREEK"].kept == 0
    assert scripts["LATIN"].kept == 3
    assert scripts[MIXED].base_tokens == 1
    assert scripts[NON_LETTER].base_tokens == 3

    categories = {group.name: group for group in report.profile.categories}
    assert categories["letter"].base_tokens == 7
    assert categories["number"].kept == 1
    assert categories["punctuation"].base_tokens == 2


def test_undecodable_tokens_get_their_own_bucket(trimmed_bpe):
    base, trimmed = trimmed_bpe

    report = compare_tokenizers(base, trimmed)

    scripts = {group.name: group for group in report.profile.scripts}
    assert scripts[UNDECODABLE].base_tokens > 0
    # A byte-level vocabulary reaches more categories than the summary shows in one line.
    assert len(report.profile.categories) > 6
    assert report.render().count(", ...") >= 1


def test_removed_tokens_are_profiled_by_id_and_quoted_lowest_first():
    base = spec_of(word_level(MULTILINGUAL_TOKENS), "base")
    other = spec_of(word_level(["[UNK]", "the"]), "other")

    report = compare_tokenizers(base, other, examples=3)

    assert sum(report.profile.removed_by_decile) == report.vocabulary.removed
    assert [example.token for example in report.profile.removed_examples] == ["cat", "кот", "кошка"]
    assert [example.token_id for example in report.profile.removed_examples] == [2, 3, 4]
    assert report.profile.removed_examples[0].surface == "cat"
    assert "first removed" in report.render()


def test_encoding_drift_is_measured_on_the_sample_texts(trimmed_bpe):
    base, trimmed = trimmed_bpe

    report = compare_tokenizers(base, trimmed, texts=["the cat sat", "quantum bureaucracy"])

    assert report.encoding.checked == 2
    assert report.encoding.length_ratio > 1.0
    assert report.encoding.identical_rate < 1.0
    # The text that gains the most tokens is quoted first.
    assert report.encoding.examples[0] == "the cat sat"
    assert "fragmented" in report.render()


def test_without_texts_the_encoding_comparison_is_skipped(trimmed_bpe):
    base, trimmed = trimmed_bpe

    report = compare_tokenizers(base, trimmed)

    assert report.encoding is None
    assert "encoding" not in report.render()


def test_the_report_survives_a_json_round_trip(trimmed_bpe, tmp_path):
    base, trimmed = trimmed_bpe
    report = compare_tokenizers(base, trimmed, texts=["the cat sat"])

    path = report.save(tmp_path / "nested" / "diff.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["base"] == "base"
    assert payload["vocabulary"]["is_subset"] is True
    assert payload["profile"]["presets"][0]["name"]


def test_unnamed_tokenizers_still_have_labels(wordpiece):
    spec = TokenizerSpec.from_tokenizer(wordpiece)
    unlabelled = dataclasses.replace(spec, source=None)

    report = compare_tokenizers(unlabelled, unlabelled)

    assert report.base == "base"
    assert report.other == "other"


def test_ratios_of_empty_groups_do_not_divide_by_zero():
    assert GroupDiff(name="empty", base_tokens=0, kept=0).kept_fraction == 1.0
    assert (
        VocabularyDiff(
            base_size=0,
            other_size=0,
            shared=0,
            removed=0,
            introduced=0,
            is_subset=True,
            ids_contiguous=True,
            order_preserved=True,
        ).removed_fraction
        == 0.0
    )
    assert EncodingDiff().length_ratio == 1.0
    assert EncodingDiff().identical_rate == 1.0
