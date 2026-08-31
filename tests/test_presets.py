import pytest

from trimbed.config import SelectionConfig
from trimbed.presets import (
    available_presets,
    describe_presets,
    register_parametrised_preset,
    register_preset,
    resolve_preset,
    resolve_presets,
)
from trimbed.selection import select_tokens
from trimbed.spec import TokenizerSpec


@pytest.fixture
def bpe_spec(byte_level_bpe):
    return TokenizerSpec.from_tokenizer(byte_level_bpe)


@pytest.fixture
def wordpiece_spec(wordpiece):
    return TokenizerSpec.from_tokenizer(wordpiece)


def test_available_presets_marks_the_parametrised_family():
    names = available_presets()
    assert "digits" in names
    assert "script:..." in names


def test_describe_presets_reports_structure_and_summaries():
    described = {preset.name: preset for preset in describe_presets()}

    assert described["special_tokens"].always_kept
    assert described["byte_alphabet"].always_kept
    assert not described["digits"].always_kept
    assert described["digits"].summary == "Existing tokens whose surface form is made up entirely of ASCII digits."
    assert described["script:..."].summary.startswith("Existing tokens whose letters all belong to one Unicode script")


@pytest.mark.parametrize("family", ["byte_level_bpe", "wordpiece", "unigram", "wordlevel"])
def test_always_kept_presets_are_covered_by_the_structural_ids(family, request):
    spec = TokenizerSpec.from_tokenizer(request.getfixturevalue(family))
    structural = spec.structural_tokens

    always_kept = [preset.name for preset in describe_presets() if preset.always_kept]
    assert "structural" in always_kept
    for name in always_kept:
        assert resolve_preset(name, spec) <= structural, name


def test_the_structural_preset_is_exactly_what_the_selector_protects(bpe_spec):
    selection = select_tokens(bpe_spec, None, SelectionConfig(keep_presets=["structural"]))

    assert selection.kept_ids == set(bpe_spec.structural_ids)
    assert resolve_preset("structural", bpe_spec) == bpe_spec.structural_tokens


def test_describe_presets_tolerates_a_preset_without_a_docstring():
    register_preset("test_only_undocumented")(lambda spec: set())

    described = {preset.name: preset for preset in describe_presets()}
    assert described["test_only_undocumented"].summary == ""


def test_special_and_added_tokens(bpe_spec):
    assert resolve_preset("special_tokens", bpe_spec) == {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}
    assert resolve_preset("added_tokens", bpe_spec) == resolve_preset("special_tokens", bpe_spec)


def test_byte_alphabet_covers_the_full_alphabet(bpe_spec):
    assert len(resolve_preset("byte_alphabet", bpe_spec)) == 256


def test_presets_match_on_surface_form_not_raw_token(bpe_spec):
    letters = resolve_preset("ascii_letters", bpe_spec)
    # "Ġthe" stands for " the", so a leading space must not disqualify it.
    assert "Ġthe" in letters
    assert "the" in letters
    assert "Ġ" not in letters


def test_digits_and_alphanumeric(wordpiece_spec):
    assert resolve_preset("digits", wordpiece_spec) == set()
    alphanumeric = resolve_preset("alphanumeric", wordpiece_spec)
    assert "cat" in alphanumeric
    assert "[UNK]" not in alphanumeric


def test_punctuation_and_whitespace(bpe_spec):
    punctuation = resolve_preset("punctuation", bpe_spec)
    assert "!" in punctuation
    whitespace = resolve_preset("whitespace", bpe_spec)
    assert "Ġ" in whitespace
    assert "!" not in whitespace


def test_single_characters(wordpiece_spec):
    singles = resolve_preset("single_characters", wordpiece_spec)
    assert "a" in singles
    assert "cat" not in singles


def test_unk_preset(wordpiece_spec, bpe_spec):
    assert resolve_preset("unk", wordpiece_spec) == {"[UNK]"}
    assert resolve_preset("unk", bpe_spec) == set()


def test_script_preset_selects_by_unicode_script(wordpiece_spec):
    latin = resolve_preset("script:Latin", wordpiece_spec)
    assert "cat" in latin
    assert resolve_preset("script:Cyrillic", wordpiece_spec) == set()


def test_script_preset_is_case_insensitive(wordpiece_spec):
    assert resolve_preset("script:latin", wordpiece_spec) == resolve_preset("script:Latin", wordpiece_spec)


def test_script_preset_needs_an_argument(wordpiece_spec):
    with pytest.raises(ValueError, match="needs an argument"):
        resolve_preset("script:", wordpiece_spec)


def test_unknown_preset_lists_the_alternatives(wordpiece_spec):
    with pytest.raises(KeyError, match="available presets"):
        resolve_preset("nonexistent", wordpiece_spec)


def test_resolve_presets_keeps_results_attributable(wordpiece_spec):
    resolved = resolve_presets(["unk", "digits"], wordpiece_spec)
    assert set(resolved) == {"unk", "digits"}
    assert resolved["unk"] == {"[UNK]"}


def test_a_project_can_register_its_own_preset(wordpiece_spec):
    @register_preset("test_only_animals")
    def _animals(spec):
        return {token for token in spec.vocabulary if token in {"cat", "dog"}}

    assert resolve_preset("test_only_animals", wordpiece_spec) == {"cat", "dog"}


def test_preset_names_may_not_contain_the_separator():
    with pytest.raises(ValueError, match="may not contain"):
        register_preset("bad:name")(lambda spec: set())


def test_duplicate_preset_names_are_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_preset("digits")(lambda spec: set())


def test_duplicate_parametrised_prefixes_are_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_parametrised_preset("script")(lambda spec, arg: set())
