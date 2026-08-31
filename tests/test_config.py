from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trimbed.config import SelectionConfig, TrimConfig, load_config, parse_overrides


MINIMAL = {"model": "some/model", "selection": {"keep_presets": ["digits"]}}


def test_defaults_are_sensible():
    config = TrimConfig.model_validate(MINIMAL)
    assert config.trim_model is True
    assert config.verify is True
    assert config.output_dir == Path("trimmed")
    assert config.corpus.datasets == []


def test_unknown_keys_are_rejected_rather_than_ignored():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrimConfig.model_validate(MINIMAL | {"trimm_model": True})


def test_a_config_with_no_selection_source_is_rejected():
    with pytest.raises(ValidationError, match="nothing to select on"):
        TrimConfig.model_validate({"model": "some/model"})


def test_a_corpus_without_a_criterion_is_rejected():
    with pytest.raises(ValidationError, match="no criterion"):
        TrimConfig.model_validate({"model": "m", "corpus": {"datasets": [{"path": "d"}]}})


def test_a_corpus_with_any_criterion_is_accepted():
    config = TrimConfig.model_validate(
        {"model": "m", "corpus": {"datasets": [{"path": "d"}]}, "selection": {"min_count": 1}}
    )
    assert config.corpus.datasets[0].path == "d"
    assert config.corpus.datasets[0].streaming is True


def test_a_model_sample_larger_than_the_tokenizer_sample_is_rejected():
    with pytest.raises(ValidationError, match="larger than verify_samples"):
        TrimConfig.model_validate(MINIMAL | {"verify_samples": 4, "verify_model_samples": 8})


def test_a_model_sample_the_tokenizer_sample_can_supply_is_accepted():
    config = TrimConfig.model_validate(MINIMAL | {"verify_samples": 8, "verify_model_samples": 8})
    assert config.verify_model_samples == 8


@pytest.mark.parametrize("coverage", [0.0, 1.1, -0.5])
def test_coverage_must_be_a_fraction(coverage):
    with pytest.raises(ValidationError):
        TrimConfig.model_validate(MINIMAL | {"selection": {"coverage": coverage}})


def test_from_yaml_round_trips(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    config = TrimConfig.from_yaml(path)
    assert config.model == "some/model"
    assert yaml.safe_load(config.to_yaml())["model"] == "some/model"


def test_from_yaml_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        TrimConfig.from_yaml(path)


def test_overrides_apply_to_nested_fields_and_skip_none():
    config = TrimConfig.model_validate(MINIMAL)
    updated = config.with_overrides({"selection.top_k": 500, "model": None, "trim_model": False})
    assert updated.selection.top_k == 500
    assert updated.model == "some/model"
    assert updated.trim_model is False
    assert config.selection.top_k is None


CORPUS = {
    "model": "m",
    "corpus": {"datasets": [{"path": "one"}, {"path": "two"}]},
    "selection": {"min_count": 1},
}


def test_overrides_index_into_a_dataset_mixture():
    config = TrimConfig.model_validate(CORPUS).with_overrides({"corpus.datasets.1.max_samples": 50})
    assert config.corpus.datasets[1].max_samples == 50
    assert config.corpus.datasets[0].max_samples is None


def test_a_non_integer_list_index_is_rejected():
    with pytest.raises(ValueError, match="must be an integer"):
        TrimConfig.model_validate(CORPUS).with_overrides({"corpus.datasets.first.path": "x"})


def test_an_out_of_range_list_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        TrimConfig.model_validate(CORPUS).with_overrides({"corpus.datasets.7.path": "x"})


def test_overrides_are_revalidated():
    config = TrimConfig.model_validate(MINIMAL)
    with pytest.raises(ValidationError):
        config.with_overrides({"selection.coverage": 5.0})


def test_has_explicit_sources():
    assert not SelectionConfig().has_explicit_sources
    assert SelectionConfig(keep_tokens=["a"]).has_explicit_sources
    assert SelectionConfig(keep_patterns=["^a"]).has_explicit_sources


def test_load_config_reads_a_yaml_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert load_config(path).model == "some/model"


def test_load_config_from_a_bare_model_keeps_the_structural_tokens():
    config = load_config(None, "some/model")
    assert config.model == "some/model"
    assert config.selection.keep_presets == ["structural"]


def test_load_config_needs_one_of_the_two():
    with pytest.raises(ValueError, match="--config or --model"):
        load_config(None)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("selection.top_k=5000", 5000),
        ("selection.top_k=false", False),
        ("selection.top_k=[1, 2]", [1, 2]),
        ("selection.top_k=alphanumeric", "alphanumeric"),
        ("selection.top_k=a=b", "a=b"),
    ],
)
def test_parse_overrides_types_values_the_way_yaml_does(override, expected):
    assert parse_overrides([override]) == {"selection.top_k": expected}


def test_parse_overrides_rejects_a_missing_equals_sign():
    with pytest.raises(ValueError, match=r"key\.path=value"):
        parse_overrides(["selection.top_k"])


def test_parse_overrides_feeds_with_overrides():
    config = load_config(None, "some/model").with_overrides(parse_overrides(["selection.min_count=3", "seed=11"]))
    assert config.selection.min_count == 3
    assert config.seed == 11


def test_keeping_the_chat_template_alone_is_not_a_selection_source():
    # It is on by default and selects nothing without a template, so counting it would
    # let a config with no criteria at all through the validator.
    with pytest.raises(ValidationError, match="nothing to select on"):
        TrimConfig.model_validate({"model": "m", "selection": {"keep_chat_template": True}})


def test_texts_to_keep_are_a_selection_source():
    config = TrimConfig.model_validate({"model": "m", "selection": {"keep_texts": ["### Response:"]}})

    assert config.selection.has_explicit_sources
    assert config.selection.keep_chat_template is True
