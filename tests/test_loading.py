"""Which class a checkpoint is loaded with, and why it is not always `AutoModel`."""

import json
from pathlib import Path

import pytest

from trimbed.config import EmbeddingTrimConfig
from trimbed.loading import load_model, resolve_model_class


pytestmark = pytest.mark.torch


@pytest.fixture
def saved_model(tiny_model_factory, tmp_path):
    """A checkpoint on disk whose config advertises an untied `BertForMaskedLM`."""
    tiny_model_factory(16, tied=False).save_pretrained(tmp_path)
    return str(tmp_path)


def _set_architectures(directory: str, value: list[str] | None) -> None:
    """Rewrite (or remove) the `architectures` entry of a saved config."""
    path = Path(directory) / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if value is None:
        payload.pop("architectures", None)
    else:
        payload["architectures"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_the_class_named_in_the_config_is_used(saved_model):
    import transformers

    assert resolve_model_class(saved_model) is transformers.BertForMaskedLM


def test_the_output_head_survives_the_load(saved_model):
    # AutoModel returns the bare encoder, which for an untied head means loading the
    # checkpoint without its trained output weights and saving it back without them.
    assert load_model(saved_model).get_output_embeddings() is not None


def test_an_explicit_auto_class_wins(saved_model):
    import transformers

    config = EmbeddingTrimConfig(auto_class="AutoModel")

    assert resolve_model_class(saved_model, config=config) is transformers.AutoModel
    assert load_model(saved_model, config=config).get_output_embeddings() is None


def test_an_auto_class_transformers_does_not_export_is_refused(saved_model):
    config = EmbeddingTrimConfig(auto_class="AutoModelForMakingTea")

    with pytest.raises(ValueError, match="no class named"):
        resolve_model_class(saved_model, config=config)


def test_a_remote_code_architecture_falls_back_to_automodel(saved_model, package_logs):
    import transformers

    _set_architectures(saved_model, ["SomeRemoteCodeModel"])

    assert resolve_model_class(saved_model) is transformers.AutoModel
    assert "SomeRemoteCodeModel" in package_logs.text


def test_a_config_without_architectures_falls_back_quietly(saved_model, package_logs):
    import transformers

    _set_architectures(saved_model, None)

    assert resolve_model_class(saved_model) is transformers.AutoModel
    assert package_logs.text == ""
