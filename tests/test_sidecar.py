"""Copying the source repository's vocabulary-independent files into the output."""

import pytest

from trimbed.sidecar import DEFAULT_SIDECAR_PATTERNS, copy_sidecar_files


@pytest.fixture
def source_repo(tmp_path):
    """A local checkpoint shaped like a sentence-transformers release."""
    source = tmp_path / "source"
    (source / "1_Pooling").mkdir(parents=True)
    (source / "modules.json").write_text('[{"idx": 0, "name": "0", "type": "Transformer"}]', encoding="utf-8")
    (source / "1_Pooling" / "config.json").write_text('{"pooling_mode_cls_token": true}', encoding="utf-8")
    (source / "config.json").write_text('{"vocab_size": 1000}', encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "README.md").write_text("the untrimmed model", encoding="utf-8")
    return source


def test_the_module_files_are_copied_and_nothing_else(source_repo, tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    copied = copy_sidecar_files(str(source_repo), output)

    assert copied == ["1_Pooling/config.json", "modules.json"]
    assert (output / "1_Pooling" / "config.json").read_text(encoding="utf-8") == '{"pooling_mode_cls_token": true}'
    # Weights, configs and the README belong to the untrimmed model.
    assert not (output / "model.safetensors").exists()
    assert not (output / "README.md").exists()


def test_files_the_trim_wrote_are_never_overwritten(source_repo, tmp_path, package_logs):
    output = tmp_path / "out"
    output.mkdir()
    (output / "config.json").write_text('{"vocab_size": 250}', encoding="utf-8")

    copied = copy_sidecar_files(str(source_repo), output, patterns=[*DEFAULT_SIDECAR_PATTERNS, "config.json"])

    assert "config.json" not in copied
    assert (output / "config.json").read_text(encoding="utf-8") == '{"vocab_size": 250}'
    assert "not overwriting" in package_logs.text


def test_a_source_without_module_files_copies_nothing(tmp_path):
    source = tmp_path / "plain"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")

    assert copy_sidecar_files(str(source), tmp_path / "out") == []


def test_a_hub_source_is_listed_and_downloaded(source_repo, tmp_path, monkeypatch):
    import huggingface_hub

    class FakeApi:
        def list_repo_files(self, repo_id, revision=None):
            assert repo_id == "someone/some-model"
            assert revision == "abc123"
            return ["modules.json", "1_Pooling/config.json", "model.safetensors"]

    def fake_download(repo_id, filename, revision=None):
        assert repo_id == "someone/some-model"
        return str(source_repo / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    copied = copy_sidecar_files("someone/some-model", tmp_path / "out", revision="abc123")

    assert copied == ["1_Pooling/config.json", "modules.json"]
    assert (tmp_path / "out" / "modules.json").exists()
