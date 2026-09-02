"""Drive the `trimbed` command the way a user would.

The arguments are monkeypatched into `sys.argv` and the router's `main()` is called, so
each test goes through the real subcommand dispatch, the real argument parsing and the
real config-override merge rather than reaching for a `run` directly.
"""

from __future__ import annotations

import json
import runpy
import sys

import pytest
import yaml
from pydantic import ValidationError

from trimbed.cli.__main__ import main


def run(monkeypatch, *argv: str) -> None:
    """Run `trimbed` with `argv` as its command line."""
    monkeypatch.setattr(sys, "argv", ["trimbed", *argv])
    main()


@pytest.fixture
def local_tokenizer(byte_level_bpe, tmp_path) -> str:
    """A checkpoint on disk, so the scripts can load a tokenizer without the Hub."""
    path = tmp_path / "source-model"
    byte_level_bpe.save_pretrained(path)
    return str(path)


@pytest.fixture
def config_file(local_tokenizer, corpus_dataset, tmp_path, monkeypatch) -> str:
    """A YAML config pointing at the local checkpoint and a stubbed corpus."""
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": local_tokenizer,
                "output_dir": str(tmp_path / "out"),
                "trim_model": False,
                "corpus": {"datasets": [{"path": "fake", "streaming": False}]},
                "selection": {"min_count": 1},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_trim_writes_the_artefacts_and_prints_a_report(monkeypatch, config_file, tmp_path, capsys):
    run(monkeypatch, "trim", "--config", config_file)

    assert (tmp_path / "out" / "tokenizer.json").exists()
    assert (tmp_path / "out" / "trim_report.json").exists()
    assert (tmp_path / "out" / "_trimbed_config.yaml").exists()
    assert "vocabulary" in capsys.readouterr().out.lower()


def test_a_dry_run_reports_without_writing(monkeypatch, config_file, tmp_path, capsys):
    run(monkeypatch, "trim", "--config", config_file, "--dry-run")

    assert not (tmp_path / "out").exists()
    assert "dry run" in capsys.readouterr().out.lower()


def test_flags_override_the_config_file(monkeypatch, config_file, tmp_path):
    # The fixture keeps 263 tokens uncapped, 259 of them structural, so a cap of 262
    # is low enough to bite and high enough to stay above the structural floor.
    run(monkeypatch, "trim", "--config", config_file, "--max-vocab-size", "262", "--no-verify")

    report = json.loads((tmp_path / "out" / "trim_report.json").read_text(encoding="utf-8"))
    assert report["vocabulary"]["trimmed_size"] == 262
    assert report["verification"] is None


def test_keeping_a_text_holds_its_tokens_against_the_corpus(monkeypatch, chat_bpe, corpus_dataset, tmp_path):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    source = tmp_path / "chat-model"
    chat_bpe.save_pretrained(source)
    output = tmp_path / "kept"

    run(
        monkeypatch,
        "trim",
        "--model",
        str(source),
        "--output-dir",
        str(output),
        "--no-trim-model",
        "--keep-text",
        "the user said",
        "corpus.datasets=[{path: fake, streaming: false}]",
        "selection.min_count=1",
    )

    report = json.loads((output / "trim_report.json").read_text(encoding="utf-8"))
    assert report["vocabulary"]["kept_by_reason"]["text"] > 0
    # The template is kept by default, so its own words are in there too.
    assert report["vocabulary"]["kept_by_reason"]["chat_template"] > 0


def test_the_chat_template_can_be_left_out(monkeypatch, chat_bpe, corpus_dataset, tmp_path):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    source = tmp_path / "chat-model"
    chat_bpe.save_pretrained(source)
    output = tmp_path / "dropped"

    run(
        monkeypatch,
        "trim",
        "--model",
        str(source),
        "--output-dir",
        str(output),
        "--no-trim-model",
        "--no-keep-chat-template",
        "corpus.datasets=[{path: fake, streaming: false}]",
        "selection.min_count=1",
    )

    report = json.loads((output / "trim_report.json").read_text(encoding="utf-8"))
    assert "chat_template" not in report["vocabulary"]["kept_by_reason"]


def test_positional_overrides_beat_the_equivalent_flag(monkeypatch, config_file, tmp_path):
    run(
        monkeypatch,
        "trim",
        "--config",
        config_file,
        "--max-vocab-size",
        "262",
        "selection.max_vocab_size=260",
    )

    report = json.loads((tmp_path / "out" / "trim_report.json").read_text(encoding="utf-8"))
    assert report["vocabulary"]["trimmed_size"] == 260


def test_overrides_are_typed_by_yaml_not_left_as_strings(monkeypatch, config_file, tmp_path, capsys):
    run(monkeypatch, "trim", "--config", config_file, "verify=false", "seed=7")

    written = yaml.safe_load((tmp_path / "out" / "_trimbed_config.yaml").read_text(encoding="utf-8"))
    assert written["verify"] is False
    assert written["seed"] == 7
    assert "verification" not in capsys.readouterr().out.lower()


def test_an_unknown_override_path_is_rejected_rather_than_ignored(monkeypatch, config_file):
    with pytest.raises(ValidationError):
        run(monkeypatch, "trim", "--config", config_file, "selection.tpo_k=5")


def test_a_malformed_override_names_the_expected_form(monkeypatch, config_file):
    with pytest.raises(ValueError, match=r"key\.path=value"):
        run(monkeypatch, "trim", "--config", config_file, "selection.top_k")


def test_trimming_without_a_config_needs_only_a_model(monkeypatch, local_tokenizer, tmp_path, capsys):
    run(
        monkeypatch,
        "trim",
        "--model",
        local_tokenizer,
        "--output-dir",
        str(tmp_path / "bare"),
        "--keep-preset",
        "byte_alphabet",
        "--keep-token",
        "the",
        "--no-trim-model",
    )

    assert (tmp_path / "bare" / "tokenizer.json").exists()
    assert "preset:byte_alphabet" in capsys.readouterr().out


def test_a_non_empty_output_directory_is_refused_unless_overwrite(monkeypatch, config_file, tmp_path):
    run(monkeypatch, "trim", "--config", config_file)

    with pytest.raises(FileExistsError, match="not empty"):
        run(monkeypatch, "trim", "--config", config_file)

    run(monkeypatch, "trim", "--config", config_file, "--overwrite")


def test_neither_a_config_nor_a_model_is_an_error(monkeypatch):
    with pytest.raises(ValueError, match="--config or --model"):
        run(monkeypatch, "trim")


def test_count_writes_a_reusable_cache(monkeypatch, config_file, tmp_path, capsys):
    output = tmp_path / "counts.json"
    run(monkeypatch, "count", "--config", config_file, "-o", str(output))

    counts = json.loads(output.read_text(encoding="utf-8"))
    assert counts["num_documents"] == 20
    assert "20 documents" in capsys.readouterr().out


def test_count_accepts_overrides_too(monkeypatch, config_file, tmp_path):
    output = tmp_path / "counts.json"
    run(monkeypatch, "count", "--config", config_file, "-o", str(output), "corpus.batch_size=2")

    assert json.loads(output.read_text(encoding="utf-8"))["num_documents"] == 20


def test_counting_without_a_corpus_says_so(monkeypatch, local_tokenizer, tmp_path):
    with pytest.raises(ValueError, match="nothing to count"):
        run(monkeypatch, "count", "--model", local_tokenizer, "-o", str(tmp_path / "counts.json"))


def test_inspect_describes_the_tokenizer_from_a_model(monkeypatch, local_tokenizer, capsys):
    run(monkeypatch, "inspect", "--model", local_tokenizer)

    summary = json.loads(capsys.readouterr().out)
    assert summary["model_type"] == "BPE"
    assert summary["supported"] is True


def test_inspect_falls_back_to_the_config_model(monkeypatch, config_file, local_tokenizer, capsys):
    run(monkeypatch, "inspect", "--config", config_file, "--verbose")

    assert json.loads(capsys.readouterr().out)["source"] == local_tokenizer


def test_inspect_needs_something_to_inspect(monkeypatch):
    with pytest.raises(ValueError, match="--config or --model"):
        run(monkeypatch, "inspect")


@pytest.fixture
def trimmed_checkpoint(byte_level_bpe, tmp_path) -> str:
    """A trimmed copy of the local checkpoint, saved next to it."""
    from trimbed.spec import TokenizerSpec
    from trimbed.tokenizer_trim import trim_tokenizer

    spec = TokenizerSpec.from_tokenizer(byte_level_bpe)
    path = tmp_path / "trimmed-model"
    trim_tokenizer(byte_level_bpe, spec, spec.structural_ids).tokenizer.save_pretrained(path)
    return str(path)


def test_compare_diffs_two_checkpoints(monkeypatch, local_tokenizer, trimmed_checkpoint, tmp_path, capsys):
    texts = tmp_path / "samples.txt"
    texts.write_text("the cat sat\n\nthe dog\n", encoding="utf-8")
    output = tmp_path / "diff.json"

    run(
        monkeypatch,
        "compare",
        local_tokenizer,
        trimmed_checkpoint,
        "--preset",
        "script:Latin",
        "--text",
        "a hat",
        "--text-file",
        str(texts),
        "--examples",
        "3",
        "-o",
        str(output),
    )

    assert "relation" in capsys.readouterr().out
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["vocabulary"]["is_subset"] is True
    assert report["encoding"]["checked"] == 3
    assert len(report["profile"]["removed_examples"]) == 3
    assert any(preset["name"] == "script:Latin" for preset in report["profile"]["presets"])


def test_compare_needs_nothing_but_two_tokenizers(monkeypatch, local_tokenizer, trimmed_checkpoint, capsys, tmp_path):
    run(monkeypatch, "compare", local_tokenizer, trimmed_checkpoint, "--quiet")

    out = capsys.readouterr().out
    assert "encoding" not in out
    assert not list(tmp_path.glob("*.json"))


def test_list_presets_prints_the_registry(monkeypatch, capsys):
    from trimbed.presets import describe_presets

    run(monkeypatch, "presets", "--quiet")

    out = capsys.readouterr().out
    for preset in describe_presets():
        assert preset.name in out
        assert preset.summary in out
    # the structural ones are printed as their own group, above the opt-in ones
    assert out.index("special_tokens") < out.index("alphanumeric")


def test_the_router_also_runs_as_a_module(monkeypatch, capsys):
    # `python -m trimbed.cli` is the no-install path, so it goes through the same routing.
    # runpy warns when the module it is about to execute is already imported, which it is
    # here because this file imports `main` from it.
    monkeypatch.delitem(sys.modules, "trimbed.cli.__main__")
    monkeypatch.setattr(sys, "argv", ["trimbed", "presets", "--quiet"])
    runpy.run_module("trimbed.cli", run_name="__main__")

    assert "Structural" in capsys.readouterr().out
