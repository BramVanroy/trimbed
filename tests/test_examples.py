"""Run every example in `examples/` against tiny local fixtures.

The examples default to Hub models, but each one takes its model as a `run` argument, so
the whole set can be executed offline against a fixture tokenizer saved to a temp
directory. That is the point of this module: an example that stops matching the API fails
the suite instead of quietly rotting.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def load_example(name: str):
    """Import an example module by file name, without needing it to be a package."""
    path = EXAMPLES_DIR / name
    if path.stem in sys.modules:
        # Executing an example twice would re-run 04's @register_preset, which refuses a
        # duplicate name, so a second load returns the module the first one produced.
        return sys.modules[path.stem]

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_names() -> list[str]:
    return sorted(path.name for path in EXAMPLES_DIR.glob("*.py"))


@pytest.fixture
def local_model(wordpiece, tmp_path) -> str:
    """Save a fixture tokenizer to disk so the examples can load it like a checkpoint."""
    directory = tmp_path / "checkpoint"
    wordpiece.save_pretrained(directory)
    return str(directory)


def test_every_example_exposes_a_working_parser(example_names, capsys, monkeypatch):
    for name in example_names:
        monkeypatch.setattr(sys, "argv", [name, "--help"])
        with pytest.raises(SystemExit) as excinfo:
            load_example(name).main()
        assert excinfo.value.code == 0
        assert "--model" in capsys.readouterr().out


def test_the_readme_lists_every_example(example_names):
    listing = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    for name in example_names:
        assert name in listing, f"{name} is missing from examples/README.md"


def test_01_inspect_reports_the_tokenizer_family(local_model, capsys):
    summary = load_example("01_inspect_tokenizer.py").run(local_model)

    assert summary["model_type"] == "WordPiece"
    assert "WordPiece" in capsys.readouterr().out


def test_02_trims_from_must_keep_rules_alone(local_model, tmp_path):
    report = load_example("02_trim_tokenizer_only.py").run(local_model, tmp_path / "out")

    assert report.vocabulary.trimmed_size < report.vocabulary.original_size
    assert report.corpus is None
    assert report.model_trim is None
    assert (tmp_path / "out" / "tokenizer.json").exists()


@pytest.mark.torch
def test_03_trims_tokenizer_and_model_over_a_corpus(
    wordpiece, tiny_model_factory, corpus_dataset, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint"
    wordpiece.save_pretrained(checkpoint)
    tiny_model_factory(len(wordpiece), with_head=False).save_pretrained(checkpoint)
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)

    report = load_example("03_trim_with_corpus.py").run(str(checkpoint), "unused", tmp_path / "out")

    assert report.corpus is not None
    assert report.corpus.documents == len(corpus_dataset)
    assert report.model_trim is not None
    # The example asks for 64-row alignment, and the padding must survive the trim.
    assert report.model_trim.new_embedding_rows % 64 == 0
    assert report.model_verification is not None
    assert report.model_verification.ok


@pytest.fixture
def chemistry_model(tmp_path) -> str:
    """A checkpoint whose vocabulary actually contains element symbols to match on."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokens = ["[UNK]", "Fe", "##Fe", "Cu", "Zn", "water", "rust", "the", "##ing"]
    tokenizer = Tokenizer(
        models.WordPiece(vocab={token: index for index, token in enumerate(tokens)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    directory = tmp_path / "chemistry"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(directory)
    return str(directory)


def test_04_registers_a_preset_and_attributes_the_tokens_to_it(chemistry_model, tmp_path):
    from trimbed.presets import available_presets

    report = load_example("04_custom_preset.py").run(chemistry_model, tmp_path / "out")

    assert "chemical_elements" in available_presets()
    # Fe, ##Fe, Cu and Zn: the preset matches the surface form, so the WordPiece
    # continuation prefix does not hide the symbol from it.
    assert report.vocabulary.kept_by_reason["preset:chemical_elements"] == 4


def test_05_selects_trims_and_verifies_without_writing(local_model, tmp_path):
    result = load_example("05_low_level_api.py").run(local_model)

    assert result.checked == 3
    assert result.ok
    # Only the checkpoint the fixture wrote; the example itself saves nothing.
    assert [path.name for path in tmp_path.iterdir()] == ["checkpoint"]


def test_06_counts_a_corpus_of_local_files(local_model, sample_texts, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.jsonl").write_text(
        "\n".join(json.dumps({"text": text}) for text in sample_texts), encoding="utf-8"
    )

    report = load_example("06_local_corpus.py").run(local_model, corpus, tmp_path / "out")

    # Nothing was fetched: both the checkpoint and the corpus came off this temp directory.
    assert report.corpus is not None
    assert report.corpus.documents == len(sample_texts)
    assert report.vocabulary.trimmed_size < report.vocabulary.original_size
    assert (tmp_path / "out" / "tokenizer.json").exists()


def test_07_compares_a_checkpoint_with_a_trim_of_it(local_model, wordpiece, tmp_path, capsys):
    from trimbed.spec import TokenizerSpec
    from trimbed.tokenizer_trim import trim_tokenizer

    spec = TokenizerSpec.from_tokenizer(wordpiece)
    trimmed = tmp_path / "trimmed"
    trim_tokenizer(wordpiece, spec, spec.structural_ids).tokenizer.save_pretrained(trimmed)

    report = load_example("07_compare_tokenizers.py").run(local_model, str(trimmed))

    assert report.vocabulary.is_subset
    assert report.vocabulary.order_preserved
    assert report.encoding.checked == len(load_example("07_compare_tokenizers.py").TEXTS)
    assert "vocabulary" in capsys.readouterr().out
