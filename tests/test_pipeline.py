import json
from pathlib import Path

import pytest

from trimbed.config import TrimConfig
from trimbed.pipeline import TrimPipeline
from trimbed.report import CONFIG_FILENAME, REPORT_FILENAME


@pytest.fixture
def local_tokenizer(byte_level_bpe, tmp_path):
    """The BPE fixture saved to disk, so the pipeline can load it like any model."""
    path = tmp_path / "source-model"
    byte_level_bpe.save_pretrained(path)
    return str(path)


@pytest.fixture
def config_for(local_tokenizer, corpus_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)

    def build(**overrides):
        base = {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "trim_model": False,
            "corpus": {"datasets": [{"path": "fake", "streaming": False}]},
            "selection": {"min_count": 1},
        }
        return TrimConfig.model_validate(base | overrides)

    return build


def test_a_full_tokenizer_only_run_writes_every_artefact(config_for, tmp_path):
    report = TrimPipeline(config_for()).run()
    output = tmp_path / "out"

    assert (output / REPORT_FILENAME).exists()
    assert (output / CONFIG_FILENAME).exists()
    assert (output / "tokenizer.json").exists()
    assert report.vocabulary.trimmed_size < report.vocabulary.original_size
    assert report.corpus is not None
    assert report.model_trim is None


def test_the_saved_tokenizer_reloads_and_matches_the_report(config_for, tmp_path):
    from transformers import AutoTokenizer

    report = TrimPipeline(config_for()).run()
    reloaded = AutoTokenizer.from_pretrained(tmp_path / "out")

    assert len(reloaded) == report.vocabulary.trimmed_size


def test_verification_runs_against_the_corpus_samples(config_for):
    report = TrimPipeline(config_for()).run()

    assert report.verification is not None
    assert report.verification.checked > 0
    assert report.verification.identical == report.verification.checked


def test_verification_can_be_switched_off(config_for):
    assert TrimPipeline(config_for(verify=False)).run().verification is None


def test_a_dry_run_writes_nothing_but_still_reports(config_for, tmp_path):
    report = TrimPipeline(config_for()).run(dry_run=True)

    assert report.dry_run is True
    assert report.output_dir is None
    assert report.vocabulary.trimmed_size > 0
    assert not (tmp_path / "out").exists()


def test_a_non_empty_output_directory_is_refused(config_for, tmp_path):
    output = tmp_path / "out"
    output.mkdir(parents=True)
    (output / "something.txt").write_text("existing work", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        TrimPipeline(config_for()).run()


def test_overwrite_allows_reusing_a_directory(config_for, tmp_path):
    output = tmp_path / "out"
    output.mkdir(parents=True)
    (output / "something.txt").write_text("existing work", encoding="utf-8")

    assert TrimPipeline(config_for(overwrite=True)).run().vocabulary.trimmed_size > 0


def test_selection_without_a_corpus_needs_no_dataset(local_tokenizer, tmp_path):
    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "trim_model": False,
            "selection": {"keep_presets": ["ascii_letters"]},
        }
    )
    report = TrimPipeline(config).run()

    assert report.corpus is None
    # Nothing was configured to verify against: no corpus, and no texts to keep either.
    assert report.verification is None


def test_texts_to_keep_are_verified_without_a_corpus(local_tokenizer, tmp_path):
    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "trim_model": False,
            "selection": {"keep_texts": ["the cat sat", "the dog"]},
        }
    )
    report = TrimPipeline(config).run()

    # A text the run was told to keep encodable is the one worth proving it kept, corpus
    # or no corpus, and it maps one-to-one by construction.
    assert report.corpus is None
    assert report.verification is not None
    assert report.verification.checked == 2
    assert report.verification.identical == 2


def test_texts_to_keep_are_verified_alongside_the_corpus(config_for):
    report = TrimPipeline(config_for(selection={"min_count": 1, "keep_texts": ["the cat sat"]})).run()

    assert report.corpus is not None
    assert report.verification is not None
    assert report.verification.checked == report.corpus.documents + 1


def test_the_report_records_provenance(config_for, tmp_path):
    TrimPipeline(config_for()).run()
    payload = json.loads((tmp_path / "out" / REPORT_FILENAME).read_text(encoding="utf-8"))

    assert "structural" in payload["vocabulary"]["kept_by_reason"]
    assert payload["vocabulary"]["kept_by_reason"]["corpus"] > 0


@pytest.mark.torch
def test_trimming_the_model_too(local_tokenizer, corpus_dataset, tmp_path, monkeypatch, tiny_model_factory):
    from transformers import AutoTokenizer

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer)
    model = tiny_model_factory(len(tokenizer), with_head=False)
    model.save_pretrained(local_tokenizer)

    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "trim_model": True,
            "corpus": {"datasets": [{"path": "fake", "streaming": False}]},
            "selection": {"min_count": 1},
            "embeddings": {"pad_to_multiple_of": 8},
        }
    )
    report = TrimPipeline(config).run()

    assert report.model_trim is not None
    assert report.model_trim.new_embedding_rows % 8 == 0
    assert report.model_trim.parameters_removed > 0
    assert (tmp_path / "out" / "config.json").exists()


@pytest.mark.torch
def test_the_model_check_compares_the_saved_model_against_the_original(
    local_tokenizer, corpus_dataset, tmp_path, monkeypatch, tiny_model_factory
):
    from transformers import AutoTokenizer

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer)
    tiny_model_factory(len(tokenizer), tied=False).save_pretrained(local_tokenizer)
    # A sentence-transformers release keeps its pooling configuration out here.
    (Path(local_tokenizer) / "modules.json").write_text("[]", encoding="utf-8")

    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "corpus": {"datasets": [{"path": "fake", "streaming": False}]},
            "selection": {"min_count": 1},
            "verify_model": True,
            "verify_model_samples": 3,
        }
    )
    report = TrimPipeline(config).run()

    assert report.model_trim is not None
    # AutoModel would have demoted this to the bare encoder and dropped the head.
    assert report.model_trim.model_class == "BertForMaskedLM"
    assert report.model_verification is not None
    assert report.model_verification.checked == 3
    assert report.model_verification.ok
    assert report.sidecar_files == ["modules.json"]
    assert (tmp_path / "out" / "modules.json").exists()


@pytest.mark.torch
def test_a_corpus_document_longer_than_the_context_is_still_compared(
    local_tokenizer, tmp_path, monkeypatch, tiny_model_factory
):
    from datasets import Dataset
    from transformers import AutoTokenizer

    # Real corpora are full of documents that are longer than the model's position table,
    # and the comparison used to hand them to the model whole and crash the whole run.
    long_document = " ".join(["the cat sat the dog"] * 60)
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *args, **kwargs: Dataset.from_dict({"text": [long_document, "the cat sat"]})
    )
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer)
    model = tiny_model_factory(len(tokenizer), with_head=False)
    model.save_pretrained(local_tokenizer)

    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "corpus": {"datasets": [{"path": "fake", "streaming": False}]},
            "selection": {"min_count": 1},
            "verify_model_samples": 2,
            "copy_sidecar_files": False,
        }
    )
    report = TrimPipeline(config).run()

    assert report.model_verification is not None
    assert report.model_verification.max_length == model.config.max_position_embeddings
    assert report.model_verification.checked == 2
    assert report.model_verification.ok


@pytest.mark.torch
def test_the_model_check_can_be_turned_off(config_for, local_tokenizer, tiny_model_factory):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer)
    tiny_model_factory(len(tokenizer)).save_pretrained(local_tokenizer)

    config = config_for(trim_model=True, verify_model=False, copy_sidecar_files=False)
    report = TrimPipeline(config).run()

    # The model is still trimmed; only the two extra loads the comparison needs are saved.
    assert report.model_trim is not None
    assert report.model_verification is None


@pytest.mark.torch
def test_the_model_check_is_skipped_without_any_texts(local_tokenizer, tmp_path, tiny_model_factory):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer)
    tiny_model_factory(len(tokenizer), with_head=False).save_pretrained(local_tokenizer)

    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "selection": {"keep_presets": ["ascii_letters"]},
            "verify_model": True,
            "copy_sidecar_files": False,
        }
    )
    report = TrimPipeline(config).run()

    assert report.model_verification is None
    assert report.sidecar_files == []


@pytest.mark.torch
def test_the_trimmed_pair_reloads_and_runs_together(
    local_tokenizer, corpus_dataset, tmp_path, monkeypatch, tiny_model_factory
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer)
    tiny_model_factory(len(tokenizer), with_head=False).save_pretrained(local_tokenizer)

    config = TrimConfig.model_validate(
        {
            "model": local_tokenizer,
            "output_dir": str(tmp_path / "out"),
            "corpus": {"datasets": [{"path": "fake", "streaming": False}]},
            "selection": {"min_count": 1},
        }
    )
    TrimPipeline(config).run()

    new_tokenizer = AutoTokenizer.from_pretrained(tmp_path / "out")
    new_model = AutoModel.from_pretrained(tmp_path / "out")
    encoded = new_tokenizer("the cat sat", return_tensors="pt")
    with torch.no_grad():
        output = new_model(**encoded)

    assert output.last_hidden_state.shape[1] == encoded["input_ids"].shape[1]
