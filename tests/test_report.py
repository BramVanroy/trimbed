import json

from trimbed.report import (
    REPORT_FILENAME,
    CorpusReport,
    ModelReport,
    ModelVerificationReport,
    TrimReport,
    VerificationReport,
    VocabularyReport,
)


def make_report(**overrides):
    vocabulary = VocabularyReport(
        model_type="BPE",
        original_size=1000,
        trimmed_size=250,
        structural_tokens=100,
        kept_by_reason={"corpus": 140, "structural": 100, "dependency": 10},
    )
    defaults = {
        "model": "some/model",
        "output_dir": "trimmed",
        "vocabulary": vocabulary,
        "corpus": CorpusReport(documents=10, total_tokens=500, distinct_tokens=300, coverage=0.998),
    }
    return TrimReport(**(defaults | overrides))


def test_reduction_is_computed_from_the_sizes():
    assert make_report().vocabulary.reduction == 0.75


def test_reduction_of_an_empty_vocabulary_is_zero():
    vocabulary = VocabularyReport(
        model_type="BPE", original_size=0, trimmed_size=0, structural_tokens=0, kept_by_reason={}
    )
    assert vocabulary.reduction == 0.0


def make_model_report(**overrides):
    defaults = {
        "model_class": "BertForMaskedLM",
        "old_embedding_rows": 1000,
        "new_embedding_rows": 250,
        "old_parameters": 10_000,
        "new_parameters": 4_000,
        "tied_embeddings": True,
        "has_output_head": False,
    }
    return ModelReport(**(defaults | overrides))


def test_parameters_removed():
    assert make_model_report().parameters_removed == 6_000


def test_render_summarises_every_stage():
    report = make_report(
        model_trim=make_model_report(),
        verification=VerificationReport(
            checked=10, identical=9, equivalent_text=10, original_tokens=1000, trimmed_tokens=1004
        ),
        model_verification=ModelVerificationReport(
            checked=8, max_hidden_diff=1e-7, max_logit_diff=2e-7, tolerance=1e-5
        ),
        sidecar_files=["modules.json", "1_Pooling/config.json"],
    )
    rendered = report.render()

    assert "some/model" in rendered
    assert "75.0% removed" in rendered
    assert "9/10 identical" in rendered
    assert "1.0040x tokens" in rendered
    assert "6,000 of 10,000 parameters removed" in rendered
    assert "BertForMaskedLM" in rendered
    assert "model check      passed on 8 texts" in rendered
    assert "1_Pooling/config.json" in rendered
    assert "trimmed" in rendered


def test_render_calls_out_a_failed_model_check():
    report = make_report(
        model_verification=ModelVerificationReport(checked=4, max_hidden_diff=0.5, tolerance=1e-5),
    )
    rendered = report.render()

    assert "model check      FAILED on 4 texts" in rendered
    assert "max |dlogit| n/a" in rendered


def test_length_ratio_without_tokens_is_one():
    assert VerificationReport(checked=0, identical=0, equivalent_text=0).length_ratio == 1.0


def test_render_marks_a_dry_run():
    report = make_report(output_dir=None, dry_run=True)
    assert "dry run, nothing written" in report.render()


def test_render_calls_out_ignored_and_dropped_requests():
    vocabulary = VocabularyReport(
        model_type="BPE",
        original_size=100,
        trimmed_size=50,
        structural_tokens=10,
        kept_by_reason={},
        unknown_requested_tokens=["kangaroo"],
        dropped_requested_tokens={"dog": ["explicit_token"]},
    )
    rendered = make_report(vocabulary=vocabulary).render()
    assert "1 requested tokens were ignored" in rendered
    assert "1 explicitly requested tokens" in rendered


def test_save_writes_json_into_a_created_directory(tmp_path):
    target = tmp_path / "nested" / "out"
    path = make_report().save(target)

    assert path == target / REPORT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["vocabulary"]["trimmed_size"] == 250
    assert payload["corpus"]["documents"] == 10
