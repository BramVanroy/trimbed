from collections import Counter

import pytest

from trimbed.config import CorpusConfig, DatasetSpec
from trimbed.counting import CorpusCounter, CorpusCounts


@pytest.fixture
def counts():
    return CorpusCounts(counts=Counter({1: 10, 2: 5, 3: 5, 4: 1}), total_num_tokens=21, num_documents=3)


def test_ranked_ids_sorts_by_frequency_then_id(counts):
    # The id tie-break keeps a run reproducible when many tokens share a count.
    assert counts.ranked_ids() == [1, 2, 3, 4]


def test_distinct_and_coverage(counts):
    assert counts.distinct_tokens == 4
    assert counts.coverage_of([1]) == pytest.approx(10 / 21)
    assert counts.coverage_of([1, 2, 3, 4]) == 1.0
    assert counts.coverage_of([]) == 0.0


def test_coverage_of_an_empty_corpus_is_total():
    assert CorpusCounts().coverage_of([1, 2]) == 1.0


def test_json_round_trip(counts, tmp_path):
    counts.samples = ["hello", "world"]
    path = tmp_path / "nested" / "counts.json"
    counts.save(path)
    restored = CorpusCounts.load_from_file(path)
    assert restored.counts == counts.counts
    assert restored.total_num_tokens == counts.total_num_tokens
    assert restored.num_documents == counts.num_documents
    assert restored.samples == ["hello", "world"]


def _count_over(tokenizer, dataset, monkeypatch, **corpus_kwargs):
    """Run a CorpusCounter against an in-memory dataset instead of the Hub."""
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: dataset)
    config = CorpusConfig(datasets=[DatasetSpec(path="fake", streaming=False)], **corpus_kwargs)
    return CorpusCounter(tokenizer, config, seed=0, sample_size=3).count()


def test_counting_accumulates_ids_and_documents(wordpiece, corpus_dataset, monkeypatch):
    counts = _count_over(wordpiece, corpus_dataset, monkeypatch, batch_size=3)
    assert counts.num_documents == len(corpus_dataset)
    assert counts.total_num_tokens > 0
    assert counts.counts[wordpiece.convert_tokens_to_ids("the")] > 0
    # Special tokens are structural, so they must not inflate the frequency ranking.
    assert counts.counts[wordpiece.convert_tokens_to_ids("[CLS]")] == 0


def test_counting_keeps_a_bounded_reservoir_of_samples(wordpiece, corpus_dataset, monkeypatch):
    counts = _count_over(wordpiece, corpus_dataset, monkeypatch, batch_size=2)
    assert len(counts.samples) == 3
    assert all(sample in corpus_dataset["text"] for sample in counts.samples)


def test_weights_scale_a_corpus_contribution(wordpiece, corpus_dataset, monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    plain = CorpusCounter(
        wordpiece, CorpusConfig(datasets=[DatasetSpec(path="f", streaming=False)]), sample_size=2
    ).count()
    doubled = CorpusCounter(
        wordpiece, CorpusConfig(datasets=[DatasetSpec(path="f", streaming=False, weight=2.0)]), sample_size=2
    ).count()
    assert doubled.total_num_tokens == 2 * plain.total_num_tokens


def test_a_missing_text_column_says_which_columns_exist(wordpiece, corpus_dataset, monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    config = CorpusConfig(datasets=[DatasetSpec(path="f", streaming=False, text_column="body")])
    with pytest.raises(KeyError, match="text"):
        CorpusCounter(wordpiece, config).count()


def test_max_samples_stops_early(wordpiece, corpus_dataset, monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    config = CorpusConfig(datasets=[DatasetSpec(path="f", streaming=False, max_samples=4)], batch_size=2)
    assert CorpusCounter(wordpiece, config).count().num_documents == 4


def test_empty_documents_are_skipped(wordpiece, monkeypatch):
    from datasets import Dataset

    dataset = Dataset.from_dict({"text": ["the cat", "", "the dog", None]})
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: dataset)
    config = CorpusConfig(datasets=[DatasetSpec(path="f", streaming=False)])

    assert CorpusCounter(wordpiece, config).count().num_documents == 2


def test_num_proc_is_only_passed_for_non_streaming_datasets(wordpiece, corpus_dataset, monkeypatch):
    seen: list[dict] = []

    def record(*args, **kwargs):
        seen.append(kwargs)
        return corpus_dataset

    monkeypatch.setattr("datasets.load_dataset", record)
    datasets = [DatasetSpec(path="f", streaming=False), DatasetSpec(path="g", streaming=True)]
    CorpusCounter(wordpiece, CorpusConfig(datasets=datasets, num_proc=2)).count()

    # `num_proc` is a download/prepare option; streaming datasets reject it.
    assert seen[0]["num_proc"] == 2
    assert "num_proc" not in seen[1]


def test_the_cache_is_written_then_reused(wordpiece, corpus_dataset, monkeypatch, tmp_path):
    cache = tmp_path / "counts.json"
    config = CorpusConfig(datasets=[DatasetSpec(path="f", streaming=False)], counts_cache=cache)
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: corpus_dataset)
    first = CorpusCounter(wordpiece, config).count()
    assert cache.exists()

    def explode(*args, **kwargs):
        raise AssertionError("the corpus should not be re-read when a cache exists")

    monkeypatch.setattr("datasets.load_dataset", explode)
    assert CorpusCounter(wordpiece, config).count().counts == first.counts
