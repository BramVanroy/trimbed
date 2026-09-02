"""Counting how often each token id occurs across one or more Hugging Face datasets."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from trimbed._logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from trimbed.config import CorpusConfig, DatasetSpec

logger = get_logger(__name__)


@dataclass
class CorpusCounts:
    """Token-frequency statistics gathered over a corpus.

    Attributes:
        counts: Occurrences per token id, weighted per dataset, e.g.
            `Counter({409: 91204, 13: 88317, ...})`.
        total_num_tokens: Sum of all counts.
        num_documents: Number of examples read.
        samples: A reservoir sample of raw texts, reused to verify the trimmed tokenizer.
    """

    counts: Counter[int] = field(default_factory=Counter)
    total_num_tokens: int = 0
    num_documents: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def distinct_tokens(self) -> int:
        """Return how many distinct token ids the corpus used at least once."""
        return len(self.counts)

    def coverage_of(self, token_ids: Iterable[int]) -> float:
        """Return the fraction of corpus occurrences covered by a set of ids.

        Args:
            token_ids: The ids that survive the trim.

        Returns:
            A value in `[0, 1]`, e.g. `0.9993`. Token frequency is skewed enough that
            even an aggressive trim lands near 1, so the interesting digits are the last
            ones. That is why the report prints four decimal places rather than two.
        """
        if self.total_num_tokens == 0:
            return 1.0
        covered = sum(self.counts.get(token_id, 0) for token_id in set(token_ids))
        return covered / self.total_num_tokens

    def ranked_ids(self) -> list[int]:
        """Return token ids sorted by descending frequency, ties broken by id to ensure determinism."""
        return sorted(self.counts, key=lambda token_id: (-self.counts[token_id], token_id))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        The counter's integer keys become strings, since JSON has no integer keys, e.g.
        `{"counts": {"13": 88317, "409": 91204}, "total_num_tokens": ..., "num_documents": ...,
        "samples": [...]}`. `from_dict` converts them back.
        """
        return {
            "counts": {str(token_id): count for token_id, count in sorted(self.counts.items())},
            "total_num_tokens": self.total_num_tokens,
            "num_documents": self.num_documents,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Rebuild counts from `to_dict` output.

        Args:
            payload: The decoded JSON mapping.

        Returns:
            The reconstructed counts.
        """
        return cls(
            counts=Counter({int(token_id): int(count) for token_id, count in payload["counts"].items()}),
            total_num_tokens=int(payload["total_num_tokens"]),
            num_documents=int(payload["num_documents"]),
            samples=list(payload.get("samples", [])),
        )

    def save(self, path: str | Path) -> None:
        """Write the counts to a JSON file, creating parent directories as needed.

        Args:
            path: Destination file.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")
        logger.info(f"wrote token counts to {destination!r}")

    @classmethod
    def load_from_file(cls, path: str | Path) -> Self:
        """Read counts back from a JSON file.

        Args:
            path: Source file written by `save`.

        Returns:
            The reconstructed counts.
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class CorpusCounter:
    """Tokenizes configured datasets and accumulates per-id frequencies."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, config: CorpusConfig, *, seed: int = 0, sample_size: int = 256
    ) -> None:
        """Build a counter.

        Args:
            tokenizer: A fast tokenizer used to encode the corpus.
            config: A CorpusConfig which specifies the datasets to read and how.
            seed: Seed for the verification-sample reservoir.
            sample_size: How many raw texts to retain for later verification, e.g. 256,
                to verify the tokenizer and the model comparison then uses only the first handful.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.sample_size = sample_size
        self._rng = random.Random(seed)

    def count(self) -> CorpusCounts:
        """Count token occurrences across every configured dataset.

        Reads from [`CorpusConfig.counts_cache`][trimbed.config.CorpusConfig.counts_cache]
        when that file exists, and writes to it otherwise, so an expensive pass can be
        reused across trimming runs.

        Returns:
            The accumulated statistics.
        """
        cache = self.config.counts_cache
        if cache is not None and Path(cache).exists():
            logger.info(f"reusing cached token counts from {cache!r}")
            return CorpusCounts.load_from_file(cache)

        totals = CorpusCounts()
        for spec in self.config.datasets:
            self._count_dataset(spec, totals)
        logger.info(
            f"counted {totals.total_num_tokens:,} tokens ({totals.distinct_tokens:,} distinct)"
            f" over {totals.num_documents:,} documents"
        )
        if cache is not None:
            totals.save(cache)
        return totals

    def _count_dataset(self, spec: DatasetSpec, totals: CorpusCounts) -> None:
        """Accumulate one dataset's contribution into `totals`.

        Args:
            spec: The dataset to read.
            totals: Accumulator, mutated in place.
        """
        logger.info(f"counting tokens over {spec.path!r} (split={spec.split}, streaming={spec.streaming})")
        seen = 0
        for batch in self._iter_batches(spec):
            self._reservoir_extend(totals, batch, totals.num_documents + seen)
            seen += len(batch)
            encodings = self.tokenizer(batch, add_special_tokens=False).input_ids
            # Add weighted counts for each token, rounded to the nearest integer
            # The weight is determined on a per-dataset basis
            for ids in encodings:
                if spec.weight == 1.0:
                    totals.counts.update(ids)
                    totals.total_num_tokens += len(ids)
                else:
                    for token_id, count in Counter(ids).items():
                        weighted = round(count * spec.weight)
                        totals.counts[token_id] += weighted
                        totals.total_num_tokens += weighted
        totals.num_documents += seen
        logger.info(f"read {seen:,} documents from {spec.path!r}")

    def _open_dataset(self, spec: DatasetSpec) -> Iterable[dict[str, Any]]:
        """Open one configured dataset, wherever it lives.

        NOTE: self.config.num_proc is ignored in streaming mode and is only used to
        speed up data (down)loading. It is NOT used to parallellize token counting since
        we rely purely on fast tokenizers, which are already highly parallellized.

        Args:
            spec: The dataset to read, e.g. `path="epfml/FineWeb2-HQ"`, `name="nld_Latn"`,
                `split="train"`, or `path="json"` with `data_files="./data/*.jsonl"`.

        Returns:
            The dataset to iterate over, streamed or in memory.

        Raises:
            KeyError: If a `save_to_disk` directory holds several splits and the
                configured one is not among them.
        """
        # `datasets` costs about ten seconds to import and only this method needs it,
        # so just load it here lazily
        from datasets import DatasetDict, load_dataset, load_from_disk

        if spec.load_from_disk:
            dataset = load_from_disk(spec.path)
            if not isinstance(dataset, DatasetDict):
                return dataset
            if spec.split not in dataset:
                raise KeyError(f"{spec.path!r} holds splits {sorted(dataset)}, so there is no {spec.split!r} to read")
            return dataset[spec.split]

        kwargs: dict[str, Any] = {
            "split": spec.split,
            "streaming": spec.streaming,
            "revision": spec.revision,
            "data_dir": spec.data_dir,
            "data_files": spec.data_files,
        }
        # `num_proc` is ignored in streaming mode, so don't pass it there or the datasets library complains
        if not spec.streaming and self.config.num_proc:
            kwargs["num_proc"] = self.config.num_proc
        return load_dataset(spec.path, spec.name, **kwargs)

    def _iter_batches(self, spec: DatasetSpec) -> Iterator[list[str]]:
        """Yield batches of raw texts from one dataset.

        Args:
            spec: The dataset to read.

        Yields:
            Lists of text strings of at most
            [`CorpusConfig.batch_size`][trimbed.config.CorpusConfig.batch_size] entries,
            where only the last one is usually shorter.

        Raises:
            KeyError: If the configured text column is absent, e.g. the default `"text"`
                against a dataset whose column is `"content"`.
        """
        dataset = self._open_dataset(spec)

        batch: list[str] = []
        for index, example in enumerate(dataset):
            if spec.max_samples is not None and index >= spec.max_samples:
                break
            if spec.text_column not in example:
                raise KeyError(
                    f"column {spec.text_column!r} not found in {spec.path!r}; available columns: {sorted(example)}"
                )
            text = example[spec.text_column]
            if not text:
                continue
            batch.append(str(text))
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _reservoir_extend(self, totals: CorpusCounts, batch: list[str], start_index: int) -> None:
        """Randomly update the sample reservoir with news texts to attempt to get a representative sample.

        Args:
            totals: Accumulator holding the reservoir.
            batch: Texts just read.
            start_index: Global index of the first text in `batch`, across all datasets.
                It has to be global, not per-batch, or later documents would be
                over-represented: with a 256-slot reservoir, text 100,000 must have a
                256/100,000 chance of being kept, not 256/1,000.
        """
        for offset, text in enumerate(batch):
            index = start_index + offset
            if len(totals.samples) < self.sample_size:
                totals.samples.append(text)
            else:
                slot = self._rng.randint(0, index)
                if slot < self.sample_size:
                    totals.samples[slot] = text
