"""Count corpus token frequencies once and cache them to JSON.

Counting is the expensive part of a trimming run and it does not depend on the selection
policy, so do it once here and point `corpus.counts_cache` at the output to reuse it
across selection experiments.

    trimbed count --config my_config.yaml -o counts.json
    trimbed count --config my_config.yaml -o counts.json \
        corpus.batch_size=4000 corpus.num_proc=8

Only the corpus half of the config matters here. The selection and embedding sections are
still validated, but they are not used.
"""

from __future__ import annotations

import argparse

from trimbed._logging import configure_logging
from trimbed.config import load_config, parse_overrides
from trimbed.counting import CorpusCounter
from trimbed.loading import load_tokenizer


def run(
    output: str,
    config: str | None = None,
    model: str | None = None,
    overrides: list[str] | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Count the configured corpus and write the frequencies to a cache file.

    Args:
        output: JSON file to write the counts to.
        config: Path to a YAML configuration file.
        model: Hub model id or local path, overriding the config.
        overrides: `key=value` strings applied on top of the config and flags.
        verbose: Emit debug logging.
        quiet: Only emit warnings and errors.

    Raises:
        ValueError: If the resolved config names no datasets to count over.
    """
    configure_logging(verbose=verbose, quiet=quiet)

    trim_config = load_config(config, model).with_overrides({"model": model, **parse_overrides(overrides or [])})
    if not trim_config.corpus.datasets:
        raise ValueError("no datasets configured under 'corpus.datasets'; nothing to count")

    tokenizer = load_tokenizer(trim_config.model, trim_config.revision, trim_config.trust_remote_code)
    counter = CorpusCounter(
        tokenizer, trim_config.corpus, seed=trim_config.seed, sample_size=trim_config.verify_samples
    )
    counts = counter.count()
    counts.save(output)
    print(
        f"counted {counts.total_num_tokens:,} tokens ({counts.distinct_tokens:,} distinct) "
        f"over {counts.num_documents:,} documents -> {output}"
    )


DESCRIPTION = (
    "Tokenize the configured corpus and write the per-token frequencies to JSON, so"
    " later trimming runs can reuse them instead of re-reading the corpus."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the counting arguments to `parser`.

    Args:
        parser: The `trimbed count` subparser to populate.
    """
    parser.add_argument("-c", "--config", help="Path to a YAML configuration file.")
    parser.add_argument("-m", "--model", help="Hub model id or local path (overrides the config).")
    parser.add_argument("-o", "--output", required=True, help="JSON file to write the counts to.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Emit debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only emit warnings and errors.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides, e.g. corpus.batch_size=4000",
    )
