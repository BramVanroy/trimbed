"""Derive the vocabulary from a corpus and trim the model along with it.

This is the main use case: point at a dataset in the language you care about, keep the
tokens that account for almost all of its token occurrences, and gather the embedding
table down to the surviving rows.

    python examples/03_trim_with_corpus.py
    python examples/03_trim_with_corpus.py --dataset epfml/FineWeb2-HQ --output-dir trimmed/f2llm-nl

Needs the model extra for the embedding surgery: `uv sync --extra model`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trimbed import TrimConfig, TrimPipeline, TrimReport


DEFAULT_MODEL = "codefuse-ai/F2LLM-v2-160M"
DEFAULT_DATASET = "epfml/FineWeb2-HQ"
DEFAULT_OUTPUT_DIR = "trimmed/f2llm-nl"


def run(
    model: str = DEFAULT_MODEL,
    dataset: str = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> TrimReport:
    """Trim tokenizer and model to the vocabulary a Dutch corpus actually uses.

    Args:
        model: Hub model id or local path.
        dataset: Hub dataset id to count tokens over.
        output_dir: Where the trimmed tokenizer and model are written.

    Returns:
        The report describing the run.
    """
    config = TrimConfig(
        model=model,
        output_dir=Path(output_dir),
        trim_model=True,
        overwrite=True,
        # Re-encoding proves the vocabulary was renumbered faithfully; running both
        # models proves the embedding rows followed it. Costs two more model loads.
        verify_model=True,
        corpus={
            "datasets": [{"path": dataset, "name": "nld_Latn", "streaming": True, "max_samples": 20_000}],
            # Counting is the slow part. Cache it so selection can be re-tuned for free.
            "counts_cache": Path(output_dir) / "counts.json",
        },
        selection={
            # Keep the frequency ranking up to 99.9% of all occurrences, but never more
            # than 32k corpus-derived tokens.
            "coverage": 0.999,
            "top_k": 32_000,
            # Kept whether or not the corpus happens to use them.
            "keep_presets": ["digits", "punctuation"],
        },
        embeddings={"pad_to_multiple_of": 64},
    )
    report = TrimPipeline(config).run()
    print(report.render())
    return report


def main() -> None:
    """Parse the command line and run the trim."""
    parser = argparse.ArgumentParser(
        description=(
            "Count a corpus, keep the tokens it actually uses plus a few must-keep presets,"
            " and gather the model's embedding table down to match."
        )
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Hub model id or local path.")
    parser.add_argument("-d", "--dataset", default=DEFAULT_DATASET, help="Hub dataset id to count tokens over.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where the trimmed tokenizer and model are written.",
    )
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
