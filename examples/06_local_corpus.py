"""Trim a local checkpoint over a corpus of local files, with nothing coming from the Hub.

A corpus usually exists as files long before it exists as a Hub dataset. `path` is the
first argument `datasets.load_dataset` takes, so the file case is spelled by naming the
loader in `path` and the files in `data_files`. Two other spellings do the same job:
`path` may be a directory, and `load_from_disk: true` reads a directory written by
`save_to_disk`.

    python examples/06_local_corpus.py --model ./my-model --corpus ./data/dutch
    python examples/06_local_corpus.py --model ./my-model --corpus ./data/dutch \
        --output-dir trimmed/my-model-nl

Both paths are local, so this runs offline. The tokenizer is trimmed on its own; see
`03_trim_with_corpus.py` for the run that takes the model's embedding table with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trimbed import TrimConfig, TrimPipeline, TrimReport


DEFAULT_OUTPUT_DIR = "trimmed/local"


def run(
    model: str,
    corpus: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> TrimReport:
    """Trim a checkpoint on disk to the vocabulary a corpus on disk actually uses.

    Args:
        model: Local checkpoint directory, or a Hub model id.
        corpus: Directory holding the corpus as JSON Lines files.
        output_dir: Where the trimmed tokenizer is written.

    Returns:
        The report describing the run.
    """
    config = TrimConfig(
        model=model,
        output_dir=Path(output_dir),
        trim_model=False,
        overwrite=True,
        corpus={
            "datasets": [
                {
                    "path": "json",
                    "data_files": f"{Path(corpus)}/*.jsonl",
                    "text_column": "text",
                    "streaming": True,
                }
            ]
        },
        selection={
            "min_count": 2,
            "keep_presets": ["digits", "punctuation"],
        },
    )
    report = TrimPipeline(config).run()
    print(report.render())
    return report


def main() -> None:
    """Parse the command line and run the trim."""
    parser = argparse.ArgumentParser(
        description="Trim a local checkpoint over a local corpus of JSON Lines files, offline."
    )
    parser.add_argument("-m", "--model", required=True, help="Local checkpoint directory, or a Hub model id.")
    parser.add_argument("-d", "--corpus", required=True, help="Directory holding the corpus as JSON Lines files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where the trimmed tokenizer is written.",
    )
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
