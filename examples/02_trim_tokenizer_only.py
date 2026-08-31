"""Trim without a corpus, using must-keep rules alone.

The smallest useful trim: keep the presets you name, a handful of literal tokens and the
prompts you cannot afford to have fragment, drop everything else, and leave the model
untouched. Useful when you know exactly which character classes your downstream text uses.

    python examples/02_trim_tokenizer_only.py
    python examples/02_trim_tokenizer_only.py --model ./my-checkpoint --output-dir trimmed/mine
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trimbed import TrimConfig, TrimPipeline, TrimReport


DEFAULT_MODEL = "google-bert/bert-base-multilingual-cased"
DEFAULT_OUTPUT_DIR = "trimmed/ascii-only"


def run(model: str = DEFAULT_MODEL, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> TrimReport:
    """Keep only ASCII-ish tokens and write the trimmed tokenizer out.

    Args:
        model: Hub model id or local path.
        output_dir: Where the trimmed tokenizer is written.

    Returns:
        The report describing the run.
    """
    config = TrimConfig(
        model=model,
        output_dir=Path(output_dir),
        # No corpus, so nothing is counted; the keep_* rules are the whole selection.
        trim_model=False,
        overwrite=True,
        selection={
            "keep_presets": ["alphanumeric", "punctuation", "whitespace"],
            "keep_tokens": ["€", "£"],
            # Encoded with the original tokenizer, so these keep the ids they have now
            # instead of fragmenting into characters. `keep_chat_template` does the same
            # for the tokenizer's own template, and is on by default.
            "keep_texts": ["### Instruction:\n", "### Response:\n"],
        },
    )
    report = TrimPipeline(config).run()
    print(report.render())
    return report


def main() -> None:
    """Parse the command line and run the trim."""
    parser = argparse.ArgumentParser(
        description="Trim a tokenizer to must-keep presets and literal tokens alone, leaving the model untouched."
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Hub model id or local path.")
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where the trimmed tokenizer is written."
    )
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
