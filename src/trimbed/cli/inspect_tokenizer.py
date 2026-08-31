"""Print a tokenizer's shape and whether trimbed can trim it.

Run this before a trimming job: it reports the backend family (BPE, WordPiece, Unigram,
WordLevel), the vocabulary size, how many tokens are added or special (those are
structural and never removed), whether it ships a chat template, and whether a backend is
registered for the family at all.

    trimbed inspect --model codefuse-ai/F2LLM-v2-160M
    trimbed inspect --config configs/f2llm_dutch.yaml

Nothing is loaded beyond the tokenizer and nothing is written. The model weights are never
touched.
"""

from __future__ import annotations

import argparse
import json

from trimbed._logging import configure_logging
from trimbed.backends import supported_model_types
from trimbed.config import load_config
from trimbed.loading import load_tokenizer
from trimbed.spec import TokenizerSpec


def run(config: str | None = None, model: str | None = None, verbose: bool = False, quiet: bool = False) -> None:
    """Describe the tokenizer as JSON on stdout.

    Args:
        config: Path to a YAML configuration file.
        model: Hub model id or local path, overriding the config.
        verbose: Emit debug logging.
        quiet: Only emit warnings and errors.
    """
    configure_logging(verbose=verbose, quiet=quiet)

    trim_config = load_config(config, model).with_overrides({"model": model})
    tokenizer = load_tokenizer(trim_config.model, trim_config.revision, trim_config.trust_remote_code)
    spec = TokenizerSpec.from_tokenizer(tokenizer, source=trim_config.model)
    summary = spec.describe()
    summary["supported"] = spec.model_type in supported_model_types()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


DESCRIPTION = (
    "Describe a tokenizer as JSON (backend family, vocabulary size, added and"
    " special tokens) and say whether trimbed supports the family. Changes nothing."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the inspection arguments to `parser`.

    Args:
        parser: The `trimbed inspect` subparser to populate.
    """
    parser.add_argument(
        "-c",
        "--config",
        help="Path to a YAML configuration file; only its model, revision and trust_remote_code are read.",
    )
    parser.add_argument("-m", "--model", help="Hub model id or local path (overrides the config).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Emit debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only emit warnings and errors.")
