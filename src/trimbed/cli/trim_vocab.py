"""Trim a tokenizer's vocabulary down to a useful subset, and optionally its model with it.

The run is described by a YAML config, documented field by field in the configuration
guide. Three layers stack, later winning over earlier: the YAML file, the typed flags
below, and the trailing `key=value` positionals. The typed flags cover the knobs worth
tuning between runs, and the positionals reach every remaining field without needing a
flag for each one.

    trimbed trim --config my_config.yaml
    trimbed trim --config my_config.yaml --dry-run \
        selection.top_k=30000 corpus.batch_size=2000

A config is not required: `--model` plus must-keep rules is enough to trim without a
corpus, which is the fastest way to see what the machinery does.

    trimbed trim --model google-bert/bert-base-multilingual-cased \
        --keep-preset alphanumeric --output-dir trimmed/bert --no-trim-model
"""

from __future__ import annotations

import argparse

from trimbed._logging import configure_logging
from trimbed.config import load_config, parse_overrides
from trimbed.pipeline import TrimPipeline


def run(
    config: str | None = None,
    model: str | None = None,
    output_dir: str | None = None,
    coverage: float | None = None,
    top_k: int | None = None,
    min_count: int | None = None,
    max_vocab_size: int | None = None,
    keep_presets: list[str] | None = None,
    keep_tokens: list[str] | None = None,
    keep_texts: list[str] | None = None,
    no_keep_chat_template: bool = False,
    no_trim_model: bool = False,
    no_verify: bool = False,
    no_verify_model: bool = False,
    trust_remote_code: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    overrides: list[str] | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Resolve the configuration, run the trimming pipeline and print its report.

    [`TrimConfig.with_overrides`][trimbed.config.TrimConfig.with_overrides] drops `None`
    values, so an argument that was not supplied leaves the config alone. The boolean
    flags therefore map to their meaningful value or to `None`, never to the config
    default. That is why a flag turning something off is spelled `--no-...` and one
    turning something on is not.

    Args:
        config: Path to a YAML configuration file.
        model: Hub model id or local path, overriding the config.
        output_dir: Directory to write the trimmed artefacts to.
        coverage: Keep tokens covering this fraction of corpus occurrences.
        top_k: Keep at most this many corpus-derived tokens.
        min_count: Keep tokens seen at least this many times.
        max_vocab_size: Hard cap on the final vocabulary size.
        keep_presets: Must-keep preset names.
        keep_tokens: Literal tokens to keep.
        keep_texts: Texts that must keep encoding the way they do now.
        no_keep_chat_template: Do not keep the tokens the chat template's own words need.
        no_trim_model: Trim only the tokenizer, not the model.
        no_verify: Skip the round-trip verification pass.
        no_verify_model: Skip running both models and comparing their outputs.
        trust_remote_code: Allow custom code shipped with the checkpoint.
        overwrite: Allow writing into a non-empty output directory.
        dry_run: Select and report without writing anything.
        overrides: `key=value` strings applied on top of the config and flags.
        verbose: Emit debug logging.
        quiet: Only emit warnings and errors.
    """
    configure_logging(verbose=verbose, quiet=quiet)

    flag_overrides = {
        "model": model,
        "output_dir": output_dir,
        "overwrite": True if overwrite else None,
        "trim_model": False if no_trim_model else None,
        "verify": False if no_verify else None,
        "verify_model": False if no_verify_model else None,
        "trust_remote_code": True if trust_remote_code else None,
        "selection.coverage": coverage,
        "selection.top_k": top_k,
        "selection.min_count": min_count,
        "selection.max_vocab_size": max_vocab_size,
        "selection.keep_presets": keep_presets,
        "selection.keep_tokens": keep_tokens,
        "selection.keep_texts": keep_texts,
        "selection.keep_chat_template": False if no_keep_chat_template else None,
    }
    trim_config = load_config(config, model).with_overrides({**flag_overrides, **parse_overrides(overrides or [])})
    report = TrimPipeline(trim_config).run(dry_run=dry_run)
    print(report.render())


DESCRIPTION = (
    "Trim a tokenizer's vocabulary to the tokens a corpus actually uses plus explicit"
    " must-keep rules, and gather the model's embedding table and output head down to"
    " match. Writes the trimmed artefacts, a report and the resolved config."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the trim arguments to `parser`.

    Args:
        parser: The `trimbed trim` subparser to populate.
    """
    parser.add_argument("-c", "--config", help="Path to a YAML configuration file.")
    parser.add_argument("-m", "--model", help="Hub model id or local path (overrides the config).")
    parser.add_argument("-o", "--output-dir", help="Directory to write the trimmed artefacts to.")
    parser.add_argument("--coverage", type=float, help="Keep tokens covering this fraction of corpus occurrences.")
    parser.add_argument("--top-k", type=int, help="Keep at most this many corpus-derived tokens.")
    parser.add_argument("--min-count", type=int, help="Keep tokens seen at least this many times.")
    parser.add_argument("--max-vocab-size", type=int, help="Hard cap on the final vocabulary size.")
    parser.add_argument(
        "--keep-preset",
        action="append",
        dest="keep_presets",
        metavar="NAME",
        help="Must-keep preset; repeatable. Use 'trimbed presets' to list them.",
    )
    parser.add_argument(
        "--keep-token", action="append", dest="keep_tokens", metavar="TOKEN", help="Literal token to keep; repeatable."
    )
    parser.add_argument(
        "--keep-text",
        action="append",
        dest="keep_texts",
        metavar="TEXT",
        help="Text that must keep encoding as it does now; repeatable. Its tokens are kept.",
    )
    parser.add_argument(
        "--no-keep-chat-template",
        action="store_true",
        help="Do not keep the tokens the chat template's own words need; they will fragment.",
    )
    parser.add_argument("--no-trim-model", action="store_true", help="Trim only the tokenizer, not the model.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the round-trip verification pass.")
    parser.add_argument(
        "--no-verify-model",
        action="store_true",
        help="Skip running both models and comparing their outputs; that check loads the model twice more.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code shipped with the checkpoint, as gte and jina models need.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Select and report without writing anything.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Emit debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only emit warnings and errors.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides, e.g. selection.max_vocab_size=32000",
    )
