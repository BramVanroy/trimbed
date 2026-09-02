"""Diff two tokenizers and print what the difference between them amounts to.

Point it at a checkpoint and a trimmed version of it to find out what the trim actually
did: whether the smaller vocabulary is a subset of the larger one and still in the
original order, whether the added, special and post-processor tokens survived, which
presets and Unicode scripts were kept or gutted, and how much longer the same text
encodes. Neither model's weights are read and nothing is written unless you ask for JSON.

    trimbed compare clips/e5-small-trm-nl clips/e5-small-trm
    trimbed compare bert-base-multilingual-cased trimmed/bert --text-file dutch.txt
    trimbed compare base/ trimmed/ --preset script:Latin --preset script:Cyrillic -o diff.json

The comparison is directional: everything is counted against the first tokenizer, so
`LATIN 45,102/60,003` means the base had 60,003 Latin tokens and 45,102 of them survive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trimbed._logging import configure_logging
from trimbed.compare import compare_tokenizers
from trimbed.loading import load_tokenizer
from trimbed.spec import TokenizerSpec


def run(
    base: str,
    other: str,
    presets: list[str] | None = None,
    texts: list[str] | None = None,
    text_file: str | None = None,
    examples: int = 10,
    output: str | None = None,
    trust_remote_code: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Compare two tokenizers and print the diff.

    Args:
        base: Hub model id or local path of the original tokenizer.
        other: Hub model id or local path of the tokenizer to compare against it.
        presets: Extra preset names to resolve, which is how the parametrised ones are
            reached, e.g. `["script:Latin"]`.
        texts: Sample texts to encode with both tokenizers.
        text_file: Text file holding one further sample text per line.
        examples: How many removed and introduced tokens to quote.
        output: JSON file to write the full report to.
        trust_remote_code: Allow tokenizer code shipped with either checkpoint.
        verbose: Emit debug logging.
        quiet: Only emit warnings and errors.
    """
    configure_logging(verbose=verbose, quiet=quiet)

    sample_texts = list(texts or [])
    if text_file is not None:
        lines = Path(text_file).read_text(encoding="utf-8").splitlines()
        sample_texts += [line for line in lines if line.strip()]

    base_spec = TokenizerSpec.from_tokenizer(load_tokenizer(base, trust_remote_code=trust_remote_code), source=base)
    other_spec = TokenizerSpec.from_tokenizer(load_tokenizer(other, trust_remote_code=trust_remote_code), source=other)

    report = compare_tokenizers(base_spec, other_spec, texts=sample_texts, presets=presets or [], examples=examples)
    print(report.render())
    if output is not None:
        print(f"full report -> {report.save(output)}")


DESCRIPTION = (
    "Diff two tokenizers: how the vocabularies relate as sets, which structural tokens,"
    " presets and Unicode scripts survived, and what the same text costs in both. Reads"
    " no weights and no corpus."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the comparison arguments to `parser`.

    Args:
        parser: The `trimbed compare` subparser to populate.
    """
    parser.add_argument("base", help="Hub model id or local path of the original tokenizer.")
    parser.add_argument("other", help="Hub model id or local path of the tokenizer to compare against it.")
    parser.add_argument(
        "--preset",
        dest="presets",
        action="append",
        help="Extra preset to report on, repeatable, e.g. --preset script:Latin. The plain presets"
        " are always included; this is how the parametrised ones are reached.",
    )
    parser.add_argument(
        "--text",
        dest="texts",
        action="append",
        help="Sample text to encode with both tokenizers, repeatable. Without any text the encoding"
        " comparison is skipped.",
    )
    parser.add_argument("--text-file", help="Text file holding one further sample text per line.")
    parser.add_argument(
        "--examples", type=int, default=10, help="How many removed and introduced tokens to quote (default: 10)."
    )
    parser.add_argument("-o", "--output", help="JSON file to write the full report to.")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow tokenizer code shipped with either checkpoint."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Emit debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only emit warnings and errors.")
