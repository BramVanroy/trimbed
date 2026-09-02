"""The main `trimbed` command.

    trimbed inspect --model codefuse-ai/F2LLM-v2-160M
    trimbed trim --config my_config.yaml --dry-run selection.top_k=30000
    trimbed presets

Each subcommand owns its own arguments, in `add_arguments`, next to the `run` they feed.
This module holds the table of subcommand names and nothing else, so adding a command is
a new module plus one line here.
"""

from __future__ import annotations

import argparse
from types import ModuleType

from trimbed.cli import compare_tokenizers, count_tokens, inspect_tokenizer, list_presets, trim_vocab


# subcommand name -> the module implementing it and the one line `trimbed --help` shows.
COMMANDS: dict[str, tuple[ModuleType, str]] = {
    "trim": (trim_vocab, "Trim a tokenizer, and optionally its model, and write the result."),
    "count": (count_tokens, "Count the corpus once and cache the frequencies to JSON."),
    "inspect": (inspect_tokenizer, "Describe a tokenizer as JSON, changing nothing."),
    "compare": (compare_tokenizers, "Diff two tokenizers, e.g. a checkpoint and a trimmed version of it."),
    "presets": (list_presets, "List the registered presets that --keep-preset accepts."),
}


def main() -> None:
    """Parse the command line and hand the arguments to the chosen subcommand's `run`."""
    parser = argparse.ArgumentParser(
        prog="trimbed",
        description=(
            "Trim a tokenizer's vocabulary, and optionally its model's embedding table, down"
            " to the subset a corpus and your must-keep rules actually need."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (module, summary) in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=summary, description=module.DESCRIPTION)
        module.add_arguments(subparser)
        subparser.set_defaults(run=module.run)

    arguments = vars(parser.parse_args())
    command = arguments.pop("run")
    del arguments["command"]
    command(**arguments)


if __name__ == "__main__":
    main()
