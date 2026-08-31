"""List the must-keep presets that `--keep-preset` and `selection.keep_presets` accept.

A preset is a named rule for tokens that must survive the trim regardless of what the
corpus says: the byte alphabet, punctuation, a whole Unicode script. Names ending in
`:...` are parametrised: supply the argument after the colon, e.g. `script:Latin`.

    trimbed presets

`trimbed.presets.render_presets()` returns the same text, for a program that wants it.

The structural presets are printed apart from the rest, because the trim keeps those
tokens whether or not you name them. Naming one is still how a run without a corpus gets
a must-keep source, which is what `trimbed trim --model ...` does by default.

The registry is user-extensible, so this reflects whatever `@register_preset` has been
applied by the time it runs. See `examples/04_custom_preset.py` for how to add your own.
"""

from __future__ import annotations

import argparse

from trimbed._logging import configure_logging
from trimbed.presets import render_presets


def run(verbose: bool = False, quiet: bool = False) -> None:
    """Print every registered preset with what it selects, structural ones first.

    Args:
        verbose: Emit debug logging.
        quiet: Only emit warnings and errors.
    """
    configure_logging(verbose=verbose, quiet=quiet)
    print(render_presets())


DESCRIPTION = "Print every registered must-keep preset with what it selects. Loads no tokenizer."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the listing arguments to `parser`.

    Args:
        parser: The `trimbed presets` subparser to populate.
    """
    parser.add_argument("-v", "--verbose", action="store_true", help="Emit debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only emit warnings and errors.")
