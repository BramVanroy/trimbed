"""Find out what somebody else's trim actually did.

A trimmed checkpoint's card tells you the recipe (this corpus, this target size) but not
the result. This compares the two vocabularies directly: whether the smaller one is a
subset of the larger, renumbered in place, whether anything structural went missing, and
which presets and scripts paid for the reduction.

    python examples/07_compare_tokenizers.py
    python examples/07_compare_tokenizers.py --model ./bert-base --other ./bert-trimmed

`trimbed compare` is the same thing as a command. This is the library call it wraps.
"""

from __future__ import annotations

import argparse

from trimbed import ComparisonReport, TokenizerSpec, compare_tokenizers
from trimbed.loading import load_tokenizer


DEFAULT_MODEL = "clips/e5-small-trm-nl"
DEFAULT_OTHER = "clips/e5-small-trm"

TEXTS = [
    "De kat zat op de mat.",
    "Trimmen mag niet veranderen hoe tekst gecodeerd wordt.",
    "The quick brown fox jumps over the lazy dog.",
]


def run(model: str = DEFAULT_MODEL, other: str = DEFAULT_OTHER) -> ComparisonReport:
    """Diff two tokenizers and print the result.

    Args:
        model: Hub model id or local path of the original tokenizer.
        other: Hub model id or local path of the tokenizer to compare against it.

    Returns:
        The comparison, whose `render()` is what was printed.
    """
    base_spec = TokenizerSpec.from_tokenizer(load_tokenizer(model), source=model)
    other_spec = TokenizerSpec.from_tokenizer(load_tokenizer(other), source=other)

    # Naming a parametrised preset is how a specific script gets its own line; the plain
    # presets are reported whether or not you ask for them.
    report = compare_tokenizers(base_spec, other_spec, texts=TEXTS, presets=["script:Latin"])
    print(report.render())

    # Everything the rendering summarises is on the report as numbers, so a script can
    # decide for itself whether a trim is worth using.
    if not report.vocabulary.is_subset:
        print(f"\n{other} is not a trim of {model}: it has {report.vocabulary.introduced:,} tokens {model} lacks")
    if report.components.structural_break:
        print(f"\n{other} lost structural tokens: {report.components.removed_post_processor_tokens}")
    return report


def main() -> None:
    """Parse the command line and compare the two tokenizers."""
    parser = argparse.ArgumentParser(
        description="Diff two tokenizers and report how their vocabularies, structural tokens and encodings differ."
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Hub model id or local path.")
    parser.add_argument("-o", "--other", default=DEFAULT_OTHER, help="The tokenizer to compare against it.")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
