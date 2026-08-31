"""Look at a tokenizer before trimming it.

Answers the three questions that decide whether a trim will go well: which family the
tokenizer belongs to, how big its id space really is, and how many of its tokens are
structural and therefore not up for removal.

    python examples/01_inspect_tokenizer.py
    python examples/01_inspect_tokenizer.py --model google-bert/bert-base-multilingual-cased

`trimbed inspect` is the same thing as a command, config file included. This
example is the library call that it wraps.
"""

from __future__ import annotations

import argparse

from trimbed import TokenizerSpec, supported_model_types
from trimbed.loading import load_tokenizer


DEFAULT_MODEL = "codefuse-ai/F2LLM-v2-160M"


def run(model: str = DEFAULT_MODEL) -> dict[str, str | int | bool | None]:
    """Describe a tokenizer's shape.

    Args:
        model: Hub model id or local path.

    Returns:
        The summary that was printed.
    """
    spec = TokenizerSpec.from_tokenizer(load_tokenizer(model), source=model)
    summary = spec.describe()

    print(f"{model}: {summary['model_type']} with {summary['vocab_size']:,} tokens")
    print(f"  supported by trimbed   {summary['model_type'] in supported_model_types()}")
    print(f"  added tokens           {summary['added_tokens']} ({summary['special_tokens']} special)")
    print(f"  byte-level             {summary['uses_byte_level']}")

    # Structural tokens survive every trim, so they are the floor on the final size.
    structural = spec.backend.structural_tokens(spec)
    print(f"  structural tokens      {len(structural) + len(spec.added_token_ids):,} (never removable)")
    return summary


def main() -> None:
    """Parse the command line and describe the tokenizer."""
    parser = argparse.ArgumentParser(
        description="Report a tokenizer's family, id-space size and structural floor. Changes nothing."
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Hub model id or local path.")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
