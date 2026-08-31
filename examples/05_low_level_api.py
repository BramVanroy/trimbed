"""Drive the stages yourself, without the pipeline.

`TrimPipeline` is a convenience wrapper around stages that are independent, so you can
just as well recombine them yourself. Here the kept set is computed by hand, the
tokenizer is trimmed in memory, and the result is checked against the original. Nothing
is written to disk.

    python examples/05_low_level_api.py
    python examples/05_low_level_api.py --model ./my-checkpoint
"""

from __future__ import annotations

import argparse

from trimbed import SelectionConfig, TokenizerSpec, VerificationReport, select_tokens, trim_tokenizer, verify_tokenizer
from trimbed.loading import load_tokenizer


DEFAULT_MODEL = "google-bert/bert-base-multilingual-cased"

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Trimming a vocabulary should not change how text is encoded.",
    "1234567890 !?.,;:",
]


def run(model: str = DEFAULT_MODEL) -> VerificationReport:
    """Select, trim and verify in three explicit steps.

    Args:
        model: Hub model id or local path.

    Returns:
        The verification result comparing both tokenizers on `TEXTS`.
    """
    tokenizer = load_tokenizer(model)
    spec = TokenizerSpec.from_tokenizer(tokenizer, source=model)

    # 1. Decide what survives. Structural tokens are added automatically, and BPE merge
    #    ancestors are pulled in so no kept token becomes unreachable.
    selection = select_tokens(spec, None, SelectionConfig(keep_presets=["ascii_printable"]))
    print(f"keeping {len(selection):,} of {spec.vocab_size:,} tokens")
    for reason, count in selection.counts_by_reason().items():
        print(f"  {reason:<24} {count:,}")

    # 2. Apply it. The result is a reloaded tokenizer plus the old-id -> new-id mapping.
    trimmed = trim_tokenizer(tokenizer, spec, selection.kept_ids)

    # 3. Prove it. Identical ids mean the trim was a pure renumbering; equivalent text
    #    means a merge was lost but the string still round-trips.
    result = verify_tokenizer(tokenizer, trimmed.tokenizer, trimmed.remap, TEXTS)
    print(f"{result.identical}/{result.checked} identical, {result.equivalent_text}/{result.checked} equivalent")
    return result


def main() -> None:
    """Parse the command line and run the select/trim/verify sequence."""
    parser = argparse.ArgumentParser(
        description="Select, trim and verify a tokenizer as three separate calls, writing nothing to disk."
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Hub model id or local path.")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
