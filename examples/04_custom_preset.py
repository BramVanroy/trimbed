"""Add your own must-keep rule and use it by name.

A preset is a function from a tokenizer to a set of its tokens. Registering one makes it
addressable from YAML and from `selection.keep_presets`, exactly like the built-ins, so
project-specific vocabulary rules live in your code rather than in a token list you have
to regenerate whenever the tokenizer changes.

    python examples/04_custom_preset.py
    python examples/04_custom_preset.py --model ./my-checkpoint --output-dir trimmed/chemistry
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trimbed import TokenizerSpec, TrimConfig, TrimPipeline, TrimReport, register_preset


DEFAULT_MODEL = "google-bert/bert-base-multilingual-cased"
DEFAULT_OUTPUT_DIR = "trimmed/chemistry"

# A stand-in for a real element table; the point is that the rule lives in code.
CHEMICAL_ELEMENTS = frozenset(["H", "He", "Li", "C", "N", "O", "Na", "Mg", "Cl", "K", "Fe", "Cu", "Zn", "Ag", "Au"])


@register_preset("chemical_elements")
def chemical_elements(spec: TokenizerSpec) -> set[str]:
    """Tokens spelling a chemical element symbol.

    Presets match on the decoded surface form, so this catches `Fe`, `##Fe` and `ĠFe`
    alike without knowing which tokenizer family produced them.

    Args:
        spec: The tokenizer being trimmed.

    Returns:
        The matching tokens.
    """
    return {
        token
        for token, surface in spec.surface_forms.items()
        if surface is not None and surface.strip() in CHEMICAL_ELEMENTS
    }


def run(model: str = DEFAULT_MODEL, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> TrimReport:
    """Trim with the custom preset applied alongside a built-in one.

    Args:
        model: Hub model id or local path.
        output_dir: Where the trimmed tokenizer is written.

    Returns:
        The report describing the run.
    """
    config = TrimConfig(
        model=model,
        output_dir=Path(output_dir),
        trim_model=False,
        overwrite=True,
        selection={"keep_presets": ["ascii_letters", "chemical_elements"]},
    )
    report = TrimPipeline(config).run()
    # The report attributes every kept token to the rule that saved it.
    print(report.render())
    print("kept by the custom preset:", report.vocabulary.kept_by_reason.get("preset:chemical_elements", 0))
    return report


def main() -> None:
    """Parse the command line and run the trim."""
    parser = argparse.ArgumentParser(
        description="Register a must-keep preset from Python and trim with it alongside a built-in one."
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Hub model id or local path.")
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where the trimmed tokenizer is written."
    )
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
