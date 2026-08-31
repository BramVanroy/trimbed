"""Named sets of tokens a user can insist on keeping.

Some of these presets are functional and can take parameters so
users can for instance write `script:Latin` to keep all tokens whose
letters are Latin script.

Note that tokens are never "added", only the intersection of the preset
and what is already in the vocabulary would be kept.
"""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trimbed.bytelevel import byte_level_alphabet


if TYPE_CHECKING:
    from trimbed.spec import TokenizerSpec

type PresetFn = Callable[[TokenizerSpec], set[str]]
type ParametrisedPresetFn = Callable[[TokenizerSpec, str], set[str]]

# preset_name -> preset function map
_PRESETS: dict[str, PresetFn] = {}
_PARAMETRISED: dict[str, ParametrisedPresetFn] = {}
# the presets whose tokens `TokenizerSpec.structural_ids` already covers, so the trim
# keeps them on its own. `tests/test_presets.py` holds that claim against every family.
_ALWAYS_KEPT: set[str] = set()

_PRINT_PARAM_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class PresetInfo:
    """What `trimbed presets` prints about one preset."""

    name: str
    always_kept: bool
    summary: str


def register_preset(name: str, always_kept: bool = False) -> Callable[[PresetFn], PresetFn]:
    """Register a preset and specify whether the trim always includes the preset's tokens.

    Args:
        name: The name users will write in `selection.keep_presets`, e.g. `"digits"`.
        always_kept: Whether the preset resolves to tokens that `TokenizerSpec.structural_ids`
            already protects, so the trim keeps them whether or not the preset is named.
            True for `"special_tokens"` and the like.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If the name is already registered or contains the separator.
    """

    def decorator(func: PresetFn) -> PresetFn:
        if _PRINT_PARAM_SEPARATOR in name:
            raise ValueError(f"preset name {name!r} may not contain {_PRINT_PARAM_SEPARATOR!r}")
        if name in _PRESETS:
            raise ValueError(f"preset {name!r} is already registered")
        _PRESETS[name] = func
        if always_kept:
            _ALWAYS_KEPT.add(name)
        return func

    return decorator


def register_parametrised_preset(prefix: str) -> Callable[[ParametrisedPresetFn], ParametrisedPresetFn]:
    """Register a preset family addressed as `prefix:argument`, such as `script:Latin`.

    Args:
        prefix: The part before the colon, e.g. `"script"`, which users then write as
            `script:Latin` or `script:Cyrillic`.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If the prefix is already registered.
    """

    def decorator(func: ParametrisedPresetFn) -> ParametrisedPresetFn:
        if prefix in _PARAMETRISED:
            raise ValueError(f"parametrised preset {prefix!r} is already registered")
        _PARAMETRISED[prefix] = func
        return func

    return decorator


def describe_presets() -> tuple[PresetInfo, ...]:
    """Describe every registered preset, plain ones first and parametrised families last.

    Returns:
        One entry per preset, carrying the name, whether the trim keeps those tokens
        anyway, and the first line of the preset function's docstring.
    """
    registered: list[tuple[str, PresetFn | ParametrisedPresetFn]] = [
        (name, _PRESETS[name]) for name in sorted(_PRESETS)
    ]
    registered += [(f"{prefix}{_PRINT_PARAM_SEPARATOR}...", _PARAMETRISED[prefix]) for prefix in sorted(_PARAMETRISED)]
    return tuple(
        PresetInfo(name=name, always_kept=name in _ALWAYS_KEPT, summary=_docstring_summary(func))
        for name, func in registered
    )


def render_presets() -> str:
    """Render every registered preset as an aligned table, the structural ones first.

    Returns:
        The table that `trimbed presets` prints.
    """
    presets = describe_presets()
    width = max(len(preset.name) for preset in presets)
    groups = (
        ("Structural (kept whether or not you name them):", [p for p in presets if p.always_kept]),
        ("Opt-in (kept only when you name them):", [p for p in presets if not p.always_kept]),
    )

    lines: list[str] = []
    for header, group in groups:
        if lines:
            lines.append("")
        lines.append(header)
        lines.extend(f"  {preset.name:<{width}}  {preset.summary}" for preset in group)
    return "\n".join(lines)


def _docstring_summary(func: PresetFn | ParametrisedPresetFn) -> str:
    """Return the first line of a preset function's docstring, empty when it has none.

    Args:
        func: A registered preset function.

    Returns:
        The summary line, e.g. `"Tokens made up entirely of ASCII digits."`.
    """
    lines = (func.__doc__ or "").strip().splitlines()
    return lines[0] if lines else ""


def available_presets() -> tuple[str, ...]:
    """Return every registered preset name, with `prefix:...` for the parametrised ones.

    E.g. `("added_tokens", "alphanumeric", "ascii_letters", ..., "unk", "whitespace",
    "script:...")`.
    """
    return tuple(info.name for info in describe_presets())


def resolve_preset(name: str, spec: TokenizerSpec) -> set[str]:
    """Resolve one preset name against a tokenizer.

    Args:
        name: A registered preset name, optionally `prefix:argument`, e.g. `"digits"`
            or `"script:Latin"`.
        spec: The tokenizer being trimmed.

    Returns:
        The tokens the preset selects, all guaranteed to exist in `spec`. Sizes vary
        enormously with the preset: against codefuse-ai/F2LLM-v2-160M, `digits` selects
        10 tokens, `byte_alphabet` 256, `single_characters` 18,747 and
        `ascii_letters` 70,096 of the 151,669.

    Raises:
        KeyError: If no preset matches the name.
    """
    if name in _PRESETS:
        return _PRESETS[name](spec)
    prefix, separator, argument = name.partition(_PRINT_PARAM_SEPARATOR)
    if separator and prefix in _PARAMETRISED:
        return _PARAMETRISED[prefix](spec, argument)
    raise KeyError(f"unknown preset {name!r}; available presets: {', '.join(available_presets())}")


def resolve_presets(names: Iterable[str], spec: TokenizerSpec) -> dict[str, set[str]]:
    """Resolve several presets, keeping the result attributable per preset.

    Args:
        names: Preset names to resolve, e.g. `["special_tokens", "digits"]`.
        spec: The tokenizer being trimmed.

    Returns:
        A mapping of preset name to the tokens it selected. Keeping it attributable per
        preset is what lets the report say `preset:digits=10` rather than one
        undifferentiated total.
    """
    return {name: resolve_preset(name, spec) for name in names}


def _by_surface(spec: TokenizerSpec, predicate: Callable[[str], bool]) -> set[str]:
    """Select vocabulary tokens whose decoded surface form satisfies a predicate.

    Args:
        spec: The tokenizer being trimmed.
        predicate: Test applied to the decoded text a token stands for, so it sees
            `" de"` rather than the stored `"Ġde"`.

    Returns:
        The matching tokens, as they are stored in the vocabulary.
    """
    return {token for token, surface in spec.surface_forms.items() if surface is not None and predicate(surface)}


def _all_in(text: str, allowed: frozenset[str]) -> bool:
    """Return whether `text` is non-empty and drawn only from `allowed`, ignoring surrounding space.

    Args:
        text: The decoded surface form of a token, e.g. `" 1990"`.
        allowed: The permitted characters, e.g. `string.digits`.

    Returns:
        True when every non-space character is allowed and at least one exists, so
        `" 1990"` passes the digits test but `" "` and `" 1990s"` do not.
    """
    core = text.strip()
    return bool(core) and all(char in allowed for char in core)


_ASCII_LETTERS = frozenset(string.ascii_letters)
_DIGITS = frozenset(string.digits)
_PUNCTUATION = frozenset(string.punctuation)
_ALPHANUMERIC = _ASCII_LETTERS | _DIGITS
_ASCII_PRINTABLE = frozenset(string.printable)


@register_preset("structural", always_kept=True)
def _structural(spec: TokenizerSpec) -> set[str]:
    """Everything the trim keeps anyway: added tokens, post-processor tokens, unk, byte alphabet.

    This resolves to `TokenizerSpec.structural_ids`, the very set the selector protects, so
    it is the honest way to say "keep the bare minimum" for a run that has no corpus.
    """
    return set(spec.structural_tokens)


@register_preset("added_tokens", always_kept=True)
def _added_tokens(spec: TokenizerSpec) -> set[str]:
    """Every entry in the tokenizer's `added_tokens` list, e.g. 26 tokens for F2LLM-v2."""
    return {token.content for token in spec.added_tokens}


@register_preset("special_tokens", always_kept=True)
def _special_tokens(spec: TokenizerSpec) -> set[str]:
    """Added tokens flagged as special, e.g. `<|endoftext|>`, `<|im_start|>`, `[SEP]`."""
    return {token.content for token in spec.added_tokens if token.special}


@register_preset("unk", always_kept=True)
def _unk(spec: TokenizerSpec) -> set[str]:
    """The backend's unknown token, if it has one, e.g. `{"[UNK]"}` for BERT."""
    unk = spec.unk_token
    return {unk} if unk else set()


@register_preset("byte_alphabet", always_kept=True)
def _byte_alphabet(spec: TokenizerSpec) -> set[str]:
    """The 256 ByteLevel alphabet characters, so no byte sequence becomes unencodable.

    Empty for a tokenizer that does not encode through those characters, e.g.
    google-bert/bert-base-cased, where they would only stand for themselves.
    """
    if not spec.uses_byte_level:
        return set()
    return set(byte_level_alphabet()) & spec.vocabulary.keys()


@register_preset("single_characters")
def _single_characters(spec: TokenizerSpec) -> set[str]:
    """Every existing token standing for exactly one character (optionally space-prefixed).

    Far from an ASCII-only set: of the 18,747 F2LLM-v2 tokens this selects, most are
    single non-Latin characters, reached through byte-level tokens such as `Ã©` for
    `"é"` and `ãĢĤ` for the CJK full stop.
    """
    return _by_surface(spec, lambda surface: len(surface.strip()) == 1)


@register_preset("ascii_letters")
def _ascii_letters(spec: TokenizerSpec) -> set[str]:
    """Existing tokens whose surface form is made up entirely of ASCII letters.

    The test runs on the decoded text, so `"Ġthe"` passes on its surface form `" the"`
    while `"Ġthe."` does not (because it has a punctuation mark).
    """
    return _by_surface(spec, lambda surface: _all_in(surface, _ASCII_LETTERS))


@register_preset("digits")
def _digits(spec: TokenizerSpec) -> set[str]:
    """Existing tokens whose surface form is made up entirely of ASCII digits.

    Often a very small set: a tokenizer whose pre-tokenizer splits numbers digit by digit,
    as the Qwen-derived ones do, has exactly ten such tokens, `"0"` through `"9"`.
    """
    return _by_surface(spec, lambda surface: _all_in(surface, _DIGITS))


@register_preset("alphanumeric")
def _alphanumeric(spec: TokenizerSpec) -> set[str]:
    """Existing tokens whose surface form is made up entirely of ASCII letters and digits.

    Where digits are split off from words this is just `ascii_letters` plus the ten
    digits, e.g. 70,106 against F2LLM-v2's 70,096.
    """
    return _by_surface(spec, lambda surface: _all_in(surface, _ALPHANUMERIC))


@register_preset("punctuation")
def _punctuation(spec: TokenizerSpec) -> set[str]:
    """Existing tokens whose surface form is made up entirely of ASCII punctuation,  e.g. `"()"`, `"->"`, `"=="`."""
    return _by_surface(spec, lambda surface: _all_in(surface, _PUNCTUATION))


@register_preset("whitespace")
def _whitespace(spec: TokenizerSpec) -> set[str]:
    """Existing tokens consisting only of whitespace, e.g. `Ġ` (a space), `Ċ` (a newline), `ĠĠĠĠ`."""
    return _by_surface(spec, lambda surface: bool(surface) and surface.isspace())


@register_preset("ascii_printable")
def _ascii_printable(spec: TokenizerSpec) -> set[str]:
    """Existing tokens whose surface form is entirely printable ASCII.

    The widest of the ASCII presets, since it admits mixed tokens the others reject:
    `"#include"`, `"(self"`, `"_name"`, `"--------"`.
    """
    return _by_surface(spec, lambda surface: bool(surface) and all(char in _ASCII_PRINTABLE for char in surface))


@register_parametrised_preset("script")
def _script(spec: TokenizerSpec, script: str) -> set[str]:
    """Existing tokens whose letters all belong to one Unicode script, e.g. `script:Latin`.

    Script membership is approximated from Unicode character names (e.g. `LATIN SMALL
    LETTER A` belongs to `Latin`), which avoids a dependency on a full Unicode
    property database.
    Non-alphabetic characters are ignored, so `"1990s"` counts as Latin.

    NOTE: this may not be accurate for all scripts, especially those that are not alphabetic.
    It is a best-effort approximation based on Unicode character names.

    Args:
        spec: The tokenizer being trimmed.
        script: Script name, matched case-insensitively, e.g. `"Latin"`, `"Cyrillic"`
            or `"Greek"`.

    Returns:
        The matching tokens. As an example, `script:Latin` selects 94,619 of F2LLM-v2's
        151,669, which is most of what a Dutch or English trim keeps.
    """
    wanted = script.strip().upper()
    if not wanted:
        raise ValueError("the 'script' preset needs an argument, e.g. 'script:Latin'")

    def matches(surface: str) -> bool:
        letters = [char for char in surface if char.isalpha()]
        if not letters:
            return False
        return all(unicodedata.name(char, "").startswith(wanted) for char in letters)

    return _by_surface(spec, matches)
