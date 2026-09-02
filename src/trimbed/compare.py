"""A structural diff between two tokenizers, typically a base and a trimmed version.

A trimmed checkpoint's README tells you the recipe (which corpus, which target size) but
not the result. This answers the other half: whether the smaller vocabulary really is a
subset of the larger one, which structural guarantees survived, which presets and which
Unicode scripts were preserved or gutted, and what the same text now costs in tokens.

Nothing here loads weights or reads a corpus. Two `tokenizer.json` documents go in, one
report comes out.
"""

from __future__ import annotations

import heapq
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from trimbed._logging import get_logger
from trimbed.presets import describe_presets, resolve_preset


if TYPE_CHECKING:
    from trimbed.spec import TokenizerSpec

logger = get_logger(__name__)

DECILES = 10
"""How many equal slices of the base id range the removed-token profile is cut into."""

MAX_REPORTED_SAMPLES = 5
"""How many fragmenting sample texts the encoding comparison quotes."""

NON_LETTER = "non-letter"
"""Script bucket for a token whose surface form holds no alphabetic character at all."""

MIXED = "mixed"
"""Script bucket for a token whose letters come from more than one script."""

UNDECODABLE = "partial-bytes"
"""Bucket for a token that stands for no well-formed text, e.g. half a UTF-8 sequence."""

_CATEGORIES = {"L": "letter", "N": "number", "P": "punctuation", "S": "symbol", "M": "mark"}
_BLOCKS = "▁▂▃▄▅▆▇█"
_LABEL_WIDTH = 17


class _Base(BaseModel):
    """Comparison models forbid unknown fields and allow `model_` names."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class GroupDiff(_Base):
    """How one group of base-vocabulary tokens fared, e.g. one Unicode script."""

    name: str = Field(description="What the group is, e.g. 'LATIN', 'letter' or 'digits'.")
    base_tokens: int = Field(description="Tokens the base tokenizer has in this group, e.g. 60003.")
    kept: int = Field(description="How many of those the other tokenizer still has, e.g. 45102.")

    @property
    def removed(self) -> int:
        """Return how many of the group's tokens the other tokenizer lacks."""
        return self.base_tokens - self.kept

    @property
    def kept_fraction(self) -> float:
        """Return the share of the group that survived, e.g. `0.752`."""
        return self.kept / self.base_tokens if self.base_tokens else 1.0


class PresetDiff(GroupDiff):
    """How one named preset fared, e.g. `digits` or `script:Latin`."""

    always_kept: bool = Field(
        description="Whether a trim keeps this preset's tokens whether or not it is named,"
        " so any loss here means a broken tokenizer rather than a policy choice."
    )


class RemovedToken(_Base):
    """One token the other tokenizer no longer has."""

    token: str = Field(description="The token as the base vocabulary stores it, e.g. 'Ġbureaucratie'.")
    token_id: int = Field(description="Its id in the base vocabulary, e.g. 1204.")
    surface: str | None = Field(
        description="The text it stands for, e.g. ' bureaucratie'; null for a partial byte sequence."
    )


class VocabularyDiff(_Base):
    """How the two vocabularies relate as sets of tokens."""

    base_size: int = Field(description="Tokens in the base vocabulary, e.g. 119547.")
    other_size: int = Field(description="Tokens in the other vocabulary, e.g. 50000.")
    shared: int = Field(description="Tokens both have.")
    removed: int = Field(description="Tokens only the base has.")
    introduced: int = Field(description="Tokens only the other has; zero for a genuine trim.")
    is_subset: bool = Field(description="Whether the other vocabulary introduces nothing new.")
    ids_contiguous: bool = Field(description="Whether the other vocabulary is numbered 0..n-1 without gaps.")
    order_preserved: bool = Field(
        description="Whether the shared tokens keep their relative order. A trim renumbers the survivors"
        " in place, so a false here means the vocabulary was rebuilt rather than trimmed."
    )
    introduced_examples: list[str] = Field(
        default_factory=list, description="A few of the tokens only the other tokenizer has."
    )

    @property
    def removed_fraction(self) -> float:
        """Return the share of the base vocabulary that is gone, e.g. `0.582`."""
        return self.removed / self.base_size if self.base_size else 0.0


class ComponentDiff(_Base):
    """How the parts of the tokenizer around the vocabulary compare."""

    base_model_type: str = Field(description="Backend family of the base tokenizer, e.g. 'WordPiece'.")
    other_model_type: str = Field(description="Backend family of the other tokenizer.")
    base_uses_byte_level: bool = Field(description="Whether the base maps text through the ByteLevel alphabet.")
    other_uses_byte_level: bool = Field(description="The same for the other tokenizer.")
    base_unk_token: str | None = Field(description="The base's unknown token, e.g. '[UNK]', or null.")
    other_unk_token: str | None = Field(description="The other's unknown token.")
    base_added_tokens: int = Field(description="Entries in the base's `added_tokens` list, e.g. 5.")
    other_added_tokens: int = Field(description="Entries in the other's `added_tokens` list.")
    base_special_tokens: int = Field(description="How many of the base's added tokens are flagged special.")
    other_special_tokens: int = Field(description="The same for the other tokenizer.")
    removed_added_tokens: list[str] = Field(
        default_factory=list, description="Added tokens the other tokenizer no longer has."
    )
    removed_special_tokens: list[str] = Field(
        default_factory=list, description="Special tokens the other tokenizer no longer has."
    )
    removed_post_processor_tokens: list[str] = Field(
        default_factory=list,
        description="Tokens the base's post-processor names that the other tokenizer no longer has,"
        " e.g. ['[SEP]']. Every one of them breaks encoding.",
    )
    chat_template: Literal["identical", "changed", "only in base", "only in other", "absent"] = Field(
        description="How the two chat templates relate."
    )

    @property
    def structural_break(self) -> bool:
        """Return whether the other tokenizer lost a token no tokenizer can do without."""
        return bool(self.removed_added_tokens or self.removed_special_tokens or self.removed_post_processor_tokens)


class ProfileDiff(_Base):
    """What kind of tokens were dropped, attributed three ways."""

    presets: list[PresetDiff] = Field(
        description="One entry per registered preset that matches anything in the base vocabulary."
    )
    scripts: list[GroupDiff] = Field(
        description="One entry per dominant Unicode script, largest group first, e.g. 'LATIN', 'CYRILLIC'."
    )
    categories: list[GroupDiff] = Field(
        description="One entry per majority Unicode category, e.g. 'letter', 'number', 'whitespace'."
    )
    removed_by_decile: list[int] = Field(
        description="Removed tokens per tenth of the base id range, lowest ids first. A trim that only cut"
        " the tail leaves the early deciles near zero."
    )
    removed_examples: list[RemovedToken] = Field(
        default_factory=list, description="The lowest-id removed tokens, which are the ones a trim rarely gives up."
    )


class EncodingDiff(_Base):
    """What the two tokenizers do to the same text."""

    checked: int = Field(default=0, description="Texts encoded with both tokenizers.")
    identical: int = Field(default=0, description="Texts both split into exactly the same tokens.")
    base_tokens: int = Field(default=0, description="Tokens the base produced over all texts.")
    other_tokens: int = Field(default=0, description="Tokens the other produced over the same texts.")
    examples: list[str] = Field(
        default_factory=list, description="The texts that fragment the most, truncated for readability."
    )

    @property
    def identical_rate(self) -> float:
        """Return the share of texts that segment identically, e.g. `0.92`."""
        return self.identical / self.checked if self.checked else 1.0

    @property
    def length_ratio(self) -> float:
        """Return how much longer the other tokenizer's output is, e.g. `1.031` for 3.1% more tokens."""
        return self.other_tokens / self.base_tokens if self.base_tokens else 1.0


class ComparisonReport(_Base):
    """The complete diff between two tokenizers."""

    base: str = Field(description="Where the base tokenizer came from, e.g. 'clips/e5-small-trm-nl'.")
    other: str = Field(description="Where the other tokenizer came from, e.g. 'clips/e5-small-trm'.")
    vocabulary: VocabularyDiff
    components: ComponentDiff
    profile: ProfileDiff
    encoding: EncodingDiff | None = None

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: File to write, e.g. `"diff.json"`. Parent directories are created.

        Returns:
            The path written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    def render(self) -> str:
        """Return the human-readable diff, one aligned `label  value` line per topic.

        Returns:
            The text `trimbed compare` prints, e.g.:

            ```
            base             clips/e5-small-trm-nl
            other            clips/e5-small-trm
            type             WordPiece -> WordPiece (byte-level no -> no, unk [UNK] -> [UNK])
            vocabulary       119,547 -> 50,000 (58.2% removed, 0 introduced)
            relation         subset, contiguous ids, original order preserved
            ```
        """
        lines = [
            _line("base", self.base),
            _line("other", self.other),
            *_render_components(self.components),
            *_render_vocabulary(self.vocabulary),
            *_render_profile(self.profile),
        ]
        if self.encoding is not None:
            check = self.encoding
            lines.append(
                _line(
                    "encoding",
                    f"{check.checked:,} texts, {check.identical:,} split identically, {check.length_ratio:.4f}x tokens",
                )
            )
            if check.examples:
                lines.append(_line("fragmented", "; ".join(check.examples)))
        return "\n".join(lines)


def compare_tokenizers(
    base: TokenizerSpec,
    other: TokenizerSpec,
    texts: Sequence[str] = (),
    presets: Sequence[str] = (),
    examples: int = 10,
) -> ComparisonReport:
    """Diff two tokenizers, reading nothing but their two documents.

    The comparison is directional: everything is attributed against `base`, so a group
    reported as `0/2,341` means the base had 2,341 such tokens and `other` has none of
    them left.

    Args:
        base: The larger, original tokenizer, e.g. clips/e5-small-trm-nl.
        other: The tokenizer to compare against it, e.g. its trimmed sibling
            clips/e5-small-trm.
        texts: Sample texts to encode with both, e.g. a handful of sentences in the
            language the trim targeted. Leave empty to skip the encoding comparison.
        presets: Extra preset names to resolve on top of the registered plain ones,
            which is how the parametrised ones are reached, e.g. `["script:Latin"]`.
        examples: How many removed tokens and introduced tokens to quote.

    Returns:
        The diff. For a healthy trim: `is_subset` and `order_preserved` both true, no
        structural break, and the removal concentrated in the later id deciles.
    """
    base_vocabulary = base.vocabulary
    other_vocabulary = other.vocabulary
    removed = base_vocabulary.keys() - other_vocabulary.keys()
    kept = base_vocabulary.keys() & other_vocabulary.keys()

    report = ComparisonReport(
        base=base.source or "base",
        other=other.source or "other",
        vocabulary=_vocabulary_diff(base_vocabulary, other_vocabulary, kept, removed, examples),
        components=_component_diff(base, other, other_vocabulary),
        profile=_profile_diff(base, kept, removed, presets, examples),
        encoding=_encoding_diff(base, other, texts) if texts else None,
    )
    logger.info(
        f"{report.base} -> {report.other}: {report.vocabulary.base_size:,} -> {report.vocabulary.other_size:,} tokens"
        f" ({report.vocabulary.removed_fraction:.1%} removed, {report.vocabulary.introduced:,} introduced)"
    )
    if report.components.structural_break:
        logger.warning(f"{report.other} is missing structural tokens that {report.base} declares")
    return report


def _vocabulary_diff(
    base_vocabulary: dict[str, int],
    other_vocabulary: dict[str, int],
    kept: set[str],
    removed: set[str],
    examples: int,
) -> VocabularyDiff:
    """Relate the two vocabularies as sets and as id spaces.

    Args:
        base_vocabulary: The base's token -> id map.
        other_vocabulary: The other's token -> id map.
        kept: Tokens both have.
        removed: Tokens only the base has.
        examples: How many introduced tokens to quote.

    Returns:
        The set-level diff, including whether the other vocabulary is what a trim of the
        base would look like: a subset, numbered contiguously, in the original order.
    """
    introduced = sorted(other_vocabulary.keys() - base_vocabulary.keys())
    other_ids = [other_vocabulary[token] for token in sorted(kept, key=base_vocabulary.__getitem__)]
    return VocabularyDiff(
        base_size=len(base_vocabulary),
        other_size=len(other_vocabulary),
        shared=len(kept),
        removed=len(removed),
        introduced=len(introduced),
        is_subset=not introduced,
        ids_contiguous=sorted(other_vocabulary.values()) == list(range(len(other_vocabulary))),
        order_preserved=all(earlier < later for earlier, later in pairwise(other_ids)),
        introduced_examples=introduced[:examples],
    )


def _component_diff(base: TokenizerSpec, other: TokenizerSpec, other_vocabulary: dict[str, int]) -> ComponentDiff:
    """Compare everything around the vocabulary: the family, the specials, the template.

    Args:
        base: The base tokenizer.
        other: The tokenizer being compared against it.
        other_vocabulary: The other's token -> id map, so membership is a set lookup.

    Returns:
        The component-level diff. The three `removed_*` lists are the ones that matter:
        a post-processor token that is gone means encoding is broken, not merely smaller.
    """
    post_processor_tokens = [base.id_to_token[token_id] for token_id in sorted(base.post_processor_token_ids)]
    return ComponentDiff(
        base_model_type=base.model_type,
        other_model_type=other.model_type,
        base_uses_byte_level=base.uses_byte_level,
        other_uses_byte_level=other.uses_byte_level,
        base_unk_token=base.unk_token,
        other_unk_token=other.unk_token,
        base_added_tokens=len(base.added_tokens),
        other_added_tokens=len(other.added_tokens),
        base_special_tokens=len(base.special_token_ids),
        other_special_tokens=len(other.special_token_ids),
        removed_added_tokens=[token.content for token in base.added_tokens if token.content not in other_vocabulary],
        removed_special_tokens=[
            token.content for token in base.added_tokens if token.special and token.content not in other_vocabulary
        ],
        removed_post_processor_tokens=[token for token in post_processor_tokens if token not in other_vocabulary],
        chat_template=_chat_template_state(base.chat_template, other.chat_template),
    )


def _chat_template_state(base: str | None, other: str | None) -> str:
    """Say how two chat templates relate.

    Args:
        base: The base tokenizer's template, or `None`.
        other: The other tokenizer's template, or `None`.

    Returns:
        One of `"identical"`, `"changed"`, `"only in base"`, `"only in other"` or
        `"absent"`. Note that a spec built from a bare `tokenizer.json` never carries a
        template, since it lives in tokenizer_config.json.
    """
    if base and other:
        return "identical" if base == other else "changed"
    if base:
        return "only in base"
    return "only in other" if other else "absent"


def _profile_diff(
    base: TokenizerSpec, kept: set[str], removed: set[str], presets: Sequence[str], examples: int
) -> ProfileDiff:
    """Attribute the base vocabulary to presets, scripts and categories, and profile the losses.

    Args:
        base: The base tokenizer.
        kept: Tokens the other tokenizer still has.
        removed: Tokens only the base has.
        presets: Extra preset names on top of the registered plain ones.
        examples: How many removed tokens to quote.

    Returns:
        The three attributions plus the id-range histogram of what was dropped.
    """
    vocabulary = base.vocabulary
    # A trim renumbers from zero, so the base id is the only shared ranking of "how early
    # a token entered the vocabulary", which for BPE and WordPiece tracks its frequency.
    span = max(base.max_id + 1, 1)
    by_decile = [0] * DECILES
    for token in removed:
        by_decile[min(vocabulary[token] * DECILES // span, DECILES - 1)] += 1

    lowest = sorted(removed, key=vocabulary.__getitem__)[:examples]
    return ProfileDiff(
        presets=_preset_diffs(base, kept, presets),
        scripts=_group_diffs(base, kept, _dominant_script),
        categories=_group_diffs(base, kept, _majority_category),
        removed_by_decile=by_decile,
        removed_examples=[
            RemovedToken(token=token, token_id=vocabulary[token], surface=base.surface_forms[token]) for token in lowest
        ],
    )


def _preset_diffs(base: TokenizerSpec, kept: set[str], extra: Sequence[str]) -> list[PresetDiff]:
    """Resolve the presets against the base and count how many of each survived.

    Args:
        base: The base tokenizer.
        kept: Tokens the other tokenizer still has.
        extra: Preset names to resolve on top of the registered plain ones. Parametrised
            presets are only reachable this way, since `script` needs its argument.

    Returns:
        One entry per preset that matched anything, ordered as `describe_presets` orders
        them with the extras last. A preset that matches nothing in this vocabulary
        (`byte_alphabet` against WordPiece, say) is left out rather than reported as 0/0.
    """
    registered = {info.name: info.always_kept for info in describe_presets() if ":" not in info.name}
    diffs = []
    for name in [*registered, *extra]:
        tokens = resolve_preset(name, base)
        if tokens:
            diffs.append(
                PresetDiff(
                    name=name,
                    base_tokens=len(tokens),
                    kept=len(tokens & kept),
                    always_kept=registered.get(name, False),
                )
            )
    return diffs


def _group_diffs(base: TokenizerSpec, kept: set[str], classify: Callable[[str | None], str]) -> list[GroupDiff]:
    """Bucket every base token by some property of its surface form and count the survivors.

    Args:
        base: The base tokenizer.
        kept: Tokens the other tokenizer still has.
        classify: Names the bucket a decoded surface form belongs to. It is handed the
            decoded text, so it sees `" de"` rather than the stored `"Ġde"`, and `None`
            for a token that stands for no well-formed text.

    Returns:
        One entry per bucket, largest first and ties broken by name so the output is
        stable across runs.
    """
    totals: Counter[str] = Counter()
    survivors: Counter[str] = Counter()
    for token, surface in base.surface_forms.items():
        group = classify(surface)
        totals[group] += 1
        if token in kept:
            survivors[group] += 1
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [GroupDiff(name=name, base_tokens=total, kept=survivors[name]) for name, total in ordered]


def _dominant_script(surface: str | None) -> str:
    """Name the Unicode script a token's letters belong to.

    Script membership is approximated from Unicode character names, the same best-effort
    approach the `script` preset takes, so this needs no Unicode property database.

    Args:
        surface: The decoded text a token stands for, or `None`.

    Returns:
        The script in the spelling the character names use, e.g. `"LATIN"`, `"CYRILLIC"`
        or `"CJK"`, or one of [`MIXED`][trimbed.compare.MIXED],
        [`NON_LETTER`][trimbed.compare.NON_LETTER] and
        [`UNDECODABLE`][trimbed.compare.UNDECODABLE].
    """
    if surface is None:
        return UNDECODABLE
    scripts = {unicodedata.name(char, "UNKNOWN").split()[0] for char in surface if char.isalpha()}
    if not scripts:
        return NON_LETTER
    return scripts.pop() if len(scripts) == 1 else MIXED


def _majority_category(surface: str | None) -> str:
    """Name the Unicode category most of a token's characters fall into.

    This is the axis the ASCII-only presets cannot cover: it sorts a vocabulary into
    letters, numbers, punctuation, symbols and whitespace regardless of script.

    Args:
        surface: The decoded text a token stands for, or `None`.

    Returns:
        One of `"letter"`, `"number"`, `"punctuation"`, `"symbol"`, `"mark"`,
        `"whitespace"`, `"other"`, or [`UNDECODABLE`][trimbed.compare.UNDECODABLE].
    """
    if surface is None:
        return UNDECODABLE
    tally: Counter[str] = Counter(
        "whitespace" if char.isspace() else _CATEGORIES.get(unicodedata.category(char)[0], "other") for char in surface
    )
    # `max` over the tally rather than `most_common`, so an empty surface form (a bare
    # WordPiece continuation marker, say) has somewhere to go instead of raising.
    winner, _ = max(tally.items(), key=lambda item: item[1], default=("other", 0))
    return winner


def _encoding_diff(base: TokenizerSpec, other: TokenizerSpec, texts: Sequence[str]) -> EncodingDiff:
    """Encode the same texts with both tokenizers and measure the drift.

    The token strings are compared rather than the ids, because a trim renumbers
    everything and identical ids would only ever mean the two vocabularies are the same
    size.

    Args:
        base: The base tokenizer.
        other: The tokenizer being compared against it.
        texts: The texts to encode.

    Returns:
        The counts, plus the texts whose token count grew the most. A ratio above 1.0 is
        what the missing tokens cost at inference time.
    """
    result = EncodingDiff()
    drifting: list[tuple[int, str]] = []
    for text in texts:
        base_tokens = [base.id_to_token[token_id] for token_id in base.encode(text)]
        other_tokens = [other.id_to_token[token_id] for token_id in other.encode(text)]
        result.checked += 1
        result.base_tokens += len(base_tokens)
        result.other_tokens += len(other_tokens)
        if base_tokens == other_tokens:
            result.identical += 1
        else:
            drifting.append((len(other_tokens) - len(base_tokens), text[:60]))

    result.examples = [text for _, text in heapq.nlargest(MAX_REPORTED_SAMPLES, drifting)]
    logger.info(
        f"encoding: {result.identical:,}/{result.checked:,} texts split identically,"
        f" {result.length_ratio:.4f}x as many tokens"
    )
    return result


def _line(label: str, value: str) -> str:
    """Return one aligned `label  value` line of the rendered report.

    Args:
        label: The left column, e.g. `"vocabulary"`.
        value: The right column.

    Returns:
        The padded line.
    """
    return f"{label:<{_LABEL_WIDTH}}{value}"


def _group_summary(groups: Sequence[GroupDiff], limit: int = 6) -> str:
    """Summarise the largest buckets as `name kept/total (pct)`, biggest first.

    Args:
        groups: The buckets to summarise, already ordered largest first.
        limit: How many to show before adding an ellipsis.

    Returns:
        E.g. `"LATIN 45,102/60,003 (75.2%), CYRILLIC 0/2,341 (0.0%), ..."`.
    """
    shown = ", ".join(
        f"{group.name} {group.kept:,}/{group.base_tokens:,} ({group.kept_fraction:.1%})" for group in groups[:limit]
    )
    return f"{shown}, ..." if len(groups) > limit else shown


def _sparkline(counts: Sequence[int]) -> str:
    """Draw a bar per value, scaled to the largest one.

    Args:
        counts: The values to draw, e.g. removed tokens per decile.

    Returns:
        One block character per value, e.g. `"▁▁▂▃▅▆▇███"`.
    """
    peak = max(counts) or 1
    return "".join(_BLOCKS[round(count * (len(_BLOCKS) - 1) / peak)] for count in counts)


def _render_components(components: ComponentDiff) -> list[str]:
    """Render the family, the special tokens and the chat template.

    Args:
        components: The component-level diff.

    Returns:
        The lines, with a `structural loss` line only when something load-bearing is gone.
    """
    byte_level = f"{_yes_no(components.base_uses_byte_level)} -> {_yes_no(components.other_uses_byte_level)}"
    unk = f"{components.base_unk_token or 'none'} -> {components.other_unk_token or 'none'}"
    lines = [
        _line("type", f"{components.base_model_type} -> {components.other_model_type} (byte-level {byte_level})"),
        _line("unk token", unk),
        _line(
            "added tokens",
            f"{components.base_added_tokens:,} -> {components.other_added_tokens:,} "
            f"({components.base_special_tokens:,} -> {components.other_special_tokens:,} special)",
        ),
        _line("chat template", components.chat_template),
    ]
    if components.structural_break:
        losses = [
            f"{kind}: {', '.join(tokens)}"
            for kind, tokens in (
                ("added", components.removed_added_tokens),
                ("special", components.removed_special_tokens),
                ("post-processor", components.removed_post_processor_tokens),
            )
            if tokens
        ]
        lines.append(_line("structural loss", "; ".join(losses)))
    return lines


def _render_vocabulary(vocabulary: VocabularyDiff) -> list[str]:
    """Render the sizes and the subset relation.

    Args:
        vocabulary: The set-level diff.

    Returns:
        Two lines: the sizes, and how the id spaces relate.
    """
    relation = ", ".join(
        [
            "subset" if vocabulary.is_subset else f"not a subset ({vocabulary.introduced:,} new tokens)",
            "contiguous ids" if vocabulary.ids_contiguous else "non-contiguous ids",
            "original order" if vocabulary.order_preserved else "reordered",
        ]
    )
    lines = [
        _line(
            "vocabulary",
            f"{vocabulary.base_size:,} -> {vocabulary.other_size:,} "
            f"({vocabulary.removed_fraction:.1%} removed, {vocabulary.shared:,} shared)",
        ),
        _line("relation", relation),
    ]
    if vocabulary.introduced_examples:
        lines.append(_line("new tokens", ", ".join(vocabulary.introduced_examples)))
    return lines


def _render_profile(profile: ProfileDiff) -> list[str]:
    """Render the preset, script and category attributions and the loss profile.

    Args:
        profile: The attributions to render.

    Returns:
        The lines, splitting the presets into the ones that survived whole, the ones that
        were partly cut and the ones that are gone entirely.
    """
    whole = [preset.name for preset in profile.presets if preset.removed == 0]
    partial = [preset for preset in profile.presets if 0 < preset.kept < preset.base_tokens]
    gone = [preset for preset in profile.presets if preset.kept == 0]

    lines = []
    if whole:
        lines.append(_line("presets kept", ", ".join(whole)))
    if partial:
        lines.append(_line("presets cut", _group_summary(partial, limit=len(partial))))
    if gone:
        lines.append(_line("presets lost", ", ".join(f"{preset.name} ({preset.base_tokens:,})" for preset in gone)))
    lines.append(_line("scripts", _group_summary(profile.scripts)))
    lines.append(_line("categories", _group_summary(profile.categories)))
    lines.append(
        _line("removed by id", f"{_sparkline(profile.removed_by_decile)}  (deciles of the base ids, lowest first)")
    )
    if profile.removed_examples:
        lines.append(
            _line(
                "first removed",
                ", ".join(f"{example.token} ({example.token_id:,})" for example in profile.removed_examples),
            )
        )
    return lines


def _yes_no(value: bool) -> str:
    """Return `"yes"` or `"no"`, since `True`/`False` reads oddly in a rendered table.

    Args:
        value: The flag to render.

    Returns:
        `"yes"` when set.
    """
    return "yes" if value else "no"
