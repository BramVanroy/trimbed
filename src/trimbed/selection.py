"""Deciding which token ids survive the trim and recording the reason."""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from trimbed._logging import get_logger
from trimbed.presets import resolve_presets


if TYPE_CHECKING:
    from trimbed.config import SelectionConfig
    from trimbed.counting import CorpusCounts
    from trimbed.spec import TokenizerSpec

logger = get_logger(__name__)

STRUCTURAL = "structural"
CORPUS = "corpus"
DEPENDENCY = "dependency"


@dataclass
class Selection:
    """The outcome of applying a `SelectionConfig`.

    Attributes:
        kept_ids: Token ids that survive, as a set.
        structural_ids: Ids that were never eligible for removal.
        provenance: For each kept id, the reasons it was kept, e.g.
            `{151645: {"structural", "chat_template"}, 9707: {"corpus"}}`. The labels are
            `"structural"`, `"corpus"`, `"dependency"`, `"text"`,
            `"chat_template"`, `"explicit_token"`, `"explicit_id"`, or a prefixed
            `"preset:digits"`, `"pattern:^Ġ"`, `"file:keep.txt"`. Tracking this is what
            lets the report say which rule saved each token, rather than only how many
            survived in total.
        dropped_requested: Ids that were requested but that the size cap had to remove
            anyway, mapped to the reasons they had been requested.
        unknown_tokens: Requested token strings that this tokenizer does not contain,
            e.g. `["<|custom|>"]` when a keep-list was written for another checkpoint.
    """

    kept_ids: set[int] = field(default_factory=set)
    structural_ids: set[int] = field(default_factory=set)
    provenance: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    dropped_requested: dict[int, set[str]] = field(default_factory=dict)
    unknown_tokens: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the size of the trimmed vocabulary."""
        return len(self.kept_ids)

    def counts_by_reason(self) -> dict[str, int]:
        """Return how many kept ids carry each provenance label.

        An id kept for several reasons is counted under each of them, so the values sum
        to more than `len(self)`.

        A preset-and-chat-template trim of codefuse-ai/F2LLM-v2-160M
        reports `{"chat_template": 88, "dependency": 101, "preset:byte_alphabet": 256,
        "preset:digits": 10, "preset:special_tokens": 14, "structural": 282, "text": 8}`
        for 461 kept tokens.
        """
        tally: Counter[str] = Counter()
        for reasons in self.provenance.values():
            tally.update(reasons)
        return dict(sorted(tally.items()))


def _read_token_file(path: Path) -> list[str]:
    """Read one token per line, skipping blanks and comments (`#`).

    Args:
        path: Text file listing tokens to keep. The tokens are written as the vocabulary
            stores them, so a byte-level file holds `Ġde` rather than ` de`.

    Returns:
        The tokens, in file order, e.g. `["Ġde", "Ġhet", "Ġeen"]`.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _requested_ids(spec: TokenizerSpec, config: SelectionConfig, selection: Selection) -> dict[int, set[str]]:
    """Resolve every explicit must-keep source into ids with provenance labels.

    Args:
        spec: The tokenizer being trimmed.
        config: The selection settings.
        selection: The selection being built. Token strings this tokenizer does not
            contain are recorded on it for the report.

    Returns:
        A mapping of token id to the labels explaining why it was requested, e.g.
        `{9707: {"text"}, 15: {"preset:digits", "pattern:^[0-9]$"}}`.

    Raises:
        ValueError: If an explicit token id is out of range.
    """
    requested: dict[int, set[str]] = defaultdict(set)

    # get the requested tokens for each preset
    for name, tokens in resolve_presets(config.keep_presets, spec).items():
        label = f"preset:{name}"
        for token in tokens:
            requested[spec.vocabulary[token]].add(label)
        logger.info(f"preset {name} selected {len(tokens):,} tokens")

    def add_literal(tokens: list[str], label: str) -> None:
        for token in tokens:
            token_id = spec.vocabulary.get(token)
            if token_id is None:
                selection.unknown_tokens.append(token)
            else:
                requested[token_id].add(label)

    # add literal tokens (as their id) that are described either in the config's
    # `keep_tokens` or in the files listed in `keep_token_files`
    add_literal(config.keep_tokens, "explicit_token")
    for path in config.keep_token_files:
        add_literal(_read_token_file(Path(path)), f"file:{path}")

    # add the requested token ids
    for token_id in config.keep_token_ids:
        if token_id not in spec.id_to_token:
            raise ValueError(f"requested token id {token_id} does not exist in {spec.source or 'this tokenizer'}")
        requested[token_id].add("explicit_id")

    # tokenize the requested texts and keep its ids
    for text in config.keep_texts:
        for token_id in spec.encode(text):
            requested[token_id].add("text")

    # tokenize the chat template and keep its ids if requested
    if config.keep_chat_template and spec.chat_template_literals.strip():
        template_ids = spec.encode(spec.chat_template_literals)
        for token_id in template_ids:
            requested[token_id].add("chat_template")
        logger.info(f"the chat template's own words need {len(set(template_ids)):,} tokens")

    # add tokens matching a specific regex pattern,
    # e.g. `^Ġ` to keep all tokens that start with a space in byte-level tokenizers
    for pattern in config.keep_patterns:
        compiled = re.compile(pattern)
        label = f"pattern:{pattern}"
        matched = 0
        for token, token_id in spec.vocabulary.items():
            surface = spec.surface_forms.get(token, token) or token
            if compiled.search(surface):
                requested[token_id].add(label)
                matched += 1
        logger.info(f"pattern {pattern} matched {matched:,} tokens")

    return requested


def _corpus_ids(counts: CorpusCounts | None, config: SelectionConfig) -> set[int]:
    """Given corpus frequency stats and a selection config, return the ids to keep.

    `coverage`, `top_k` and `min_count` each cut the frequency-ranked vocabulary off at
    some point. Setting more than one is fine: the result is their intersection, so the
    strictest of them decides.

    Args:
        counts: Corpus statistics, or `None` when no corpus was configured.
        config: The selection settings. E.g. `coverage=0.999` with `top_k=32000` keeps
            whichever of the two prefixes is shorter (smaller resulting vocabulary).

    Returns:
        The ids selected from corpus frequency, and an empty set when no corpus was read.
    """
    if counts is None or not counts.counts:
        return set()

    # token ids ranked by highest frequency first
    ranked = counts.ranked_ids()
    num_kept = len(ranked)

    # Only keep tokens whose count is >= config.min_count
    if config.min_count is not None:
        num_kept = min(num_kept, sum(1 for token_id in ranked if counts.counts[token_id] >= config.min_count))
    # Strictly cut off the top_k most frequent tokens, if requested
    if config.top_k is not None:
        num_kept = min(num_kept, config.top_k)

    # Of the remaining tokens, only keep the top-n that account for a total
    # corpus coverage of config.coverage, if requested. This is a cumulative sum of the
    # counts of the ranked tokens, stopping when the sum reaches the target coverage.
    if config.coverage is not None:
        # given our corpus size and the requested coverage (e.g. 0.99)
        # how many tokens should we account for?
        target = config.coverage * counts.total_num_tokens
        running = 0
        reached = len(ranked)
        for index, token_id in enumerate(ranked):
            running += counts.counts[token_id]
            if running >= target:
                reached = index + 1
                break
        num_kept = min(num_kept, reached)

    return set(ranked[:num_kept])


def _expand_dependencies(kept: set[int], dependencies: dict[int, tuple[int, ...]]) -> set[int]:
    """Return the extra ids needed to keep every kept token reachable (BPE merges particularly).

    Args:
        kept: Ids selected so far.
        dependencies: Per-token structural dependencies from the backend, e.g.
            `Ġthe -> (Ġth, e)`.

    Returns:
        Ids that must be added to `kept`. Empty for backends without dependencies. The
        walk is transitive, so keeping `Ġthe` pulls in `Ġth`, then whatever `Ġth` is
        itself merged from, down to the byte alphabet.
    """
    if not dependencies:
        return set()
    added: set[int] = set()
    stack = [token_id for token_id in kept if token_id in dependencies]
    # recursively go through the dependency graph, recording all encountered (sub)tokens
    # in both added/stack. Popping from "stack" to keep recursing in until it is empty
    while stack:
        for parent in dependencies.get(stack.pop(), ()):
            if parent not in kept and parent not in added:
                added.add(parent)
                stack.append(parent)
    return added


def _apply_cap(
    selection: Selection,
    counts: CorpusCounts | None,
    dependencies: dict[int, tuple[int, ...]],
    max_vocab_size: int,
) -> None:
    """Shrink the selection to the configured hard cap.

    Least-frequent tokens are removed first but only once nothing still depends on them.
    Removing a token can free the tokens it was merged from, so the candidates are
    re-evaluated as the heap drains. Structural tokens are never touched, and requested
    tokens the cap removes anyway are recorded on the selection so the report can call
    them out.

    Args:
        selection: The selection to shrink, updated in place with provenance info. Its
            `kept_ids` must already be closed over `dependencies`, which `select_tokens`
            guarantees, or the bookkeeping below counts tokens that were never there.
        counts: Corpus statistics used to rank droppable tokens. This may be `None`, in
            which case every droppable token ranks at frequency zero and the id order
            decides, which only makes sense in tests.
        dependencies: Per-token structural dependencies from the backend, mapping a
            token to the tokens it is merged from, e.g. `Ġthe -> (Ġth, e)`.
        max_vocab_size: The hard cap, e.g. 32000.

    Raises:
        ValueError: If the structural tokens alone exceed the cap, e.g. `max_vocab_size:
            128` against the 282 structural tokens codefuse-ai/F2LLM-v2-160M cannot
            give up.
    """
    if len(selection.kept_ids) <= max_vocab_size:
        return
    if len(selection.structural_ids) > max_vocab_size:
        raise ValueError(
            f"max_vocab_size={max_vocab_size:,} is below the {len(selection.structural_ids):,} structural tokens "
            f"(added/special tokens, post-processor tokens, unk, byte alphabet) that cannot be removed "
            f"without breaking the tokenizer"
        )

    frequency = counts.counts if counts is not None else {}
    # Initially set to the ids that survived the union of structural, requested and corpus sources
    kept = selection.kept_ids

    # Each kept token can only be removed if nothing still depends on it. For example,
    # `Ġthe` depends on `Ġth` and `e`, so we do not drop `Ġth` while `Ġthe` is still kept.
    # This keeps the vocabulary reachable after the merge graph has been closed.
    dependent_count: dict[int, int] = defaultdict(int)
    for token_id in kept:
        for parent in dependencies.get(token_id, ()):
            dependent_count[parent] += 1

    def sort_key(token_id: int) -> tuple[int, int, int]:
        # Lowest corpus frequency gets deleted first. In case of a tie, always remove
        # the token with the highest token id first. The token id itself is the last
        # item so that we can easily recover it after popping from the heap
        return (frequency.get(token_id, 0), -token_id, token_id)

    # rm_candidates only has "leaf" tokens that are currently removable: tokens that are
    # kept, not structural, and whose dependent count is zero. We never add structural ids to
    # this heap because they are intentionally non-removable even if they are leaves
    rm_candidates = [sort_key(t) for t in kept - selection.structural_ids if dependent_count[t] == 0]
    # Because we need to easily get access to the least frequent token, we use a min-heap. It's
    # basically a binary tree where the root is the smallest element and each two descendants
    # are larger than the parent. Pushing and popping to/from it automatically and efficiently
    # keeps it sorted so that the next pop is always the least frequent token
    heapq.heapify(rm_candidates)

    removed = 0
    # A token enters the heap exactly once, either as an initial leaf or on the decrement
    # that takes its dependent count to zero, so whatever is popped is still kept and is
    # free to go: the pop needs no second check.
    while (len(kept) > max_vocab_size) and rm_candidates:
        token_id = heapq.heappop(rm_candidates)[2]
        kept.discard(token_id)
        removed += 1

        reasons = selection.provenance.pop(token_id, set()) - {CORPUS, DEPENDENCY}
        if reasons:
            selection.dropped_requested[token_id] = reasons

        # Dropping a token frees the tokens it was merged from, but only once their last
        # dependent is gone: removing `Ġthe` frees `Ġth` unless `Ġthese` still needs it
        for parent in dependencies.get(token_id, ()):
            dependent_count[parent] -= 1
            if dependent_count[parent] == 0 and parent not in selection.structural_ids:
                heapq.heappush(rm_candidates, sort_key(parent))

    if removed:
        logger.warning(f"max_vocab_size={max_vocab_size:,} forced the removal of {removed:,} further tokens")
    if len(kept) > max_vocab_size:
        logger.warning(
            f"could not shrink below {max_vocab_size:,} tokens without breaking token reachability;"
            f" keeping {len(kept):,} tokens"
        )
    if selection.dropped_requested:
        logger.warning(
            f"{len(selection.dropped_requested):,} explicitly requested tokens were dropped to satisfy max_vocab_size",
        )


def select_tokens(spec: TokenizerSpec, counts: CorpusCounts | None, config: SelectionConfig) -> Selection:
    """Decide which token ids survive the trim.

    The rule is the union of `structural`, `requested` and `corpus`, closed over the
    backend's token dependencies, followed by the `max_vocab_size` cap if one was asked
    for. Presets and explicit lists therefore act as a floor on the vocabulary rather
    than as a filter on it.

    Args:
        spec: The tokenizer being trimmed.
        counts: Corpus statistics, or `None` when selecting from explicit sources only.
        config: The selection settings.

    Returns:
        The kept ids together with the provenance of each. Keeping the special tokens,
        the byte alphabet, the digits, the chat template's words and one Dutch sentence
        leaves 461 of codefuse-ai/F2LLM-v2-160M's 151,669 tokens, 282 of them
        structural.
    """
    selection = Selection()

    selection.structural_ids = set(spec.structural_ids)
    for token_id in selection.structural_ids:
        selection.provenance[token_id].add(STRUCTURAL)

    for token_id, labels in _requested_ids(spec, config, selection).items():
        selection.provenance[token_id].update(labels)

    for token_id in _corpus_ids(counts, config):
        selection.provenance[token_id].add(CORPUS)

    # initially all tokens are kept
    selection.kept_ids = set(selection.provenance)

    dependencies = spec.backend.dependencies(spec)
    for token_id in _expand_dependencies(selection.kept_ids, dependencies):
        selection.provenance[token_id].add(DEPENDENCY)
        selection.kept_ids.add(token_id)

    if selection.unknown_tokens:
        logger.warning(
            f"{len(selection.unknown_tokens):,} requested tokens are not in this vocabulary"
            f" and were ignored (e.g. {', '.join(repr(token) for token in selection.unknown_tokens[:5])})"
        )

    if config.max_vocab_size is not None:
        _apply_cap(selection, counts, dependencies, config.max_vocab_size)

    logger.info(
        f"keeping {len(selection):,} of {spec.vocab_size:,} tokens ({100 * len(selection) / spec.vocab_size:.1f}%)",
    )
    return selection
