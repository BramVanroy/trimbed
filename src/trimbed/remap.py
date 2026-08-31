"""Contiguous, order-preserving remapping of token ids."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class IdRemap:
    """A mapping from a kept subset of old token ids onto `0..n-1`.

    Old ids are sorted ascending before renumbering, so the surviving tokens keep their
    relative order. That is what makes `new_to_old` usable as-is when pruning the
    embeddings: it is the gather index that picks out the rows to keep.

    E.g. keeping old ids `{0, 5, 9}` out of ten gives `new_to_old == (0, 5, 9)` and
    `old_to_new == {0: 0, 5: 1, 9: 2}`.
    """

    new_to_old: tuple[int, ...]
    old_to_new: dict[int, int]

    @classmethod
    def from_kept(cls, kept_ids: Iterable[int]) -> Self:
        """Build a remap from the set of old ids that survive.

        Args:
            kept_ids: Old token ids to keep, e.g. `{9, 0, 5}`. Duplicates are collapsed
                and the order is irrelevant.

        Returns:
            A remap numbering the sorted kept ids from zero, so that example yields
            `new_to_old == (0, 5, 9)`.

        Raises:
            ValueError: If no ids were kept or an id is negative.
        """
        ordered = tuple(sorted(set(kept_ids)))
        if not ordered:
            raise ValueError("cannot build a remap from an empty set of kept ids")
        if ordered[0] < 0:
            raise ValueError(f"token ids must be non-negative, got {ordered[0]}")
        return cls(new_to_old=ordered, old_to_new={old: new for new, old in enumerate(ordered)})

    @classmethod
    def from_vocabularies(cls, old: dict[str, int], new: dict[str, int]) -> Self:
        """Build a mapping between two vocabularies, e.g. the original and the trimmed one.

        Only the tokens present in both are kept, numbered contiguously from zero.

        Args:
            old: Token -> id map before trimming, e.g. codefuse-ai/F2LLM-v2-160M's 151,669
                entries.
            new: Token -> id map after trimming, numbered contiguously from zero, e.g. the
                32,000 entries skeletoken left behind.

        Returns:
            A remap covering every token present in both vocabularies.

        Raises:
            ValueError: If the trimmed vocabulary introduces tokens the original
                lacked, or is not numbered contiguously from zero.
        """
        introduced = new.keys() - old.keys()
        if introduced:
            sample = ", ".join(sorted(introduced)[:5])
            raise ValueError(f"the trimmed vocabulary contains {len(introduced)} unknown tokens: {sample}")

        new_to_old = [0] * len(new)
        for token, new_id in new.items():
            if not 0 <= new_id < len(new):
                raise ValueError(f"trimmed id {new_id} for {token!r} is outside 0..{len(new) - 1}")
            new_to_old[new_id] = old[token]
        return cls(new_to_old=tuple(new_to_old), old_to_new={old_id: i for i, old_id in enumerate(new_to_old)})

    def __len__(self) -> int:
        """Return the number of surviving tokens."""
        return len(self.new_to_old)

    def __contains__(self, old_id: int) -> bool:
        """Return whether `old_id` survives the trim."""
        return old_id in self.old_to_new

    def __iter__(self) -> Iterator[int]:
        """Iterate over the kept old ids in ascending order."""
        return iter(self.new_to_old)

    def to_new(self, old_id: int) -> int:
        """Map an old id to its new id.

        Args:
            old_id: Id in the original vocabulary.

        Returns:
            The corresponding id in the trimmed vocabulary.

        Raises:
            KeyError: If `old_id` did not survive the trim.
        """
        return self.old_to_new[old_id]

    def to_old(self, new_id: int) -> int:
        """Map a new id back to the id it had in the original vocabulary.

        Args:
            new_id: Id in the trimmed vocabulary.

        Returns:
            The original id.
        """
        return self.new_to_old[new_id]

    def map_sequence(self, old_ids: Iterable[int]) -> list[int] | None:
        """Map a whole sequence or report that it cannot be mapped.

        Args:
            old_ids: Ids produced by the original tokenizer, e.g. `[9707, 1879]` for
                `"Hello world"` under codefuse-ai/F2LLM-v2-160M.

        Returns:
            The remapped ids, or `None` if any id was dropped by the trim. `None` is the
            signal `verify_model` uses to skip a text rather than
            compare two sequences that no longer correspond.
        """
        out: list[int] = []
        for old in old_ids:
            new = self.old_to_new.get(old)
            if new is None:
                return None
            out.append(new)
        return out
