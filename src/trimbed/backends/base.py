"""The base interface for different tokenizer families.

We luckily can rely on `skeletoken` to do the vocabulary surgery for every tokenizer
family, so these adapters only carry what a trimmer has to know and a serialiser does
not: which tokens can never be dropped, and which tokens depend on which.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    from trimbed.spec import TokenizerSpec


class VocabBackend(ABC):
    """Describes the structural constraints of one `tokenizers` model type.

    Subclasses answer which tokens can never be removed without breaking encoding
    outright and which tokens can only be produced if some other token survives.
    """

    model_type: ClassVar[str]
    """Value of `model.type` in tokenizer.json that this adapter handles.

    E.g. `"BPE"` for codefuse-ai/F2LLM-v2-160M, `"WordPiece"` for
    google-bert/bert-base-cased and `"Unigram"` for google-t5/t5-small.
    """

    def structural_tokens(self, spec: TokenizerSpec) -> set[str]:
        """Return tokens that must survive or the tokenizer stops working.

        The default covers the unknown token. Byte-level backends extend it with the byte
        alphabet.

        Args:
            spec: The tokenizer being trimmed.

        Returns:
            Token strings that may never be dropped. E.g. `{"[UNK]"}` for
            google-bert/bert-base-cased, and an empty set for codefuse-ai/F2LLM-v2-160M,
            which declares no unknown token at all.
        """
        unk = spec.unk_token
        return {unk} if unk else set()

    def dependencies(self, spec: TokenizerSpec) -> dict[int, tuple[int, ...]]:
        """Return which other tokens each token needs in order to stay reachable.

        Keeping a token while dropping something it is assembled from leaves it in the
        vocabulary but unreachable, so text silently fragments.
        Backends with no such structure return an empty mapping.

        Args:
            spec: The tokenizer being trimmed.

        Returns:
            A mapping of token id to the ids it directly depends on. Empty here, since
            BPE is the only family that builds tokens out of other tokens. See
            `BpeBackend.dependencies` for how it fills this in.
        """
        return {}

    def _byte_fallback_tokens(self, spec: TokenizerSpec) -> set[str]:
        """Return the `<0xNN>` tokens present in the vocabulary, if byte fallback is on.

        Args:
            spec: The tokenizer being trimmed.

        Returns:
            The byte-fallback tokens, e.g. `{"<0x00>", "<0x01>", ..., "<0xFF>"}`, or an
            empty set when the feature is off, as it is for google-t5/t5-small and
            FacebookAI/xlm-roberta-base. A byte-level BPE never needs it: its alphabet
            already covers all 256 bytes.
        """
        if not getattr(spec.model, "byte_fallback", False):
            return set()
        return {f"<0x{byte:02X}>" for byte in range(256)} & spec.vocabulary.keys()

    def __repr__(self) -> str:
        """Return a short debug representation."""
        return f"<{type(self).__name__} model_type={self.model_type!r}>"
