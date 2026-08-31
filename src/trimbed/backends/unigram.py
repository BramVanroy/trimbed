"""Selection constraints for Unigram tokenizers (XLM-R, mT5, ALBERT, ...)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trimbed.backends import register_backend
from trimbed.backends.base import VocabBackend


if TYPE_CHECKING:
    from trimbed.spec import TokenizerSpec


@register_backend
class UnigramBackend(VocabBackend):
    """Keeps a Unigram vocabulary usable.

    Unigram scores whole candidate pieces independently rather than composing them from
    merges.
    """

    model_type = "Unigram"

    def structural_tokens(self, spec: TokenizerSpec) -> set[str]:
        """Return the unk token plus any byte-fallback tokens the model relies on.

        Args:
            spec: The tokenizer being trimmed.

        Returns:
            Token strings that may never be dropped, e.g. `{"<unk>"}` for
            google-t5/t5-small and FacebookAI/xlm-roberta-base.
        """
        return super().structural_tokens(spec) | self._byte_fallback_tokens(spec)
