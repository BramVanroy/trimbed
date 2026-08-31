"""Selection constraints for WordLevel tokenizers."""

from __future__ import annotations

from trimbed.backends import register_backend
from trimbed.backends.base import VocabBackend


@register_backend
class WordLevelBackend(VocabBackend):
    """Keeps a flat word-level vocabulary usable.

    The simplest case: no merges, no subwords, so no dependencies. Anything dropped
    becomes unk, which makes the unk token the only structural requirement.
    """

    model_type = "WordLevel"
