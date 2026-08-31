"""Selection constraints for WordPiece tokenizers (BERT and friends)."""

from __future__ import annotations

from trimbed.backends import register_backend
from trimbed.backends.base import VocabBackend


@register_backend
class WordPieceBackend(VocabBackend):
    """Keeps a WordPiece vocabulary usable.

    WordPiece greedily matches the longest prefix in the vocabulary and needs no merge
    table, so tokens carry no dependencies: dropping `"##ing"` costs coverage but
    never makes another vocabulary token unreachable.
    """

    model_type = "WordPiece"
