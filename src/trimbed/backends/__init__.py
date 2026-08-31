"""Registry of per-tokenizer-family trimming adapters.

Adding support for a new tokenizer family means adding one module here and decorating
its class with `register_backend`, following one of the existing backends. Nothing else
in the package needs to change.
"""

from __future__ import annotations

from trimbed.backends.base import VocabBackend


_REGISTRY: dict[str, VocabBackend] = {}


def register_backend[BackendT: type[VocabBackend]](cls: BackendT) -> BackendT:
    """Register a backend adapter under its declared `model_type`.

    Args:
        cls: A concrete `VocabBackend` subclass.

    Returns:
        The class unchanged, so this can be used as a decorator.

    Raises:
        ValueError: If the class has no `model_type` or the type is already taken.
    """
    model_type = getattr(cls, "model_type", None)
    if not model_type:
        raise ValueError(f"{cls.__name__} must declare a `model_type` class attribute")
    if model_type in _REGISTRY:
        raise ValueError(f"a backend for model type {model_type!r} is already registered")
    _REGISTRY[model_type] = cls()
    return cls


def get_backend(model_type: str) -> VocabBackend:
    """Look up the adapter for a `tokenizer.json` model type.

    Args:
        model_type: Value of `model.type`, e.g. `"BPE"` for codefuse-ai/F2LLM-v2-160M
            or `"WordPiece"` for google-bert/bert-base-cased.

    Returns:
        The registered adapter instance, e.g. a `BpeBackend`.

    Raises:
        KeyError: If no adapter handles that model type.
    """
    try:
        return _REGISTRY[model_type]
    except KeyError:
        supported = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(
            f"no trimming backend registered for tokenizer model type {model_type!r} (supported: {supported})"
        ) from None


def supported_model_types() -> tuple[str, ...]:
    """Return the tokenizer model types that can currently be trimmed.

    E.g. `("BPE", "Unigram", "WordLevel", "WordPiece")`.
    """
    return tuple(sorted(_REGISTRY))


# Importing the adapters is what populates the registry, since that is when their
# `register_backend` decorators run.
from trimbed.backends.bpe import BpeBackend
from trimbed.backends.unigram import UnigramBackend
from trimbed.backends.wordlevel import WordLevelBackend
from trimbed.backends.wordpiece import WordPieceBackend


__all__ = [
    "BpeBackend",
    "UnigramBackend",
    "VocabBackend",
    "WordLevelBackend",
    "WordPieceBackend",
    "get_backend",
    "register_backend",
    "supported_model_types",
]
