"""Loading tokenizers and models."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from trimbed._logging import get_logger
from trimbed.exceptions import MissingDependencyError


if TYPE_CHECKING:
    from transformers import AutoModel, PreTrainedModel

    from trimbed.config import EmbeddingTrimConfig

logger = get_logger(__name__)


def load_tokenizer(model: str, revision: str | None = None, trust_remote_code: bool = False) -> PreTrainedTokenizerFast:
    """Load a tokenizer as a fast tokenizer, converting it if necessary.

    Tokenizers distributed only as a SentencePiece model (mT5's `spiece.model`, say)
    are converted to the `tokenizers` format on load so we can trim them following
    the same code path as everything else.

    Args:
        model: Hub model id or local path, e.g. `"codefuse-ai/F2LLM-v2-160M"` or
            `"./trimmed/f2llm-nl"`.
        revision: Optional revision to pin, e.g. `"refs/pr/1"` or a commit sha.
        trust_remote_code: Allow tokenizer code shipped with the checkpoint.

    Returns:
        A `transformers` fast tokenizer, e.g. a `Qwen2Tokenizer` for
        codefuse-ai/F2LLM-v2-160M or a `BertTokenizer` for google-bert/bert-base-cased.

    Raises:
        MissingDependencyError: If conversion needs `sentencepiece`/`protobuf`.
        ValueError: If the result is not backed by a `tokenizers.Tokenizer`.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, use_fast=True, trust_remote_code=trust_remote_code
        )
    except ImportError as exc:
        missing = "sentencepiece" if "sentencepiece" in str(exc).lower() else "protobuf"
        raise MissingDependencyError(
            missing, "convert", f"Converting the tokenizer of {model!r} to the fast format"
        ) from exc
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"{model!r} loaded as a slow tokenizer ({type(tokenizer).__name__})."
            " `trimbed` can only trim fast tokenizers."
            " There might be an underlying issue with the tokenizer that makes it"
            " impossible to convert it to a fast tokenizer."
        )
    return tokenizer


def require_torch() -> ModuleType:
    """Import and return torch or explain how to install it.

    Returns:
        The imported `torch` module.

    Raises:
        MissingDependencyError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError("torch", "model", "Trimming a model's embedding table") from exc
    return torch


def resolve_model_class(
    model: str,
    revision: str | None = None,
    config: EmbeddingTrimConfig | None = None,
    trust_remote_code: bool = False,
) -> type[PreTrainedModel] | type[AutoModel]:
    """Find the model class to load a checkpoint with.

    The class named in `config.architectures` is what the checkpoint actually contains,
    so that is what gets loaded.

    Args:
        model: Hub model id or local path.
        revision: Optional revision to pin.
        config: Embedding-trimming settings. Set `auto_class` to override the inference.
        trust_remote_code: Allow modelling code shipped with the checkpoint.

    Returns:
        A class exposing `from_pretrained`, read off `config.architectures`: e.g.
        `Qwen3ForCausalLM` for Qwen/Qwen3-0.6B, `BertForMaskedLM` for
        google-bert/bert-base-cased, `T5ForConditionalGeneration` for google-t5/t5-small,
        and `Qwen3Model` for codefuse-ai/F2LLM-v2-160M, which really is a base model with
        no head (it is an embedding model that uses last-token pooling). The second arm of
        the return type is the `AutoModel` fallback for a remote-code checkpoint whose
        class transformers does not export, which is separate because the auto classes
        are factories rather than `PreTrainedModel` subclasses.

    Raises:
        ValueError: If `EmbeddingTrimConfig.auto_class` names something transformers
            does not export, e.g. a typo like `"AutoModelForCasualLM"`, although
            perhaps it would not be bad if we'd have some more casual lm models. :-)
    """
    requested = config.auto_class if config is not None else None
    if requested is not None:
        model_class = getattr(transformers, requested, None)
        if model_class is None:
            raise ValueError(f"transformers has no class named {requested!r}. Check embeddings.auto_class")
        return model_class

    architectures = getattr(
        transformers.AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code),
        "architectures",
        None,
    )
    for architecture in architectures or ():
        model_class = getattr(transformers, architecture, None)
        if model_class is not None:
            return model_class

    if architectures:
        logger.warning(f"no transformers class named {', '.join(architectures)}; falling back to AutoModel")
    return transformers.AutoModel


def load_model(
    model: str,
    revision: str | None = None,
    config: EmbeddingTrimConfig | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    """Load a model for embedding trimming.

    Args:
        model: Hub model id or local path.
        revision: Optional revision to pin.
        config: Embedding-trimming settings, which control the class, dtype and
            placement, e.g. `dtype="bfloat16"` with `device="cuda"`.
        trust_remote_code: Allow modelling code shipped with the checkpoint, as the gte
            and jina encoders need.

    Returns:
        A `transformers` model in eval mode.
    """
    torch = require_torch()

    model_class = resolve_model_class(model, revision, config, trust_remote_code)
    kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": trust_remote_code}
    if config is not None and config.dtype:
        kwargs["dtype"] = getattr(torch, config.dtype)
    model = model_class.from_pretrained(model, **kwargs)
    if config is not None and config.device:
        model = model.to(config.device)
    model.eval()
    logger.info(f"loaded {model!r} ({type(model).__name__}) in eval mode")
    return model
