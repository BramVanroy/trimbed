"""Trim a tokenizer's vocabulary, and optionally its model, down to what you need.

Typical use:

    >>> from trimbed import TrimConfig, TrimPipeline
    >>> config = TrimConfig.from_yaml("config.yaml")
    >>> report = TrimPipeline(config).run()
"""

from __future__ import annotations

from importlib.metadata import version

from trimbed.backends import VocabBackend, get_backend, register_backend, supported_model_types
from trimbed.compare import ComparisonReport, compare_tokenizers
from trimbed.config import (
    CorpusConfig,
    DatasetSpec,
    EmbeddingTrimConfig,
    SelectionConfig,
    TrimConfig,
    load_config,
    parse_overrides,
)
from trimbed.counting import CorpusCounter, CorpusCounts
from trimbed.exceptions import MissingDependencyError
from trimbed.model_trim import trim_model
from trimbed.pipeline import TrimPipeline
from trimbed.presets import available_presets, register_preset, resolve_preset
from trimbed.remap import IdRemap
from trimbed.report import ModelReport, ModelVerificationReport, TrimReport, VerificationReport
from trimbed.selection import Selection, select_tokens
from trimbed.sidecar import DEFAULT_SIDECAR_PATTERNS, copy_sidecar_files
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import TrimmedTokenizer, trim_tokenizer
from trimbed.verify import verify_model, verify_tokenizer


__version__ = version("trimbed")

__all__ = [
    "DEFAULT_SIDECAR_PATTERNS",
    "ComparisonReport",
    "CorpusConfig",
    "CorpusCounter",
    "CorpusCounts",
    "DatasetSpec",
    "EmbeddingTrimConfig",
    "IdRemap",
    "MissingDependencyError",
    "ModelReport",
    "ModelVerificationReport",
    "Selection",
    "SelectionConfig",
    "TokenizerSpec",
    "TrimConfig",
    "TrimPipeline",
    "TrimReport",
    "TrimmedTokenizer",
    "VerificationReport",
    "VocabBackend",
    "__version__",
    "available_presets",
    "compare_tokenizers",
    "copy_sidecar_files",
    "get_backend",
    "load_config",
    "parse_overrides",
    "register_backend",
    "register_preset",
    "resolve_preset",
    "select_tokens",
    "supported_model_types",
    "trim_model",
    "trim_tokenizer",
    "verify_model",
    "verify_tokenizer",
]
