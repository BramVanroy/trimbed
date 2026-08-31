"""Pydantic configuration models, basically extended the models provided by `skeletoken`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trimbed.sidecar import DEFAULT_SIDECAR_PATTERNS


def _override_key(container: object, segment: str, dotted: str) -> str | int:
    """Turn one segment of a dotted override path into a usable key.

    Args:
        container: The dict or list the segment addresses.
        segment: The path segment, e.g. `"top_k"` or, for a list, `"0"`.
        dotted: The full path, for the error message, e.g.
            `"corpus.datasets.0.max_samples"`.

    Returns:
        `segment` itself for a mapping, or the integer index for a list. Negative indices
        work, so `corpus.datasets.-1.weight` addresses the last dataset.

    Raises:
        ValueError: If the container is a list and the segment is not a valid index.
    """
    if not isinstance(container, list):
        # if the container is a dict, then the segment is simply the key to access the value
        return segment

    # if the container is a list, then the segment must be an integer index
    try:
        index = int(segment)
    except ValueError as exc:
        raise ValueError(f"override {dotted!r}: {segment!r} addresses a list, so it must be an integer") from exc
    if not -len(container) <= index < len(container):
        raise ValueError(f"override {dotted!r}: index {index} is out of range for a list of {len(container)}")
    return index


class _StrictBase(BaseModel):
    """Shared model settings: unknown YAML keys are an error."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DatasetSpec(_StrictBase):
    """Hugging Face dataset to derive token frequencies from."""

    path: str = Field(description="Hub dataset id or local path, e.g. 'epfml/FineWeb2-HQ'.")
    name: str | None = Field(default=None, description="Dataset configuration name, e.g. 'nld_Latn'.")
    split: str = Field(default="train", description="Split expression, e.g. 'train' or even 'train[:1%]'.")
    text_column: str = Field(default="text", description="Column holding the raw text, e.g. 'text' or 'content'.")
    streaming: bool = Field(default=True, description="Stream instead of downloading the whole split.")
    max_samples: int | None = Field(default=None, ge=1, description="Stop after this many examples, e.g. 200000.")
    revision: str | None = Field(default=None, description="Dataset revision to pin. Recommended for reproducibility.")
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Multiplier applied to this corpus' token counts, e.g. 2.0 to let a small"
        " in-domain corpus weigh as much as twice its size against a large generic one.",
    )


class CorpusConfig(_StrictBase):
    """How the corpus, containing one or more datasets, is read and counted."""

    datasets: list[DatasetSpec] = Field(default_factory=list, description="Datasets to count tokens over.")
    batch_size: int = Field(default=1000, ge=1, description="Examples tokenized per batch.")
    num_proc: int | None = Field(default=None, ge=1, description="Worker processes (non-streaming datasets only).")
    counts_cache: Path | None = Field(
        default=None,
        description="Read counts from this JSON file if it exists, otherwise write them there, e.g. 'counts.json'.",
    )


class SelectionConfig(_StrictBase):
    """Which tokens survive the trim.

    The criteria are the union of `structural`, `requested` and `corpus`, then the
    `max_vocab_size` cap is applied to the remainder. Structural tokens (added/special
    tokens, the unk token, and the byte alphabet of byte-level tokenizers) are always
    kept and never counted against a criterion, because dropping them breaks the tokenizer.
    """

    coverage: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Keep the most frequent tokens until they cover this fraction of corpus occurrences,"
        " e.g. 0.9999. Values below about 0.99 trim far harder than you might expect!",
    )
    top_k: int | None = Field(
        default=None, ge=1, description="Keep at most this many corpus-derived tokens, e.g. 32000."
    )
    min_count: int | None = Field(
        default=None,
        ge=1,
        description="Keep tokens seen at least this many times,"
        " e.g. 10. 1 would mean: keep everything the corpus used.",
    )
    max_vocab_size: int | None = Field(
        default=None,
        ge=1,
        description="Hard cap on the final vocabulary: least-frequent non-structural tokens are dropped to fit."
        " E.g. 32000. It cannot go below the structural tokens, of which a byte-level BPE has at least 256.",
    )
    keep_presets: list[str] = Field(
        default_factory=list,
        description="Named presets to keep, e.g. 'alphanumeric'. Some presets are always included. Run"
        " `trimbed presets` to see the available presets.",
    )
    keep_tokens: list[str] = Field(
        default_factory=list,
        description="Literal token strings to keep, written as the vocabulary stores them,"
        " so 'Ġde' rather than ' de' for a byte-level tokenizer.",
    )
    keep_token_ids: list[int] = Field(default_factory=list, description="Literal token ids to keep, e.g. [151643].")
    keep_token_files: list[Path] = Field(
        default_factory=list,
        description="Text files with one token per line. Blank lines and '#' comments are ignored.",
    )
    keep_patterns: list[str] = Field(
        default_factory=list,
        description="Regular expressions matched against each token's decoded surface form,"
        r" e.g. '^\d+$' for numbers or '[À-ÿ]' for accented Latin characters. The surface form is matched,"
        " so a pattern is written against ' de' rather than the Unicode representation 'Ġde'.",
    )
    keep_texts: list[str] = Field(
        default_factory=list,
        description="Texts to keep encodable as they are now. E.g. a representative prompt, or the"
        " instruction format an SFT run will use.",
    )
    keep_chat_template: bool = Field(
        default=True,
        description="Keep the tokens the chat template's own words need, so prompts do not fragment.",
    )

    @property
    def has_explicit_sources(self) -> bool:
        """Return whether any must-keep source was configured.

        `keep_chat_template` is deliberately not counted: it is on by default and selects
        nothing at all for a tokenizer without a chat template, so treating it as a source
        would defeat the "nothing to select on" check.
        """
        return bool(
            self.keep_presets
            or self.keep_tokens
            or self.keep_token_ids
            or self.keep_token_files
            or self.keep_patterns
            or self.keep_texts
        )


class EmbeddingTrimConfig(_StrictBase):
    """How the model's embedding table and output head are resized."""

    pad_to_multiple_of: int | None = Field(
        default=None,
        ge=1,
        description="Pad the resized embedding matrix up to a multiple of this, for tensor-core alignment."
        " See for instance https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/#h.9yili3t5wcy5",
    )
    dtype: str | None = Field(default=None, description="Cast the model to this dtype, e.g. 'bfloat16'.")
    # TODO: check/benchmark if using a GPU is actually faster than CPU for the gather
    device: str = Field(default="cpu", description="Device used while selecting embedding rows.")
    auto_class: str | None = Field(
        default=None,
        description=(
            "transformers class used to load the model, e.g. 'AutoModelForCausalLM'. "
            "By default the class named in the checkpoint's config.architectures is used."
        ),
    )


class TrimConfig(_StrictBase):
    """Top-level configuration for one trimming run."""

    model: str = Field(
        description="Hub model id or local path whose tokenizer is trimmed, e.g. 'codefuse-ai/F2LLM-v2-160M'."
    )
    revision: str | None = Field(
        default=None, description="Model revision to pin, e.g. a commit sha. Recommended for reproducibility."
    )
    output_dir: Path = Field(
        default=Path("trimmed"),
        description="Directory the trimmed artefacts are written to, e.g. 'trimmed/f2llm-nl'.",
    )
    trim_model: bool = Field(default=True, description="Also trim the model's embeddings, not just the tokenizer.")
    overwrite: bool = Field(default=False, description="Allow (over)writing into a non-empty output directory.")
    trust_remote_code: bool = Field(
        default=False,
        description="Allow custom modelling/tokenizer code from the checkpoint, as gte and jina models need.",
    )
    verify: bool = Field(
        default=True,
        description="Re-encode sample texts to prove the trim is non-destructive and behaviour-preserving.",
    )
    verify_samples: int = Field(
        default=256,
        ge=1,
        description="How many corpus texts to verify the trimmed tokenizer against."
        " Also the pool the model comparison draws from, so it cannot be smaller"
        " than 'verify_model_samples'.",
    )
    verify_model: bool = Field(
        default=True,
        description="Also run both models on sample texts and compare their outputs.",
    )
    verify_model_samples: int = Field(
        default=8,
        ge=1,
        description="How many texts the model comparison runs on, taken from the front of the"
        " tokenizer verification sample. At most 'verify_samples'.",
    )
    verify_tolerance: float = Field(
        default=1e-5,
        gt=0,
        description="Largest absolute output difference the model comparison accepts.",
    )
    copy_sidecar_files: bool = Field(
        default=True,
        description="Copy vocabulary-independent files (sentence-transformers modules and the like) from the source.",
    )
    sidecar_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SIDECAR_PATTERNS),
        description="Glob patterns, relative to the source repository, selecting which files to copy,"
        " e.g. 'modules.json' and '[0-9]_*/*' for the numbered sentence-transformers modules."
        " Defaults to all sentence-transformers related files.",
    )
    seed: int = Field(default=0, description="Seed for corpus sampling.")

    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    embeddings: EmbeddingTrimConfig = Field(default_factory=EmbeddingTrimConfig)

    @model_validator(mode="after")
    def _require_a_selection_source(self) -> Self:
        """Reject configs that would keep either nothing or everything by accident."""
        if not self.corpus.datasets and not self.selection.has_explicit_sources:
            raise ValueError(
                "nothing to select on: configure at least one entry under 'corpus.datasets' "
                "or one of the 'selection.keep_*' options"
            )
        return self

    @model_validator(mode="after")
    def _fit_the_model_sample_inside_the_tokenizer_sample(self) -> Self:
        """Reject a model sample the tokenizer sample cannot supply."""
        if self.verify_model_samples > self.verify_samples:
            raise ValueError(
                f"verify_model_samples ({self.verify_model_samples}) is larger than verify_samples"
                f" ({self.verify_samples}), but the model comparison runs on the first"
                " verify_model_samples of the very texts the tokenizer verification uses, so it would"
                " quietly settle for fewer. Raise 'verify_samples' to at least"
                f" {self.verify_model_samples}"
            )
        return self

    @model_validator(mode="after")
    def _require_a_criterion_with_a_corpus(self) -> Self:
        """Error out when a corpus is given but no criterion narrows it down."""
        selection = self.selection
        if self.corpus.datasets and not any(
            criterion is not None for criterion in (selection.coverage, selection.top_k, selection.min_count)
        ):
            raise ValueError(
                "a corpus was configured but no criterion to narrow it down; set at least one of "
                "'selection.coverage', 'selection.top_k' or 'selection.min_count'"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a configuration from a YAML file.

        Args:
            path: Path to the YAML document, e.g. `"configs/f2llm_dutch.yaml"`.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the document is not a mapping, e.g. a file holding a bare list.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level, got {type(raw).__name__}")
        return cls.model_validate(raw)

    def with_overrides(self, overrides: dict[str, Any]) -> Self:
        """Return a copy with dotted-path overrides applied and revalidated.

        A path segment addressing a list is an index, so a single dataset in a mixture can
        be tuned without restating the rest: `corpus.datasets.0.max_samples`.

        Args:
            overrides: Mapping of dotted paths to values, e.g. `{"selection.top_k": 5000}`.
                Entries whose value is `None` are ignored, so unset flags are no-ops.

        Returns:
            A new, validated configuration.

        Raises:
            ValueError: If a segment indexing a list is not an integer, or is out of range.
        """
        data = self.model_dump(mode="json")
        for dotted, value in overrides.items():
            if value is None:
                continue
            *parents, leaf = dotted.split(".")
            target: Any = data
            for part in parents:
                key = _override_key(target, part, dotted)
                if isinstance(target, dict):
                    # A path into a section the YAML left out is still valid; the missing
                    # mapping is created and pydantic validates the result either way.
                    target.setdefault(key, {})
                target = target[key]
            target[_override_key(target, leaf, dotted)] = value
        return type(self).model_validate(data)

    def to_yaml(self) -> str:
        """Serialise the resolved configuration back to YAML, for provenance."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def load_config(config_path: str | Path | None, model: str | None = None) -> TrimConfig:
    """Build a configuration from a YAML file, or from a bare model id.

    The `trimbed` commands all accept `--config` and `--model`, and this is the shared
    resolution of the two. Overrides are deliberately not applied here, so a caller can
    merge flag-derived and `key=value` overrides in one `TrimConfig.with_overrides`
    call with a single precedence rule.

    Args:
        config_path: Path to a YAML document, or `None`.
        model: Hub model id or local path, used when no YAML file is given, e.g.
            `"codefuse-ai/F2LLM-v2-160M"`.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If neither a config file nor a model was supplied.
    """
    if config_path:
        return TrimConfig.from_yaml(config_path)
    if model:
        # Without a corpus something must still be kept, so minimally the structural tokens are kept
        # That includes the unk token and/or, for byte-level BPE, the whole byte alphabet
        return TrimConfig.model_validate({"model": model, "selection": {"keep_presets": ["structural"]}})
    raise ValueError("supply either --config or --model")


def parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    """Turn `key.path=value` strings into dotted-path overrides.

    Values go through `yaml.safe_load`, so `5000` becomes an int, `false` a bool and
    `[a, b]` a list, while a bare word stays a string. `null` (or an empty one) resolves to
    `None`, which `TrimConfig.with_overrides` skips: fields cannot be unset from the command line.

    Args:
        overrides: Strings of the form `key.path=value`, e.g.
            `["selection.top_k=5000", "trim_model=false", "corpus.datasets.0.max_samples=1000"]`.

    Returns:
        A mapping suitable for `TrimConfig.with_overrides`, e.g.
        `{"selection.top_k": 5000, "trim_model": False}`.

    Raises:
        ValueError: If an entry contains no `=`, e.g. a bare `"top_k"`.
    """
    parsed: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} must be of the form 'key.path=value'")
        dotted, raw_value = override.split("=", 1)
        parsed[dotted] = yaml.safe_load(raw_value)
    return parsed
