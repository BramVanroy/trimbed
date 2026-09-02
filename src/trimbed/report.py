"""The machine-readable and human-readable record of a trimming run."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


REPORT_FILENAME = "trim_report.json"
"""Name of the serialised report a run writes into its output directory."""

CONFIG_FILENAME = "_trimbed_config.yaml"
"""Name of the fully resolved configuration a run writes alongside the report."""


class _Base(BaseModel):
    """Report models forbid unknown fields and allow `model_` names."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class CorpusReport(_Base):
    """What the corpus pass saw."""

    documents: int = Field(description="Examples read across all datasets, e.g. 200000.")
    total_tokens: int = Field(description="Total token occurrences counted, e.g. 91204338.")
    distinct_tokens: int = Field(
        description="Distinct token ids the corpus used; the ceiling on what a corpus-only selection can keep."
    )
    coverage: float = Field(description="Fraction of corpus occurrences covered by the kept vocabulary, e.g. 0.9993.")


class VocabularyReport(_Base):
    """How the vocabulary changed."""

    model_type: str = Field(description="Tokenizer backend type, e.g. 'BPE'.")
    original_size: int = Field(description="Token count before trimming, e.g. 151669.")
    trimmed_size: int = Field(description="Token count after trimming, e.g. 32000.")
    structural_tokens: int = Field(
        description="Tokens that were never eligible for removal, e.g. 282 for a byte-level BPE with 26 added tokens."
    )
    kept_by_reason: dict[str, int] = Field(
        description="Kept-token counts per provenance label, e.g."
        " {'structural': 282, 'preset:byte_alphabet': 256, 'chat_template': 88, 'dependency': 101}."
        " An id kept for two reasons is counted under both."
    )
    unknown_requested_tokens: list[str] = Field(
        default_factory=list,
        description="Requested token strings absent from this vocabulary; usually a keep-list written"
        " for a different checkpoint.",
    )
    dropped_requested_tokens: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Requested tokens the size cap removed anyway, mapped to why they were requested.",
    )

    @property
    def reduction(self) -> float:
        """Return the fraction of the vocabulary that was removed, e.g. `0.789`."""
        return 1 - self.trimmed_size / self.original_size if self.original_size else 0.0


class ModelReport(_Base):
    """How the model changed.

    This is what `trimbed.model_trim.trim_model` returns, so the surgery's own
    result is already the thing the report serialises.

    Attributes:
        model_class: Class the checkpoint was loaded as, e.g. `BertForMaskedLM` or
            `Qwen3ForCausalLM`.
        old_embedding_rows: Rows in the input embedding matrix before trimming, e.g.
            151,936 for Qwen3-0.6B. Read off the matrix, so not necessarily the 151,669
            the tokenizer reports.
        new_embedding_rows: Rows after trimming, including any alignment padding, so
            `pad_to_multiple_of: 128` turns 32,000 kept tokens into 32,000 exactly and
            32,001 into 32,128.
        old_parameters: Total model parameters before trimming.
        new_parameters: Total model parameters after trimming.
        tied_embeddings: Whether input and output embeddings share weights. This is true
            for most current decoders, including every checkpoint the test suite covers.
        has_output_head: Whether the model has a separate output embedding at all. False
            for an encoder loaded as a base model, e.g. codefuse-ai/F2LLM-v2-160M.
    """

    model_class: str
    old_embedding_rows: int
    new_embedding_rows: int
    old_parameters: int
    new_parameters: int
    tied_embeddings: bool
    has_output_head: bool

    @property
    def parameters_removed(self) -> int:
        """Return how many parameters the trim eliminated."""
        return self.old_parameters - self.new_parameters


class VerificationReport(_Base):
    """How faithfully the trimmed tokenizer reproduces the original.

    `trimbed.verify.verify_tokenizer` fills this in as it walks the sample texts,
    so every field starts at its empty value.

    Attributes:
        checked: How many texts were compared.
        identical: Texts whose token sequence maps exactly through the remap.
        equivalent_text: Texts that decode back to the same string, even if the ids took a
            different route (possible when a merge was dropped).
        original_tokens: Tokens the original tokenizer produced over all samples.
        trimmed_tokens: Tokens the trimmed tokenizer produced over the same samples.
        failures: Sample texts that decoded differently, truncated for readability.
    """

    checked: int = 0
    identical: int = 0
    equivalent_text: int = 0
    original_tokens: int = 0
    trimmed_tokens: int = 0
    failures: list[str] = Field(default_factory=list)

    @property
    def exact_rate(self) -> float:
        """Return the fraction of texts whose ids mapped one-to-one."""
        return self.identical / self.checked if self.checked else 1.0

    @property
    def text_rate(self) -> float:
        """Return the fraction of texts that still decode to the original string."""
        return self.equivalent_text / self.checked if self.checked else 1.0

    @property
    def length_ratio(self) -> float:
        """Return how much longer the trimmed tokenizer's output is.

        Dropping a merge splits the affected words into more pieces, and every extra
        piece is paid for at inference time. A ratio above 1.0 is the size of that cost,
        e.g. `1.0000` when nothing the sample uses was lost, `1.03` when the trim costs
        3% more tokens on the same text.
        """
        return self.trimmed_tokens / self.original_tokens if self.original_tokens else 1.0

    @property
    def ok(self) -> bool:
        """Return whether every checked text round-tripped to the same string."""
        return not self.failures


class ModelVerificationReport(_Base):
    """How closely the trimmed model reproduces the original's outputs.

    Attributes:
        checked: How many texts both models ran on.
        skipped: Texts left out because their ids did not map one-to-one.
        max_hidden_diff: Largest absolute difference between the last hidden states,
            e.g. `4.8e-07` for a correct float32 trim.
        max_logit_diff: The same over the output head's logits, or `None` when the model
            has no head.
        tolerance: The threshold `ok` compares against, e.g. `1e-05`.
        max_length: How many tokens of each text the models were run on, e.g. `512` for a
            BERT, or `None` when nothing bounded the length.
    """

    checked: int = 0
    skipped: int = 0
    max_hidden_diff: float = 0.0
    max_logit_diff: float | None = None
    tolerance: float = 1e-5
    max_length: int | None = None

    @property
    def ok(self) -> bool:
        """Return whether both outputs stayed within `tolerance`."""
        return max(self.max_hidden_diff, self.max_logit_diff or 0.0) <= self.tolerance


class TrimReport(_Base):
    """The complete record of one trimming run."""

    model: str = Field(description="Model the tokenizer came from, e.g. 'codefuse-ai/F2LLM-v2-160M'.")
    output_dir: str | None = Field(
        default=None, description="Where artefacts were written, e.g. 'trimmed/f2llm-nl'; null for a dry run."
    )
    dry_run: bool = Field(default=False, description="Whether writing was skipped.")
    vocabulary: VocabularyReport
    corpus: CorpusReport | None = None
    model_trim: ModelReport | None = None
    verification: VerificationReport | None = None
    model_verification: ModelVerificationReport | None = None
    sidecar_files: list[str] = Field(
        default_factory=list, description="Files copied verbatim from the source repository."
    )

    def save(self, directory: str | Path) -> Path:
        """Write the report as JSON into a directory.

        Args:
            directory: Destination directory, created if it does not exist yet.

        Returns:
            The path written.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / REPORT_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    def render(self) -> str:
        """Return a compact human-readable summary for the terminal.

        One aligned `label  value` line per stage that ran, e.g.:

        ```
        model            codefuse-ai/F2LLM-v2-160M
        tokenizer type   BPE
        vocabulary       151,669 -> 32,000 (78.9% removed, 282 structural)
        ```
        """
        vocabulary = self.vocabulary
        lines = [
            f"model            {self.model}",
            f"tokenizer type   {vocabulary.model_type}",
            f"vocabulary       {vocabulary.original_size:,} -> {vocabulary.trimmed_size:,} "
            f"({vocabulary.reduction:.1%} removed, {vocabulary.structural_tokens:,} structural)",
        ]
        if self.corpus is not None:
            lines.append(
                f"corpus           {self.corpus.documents:,} docs, {self.corpus.total_tokens:,} tokens, "
                f"{self.corpus.coverage:.4%} covered by the kept vocabulary"
            )
        if self.model_trim is not None:
            trim = self.model_trim
            lines.append(f"architecture     {trim.model_class}")
            lines.append(
                f"embeddings       {trim.old_embedding_rows:,} -> {trim.new_embedding_rows:,} rows, "
                f"{trim.parameters_removed:,} of {trim.old_parameters:,} parameters removed "
                f"({trim.parameters_removed / trim.old_parameters:.1%})"
            )
        if self.verification is not None:
            check = self.verification
            lines.append(
                f"verification     {check.identical}/{check.checked} identical, "
                f"{check.equivalent_text}/{check.checked} decode-equivalent, "
                f"{check.length_ratio:.4f}x tokens"
            )
        if self.model_verification is not None:
            model_check = self.model_verification
            logit_diff = "n/a" if model_check.max_logit_diff is None else f"{model_check.max_logit_diff:.3g}"
            truncation = "" if model_check.max_length is None else f" of {model_check.max_length:,} tokens"
            lines.append(
                f"model check      {'passed' if model_check.ok else 'FAILED'} on {model_check.checked} texts"
                f"{truncation}: max |dh| {model_check.max_hidden_diff:.3g}, max |dlogit| {logit_diff} "
                f"(tolerance {model_check.tolerance:g})"
            )
        if self.sidecar_files:
            lines.append(f"copied files     {', '.join(self.sidecar_files)}")
        if vocabulary.kept_by_reason:
            reasons = ", ".join(f"{reason}={count:,}" for reason, count in vocabulary.kept_by_reason.items())
            lines.append(f"kept by          {reasons}")
        if vocabulary.unknown_requested_tokens:
            lines.append(f"not in vocab     {len(vocabulary.unknown_requested_tokens)} requested tokens were ignored")
        if vocabulary.dropped_requested_tokens:
            lines.append(
                f"cap removed      {len(vocabulary.dropped_requested_tokens)} explicitly requested tokens "
                "(max_vocab_size)"
            )
        lines.append(f"output           {self.output_dir or '(dry run, nothing written)'}")
        return "\n".join(lines)
