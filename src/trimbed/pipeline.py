"""End-to-end trimming pipeline."""

from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedTokenizerFast

from trimbed._logging import get_logger
from trimbed.config import TrimConfig
from trimbed.counting import CorpusCounter, CorpusCounts
from trimbed.loading import load_model, load_tokenizer
from trimbed.model_trim import trim_model
from trimbed.report import (
    CONFIG_FILENAME,
    CorpusReport,
    ModelReport,
    ModelVerificationReport,
    TrimReport,
    VerificationReport,
    VocabularyReport,
)
from trimbed.selection import Selection, select_tokens
from trimbed.sidecar import copy_sidecar_files
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import TrimmedTokenizer, trim_tokenizer
from trimbed.verify import verify_model, verify_tokenizer


logger = get_logger(__name__)


class TrimPipeline:
    """Runs a configured trim from corpus counting through to saved artefacts.

    The stages are exposed as methods so callers can drive them individually, like
    our entrypoints `trimbed count` and `trimbed inspect` do.
    """

    def __init__(self, config: TrimConfig) -> None:
        """Build a pipeline.

        Args:
            config: The validated run configuration.
        """
        self.config = config

    def load(self) -> tuple[PreTrainedTokenizerFast, TokenizerSpec]:
        """Load the tokenizer and parse its backend document (typically tokenizer.json).

        Returns:
            The fast tokenizer and a spec over its tokenizer.json, e.g. a `Qwen2Tokenizer`
            and a BPE spec of 151,669 tokens.
        """
        tokenizer = load_tokenizer(self.config.model, self.config.revision, self.config.trust_remote_code)
        spec = TokenizerSpec.from_tokenizer(tokenizer, source=self.config.model)
        logger.info(
            f"loaded {spec.model_type} tokenizer with {spec.vocab_size:,} tokens ({len(spec.added_tokens):,} added)"
        )
        return tokenizer, spec

    def count(self, tokenizer: PreTrainedTokenizerFast) -> CorpusCounts | None:
        """Count token frequencies over the configured corpus.

        Args:
            tokenizer: The tokenizer used to encode the corpus.

        Returns:
            The corpus statistics, or `None` when no dataset was configured.
        """
        if not self.config.corpus.datasets:
            logger.info("no datasets configured. Selecting tokens from explicit sources only")
            return None
        counter = CorpusCounter(
            tokenizer,
            self.config.corpus,
            seed=self.config.seed,
            sample_size=self.config.verify_samples,
        )
        return counter.count()

    def run(self, dry_run: bool = False) -> TrimReport:
        """Execute the whole pipeline.

        Args:
            dry_run: Select and report but write nothing to disk. Useful for sweeping
                different `top_k` values against a cached count file to see what each
                value would keep.

        Returns:
            The report describing what happened.
        """
        tokenizer, spec = self.load()
        counts = self.count(tokenizer)
        selection = select_tokens(spec, counts, self.config.selection)

        if dry_run:
            logger.info("dry run: skipping tokenizer and model trimming")
            return self._build_report(spec, selection, counts, dry_run=True)

        output_dir = self._prepare_output_dir()
        trimmed = trim_tokenizer(tokenizer, spec, selection.kept_ids)
        trimmed.tokenizer.save_pretrained(output_dir)

        texts = self._verification_texts(counts)
        verification = None
        if self.config.verify:
            if texts:
                verification = verify_tokenizer(tokenizer, trimmed.tokenizer, trimmed.remap, texts)
            else:
                logger.info("verification skipped: no texts to verify against")

        model_trim = None
        model_verification = None
        sidecar_files: list[str] = []
        if self.config.trim_model:
            model_trim = self._trim_model(trimmed, output_dir)
            if self.config.copy_sidecar_files:
                sidecar_files = copy_sidecar_files(
                    self.config.model, output_dir, self.config.sidecar_patterns, self.config.revision
                )
            model_verification = self._verify_model(tokenizer, trimmed, output_dir, texts)

        report = self._build_report(
            spec,
            selection,
            counts,
            model_trim=model_trim,
            verification=verification,
            model_verification=model_verification,
            sidecar_files=sidecar_files,
        )
        report.save(output_dir)
        # also copy the config to the output dir (including its potential CLI overrides)
        (output_dir / CONFIG_FILENAME).write_text(self.config.to_yaml(), encoding="utf-8")
        logger.info(f"wrote trimmed artefacts to {output_dir!r}")
        return report

    def _verification_texts(self, counts: CorpusCounts | None) -> list[str]:
        """Return the texts both verification passes run on.

        `keep_texts` comes first and is used whether or not a corpus was configured: a
        text the run was told to keep encodable is the one worth proving it kept, and it
        is the only thing to verify against for a trim driven by must-keep rules alone.
        Putting it first also puts it inside the much smaller sample the model comparison
        takes, which is the front `verify_model_samples` of this same list. `TrimConfig`
        caps `verify_model_samples` at `verify_samples` so that sample is one the corpus
        can actually supply.

        Args:
            counts: Corpus statistics, or `None` when no corpus was configured.

        Returns:
            The sample texts, possibly empty.
        """
        return [*self.config.selection.keep_texts, *(counts.samples if counts is not None else [])]

    def _trim_model(self, trimmed: TrimmedTokenizer, output_dir: Path) -> ModelReport:
        """Load, trim and save the model that goes with the tokenizer.

        Args:
            trimmed: The already-trimmed tokenizer, whose remap is reused.
            output_dir: Where the trimmed model is written.

        Returns:
            The model-side statistics.
        """
        model = load_model(
            self.config.model, self.config.revision, self.config.embeddings, self.config.trust_remote_code
        )
        report = trim_model(model, trimmed.remap, self.config.embeddings)
        model.save_pretrained(output_dir)
        return report

    def _verify_model(
        self,
        tokenizer: PreTrainedTokenizerFast,
        trimmed: TrimmedTokenizer,
        output_dir: Path,
        texts: list[str],
    ) -> ModelVerificationReport | None:
        """Compare the trimmed model against the original on sample texts.

        Args:
            tokenizer: The original tokenizer.
            trimmed: The trimmed tokenizer and its remap.
            output_dir: Where the trimmed model was written.
            texts: The texts the tokenizer verification used. The comparison runs on the
                first `verify_model_samples` of them, so both passes see the same texts.

        Returns:
            The comparison result, or `None` when it was skipped.
        """
        if not self.config.verify_model:
            return None
        if not texts:
            logger.info("model verification skipped: no texts to verify against")
            return None

        original = load_model(
            self.config.model, self.config.revision, self.config.embeddings, self.config.trust_remote_code
        )
        reloaded = load_model(str(output_dir), None, self.config.embeddings, self.config.trust_remote_code)
        return verify_model(
            original,
            reloaded,
            tokenizer,
            trimmed.tokenizer,
            trimmed.remap,
            texts[: self.config.verify_model_samples],
            tolerance=self.config.verify_tolerance,
        )

    def _prepare_output_dir(self) -> Path:
        """Create the output directory.

        Returns:
            The prepared directory, e.g. `Path("trimmed/f2llm-nl")`.

        Raises:
            FileExistsError: If the directory has contents and `overwrite` is not set,
                which is what stops a second run from writing a trimmed model next to the
                first run's tokenizer.
        """
        output_dir = self.config.output_dir
        if output_dir.exists() and any(output_dir.iterdir()) and not self.config.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass overwrite: true (or --overwrite) to write into it")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_report(
        self,
        spec: TokenizerSpec,
        selection: Selection,
        counts: CorpusCounts | None,
        *,
        dry_run: bool = False,
        model_trim: ModelReport | None = None,
        verification: VerificationReport | None = None,
        model_verification: ModelVerificationReport | None = None,
        sidecar_files: list[str] | None = None,
    ) -> TrimReport:
        """Assemble the run report from the individual stage results.

        Every stage after selection can be switched off.

        Args:
            spec: The original tokenizer spec.
            selection: Which tokens survived and why.
            counts: Corpus statistics, if a corpus was used.
            dry_run: Whether writing was skipped.
            model_trim: Model-side statistics, if the model was trimmed.
            verification: Verification result, if verification ran.
            model_verification: Model comparison result, if that check ran.
            sidecar_files: Files copied verbatim from the source repository.

        Returns:
            The assembled report.
        """
        vocabulary = VocabularyReport(
            model_type=spec.model_type,
            original_size=spec.vocab_size,
            trimmed_size=len(selection),
            structural_tokens=len(selection.structural_ids),
            kept_by_reason=selection.counts_by_reason(),
            unknown_requested_tokens=selection.unknown_tokens,
            dropped_requested_tokens={
                spec.id_to_token.get(token_id, str(token_id)): sorted(reasons)
                for token_id, reasons in selection.dropped_requested.items()
            },
        )
        corpus = None
        if counts is not None:
            corpus = CorpusReport(
                documents=counts.num_documents,
                total_tokens=counts.total_num_tokens,
                distinct_tokens=counts.distinct_tokens,
                coverage=counts.coverage_of(selection.kept_ids),
            )
        return TrimReport(
            model=self.config.model,
            output_dir=None if dry_run else str(self.config.output_dir),
            dry_run=dry_run,
            vocabulary=vocabulary,
            corpus=corpus,
            model_trim=model_trim,
            verification=verification,
            model_verification=model_verification,
            sidecar_files=sidecar_files or [],
        )
