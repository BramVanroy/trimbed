"""Proving that a trimmed tokenizer still behaves like the original."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from trimbed._logging import get_logger
from trimbed.loading import require_torch
from trimbed.report import ModelVerificationReport, VerificationReport


if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from trimbed.remap import IdRemap

logger = get_logger(__name__)

MAX_REPORTED_FAILURES = 5

# Covers both a `BatchEncoding` straight from a tokenizer and the plain dict
type TensorInputs = Mapping[str, torch.Tensor]


def verify_tokenizer(
    original: PreTrainedTokenizerFast,
    trimmed: PreTrainedTokenizerFast,
    remap: IdRemap,
    texts: Sequence[str],
) -> VerificationReport:
    """Compare the trimmed tokenizer against the original on real text.

    Id-level identity (`new_ids == remap(old_ids)`) proves the trim was purely a
    renumbering, while text-level identity can still hold when a dropped merge splits a
    word into more pieces. Both are reported because they fail for different reasons.

    Args:
        original: The tokenizer before trimming.
        trimmed: The tokenizer after trimming.
        remap: The mapping applied during the trim.
        texts: Sample texts to compare on, e.g. the corpus plus anything named
            in `keep_texts`.

    Returns:
        Counts of exact and text-equivalent matches, plus a few failing samples. A healthy
        trim reports every text identical and a length ratio of 1.0. A ratio of, say,
        1.02 means dropped merges cost 2% more tokens on this sample.
    """
    result = VerificationReport()
    for text in texts:
        old_ids = original(text, add_special_tokens=False).input_ids
        new_ids = trimmed(text, add_special_tokens=False).input_ids
        result.checked += 1
        result.original_tokens += len(old_ids)
        result.trimmed_tokens += len(new_ids)

        if remap.map_sequence(old_ids) == new_ids:
            result.identical += 1
            result.equivalent_text += 1
            continue

        if trimmed.decode(new_ids) == original.decode(old_ids):
            result.equivalent_text += 1
        elif len(result.failures) < MAX_REPORTED_FAILURES:
            result.failures.append(text[:200])

    logger.info(
        f"verification: {result.identical:,}/{result.checked:,} texts encoded identically,"
        f" {result.equivalent_text:,}/{result.checked:,} decode to the same text,"
        f" {result.length_ratio:.4f}x as many tokens"
    )
    if result.failures:
        logger.warning(f"{len(result.failures):,} sample texts no longer round-trip. See the report for examples")
    return result


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the largest absolute elementwise difference between two tensors.

    Mostly used for sanity checks.

    Args:
        left: A tensor.
        right: A tensor of the same shape.

    Returns:
        The maximum absolute difference, as a float. A correct trim gives something around
        `1e-7` in float32, which is floating-point noise.
    """
    return float((left.detach().cpu().float() - right.detach().cpu().float()).abs().max())


def _decoder_primed(model: PreTrainedModel, inputs: TensorInputs) -> TensorInputs:
    """Add the one decoder step an encoder-decoder needs to produce logits.

    A seq2seq model needs an initial decoder input token to produce logits.

    Args:
        model: A `transformers` model, e.g. a `T5ForConditionalGeneration`, whose
            `decoder_start_token_id` is 0.
        inputs: Keyword tensors from a tokenizer, already on the model's device.

    Returns:
        `inputs` unchanged for a decoder-only or encoder-only model, e.g. a Qwen3 or a
        BERT, otherwise a copy with `decoder_input_ids` added.

    Raises:
        ValueError: If the trim dropped the model's decoder start token.
    """
    if not getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        return inputs

    torch = require_torch()
    start_id = getattr(model.config, "decoder_start_token_id", None)
    if start_id is None:
        raise ValueError(
            f"{type(model).__name__} is an encoder-decoder whose decoder_start_token_id did not survive the trim, "
            "so it cannot be run. Make sure the token is included in the trimmed model or set `verify_model: false`"
        )
    batch = inputs["input_ids"].shape[0]
    start = torch.full((batch, 1), start_id, dtype=torch.long, device=inputs["input_ids"].device)
    return {**inputs, "decoder_input_ids": start}


def _outputs_of(model: PreTrainedModel, inputs: TensorInputs) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a model and pull out its last hidden state and, if any, its logits.

    Args:
        model: A `transformers` model.
        inputs: Keyword tensors from a tokenizer, already on the model's device.

    Returns:
        The last hidden state and the logits. The logits are `None` without an output
        head, as for codefuse-ai/F2LLM-v2-160M, which loads as a bare `Qwen3Model`.

    Raises:
        ValueError: If the model exposes no hidden state to compare on.
    """
    outputs = model(**_decoder_primed(model, inputs), output_hidden_states=True)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        states = getattr(outputs, "hidden_states", None) or getattr(outputs, "decoder_hidden_states", None)
        hidden = states[-1] if states else None
    if hidden is None:
        raise ValueError(
            f"{type(model).__name__} returned neither last_hidden_state nor hidden_states, "
            "so there is nothing to compare; set verify_model: false for this architecture"
        )
    return hidden, getattr(outputs, "logits", None)


def verify_model(
    original: PreTrainedModel,
    trimmed: PreTrainedModel,
    original_tokenizer: PreTrainedTokenizerFast,
    trimmed_tokenizer: PreTrainedTokenizerFast,
    remap: IdRemap,
    texts: Sequence[str],
    tolerance: float = 1e-5,
) -> ModelVerificationReport:
    """Run both models on the same texts and compare what they produce.

    Only with a foreward pass we can check that the trimmed model behaves like the original
    (except for doing a full elementwise comparison, which is not feasible for large models).

    Logits are compared through the remap, since the trimmed head has one
    column per kept token. Alignment padding and texts whose ids do not map one-to-one are
    left out, as neither corresponds to anything in the original.

    Args:
        original: The model before trimming.
        trimmed: The model after trimming.
        original_tokenizer: The tokenizer before trimming.
        trimmed_tokenizer: The tokenizer after trimming.
        remap: The mapping applied during the trim.
        texts: Sample texts to compare on.
        tolerance: Largest absolute difference accepted, e.g. `1e-5`. Raise it for a
            model loaded in bfloat16 where accumulated error may be far higher than in float32.

    Returns:
        The largest differences observed, and how many texts they came from.
    """
    torch = require_torch()

    result = ModelVerificationReport(tolerance=tolerance)
    index = torch.tensor(remap.new_to_old, dtype=torch.long)
    original_device = next(original.parameters()).device
    trimmed_device = next(trimmed.parameters()).device

    with torch.inference_mode():
        for text in texts:
            old_inputs = original_tokenizer(text, return_tensors="pt")
            new_inputs = trimmed_tokenizer(text, return_tensors="pt")
            if remap.map_sequence(old_inputs["input_ids"][0].tolist()) != new_inputs["input_ids"][0].tolist():
                result.skipped += 1
                continue

            old_hidden, old_logits = _outputs_of(original, old_inputs.to(original_device))
            new_hidden, new_logits = _outputs_of(trimmed, new_inputs.to(trimmed_device))
            result.checked += 1

            result.max_hidden_diff = max(result.max_hidden_diff, _max_diff(old_hidden, new_hidden))
            if old_logits is not None and new_logits is not None:
                gathered = old_logits.index_select(-1, index.to(old_logits.device))
                difference = _max_diff(gathered, new_logits[..., : len(remap)])
                result.max_logit_diff = max(result.max_logit_diff or 0.0, difference)

    logger.info(
        f"model verification: {result.checked:,} texts, max hidden difference {result.max_hidden_diff:.3g},"
        f" max logit difference {'n/a' if result.max_logit_diff is None else f'{result.max_logit_diff:.3g}'}"
        f" (tolerance {tolerance:g})"
    )
    if result.skipped:
        logger.info(f"{result.skipped:,} texts were skipped because their ids no longer map one-to-one")
    if not result.ok:
        logger.warning(f"the trimmed model does not reproduce the original within {tolerance:g}; see the report")
    return result
