"""Trimming a model's embedding table (and output head) down to just the kept tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trimbed._logging import get_logger
from trimbed.loading import require_torch
from trimbed.report import ModelReport


if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedConfig, PreTrainedModel

    from trimbed.config import EmbeddingTrimConfig
    from trimbed.remap import IdRemap

logger = get_logger(__name__)

TOKEN_ID_ATTRIBUTES = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sep_token_id",
    "cls_token_id",
    "unk_token_id",
    "mask_token_id",
    "decoder_start_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
)
"""Config attributes holding a single token id that must follow the remap.

Which of them a checkpoint sets varies: Qwen3 has only `eos_token_id`, T5 adds
`decoder_start_token_id` and `pad_token_id`, BERT sets `pad_token_id` only.
"""


def _remap_config_token_ids(config: PreTrainedConfig | GenerationConfig, remap: IdRemap, label: str) -> None:
    """Point every token id stored on a config, and on its sub-configs, at its new value.

    Ids that did not survive are cleared to `None` since a stale id would silently
    index the wrong embedding row.

    Args:
        config: A `transformers` config or generation config. E.g. a Qwen3 config, whose
            `eos_token_id` is 151645 before the trim and a much smaller number after.
        remap: The old-id to new-id mapping.
        label: Name used in log messages, e.g. `"config"` or `"config.text_config"`.
    """
    for name in getattr(type(config), "sub_configs", None) or ():
        child = getattr(config, name, None)
        if child is not None:
            _remap_config_token_ids(child, remap, f"{label}.{name}")

    for attribute in TOKEN_ID_ATTRIBUTES:
        value = getattr(config, attribute, None)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            new_value = remap.old_to_new.get(value)
            if new_value is None:
                logger.warning(f"{label}.{attribute}={value} refers to a dropped token; clearing it")
            setattr(config, attribute, new_value)
        elif isinstance(value, list):
            setattr(config, attribute, [remap.to_new(item) for item in value if item in remap])


def trim_model(model: PreTrainedModel, remap: IdRemap, config: EmbeddingTrimConfig | None = None) -> ModelReport:
    """Shrink a model's vocabulary-sized tensors (embeddings, lm head) down to just the kept tokens.

    Rows are gathered before resizing and written back afterwards, because
    `resize_token_embeddings` keeps only the first `n` of the original rows, which is almost
    never the set we want (we select the right ones non-contiguously).

    Args:
        model: A loaded `transformers` model, e.g. a `Qwen3ForCausalLM` or a
            `BertForMaskedLM`.
        remap: Mapping from surviving old ids to contiguous new ids. Its `new_to_old`
            is used directly as the gather index.
        config: Embedding-trimming settings. Defaults are used when `None`.

    Returns:
        Statistics describing the change, e.g. 151,936 embedding rows down to 32,000.

    Raises:
        ValueError: If an id to keep lies outside the existing embedding matrix, which
            means the remap and the checkpoint disagree about the vocabulary.
    """
    torch = require_torch()

    input_embeddings = model.get_input_embeddings()
    num_old_embeds = int(input_embeddings.weight.shape[0])
    num_old_parameters = sum(parameter.numel() for parameter in model.parameters())

    if remap.new_to_old[-1] >= num_old_embeds:
        raise ValueError(
            f"token id {remap.new_to_old[-1]} is outside the embedding matrix, which has {num_old_embeds} rows"
        )
    num_declared = getattr(model.config, "vocab_size", None)
    if num_declared is not None and num_declared != num_old_embeds:
        logger.info(
            f"config.vocab_size={num_declared} differs from the {num_old_embeds} embedding rows; using the matrix"
        )

    index = torch.tensor(remap.new_to_old, dtype=torch.long, device=input_embeddings.weight.device)
    gathered_input = input_embeddings.weight.data.index_select(0, index).clone()

    output_embeddings = model.get_output_embeddings()
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    gathered_output = None
    gathered_bias = None
    if output_embeddings is not None:
        # if embeddings are tied, the output head is just a view of the input embeddings
        # so we don't need to set it separately
        if not tied:
            gathered_output = output_embeddings.weight.data.index_select(0, index).clone()
        # The head's bias is its own parameter even when the weights are tied, so tying
        # does not carry it along. We have to explicitly index-select on the bias too
        if getattr(output_embeddings, "bias", None) is not None:
            gathered_bias = output_embeddings.bias.data.index_select(0, index).clone()

    # do old-school embedding resizing with optional padding-to-multiple
    # this just sets the shape, to fill up later
    old_padding_idx = getattr(input_embeddings, "padding_idx", None)
    pad_to = config.pad_to_multiple_of if config is not None else None
    model.resize_token_embeddings(len(remap), pad_to_multiple_of=pad_to)

    # actually fill it back up again
    with torch.no_grad():
        new_embeddings = model.get_input_embeddings()
        new_embeddings.weight.data[: len(remap)] = gathered_input
        head = model.get_output_embeddings()
        if gathered_output is not None:
            head.weight.data[: len(remap)] = gathered_output
        if gathered_bias is not None:
            head.bias.data[: len(remap)] = gathered_bias

        # Personal taste: `pad_to_multiple_of` leaves rows past the end of the vocabulary
        # but `transformers` fills them from the mean of the existing embeddings, which makes
        # their logits more likely to end up as an "accidental generation". Instead, we set
        # them to zero so that is not possible
        if new_embeddings.weight.shape[0] > len(remap):
            new_embeddings.weight.data[len(remap) :].zero_()
            if gathered_output is not None:
                head.weight.data[len(remap) :].zero_()
            if gathered_bias is not None:
                head.bias.data[len(remap) :].zero_()

    # `resize_token_embeddings` carries `padding_idx` over unchanged so we
    # remap it manually
    if old_padding_idx is not None:
        new_embeddings.padding_idx = remap.old_to_new.get(old_padding_idx)
        if new_embeddings.padding_idx is None:
            logger.warning(f"the embedding's padding_idx={old_padding_idx} was dropped by the trim")

    _remap_config_token_ids(model.config, remap, "config")
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        _remap_config_token_ids(generation_config, remap, "generation_config")

    num_new_rows = int(model.get_input_embeddings().weight.shape[0])
    stats = ModelReport(
        model_class=type(model).__name__,
        old_embedding_rows=num_old_embeds,
        new_embedding_rows=num_new_rows,
        old_parameters=num_old_parameters,
        new_parameters=sum(parameter.numel() for parameter in model.parameters()),
        tied_embeddings=tied,
        has_output_head=output_embeddings is not None,
    )
    logger.info(
        f"trimmed embeddings: {stats.old_embedding_rows:,} -> {stats.new_embedding_rows:,}"
        f" rows, {stats.parameters_removed:,} parameters removed"
    )
    return stats
