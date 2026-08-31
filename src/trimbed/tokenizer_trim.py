"""Applying a trim to a tokenizer, via skeletoken's typed data model."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from trimbed._logging import get_logger
from trimbed.remap import IdRemap
from trimbed.spec import TokenizerSpec


if TYPE_CHECKING:
    from skeletoken import TokenizerModel

logger = get_logger(__name__)

LEGACY_VOCAB_FILES = (
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "source.spm",
    "target.spm",
)
"""Vocabulary files a fast tokenizer may write alongside tokenizer.json.

They are regenerated from the backend on save, so a stale copy left in the output
directory would describe the untrimmed vocabulary.
"""


@dataclass
class TrimmedTokenizer:
    """A trimmed tokenizer and the mapping that produced it.

    Attributes:
        tokenizer: The reloaded `transformers` fast tokenizer, read back from disk rather
            than constructed in memory, which is what proves it loads.
        spec: A spec over the trimmed tokenizer, whose `source` is the original's with
            `" (trimmed)"` added to disambiguate.
        remap: The old-id to new-id mapping that was applied.
    """

    tokenizer: PreTrainedTokenizerFast
    spec: TokenizerSpec
    remap: IdRemap


def build_trimmed_model(spec: TokenizerSpec, kept_ids: Iterable[int]) -> TokenizerModel:
    """Remove everything outside `kept_ids` from the tokenizer.

    skeletoken performs the surgery and validates the result: it compacts the ids, filters
    the merge table, prunes the added tokens and renumbers every id another component
    hard-codes (the post-processor's `special_tokens`, `padding.pad_id`, Unigram's
    `unk_id`, maybe others...)

    Args:
        spec: The tokenizer being trimmed.
        kept_ids: Ids that survive, e.g. the 461 that a preset-and-chat-template selection
            leaves of codefuse-ai/F2LLM-v2-160M's 151,669.

    Returns:
        The trimmed skeletoken model, renumbered from zero.
    """
    keep = set(kept_ids)
    doomed_tokens = [token for token, token_id in spec.vocabulary.items() if token_id not in keep]
    logger.info(f"removing {len(doomed_tokens):,} of {spec.vocab_size:,} tokens")
    return spec.tokenizer_model.remove_tokens_from_vocabulary(doomed_tokens)


def _rewrite_added_tokens_decoder(config_path: Path, remap: IdRemap) -> None:
    """Remap the id-keyed `added_tokens_decoder` map inside tokenizer_config.json.

    This mirrors `added_tokens` in tokenizer.json. Not updating it is a problem because
    reloading then re-registers the special tokens at their original ids, which silently
    corrupts the trimmed vocabulary.

    TODO: Unsure if this is something that we'd expect upstream in skeletoken.

    Args:
        config_path: Path to tokenizer_config.json, whose `added_tokens_decoder` looks
            like `{"151643": {"content": "<|endoftext|>", ...}, ...}`.
        remap: The old-id to new-id mapping.
    """
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        return
    config["added_tokens_decoder"] = {
        str(remap.to_new(int(old_id))): entry for old_id, entry in decoder.items() if int(old_id) in remap
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def trim_tokenizer(
    tokenizer: PreTrainedTokenizerFast, spec: TokenizerSpec, kept_ids: Iterable[int]
) -> TrimmedTokenizer:
    """Apply a trim to a fast tokenizer and return the trimmed result.

    The trimmed tokenizer config is round-tripped through `save_pretrained` and
    `AutoTokenizer.from_pretrained` rather than through skeletoken's `to_transformers`,
    because the latter constructs a fresh tokenizer object and so loses what lives outside
    tokenizer.json: the chat template and the rest of tokenizer_config.json. Reloading also
    proves the result actually loads.

    Args:
        tokenizer: The original fast tokenizer.
        spec: A spec over that tokenizer.
        kept_ids: Ids that survive the trim.

    Returns:
        The trimmed tokenizer, its spec, and the remap that was applied.

    Raises:
        RuntimeError: If the reloaded tokenizer does not match the trimmed model.
    """
    trimmed_tok_model = build_trimmed_model(spec, kept_ids)
    remap = IdRemap.from_vocabularies(spec.vocabulary, dict(trimmed_tok_model.vocabulary))

    with tempfile.TemporaryDirectory(prefix="trimbed-") as staging:
        p_stage = Path(staging)
        # Save original tokenizer
        tokenizer.save_pretrained(p_stage)
        # ... but overwrite the tokenizer.json and tokenizer_config.json with the trimmed model
        (p_stage / "tokenizer.json").write_text(trimmed_tok_model.to_tokenizer().to_str(), encoding="utf-8")
        _rewrite_added_tokens_decoder(p_stage / "tokenizer_config.json", remap)
        # remove legacy files that now may conflict with the new ground truth in tokenizer*.json
        for name in LEGACY_VOCAB_FILES:
            (p_stage / name).unlink(missing_ok=True)
        reloaded = AutoTokenizer.from_pretrained(p_stage)

    new_spec = TokenizerSpec.from_tokenizer(reloaded, source=f"{spec.source or 'tokenizer'} (trimmed)")
    if new_spec.vocab_size != len(remap):
        raise RuntimeError(
            f"trimmed tokenizer has {new_spec.vocab_size:,} tokens but the remap expected {len(remap):,}; "
            "this usually means added tokens were re-registered at their original ids"
        )
    logger.info(f"trimmed tokenizer: {spec.vocab_size:,} -> {new_spec.vocab_size:,} tokens")
    return TrimmedTokenizer(tokenizer=reloaded, spec=new_spec, remap=remap)
