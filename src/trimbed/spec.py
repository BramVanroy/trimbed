"""An inspectable view of a tokenizer, backed by skeletoken's typed data model.

Every Hugging Face (fast) tokenizer serialises to one JSON file (`tokenizer.json`),
regardless of whether it is BPE, WordPiece, Unigram or WordLevel.
[skeletoken](https://github.com/stephantul/skeletoken) supplies a typed, validated
model of that document, so here we only add what is needed for trimming:
decoded surface forms for preset matching, and a lookup of the backend adapter that
knows which tokens must never be dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Self

from skeletoken import TokenizerModel
from skeletoken.addedtoken import AddedToken
from skeletoken.models import Model
from skeletoken.post_processors import get_tokens_from_post_processor

from trimbed.bytelevel import decode_byte_level


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from trimbed.backends.base import VocabBackend

JINJA_CONSTRUCT = re.compile(r"\{[{%#].*?[}%#]\}", re.DOTALL)
"""`{{ expression }}`, `{% statement %}` and `{# comment #}`.

Everything a Jinja template evaluates rather than outputs literally. `re.DOTALL` makes
`.` match newlines.
"""

QUOTED_STRING = re.compile(r"'([^']*)'|\"([^\"]*)\"")
"""A quoted string inside a Jinja construct.

Role names reach the output through a comparison (`message['role'] == 'user'`) rather
than as literal text, so they only show up here.
"""

CHAT_ROLES = ("system", "user", "assistant", "tool", "function", "developer")
"""Roles a ChatML-style template substitutes from the message without ever naming.

Such a template writes `{{ message['role'] }}` without being explicit about which roles
are expected, so as a precaution the standard ones are kept even when the template does
not mention them anywhere.
"""


def _join_chat_templates(template: str | dict[str, str] | None) -> str | None:
    """Normalise a tokenizer's `chat_template` attribute to one string.

    Args:
        template: A template string, a mapping of named templates, or `None`. Most
            checkpoints carry one string, but a mapping shows up where a repository
            offers variants side by side, e.g. `{"default": ..., "tool_use": ...}`.

    Returns:
        The template text, with several named templates concatenated, or `None`.
    """
    if isinstance(template, dict):
        return "\n".join(str(value) for value in template.values()) or None
    return str(template) if template else None


@dataclass
class TokenizerSpec:
    """A tokenizer, ready to be inspected and trimmed.

    Attributes:
        tokenizer_model: The skeletoken model of the tokenizer.json document.
        source: Where the tokenizer came from (model id or path), for error messages.
            E.g. `"codefuse-ai/F2LLM-v2-160M"` or `"trimmed/f2llm-nl/tokenizer.json"`.
        chat_template: The Jinja chat template, when the tokenizer ships one. It lives in
            tokenizer_config.json rather than in tokenizer.json, so a spec built from a
            bare document does not have it.
    """

    tokenizer_model: TokenizerModel
    source: str | None = None
    chat_template: str | None = None

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerFast, source: str | None = None) -> Self:
        """Build a spec from a fast tokenizer object.

        Args:
            tokenizer: A `transformers` fast tokenizer.
            source: Optional provenance label.

        Returns:
            The parsed spec.

        Raises:
            ValueError: If the object is not backed by a `tokenizers.Tokenizer`.
        """
        if not hasattr(tokenizer, "backend_tokenizer"):
            raise ValueError(
                f"{type(tokenizer).__name__} is not backed by a fast `tokenizers.Tokenizer`; "
                "trimbed can only trim fast tokenizers"
            )
        label = source or getattr(tokenizer, "name_or_path", None)
        return cls(
            tokenizer_model=TokenizerModel.from_transformers_tokenizer(tokenizer),
            source=label,
            chat_template=_join_chat_templates(getattr(tokenizer, "chat_template", None)),
        )

    @classmethod
    def from_json_str(cls, payload: str, source: str | None = None) -> Self:
        """Build a spec from a serialised `tokenizer.json`.

        Args:
            payload: The JSON text.
            source: Optional provenance label.

        Returns:
            The parsed spec.
        """
        return cls(tokenizer_model=TokenizerModel.from_string(payload), source=source)

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Build a spec from a `tokenizer.json` file on disk.

        Args:
            path: Path to the JSON file.

        Returns:
            The parsed spec.
        """
        return cls.from_json_str(Path(path).read_text(encoding="utf-8"), source=str(path))

    @property
    def model(self) -> Model:
        """Return the typed `model` sub-object (BPE, WordPiece, Unigram or WordLevel)."""
        return self.tokenizer_model.model

    @property
    def model_type(self) -> str:
        """Return the backend model type.

        E.g. `"BPE"` for codefuse-ai/F2LLM-v2-160M, `"WordPiece"` for
        google-bert/bert-base-cased, `"Unigram"` for google-t5/t5-small.
        """
        return str(self.model.type.value)

    @cached_property
    def backend(self) -> VocabBackend:
        """Return the registered adapter for this tokenizer's model type."""
        # Imported here to avoid circular imports because every backend module imports TokenizerSpec in turn
        from trimbed.backends import get_backend

        return get_backend(self.model_type)

    @cached_property
    def vocabulary(self) -> dict[str, int]:
        """Return the token -> id map, added tokens included, since skeletoken folds those in.

        E.g. for codefuse-ai/F2LLM-v2-160M this has 151,669 entries, holding ordinary
        tokens like `"Ġthe"` alongside added ones like `"<|im_end|>"` (id 151,645) with
        nothing to tell them apart here.
        """
        return dict(self.tokenizer_model.vocabulary)

    @cached_property
    def id_to_token(self) -> dict[int, str]:
        """Return the id -> token map, e.g. `{9707: "Hello", 1879: "Ġworld", ...}`."""
        return {token_id: token for token, token_id in self.vocabulary.items()}

    @property
    def added_tokens(self) -> list[AddedToken]:
        """Return the typed `added_tokens` entries.

        E.g. 26 of them for codefuse-ai/F2LLM-v2-160M, starting at `<|endoftext|>`
        (id 151643), `<|im_start|>` (151644) and `<|im_end|>` (151645).
        """
        return list(self.tokenizer_model.added_tokens.root)

    @cached_property
    def added_token_ids(self) -> frozenset[int]:
        """Return the ids of all added tokens, e.g. 26 ids for codefuse-ai/F2LLM-v2-160M."""
        return frozenset(token.id for token in self.added_tokens)

    @cached_property
    def special_token_ids(self) -> frozenset[int]:
        """Return the ids of added tokens flagged `special`.

        Flagging happens in the `added_tokens` section of tokenizer.json (mirrored by
        `added_tokens_decoder` in tokenizer_config.json), where each added token carries
        a handful of flags, one of them `"special": true`. Qwen 3 marks `"<|endoftext|>"`
        and `"<|im_end|>"` as special but leaves `"<tool_call>"` and `"<think>"`
        ordinary.

        A subset of `added_token_ids`: codefuse-ai/F2LLM-v2-160M flags 14 of its 26,
        the rest being ordinary additions the checkpoint made no promises about.
        """
        return frozenset(token.id for token in self.added_tokens if token.special)

    @cached_property
    def post_processor_token_ids(self) -> frozenset[int]:
        """Return the ids of tokens the post-processor names.

        E.g. `{101, 102}` for google-bert/bert-base-cased, which is `[CLS]` and `[SEP]`,
        or `{151645}` (`<|im_end|>`) for codefuse-ai/F2LLM-v2-160M. You get an empty set
        for HuggingFaceTB/SmolLM2-135M-Instruct, which has no post-processor.
        """
        post_processor = self.tokenizer_model.post_processor
        if post_processor is None:
            return frozenset()
        vocabulary = self.vocabulary
        tokens = get_tokens_from_post_processor(post_processor)
        return frozenset(vocabulary[token] for token in tokens if token in vocabulary)

    @cached_property
    def structural_ids(self) -> frozenset[int]:
        """Return the ids that must survive for the tokenizer to keep working.

        That is every added token, every token the post-processor names, plus whatever the
        backend declares important (typically the unk token, and the byte alphabet for
        byte-level tokenizers). This is what the trim keeps whether or not a corpus or a
        preset asks for it, and it is what the `structural` preset resolves to.

        For codefuse-ai/F2LLM-v2-160M that is 282 ids: the 256 byte-alphabet characters
        plus the 26 added tokens. The three sources happily overlap, so this is their
        union rather than their sum. google-bert/bert-base-cased is the clearest case: it
        contributes `[CLS]`/`[SEP]` from the post-processor and `[UNK]` from the backend,
        and all three are already among its 5 added tokens.
        """
        vocabulary = self.vocabulary
        ids = set(self.added_token_ids) | set(self.post_processor_token_ids)
        ids |= {vocabulary[token] for token in self.backend.structural_tokens(self) if token in vocabulary}
        return frozenset(ids)

    @cached_property
    def structural_tokens(self) -> set[str]:
        """Return the token strings behind `structural_ids`, e.g. `{"[UNK]", "[CLS]", ...}` for BERT."""
        return {self.id_to_token[token_id] for token_id in self.structural_ids}

    @property
    def vocab_size(self) -> int:
        """Return the number of distinct token ids in the tokenizer.

        This is the tokenizer's own count, which is not the model's `config.vocab_size`:
        codefuse-ai/F2LLM-v2-160M reports 151,669 here while its config declares 151,936
        rows. That is because the embedding matrix is padded but the vocabulary is not.
        """
        return self.tokenizer_model.vocabulary_size

    @property
    def max_id(self) -> int:
        """Return the largest token id in use, e.g. 151,668 for codefuse-ai/F2LLM-v2-160M."""
        return max(self.id_to_token) if self.id_to_token else -1

    @property
    def uses_byte_level(self) -> bool:
        """Return whether the tokenizer maps text through the ByteLevel alphabet.

        When it does, every one of the 256 byte-alphabet characters must survive
        trimming or some inputs become unencodable. True for BPE checkpoints
        (e.g. codefuse-ai/F2LLM-v2-160M, HuggingFaceTB/SmolLM2-135M-Instruct), false for
        e.g. google-bert/bert-base-cased and google-t5/t5-small.
        """
        return bool(self.tokenizer_model.transforms_into_bytes)

    @cached_property
    def surface_forms(self) -> dict[str, str | None]:
        """Map every vocabulary token to the text it actually stands for.

        Byte-level tokens are decoded back to text, and the prefixes skeletoken reports, such as
        WordPiece's `##` continuation marker and the `▁` character standing for a
        leading space, are undone. Tokens that are partial UTF-8 sequences map to `None`,
        since they stand for no well-formed text on their own.

        E.g. `"Ġde" -> " de"` and `"Ġ" -> " "` for codefuse-ai/F2LLM-v2-160M, where the
        lone byte `"¡"` maps to `None`. For google-bert/bert-base-cased you get
        `"##ing" -> "ing"`, and for google-t5/t5-small `"▁the" -> " the"`.
        """
        if self.uses_byte_level:
            return {token: decode_byte_level(token) for token in self.vocabulary}

        # continuation marker for a broken subword, e.g. `##` for WordPiece
        continuing = self.tokenizer_model.continuing_subword_prefix or ""
        # leading-space marker for a subword that starts a new word, e.g. `▁` for Unigram
        initial = self.tokenizer_model.initial_subword_prefix
        surfaces: dict[str, str | None] = {}
        for token in self.vocabulary:
            text = token.removeprefix(continuing)
            surfaces[token] = text.replace(initial, " ") if initial else text
        return surfaces

    def encode(self, text: str) -> list[int]:
        """Return the ids this tokenizer currently produces for a piece of text.

        Args:
            text: The text to encode, e.g. `"De kat zat op de mat."`.

        Returns:
            The token ids. codefuse-ai/F2LLM-v2-160M answers that example with
            `[1912, 44256, 1147, 266, 1179, 409, 5517, 13]`, which is
            `De | Ġkat | Ġz | at | Ġop | Ġde | Ġmat | .`.
        """
        return list(self.tokenizer_model.tokenizer.encode(text, add_special_tokens=False).ids)

    @property
    def chat_template_literals(self) -> str:
        """Return the fixed text a chat template works with, with its Jinja removed.

        The Jinja syntax is removed so what is left is every literal text the template may
        put around the message content, including the ones only a tool-call or system-prompt
        branch reaches.

        Quoted strings inside the markup are kept too, since a role name
        often reaches the output through `message['role'] == 'user'`, and the standard
        role names are added outright (see [`CHAT_ROLES`][trimbed.spec.CHAT_ROLES]).

        Returns:
            The literals, newline-separated, or an empty string without a template. For
            codefuse-ai/F2LLM-v2-160M this starts with the role names and runs on
            through the template's whitespace and its quoted strings. Encoding it needs 89
            distinct tokens, which is what the trim has to keep.
        """
        if not self.chat_template:
            return ""
        parts = [*CHAT_ROLES, *JINJA_CONSTRUCT.split(self.chat_template)]
        for construct in JINJA_CONSTRUCT.findall(self.chat_template):
            for match in QUOTED_STRING.finditer(construct):
                parts.append(match.group(1) if match.group(1) is not None else match.group(2))
        return "\n".join(parts)

    @property
    def unk_token(self) -> str | None:
        """Return the backend's "unknown" token if it declares one.

        E.g. `"[UNK]"` for google-bert/bert-base-cased, `"<unk>"` for
        google-t5/t5-small, `"<|endoftext|>"` for HuggingFaceTB/SmolLM2-135M-Instruct.
        A byte-level BPE needs none, so codefuse-ai/F2LLM-v2-160M returns `None`.
        """
        return self.tokenizer_model.unk_token

    def describe(self) -> dict[str, str | int | bool | None]:
        """Return a small summary suitable for logging.

        Returns:
            A JSON-serialisable summary of the tokenizer's shape. For
            google-bert/bert-base-cased: `{"model_type": "WordPiece", "vocab_size": 28996,
            "added_tokens": 5, "special_tokens": 5, "max_token_id": 28995,
            "uses_byte_level": false, "unk_token": "[UNK]", "has_post_processor": true,
            "has_chat_template": false}`, plus `source`.
        """
        return {
            "source": self.source,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "added_tokens": len(self.added_tokens),
            "special_tokens": len(self.special_token_ids),
            "max_token_id": self.max_id,
            "uses_byte_level": self.uses_byte_level,
            "unk_token": self.unk_token,
            "has_post_processor": self.tokenizer_model.post_processor is not None,
            "has_chat_template": self.chat_template is not None,
        }
