"""Selection constraints for byte-pair-encoding tokenizers (GPT-2, Qwen, Llama, ...)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trimbed.backends import register_backend
from trimbed.backends.base import VocabBackend
from trimbed.bytelevel import byte_level_alphabet


if TYPE_CHECKING:
    from trimbed.spec import TokenizerSpec


@register_backend
class BpeBackend(VocabBackend):
    """Keeps a BPE vocabulary encodable and its merge chains intact."""

    model_type = "BPE"

    def structural_tokens(self, spec: TokenizerSpec) -> set[str]:
        """Return the unk token plus, for byte-level BPE, the whole byte alphabet.

        Args:
            spec: The tokenizer being trimmed.

        Returns:
            Token strings that may never be dropped. For codefuse-ai/F2LLM-v2-160M that is
            the 256 alphabet characters (`"!"`, `"Ġ"` for a space, `"Ċ"` for a
            newline, ...) and nothing else, since it has no unknown token.
        """
        required = super().structural_tokens(spec)
        if spec.uses_byte_level:
            # add the byte-level alphabet, but only the bytes that are actually in the vocabulary
            required |= set(byte_level_alphabet()) & spec.vocabulary.keys()
        return required | self._byte_fallback_tokens(spec)

    def dependencies(self, spec: TokenizerSpec) -> dict[int, tuple[int, ...]]:
        """Map each merged token to the pair it is assembled from.

        codefuse-ai/F2LLM-v2-160M builds `"Ġthe"` by merging `"Ġth"` with `"e"`, so
        dropping `"Ġth"` leaves `"Ġthe"` in the vocabulary but unreachable and the text
        quietly tokenizes into characters instead. The selector uses this mapping to pull
        in whatever a kept token is built from.

        Args:
            spec: The tokenizer being trimmed.

        Returns:
            A mapping of merged token id to its two parent ids, e.g. `Ġthe -> (Ġth, e)`
            and `Ġworld -> (Ġw, orld)`. That is 151,387 entries for
            codefuse-ai/F2LLM-v2-160M, one per merge rule whose three tokens all survive.
        """
        vocabulary = spec.vocabulary
        dependencies: dict[int, tuple[int, ...]] = {}
        for left, right in spec.model.merges.root:
            # concatenate the strings and check if the merged token is actually in vocab
            merged = left + right
            if merged in vocabulary and left in vocabulary and right in vocabulary:
                # The first rule producing a token is the one BPE actually applies.
                dependencies.setdefault(vocabulary[merged], (vocabulary[left], vocabulary[right]))
        return dependencies
