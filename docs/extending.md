# Extending trimbed

## A new preset

The likely case: you want a custom rule for which tokens must be kept. Register one by
name with [`register_preset`][trimbed.presets.register_preset] and it becomes available to
`--keep-preset` and to `selection.keep_presets` like any built-in.

```python
from trimbed.presets import register_preset


@register_preset("emoji")
def _emoji(spec):
    """Keep every token whose surface form is an emoji."""
    return {t for t, surface in spec.surface_forms.items() if surface and _is_emoji(surface)}
```

The function is handed a [`TokenizerSpec`][trimbed.spec.TokenizerSpec] and returns the
token ids to keep. Its docstring is what `trimbed presets` prints as the description, so
write one. [`examples/04_custom_preset.py`](examples.md) is a runnable version.

A preset name ending in `:` is parametrised, and the argument arrives after the colon, as
in `script:Latin`. [`available_presets`][trimbed.presets.available_presets] lists what is
registered and [`resolve_preset`][trimbed.presets.resolve_preset] turns one name into ids.

## A new tokenizer family

trimbed relies on [skeletoken](https://github.com/stephantul/skeletoken) for the
serialisation. As long as your tokenizer type is supported there and in `transformers`,
adding a backend is one decorated class.

```python
from trimbed.backends import register_backend
from trimbed.backends.base import VocabBackend


@register_backend
class MyBackend(VocabBackend):
    """Selection constraints for MyModel tokenizers."""

    model_type = "MyModel"  # matches model.type in tokenizer.json

    def structural_tokens(self, spec):
        return super().structural_tokens(spec) | {"<my-required-token>"}
```

A [`VocabBackend`][trimbed.backends.base.VocabBackend] answers exactly two questions:

- [`structural_tokens`][trimbed.backends.base.VocabBackend.structural_tokens] — what can
  never be removed. For a byte-level tokenizer that includes all 256 alphabet characters;
  drop one and some byte sequences become unencodable.
- [`dependencies`][trimbed.backends.base.VocabBackend.dependencies] — what needs what. Only
  BPE builds tokens out of other tokens, so
  [`BpeBackend`][trimbed.backends.bpe.BpeBackend] is the only family that fills this in.

Both are properties of how a family *encodes* text and are invisible in the JSON, which is
why they live here rather than in skeletoken. Backends are selection constraints, not
serialisers: adding a family is one decorated class and nothing else changes.

[`supported_model_types`][trimbed.backends.supported_model_types] lists what is registered,
and `trimbed inspect` reports whether a given checkpoint's family is among them.
