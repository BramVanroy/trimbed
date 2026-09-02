# Examples

[`examples/`](https://github.com/BramVanroy/trimbed/tree/main/examples) has one runnable
script per usage pattern. Each takes `-m/--model`, so you can point any of them at a local
checkpoint, and each returns its report or result rather than printing and discarding it,
so the body copies straight into your own code.

| Script | What it shows |
|---|---|
| [`01_inspect_tokenizer.py`](https://github.com/BramVanroy/trimbed/blob/main/examples/01_inspect_tokenizer.py) | Build a [`TokenizerSpec`][trimbed.spec.TokenizerSpec] and describe a tokenizer without changing anything |
| [`02_trim_tokenizer_only.py`](https://github.com/BramVanroy/trimbed/blob/main/examples/02_trim_tokenizer_only.py) | Trim from must-keep rules alone, with no corpus and no model |
| [`03_trim_with_corpus.py`](https://github.com/BramVanroy/trimbed/blob/main/examples/03_trim_with_corpus.py) | The full [`TrimPipeline`][trimbed.pipeline.TrimPipeline] over a corpus, model included |
| [`04_custom_preset.py`](https://github.com/BramVanroy/trimbed/blob/main/examples/04_custom_preset.py) | Register your own preset with [`register_preset`][trimbed.presets.register_preset] |
| [`05_low_level_api.py`](https://github.com/BramVanroy/trimbed/blob/main/examples/05_low_level_api.py) | Drive the stages yourself: count, select, trim, verify |
| [`06_local_corpus.py`](https://github.com/BramVanroy/trimbed/blob/main/examples/06_local_corpus.py) | Count a corpus of local files with a [`DatasetSpec`][trimbed.config.DatasetSpec], offline |

The low-level path is worth a look if you want to slot trimbed into a larger job:
[`CorpusCounter`][trimbed.counting.CorpusCounter] →
[`select_tokens`][trimbed.selection.select_tokens] →
[`trim_tokenizer`][trimbed.tokenizer_trim.trim_tokenizer] →
[`trim_model`][trimbed.model_trim.trim_model] →
[`verify_tokenizer`][trimbed.verify.verify_tokenizer].

Every example runs in the test suite against tiny in-process fixtures, so none of them can
drift from the API without the suite going red.
