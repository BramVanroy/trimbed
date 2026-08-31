# Examples

Small, self-contained scripts, roughly in order of how much of `trimbed` they touch.
Each one takes `-m/--model`, so any of them can be pointed at a local directory instead
of the Hub:

```bash
uv run python examples/01_inspect_tokenizer.py --model ./my-local-model
uv run python examples/02_trim_tokenizer_only.py --help
```

| Script | What it shows | Needs |
|---|---|---|
| [`01_inspect_tokenizer.py`](01_inspect_tokenizer.py) | Tokenizer family, real id-space size, structural floor | Hub |
| [`02_trim_tokenizer_only.py`](02_trim_tokenizer_only.py) | Trimming from must-keep rules alone, no corpus, no model | Hub |
| [`03_trim_with_corpus.py`](03_trim_with_corpus.py) | The main use case: corpus-derived vocabulary + embedding surgery | Hub, `trimbed[model]` |
| [`04_custom_preset.py`](04_custom_preset.py) | Registering your own must-keep rule and using it by name | Hub |
| [`05_low_level_api.py`](05_low_level_api.py) | Select / trim / verify as separate steps, writing nothing | Hub |

The [configuration guide](https://BramVanroy.github.io/trimbed/latest/configuration/)
holds the same ideas as YAML, for the `trimbed` command rather than the Python API.

Each example is split the same way the commands are: `run(...)` holds the logic and
returns its result, `main()` builds the argument parser and hands the parsed arguments
over as keyword arguments. `tests/test_examples.py` calls `run()` directly against tiny
in-process fixtures, so every example is executed on each `make test` and cannot drift
away from the API.
