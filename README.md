# 🛏️ `trimbed`

[![CI](https://github.com/BramVanroy/trimbed/actions/workflows/ci.yml/badge.svg)](https://github.com/BramVanroy/trimbed/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BramVanroy/trimbed/branch/main/graph/badge.svg)](https://codecov.io/gh/BramVanroy/trimbed)
[![Documentation](https://img.shields.io/badge/docs-github.io-teal)](https://BramVanroy.github.io/trimbed)
![PyPI version](https://img.shields.io/pypi/v/trimbed)
[![Python versions](https://img.shields.io/pypi/pyversions/trimbed.svg)](https://pypi.org/project/trimbed/)
[![License](https://img.shields.io/github/license/BramVanroy/trimbed)](LICENSE)

Trim a tokenizer's vocabulary, and optionally its model's embedding table, down to the
subset you actually need. You can specify "what you need" explicitly, by presets, or by
providing a corpus (one or more datasets) as an anchor for what you'd actually like to keep.

The most typical use would be to provide a corpus of the domain and/or language that is relevant
for you, and then specifying to, for example, keep only the top 32k tokens from that corpus,
or to remove the tail of the corpus by only keeping 99.95% of its tokens. But more options
are available!

**[Read the documentation](https://BramVanroy.github.io/trimbed)** for the full guides and
the API reference.

## Installation

```bash
pip install trimbed                       # tokenizer trimming only
pip install "trimbed[model]"              # + torch, for trimming embeddings
pip install "trimbed[model,convert]"      # + sentencepiece/protobuf, for spiece-only tokenizers
```

Or preferably with `uv add`.

From a git clone the above would be `uv sync`, `uv sync --extra model` and
`uv sync --all-extras`.

## Quickstart

Installing the package comes with the `trimbed` command, which is the main entrypoint
for users. It has a number of subcommands: `trim`, `count`, `inspect` and `presets`.
`trimbed --help` lists them.

If you have only cloned the repository, `python -m trimbed.cli` works too.

```bash
# What am I dealing with?
trimbed inspect --model codefuse-ai/F2LLM-v2-160M

# Trim tokenizer + model from a config file
trimbed trim --config my_config.yaml

# Same config but overridden one value and nothing written (dry-run)
trimbed trim --config my_config.yaml --dry-run selection.top_k=30000

# Or without a config, keeping a preset
# (only all alphanumeric tokens in the vocab are kept),
# and the model is not trimmed
trimbed trim --model google-bert/bert-base-multilingual-cased \
    --keep-preset alphanumeric --output-dir trimmed/bert --no-trim-model
```

From Python:

```python
from trimbed import TrimConfig, TrimPipeline

config = TrimConfig.from_yaml("my_config.yaml")
report = TrimPipeline(config).run()
print(report.render())
```

[`examples/`](examples/) has a few Python examples: inspecting a tokenizer, trimming from
rules alone, trimming over a corpus with the model, and registering your own preset.

## What it does

- **One code path for every family.** trimbed operates on the `tokenizers` backend
  document (`tokenizer.json`) rather than on SentencePiece protobufs, so BPE, WordPiece,
  Unigram and WordLevel are all covered. The vocabulary surgery itself is delegated to
  [skeletoken](https://github.com/stephantul/skeletoken).
- **Selection with provenance.** Structural tokens, must-keep rules and the corpus
  frequency ranking are unioned, closed over BPE merge dependencies, then capped. Every
  kept token records why it survived.
- **Prompts and chat templates stay intact.** `keep_texts` and `keep_chat_template` keep
  the ids a prompt is made of, so it goes on tokenizing exactly as it does now.
- **The model follows.** Embedding table, output head, head bias, tied and untied,
  encoder-decoders, `pad_to_multiple_of` alignment rows, and the token ids stored on the
  config and generation config.
- **It proves its work.** Both tokenizers re-encode sampled texts, and both models can be
  run and compared, before anything is called a success.
- **Nothing has to come from the Hub.** The checkpoint can be a local directory, and the
  corpus a directory of JSON Lines, a `json`/`csv`/`parquet` loader pointed at your own
  files, or a dataset written with `save_to_disk`.

## Documentation

| | |
|---|---|
| [Command line](https://BramVanroy.github.io/trimbed/latest/cli/) | The four subcommands and their flags |
| [Configuration](https://BramVanroy.github.io/trimbed/latest/configuration/) | Every config field, and how the override layers stack |
| [How selection works](https://BramVanroy.github.io/trimbed/latest/selection/) | What survives a trim and why |
| [Trimming a model](https://BramVanroy.github.io/trimbed/latest/trimming-a-model/) | Embedding and head surgery, and verification |
| [Output and reports](https://BramVanroy.github.io/trimbed/latest/output/) | What lands in the output directory |
| [Extending trimbed](https://BramVanroy.github.io/trimbed/latest/extending/) | A new tokenizer family or a new preset |
| [API reference](https://BramVanroy.github.io/trimbed/latest/api/pipeline/) | Every public symbol |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `make style`, `make test` and
`make build-docs` before you push.

## Acknowledegments

We rely heavily on the internals of [`skeletoken`](https://github.com/stephantul/skeletoken/) to map out tokenizers to a standardized Pydantic format.
