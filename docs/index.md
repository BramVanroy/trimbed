# 🛏️ trimbed

Trim a tokenizer's vocabulary, and optionally its model's embedding table, down to the
subset you actually need. You can specify "what you need" explicitly, by presets, or by
providing a corpus (one or more datasets) as an anchor for what you'd actually like to
keep.

The most typical use would be to provide a corpus of the domain and/or language that is
relevant for you, and then specifying to, for example, keep only the top 32k tokens from
that corpus, or to remove the tail of the corpus by only keeping 99.95% of its tokens.
But more options are available!

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

Installing the package comes with the `trimbed` command, which is the main entrypoint for
users. It has four subcommands: `trim`, `count`, `inspect` and `presets`. See
[Command line](cli.md), or run `trimbed --help`.

```bash
# What am I dealing with?
trimbed inspect --model codefuse-ai/F2LLM-v2-160M

# Trim tokenizer + model from a config file
trimbed trim --config my_config.yaml

# Same config with one value overridden and nothing written (dry run)
trimbed trim --config my_config.yaml --dry-run selection.top_k=30000

# Or without a config, keeping a preset
# (only the alphanumeric tokens in the vocab are kept),
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

[`TrimConfig`][trimbed.config.TrimConfig] is the whole run description and
[`TrimPipeline.run`][trimbed.pipeline.TrimPipeline.run] returns a
[`TrimReport`][trimbed.report.TrimReport].

## Where to go next

- [Command line](cli.md) — the four subcommands and their flags.
- [Configuration](configuration.md) — every config field, and how the three override
  layers stack.
- [How selection works](selection.md) — what survives a trim and why.
- [Trimming a model](trimming-a-model.md) — embedding and head surgery, and verification.
- [Output and reports](output.md) — what lands in the output directory.
- [Extending trimbed](extending.md) — a new tokenizer family or a new preset.
- [Examples](examples.md) — runnable scripts, one per usage pattern.
- [API reference](api/pipeline.md) — every public symbol.

## How it works

trimbed operates on the `tokenizers` backend document (`tokenizer.json`) rather than on
SentencePiece protobufs. Every Hugging Face fast tokenizer serialises to that format, and
`transformers` produces it even for spiece-only tokenizers, so one code path covers BPE,
WordPiece, Unigram and WordLevel.

The vocabulary surgery itself is delegated to
[skeletoken](https://github.com/stephantul/skeletoken), which owns the typed model of
`tokenizer.json`: compacting ids, filtering the merge table, pruning added tokens and
validating the result. trimbed owns everything around it — corpus counting, the selection
policy, model trimming, the commands and the report.
