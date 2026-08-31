# 🛏️ `trimbed`

Trim a tokenizer's vocabulary, and optionally its model's embedding table, down to the
subset you actually need. You can specify "what you need" explicitly, by presets, or by
providing a corpus (one or more datasets) as an anchor for what you'd actually like to keep.

The most typical use would be to provide a corpus of the domain and/or language that is relevant
for you, and then specifying to, for example, keep only the top 32k tokens from that corpus,
or to remove the tail of the corpus by only keeping 99.95% of its tokens. But more options
are available!

## Installation

PyPi install coming soon.

```bash
uv sync                      # tokenizer trimming only
uv sync --extra model        # + torch, for trimming embeddings
uv sync --all-extras         # + sentencepiece/protobuf, for spiece-only tokenizers
```

## Quickstart

Installing the package comes with the `trimbed` command, which is the main entrypoint
for users. It has a number of subcommands: `trim`, `count`, `inspect` and `presets`.
`trimbed --help` lists them.

If you have only cloned the repository, `python -m trimbed.cli` works too.

```bash
# What am I dealing with?
uv run trimbed inspect --model codefuse-ai/F2LLM-v2-160M

# Trim tokenizer + model from a config file
uv run trimbed trim --config configs/f2llm_dutch.yaml

# Same config but overridden one value and nothing written (dry-run)
uv run trimbed trim --config configs/f2llm_dutch.yaml --dry-run \
    selection.top_k=30000

# Or without a config, keeping a preset
# (only all alphanumeric tokens in the vocab are kept),
# and the model is not trimmed
uv run trimbed trim --model bert-base-multilingual-cased \
    --keep-preset alphanumeric --output-dir trimmed/bert --no-trim-model
```

From Python:

```python
from trimbed import TrimConfig, TrimPipeline

config = TrimConfig.from_yaml("configs/f2llm_dutch.yaml")
report = TrimPipeline(config).run()
print(report.render())
```

[`examples/`](examples/) has a few Python examples: inspecting a
tokenizer, trimming from rules alone, trimming over a corpus with the model,
and registering your own preset.

## Configuration

Everything is focused on composable config files backed by Pydantic.

```yaml
model: codefuse-ai/F2LLM-v2-160M   # Hub id or local path
revision: null
output_dir: trimmed/f2llm-nl
trim_model: true                   # false => tokenizer only
overwrite: false                   # allow writing into a non-empty directory
trust_remote_code: false           # needed by gte, jina and other custom-code checkpoints
verify: true                       # re-encode corpus samples to prove the trim is faithful
verify_samples: 256                # corpus texts sampled; keep_texts are added to them
verify_model: true                 # also run both models and compare their outputs
verify_model_samples: 8            # texts the model comparison runs on; at most verify_samples
verify_tolerance: 1.0e-5           # largest output difference it accepts
copy_sidecar_files: true           # carry sentence-transformers modules into the output
sidecar_patterns:                  # what "sidecar" has to look for
  - modules.json
  - config_sentence_transformers.json
  - sentence_bert_config.json
  - "[0-9]_*/*"
seed: 0

corpus:
  datasets:
    - path: epfml/FineWeb2-HQ
      name: nld_Latn               # dataset config
      split: train
      text_column: text
      streaming: true
      max_samples: 200000
      weight: 1.0                  # multiplier on this corpus' counts
  batch_size: 1000
  num_proc: 8                      # dataset loading workers (non-streaming)
  counts_cache: counts.json        # read if present, else written

selection:
  coverage: 0.9995                 # keep tokens covering this share of occurrences
  top_k: 48000                     # and/or an absolute cap on corpus-derived tokens
  min_count: null                  # and/or a minimum frequency
  max_vocab_size: null             # hard cap, applied last
  keep_presets: [alphanumeric, punctuation, "script:Latin"]
  keep_tokens: ["€"]
  keep_token_ids: []
  keep_token_files: [my_tokens.txt]   # one token per line, '#' comments allowed
  keep_patterns: ['^\d+$']            # regex against each token's decoded surface form
  keep_texts: ["### Instruction:"]    # keep these encoding exactly as they do now
  keep_chat_template: true            # same, for the text the chat template works with

embeddings:
  pad_to_multiple_of: 64           # keep the matrix tensor-core aligned
  dtype: null                      # null preserves the original
  device: cpu
  auto_class: null                 # null will just load the class named in the checkpoint's config
```

You can override config values by extra key-value pairs. Typically useful if you want to
test out different values. E.g., if you want to see what the effect of different
top-k values would be so you run it in a bash script with different `k` values with `dry-run`.

```bash
uv run trimbed trim --config cfg.yaml --top-k 32000 --dry-run
uv run trimbed trim --config cfg.yaml --dry-run \
    selection.top_k=32000 corpus.batch_size=2000 embeddings.dtype=bfloat16
```

### How selection works

```
kept = union(structural, requested, corpus)  ->  close over dependencies  ->  apply cap
```

1. Always kept: structural tokens, i.e. added/special tokens, whatever the post-processor names,
   the unk token, and (for byte-level tokenizers) all 256 byte-alphabet characters
2. Requested: `keep_presets`, `keep_tokens`, `keep_token_ids`, `keep_token_files`,
   `keep_patterns`, `keep_texts` and `keep_chat_template`. These are a floor, not a
   filter: they can only grow the vocabulary by specifying what you want to *keep*.
3. Corpus: the frequency ranking cut by whichever of `coverage`, `top_k` and `min_count`
   are set. Setting several applies the strictest, most narrow view.
4. Dependency closure: see below.
5. Cap: if `max_vocab_size` is exceeded, the least-frequent tokens are dropped until it
   fits. This can remove tokens you explicitly requested so the output report lists them.

Every kept token records why it survived, and the report breaks that down by reason.

#### Keeping a prompt intact

`keep_tokens` needs the exact vocabulary entry, which nobody knows by hand: ` assistant` is
`Ġassistant` in one tokenizer and `▁assistant` in the next, and may not be one token at
all.

So to make it easier for you, `keep_texts` asks the question the other way round (it
encodes the text with the original tokenizer and keeps whatever ids came out),
so the text goes on tokenizing exactly as it does now. It needs no corpus, and it
is verified afterwards:

```yaml
selection:
  keep_texts: ["### Instruction:\n", "### Response:\n"]
```

`keep_chat_template` does the same for the tokenizer's own chat template, and is on by
default. The template's Jinja is stripped to only the literals (such as `<|im_start|>`).

#### Dependency closure

BPE builds `"the"` by merging `"t"` with `"he"`. If we keep `"the"` but drop `"he"`,
`"the"` becomes unreachable. `trimbed` therefore closes the kept set over each
token's merge ancestry, and the size cap only ever removes tokens nothing else depends
on.

### Presets

`trimbed presets` prints them with a line on what each one selects. Highly recommended
to run this to better understand which ones are available and which ones are on by default.

## What the trimmed tokenizer keeps

The trimmed document is round-tripped through `save_pretrained` and
`AutoTokenizer.from_pretrained` so everything outside tokenizer.json survives:
the chat template, `model_max_length`, the special-tokens map and the rest of tokenizer_config.json.
Added tokens keep their id-linked entries in `added_tokens_decoder`, and their matching flags (`single_word`, `lstrip`, `rstrip`,
`normalized`, `special`) are restored from the source, which is what keeps a chat template
tokenizing to exactly the ids it did before.

## Trimming the model

With `trim_model: true` the embedding table, and the output head if there is one, is
trimmed too by only select the relevant rows of the table. We cover tied and untied heads, a head bias (a
masked-LM head has one even when its weights are tied, and tying does not carry it along),
encoder-decoders, and encoders with no head at all. Token ids stored on the config and the
generation config follow the remap, including the ones a multimodal checkpoint keeps on
`config.text_config` rather than at the top level. That said, multimodal models have not been
thoroughly tested so make sure to add a bug report in case you encounter issues.

As is relatively well known and common, `pad_to_multiple_of` rounds the matrix up past
the end of the vocabulary to a value that ensures efficient tensor-core processing.

## Output

The output directory holds a ready-to-load tokenizer (and model, if trimmed), plus:

- `trim_report.json`: sizes, reduction, corpus coverage, per-reason token counts,
  verification results, parameters removed
- `_trimbed_config.yaml`: the fully resolved configuration, for provenance

`report.render()` prints a summary in this shape, with illustrative figures; your corpus
and settings decide the actual numbers:

```
model            codefuse-ai/F2LLM-v2-160M
tokenizer type   BPE
vocabulary       151,669 -> 32,412 (78.6% removed, 282 structural)
corpus           200,000 docs, 91,204,331 tokens, 99.9612% covered by the kept vocabulary
architecture     Qwen3Model
embeddings       151,936 -> 32,448 rows, 122,355,712 of 160,384,000 parameters removed (76.3%)
verification     512/512 identical, 512/512 decode-equivalent, 1.0004x tokens
model check      passed on 8 texts: max |dh| 2.38e-07, max |dlogit| n/a (tolerance 1e-05)
kept by          chat_template=41, corpus=32,004, dependency=126, structural=282
output           trimmed/f2llm-nl
```

## Extending

### A new tokenizer family

We happily rely on `skeletoken` which handles the serialisation. As long as your tokenizer-type
is supported in `skeletoken` and `transformers`, you can add a new tokenizer backend like so:

```python
from trimbed.backends import register_backend
from trimbed.backends.base import VocabBackend

@register_backend
class MyBackend(VocabBackend):
    model_type = "MyModel"          # matches model.type in tokenizer.json

    def structural_tokens(self, spec):
        return super().structural_tokens(spec) | {"<my-required-token>"}
```

### A new preset

More likely, though, is that you want to add custom rules that determine which tokens should be kept.

For instance, registering a preset by name that keeps emojis could work like so:

```python
from trimbed.presets import register_preset

@register_preset("emoji")
def _emoji(spec):
    return {t for t, surface in spec.surface_forms.items() if surface and _is_emoji(surface)}
```
