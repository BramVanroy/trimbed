# trimbed

Trim a tokenizer's vocabulary, and optionally its model's embedding table, down to the
subset you actually need.

Multilingual encoders carry vocabularies covering dozens of languages. If you only ever
run one language, most of that vocabulary is dead weight: on `codefuse-ai/F2LLM-v2-160M`
the 151k-token embedding table is a large fraction of a 160M-parameter model. `trimbed`
derives the subset you need from a corpus, keeps whatever you insist on keeping, and
rewrites both the tokenizer and the model consistently.

## Why not `lm-vocab-trimmer`?

[`lm-vocab-trimmer`](https://github.com/asahi417/lm-vocab-trimmer) edits the SentencePiece
protobuf, so it only handles SentencePiece-backed families (XLM-R, mT5, mBART). It cannot touch a byte-level BPE tokenizer like Qwen's or F2LLM's, which has no
SentencePiece model at all.

`trimbed` works one layer up, on the `tokenizers` backend document (`tokenizer.json`) that
every Hugging Face fast tokenizer serialises to, and that `transformers` produces even for
tokenizers shipped only as a `spiece.model`. One code path therefore covers:

| Family | Example | Supported |
|---|---|---|
| BPE (byte-level) | `codefuse-ai/F2LLM-v2-160M`, Qwen, GPT-2 | yes |
| WordPiece | `bert-base-multilingual-cased` | yes |
| Unigram | `xlm-roberta-base`, `mt5-small` (spiece-only) | yes |
| WordLevel | any word-level checkpoint | yes |

The typed model of that document comes from
[skeletoken](https://github.com/stephantul/skeletoken), which performs the vocabulary
surgery and validates the result.

## Installation

```bash
uv sync                      # tokenizer trimming only
uv sync --extra model        # + torch, for trimming embeddings
uv sync --all-extras         # + sentencepiece/protobuf, for spiece-only tokenizers
```

## Quickstart

Installing the package puts one command on your PATH, `trimbed`, with a subcommand per
job: `trim`, `count`, `inspect` and `presets`. `trimbed --help` lists them. The `uv run`
prefix below is only for a checkout, where it resolves the environment for you; on a
checkout without an install, `python -m trimbed.cli` works too.

```bash
# What am I dealing with?
uv run trimbed inspect --model codefuse-ai/F2LLM-v2-160M

# Trim tokenizer + model from a config file
uv run trimbed trim --config configs/f2llm_dutch.yaml

# Same config, one value changed and nothing written
uv run trimbed trim --config configs/f2llm_dutch.yaml --dry-run \
    selection.top_k=30000

# Or without a config, keeping a preset and nothing else
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

[`examples/`](examples/) has a short runnable script per usage pattern: inspecting a
tokenizer, trimming from must-keep rules alone, trimming over a corpus with the model,
registering your own preset, and driving the stages without the pipeline.

## Configuration

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
sidecar_patterns:                  # what "sidecar" means, if the default is wrong
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
  auto_class: null                 # null => the class named in the checkpoint's config
```

Three layers stack, later winning over earlier: the YAML file, then the typed flags, then
the trailing `key=value` positionals. One config can therefore be reused across experiments
with a value or two changed on the command line:

```bash
uv run trimbed trim --config cfg.yaml --top-k 32000 --dry-run
uv run trimbed trim --config cfg.yaml --dry-run \
    selection.top_k=32000 corpus.batch_size=2000 embeddings.dtype=bfloat16
```

Override values are parsed as YAML scalars, so `32000` is an int, `false` a bool and
`[a, b]` a list, while a bare word stays a string, so no shell quoting is needed.

Unknown YAML keys are rejected rather than ignored, so typos fail loudly. The same
applies to a mistyped override path: `selection.tpo_k=5` is an error, not a no-op.

### How selection works

```
kept = union(structural, requested, corpus)  ->  close over dependencies  ->  apply cap
```

1. Structural: added/special tokens, whatever the post-processor names, the unk token,
   and (for byte-level tokenizers) all 256 byte-alphabet characters. Always kept;
   dropping them breaks encoding outright.
2. Requested: `keep_presets`, `keep_tokens`, `keep_token_ids`, `keep_token_files`,
   `keep_patterns`, `keep_texts` and `keep_chat_template`. These are a floor, not a
   filter: they can only grow the vocabulary.
3. Corpus: the frequency ranking cut by whichever of `coverage`, `top_k` and `min_count`
   are set. Setting several applies the strictest.
4. Dependency closure: see below.
5. Cap: if `max_vocab_size` is exceeded, the least-frequent tokens are dropped until it
   fits. This can remove tokens you explicitly requested; the report lists them.

Every kept token records why it survived, and the report breaks that down by reason.

#### Keeping a prompt intact

`keep_tokens` needs the vocabulary entry, which nobody knows by hand: ` assistant` is
`Ġassistant` in one tokenizer and `▁assistant` in the next, and may not be one token at
all. `keep_texts` asks the question the other way round (it encodes the text with the
original tokenizer and keeps whatever ids came out), so the text goes on tokenizing
exactly as it does now. It needs no corpus, and it is verified afterwards:

```yaml
selection:
  keep_texts: ["### Instruction:\n", "### Response:\n"]
```

The text is encoded with `add_special_tokens=False`, so what is kept is what the text
itself is made of, not the `[CLS]`/`[SEP]` pair a post-processor would wrap around it --
those are structural and kept regardless. A special token written out inside the text is
still matched as one, since that is genuinely what the text encodes to.

`keep_chat_template` does the same for the tokenizer's own chat template, and is on by
default. The template's Jinja is stripped rather than rendered, so nothing has to be
invented and no template can raise: what is kept is the literal text it emits (including
branches a single rendered conversation would never reach), the strings quoted inside its
markup, and the standard role names, which a ChatML-style template substitutes from the
message and therefore never names.

This matters more than it looks. Nothing breaks without it, since the template still
renders and the special tokens are structural anyway, but the words around them fragment
into characters, so every SFT or RL example silently pays extra tokens on boilerplate
and the prompt no longer tokenizes to the ids it trained on.

#### Dependency closure

BPE builds `"the"` by merging `"t"` with `"he"`. Keep `"the"` but drop `"he"` and the
token stays in the vocabulary while becoming unreachable: text silently fragments into
`t`, `h`, `e` instead of failing. `trimbed` therefore closes the kept set over each
token's merge ancestry, and the size cap only ever removes tokens nothing else depends
on. WordPiece, Unigram and WordLevel compose no tokens from others, so they need none of
this and declare no dependencies.

### Presets

`uv run trimbed presets` prints them with a line on what each one selects.
An installed copy of the package has the same listing under `trimbed presets`.

`structural`, and its parts `added_tokens`, `special_tokens`, `unk` and `byte_alphabet`,
resolve to tokens the trim keeps whether or not you name them. Naming one is still how a
run without a corpus gets a must-keep source, which is why a bare `--model` falls back to
`structural`.

The opt-in ones are `single_characters`, `ascii_letters`, `digits`, `alphanumeric`,
`punctuation`, `whitespace`, `ascii_printable`, and `script:<Name>` (e.g. `script:Latin`,
`script:Cyrillic`).

Presets match on each token's decoded surface form, so `Ġthe` (byte-level for `" the"`) and
`##ing` (WordPiece for `"ing"`) match the same way their plain text would.

## What the trimmed tokenizer keeps

The trimmed document is round-tripped through `save_pretrained` and
`AutoTokenizer.from_pretrained` rather than rebuilt from scratch, so everything outside
tokenizer.json survives: the chat template, `model_max_length`, the special-tokens map and
the rest of tokenizer_config.json. Added tokens keep their id-linked entries in
`added_tokens_decoder`, and their matching flags (`single_word`, `lstrip`, `rstrip`,
`normalized`, `special`) are restored from the source, which is what keeps a chat template
tokenizing to exactly the ids it did before. `keep_chat_template` covers the other half of
that: the flags keep the control tokens matching, the selection keeps the words between
them from fragmenting.

## Trimming the model

With `trim_model: true` the embedding table, and the output head if there is one, is
gathered down to the surviving rows. That covers tied and untied heads, a head bias (a
masked-LM head has one even when its weights are tied, and tying does not carry it along),
encoder-decoders, and encoders with no head at all. Token ids stored on the config and the
generation config follow the remap, including the ones a multimodal checkpoint keeps on
`config.text_config` rather than at the top level.

`pad_to_multiple_of` rounds the matrix up past the end of the vocabulary, and transformers
fills the rows it adds from the mean of the existing embeddings, which leaves `generate`
able to emit an id the tokenizer cannot decode. Those rows, and the matching head rows and
bias entries, are zeroed, so their logits sit at the bias rather than among the real ones.

Three more things are worth knowing.

### The checkpoint is loaded as the class it says it is

`AutoModel` returns the base model, so a `...ForCausalLM` or `...ForMaskedLM` checkpoint
would load without its head: for an untied head that silently discards trained weights, and
the saved config would advertise the demoted architecture. `trimbed` therefore reads
`config.architectures` and loads that class, falling back to `AutoModel` only for
remote-code checkpoints whose class transformers does not export. Override it with
`embeddings.auto_class` when the checkpoint is wrong or you deliberately want the bare
encoder.

### Verification comes in two strengths

`verify` re-encodes sample texts and proves the vocabulary was renumbered faithfully. The
samples are `selection.keep_texts` followed by a reservoir sample of the corpus, so a trim
driven by must-keep rules alone still gets checked, and the texts you asked to keep
encodable are the first thing proven and the first thing the model comparison sees. It
cannot see whether the embedding rows followed that renumbering, because it never runs the
model. `verify_model` does: it reloads the original and the just-written trimmed model,
runs both on a handful of samples and compares last hidden states plus, when there is an
output head, its logits gathered through the remap. That is the end-to-end proof that the
gather index, the head and the remapped config ids agree. An encoder-decoder is given one
decoder step at its own `decoder_start_token_id`, since it will not run on `input_ids`
alone and its head would otherwise stay out of the comparison. It costs two more model
loads; turn it off with `--no-verify-model` when that is too expensive.

Note what neither check covers: they prove the trimmed model computes the same thing as the
original for text made of kept tokens. They say nothing about how well the model does on
text it now tokenizes into more pieces, and nothing about representation quality if you
trim away a language the model was relying on.

### Sidecar files are carried over

`save_pretrained` writes weights, configs and the tokenizer, and nothing else, so a
sentence-transformers checkpoint would lose `modules.json` and `1_Pooling/`, and
`SentenceTransformer` would silently reopen the result with default mean pooling. Files
matching `sidecar_patterns` are copied from the source repository (Hub or local); nothing
the trim itself wrote is ever overwritten.

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

skeletoken handles the serialisation; a backend only declares the two things a trimmer has
to know: which tokens are load-bearing, and which tokens depend on which.

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

Registering one makes it referenceable from YAML by name:

```python
from trimbed.presets import register_preset

@register_preset("emoji")
def _emoji(spec):
    return {t for t, surface in spec.surface_forms.items() if surface and _is_emoji(surface)}
```

## Architecture

```
configs/                   YAML configs for the commands; each opens with a header comment
  minimal.yaml             the smallest config that does something useful
  f2llm_dutch.yaml         the worked example: a 151k multilingual vocab down to Dutch
src/trimbed/
  cli/                     the command surface; each subcommand is one self-contained module
    __main__.py            the `trimbed` command: the subcommand table and the routing
    trim_vocab.py          trimbed trim: count -> select -> trim -> verify -> report
    count_tokens.py        trimbed count: corpus counting alone, cached to JSON for reuse
    inspect_tokenizer.py   trimbed inspect: describe a tokenizer, change nothing
    list_presets.py        trimbed presets: print the must-keep preset registry
  config.py                pydantic config models, YAML loading, key=value overrides
  spec.py                  TokenizerSpec: a skeletoken TokenizerModel + surface forms
  backends/                per-family selection constraints (structural tokens, dependencies)
  presets.py               named must-keep token sets, extensible via @register_preset
  counting.py              CorpusCounter over HF datasets; cacheable CorpusCounts
  selection.py             the union / closure / cap policy, with per-token provenance
  remap.py                 old id <-> new id, and the embedding gather index
  tokenizer_trim.py        skeletoken's surgery plus a save/reload round trip
  model_trim.py            embedding + lm_head surgery (lazy torch import)
  loading.py               tokenizer/model loading, including which class to load as
  sidecar.py               copying the source repo's vocabulary-independent files
  verify.py                re-encode samples, and run both models, to prove the trim
  report.py                the report models the stages return, and render()
  pipeline.py              TrimPipeline.run() ties the stages together
examples/                  one short runnable script per library-API usage pattern
tests/                     offline by default; conftest.py builds tiny tokenizers in-process
```

Errors propagate. `cli/` catches nothing and calls no `sys.exit`, so a failure is a
traceback naming what went wrong. Those are built-in exceptions: `ValueError` for input
or configuration that cannot be used, `KeyError` for an unknown preset or tokenizer
family, `FileExistsError` for a non-empty output directory, and `RuntimeError` for a
surgery invariant that did not hold. The one type trimbed defines is
`MissingDependencyError`, an `ImportError` naming the extra to install.

## Development

```bash
make style      # ruff check --fix + format
make quality    # non-mutating; the CI entrypoint
make test       # the offline suite, with statement + branch coverage
```

The suite runs offline: fixtures build tiny BPE/WordPiece/Unigram/WordLevel tokenizers and
models in-process, every script in `examples/` is executed against them, and every
subcommand is driven end to end through the real `trimbed` router. Coverage of
`src/trimbed` is 100% of both statements and branches. The Hub tests are excluded by
default and have to be asked for, and the ones that download a checkpoint and run a forward pass over it are
`slow` on top of that:

```bash
make test-network                     # the Hub tokenizer trims
make test-slow                        # + real weights, minutes each
make test-all                         # everything
uv run pytest -m "not torch"          # skip everything needing torch
```

The Hub tests need `--all-extras`, which those targets pass: one of them trims mT5, which
ships `spiece.model` with no `tokenizer.json`, so converting it needs `trimbed[convert]`.

### CI

`.gitlab-ci.yml` runs `make quality` and then `make test` on Python 3.12 and 3.13 for
every merge request and every push to the default branch, publishing the coverage and
JUnit reports back into the merge request. The Hub tests are deliberately off that path:
`test:hub` and `test:hub-models` run on a pipeline schedule (add a nightly under
Build > Pipeline schedules) and are otherwise `when: manual`, so they can be started from
any pipeline without blocking it.
