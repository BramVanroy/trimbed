# Configuration

Everything is focused on composable config files backed by pydantic. The whole run is one
[`TrimConfig`][trimbed.config.TrimConfig], which nests
[`CorpusConfig`][trimbed.config.CorpusConfig],
[`SelectionConfig`][trimbed.config.SelectionConfig] and
[`EmbeddingTrimConfig`][trimbed.config.EmbeddingTrimConfig]. Unknown keys are rejected, so
a typo is an error rather than a silently ignored setting.

## A complete config

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
      data_dir: null               # for a loader name, see "Corpora on disk" below
      data_files: null
      load_from_disk: false        # true for a directory written by save_to_disk
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

Every field, with its own description, is in the
[Config API reference](api/config.md). Each dataset entry is a
[`DatasetSpec`][trimbed.config.DatasetSpec]. A leading `~` is expanded in every path,
including `model`, so `output_dir: ~/trimmed` lands in your home directory rather than in
a directory literally named `~`.

## Corpora on disk

`path` is the first argument `datasets.load_dataset` takes, so a corpus that never went to
the Hub is spelled the same way it would be there. Four shapes cover everything:

```yaml
corpus:
  datasets:
    # A Hub dataset.
    - path: epfml/FineWeb2-HQ
      name: nld_Latn
    # A directory of data files, with the loader inferred from the extensions.
    - path: ./data/dutch
    # A loader named explicitly, reading the files you point it at. Globs are allowed,
    # and this is the spelling to use for one file: `path` itself cannot be a file.
    - path: json
      data_files: ./data/dutch/*.jsonl
    # A dataset written by `save_to_disk`, which `load_dataset` cannot read.
    - path: ./data/prepared
      load_from_disk: true
```

`data_dir` is the alternative to `data_files`: it hands the loader a whole directory
(`path: json` with `data_dir: ./data/dutch`) instead of a file list.

A `save_to_disk` dataset is already built, so it is never streamed, and it takes no
`name`, `revision`, `data_dir` or `data_files`: there is nothing left to resolve.
`split` still applies when the directory holds a `DatasetDict`.

The `model` side needs nothing special. A local checkpoint is the directory holding
`config.json` and the tokenizer files, exactly what `from_pretrained` takes, and
`sidecar_patterns` are matched against that directory rather than against a Hub listing.
`revision` only means something for a Hub id, so pinning one next to a local path is an
error rather than a setting that quietly does nothing.

## Overriding values

Three layers stack, later winning over earlier: the YAML file, the typed flags, and the
trailing `key=value` positionals. The typed flags cover the knobs worth tuning between
runs, and the positionals reach every remaining field without needing a flag for each one.

```bash
trimbed trim --config cfg.yaml --top-k 32000 --dry-run
trimbed trim --config cfg.yaml --dry-run \
    selection.top_k=32000 corpus.batch_size=2000 embeddings.dtype=bfloat16
```

This is what makes a sweep cheap: run the same config in a bash loop with a different `k`
each time and `--dry-run`, and compare the reports.

From Python the same precedence is one expression, built from
[`load_config`][trimbed.config.load_config],
[`parse_overrides`][trimbed.config.parse_overrides] and
[`TrimConfig.with_overrides`][trimbed.config.TrimConfig.with_overrides]:

```python
from trimbed import load_config, parse_overrides

config = load_config("cfg.yaml", "codefuse-ai/F2LLM-v2-160M").with_overrides(
    {**flag_overrides, **parse_overrides(["selection.top_k=32000"])}
)
```

`with_overrides` drops `None`, so a flag that was not passed changes nothing. That is also
why a boolean flag whose config field defaults to true is spelled `--no-...` and maps to
`False`, while one defaulting to false is spelled plainly and maps to `True`.

The fully resolved configuration is written to the output directory as
`_trimbed_config.yaml`, so a finished run records exactly what produced it. See
[Output and reports](output.md).
