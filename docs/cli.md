# Command line

Installing the package gives you the `trimbed` command. From a checkout without an
install, `python -m trimbed.cli` runs the same router.

```bash
trimbed --help
```

Each subcommand owns its own arguments, declared next to the `run` they feed, so
`trimbed <command> --help` is the authoritative list. Four flags are shared by all of
them: `-c/--config`, `-m/--model`, `-v/--verbose` and `-q/--quiet`, plus trailing
`key=value` positionals.

## `trimbed trim`

Trim a tokenizer, and optionally its model, and write the result.
Implemented by [`trimbed.cli.trim_vocab.run`][trimbed.cli.trim_vocab.run].

```bash
trimbed trim --config my_config.yaml

trimbed trim --config my_config.yaml --dry-run \
    selection.top_k=30000 corpus.batch_size=2000
```

A config is not required: `--model` plus must-keep rules is enough to trim without a
corpus, which is the fastest way to see what the machinery does.

```bash
trimbed trim --model google-bert/bert-base-multilingual-cased \
    --keep-preset alphanumeric --output-dir trimmed/bert --no-trim-model
```

`--dry-run` runs the whole selection and reports on it without writing anything, which is
what makes a sweep over `selection.top_k` values cheap.

## `trimbed count`

Count the corpus once and cache the frequencies to JSON.
Implemented by [`trimbed.cli.count_tokens.run`][trimbed.cli.count_tokens.run].

```bash
trimbed count --config my_config.yaml -o counts.json

trimbed count --config my_config.yaml -o counts.json \
    corpus.batch_size=4000 corpus.num_proc=8
```

Counting is the expensive part of a trimming run and it does not depend on the selection
policy. Do it once here, then point `corpus.counts_cache` at the output to reuse it
across selection experiments. Only the corpus half of the config matters: the selection
and embedding sections are still validated, but unused.

The written file is a serialised [`CorpusCounts`][trimbed.counting.CorpusCounts].

## `trimbed inspect`

Describe a tokenizer as JSON, changing nothing.
Implemented by [`trimbed.cli.inspect_tokenizer.run`][trimbed.cli.inspect_tokenizer.run].

```bash
trimbed inspect --model codefuse-ai/F2LLM-v2-160M
trimbed inspect --config my_config.yaml
```

Run this before a trimming job. It reports the backend family (BPE, WordPiece, Unigram,
WordLevel), the vocabulary size, how many tokens are added or special (those are
structural and never removed), whether it ships a chat template, and whether a backend is
registered for the family at all. The output is
[`TokenizerSpec.describe`][trimbed.spec.TokenizerSpec.describe] plus a `supported` flag.

Nothing is loaded beyond the tokenizer and nothing is written. The model weights are never
touched.

## `trimbed presets`

List the registered presets that `--keep-preset` and `selection.keep_presets` accept.
Implemented by [`trimbed.cli.list_presets.run`][trimbed.cli.list_presets.run].

```bash
trimbed presets
```

The structural presets are printed apart from the rest, because the trim keeps those
tokens whether or not you name them. Naming one is still how a run without a corpus gets
a must-keep source, which is what `trimbed trim --model ...` does by default.

The registry is user-extensible, so this reflects whatever
[`register_preset`][trimbed.presets.register_preset] has been applied by the time it runs.
See [Extending trimbed](extending.md). For a program that wants the same text,
[`render_presets`][trimbed.presets.render_presets] returns it.

## Adding a subcommand

`src/trimbed/cli/` is the whole command surface, one module per job.
[`trimbed.cli.__main__`][trimbed.cli.__main__] holds the `COMMANDS` table and nothing
else: the name, the module, and the one line `trimbed --help` prints for it. Adding a
command is a new module plus one line in that table, so reading a command still means
reading one file.
