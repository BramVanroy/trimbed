# Command line

Installing the package gives you the `trimbed` command. From a checkout without an
install, `python -m trimbed.cli` runs the same router.

```bash
trimbed --help
```

Each subcommand owns its own arguments, declared next to the `run` they feed, so
`trimbed <command> --help` is the authoritative list. Four flags are shared by the
config-driven ones: `-c/--config`, `-m/--model`, `-v/--verbose` and `-q/--quiet`, plus
trailing `key=value` positionals. `trimbed compare` is the odd one out: it takes two
tokenizers as positional arguments and reads no config at all.

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

## `trimbed compare`

Diff two tokenizers, e.g. a checkpoint and a trimmed version of it.
Implemented by [`trimbed.cli.compare_tokenizers.run`][trimbed.cli.compare_tokenizers.run].

```bash
trimbed compare intfloat/multilingual-e5-small clips/e5-small-trm-nl

trimbed compare base/ trimmed/ --preset script:Latin \
    --text-file dutch.txt -o diff.json
```

A model card says which corpus a trim was made from and what size it targeted, which does
not tell you what it left behind. This does, reading nothing but the two `tokenizer.json`
documents:

```
base             intfloat/multilingual-e5-small
other            clips/e5-small-trm-nl
type             Unigram -> Unigram (byte-level no -> no)
unk token        <unk> -> <unk>
added tokens     5 -> 5 (5 -> 5 special)
chat template    absent
vocabulary       250,002 -> 50,002 (80.0% removed, 50,002 shared)
relation         subset, contiguous ids, original order
presets kept     added_tokens, special_tokens, structural, unk, whitespace
presets cut      alphanumeric 42,990/81,320 (52.9%), digits 1,273/1,274 (99.9%), ...
scripts          LATIN 44,795/110,090 (40.7%), CYRILLIC 585/31,670 (1.8%), CJK 384/17,134 (2.2%), ...
categories       letter 46,160/237,488 (19.4%), number 2,429/3,167 (76.7%), ...
removed by id    ▅▆▆▇▇▇▇███  (deciles of the base ids, lowest first)
first removed    ، (50), । (125), ▁të (134), ▁کے (216), ▁از (270)
encoding         2 texts, 2 split identically, 1.0000x tokens
```

Everything is counted against the first tokenizer, so `CYRILLIC 585/31,670` means the base
had 31,670 Cyrillic tokens and 585 of them survive.

Three lines carry most of the weight:

`relation` says whether this is a trim at all. A trim is a subset of the original,
renumbered contiguously from zero in the original order, which is what
[`IdRemap`][trimbed.remap.IdRemap] produces. `reordered` or `not a subset` means the
vocabulary was rebuilt, and an embedding table gathered against the original would be
wrong.

`structural loss` only appears when something load-bearing is missing: an added token, a
special token, or a token the base's post-processor names. Any of those is a broken
tokenizer rather than a smaller one, so the command also logs a warning.

`removed by id` is a histogram of the removed tokens over ten equal slices of the base id
range. A vocabulary is roughly frequency-ordered, so a trim that only gave up the tail
leaves the early deciles near empty. Bars on the left mean it cut into tokens the base
considered common, which is what the example above shows for a multilingual base narrowed
to one language.

Sample texts are optional. With `--text` or `--text-file` the report adds how many of them
both tokenizers split the same way and what the second one costs in tokens, which is the
number that matters if you are choosing between two trims. `--preset` adds a preset to the
report, and is the only way to reach the parametrised ones, e.g. `--preset script:Cyrillic`.
With `-o` the full report is written as JSON: everything the rendering summarises, plus the
per-preset, per-script and per-category counts in full.

The library call behind it is [`compare_tokenizers`][trimbed.compare.compare_tokenizers],
which returns a [`ComparisonReport`][trimbed.compare.ComparisonReport].

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
