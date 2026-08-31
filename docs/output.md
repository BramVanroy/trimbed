# Output and reports

The output directory holds a ready-to-load tokenizer (and model, if trimmed), plus two
files that record the run:

- `trim_report.json` — sizes, reduction, corpus coverage, per-reason token counts,
  verification results, parameters removed. This is a serialised
  [`TrimReport`][trimbed.report.TrimReport].
- `_trimbed_config.yaml` — the fully resolved configuration, for provenance.

[`TrimPipeline.run`][trimbed.pipeline.TrimPipeline.run] returns the same
[`TrimReport`][trimbed.report.TrimReport] it writes, so a caller never has to read the
file back.

## The rendered summary

[`TrimReport.render`][trimbed.report.TrimReport.render] prints a summary in this shape.
The figures below are illustrative; your corpus and settings decide the actual numbers.

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

```python
from trimbed import TrimConfig, TrimPipeline

report = TrimPipeline(TrimConfig.from_yaml("my_config.yaml")).run()
print(report.render())
print(report.vocabulary.trimmed_size)
```

## Reading the sections

`vocabulary` is a [`VocabularyReport`][trimbed.report.VocabularyReport]. Its
`kept_by_reason` is the provenance breakdown from
[`Selection`][trimbed.selection.Selection]: an id kept for two reasons is counted under
both, so the numbers do not sum to the vocabulary size. Two fields are worth checking on
every run:

- `unknown_requested_tokens` — token strings you asked for that this vocabulary does not
  contain. Usually a keep-list written for a different checkpoint.
- `dropped_requested_tokens` — tokens that were requested but that `max_vocab_size` had to
  remove anyway, mapped to the reasons they had been requested.

`corpus` is a [`CorpusReport`][trimbed.report.CorpusReport], where `distinct_tokens` is the
ceiling on what a corpus-only selection can keep, and `coverage` is the share of corpus
occurrences the kept vocabulary still covers.

`model` is a [`ModelReport`][trimbed.report.ModelReport], present only when `trim_model` was
on. `verification` and `model_verification` are the two report models described in
[Trimming a model](trimming-a-model.md).

## Dry runs

`--dry-run` (or `dry_run` on the pipeline) runs the whole selection and produces the same
report without writing anything. That is what makes a sweep over `selection.top_k` values
cheap. See [Configuration](configuration.md#overriding-values).
