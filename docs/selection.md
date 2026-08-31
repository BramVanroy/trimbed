# How selection works

Selection decides which token ids survive. It is implemented by
[`select_tokens`][trimbed.selection.select_tokens], which returns a
[`Selection`][trimbed.selection.Selection] recording not just what was kept but why.

```
kept = union(structural, requested, corpus)  ->  close over dependencies  ->  apply cap
```

1. **Always kept**: structural tokens, i.e. added/special tokens, whatever the
   post-processor names, the unk token, and (for byte-level tokenizers) all 256
   byte-alphabet characters. These are
   [`TokenizerSpec.structural_ids`][trimbed.spec.TokenizerSpec.structural_ids].
2. **Requested**: `keep_presets`, `keep_tokens`, `keep_token_ids`, `keep_token_files`,
   `keep_patterns`, `keep_texts` and `keep_chat_template`. These are a floor, not a
   filter: they can only grow the vocabulary by specifying what you want to *keep*.
3. **Corpus**: the frequency ranking cut by whichever of `coverage`, `top_k` and
   `min_count` are set. Setting several applies the strictest, most narrow view.
4. **Dependency closure**: see below.
5. **Cap**: if `max_vocab_size` is exceeded, the least-frequent tokens are dropped until
   it fits. This can remove tokens you explicitly requested, so the report lists them
   under `dropped_requested_tokens`.

Every kept token records why it survived in
[`Selection.provenance`][trimbed.selection.Selection], and
[`Selection.counts_by_reason`][trimbed.selection.Selection.counts_by_reason] is what the
report breaks down.

## Keeping a prompt intact

`keep_tokens` needs the exact vocabulary entry, which nobody knows by hand: ` assistant`
is `Ġassistant` in one tokenizer and `▁assistant` in the next, and may not be one token at
all.

So to make it easier for you, `keep_texts` asks the question the other way round. It
encodes the text with the original tokenizer and keeps whatever ids came out, so the text
goes on tokenizing exactly as it does now. It needs no corpus, and it is verified
afterwards:

```yaml
selection:
  keep_texts: ["### Instruction:\n", "### Response:\n"]
```

`keep_chat_template` does the same for the tokenizer's own chat template, and is on by
default. This matters more than it looks. `<|im_start|>` is an added token and therefore
structural, but `assistant`, `system` and the rest of the words the template puts around
the message content are ordinary vocabulary entries that a corpus of plain prose never
uses. Drop them and nothing fails: the template still renders, and every prompt just
quietly fragments into characters, so SFT pays extra tokens on boilerplate and no longer
matches the ids the model trained on.

Rather than rendering the template (which would mean inventing messages, and a template
that can raise), trimbed strips the Jinja and keeps the literals, the strings quoted
inside the markup, and the chat roles a ChatML template substitutes from the message and
never names.

Both `keep_texts` and `keep_chat_template` encode with `add_special_tokens=False`: the ids
wanted are the ones the text is made of, and a post-processor's `[CLS]`/`[SEP]` are
structural anyway.

## Dependency closure

BPE builds `"the"` by merging `"t"` with `"he"`. If we keep `"the"` but drop `"he"`,
`"the"` becomes unreachable and text silently fragments — a quality bug with no error
message. trimbed therefore closes the kept set over each token's merge ancestry, using
[`BpeBackend.dependencies`][trimbed.backends.bpe.BpeBackend.dependencies], and the size cap
only ever removes tokens nothing else depends on.

## Presets

A preset is a named rule for tokens that must survive the trim regardless of what the
corpus says: the byte alphabet, punctuation, a whole Unicode script. Names ending in `:`
are parametrised — supply the argument after the colon, as in `script:Latin`.

```bash
trimbed presets
```

Highly recommended to run this to better understand which ones are available and which
ones are on by default. [`available_presets`][trimbed.presets.available_presets] and
[`resolve_preset`][trimbed.presets.resolve_preset] are the programmatic equivalents, and
[Extending trimbed](extending.md) covers registering your own.

## Counting a corpus

Corpus frequencies come from [`CorpusCounter`][trimbed.counting.CorpusCounter], which
streams the configured datasets and produces
[`CorpusCounts`][trimbed.counting.CorpusCounts]. Counting is the expensive part of a run
and independent of the selection policy, so cache it once with `trimbed count` and point
`corpus.counts_cache` at the result. See [Command line](cli.md).
