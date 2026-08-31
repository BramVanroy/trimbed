# Trimming a model

## What the trimmed tokenizer keeps

[`trim_tokenizer`][trimbed.tokenizer_trim.trim_tokenizer] drives the vocabulary surgery
and then round-trips the result through `save_pretrained` and
`AutoTokenizer.from_pretrained`, so everything outside `tokenizer.json` survives: the chat
template, `model_max_length`, the special-tokens map and the rest of
`tokenizer_config.json`.

Added tokens keep their id-linked entries in `added_tokens_decoder`, and their matching
flags (`single_word`, `lstrip`, `rstrip`, `normalized`, `special`) are preserved, which is
what keeps a chat template tokenizing to exactly the ids it did before.

Added tokens are ordinary vocabulary entries here: skeletoken folds them in on load, so
there is no separate id space above the base vocab and no Qwen-style `vocab_size` versus
`len()` trap.

## The embedding table and the head

With `trim_model: true`, [`trim_model`][trimbed.model_trim.trim_model] trims the embedding
table and the output head if there is one, by selecting the relevant rows of the table.
The row order comes from an [`IdRemap`][trimbed.remap.IdRemap], built from the two
vocabularies and matched on token strings, which is what keeps every added token's trained
embedding row.

Covered shapes: tied and untied heads, a head bias (a masked-LM head has one even when its
weights are tied, and tying does not carry it along), encoder-decoders, and encoders with
no head at all. Token ids stored on the config and the generation config follow the remap,
including the ones a multimodal checkpoint keeps on `config.text_config` rather than at the
top level.

!!! note
    Multimodal models have not been thoroughly tested. Please open an issue if you run
    into problems.

[`resolve_model_class`][trimbed.loading.resolve_model_class] picks the class to load with,
reading the checkpoint's own `architectures` rather than reaching for `AutoModel`. That
matters: `AutoModel` returns the *base* model, so a `...ForCausalLM` or `...ForMaskedLM`
checkpoint would lose its head, silently discarding trained weights when the head is untied
and demoting `config.architectures` on save.

### Alignment padding

As is relatively well known and common, `pad_to_multiple_of` rounds the matrix up past the
end of the vocabulary to a value that keeps tensor cores efficient. Those extra rows are
reachable output: `transformers` fills them from the mean of the existing embeddings, so
their logits compete with the real ones and `generate` can emit an id past the end of the
vocabulary. trimbed zeroes them, and the head rows and bias entries with them.

## Sidecar files

sentence-transformers keeps its pooling and dense modules in separate files that
`save_pretrained` knows nothing about. Without them, `SentenceTransformer` reopens the
output with default mean pooling and no error at all.
[`copy_sidecar_files`][trimbed.sidecar.copy_sidecar_files] carries those
vocabulary-independent files across, matching
[`DEFAULT_SIDECAR_PATTERNS`][trimbed.sidecar.DEFAULT_SIDECAR_PATTERNS] unless
`sidecar_patterns` says otherwise. Nothing the trim itself writes is ever copied.

## Verification

A trim that quietly changes how text tokenizes is worse than one that fails, so the
pipeline proves its work.

[`verify_tokenizer`][trimbed.verify.verify_tokenizer] re-encodes sampled corpus texts with
both tokenizers and reports how many came out identical, how many decode to the same
string, and the token-count ratio. A run with no corpus verifies against its `keep_texts`
and chat-template literals instead.

[`verify_model`][trimbed.verify.verify_model] goes further and runs both models on the same
texts, comparing hidden states and logits against `verify_tolerance`. For an encoder-decoder
it primes one decoder step at the model's own `decoder_start_token_id`, which is also what
puts the trimmed output head into the comparison.

Both return the report models the run serialises:
[`VerificationReport`][trimbed.report.VerificationReport] and
[`ModelVerificationReport`][trimbed.report.ModelVerificationReport].
