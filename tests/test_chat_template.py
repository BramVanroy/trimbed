"""The chat template, and the tokens it needs, surviving a trim.

`trim_tokenizer` round-trips through `save_pretrained` rather than rebuilding the
tokenizer, which is what carries the template over at all; `keep_chat_template` is what
stops a corpus-driven selection from dropping the tokens the template's own words are
made of. Both used to be covered only by the Hub tests, which do not run by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trimbed.config import SelectionConfig
from trimbed.counting import CorpusCounts
from trimbed.selection import select_tokens
from trimbed.spec import TokenizerSpec
from trimbed.tokenizer_trim import trim_tokenizer


# Qwen/Qwen3-0.6B's template, copied verbatim out of its tokenizer_config.json. The
# fixture template is a plain ChatML one; this is what a shipped template really looks
# like, with tool calls and reasoning behind branches no ordinary conversation reaches.
QWEN3_CHAT_TEMPLATE = (Path(__file__).resolve().parent / "qwen3_chat_template.jinja").read_text()


def _spec(tokenizer):
    return TokenizerSpec.from_tokenizer(tokenizer, source="chat")


def _counts_over(tokenizer, texts):
    counts = CorpusCounts()
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        counts.counts.update(ids)
        counts.total_num_tokens += len(ids)
        counts.num_documents += 1
    return counts


def test_the_template_is_captured_on_the_spec(chat_bpe):
    assert _spec(chat_bpe).chat_template == chat_bpe.chat_template


def test_a_spec_without_a_template_has_no_literals(byte_level_bpe):
    spec = _spec(byte_level_bpe)

    assert spec.chat_template is None
    assert spec.chat_template_literals == ""
    assert spec.describe()["has_chat_template"] is False


def test_the_summary_reports_a_template(chat_bpe):
    assert _spec(chat_bpe).describe()["has_chat_template"] is True


def test_the_literals_cover_text_and_quoted_role_names(chat_bpe):
    literals = _spec(chat_bpe).chat_template_literals

    # Emitted as text, and only ever compared against inside `{% ... %}` respectively.
    assert "assistant" in literals
    assert "system" in literals
    # The Jinja itself is gone, expressions and all.
    assert "{%" not in literals
    assert "message" not in literals


@pytest.mark.parametrize(
    "literal",
    [
        # Control tokens, and the role names the template writes around the content.
        "<|im_start|>",
        "<|im_end|>",
        "system",
        "user",
        "assistant",
        # Reachable only through the tool-call branches, which no plain chat renders.
        "<tools>",
        "</tools>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "arguments",
        # The reasoning markers, and the prose of the system prompt the template injects.
        "<think>",
        "</think>",
        "# Tools",
        "You may call one or more functions to assist with the user query.",
    ],
)
def test_a_shipped_template_yields_the_literals_it_surrounds_content_with(chat_bpe, literal):
    chat_bpe.chat_template = QWEN3_CHAT_TEMPLATE

    assert literal in _spec(chat_bpe).chat_template_literals


@pytest.mark.parametrize("expression", ["{%", "{{", "endfor", "messages", "namespace", "tojson", "loop.index0"])
def test_a_shipped_templates_jinja_is_left_out(chat_bpe, expression):
    chat_bpe.chat_template = QWEN3_CHAT_TEMPLATE

    assert expression not in _spec(chat_bpe).chat_template_literals


def test_several_named_templates_are_all_considered(chat_bpe):
    chat_bpe.chat_template = {"default": "<|a|>tool_call", "rag": "<|b|>citation"}
    literals = _spec(chat_bpe).chat_template_literals

    assert "tool_call" in literals
    assert "citation" in literals


def test_keeping_the_template_saves_the_tokens_its_words_are_made_of(chat_bpe):
    spec = _spec(chat_bpe)
    counts = _counts_over(chat_bpe, ["the cat sat", "the dog"])
    assistant = spec.vocabulary["assistant"]

    without = select_tokens(spec, counts, SelectionConfig(min_count=1, keep_chat_template=False))
    with_template = select_tokens(spec, counts, SelectionConfig(min_count=1))

    assert assistant not in without.kept_ids
    assert assistant in with_template.kept_ids
    assert "chat_template" in with_template.provenance[assistant]


def test_encoding_leaves_the_post_processors_tokens_out(wordpiece):
    spec = _spec(wordpiece)

    ids = spec.encode("the cat")

    # `[CLS] the cat [SEP]` is what a plain call would give. Those two are structural and
    # kept regardless; recording them here would only credit them to the text.
    assert [spec.id_to_token[token_id] for token_id in ids] == ["the", "cat"]


def test_keep_texts_holds_a_prompt_to_the_ids_it_has_now(chat_bpe):
    spec = _spec(chat_bpe)
    counts = _counts_over(chat_bpe, ["the cat sat"])
    prompt = "the user said"

    selection = select_tokens(spec, counts, SelectionConfig(min_count=1, keep_texts=[prompt]))
    trimmed = trim_tokenizer(chat_bpe, spec, selection.kept_ids)

    assert spec.vocabulary["user"] in selection.kept_ids
    old_ids = chat_bpe(prompt, add_special_tokens=False)["input_ids"]
    assert trimmed.remap.map_sequence(old_ids) == trimmed.tokenizer(prompt, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("keep_template", [True, False])
def test_the_template_renders_the_same_conversation_after_a_trim(chat_bpe, chat_messages, keep_template):
    spec = _spec(chat_bpe)
    counts = _counts_over(chat_bpe, ["the cat sat", "the dog"])
    selection = select_tokens(spec, counts, SelectionConfig(min_count=1, keep_chat_template=keep_template))

    trimmed = trim_tokenizer(chat_bpe, spec, selection.kept_ids)

    assert trimmed.tokenizer.chat_template == chat_bpe.chat_template
    rendered = chat_bpe.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    assert trimmed.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True) == rendered


def test_the_rendered_prompt_keeps_its_ids_only_when_the_template_is_kept(chat_bpe, chat_messages):
    spec = _spec(chat_bpe)
    counts = _counts_over(chat_bpe, ["the cat sat", "the dog"])

    def prompt_ids(config: SelectionConfig) -> tuple[list[int], list[int]]:
        selection = select_tokens(spec, counts, config)
        trimmed = trim_tokenizer(chat_bpe, spec, selection.kept_ids)
        old = chat_bpe.apply_chat_template(chat_messages, add_generation_prompt=True, return_dict=False)
        new = trimmed.tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True, return_dict=False)
        return trimmed.remap.map_sequence(old), new

    mapped, actual = prompt_ids(SelectionConfig(min_count=1))
    assert mapped == actual

    # Dropped, the prompt still renders but `assistant` fragments into characters, which
    # is the silent quality loss the option exists to prevent.
    mapped, actual = prompt_ids(SelectionConfig(min_count=1, keep_chat_template=False))
    assert mapped != actual
