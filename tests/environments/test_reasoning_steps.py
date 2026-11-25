"""Reasoning trace formatting matches the requested split and output semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from agent_system.reasoning_steps import (
    DEFAULT_THINKING_SPLIT_PHRASES,
    ReasoningStepFormatter,
    attach_reasoning_steps,
    resolve_thinking_split_phrases,
)


def trace(thought):
    return f"<think>{thought}</think>\n<action>Lima, Peru</action>"


def test_default_phrases_keep_longest_match_and_original_text():
    thought = "First idea. However, another option. But wait, reconsider. WAIT, retry. Alternatively, finish."
    assert ReasoningStepFormatter().format(trace(thought)) == (
        "[1] First idea.\n\n[2] However, another option.\n\n[3] But wait, reconsider."
        "\n\n[4] WAIT, retry.\n\n[5] Alternatively, finish.\n\n<action>Lima, Peru</action>"
    )


def test_no_empty_steps_and_no_matches_within_words():
    assert ReasoningStepFormatter().format(trace("\n Wait, first. await, x. however,next. However, last.\n")) == (
        "[1] Wait, first. await, x. however,next.\n\n[2] However, last.\n\n<action>Lima, Peru</action>"
    )


def test_custom_phrases_are_literal_and_empty_list_disables_splitting():
    thought = "A. However, B. Next (idea), C."
    formatter = ReasoningStepFormatter(OmegaConf.create(["next (idea),"]))
    assert formatter.format(trace(thought)).startswith("[1] A. However, B.\n\n[2] Next (idea), C.")
    assert ReasoningStepFormatter([]).format(trace(thought)) == f"[1] {thought}\n\n<action>Lima, Peru</action>"


def test_prompt_opening_think_tag_only_adds_generated_reasoning():
    output = "First. Wait, next.</think><action>population of Lima</action>"
    assert ReasoningStepFormatter().format(output, prompt="User context.\n<think>\n") == (
        "[1] First.\n\n[2] Wait, next.\n\n<action>population of Lima</action>"
    )
    with pytest.raises(ValueError):
        ReasoningStepFormatter().format(output, prompt="There was a <think> tag earlier.\n")


@pytest.mark.parametrize("output", [
    "<action>Lima, Peru</action>",
    trace("   "),
    "<think>unfinished<action>Lima, Peru</action>",
    trace("nested <think>thought</think>"),
    trace("first") + trace("second"),
    "<think>thought</think><action></action>",
    "<think>thought</think><search>unfinished\n<answer>Lima</answer>",
])
def test_invalid_or_missing_thinking_is_rejected(output):
    with pytest.raises(ValueError):
        ReasoningStepFormatter().format(output)


@pytest.mark.parametrize("phrases", ["wait,", [""], [" wait,"], ["wait, "], [None], [3]])
def test_bad_split_configuration_is_rejected(phrases):
    with pytest.raises(ValueError, match="thinking_split_phrases"):
        ReasoningStepFormatter(phrases)


@pytest.mark.parametrize("model", ["Qwen/Qwen3-8B", "/models/qWeN2.5/checkpoint", "Qwen2TokenizerFast"])
def test_qwen_automatic_default_and_explicit_overrides(model):
    assert resolve_thinking_split_phrases(model) == DEFAULT_THINKING_SPLIT_PHRASES
    assert resolve_thinking_split_phrases(model, []) == []
    assert resolve_thinking_split_phrases(model, ["next,"]) == ["next,"]


@pytest.mark.parametrize("model", ["meta-llama/Llama-3.1-8B", "google/gemma-3", "", "/checkpoints/policy"])
def test_other_models_default_to_sentence_newlines_and_honor_overrides(model):
    assert resolve_thinking_split_phrases(model) is None
    assert resolve_thinking_split_phrases(model, []) == []
    assert resolve_thinking_split_phrases(model, ["next,"]) == ["next,"]


@pytest.mark.parametrize("prompt_supplies_think", [False, True])
def test_sentence_newlines_keep_colons_wrapped_lines_and_inline_sentences(prompt_supplies_think):
    thought = ("  First sentence. Still the first step? \t\r\n\n"
               "Options:\n- one\n- two;\nNot finished!\nWrapped\nline.\n"
               "However, same step. Wait, still same step?\n  Last.  ")
    output = trace(thought)
    prompt = ""
    if prompt_supplies_think:
        output = output.removeprefix("<think>")
        prompt = "context <think>\n"
    steps, action, spans = ReasoningStepFormatter(None).parse_with_spans(output, prompt=prompt)
    assert steps == ["First sentence. Still the first step?",
                     "Options:\n- one\n- two;\nNot finished!\nWrapped\nline.",
                     "However, same step. Wait, still same step?", "Last."]
    assert [output[start:end] for start, end in spans] == steps
    assert action == "Lima, Peru"


@pytest.mark.parametrize("operation", ["action", "search", "answer", "code"])
@pytest.mark.parametrize("intervening_text", ["", "\nHowever, proceed.\n"])
def test_native_environment_operation_tags(operation, intervening_text):
    output = f"<think>First.\nSecond?</think>{intervening_text}<{operation}>chosen</{operation}>"
    steps, action, spans = ReasoningStepFormatter(None).parse_with_spans(output)
    assert steps == ["First.", "Second?"]
    assert [output[start:end] for start, end in spans] == steps
    assert action == ("chosen" if operation == "action" else f"<{operation}>chosen</{operation}>")


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("phrases,expected", [
    (None, ['Read "cards" é 😀.', 'Options:\nHit?']),
    (["Options:"], ['Read "cards" é 😀.', 'Options:\nHit?']),
    ([], ['Read "cards" é 😀.\nOptions:\nHit?']),
])
def test_gym_cards_json_thoughts_and_escaped_source_spans(fenced, ensure_ascii, phrases, expected):
    output = json.dumps({"thoughts": 'Read "cards" é 😀.\nOptions:\nHit?', "action": "hit"}, ensure_ascii=ensure_ascii)
    if fenced:
        output = f"```json\n{output}\n```"
    steps, action, spans = ReasoningStepFormatter(phrases).parse_with_spans(output)
    assert steps == expected
    assert [json.loads('"' + output[start:end] + '"') for start, end in spans] == steps
    assert action == "hit"


@pytest.mark.parametrize("output", [
    '<think>First.\nSecond.</think><action>take apple 1 from table 1</action>',
    '<think>First.\nSecond.</think><search>query</search>',
    '<think>First.\nSecond.</think><code>print(1)</code>',
    json.dumps({"thoughts": "First.\nSecond.", "action": "+"}),
])
def test_attach_reasoning_skips_invalid_actions(output):
    batch = SimpleNamespace(batch={}, non_tensor_batch={"is_action_valid": np.array([True, False, True])})
    tokenizer = SimpleNamespace()
    outputs = [output, output, "<action>missing reasoning</action>"]
    attach_reasoning_steps(batch, tokenizer, None, outputs=outputs)
    assert batch.non_tensor_batch["reasoning_trace"][0]["steps"] == ["First.", "Second."]
    assert batch.non_tensor_batch["reasoning_trace"][1:].tolist() == [None, None]


@pytest.mark.parametrize("output", [
    '{"thoughts": null, "action": "hit"}',
    '{"thoughts": "valid", "action": ""}',
    '{"thoughts": "valid", "action": 3}',
    '{"thoughts": "unterminated',
    '{"action": "hit"}',
])
def test_malformed_json_reasoning_is_rejected(output):
    with pytest.raises(ValueError):
        ReasoningStepFormatter(None).parse(output)
