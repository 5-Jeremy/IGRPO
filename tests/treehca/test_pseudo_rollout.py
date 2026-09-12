"""Prompt preservation and tokenizer-backed checks for action-choice labels."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout import build_product_page_pseudo_rollout_prompt, select_action_labels

_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = runpy.run_path(str(_ROOT / "agent_system/environments/prompts/webshop.py"))
_EXAMPLES = json.loads((Path(__file__).parent / "fixtures/product_page_contexts.json").read_text())


@pytest.fixture(scope="module")
def tokenizer():
    # Use the configured model's cached tokenizer; no downloads or model weights are needed.
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)


@pytest.fixture
def parts():
    return extract_product_page_contexts([_EXAMPLES[0]["raw_context"]])[0]


def test_55_labels_have_distinct_bare_and_spaced_single_tokens(tokenizer, parts):
    labels = select_action_labels(tokenizer, 55)
    assert labels == tuple("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AZ BA BB BC BD".split())
    variants = [variant for label in labels for variant in (label, " " + label)]
    ids = [tokenizer.encode(variant, add_special_tokens=False) for variant in variants]
    assert all(len(tokens) == 1 for tokens in ids)
    assert len({tokens[0] for tokens in ids}) == 110

    components = replace(parts, admissible_actions=tuple(f"click[option {i}]" for i in range(55)))
    body, label_to_action = build_product_page_pseudo_rollout_prompt(components, tokenizer)
    assert tuple(label_to_action) == labels
    assert tuple(label_to_action.values()) == components.admissible_actions
    assert len(label_to_action) == 55
    assert "AY" not in label_to_action
    assert label_to_action["BD"] == "click[option 54]"
    assert "BD: click[option 54]" in body
    # Validate both spellings at the actual Qwen assistant generation boundary, not only in isolation.
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for variant, tokens in zip(variants, ids):
        assert tokenizer.encode(prompt + variant, add_special_tokens=False) == prompt_ids + tokens


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda example: example["name"])
@pytest.mark.parametrize("with_history", [False, True])
def test_only_action_entries_and_final_instructions_change(tokenizer, example, with_history):
    original = example["raw_context"]
    components = extract_product_page_contexts([original])[0]
    if not with_history:
        original = _TEMPLATES["WEBSHOP_TEMPLATE_NO_HIS"].format(
            task_description=components.shopping_task,
            current_observation=components.current_observation,
            available_actions="\n".join(f"'{action}'," for action in components.admissible_actions),
        )
        components = extract_product_page_contexts([original])[0]
    prompt, label_to_action = build_product_page_pseudo_rollout_prompt(components, tokenizer)

    labels = select_action_labels(tokenizer, len(components.admissible_actions))
    assert label_to_action == dict(zip(labels, components.admissible_actions))
    original_action_block = "[\n" + "\n".join(f"'{action}'," for action in components.admissible_actions) + "\n]."
    expected_action_block = "[\n" + "\n".join(f"{label}: {action}" for label, action in zip(labels, components.admissible_actions)) + "\n]."
    expected_footer = "Now it's your turn to take one action for the current step.\nYou must give the label corresponding to the action you want to take. You should only respond with a single label from the list"
    # Independently edit the original body; all other bytes must agree apart from surrounding newlines.
    assert original.count(original_action_block) == 1
    expected = original.replace(original_action_block, expected_action_block).replace(components.response_instructions, expected_footer)
    assert prompt.strip("\n") == expected.strip("\n")


def test_labels_follow_action_order_without_sorting(tokenizer, parts):
    components = replace(parts, admissible_actions=("click[zebra]", "click[apple]", "click[buy now]"))
    prompt, label_to_action = build_product_page_pseudo_rollout_prompt(components, tokenizer)
    assert label_to_action == {"A": "click[zebra]", "B": "click[apple]", "C": "click[buy now]"}
    assert "[\nA: click[zebra]\nB: click[apple]\nC: click[buy now]\n]." in prompt


def test_label_selection_uses_the_supplied_tokenizer(tokenizer, monkeypatch):
    original_encode = tokenizer.encode

    def encode_with_split_space_a(text, **kwargs):
        if text == " A":
            return [1, 2]
        return original_encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", encode_with_split_space_a)
    assert select_action_labels(tokenizer, 2) == ("B", "C")


def test_insufficient_single_token_labels_raise(tokenizer, monkeypatch):
    monkeypatch.setattr(tokenizer, "encode", lambda *args, **kwargs: [1, 2])
    with pytest.raises(ValueError, match="only 0 usable action labels"):
        select_action_labels(tokenizer, 1)


def test_labels_must_round_trip_to_the_original_spelling(tokenizer, monkeypatch):
    monkeypatch.setattr(tokenizer, "decode", lambda *args, **kwargs: "<unk>")
    with pytest.raises(ValueError, match="only 0 usable action labels"):
        select_action_labels(tokenizer, 1)


def test_empty_actions_raise(tokenizer, parts):
    with pytest.raises(ValueError, match="without admissible actions"):
        build_product_page_pseudo_rollout_prompt(replace(parts, admissible_actions=()), tokenizer)


def test_inconsistent_history_metadata_raises(tokenizer, parts):
    with pytest.raises(ValueError, match="History block and current-step"):
        build_product_page_pseudo_rollout_prompt(replace(parts, current_step=None), tokenizer)


@pytest.mark.parametrize("count", [-1, 703, True])
def test_invalid_action_counts_raise(tokenizer, count):
    with pytest.raises(ValueError, match="num_actions must be an integer"):
        select_action_labels(tokenizer, count)


def test_zero_actions_selects_no_labels(tokenizer):
    assert select_action_labels(tokenizer, 0) == ()
