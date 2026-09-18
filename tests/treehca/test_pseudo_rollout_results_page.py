"""Tests for byte-preserving results-page pseudo-rollouts."""

import pytest
import torch

from treehca.pseudo_rollout_results_page import build_results_page_pseudo_rollout, compute_results_page_answer_probability, prepare_results_page_answer_probe
from verl import DataProto


class _CharacterTokenizer:
    """Small offset-aware tokenizer that makes every character one token."""

    def __init__(self):
        self.chat_template_calls = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.chat_template_calls.append(conversation)
        rendered = f"CHAT:{conversation[0]['content']}ASSISTANT:"
        return [ord(character) for character in rendered]

    def __call__(self, text, *, return_tensors, add_special_tokens, return_offsets_mapping):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": torch.tensor([[ord(character) for character in text]], dtype=torch.long),
            "offset_mapping": torch.tensor([[[index, index + 1] for index in range(len(text))]], dtype=torch.long),
        }


class _FakeActorRolloutWorkerGroup:
    def __init__(self):
        self.batch = None

    def compute_log_prob(self, batch):
        self.batch = batch
        response_length = batch.batch["responses"].shape[1]
        log_probs = -torch.arange(1, response_length + 1, dtype=torch.float32).unsqueeze(0) / 10
        return DataProto.from_dict(tensors={"old_log_probs": log_probs})


@pytest.fixture
def prompt():
    return (
        "\nYou are an expert autonomous agent operating in the WebShop e-commerce environment.\n"
        "Your task is to: buy a desk.\n"
        "Your current observation is: 'Back to Search' [SEP] 'Page 1 (Total results: 2)'.\n"
        "Your admissible actions of the current situation are: \n"
        "[\n"
        "'click[back to search]',\n"
        "'click[item-1]',\n"
        "'click[item-2]',\n"
        "].\n\n"
        "Now it's your turn to take one action for the current step.\n"
        "Think, then respond.\n"
    )


def test_keeps_prompt_exactly_and_appends_answer(prompt):
    rollout = build_results_page_pseudo_rollout(prompt, "click[item-2]")

    assert rollout[: len(prompt)] == prompt
    assert rollout[len(prompt) :] == "<answer>click[item-2]</answer>"


def test_keeps_generated_thinking_prefix_exactly(prompt):
    thinking = "<think>Item 1 is cheaper.</think>\n"

    rollout = build_results_page_pseudo_rollout(prompt, "click[item-1]", assistant_response_prefix=thinking)

    assert rollout == prompt + thinking + "<answer>click[item-1]</answer>"


def test_requires_an_exact_admissible_action(prompt):
    with pytest.raises(ValueError, match="not one of"):
        build_results_page_pseudo_rollout(prompt, "click[ITEM-1]")


def test_computes_joint_probability_without_click_prefix_tokens(prompt):
    tokenizer = _CharacterTokenizer()
    worker = _FakeActorRolloutWorkerGroup()
    thinking = "<think>Item 1 is cheaper.</think>\n"

    probability = compute_results_page_answer_probability(
        prompt,
        "click[item-1]",
        tokenizer,
        worker,
        assistant_response_prefix=thinking,
    )

    response = thinking + "<answer>click[item-1]</answer>"
    answer_start = response.index("item-1]")
    answer_indices = torch.arange(answer_start, answer_start + len("item-1]"))
    expected_joint_log_probability = (-(answer_indices + 1).double() / 10).sum()
    assert probability == pytest.approx(expected_joint_log_probability.exp())
    assert tokenizer.chat_template_calls == [[{"role": "user", "content": prompt}]]
    assert worker.batch.batch["responses"].shape == (1, len(response))
    assert worker.batch.batch["input_ids"].shape[1] > len(response)


def test_counts_token_that_merges_click_prefix_with_action_text(prompt):
    class MergedTokenizer(_CharacterTokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            start = text.index("click[item-1]")
            merge_start = start + len("click")
            end = start + len("click[i")
            ids = encoded["input_ids"][0].tolist()
            offsets = encoded["offset_mapping"][0].tolist()
            encoded["input_ids"] = torch.tensor([ids[:merge_start] + [ids[merge_start]] + ids[end:]])
            encoded["offset_mapping"] = torch.tensor([offsets[:merge_start] + [[merge_start, end]] + offsets[end:]])
            return encoded

    probe = prepare_results_page_answer_probe(prompt, "click[item-1]", MergedTokenizer())
    response = "<answer>click[item-1]</answer>"
    merged_index = response.index("click[item-1]") + len("click")

    assert probe.answer_token_indices[0] == merged_index
    assert len(probe.answer_token_indices) == len("item-1]")
    assert all(index >= merged_index for index in probe.answer_token_indices)


def test_probability_rejects_malformed_log_prob_output(prompt):
    class BadWorker:
        def compute_log_prob(self, batch):
            return DataProto.from_dict(tensors={"old_log_probs": torch.zeros((1, 1))})

    with pytest.raises(ValueError, match="matching the response shape"):
        compute_results_page_answer_probability(prompt, "click[item-1]", _CharacterTokenizer(), BadWorker())


def test_joint_probability_does_not_underflow_at_float32_range(prompt):
    class LowProbabilityWorker:
        def compute_log_prob(self, batch):
            log_probs = torch.full(batch.batch["responses"].shape, -50.0)
            return DataProto.from_dict(tensors={"old_log_probs": log_probs})

    probability = compute_results_page_answer_probability(
        prompt,
        "click[item-1]",
        _CharacterTokenizer(),
        LowProbabilityWorker(),
    )

    assert probability.dtype == torch.float64
    assert probability > 0.0
