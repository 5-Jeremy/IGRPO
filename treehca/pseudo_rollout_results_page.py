"""Construct answer-scoring pseudo-rollouts for WebShop results pages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

_ACTION_BLOCK = re.compile(
    r"\nYour admissible actions of the current situation are:[ \t]*\n\[\n"
    r"(?P<actions>.*?)"
    r"\n\]\."
    r"(?=\n\nNow it's your turn to take one action for the current step\.)",
    re.DOTALL,
)


def _extract_admissible_actions(prompt: str) -> tuple[str, ...]:
    """Read the displayed actions without rebuilding or otherwise editing the prompt."""
    matches = list(_ACTION_BLOCK.finditer(prompt))
    if len(matches) != 1:
        raise ValueError(f"Expected one admissible-action block, found {len(matches)}")

    actions = []
    for line in matches[0]["actions"].splitlines():
        if not line.startswith("'") or not line.endswith("',"):
            raise ValueError("Each admissible action must be single-quoted and followed by a comma")
        actions.append(line[1:-2])
    if not actions:
        raise ValueError("A results-page prompt must contain at least one admissible action")
    if len(set(actions)) != len(actions):
        raise ValueError("Duplicate admissible actions are ambiguous")
    return tuple(actions)


def build_results_page_pseudo_rollout(prompt: str, action: str, *, assistant_response_prefix: str = "") -> str:
    """Append an exact admissible action as an ``<answer>`` response.

    ``prompt`` is retained byte-for-byte. ``assistant_response_prefix`` may be
    empty or may contain an already-generated prefix such as a complete
    ``<think>...</think>`` block; it too is appended without modification.
    The caller can therefore tokenize the returned string and obtain prompt
    log-probabilities for the tokens belonging to ``action``.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    if not isinstance(assistant_response_prefix, str):
        raise ValueError("assistant_response_prefix must be a string")

    admissible_actions = _extract_admissible_actions(prompt)
    if action not in admissible_actions:
        raise ValueError(f"Action {action!r} is not one of the prompt's admissible actions")
    return f"{prompt}{assistant_response_prefix}<answer>{action}</answer>"


def _tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Apply the model's chat template while retaining the prompt as user content."""
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one tokenized conversation")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("Chat template must return a flat list of integer token IDs")
    return token_ids


@dataclass(frozen=True)
class ResultsPageAnswerProbe:
    """Unpadded teacher-forcing input and the response tokens to measure."""

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    answer_token_indices: tuple[int, ...]

    @property
    def input_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.response_token_ids


def prepare_results_page_answer_probe(
    prompt: str,
    action: str,
    tokenizer: Any,
    *,
    assistant_response_prefix: str = "",
    prompt_token_ids: tuple[int, ...] | None = None,
) -> ResultsPageAnswerProbe:
    """Prepare scalar-scoring tokens; reuse prompt IDs for the same prompt."""
    pseudo_rollout = build_results_page_pseudo_rollout(
        prompt,
        action,
        assistant_response_prefix=assistant_response_prefix,
    )
    # Tokenize the assistant response independently so its offsets refer to
    # the exact string that the model is being asked to generate.
    response = pseudo_rollout[len(prompt) :]
    encoded_response = tokenizer(
        response,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = encoded_response["input_ids"]
    offsets = encoded_response["offset_mapping"]
    if not isinstance(response_ids, torch.Tensor) or response_ids.ndim != 2 or response_ids.shape[0] != 1:
        raise ValueError("Tokenizer must return input_ids with shape [1, response_length]")
    if not isinstance(offsets, torch.Tensor) or offsets.shape != (*response_ids.shape, 2):
        raise ValueError("Tokenizer must return offset_mapping with shape [1, response_length, 2]")

    answer_start = response.rfind("<answer>")
    answer_end = response.find("</answer>", answer_start + len("<answer>"))
    if answer_start < 0 or answer_end < 0:
        raise ValueError("Pseudo-rollout response must contain a complete <answer> block")
    answer_interval = (answer_start + len("<answer>"), answer_end)
    answer_token_indices = [index for index, (start, end) in enumerate(offsets[0].tolist()) if start != end and max(start, answer_interval[0]) < min(end, answer_interval[1])]
    if not answer_token_indices:
        raise ValueError(f"No tokens overlap the action span {answer_interval} for action {action!r}")

    if prompt_token_ids is None:
        prompt_token_ids = tuple(_tokenize_prompt(tokenizer, prompt))
    return ResultsPageAnswerProbe(prompt_token_ids, tuple(response_ids[0].tolist()), tuple(answer_token_indices))


def compute_results_page_answer_probability(
    prompt: str,
    action: str,
    tokenizer: Any,
    actor_rollout_wg: Any,
    *,
    assistant_response_prefix: str = "",
) -> torch.Tensor:
    """Return joint action-token probability, retaining the original prompt.

    Sum log probabilities for response tokens overlapping the final answer's
    action text. Neither the answer tags nor an optional thinking prefix enter
    that sum, although they remain part of the conditioning context.
    """
    probe = prepare_results_page_answer_probe(prompt, action, tokenizer, assistant_response_prefix=assistant_response_prefix)
    if actor_rollout_wg is None or not callable(getattr(actor_rollout_wg, "compute_log_prob", None)):
        raise ValueError("actor_rollout_wg must provide compute_log_prob")
    response_ids = torch.tensor([probe.response_token_ids], dtype=torch.long)
    prompt_ids = torch.tensor([probe.prompt_token_ids], dtype=torch.long)
    input_ids = torch.cat((prompt_ids, response_ids), dim=1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)

    # Keep this import local so callers that only construct text rollouts do
    # not need to import the training stack.
    from verl import DataProto

    scoring_batch = DataProto.from_dict(
        tensors={
            "responses": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    )
    log_prob_output = actor_rollout_wg.compute_log_prob(scoring_batch)
    response_log_probs = log_prob_output.batch.get("old_log_probs")
    if not isinstance(response_log_probs, torch.Tensor) or response_log_probs.shape != response_ids.shape:
        raise ValueError("compute_log_prob must return old_log_probs matching the response shape")

    # Joint probabilities for multi-token actions can fall below float32's
    # range even when every log-probability is finite.
    joint_log_probability = response_log_probs[0, list(probe.answer_token_indices)].double().sum()
    return joint_log_probability.exp()
