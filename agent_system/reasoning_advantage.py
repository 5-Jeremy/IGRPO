"""Align reasoning spans and shape each turn's estimated policy advantages."""

from copy import deepcopy

import numpy as np
import torch

from agent_system.reasoning_steps import DEFAULT_THINKING_SPLIT_PHRASES, ReasoningStepFormatter
from agent_system.token_alignment import response_token_offsets


def attach_reasoning_token_ids(batch, tokenizer, thinking_split_phrases=DEFAULT_THINKING_SPLIT_PHRASES):
    """Attach verified response-shaped step IDs before batch padding/reordering.

    Zero means outside a reasoning step or an unaligned token. Only tokens wholly
    contained in one trimmed step are assigned; boundary-crossing tokens stay zero.
    ByteLevel offsets come from the original token bytes; other decoders require
    verified re-encoding. Decoding preserves whitespace throughout the pipeline.
    """
    responses = batch.batch["responses"]
    step_ids = torch.zeros_like(responses)
    formatter = ReasoningStepFormatter(thinking_split_phrases)
    outputs = tokenizer.batch_decode(responses, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    statuses = []
    for row, (output, trace) in enumerate(zip(outputs, batch.non_tensor_batch["reasoning_trace"], strict=True)):
        if trace is None:
            statuses.append("no_trace")
            continue
        prompt = tokenizer.decode(batch.batch["prompts"][row], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False) if "prompts" in batch.batch else ""
        steps, action, spans = formatter.parse_with_spans(output, prompt=prompt)
        if steps != trace["steps"] or action != trace["action"]:
            raise ValueError("Reasoning token mapping disagrees with submitted trace")
        positions, offsets, status = response_token_offsets(tokenizer, responses[row].tolist(), output)
        if status != "aligned":
            statuses.append(status)
            continue
        for position, (start, end) in zip(positions, offsets, strict=True):
            for number, (step_start, step_end) in enumerate(spans, 1):
                if step_start <= start < end <= step_end:
                    step_ids[row, position] = number
                    break
        statuses.append("aligned")
    batch.batch["reasoning_step_ids"] = step_ids
    batch.non_tensor_batch["reasoning_alignment_status"] = np.array(statuses, dtype=object)


def apply_reasoning_advantage_adjustment(
    original_advantage, *, step_number, step_text, step_probabilities,
    all_step_probabilities, reasoning_steps, turn_id, turn_reward=None,
):
    """Scale this step's trainable token advantages using its action judgment.

    original_advantage includes any turn credit assigned by the estimator.
    turn_reward supplies IGPO's pre-normalization reward or the environment's
    turn reward when available, for rules that also need the reward itself.
    """
    if not step_probabilities:
        return original_advantage

    support_label = "The reasoning step supports the chosen action"
    support = step_probabilities.get(support_label, 0)
    contradict = step_probabilities.get("The reasoning step contradicts the chosen action", 0)
    neutral = step_probabilities.get("The reasoning step makes no claim on what the correct action is", 0)
    # Step 1 has no preceding support, so it can earn the initial support bonus.
    previous_support = all_step_probabilities.get(f"step_{step_number - 1}", {}).get(support_label, 0)

    if neutral > 0.5 and (contradict <= support + 0.01 or contradict <= 0.1):
        return 0.3 * original_advantage
    if neutral > 0.5 and contradict > support + 0.01 and contradict > 0.2:
        return -0.3 * original_advantage
    if support >= 0.5:
        return original_advantage
    if contradict >= 0.5:
        return -original_advantage
    return original_advantage


@torch.no_grad()
def adjust_reasoning_advantages(batch):
    """Invoke the hook per step, scatter results, and save per-step summaries."""
    original = batch.batch["advantages"]
    adjusted = original.clone()  # GRPO advantages and returns can share storage.
    ids = batch.batch["reasoning_step_ids"]
    mask = batch.batch["response_mask"].bool()
    if "loss_mask" in batch.batch:
        mask = mask & batch.batch["loss_mask"][:, -original.shape[1]:].bool()
    probabilities = batch.non_tensor_batch.get("decision_model_probabilities", [None] * len(batch))
    metadata = batch.non_tensor_batch
    turn_rewards = metadata.get("igpo_step_reward", metadata.get("turn_reward", metadata.get("rewards")))
    logs = []
    for row, trace in enumerate(batch.non_tensor_batch["reasoning_trace"]):
        if trace is None:
            logs.append(None)
            continue
        step_log = {}
        for number, text in enumerate(trace["steps"], 1):
            key = f"step_{number}"
            selected = mask[row] & (ids[row] == number)
            before = original[row, selected].clone()
            entry = {"token_count": before.numel(), "original_advantage": None, "advantage": None,
                     "min": None, "max": None, "probabilities_available": key in (probabilities[row] or {})}
            if before.numel():
                all_probabilities = deepcopy(probabilities[row] or {})
                result = apply_reasoning_advantage_adjustment(
                    before.clone(), step_number=number, step_text=text,
                    step_probabilities=all_probabilities.get(key), all_step_probabilities=all_probabilities,
                    reasoning_steps=list(trace["steps"]), turn_id=metadata["id"][row],
                    turn_reward=float(turn_rewards[row]) if turn_rewards is not None else None,
                )
                if (not isinstance(result, torch.Tensor) or result.shape != before.shape
                        or result.dtype != before.dtype or result.device != before.device
                        or not torch.isfinite(result).all()):
                    raise ValueError(f"Invalid reasoning advantage result for turn {batch.non_tensor_batch['id'][row]}, {key}")
                adjusted[row, selected] = result
                entry.update(original_advantage=before.float().mean().item(), advantage=result.float().mean().item(),
                             min=result.min().item(), max=result.max().item())
            step_log[key] = entry
        logs.append(step_log)
    batch.batch["advantages"] = adjusted
    batch.non_tensor_batch["reasoning_step_advantages"] = np.array(logs, dtype=object)
    # Count each actual rollout once, excluding training padding copies.
    unique = dict(zip(batch.non_tensor_batch["id"], range(len(batch)), strict=True))
    entries = [entry for row in unique.values() for entry in (logs[row] or {}).values()]
    values = [entry["advantage"] for entry in entries if entry["advantage"] is not None]
    statuses = batch.non_tensor_batch["reasoning_alignment_status"]
    return {
        "reasoning_advantage/step_count": len(entries),
        "reasoning_advantage/scored_step_count": len(values),
        "reasoning_advantage/unmapped_step_count": len(entries) - len(values),
        "reasoning_advantage/alignment_failure_count": sum(statuses[row] not in ("aligned", "no_trace") for row in unique.values()),
        **({"reasoning_advantage/mean": float(np.mean(values))} if values else {}),
    }
