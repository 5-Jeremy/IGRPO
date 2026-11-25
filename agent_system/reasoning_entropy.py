"""Summarize actor token-distribution entropy for generated reasoning and answers."""

import json
import re

import numpy as np
import torch

from agent_system.reasoning_steps import DEFAULT_THINKING_SPLIT_PHRASES, ReasoningStepFormatter
from agent_system.token_alignment import response_token_offsets


def answer_span(output):
    """Locate answer content in native tagged or Gym Cards JSON responses.

    This also works without a reasoning trace (e.g. an action-only response).
    Tags and JSON keys/quotes are excluded; tagged answer content is trimmed.
    """
    if output.lstrip().startswith(("{", "```json", "```\n")):
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", output, re.S)
        source = fenced.group(1) if fenced else output
        try:
            payload = json.loads(source)
        except ValueError:
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
            return None
        matches = list(re.finditer(r'"action"\s*:\s*"((?:[^"\\]|\\.)*)"', source))
        if len(matches) != 1 or not payload["action"].strip():
            return None
        match = matches[0]
        offset = fenced.start(1) if fenced else 0
        return offset + match.start(1), offset + match.end(1)
    matches = list(re.finditer(r"<(?P<tag>action|answer|search|code)>(.*?)</(?P=tag)>", output, re.S | re.I))
    if len(matches) != 1:
        return None
    match = matches[0]
    content = match.group(2)
    if not content.strip() or re.search(r"</?(?:think|action|answer|search|code)>", content, re.I):
        return None
    return match.start(2) + len(content) - len(content.lstrip()), match.end(2) - len(content) + len(content.rstrip())


def _summary(values):
    values = values.float()
    valid = values.numel() > 0 and bool(torch.isfinite(values).all())
    return {"mean": values.mean().item() if valid else None,
            "sum": values.sum().item() if valid else None, "token_count": values.numel()}


@torch.no_grad()
def attach_reasoning_entropy(batch, tokenizer, entropys,
                             thinking_split_phrases=DEFAULT_THINKING_SPLIT_PHRASES):
    """Save mean/sum entropy in nats, independently of decision-model requests.

    Offsets are verified against the original response tokens and decoded text.
    Each content-overlapping token belongs to the span with greatest overlap
    (earlier span wins ties), so boundary tokens are counted once. Padding and
    special tokens are excluded. Loss masks do not exclude generated content.
    Failed alignment produces null summaries, never a fabricated zero entropy.
    """
    responses = batch.batch["responses"]
    if entropys.shape != responses.shape:
        raise ValueError("Entropy must have the same shape as response tokens")
    formatter = ReasoningStepFormatter(thinking_split_phrases)
    outputs = tokenizer.batch_decode(responses, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    traces = batch.non_tensor_batch.get("reasoning_trace", [None] * len(outputs))
    # response_mask includes the training loss mask; use attention to keep all
    # content tokens even when they are excluded from the policy objective.
    mask = batch.batch["attention_mask"][:, -responses.shape[1]:].bool()
    step_logs, answer_logs, statuses = [], [], []
    for row, (output, trace) in enumerate(zip(outputs, traces, strict=True)):
        spans = []
        if trace is not None:
            prompt = tokenizer.decode(batch.batch["prompts"][row], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False) if "prompts" in batch.batch else ""
            steps, action, spans = formatter.parse_with_spans(output, prompt=prompt)
            if steps != trace["steps"] or action != trace["action"]:
                raise ValueError("Entropy token mapping disagrees with saved trace")
        answer = answer_span(output)
        step_log = {f"step_{i}": _summary(entropys[row, :0]) for i in range(1, len(spans) + 1)}
        step_logs.append(step_log if trace is not None else None)
        answer_logs.append(None)
        if not spans and answer is None:
            statuses.append("no_content")
            continue
        positions, offsets, status = response_token_offsets(tokenizer, responses[row].tolist(), output)
        if status != "aligned":
            statuses.append(status)
            continue
        regions = [*spans, *([answer] if answer is not None else [])]
        owned = [[] for _ in regions]
        for position, (start, end) in zip(positions, offsets, strict=True):
            if not mask[row, position] or end <= start:
                continue
            overlaps = [max(0, min(end, right) - max(start, left)) for left, right in regions]
            if max(overlaps) > 0:
                owned[overlaps.index(max(overlaps))].append(position)
        for i in range(len(spans)):
            step_log[f"step_{i + 1}"] = _summary(entropys[row, owned[i]])
        if answer is not None:
            answer_logs[-1] = _summary(entropys[row, owned[-1]])
        statuses.append("aligned")
    batch.non_tensor_batch["reasoning_step_entropies"] = np.array(step_logs, dtype=object)
    batch.non_tensor_batch["answer_entropy"] = np.array(answer_logs, dtype=object)
    batch.non_tensor_batch["entropy_alignment_status"] = np.array(statuses, dtype=object)
