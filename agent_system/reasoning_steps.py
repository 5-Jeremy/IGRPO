"""Split generated reasoning for rollout logging and per-step processing."""

import json
import re
from collections.abc import Sequence

import numpy as np


DEFAULT_THINKING_SPLIT_PHRASES = ("however,", "wait,", "but wait,", "alternatively,")


def resolve_thinking_split_phrases(model_name, thinking_split_phrases=None):
    """Resolve automatic model defaults; an explicit list always takes precedence.

    None selects sentence-ending newlines in the formatter. Keep the historical
    phrase defaults for Qwen models, including locally saved Qwen checkpoints.
    """
    if thinking_split_phrases is not None:
        return thinking_split_phrases
    return DEFAULT_THINKING_SPLIT_PHRASES if "qwen" in model_name.lower() else None


def format_reasoning_trace(steps, action):
    numbered = "\n\n".join(f"[{number}] {step}" for number, step in enumerate(steps, 1))
    return f"{numbered}\n\n<action>{action}</action>"


class ReasoningStepFormatter:
    """Split thinking into disjoint spans."""

    def __init__(self, thinking_split_phrases=DEFAULT_THINKING_SPLIT_PHRASES):
        if thinking_split_phrases is None:
            # Split only at a newline after '.' or '?', allowing trailing spaces
            # and Windows line endings. Colons, list items and wrapped lines stay.
            self.split_pattern = re.compile(r"(?<=[.?])[^\S\r\n]*\r?\n")
            return
        if (not isinstance(thinking_split_phrases, Sequence) or isinstance(thinking_split_phrases, str)
                or any(not isinstance(phrase, str) or not phrase.strip() or phrase != phrase.strip()
                       for phrase in thinking_split_phrases)):
            raise ValueError("thinking_split_phrases must be a list of nonempty strings without surrounding whitespace")
        # Longest alternative first prevents splitting 'but wait,' again at 'wait,'.
        phrases = sorted(set(thinking_split_phrases), key=lambda phrase: (-len(phrase), phrase))
        self.split_pattern = (re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)", re.I)
                              if phrases else None)

    def format(self, output, *, prompt=""):
        steps, action = self.parse(output, prompt=prompt)
        return format_reasoning_trace(steps, action)

    def parse(self, output, *, prompt=""):
        """Return disjoint reasoning-step texts and the generated action.

        A prompt ending in <think> can supply the opening tag, but only generated
        response text is numbered. Malformed/empty thinking raises ValueError.
        Prose between </think> and the action block is ignored.
        Action syntax is checked by the environment before this formatter is used.
        Search/answer and code blocks, and Gym Cards' JSON thoughts, are also
        supported so the same splitting rules apply to every environment.
        """
        steps, action, _ = self.parse_with_spans(output, prompt=prompt)
        return steps, action

    def parse_with_spans(self, output, *, prompt=""):
        """Also return half-open character spans in the original generated output."""
        if output.lstrip().startswith(("{", "```json", "```\n")):
            thought, action, offsets = self._parse_json(output)
        else:
            thought, action, offsets = self._parse_tagged(output, prompt)
        cuts = {0, len(thought)}
        if self.split_pattern is not None:
            cuts.update(match.start() for match in self.split_pattern.finditer(thought))
        boundaries = sorted(cuts)
        steps, spans = [], []
        for start, end in zip(boundaries, boundaries[1:]):
            segment = thought[start:end]
            if not segment.strip():
                continue
            steps.append(segment.strip())
            spans.append((offsets[start + len(segment) - len(segment.lstrip())],
                          offsets[end - (len(segment) - len(segment.rstrip()))]))
        if not steps:
            raise ValueError("No nonempty reasoning steps")
        return steps, action.strip(), spans

    @staticmethod
    def _parse_tagged(output, prompt):
        prefix_length = 0
        if not re.search(r"<think>", output, re.I) and re.search(r"<think>\s*$", prompt, re.I):
            prefix_length = len("<think>")
            output = "<think>" + output
        match = re.fullmatch(
            r"\s*<think>(.*?)</think>(.*?)<(?P<tag>action|search|answer|code)>(.*?)</(?P=tag)>\s*",
            output, re.S | re.I)
        if match is None:
            raise ValueError("Expected one think block followed by one action, search, answer or code block")
        thought, intervening_text, tag, action = match.groups()
        if not action.strip() or any(
            re.search(r"</?(?:think|action|search|answer|code)>", part, re.I)
            for part in (thought, intervening_text, action)
        ):
            raise ValueError("Nested/repeated tags or an empty action are not supported")
        if tag.lower() != "action":
            # Preserve which operation was chosen in the environment's format.
            action = f"<{tag.lower()}>{action.strip()}</{tag.lower()}>"
        offset = match.start(1) - prefix_length
        return thought, action, range(offset, offset + len(thought) + 1)

    @staticmethod
    def _parse_json(output):
        """Read Gym Cards thoughts and retain offsets through JSON escapes."""
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", output, re.S)
        source = fenced.group(1) if fenced else output
        payload = json.loads(source)
        if (not isinstance(payload, dict) or not isinstance(payload.get("thoughts"), str)
                or not isinstance(payload.get("action"), str) or not payload["action"].strip()):
            raise ValueError("Expected JSON thoughts and a nonempty action")
        match = re.search(r'"thoughts"\s*:\s*"((?:[^"\\]|\\.)*)"', source)
        if match is None:
            raise ValueError("Missing JSON thoughts string")
        thought = payload["thoughts"]
        offset = (fenced.start(1) if fenced else 0) + match.start(1)
        offsets = [offset]
        decoded = []
        # A JSON escape (including a UTF-16 surrogate pair) represents one
        # decoded character, but its entire source span belongs to that step.
        for char in re.finditer(
            r'\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\.|[^\\]',
            match.group(1),
        ):
            raw = char.group()
            decoded.append(json.loads('"' + raw + '"') if raw.startswith("\\") else raw)
            offsets.append(offset + char.end())
        if "".join(decoded) != thought:
            raise ValueError("Cannot map JSON thoughts to source characters")
        return thought, payload["action"], offsets


def attach_reasoning_steps(batch, tokenizer, thinking_split_phrases=DEFAULT_THINKING_SPLIT_PHRASES,
                           *, outputs=None, actions=None):
    """Save per-turn text for syntactically valid actions, without changing tensors."""
    formatter = ReasoningStepFormatter(thinking_split_phrases)
    if outputs is None:
        outputs = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if actions is None:
        actions = batch.non_tensor_batch.get("is_action_valid", [True] * len(outputs))
    values = []
    traces = []
    for index, (output, action) in enumerate(zip(outputs, actions, strict=True)):
        if not action:
            values.append(None)
            traces.append(None)
            continue
        prompt = ""
        if not re.search(r"<think>", output, re.I) and "prompts" in batch.batch:
            prompt = tokenizer.decode(batch.batch["prompts"][index], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        try:
            steps, action = formatter.parse(output, prompt=prompt)
            values.append(format_reasoning_trace(steps, action))
            traces.append({"steps": steps, "action": action})
        except ValueError:
            # An invalid/missing reasoning block must not change training behavior.
            values.append(None)
            traces.append(None)
    batch.non_tensor_batch["reasoning_steps"] = np.array(values, dtype=object)
    batch.non_tensor_batch["reasoning_trace"] = np.array(traces, dtype=object)
