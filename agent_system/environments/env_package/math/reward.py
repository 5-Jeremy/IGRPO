"""Terminal math answer reward."""

import re

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed


ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
CHOICE = re.compile(r"^\(?([A-E])\)?$", re.IGNORECASE)
VERIFY = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def choice_letter(value):
    value = value.strip()
    for wrapper in (r"\text", r"\mathrm", r"\mathbf"):
        if value.startswith(wrapper + "{") and value.endswith("}"):
            value = value[len(wrapper) + 1 : -1].strip()
    match = CHOICE.fullmatch(value)
    return match.group(1).upper() if match else None


def score_answer(action, ground_truth, used_python):
    if not used_python:
        return 0.0
    matches = ANSWER.findall(action)
    if len(matches) != 1:
        return 0.1
    target = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
    if not isinstance(target, str) or not target.strip():
        return 0.1
    boxed = last_boxed_only_string(matches[0])
    if boxed is None or not boxed.startswith(r"\boxed"):
        return 0.1
    correct = float(compute_score(matches[0], target))
    if not correct:
        predicted = choice_letter(remove_boxed(boxed))
        correct = float(predicted is not None and predicted == choice_letter(target))
    if not correct:
        try:
            correct, _ = VERIFY([r"\boxed{" + target + "}"], [boxed])
        except Exception:
            correct = 0.0
    return 0.1 + 0.9 * correct
