"""Empirically validate product-option-group pseudo-rollout probabilities.

The constrained pseudo distribution for each option group is compared with the
ordinary product-page policy under a configurable empirical denominator. The
pseudo probe can use either an artificial cue or thinking sampled from the
ordinary policy. Singleton groups are excluded because they are not useful
calibration cases.
"""

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from treehca.pseudo_rollout_product_page import (
    GROUP_NONE_ACTION,
    ProductOptionGroupPseudoRollout,
    ProductOptionGroupPseudoRolloutScores,
    prepare_product_page_grouped_choice_rollouts,
    required_max_logprobs,
    score_product_page_grouped_choice_rollouts,
)
from treehca.pseudo_rollout_product_page_choices_testbed import (
    EmpiricalActionSamples,
    RenderedProductPage,
    SampledAgentResponse,
    average_pseudo_rollout_scores,
    compare_action_distributions,
    load_webshop_products_and_goals,
    project_completions,
    render_product_pages,
    sample_diverse_products,
    sample_real_prompt_completions,
    tokenize_real_prompts,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_WEBSHOP_ROOT = _ROOT / "agent_system/environments/env_package/webshop/webshop"
_RESULTS_ROOT = _ROOT / "pseudo_prob_test_results"
_DEFAULT_CATALOG = _WEBSHOP_ROOT / "data/items_shuffle_1000.json"
_DEFAULT_ATTRIBUTES = _WEBSHOP_ROOT / "data/items_ins_v2_1000.json"
_DEFAULT_OUTPUT = _RESULTS_ROOT / "pseudo_rollout_grouped_choice_calibration_report.json"
_DEFAULT_HIGH_PROBABILITY_THRESHOLD = 0.6
_GROUP_DECISION_CUE_TEMPLATE = "The best choice for the {group_name} group corresponds to the label:"
_ARTIFICIAL_THINKING_TEMPLATE = f"<think>{_GROUP_DECISION_CUE_TEMPLATE}"


@dataclass(frozen=True)
class SelectedProductGoal:
    """A sampled product and the exact synthetic goal used to render it."""

    product: dict[str, Any]
    goal: dict[str, Any]


@dataclass(frozen=True)
class GroupResponsePrefix:
    """Assistant-side context placed before a grouped-choice label probe."""

    text: str
    token_ids: tuple[int, ...]
    source: str
    status: str = "artificial_group_thinking"


@dataclass(frozen=True)
class EmpiricalGroupOptionSamples(EmpiricalActionSamples):
    """Projected samples plus the configured option-probability denominator."""

    probability_denominator: int = 0
    included_valid_non_option_completions: int = 0
    different_group_option_completions: int = 0
    denominator_inclusions: tuple[bool, ...] = ()


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _goal_option_mapping(goal: Mapping[str, Any]) -> dict[str, str] | None:
    goal_options = goal.get("goal_options")
    if not isinstance(goal_options, Mapping) or not goal_options:
        return None
    if any(not isinstance(name, str) or not isinstance(option, str) for name, option in goal_options.items()):
        return None
    return dict(goal_options)


def sample_diverse_product_goals(
    products: Sequence[dict[str, Any]],
    goals: Sequence[dict[str, Any]],
    num_pages: int,
    seed: int,
) -> tuple[SelectedProductGoal, ...]:
    """Sample diverse products having synthetic goals and nontrivial groups."""
    eligible_products = [product for product in products if any(isinstance(values, list) and len(values) >= 2 for values in product.get("options", {}).values())]
    eligible_asins = {product.get("asin") for product in eligible_products}
    eligible_goals = [goal for goal in goals if goal.get("asin") in eligible_asins and _goal_option_mapping(goal) is not None]
    selected = sample_diverse_products(eligible_products, eligible_goals, num_pages, seed)

    goals_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for goal in eligible_goals:
        key = (goal["asin"], goal["instruction_text"])
        goals_by_key.setdefault(key, []).append(goal)

    result = []
    for product, task in selected:
        matches = goals_by_key[(product["asin"], task)]
        option_mappings = {_goal_options_key(goal) for goal in matches}
        if len(option_mappings) != 1:
            raise ValueError(f"Selected task for ASIN {product['asin']!r} maps to conflicting goal_options")
        result.append(SelectedProductGoal(product=product, goal=matches[0]))
    return tuple(result)


def _goal_options_key(goal: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    goal_options = _goal_option_mapping(goal)
    if goal_options is None:
        raise ValueError("Expected a nonempty synthetic goal_options mapping")
    return tuple(sorted(goal_options.items()))


def build_artificial_group_response_prefixes(
    pseudo_rollouts: Sequence[ProductOptionGroupPseudoRollout],
    tokenizer: Any,
) -> tuple[GroupResponsePrefix, ...]:
    """Tokenize the deterministic thinking cue for every flattened group row.

    Returning explicit response-prefix objects keeps this path aligned with the
    generated-thinking strategy, which supplies a different prefix per sample.
    """
    prefixes = []
    for pseudo_rollout in pseudo_rollouts:
        prefix_text = _ARTIFICIAL_THINKING_TEMPLATE.format(group_name=pseudo_rollout.option_group.name)
        prefixes.append(_encode_group_response_prefix(prefix_text, tokenizer, source="artificial_group_thinking", status="artificial_group_thinking"))
    return tuple(prefixes)


def _encode_group_response_prefix(text: str, tokenizer: Any, *, source: str, status: str) -> GroupResponsePrefix:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("Tokenizer must encode each group response prefix as integer token IDs")
    return GroupResponsePrefix(text=text, token_ids=tuple(token_ids), source=source, status=status)


def extract_generated_group_thinking_prefix(
    completion: SampledAgentResponse,
    pseudo_rollout: ProductOptionGroupPseudoRollout,
    tokenizer: Any,
) -> GroupResponsePrefix:
    """Retain sampled thinking and replace its closing suffix with a group cue."""
    cue = _GROUP_DECISION_CUE_TEMPLATE.format(group_name=pseudo_rollout.option_group.name)
    think_start = completion.text.find("<think>")
    think_end = completion.text.find("</think>", think_start + len("<think>")) if think_start >= 0 else -1
    if think_start < 0 or think_end < 0:
        prefix_text = cue
        status = "no_complete_thinking_block"
    else:
        # The edit boundary need not coincide with an original sampled token.
        prefix_text = f"{completion.text[:think_end].rstrip()}\n{cue}"
        status = "complete_thinking_block"
    return _encode_group_response_prefix(prefix_text, tokenizer, source="generated_thinking", status=status)


def score_generated_thinking_group_pseudo_rollouts(
    inference_engine: Any,
    tokenizer: Any,
    pseudo_rollouts: Sequence[ProductOptionGroupPseudoRollout],
    completion_batches_by_page: Mapping[int, Sequence[SampledAgentResponse]],
    *,
    max_model_len: int,
    batch_size: int,
) -> tuple[tuple[tuple[ProductOptionGroupPseudoRolloutScores, ...], ...], tuple[tuple[GroupResponsePrefix, ...], ...]]:
    """Score every group once per ordinary completion using its sampled thought."""
    _validate_positive_integer(batch_size, "pseudo_score_batch_size")
    prefix_batches = []
    flat_rollouts = []
    flat_prefixes = []
    batch_lengths = []
    for pseudo_rollout in pseudo_rollouts:
        completions = completion_batches_by_page.get(pseudo_rollout.source_page_index)
        if completions is None:
            raise ValueError(f"No completion batch exists for source page {pseudo_rollout.source_page_index}")
        prefixes = tuple(extract_generated_group_thinking_prefix(completion, pseudo_rollout, tokenizer) for completion in completions)
        prefix_batches.append(prefixes)
        batch_lengths.append(len(prefixes))
        flat_rollouts.extend([pseudo_rollout] * len(prefixes))
        flat_prefixes.extend(prefix.token_ids for prefix in prefixes)

    flat_scores: list[ProductOptionGroupPseudoRolloutScores] = []
    for start in range(0, len(flat_rollouts), batch_size):
        scores = score_product_page_grouped_choice_rollouts(
            inference_engine,
            flat_rollouts[start : start + batch_size],
            max_model_len=max_model_len,
            assistant_response_prefix_token_ids=flat_prefixes[start : start + batch_size],
        )
        if any(score is None for score in scores):
            raise RuntimeError("A generated-thinking group pseudo-rollout exceeded the validated model context limit")
        flat_scores.extend(score for score in scores if score is not None)

    score_batches = []
    offset = 0
    for length in batch_lengths:
        next_offset = offset + length
        score_batches.append(tuple(flat_scores[offset:next_offset]))
        offset = next_offset
    if offset != len(flat_scores):
        raise RuntimeError("Generated-thinking group scores lost rollout alignment")
    return tuple(score_batches), tuple(prefix_batches)


def average_group_pseudo_rollout_scores(
    pseudo_rollout: ProductOptionGroupPseudoRollout,
    scores: Sequence[ProductOptionGroupPseudoRolloutScores],
) -> ProductOptionGroupPseudoRolloutScores:
    """Average aligned conditioned rows while retaining group metadata."""
    scores = tuple(scores)
    if any(score.source_page_index != pseudo_rollout.source_page_index or score.option_group != pseudo_rollout.option_group or score.none_is_correct != pseudo_rollout.none_is_correct for score in scores):
        raise ValueError("Every conditioned score must match the supplied group pseudo-rollout")
    averaged = average_pseudo_rollout_scores(scores)
    return ProductOptionGroupPseudoRolloutScores(
        choices=averaged.choices,
        source_page_index=pseudo_rollout.source_page_index,
        option_group=pseudo_rollout.option_group,
        none_is_correct=pseudo_rollout.none_is_correct,
    )


def project_group_completions(
    completions: Sequence[str | SampledAgentResponse],
    pseudo_rollout: ProductOptionGroupPseudoRollout,
    admissible_actions: Sequence[str],
    all_option_actions: Sequence[str],
    *,
    include_valid_non_option_actions_in_denominator: bool = False,
    projection: Callable[[list[str]], tuple[list[str], list[int]]] | None = None,
) -> EmpiricalGroupOptionSamples:
    """Project responses and construct one group's empirical denominator."""
    group_action_to_index = {action.lower(): index for index, action in enumerate(pseudo_rollout.actions) if action != GROUP_NONE_ACTION}
    all_option_action_set = {action.lower() for action in all_option_actions if action != GROUP_NONE_ACTION}
    admissible_action_set = {action.lower() for action in admissible_actions}
    if not isinstance(include_valid_non_option_actions_in_denominator, bool):
        raise ValueError("include_valid_non_option_actions_in_denominator must be a boolean")
    if not set(group_action_to_index).issubset(all_option_action_set) or not all_option_action_set.issubset(admissible_action_set):
        raise ValueError("Group and all-option actions must be subsets of the admissible actions")
    full_projection = project_completions(completions, admissible_actions, projection=projection)
    counts = [0] * len(pseudo_rollout.actions)
    group_selections = 0
    group_selections_with_invalid_format = 0
    valid_non_option_selections = 0
    different_group_selections = 0
    denominator_inclusions = []
    for action, format_valid in zip(full_projection.projected_actions, full_projection.format_valids):
        normalized_action = action.lower()
        group_index = group_action_to_index.get(normalized_action)
        if group_index is not None:
            counts[group_index] += 1
            group_selections += 1
            group_selections_with_invalid_format += int(not format_valid)
            included = True
        elif normalized_action in all_option_action_set:
            different_group_selections += 1
            included = False
        elif normalized_action in admissible_action_set:
            valid_non_option_selections += 1
            included = include_valid_non_option_actions_in_denominator
        else:
            included = False
        denominator_inclusions.append(included)

    probability_denominator = group_selections
    if include_valid_non_option_actions_in_denominator:
        probability_denominator += valid_non_option_selections
    return EmpiricalGroupOptionSamples(
        action_counts=tuple(counts),
        total_completions=full_projection.total_completions,
        format_valid_completions=full_projection.format_valid_completions,
        recognized_completions=group_selections,
        recognized_with_invalid_format=group_selections_with_invalid_format,
        invalid_format_completions=full_projection.invalid_format_completions,
        inadmissible_completions=full_projection.inadmissible_completions,
        invalid_examples=full_projection.invalid_examples,
        projected_actions=full_projection.projected_actions,
        format_valids=full_projection.format_valids,
        probability_denominator=probability_denominator,
        included_valid_non_option_completions=valid_non_option_selections if include_valid_non_option_actions_in_denominator else 0,
        different_group_option_completions=different_group_selections,
        denominator_inclusions=tuple(denominator_inclusions),
    )


def build_generated_thinking_records(
    pseudo_rollout: ProductOptionGroupPseudoRollout,
    scores: Sequence[ProductOptionGroupPseudoRolloutScores],
    prefixes: Sequence[GroupResponsePrefix],
    empirical: EmpiricalGroupOptionSamples,
) -> list[dict[str, Any]]:
    """Serialize conditioned probes without storing sampled chain-of-thought text."""
    if not len(scores) == len(prefixes) == empirical.total_completions == len(empirical.denominator_inclusions):
        raise ValueError("Conditioned scores, prefixes, projections, and denominator flags must align")
    action_to_label = {action.lower(): label for label, action in zip(pseudo_rollout.labels, pseudo_rollout.actions) if action != GROUP_NONE_ACTION}
    correct_labels = set(pseudo_rollout.correct_labels)
    records = []
    for sample_index, (score, prefix, projected_action, format_valid, included) in enumerate(zip(scores, prefixes, empirical.projected_actions, empirical.format_valids, empirical.denominator_inclusions)):
        projected_label = action_to_label.get(projected_action.lower())
        records.append(
            {
                "sample_index": sample_index,
                "thinking_status": prefix.status,
                "assistant_prefix_tokens": len(prefix.token_ids),
                "projected_action": projected_action,
                "projected_label": projected_label,
                "projected_option_is_correct": projected_label in correct_labels if projected_label is not None else None,
                "included_in_empirical_denominator": included,
                "format_valid": bool(format_valid),
                "pseudo_entropy_nats": score.entropy,
                "pseudo_correct_probability": score.correct_probability,
                "label_probabilities": score.label_probabilities,
            }
        )
    return records


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return center - half_width, center + half_width


def build_group_result(
    page: RenderedProductPage,
    pseudo_rollout: ProductOptionGroupPseudoRollout,
    scores: ProductOptionGroupPseudoRolloutScores,
    empirical: EmpiricalGroupOptionSamples,
    *,
    real_prompt_tokens: int,
    bootstrap_replicates: int,
    seed: int,
    response_prefix: GroupResponsePrefix | None = None,
    pseudo_probability_mode: str = "artificial_group_thinking_prefix",
    all_sample_scores: ProductOptionGroupPseudoRolloutScores | None = None,
    conditioned_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one serializable group-level pseudo/empirical comparison."""
    if pseudo_rollout.option_group != scores.option_group or pseudo_rollout.source_page_index != scores.source_page_index:
        raise ValueError("Pseudo-rollout scores do not match the supplied option group")
    if len(empirical.action_counts) != len(scores.choices):
        raise ValueError("Empirical counts and pseudo choices must align")

    none_record = None
    if pseudo_rollout.include_none:
        none_choice = scores.choices[-1]
        none_record = {"label": none_choice.label, "pseudo_probability": none_choice.probability, "is_correct": pseudo_rollout.none_is_correct, "empirical_probability": None, "reason": "A single-turn response cannot establish that a group will never be selected"}

        # A non-group action this turn does not mean the group is never selected.
        # Condition only the click comparison, and retain the full none mass.
        def condition_on_clicks(score):
            choices = score.choices[:-1]
            mass = math.fsum(choice.probability for choice in choices)
            if mass <= 0:
                raise ValueError("Cannot condition a grouped pseudo distribution with zero option-click mass")
            return replace(score, choices=tuple(replace(choice, probability=choice.probability / mass, log_probability=math.log(choice.probability / mass) if choice.probability > 0 else -math.inf) for choice in choices), none_is_correct=False)

        scores = condition_on_clicks(scores)
        if all_sample_scores is not None:
            all_sample_scores = condition_on_clicks(all_sample_scores)
        empirical = replace(empirical, action_counts=empirical.action_counts[:-1])
        pseudo_rollout = replace(pseudo_rollout, actions=pseudo_rollout.actions[:-1], labels=pseudo_rollout.labels[:-1], variant_token_ids=pseudo_rollout.variant_token_ids[:-1], include_none=False, none_is_correct=False)

    comparison = compare_action_distributions(
        [choice.probability for choice in scores.choices],
        empirical.action_counts,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    conditional_total = empirical.probability_denominator
    correct_options = set(pseudo_rollout.option_group.correct_options)
    all_sample_probabilities = None if all_sample_scores is None else all_sample_scores.label_probabilities
    option_rows = []
    for option, choice, count in zip(pseudo_rollout.option_group.values, scores.choices, empirical.action_counts):
        lower, upper = _wilson_interval(count, conditional_total)
        option_rows.append(
            {
                "label": choice.label,
                "option": option,
                "action": choice.action,
                "is_correct": option in correct_options,
                "pseudo_probability": choice.probability,
                **({"pseudo_probability_all_samples": all_sample_probabilities[choice.label]} if all_sample_probabilities is not None else {}),
                "empirical_count": count,
                "empirical_probability_conditional": count / conditional_total if conditional_total else None,
                "empirical_wilson_95_interval": [lower, upper],
            }
        )

    empirical_correct_count = sum(row["empirical_count"] for row in option_rows if row["is_correct"])
    total_completions = empirical.total_completions
    result = {
        "source_page_index": pseudo_rollout.source_page_index,
        "asin": page.asin,
        "category": page.category,
        "shopping_task": page.shopping_task,
        "group_name": pseudo_rollout.option_group.name,
        "correct_options": list(pseudo_rollout.option_group.correct_options),
        "option_count": len(option_rows),
        "real_prompt_tokens": real_prompt_tokens,
        "pseudo_prompt_tokens": len(pseudo_rollout.prompt_token_ids),
        "pseudo_entropy_nats": scores.entropy,
        "pseudo_probability_mode": pseudo_probability_mode,
        "pseudo_correct_probability": scores.correct_probability,
        **({"pseudo_correct_probability_all_samples": all_sample_scores.correct_probability} if all_sample_scores is not None else {}),
        "empirical_correct_probability_conditional": empirical_correct_count / conditional_total if conditional_total else None,
        "monte_carlo": {
            "total_page_completions": total_completions,
            "recognized_group_option_completions": empirical.recognized_completions,
            "recognized_group_option_rate": empirical.recognized_completions / total_completions if total_completions else None,
            "empirical_option_probability_denominator": conditional_total,
            "included_valid_non_option_completions": empirical.included_valid_non_option_completions,
            "different_group_option_completions": empirical.different_group_option_completions,
            "production_format_valid_completions": empirical.format_valid_completions,
            "production_format_valid_rate": empirical.format_valid_completions / total_completions if total_completions else None,
            "recognized_with_invalid_format": empirical.recognized_with_invalid_format,
            "invalid_format_completions": empirical.invalid_format_completions,
            "inadmissible_completions": empirical.inadmissible_completions,
            "invalid_examples": list(empirical.invalid_examples),
        },
        "comparison_conditional_on_group_option": comparison,
        "options": option_rows,
    }
    if none_record is not None:
        result["none_choice"] = none_record
        result["pseudo_click_conditioning"] = "option probabilities renormalized after excluding the none pseudo-choice"
    if response_prefix is not None:
        result["pseudo_response_prefix"] = {
            "source": response_prefix.source,
            "text": response_prefix.text,
            "token_count": len(response_prefix.token_ids),
        }
    if conditioned_records is not None:
        result["conditioned_pseudo_rollouts"] = list(conditioned_records)
        result["complete_thinking_fraction"] = sum(record["thinking_status"] == "complete_thinking_block" for record in conditioned_records) / len(conditioned_records) if conditioned_records else None
        result["denominator_conditioned_pseudo_probes"] = sum(record["included_in_empirical_denominator"] for record in conditioned_records)
    return result


def summarize_group_results(
    group_results: Sequence[dict[str, Any]],
    *,
    requested_pages: int,
    high_probability_threshold: float,
) -> dict[str, Any]:
    """Summarize agreement after aggregating all correct-option probability."""
    if not 0.5 < high_probability_threshold < 1.0:
        raise ValueError("high_probability_threshold must be greater than 0.5 and less than 1")
    evaluated = [group for group in group_results if group["empirical_correct_probability_conditional"] is not None]
    pseudo_high = [group for group in evaluated if group["pseudo_correct_probability"] > high_probability_threshold]
    empirical_high = [group for group in evaluated if group["empirical_correct_probability_conditional"] > high_probability_threshold]
    both_high = [group for group in evaluated if group["pseudo_correct_probability"] > high_probability_threshold and group["empirical_correct_probability_conditional"] > high_probability_threshold]
    union_high = [group for group in evaluated if group["pseudo_correct_probability"] > high_probability_threshold or group["empirical_correct_probability_conditional"] > high_probability_threshold]
    recognized = [group["monte_carlo"]["recognized_group_option_completions"] for group in group_results]
    return {
        "pages_requested": requested_pages,
        "pages_evaluated": len({group["source_page_index"] for group in group_results}),
        "groups_evaluated": len(group_results),
        "groups_with_no_recognized_option": sum(count == 0 for count in recognized),
        "total_recognized_group_option_selections": sum(recognized),
        "mean_recognized_option_selections_per_group": float(np.mean(recognized)) if recognized else None,
        "high_probability_threshold": high_probability_threshold,
        "pseudo_high_correct_groups": len(pseudo_high),
        "pseudo_high_correct_also_empirical_high": len(both_high),
        "pseudo_high_confirmation_rate": len(both_high) / len(pseudo_high) if pseudo_high else None,
        "empirical_high_correct_groups": len(empirical_high),
        "empirical_high_correct_also_pseudo_high": len(both_high),
        "empirical_high_confirmation_rate": len(both_high) / len(empirical_high) if empirical_high else None,
        "decisive_correct_group_union": len(union_high),
        "bidirectional_high_agreement": len(both_high) / len(union_high) if union_high else None,
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render a concise summary of aggregate correct-option probabilities."""
    summary = report["summary"]
    config = report["configuration"]

    def display(value: Any) -> str:
        return "n/a" if value is None else f"{value:.2%}" if isinstance(value, float) else str(value)

    threshold = summary["high_probability_threshold"]
    pseudo_mode = config.get("pseudo_probability_mode", "artificial_group_thinking_prefix")
    inclusive_denominator = bool(config.get("include_valid_non_option_actions_in_empirical_denominator", False))
    lines = [
        "# Grouped-choice pseudo-probability testbed",
        "",
        f"Model: `{config['model']}`  ",
        f"Pages: `{summary['pages_evaluated']}` of `{summary['pages_requested']}` requested  ",
        f"Multi-option groups: `{summary['groups_evaluated']}`  ",
        f"Ordinary completions per page: `{config['samples_per_page']}`  ",
        f"High-probability threshold: `>{threshold:g}`  ",
        f"Pseudo probability mode: `{pseudo_mode}`  ",
        f"Empirical denominator: `{'same-group options plus valid non-option actions' if inclusive_denominator else 'same-group options only'}`",
        "",
        "## Aggregate correct-option results",
        "",
        "| Direction | Confirmed high | Total high | Rate |",
        "| --- | ---: | ---: | ---: |",
        f"| Pseudo correct mass high implies empirical correct mass high | {summary['pseudo_high_correct_also_empirical_high']} | {summary['pseudo_high_correct_groups']} | {display(summary['pseudo_high_confirmation_rate'])} |",
        f"| Empirical correct mass high implies pseudo correct mass high | {summary['empirical_high_correct_also_pseudo_high']} | {summary['empirical_high_correct_groups']} | {display(summary['empirical_high_confirmation_rate'])} |",
        f"| Bidirectional union agreement | {summary['pseudo_high_correct_also_empirical_high']} | {summary['decisive_correct_group_union']} | {display(summary['bidirectional_high_agreement'])} |",
        "",
        "Each probability is first summed over every fuzzy-correct option in the group. Empirical probabilities are conditional on the ordinary "
        "product-page response selecting an option from the named group. Singleton groups are excluded. Groups whose aggregate correct mass is not high "
        "in either distribution do not enter the bidirectional headline metric.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_path: Path, markdown_path: Path | None) -> None:
    """Atomically write the JSON report and optional Markdown companion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_path.with_name(output_path.name + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary_json, output_path)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_markdown = markdown_path.with_name(markdown_path.name + ".tmp")
        temporary_markdown.write_text(format_markdown_summary(report))
        os.replace(temporary_markdown, markdown_path)


def _engine_model_limit(inference_engine: Any) -> int:
    model_config = getattr(getattr(inference_engine, "llm_engine", None), "model_config", None)
    model_limit = getattr(model_config, "max_model_len", None)
    if isinstance(model_limit, bool) or not isinstance(model_limit, int) or model_limit <= 0:
        raise ValueError("Could not discover max_model_len from the inference engine")
    return model_limit


def _validate_run_arguments(args: argparse.Namespace) -> None:
    for name in ("num_pages", "samples_per_page", "bootstrap_replicates", "max_new_tokens", "page_batch_size", "pseudo_score_batch_size", "tensor_parallel_size"):
        _validate_positive_integer(getattr(args, name), name)
    if args.max_model_len is not None:
        _validate_positive_integer(args.max_model_len, "max_model_len")
    if not math.isfinite(args.gpu_memory_utilization) or not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if not 0.5 < args.high_probability_threshold < 1.0:
        raise ValueError("high_probability_threshold must be greater than 0.5 and less than 1")


def _configure_vllm_engine() -> None:
    """Select V0, whose returned logprobs include the allowed-token mask."""
    if "vllm" in sys.modules:
        raise RuntimeError("The testbed must configure VLLM_USE_V1 before vLLM is imported; run it as `python -m treehca.pseudo_rollout_product_page_grouped_choices_testbed`")
    if os.environ.get("VLLM_USE_V1") == "1":
        logger.warning("Overriding VLLM_USE_V1=1 because vLLM V1 reports logprobs before applying allowed_token_ids")
    os.environ["VLLM_USE_V1"] = "0"


def run_testbed(args: argparse.Namespace) -> dict[str, Any]:
    """Run catalog sampling, pseudo scoring, ordinary sampling, and reporting."""
    _validate_run_arguments(args)
    _configure_vllm_engine()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    products, goals = load_webshop_products_and_goals(args.catalog, args.attributes, args.seed)
    selected = sample_diverse_product_goals(products, goals, args.num_pages, args.seed)
    pages = render_product_pages([(item.product, item.goal["instruction_text"]) for item in selected])
    goal_options_by_page = [_goal_option_mapping(item.goal) for item in selected]
    if any(goal_options is None for goal_options in goal_options_by_page):
        raise RuntimeError("A selected synthetic goal unexpectedly lacks goal_options")
    all_group_rollouts = prepare_product_page_grouped_choice_rollouts(
        [page.context_parts for page in pages],
        tokenizer,
        goal_options_by_page,
    )
    all_option_actions_by_page = {page_index: tuple(action for pseudo in all_group_rollouts if pseudo.source_page_index == page_index for action in pseudo.actions if action != GROUP_NONE_ACTION) for page_index in range(len(pages))}
    singleton_groups = [pseudo for pseudo in all_group_rollouts if len(pseudo.option_group.values) == 1]
    pseudo_rollouts = tuple(pseudo for pseudo in all_group_rollouts if len(pseudo.option_group.values) >= 2)
    if not pseudo_rollouts:
        raise ValueError("No sampled page produced a multi-option group")
    artificial_response_prefixes = build_artificial_group_response_prefixes(pseudo_rollouts, tokenizer)
    real_prompt_ids = tokenize_real_prompts(pages, tokenizer)
    required_logprobs = required_max_logprobs(pseudo_rollouts)

    engine_kwargs = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "max_logprobs": required_logprobs,
        "enable_prefix_caching": True,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    inference_engine = LLM(**engine_kwargs)
    model_limit = _engine_model_limit(inference_engine)

    real_eligible_pages = {index for index, token_ids in enumerate(real_prompt_ids) if len(token_ids) + args.max_new_tokens <= model_limit}
    eligible_pairs = tuple(
        (pseudo, response_prefix)
        for pseudo, response_prefix in zip(pseudo_rollouts, artificial_response_prefixes)
        if pseudo.source_page_index in real_eligible_pages and len(pseudo.prompt_token_ids) + (args.max_new_tokens + len(response_prefix.token_ids) if args.condition_pseudo_on_thinking else len(response_prefix.token_ids)) + 1 <= model_limit
    )
    eligible_rollouts = tuple(pseudo for pseudo, _ in eligible_pairs)
    eligible_response_prefixes = tuple(response_prefix for _, response_prefix in eligible_pairs)
    eligible_page_indices = sorted({pseudo.source_page_index for pseudo in eligible_rollouts})
    eligible_rollout_ids = {id(pseudo) for pseudo in eligible_rollouts}
    omitted_groups = [
        {
            "source_page_index": pseudo.source_page_index,
            "asin": pages[pseudo.source_page_index].asin,
            "group_name": pseudo.option_group.name,
            "reason": ("real prompt or worst-case generated-thinking pseudo prompt exceeds the model context limit" if args.condition_pseudo_on_thinking else "real prompt or response-prefixed pseudo prompt exceeds the model context limit"),
        }
        for pseudo in pseudo_rollouts
        if id(pseudo) not in eligible_rollout_ids
    ]
    if not eligible_rollouts:
        raise ValueError(f"No multi-option group fits max_model_len={model_limit} with max_new_tokens={args.max_new_tokens}")

    pseudo_scores = None
    if not args.condition_pseudo_on_thinking:
        logger.info("Scoring %d group-specific pseudo-rollout prompts", len(eligible_rollouts))
        pseudo_scores = score_product_page_grouped_choice_rollouts(
            inference_engine,
            eligible_rollouts,
            max_model_len=model_limit,
            assistant_response_prefix_token_ids=[response_prefix.token_ids for response_prefix in eligible_response_prefixes],
        )
        if any(score is None for score in pseudo_scores):
            raise RuntimeError("An eligible group pseudo-rollout unexpectedly produced no score")

    eligible_real_ids = tuple(real_prompt_ids[index] for index in eligible_page_indices)
    logger.info("Sampling %d ordinary completions for each of %d pages", args.samples_per_page, len(eligible_page_indices))
    completion_batches = sample_real_prompt_completions(
        inference_engine,
        tokenizer,
        eligible_real_ids,
        samples_per_page=args.samples_per_page,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        seed=args.seed + 10_000,
        page_batch_size=args.page_batch_size,
    )
    completions_by_page = dict(zip(eligible_page_indices, completion_batches))

    conditioned_score_batches = None
    generated_prefix_batches = None
    if args.condition_pseudo_on_thinking:
        total_probes = len(eligible_rollouts) * args.samples_per_page
        logger.info("Scoring %d group pseudo-rollouts conditioned on sampled thinking", total_probes)
        conditioned_score_batches, generated_prefix_batches = score_generated_thinking_group_pseudo_rollouts(
            inference_engine,
            tokenizer,
            eligible_rollouts,
            completions_by_page,
            max_model_len=model_limit,
            batch_size=args.pseudo_score_batch_size,
        )

    group_results = []
    for group_index, pseudo in enumerate(eligible_rollouts):
        page_index = pseudo.source_page_index
        empirical = project_group_completions(
            completions_by_page[page_index],
            pseudo,
            pages[page_index].context_parts.admissible_actions,
            all_option_actions_by_page[page_index],
            include_valid_non_option_actions_in_denominator=args.include_valid_non_option_actions_in_empirical_denominator,
        )
        result_kwargs = {}
        if args.condition_pseudo_on_thinking:
            assert conditioned_score_batches is not None and generated_prefix_batches is not None
            sample_scores = conditioned_score_batches[group_index]
            denominator_scores = [score for score, included in zip(sample_scores, empirical.denominator_inclusions) if included]
            score = average_group_pseudo_rollout_scores(pseudo, denominator_scores or sample_scores)
            all_sample_score = average_group_pseudo_rollout_scores(pseudo, sample_scores)
            conditioned_records = build_generated_thinking_records(pseudo, sample_scores, generated_prefix_batches[group_index], empirical)
            result_kwargs = {
                "pseudo_probability_mode": "mean_generated_thinking_for_empirical_denominator",
                "all_sample_scores": all_sample_score,
                "conditioned_records": conditioned_records,
            }
        else:
            assert pseudo_scores is not None
            score = pseudo_scores[group_index]
            assert score is not None
            result_kwargs = {"response_prefix": eligible_response_prefixes[group_index]}
        group_results.append(
            build_group_result(
                pages[page_index],
                pseudo,
                score,
                empirical,
                real_prompt_tokens=len(real_prompt_ids[page_index]),
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed + 100_000 + group_index,
                **result_kwargs,
            )
        )

    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "model": args.model,
            "tokenizer": tokenizer_name,
            "catalog": str(args.catalog),
            "attributes": str(args.attributes),
            "seed": args.seed,
            "num_pages": args.num_pages,
            "samples_per_page": args.samples_per_page,
            "bootstrap_replicates": args.bootstrap_replicates,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": model_limit,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "required_max_logprobs": required_logprobs,
            "pseudo_probability_mode": ("mean_generated_thinking_for_empirical_denominator" if args.condition_pseudo_on_thinking else "artificial_group_thinking_prefix"),
            "condition_pseudo_on_thinking": args.condition_pseudo_on_thinking,
            "pseudo_score_batch_size": args.pseudo_score_batch_size,
            "artificial_thinking_template": _ARTIFICIAL_THINKING_TEMPLATE,
            "high_probability_threshold": args.high_probability_threshold,
            "include_valid_non_option_actions_in_empirical_denominator": args.include_valid_non_option_actions_in_empirical_denominator,
            "history_omitted": True,
            "goal_source": "WebShop get_goals(..., human_goals=False)",
            "available_training_goals": len(goals),
            "vllm_engine_version": "V0",
            "empirical_conditioning": ("same-group options plus valid non-option actions; different-group options excluded" if args.include_valid_non_option_actions_in_empirical_denominator else "same-group option selections only"),
        },
        "summary": summarize_group_results(
            group_results,
            requested_pages=args.num_pages,
            high_probability_threshold=args.high_probability_threshold,
        ),
        "singleton_groups_excluded": len(singleton_groups),
        "omitted_groups": omitted_groups,
        "groups": group_results,
    }
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    write_report(report, args.output, markdown_output)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model name or local merged checkpoint accepted by vLLM")
    parser.add_argument("--tokenizer", help="Tokenizer name/path; defaults to --model")
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--attributes", type=Path, default=_DEFAULT_ATTRIBUTES)
    parser.add_argument("--num-pages", type=int, default=20)
    parser.add_argument("--samples-per-page", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--high-probability-threshold", type=float, default=_DEFAULT_HIGH_PROBABILITY_THRESHOLD)
    parser.add_argument(
        "--include-valid-non-option-actions-in-empirical-denominator",
        action="store_true",
        help="Count valid non-option actions in every group's empirical denominator while excluding options from other groups",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, help="Optional vLLM model context limit")
    parser.add_argument("--page-batch-size", type=int, default=8, help="Number of ordinary prompts submitted per vLLM call")
    parser.add_argument("--pseudo-score-batch-size", type=int, default=1024, help="Generated-thinking group pseudo prompts submitted per vLLM call")
    parser.add_argument(
        "--condition-pseudo-on-thinking",
        action="store_true",
        help="Score each group after the thinking sampled in every ordinary completion instead of using the artificial thinking prefix",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Full JSON report")
    parser.add_argument("--markdown-output", type=Path, help="Concise Markdown summary; defaults to the JSON path with a .md suffix")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argument_parser().parse_args()
    report = run_testbed(args)
    summary = report["summary"]
    print(f"Wrote {args.output} with {summary['groups_evaluated']} evaluated groups")
    print(f"Bidirectional high-probability agreement: {summary['bidirectional_high_agreement']}")


if __name__ == "__main__":
    main()
