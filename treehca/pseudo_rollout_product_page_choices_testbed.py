"""Empirically validate product-page pseudo-rollout action probabilities."""

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

from treehca.product_page_parser import ProductPageContextParts, extract_product_page_contexts, parse_product_page_fields
from treehca.pseudo_rollout_product_page import ActionChoiceScore, ProductPagePseudoRollout, ProductPagePseudoRolloutScores, prepare_product_page_pseudo_rollouts, required_max_logprobs, score_product_page_pseudo_rollouts

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_WEBSHOP_ROOT = _ROOT / "agent_system/environments/env_package/webshop/webshop"
_RESULTS_ROOT = _ROOT / "pseudo_prob_test_results"
_DEFAULT_CATALOG = _WEBSHOP_ROOT / "data/items_shuffle_1000.json"
_DEFAULT_ATTRIBUTES = _WEBSHOP_ROOT / "data/items_ins_v2_1000.json"
_DEFAULT_OUTPUT = _RESULTS_ROOT / "pseudo_rollout_calibration_report.json"
_THINKING_DECISION_CUE = "The best next action is:"


@dataclass(frozen=True)
class RenderedProductPage:
    """One real no-history WebShop prompt and its parsed components."""

    asin: str
    category: str
    shopping_task: str
    real_prompt: str
    context_parts: ProductPageContextParts


@dataclass(frozen=True)
class SampledAgentResponse:
    """One decoded real-prompt completion and its exact generated token IDs."""

    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ThinkingPrefix:
    """The edited assistant prefix retained for a conditioned pseudo probe."""

    token_ids: tuple[int, ...]
    text: str
    status: str


@dataclass(frozen=True)
class EmpiricalActionSamples:
    """Projected Monte Carlo completions for one product page."""

    action_counts: tuple[int, ...]
    total_completions: int
    format_valid_completions: int
    recognized_completions: int
    recognized_with_invalid_format: int
    invalid_format_completions: int
    inadmissible_completions: int
    invalid_examples: tuple[dict[str, str], ...]
    projected_actions: tuple[str, ...] = ()
    format_valids: tuple[int, ...] = ()


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _expected_action_count(product: dict[str, Any]) -> int:
    controls = ("back to search", "< prev", "description", "features", "buy now")
    commands = [f"click[{control}]" for control in controls]
    commands.extend(f"click[{value}]" for values in product.get("options", {}).values() for value in values)
    return len(dict.fromkeys(commands))


def _has_unique_option_fragments(product: dict[str, Any]) -> bool:
    fragments = [fragment for name, values in product.get("options", {}).items() for fragment in (name, *values)]
    return len(fragments) == len(set(fragments))


def sample_diverse_products(
    products: Sequence[dict[str, Any]],
    goals: Sequence[dict[str, Any]],
    num_pages: int,
    seed: int,
) -> tuple[tuple[dict[str, Any], str], ...]:
    """Randomly sample goal-bearing products while balancing action counts.

    A sampling cycle takes at most one item from each available action-count
    stratum. Within a stratum, it prefers a category not yet represented, then
    chooses one of the selected product's real generated training goals.
    """
    _validate_positive_integer(num_pages, "num_pages")
    rng = random.Random(seed)
    goals_by_asin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for goal in goals:
        instruction = goal.get("instruction_text")
        asin = goal.get("asin")
        if isinstance(asin, str) and isinstance(instruction, str) and instruction.strip():
            goals_by_asin[asin].append(goal)

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        if product.get("Title", "").strip() and goals_by_asin.get(product.get("asin")) and _has_unique_option_fragments(product):
            buckets[_expected_action_count(product)].append(product)
    candidate_count = sum(len(bucket) for bucket in buckets.values())
    if num_pages > candidate_count:
        raise ValueError(f"Requested {num_pages} pages, but only {candidate_count} parser-compatible products with generated training goals are available")

    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[tuple[dict[str, Any], str]] = []
    used_categories: set[str] = set()
    while len(selected) < num_pages:
        action_counts = [count for count, bucket in buckets.items() if bucket]
        rng.shuffle(action_counts)
        for action_count in action_counts:
            bucket = buckets[action_count]
            preferred_index = next((index for index, product in enumerate(bucket) if product.get("category", "") not in used_categories), len(bucket) - 1)
            product = bucket.pop(preferred_index)
            task = rng.choice(goals_by_asin[product["asin"]])["instruction_text"]
            selected.append((product, task))
            used_categories.add(product.get("category", ""))
            if len(selected) == num_pages:
                break
    return tuple(selected)


def load_webshop_products_and_goals(catalog_path: Path, attributes_path: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run WebShop's synthetic training-goal pipeline reproducibly."""
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        products, _, product_prices, _ = load_products(str(catalog_path), str(attributes_path), human_goals=False)
        goals = get_goals(products, product_prices, human_goals=False)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    return products, goals


def render_product_pages(selected: Sequence[tuple[dict[str, Any], str]]) -> tuple[RenderedProductPage, ...]:
    """Render selected items and build production no-history agent prompts."""
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    from web_agent_site.engine import engine
    from web_agent_site.envs import web_agent_text_env

    from agent_system.environments.env_manager import WebshopEnvironmentManager

    text_env = object.__new__(web_agent_text_env.WebAgentTextEnv)
    text_env.observation_mode = "text"
    manager = object.__new__(WebshopEnvironmentManager)
    manager.config = SimpleNamespace(env=SimpleNamespace(history_length=0))
    manager.tasks = [task for _, task in selected]
    raw_observations = []
    infos = []

    with web_agent_text_env.app.test_request_context("/"):
        for index, (product, task) in enumerate(selected):
            html = engine.map_action_to_html(
                "click",
                session_id=f"pseudo-rollout-testbed-{index}",
                product_info=product,
                keywords=[product.get("query", "catalog")],
                page=1,
                asin=product["asin"],
                options={},
                instruction_text=task,
                show_attrs=False,
            )
            text_env.browser = SimpleNamespace(page_source=html, current_url=f"/item_page/pseudo-rollout-testbed-{index}")
            text_env.instruction_text = task
            raw_observations.append(text_env.observation)
            infos.append({"available_actions": text_env.get_available_actions()})

    observations = manager.format_obs(raw_observations)
    real_prompts = manager.build_text_obs(observations, infos)
    parts = extract_product_page_contexts(real_prompts)
    rendered = []
    for (product, task), real_prompt, context_parts in zip(selected, real_prompts, parts):
        # Fail before model loading if a renderer/parser contract changed.
        parse_product_page_fields(context_parts.current_observation, context_parts.admissible_actions)
        rendered.append(
            RenderedProductPage(
                asin=product["asin"],
                category=product.get("category", ""),
                shopping_task=task,
                real_prompt=real_prompt,
                context_parts=context_parts,
            )
        )
    return tuple(rendered)


def tokenize_real_prompts(pages: Sequence[RenderedProductPage], tokenizer: Any) -> tuple[tuple[int, ...], ...]:
    """Apply the model's chat template to real WebShop prompts."""
    tokenized = []
    for page in pages:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": page.real_prompt}],
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
        tokenized.append(tuple(token_ids))
    return tuple(tokenized)


def sample_real_prompt_completions(
    inference_engine: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    samples_per_page: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    page_batch_size: int,
) -> tuple[tuple[SampledAgentResponse, ...], ...]:
    """Sample full agent responses, sharing each prompt prefill across ``n`` outputs."""
    _validate_positive_integer(samples_per_page, "samples_per_page")
    _validate_positive_integer(max_new_tokens, "max_new_tokens")
    _validate_positive_integer(page_batch_size, "page_batch_size")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    try:
        from vllm import SamplingParams
    except ImportError as error:
        raise RuntimeError("vLLM is required to sample real product-page prompts") from error

    all_completions: list[tuple[SampledAgentResponse, ...]] = []
    for batch_start in range(0, len(prompt_token_ids), page_batch_size):
        batch = prompt_token_ids[batch_start : batch_start + page_batch_size]
        params = [
            SamplingParams(
                n=samples_per_page,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=seed + batch_start + row,
                detokenize=False,
            )
            for row in range(len(batch))
        ]
        outputs = inference_engine.generate(
            prompts=[{"prompt_token_ids": list(ids)} for ids in batch],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != len(batch):
            raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(batch)} real prompts")
        for output in outputs:
            candidates = getattr(output, "outputs", None)
            if not isinstance(candidates, list) or len(candidates) != samples_per_page:
                actual = len(candidates) if isinstance(candidates, list) else None
                raise ValueError(f"Expected {samples_per_page} completions for a real prompt, got {actual}")
            candidate_token_ids = [tuple(int(token_id) for token_id in candidate.token_ids) for candidate in candidates]
            decoded = tokenizer.batch_decode(candidate_token_ids, skip_special_tokens=True)
            all_completions.append(tuple(SampledAgentResponse(text=text, token_ids=token_ids) for text, token_ids in zip(decoded, candidate_token_ids)))
    return tuple(all_completions)


def extract_thinking_prefix(completion: SampledAgentResponse, tokenizer: Any) -> ThinkingPrefix:
    """Replace the first closing think tag and everything after it with a cue.

    The returned prefix ends in ``The best next action is:`` so the constrained
    label is the next token. A malformed response uses only that cue, allowing
    every Monte Carlo sample to remain scoreable without retaining a partial
    thought or its action.
    """
    think_start = completion.text.find("<think>")
    think_end = completion.text.find("</think>", think_start + len("<think>")) if think_start >= 0 else -1
    if think_start < 0 or think_end < 0:
        prefix_text = _THINKING_DECISION_CUE
        status = "no_complete_thinking_block"
    else:
        # Re-tokenization is necessary because the source suffix is edited at a
        # text boundary that need not coincide with a sampled-token boundary.
        prefix_text = f"{completion.text[:think_end].rstrip()}\n{_THINKING_DECISION_CUE}"
        status = "complete_thinking_block"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if hasattr(prefix_ids, "tolist"):
        prefix_ids = prefix_ids.tolist()
    if not isinstance(prefix_ids, list) or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in prefix_ids):
        raise ValueError("Tokenizer must encode the thinking-conditioned assistant prefix as integer token IDs")
    return ThinkingPrefix(token_ids=tuple(prefix_ids), text=prefix_text, status=status)


def score_thinking_conditioned_pseudo_rollouts(
    inference_engine: Any,
    tokenizer: Any,
    pseudo_rollouts: Sequence[ProductPagePseudoRollout],
    completion_batches: Sequence[Sequence[SampledAgentResponse]],
    *,
    max_model_len: int,
    batch_size: int,
) -> tuple[tuple[tuple[ProductPagePseudoRolloutScores, ...], ...], tuple[tuple[ThinkingPrefix, ...], ...]]:
    """Score every real completion's pseudo prompt after its sampled thinking."""
    _validate_positive_integer(batch_size, "pseudo_score_batch_size")
    if len(pseudo_rollouts) != len(completion_batches):
        raise ValueError("Pseudo-rollout pages and completion batches must align")

    page_prefixes = tuple(tuple(extract_thinking_prefix(completion, tokenizer) for completion in completions) for completions in completion_batches)
    flat_rollouts = [pseudo for pseudo, completions in zip(pseudo_rollouts, completion_batches) for _ in completions]
    flat_prefixes = [prefix.token_ids for prefixes in page_prefixes for prefix in prefixes]
    flat_scores: list[ProductPagePseudoRolloutScores] = []
    for start in range(0, len(flat_rollouts), batch_size):
        batch_rollouts = flat_rollouts[start : start + batch_size]
        batch_prefixes = flat_prefixes[start : start + batch_size]
        batch_scores = score_product_page_pseudo_rollouts(
            inference_engine,
            batch_rollouts,
            max_model_len=max_model_len,
            assistant_response_prefix_token_ids=batch_prefixes,
        )
        if any(score is None for score in batch_scores):
            raise RuntimeError("A thinking-conditioned pseudo-rollout exceeded the validated model context limit")
        flat_scores.extend(score for score in batch_scores if score is not None)

    page_scores = []
    offset = 0
    for completions in completion_batches:
        next_offset = offset + len(completions)
        page_scores.append(tuple(flat_scores[offset:next_offset]))
        offset = next_offset
    if offset != len(flat_scores):
        raise RuntimeError("Thinking-conditioned pseudo-rollout scores lost page alignment")
    return tuple(page_scores), page_prefixes


def project_completions(
    completions: Sequence[str | SampledAgentResponse],
    admissible_actions: Sequence[str],
    *,
    projection: Callable[[list[str]], tuple[list[str], list[int]]] | None = None,
    max_invalid_examples: int = 5,
) -> EmpiricalActionSamples:
    """Apply the production projection and count actions the environment can take."""
    if projection is None:
        from agent_system.environments.env_package.webshop.projection import webshop_projection

        projection = webshop_projection
    completion_texts = [completion.text if isinstance(completion, SampledAgentResponse) else completion for completion in completions]
    projected, format_valids = projection(completion_texts.copy())
    if len(projected) != len(completions) or len(format_valids) != len(completions):
        raise ValueError("WebShop projection returned a misaligned result")

    action_to_index = {action.lower(): index for index, action in enumerate(admissible_actions)}
    counts = [0] * len(admissible_actions)
    recognized = 0
    recognized_with_invalid_format = 0
    invalid_format = 0
    inadmissible = 0
    invalid_examples = []
    for response, action, format_valid in zip(completion_texts, projected, format_valids):
        action_index = action_to_index.get(action.lower())
        if action_index is not None:
            counts[action_index] += 1
            recognized += 1
            if not format_valid:
                recognized_with_invalid_format += 1
        else:
            inadmissible += 1
        if not format_valid:
            invalid_format += 1
        if (not format_valid or action_index is None) and len(invalid_examples) < max_invalid_examples:
            invalid_examples.append({"response": response, "projected_action": action, "reason": "invalid_format" if not format_valid else "inadmissible_action"})
    return EmpiricalActionSamples(
        action_counts=tuple(counts),
        total_completions=len(completions),
        format_valid_completions=sum(bool(value) for value in format_valids),
        recognized_completions=recognized,
        recognized_with_invalid_format=recognized_with_invalid_format,
        invalid_format_completions=invalid_format,
        inadmissible_completions=inadmissible,
        invalid_examples=tuple(invalid_examples),
        projected_actions=tuple(projected),
        format_valids=tuple(int(value) for value in format_valids),
    )


def _likelihood_ratio_statistic(counts: np.ndarray, probabilities: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return math.nan
    observed = counts > 0
    if np.any(probabilities[observed] <= 0.0):
        return math.inf
    return float(2.0 * np.sum(counts[observed] * np.log(counts[observed] / (total * probabilities[observed]))))


def _jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0

    def kl_divergence(distribution: np.ndarray) -> float:
        nonzero = distribution > 0.0
        return float(np.sum(distribution[nonzero] * np.log(distribution[nonzero] / midpoint[nonzero])))

    return (kl_divergence(left) + kl_divergence(right)) / 2.0


def compare_action_distributions(
    expected_probabilities: Sequence[float],
    observed_counts: Sequence[int],
    *,
    bootstrap_replicates: int,
    seed: int,
    per_sample_expected_probabilities: Sequence[Sequence[float]] | None = None,
) -> dict[str, float | int | str | None]:
    """Compare one conditional empirical distribution to the pseudo distribution."""
    _validate_positive_integer(bootstrap_replicates, "bootstrap_replicates")
    expected = np.asarray(expected_probabilities, dtype=np.float64)
    counts = np.asarray(observed_counts, dtype=np.int64)
    if expected.ndim != 1 or counts.ndim != 1 or len(expected) == 0 or len(expected) != len(counts):
        raise ValueError("Expected probabilities and observed counts must be nonempty aligned vectors")
    if np.any(expected < 0.0) or not np.all(np.isfinite(expected)) or not math.isclose(float(expected.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Expected probabilities must be finite, nonnegative, and sum to one")
    if np.any(counts < 0):
        raise ValueError("Observed counts must be nonnegative")
    total = int(counts.sum())
    if total == 0:
        return {
            "conditional_sample_size": 0,
            "total_variation_distance": None,
            "jensen_shannon_divergence_nats": None,
            "multinomial_g_statistic": None,
            "parametric_bootstrap_p_value": None,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_null": "multinomial" if per_sample_expected_probabilities is None else "poisson_multinomial_from_conditioned_rows",
        }

    probability_rows = None
    if per_sample_expected_probabilities is not None:
        probability_rows = np.asarray(per_sample_expected_probabilities, dtype=np.float64)
        if probability_rows.shape != (total, len(expected)):
            raise ValueError(f"Per-sample expected probabilities must have shape {(total, len(expected))}, got {probability_rows.shape}")
        if np.any(probability_rows < 0.0) or not np.all(np.isfinite(probability_rows)) or not np.allclose(probability_rows.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("Every per-sample expected-probability row must be finite, nonnegative, and sum to one")
        row_mean = probability_rows.mean(axis=0)
        if not np.allclose(row_mean, expected, rtol=1e-8, atol=1e-10):
            raise ValueError("Expected probabilities must equal the mean of the per-sample rows")

    empirical = counts / total
    statistic = _likelihood_ratio_statistic(counts, expected)
    rng = np.random.default_rng(seed)
    if probability_rows is None:
        simulated_counts = rng.multinomial(total, expected, size=bootstrap_replicates)
        bootstrap_null = "multinomial"
    else:
        # Independent non-identical categorical draws form a Poisson multinomial
        # null; using multinomial(total, mean_probability) would overstate its
        # variance when the thought-conditioned distributions differ.
        simulated_counts = np.zeros((bootstrap_replicates, len(expected)), dtype=np.int64)
        replicate_indices = np.arange(bootstrap_replicates)
        for probability_row in probability_rows:
            sampled_actions = rng.choice(len(expected), size=bootstrap_replicates, p=probability_row)
            np.add.at(simulated_counts, (replicate_indices, sampled_actions), 1)
        bootstrap_null = "poisson_multinomial_from_conditioned_rows"
    expected_counts = total * expected
    ratios = np.ones_like(simulated_counts, dtype=np.float64)
    np.divide(simulated_counts, expected_counts, out=ratios, where=simulated_counts > 0)
    simulated_terms = np.zeros_like(ratios)
    positive = simulated_counts > 0
    simulated_terms[positive] = simulated_counts[positive] * np.log(ratios[positive])
    simulated_statistics = 2.0 * simulated_terms.sum(axis=1)
    p_value = float((1 + np.count_nonzero(simulated_statistics >= statistic)) / (bootstrap_replicates + 1))
    return {
        "conditional_sample_size": total,
        "total_variation_distance": float(np.abs(empirical - expected).sum() / 2.0),
        "jensen_shannon_divergence_nats": _jensen_shannon_divergence(empirical, expected),
        "multinomial_g_statistic": statistic if math.isfinite(statistic) else None,
        "parametric_bootstrap_p_value": p_value,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_null": bootstrap_null,
    }


def average_pseudo_rollout_scores(scores: Sequence[ProductPagePseudoRolloutScores]) -> ProductPagePseudoRolloutScores:
    """Average aligned conditional distributions into one marginal distribution."""
    scores = tuple(scores)
    if not scores:
        raise ValueError("Cannot average an empty collection of pseudo-rollout scores")
    reference = scores[0].choices
    for row, score in enumerate(scores[1:], start=1):
        if [(choice.label, choice.action) for choice in score.choices] != [(choice.label, choice.action) for choice in reference]:
            raise ValueError(f"Pseudo-rollout score row {row} does not align with the first row")
    probabilities = np.mean([[choice.probability for choice in score.choices] for score in scores], axis=0)
    choices = tuple(
        ActionChoiceScore(
            label=reference_choice.label,
            action=reference_choice.action,
            probability=float(probability),
            log_probability=math.log(float(probability)) if probability > 0.0 else -math.inf,
            # Variant-level logits are conditional on individual thoughts and
            # have no single aggregate equivalent.
            bare_log_probability=-math.inf,
            spaced_log_probability=-math.inf,
        )
        for reference_choice, probability in zip(reference, probabilities)
    )
    return ProductPagePseudoRolloutScores(choices=choices)


def build_conditioned_rollout_records(
    pseudo_rollout: ProductPagePseudoRollout,
    scores: Sequence[ProductPagePseudoRolloutScores],
    thinking_prefixes: Sequence[ThinkingPrefix],
    empirical: EmpiricalActionSamples,
) -> list[dict[str, Any]]:
    """Serialize per-completion conditional probabilities without bulky CoT text."""
    if not len(scores) == len(thinking_prefixes) == empirical.total_completions:
        raise ValueError("Conditioned scores, thinking prefixes, and empirical completions must align")
    if len(empirical.projected_actions) != empirical.total_completions or len(empirical.format_valids) != empirical.total_completions:
        raise ValueError("Empirical per-completion projection details are missing")
    action_to_label = {action.lower(): label for label, action in zip(pseudo_rollout.labels, pseudo_rollout.actions)}
    records = []
    for sample_index, (score, thinking, projected_action, format_valid) in enumerate(zip(scores, thinking_prefixes, empirical.projected_actions, empirical.format_valids)):
        projected_label = action_to_label.get(projected_action.lower())
        maximum_probability = max(choice.probability for choice in score.choices)
        top_labels = [choice.label for choice in score.choices if math.isclose(choice.probability, maximum_probability, rel_tol=0.0, abs_tol=max(1e-12, maximum_probability * 1e-12))]
        records.append(
            {
                "sample_index": sample_index,
                "thinking_status": thinking.status,
                "assistant_prefix_tokens": len(thinking.token_ids),
                "projected_action": projected_action,
                "projected_label": projected_label,
                "format_valid": bool(format_valid),
                "pseudo_entropy_nats": score.entropy,
                "label_probabilities": score.label_probabilities,
                "projected_action_probability": score.label_probabilities[projected_label] if projected_label is not None else None,
                "projected_action_is_pseudo_argmax": projected_label in top_labels if projected_label is not None else None,
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


def build_page_result(
    page: RenderedProductPage,
    pseudo_rollout: ProductPagePseudoRollout,
    scores: ProductPagePseudoRolloutScores,
    empirical: EmpiricalActionSamples,
    *,
    real_prompt_tokens: int,
    bootstrap_replicates: int,
    seed: int,
    pseudo_probability_mode: str = "single_no_thinking",
    all_sample_scores: ProductPagePseudoRolloutScores | None = None,
    conditioned_rollouts: Sequence[dict[str, Any]] | None = None,
    conditioned_comparison_scores: Sequence[ProductPagePseudoRolloutScores] | None = None,
) -> dict[str, Any]:
    """Build the serializable comparison for one aligned page."""
    probabilities = [choice.probability for choice in scores.choices]
    per_sample_expected = None
    if conditioned_comparison_scores is not None:
        per_sample_expected = [[choice.probability for choice in sample_score.choices] for sample_score in conditioned_comparison_scores]
    comparison = compare_action_distributions(
        probabilities,
        empirical.action_counts,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        per_sample_expected_probabilities=per_sample_expected,
    )
    conditional_total = empirical.recognized_completions
    action_rows = []
    all_sample_probabilities = None if all_sample_scores is None else all_sample_scores.label_probabilities
    for choice, count in zip(scores.choices, empirical.action_counts):
        lower, upper = _wilson_interval(count, conditional_total)
        action_rows.append(
            {
                "label": choice.label,
                "action": choice.action,
                "pseudo_probability": choice.probability,
                **({"pseudo_probability_all_samples": all_sample_probabilities[choice.label]} if all_sample_probabilities is not None else {}),
                "empirical_count": count,
                "empirical_probability_conditional": count / conditional_total if conditional_total else None,
                "empirical_wilson_95_interval": [lower, upper],
            }
        )
    total = empirical.total_completions
    result = {
        "asin": page.asin,
        "category": page.category,
        "shopping_task": page.shopping_task,
        "action_count": len(action_rows),
        "real_prompt_tokens": real_prompt_tokens,
        "pseudo_prompt_tokens": len(pseudo_rollout.prompt_token_ids),
        "pseudo_entropy_nats": scores.entropy,
        "pseudo_probability_mode": pseudo_probability_mode,
        "monte_carlo": {
            "total_completions": total,
            "format_valid_completions": empirical.format_valid_completions,
            "format_valid_rate": empirical.format_valid_completions / total if total else None,
            "recognized_action_completions": empirical.recognized_completions,
            "recognized_action_rate": empirical.recognized_completions / total if total else None,
            "recognized_with_invalid_format": empirical.recognized_with_invalid_format,
            "invalid_format_completions": empirical.invalid_format_completions,
            "inadmissible_completions": empirical.inadmissible_completions,
            "invalid_examples": list(empirical.invalid_examples),
        },
        "comparison_conditional_on_recognized_action": comparison,
        "actions": action_rows,
    }
    if conditioned_rollouts is not None:
        result["conditioned_pseudo_rollouts"] = list(conditioned_rollouts)
        result["mean_conditioned_pseudo_entropy_nats"] = float(np.mean([row["pseudo_entropy_nats"] for row in conditioned_rollouts]))
        result["complete_thinking_fraction"] = sum(row["thinking_status"] == "complete_thinking_block" for row in conditioned_rollouts) / len(conditioned_rollouts) if conditioned_rollouts else None
        recognized_records = [row for row in conditioned_rollouts if row["projected_label"] is not None]
        result["conditioned_top_action_agreement"] = sum(row["projected_action_is_pseudo_argmax"] for row in recognized_records) / len(recognized_records) if recognized_records else None
        result["mean_probability_assigned_to_projected_action"] = float(np.mean([row["projected_action_probability"] for row in recognized_records])) if recognized_records else None
    return result


def _benjamini_hochberg_rejections(p_values: Sequence[float], alpha: float) -> int:
    ordered = sorted(p_values)
    largest = 0
    for rank, p_value in enumerate(ordered, start=1):
        if p_value <= alpha * rank / len(ordered):
            largest = rank
    return largest


def summarize_page_results(page_results: Sequence[dict[str, Any]], requested_pages: int) -> dict[str, Any]:
    """Aggregate validity, effect sizes, and page-level goodness-of-fit tests."""
    comparisons = [page["comparison_conditional_on_recognized_action"] for page in page_results]
    tested = [comparison for comparison in comparisons if comparison["parametric_bootstrap_p_value"] is not None]
    p_values = [comparison["parametric_bootstrap_p_value"] for comparison in tested]
    total_completions = sum(page["monte_carlo"]["total_completions"] for page in page_results)
    total_recognized = sum(page["monte_carlo"]["recognized_action_completions"] for page in page_results)
    total_format_valid = sum(page["monte_carlo"]["format_valid_completions"] for page in page_results)
    tv_values = [comparison["total_variation_distance"] for comparison in tested]
    js_values = [comparison["jensen_shannon_divergence_nats"] for comparison in tested]
    action_counts = [page["action_count"] for page in page_results]
    summary = {
        "pages_requested": requested_pages,
        "pages_evaluated": len(page_results),
        "distinct_action_counts": sorted(set(action_counts)),
        "distinct_categories": len({page["category"] for page in page_results}),
        "total_completions": total_completions,
        "overall_format_valid_rate": total_format_valid / total_completions if total_completions else None,
        "overall_recognized_action_rate": total_recognized / total_completions if total_completions else None,
        "mean_page_total_variation_distance": float(np.mean(tv_values)) if tv_values else None,
        "median_page_total_variation_distance": float(np.median(tv_values)) if tv_values else None,
        "mean_page_jensen_shannon_divergence_nats": float(np.mean(js_values)) if js_values else None,
        "pages_with_no_recognized_actions": len(page_results) - len(tested),
        "goodness_of_fit_pages_tested": len(tested),
        "goodness_of_fit_rejections_at_0_05_uncorrected": sum(p_value < 0.05 for p_value in p_values),
        "goodness_of_fit_rejections_at_0_05_bh_fdr": _benjamini_hochberg_rejections(p_values, 0.05) if p_values else 0,
    }
    conditioned_records = [record for page in page_results for record in page.get("conditioned_pseudo_rollouts", [])]
    if conditioned_records:
        summary["conditioned_pseudo_probes"] = len(conditioned_records)
        summary["complete_thinking_fraction"] = sum(record["thinking_status"] == "complete_thinking_block" for record in conditioned_records) / len(conditioned_records)
        recognized_records = [record for record in conditioned_records if record["projected_label"] is not None]
        summary["conditioned_top_action_agreement"] = sum(record["projected_action_is_pseudo_argmax"] for record in recognized_records) / len(recognized_records) if recognized_records else None
        summary["mean_probability_assigned_to_projected_action"] = float(np.mean([record["projected_action_probability"] for record in recognized_records])) if recognized_records else None
    return summary


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render a concise human-readable companion to the full JSON report."""
    summary = report["summary"]
    config = report["configuration"]

    def display(value: Any, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    lines = [
        "# Pseudo-rollout action-probability testbed",
        "",
        f"Model: `{config['model']}`  ",
        f"Seed: `{config['seed']}`  ",
        f"Pages: `{summary['pages_evaluated']}` of `{summary['pages_requested']}` requested  ",
        f"Samples per page: `{config['samples_per_page']}`  ",
        f"Pseudo probability mode: `{'conditioned on each sampled thought' if config.get('condition_pseudo_on_thinking') else 'single no-thinking probe per page'}`  ",
        f"Temperature: `{config['temperature']}`",
        "",
        "## Aggregate results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Recognized-action rate | {display(summary['overall_recognized_action_rate'])} |",
        f"| Production-format-valid rate | {display(summary['overall_format_valid_rate'])} |",
        f"| Mean page total-variation distance | {display(summary['mean_page_total_variation_distance'])} |",
        f"| Median page total-variation distance | {display(summary['median_page_total_variation_distance'])} |",
        f"| Mean page Jensen-Shannon divergence (nats) | {display(summary['mean_page_jensen_shannon_divergence_nats'])} |",
        f"| GOF rejections at 0.05, uncorrected | {summary['goodness_of_fit_rejections_at_0_05_uncorrected']} / {summary['goodness_of_fit_pages_tested']} |",
        f"| GOF rejections at 0.05, BH-FDR | {summary['goodness_of_fit_rejections_at_0_05_bh_fdr']} / {summary['goodness_of_fit_pages_tested']} |",
        "",
        "## Per-page results",
        "",
        "| ASIN | Category | Actions | Recognized | TV | JS (nats) | Bootstrap p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if "conditioned_pseudo_probes" in summary:
        metric_insert = lines.index("", lines.index("| Metric | Value |"))
        lines[metric_insert:metric_insert] = [
            f"| Thinking-conditioned pseudo probes | {summary['conditioned_pseudo_probes']} |",
            f"| Complete-thinking fraction | {display(summary['complete_thinking_fraction'])} |",
            f"| Conditioned top-action agreement | {display(summary['conditioned_top_action_agreement'])} |",
            f"| Mean probability assigned to projected action | {display(summary['mean_probability_assigned_to_projected_action'])} |",
        ]
    for page in report["pages"]:
        monte_carlo = page["monte_carlo"]
        comparison = page["comparison_conditional_on_recognized_action"]
        escaped_category = page["category"].replace("|", "\\|")
        lines.append(
            f"| {page['asin']} | {escaped_category} | {page['action_count']} | "
            f"{monte_carlo['recognized_action_completions']}/{monte_carlo['total_completions']} | "
            f"{display(comparison['total_variation_distance'])} | {display(comparison['jensen_shannon_divergence_nats'])} | "
            f"{display(comparison['parametric_bootstrap_p_value'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "TV and Jensen-Shannon are effect sizes; smaller is closer. The bootstrap p-value tests whether recognized actions came from the pseudo distribution. "
            "It uses a multinomial null in the default mode and the individual conditional distributions' Poisson-multinomial null in thinking-conditioned mode. "
            "The BH-FDR count corrects the page-level tests for multiple comparisons.",
            "",
            "Comparisons are conditional on the completion mapping to an admissible action, matching the forced-choice pseudo distribution. "
            "The recognized-action and format-valid rates must therefore be considered alongside the conditional fit. Full action counts, "
            "probabilities, confidence intervals, and invalid examples are in the JSON report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_path: Path, markdown_path: Path | None) -> None:
    """Atomically write the full JSON report and optional Markdown summary."""
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


def _configure_vllm_engine() -> None:
    """Select V0, whose returned logprobs include the allowed-token mask."""
    if "vllm" in sys.modules:
        raise RuntimeError("The testbed must configure VLLM_USE_V1 before vLLM is imported; run it as `python -m treehca.pseudo_rollout_product_page_choices_testbed`")
    if os.environ.get("VLLM_USE_V1") == "1":
        logger.warning("Overriding VLLM_USE_V1=1 because vLLM 0.8.5 V1 reports logprobs before applying allowed_token_ids")
    os.environ["VLLM_USE_V1"] = "0"


def run_testbed(args: argparse.Namespace) -> dict[str, Any]:
    """Run catalog sampling, both inference paths, analysis, and report writing."""
    _validate_run_arguments(args)
    _configure_vllm_engine()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    products, goals = load_webshop_products_and_goals(args.catalog, args.attributes, args.seed)
    selected = sample_diverse_products(products, goals, args.num_pages, args.seed)
    pages = render_product_pages(selected)
    pseudo_rollouts = prepare_product_page_pseudo_rollouts([page.context_parts for page in pages], tokenizer)
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

    eligible_indices = [
        index
        for index, (real_ids, pseudo) in enumerate(zip(real_prompt_ids, pseudo_rollouts))
        if len(real_ids) + args.max_new_tokens <= model_limit and len(pseudo.prompt_token_ids) + 1 <= model_limit and (not args.condition_pseudo_on_thinking or len(pseudo.prompt_token_ids) + args.max_new_tokens + 1 <= model_limit)
    ]
    eligible_index_set = set(eligible_indices)
    omitted = [
        {
            "asin": pages[index].asin,
            "reason": "real, pseudo, or worst-case thinking-conditioned pseudo prompt exceeds the model context limit",
            "real_prompt_tokens": len(real_prompt_ids[index]),
            "pseudo_prompt_tokens": len(pseudo_rollouts[index].prompt_token_ids),
        }
        for index in range(len(pages))
        if index not in eligible_index_set
    ]
    if not eligible_indices:
        raise ValueError(f"No sampled page fits max_model_len={model_limit} with max_new_tokens={args.max_new_tokens}")

    pages = tuple(pages[index] for index in eligible_indices)
    pseudo_rollouts = tuple(pseudo_rollouts[index] for index in eligible_indices)
    real_prompt_ids = tuple(real_prompt_ids[index] for index in eligible_indices)
    pseudo_scores = None
    if not args.condition_pseudo_on_thinking:
        logger.info("Scoring %d pseudo-rollout prompts", len(pages))
        pseudo_scores = score_product_page_pseudo_rollouts(inference_engine, pseudo_rollouts, max_model_len=model_limit)
        if any(score is None for score in pseudo_scores):
            raise RuntimeError("An eligible pseudo-rollout unexpectedly produced no score")

    logger.info("Sampling %d real completions for each of %d pages", args.samples_per_page, len(pages))
    completion_batches = sample_real_prompt_completions(
        inference_engine,
        tokenizer,
        real_prompt_ids,
        samples_per_page=args.samples_per_page,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        seed=args.seed + 10_000,
        page_batch_size=args.page_batch_size,
    )
    conditioned_page_scores = None
    thinking_prefix_batches = None
    if args.condition_pseudo_on_thinking:
        total_probes = sum(len(completions) for completions in completion_batches)
        logger.info("Scoring %d pseudo-rollout prompts conditioned on sampled thinking", total_probes)
        conditioned_page_scores, thinking_prefix_batches = score_thinking_conditioned_pseudo_rollouts(
            inference_engine,
            tokenizer,
            pseudo_rollouts,
            completion_batches,
            max_model_len=model_limit,
            batch_size=args.pseudo_score_batch_size,
        )

    page_results = []
    for index, (page, pseudo, completions, prompt_ids) in enumerate(zip(pages, pseudo_rollouts, completion_batches, real_prompt_ids)):
        empirical = project_completions(completions, page.context_parts.admissible_actions)
        result_kwargs = {}
        if args.condition_pseudo_on_thinking:
            assert conditioned_page_scores is not None and thinking_prefix_batches is not None
            sample_scores = conditioned_page_scores[index]
            recognized_action_set = {action.lower() for action in page.context_parts.admissible_actions}
            recognized_scores = [score for score, action in zip(sample_scores, empirical.projected_actions) if action.lower() in recognized_action_set]
            score = average_pseudo_rollout_scores(recognized_scores or sample_scores)
            all_sample_score = average_pseudo_rollout_scores(sample_scores)
            conditioned_records = build_conditioned_rollout_records(pseudo, sample_scores, thinking_prefix_batches[index], empirical)
            result_kwargs = {
                "pseudo_probability_mode": "mean_conditioned_on_sampled_thinking_for_recognized_actions",
                "all_sample_scores": all_sample_score,
                "conditioned_rollouts": conditioned_records,
                "conditioned_comparison_scores": recognized_scores or None,
            }
        else:
            assert pseudo_scores is not None
            score = pseudo_scores[index]
            assert score is not None
        page_results.append(
            build_page_result(
                page,
                pseudo,
                score,
                empirical,
                real_prompt_tokens=len(prompt_ids),
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed + 100_000 + index,
                **result_kwargs,
            )
        )

    report = {
        "schema_version": 1,
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
            "condition_pseudo_on_thinking": args.condition_pseudo_on_thinking,
            "pseudo_score_batch_size": args.pseudo_score_batch_size,
            "history_omitted": True,
            "goal_source": "WebShop get_goals(..., human_goals=False)",
            "available_training_goals": len(goals),
            "vllm_engine_version": "V0",
            "catalog_sampling_strategy": "round_robin_action_count_then_prefer_new_category",
        },
        "summary": summarize_page_results(page_results, args.num_pages),
        "omitted_pages": omitted,
        "pages": page_results,
    }
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    write_report(report, args.output, markdown_output)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model name or local merged checkpoint accepted by vLLM")
    parser.add_argument("--tokenizer", help="Tokenizer name/path; defaults to --model")
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--attributes", type=Path, default=_DEFAULT_ATTRIBUTES)
    parser.add_argument("--num-pages", type=int, default=20)
    parser.add_argument("--samples-per-page", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, help="Optional vLLM model context limit")
    parser.add_argument("--page-batch-size", type=int, default=8, help="Number of distinct real prompts submitted per vLLM call")
    parser.add_argument("--pseudo-score-batch-size", type=int, default=1024, help="Conditioned pseudo prompts submitted per vLLM call")
    parser.add_argument(
        "--condition-pseudo-on-thinking",
        action="store_true",
        help="Score one pseudo prompt per Monte Carlo completion after replacing </think> with 'The best next action is:'",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Full JSON report")
    parser.add_argument("--markdown-output", type=Path, help="Concise Markdown summary; defaults to the JSON path with a .md suffix")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_argument_parser().parse_args()
    report = run_testbed(args)
    summary = report["summary"]
    logger.info(
        "Wrote %s pages; recognized-action rate=%.4f, mean TV=%s",
        summary["pages_evaluated"],
        summary["overall_recognized_action_rate"],
        summary["mean_page_total_variation_distance"],
    )
    logger.info("JSON report: %s", args.output)
    logger.info("Markdown summary: %s", args.markdown_output or args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
