"""Compare grouped-choice pseudo probabilities with real WebShop purchase outcomes.

See docs/treehca/pseudo_rollout_product_page_outcomes_testbed.md for the state
construction, training semantics, denominators, and representative prompts.
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

from treehca.product_page_parser import extract_product_page_contexts, parse_product_page_fields
from treehca.pseudo_rollout_product_page import GROUP_NONE_ACTION, prepare_product_page_grouped_choice_rollouts, required_max_logprobs, score_product_page_grouped_choice_rollouts
from treehca.pseudo_rollout_product_page_choices_testbed import _DEFAULT_ATTRIBUTES, _DEFAULT_CATALOG, _WEBSHOP_ROOT, _wilson_interval
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import (
    SelectedProductGoal,
    _configure_vllm_engine,
    _engine_model_limit,
    _validate_positive_integer,
    build_artificial_group_response_prefixes,
    sample_diverse_product_goals,
)

logger = logging.getLogger(__name__)
_DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "pseudo_prob_test_results/pseudo_rollout_product_page_outcomes_report.json"


def _webshop_modules():
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    from web_agent_site.engine import engine
    from web_agent_site.envs import web_agent_text_env

    return engine, web_agent_text_env


def create_server(catalog: Path, attributes: Path, seed: int, num_products: int):
    """Load the native server once; clones share only catalog/search resources."""
    _, native = _webshop_modules()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        return native.SimServer(seed, "http://127.0.0.1:3000", str(catalog), str(attributes), num_products=num_products, human_goals=False)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass
class ProductEpisode:
    """A native browser/environment plus the production prompt memory."""

    env: Any
    manager: Any
    prompt: str
    setup: dict[str, Any] = field(default_factory=dict)

    def clone(self):
        # Keep session IDs (and thus rendered observations) identical, but isolate
        # every mutable session, browser, and memory, including auto-reset state.
        server = copy.copy(self.env.server)
        server.user_sessions = copy.deepcopy(self.env.server.user_sessions)
        server.goals = copy.deepcopy(self.env.server.goals)
        # BeautifulSoup clickables form a deep cyclic DOM; the native step
        # reparses them from page_source, so discard that derived cache.
        memo = {id(self.env.server): server}
        if self.env.text_to_clickable is not None:
            memo[id(self.env.text_to_clickable)] = None
        return copy.deepcopy(self, memo)

    def advance(self, action: str) -> dict[str, Any]:
        session_id = self.env.session
        session = self.env.server.user_sessions[session_id]
        observation, reward, done, _ = self.env.step(action)
        # step() auto-resets on purchase. The old session still holds the receipt.
        purchased = bool(done and session["actions"].get("purchase", 0))
        result = {
            "raw_reward": float(reward),
            "done": bool(done),
            "purchased": purchased,
            "full_reward": bool(purchased and reward == 1.0),
            "training_reward": 10.0 if done and reward == 1.0 else 0.0,
            "selected_options": dict(session["options"]),
            "purchased_asin": session["asin"] if purchased else None,
            "reward_info": copy.deepcopy(session.get("verbose_info")) if purchased else None,
            "observation": observation,
        }
        self.manager.memory.store({"text_obs": self.manager.pre_text_obs, "action": [action]})
        self.manager.pre_text_obs = self.manager.format_obs([observation])
        if not done:
            self.prompt = self.manager.build_text_obs(self.manager.pre_text_obs, [{"available_actions": self.env.get_available_actions()}])[0]
        return result


def construct_start_episode(server: Any, selected: SelectedProductGoal, *, seed: int, history_length: int = 2, results_size: int = 10) -> ProductEpisode:
    """Construct a results page, then actually click its target product link."""
    engine, native = _webshop_modules()
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.memory import SimpleMemory

    # A small private server avoids copying the full goal list per MC sample.
    private_server = copy.copy(server)
    private_server.user_sessions = {}
    private_server.goals = [copy.deepcopy(selected.goal)]
    private_server.weights = [1]
    private_server.cum_weights = [0, 1]
    env = native.WebAgentTextEnv(observation_mode="text", server=private_server, seed=seed, session=0)
    initial_observation, _ = env.reset(session=0)
    private_server.user_sessions = {env.session: private_server.user_sessions[env.session]}
    session = private_server.user_sessions[env.session]
    query = selected.product.get("query") or selected.product["Title"]
    keywords = query.split()
    rng = random.Random(seed)
    others = [p for p in server.all_products if p["asin"] != selected.product["asin"]]
    # Prefer distractors in the same search category, with seeded ties.
    rng.shuffle(others)
    others.sort(key=lambda p: (p.get("query") != selected.product.get("query"), p.get("category") != selected.product.get("category")))
    results = [selected.product, *others[: results_size - 1]]
    rng.shuffle(results)
    session.update(keywords=keywords, page=1, asin=None, options={})
    session["actions"]["search"] += 1
    with native.app.test_request_context("/"):
        env.browser.page_source = engine.map_action_to_html("search", session_id=env.session, products=results, keywords=keywords, page=1, total=len(results), instruction_text=selected.goal["instruction_text"])
    env.browser.current_url = f"{env.base_url}/search_results/{env.session}/{'+'.join(keywords)}/1"
    results_observation = env.observation
    search_action = f"search[{query}]"
    env.prev_actions.append(search_action)
    env.prev_obs.append(results_observation)

    manager = WebshopEnvironmentManager(None, None, SimpleNamespace(env=SimpleNamespace(history_length=history_length)))
    manager.memory = SimpleMemory()
    manager.memory.reset(batch_size=1)
    manager.tasks = manager.extract_task([initial_observation])
    manager.memory.store({"text_obs": manager.format_obs([initial_observation]), "action": [search_action]})
    manager.pre_text_obs = manager.format_obs([results_observation])
    episode = ProductEpisode(env, manager, "")
    target_action = f"click[{selected.product['asin'].lower()}]"
    if target_action[6:-1] not in env.get_available_actions()["clickables"]:
        raise ValueError("Constructed results page does not expose the target product link")
    episode.advance(target_action)
    parts = extract_product_page_contexts([episode.prompt])[0]
    parse_product_page_fields(parts.current_observation, parts.admissible_actions)
    if parts.current_step != 3 or target_action not in (parts.history_block or ""):
        raise ValueError("Starting prompt lost the search-results selection history (possibly the training formatter's 13,000-character fallback)")
    if session["options"] or session["asin"] != selected.product["asin"]:
        raise ValueError("Starting state must be the target product with no selected options")
    episode.setup = {
        "search_query": query,
        "search_action": search_action,
        "selection_action": target_action,
        "results_asins": [product["asin"] for product in results],
        "initial_observation": initial_observation,
        "results_observation": results_observation,
        "product_observation": env.observation,
        "initial_selected_options": {},
        "real_prompt": episode.prompt,
    }
    return episode


def validate_full_reward(episode: ProductEpisode, goal_options: dict[str, str]) -> dict[str, Any]:
    """Use actual option clicks and a purchase to verify that full reward is reachable."""
    probe = episode.clone()
    for option in goal_options.values():
        probe.advance(f"click[{option}]")
    receipt = probe.advance("click[buy now]")
    return {key: receipt[key] for key in ("raw_reward", "full_reward", "selected_options", "reward_info")}


def exit_reason(env: Any, action: str) -> str | None:
    """Recognize transitions that would leave the product before executing them."""
    engine, _ = _webshop_modules()
    name, argument = engine.parse_action(action)
    # WebAgentTextEnv accepts nonempty search actions even without a search bar.
    if name == "search" and argument is not None and argument != "":
        return "search_action"
    if name != "click" or argument is None:
        return None
    argument = argument.lower()
    if argument not in env.get_available_actions()["clickables"]:
        return None
    if argument == engine.BACK_TO_SEARCH.lower():
        return "back_to_search"
    if argument == engine.PREV_PAGE.lower() and env.server.get_page_name(env.browser.current_url) == "item_page":
        return "back_to_results"
    return None


def collect_rollouts(
    start: ProductEpisode,
    sample_responses: Callable[[Sequence[str], Sequence[int]], Sequence[str]],
    *,
    samples_per_page: int,
    max_steps: int = 13,
    seed: int = 0,
    store_prompts: bool = False,
) -> list[dict[str, Any]]:
    """Sample one unconstrained response per live environment, then step it."""
    from agent_system.environments.env_package.webshop.projection import webshop_projection

    _validate_positive_integer(samples_per_page, "samples_per_page")
    _validate_positive_integer(max_steps, "max_steps")
    if max_steps > 13:
        raise ValueError("max_steps must be at most 13")
    episodes = [start.clone() for _ in range(samples_per_page)]
    records = [{"sample_index": index, "steps": [], "termination_reason": None, "purchased": False, "full_reward": False, "raw_reward": 0.0, "selected_options": {}} for index in range(samples_per_page)]
    active = list(range(samples_per_page))
    for step in range(max_steps):
        # Seeds belong to trajectory/step, so batching and early exits don't shift them.
        seeds = [seed + index * max_steps + step for index in active]
        responses = list(sample_responses([episodes[index].prompt for index in active], seeds))
        if len(responses) != len(active):
            raise ValueError("Sampler returned the wrong number of live-trajectory responses")
        actions, valids = webshop_projection(responses.copy())
        next_active = []
        for index, action, valid, step_seed in zip(active, actions, valids, seeds):
            episode, record = episodes[index], records[index]
            reason = exit_reason(episode.env, action)
            step_record = {"continuation_step": step + 1, "training_step": step + 3, "seed": step_seed, "projected_action": action, "format_valid": bool(valid), "executed": reason is None}
            if store_prompts:
                step_record["prompt"] = episode.prompt
            if reason is not None:
                record["termination_reason"] = reason
                record["selected_options"] = dict(episode.env.server.user_sessions[episode.env.session]["options"])
            else:
                receipt = episode.advance(action)
                step_record.update(raw_reward=receipt["raw_reward"], selected_options=receipt["selected_options"], observation=receipt["observation"])
                record.update({key: receipt[key] for key in ("purchased", "full_reward", "raw_reward", "selected_options")})
                if receipt["done"]:
                    record.update(termination_reason="purchase" if receipt["purchased"] else "environment_done", purchased_asin=receipt["purchased_asin"], reward_info=receipt["reward_info"])
                elif step + 1 == max_steps:
                    record["termination_reason"] = "step_limit"
                else:
                    next_active.append(index)
            record["steps"].append(step_record)
        active = next_active
        if not active:
            break
    return records


def compute_pseudo_rewards(start: ProductEpisode, groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Integrate the native purchase reward over independent option combinations."""
    engine, native = _webshop_modules()
    session = start.env.server.user_sessions[start.env.session]
    item = start.env.server.product_item_dict[session["asin"]]
    names = [group["group_name"].lower() for group in groups]
    if len(set(names)) != len(names) or set(names) != {name.lower() for name in item["options"]}:
        raise ValueError("Pseudo reward integration requires every product option group exactly once")
    choices = []
    for name, group in zip(names, groups):
        values = []
        for choice in group["options"]:
            action, value = engine.parse_action(choice["action"])
            probability = float(choice["probability"])
            is_none = choice["action"] == GROUP_NONE_ACTION
            if (not is_none and (action != "click" or value is None)) or not math.isfinite(probability) or probability < 0:
                raise ValueError("Expected option clicks with finite nonnegative pseudo probabilities")
            values.append((None if is_none else value.lower(), probability))
        expected_values = {value.lower() for key, options in item["options"].items() if key.lower() == name for value in options}
        if any(value is None for value, _ in values):
            expected_values.add(None)
        if len({value for value, _ in values}) != len(values) or {value for value, _ in values} != expected_values:
            raise ValueError("Pseudo reward integration requires every option value exactly once")
        if not math.isclose(math.fsum(probability for _, probability in values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Each pseudo option distribution must sum to one")
        choices.append(values)

    mass_by_reward, counts_by_reward = Counter(), Counter()
    for combination in product(*choices):
        options = {name: value for name, (value, _) in zip(names, combination) if value is not None}
        probability = math.prod(probability for _, probability in combination)
        # Call the very same function as SimServer.done, including set-wise fuzzy
        # matching and attribute/price/type components; no reward approximation.
        reward = float(native.get_reward(item, session["goal"], price=start.env.server.product_prices[session["asin"]], options=options))
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Native purchase reward must be finite and in [0, 1]")
        mass_by_reward[reward] += probability
        counts_by_reward[reward] += 1
    total_mass = math.fsum(mass_by_reward.values())
    distribution = [{"raw_reward": reward, "probability": mass / total_mass, "combination_count": counts_by_reward[reward]} for reward, mass in sorted(mass_by_reward.items())]
    return {
        "combination_count": sum(counts_by_reward.values()),
        "reward_distribution": distribution,
        "full_reward_probability": math.fsum(row["probability"] for row in distribution if row["raw_reward"] == 1.0),
        "positive_reward_probability": math.fsum(row["probability"] for row in distribution if row["raw_reward"] > 0),
        "expected_reward": math.fsum(row["probability"] * row["raw_reward"] for row in distribution),
    }


def summarize_outcomes(
    trajectories: Sequence[dict[str, Any]],
    pseudo_probability: float | None,
    *,
    count_partial_reward: bool = False,
    pseudo_rewards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain both unconditional and purchase-conditioned denominators."""
    total = len(trajectories)
    purchased = sum(record["purchased"] for record in trajectories)
    full = sum(record["purchased"] and record["raw_reward"] == 1.0 for record in trajectories)
    positive = sum(record["purchased"] and record["raw_reward"] > 0 for record in trajectories)
    reward_sum = math.fsum(record["raw_reward"] for record in trajectories if record["purchased"]) if count_partial_reward else full
    counted = positive if count_partial_reward else full
    selected_pseudo = pseudo_rewards["positive_reward_probability" if count_partial_reward else "full_reward_probability"] if pseudo_rewards is not None else (None if count_partial_reward else pseudo_probability)
    pseudo_expected = pseudo_rewards["expected_reward"] if count_partial_reward and pseudo_rewards is not None else selected_pseudo
    result = {
        "trajectories": total,
        "no_purchase_count": total - purchased,
        "purchase_count": purchased,
        "partial_reward_purchase_count": purchased - full,
        "full_reward_purchase_count": full,
        "termination_counts": dict(Counter(record["termination_reason"] for record in trajectories)),
        "pseudo_all_correct_probability": pseudo_probability,
        "positive_reward_purchase_count": positive,
        "positive_partial_reward_purchase_count": positive - full,
        "zero_reward_purchase_count": purchased - positive,
        "count_partial_reward": count_partial_reward,
        "reward_event": "positive_native_reward" if count_partial_reward else "full_native_reward",
        "reward_scale": "native_fractional" if count_partial_reward else "training_binary_0_1",
        "reward_purchase_count": counted,
        "pseudo_reward_probability": selected_pseudo,
        "empirical_expected_reward_all": reward_sum / total if total else None,
        "empirical_expected_reward_given_purchase": reward_sum / purchased if purchased else None,
        "pseudo_expected_reward": pseudo_expected,
    }
    # Count sampled continuation turns, excluding the two setup actions.
    first_turn_counts = Counter(record["termination_reason"] for record in trajectories if len(record.get("steps", [])) == 1)
    for name, count in (
        ("first_turn_back_to_results", first_turn_counts["back_to_results"]),
        ("first_turn_other_search_exit", first_turn_counts["back_to_search"] + first_turn_counts["search_action"]),
        ("first_turn_purchase", first_turn_counts["purchase"]),
    ):
        result[f"{name}_count"] = count
        result[f"{name}_rate"] = count / total if total else None
    for name, count, denominator in (
        ("no_purchase", total - purchased, total),
        ("partial_reward_purchase", purchased - full, total),
        ("full_reward_purchase", full, total),
        ("full_reward_given_purchase", full, purchased),
        ("positive_reward_purchase", positive, total),
        ("positive_reward_given_purchase", positive, purchased),
        ("reward_purchase", counted, total),
        ("reward_given_purchase", counted, purchased),
    ):
        result[f"{name}_rate"] = count / denominator if denominator else None
        result[f"{name}_wilson_95"] = list(_wilson_interval(count, denominator))
    for name in ("full_reward_purchase", "full_reward_given_purchase"):
        rate = result[f"{name}_rate"]
        full_pseudo = pseudo_rewards["full_reward_probability"] if pseudo_rewards is not None else pseudo_probability
        result[f"pseudo_minus_{name}_rate"] = full_pseudo - rate if full_pseudo is not None and rate is not None else None
    for name in ("reward_purchase", "reward_given_purchase"):
        rate = result[f"{name}_rate"]
        result[f"pseudo_minus_{name}_rate"] = selected_pseudo - rate if selected_pseudo is not None and rate is not None else None
    for name in ("all", "given_purchase"):
        empirical_expected = result[f"empirical_expected_reward_{name}"]
        result[f"pseudo_minus_expected_reward_{name}"] = pseudo_expected - empirical_expected if pseudo_expected is not None and empirical_expected is not None else None
    option_sets = Counter((record["purchased"], record["full_reward"], record["raw_reward"], tuple(sorted(record["selected_options"].items()))) for record in trajectories)
    result["final_option_sets"] = [{"purchased": bought, "full_reward": won, "raw_reward": reward, "selected_options": dict(options), "count": count} for (bought, won, reward, options), count in option_sets.items()]
    return result


class VllmPolicy:
    """The training chat template, token budget, and unconstrained sampling."""

    def __init__(self, engine: Any, tokenizer: Any, args: argparse.Namespace, model_limit: int):
        self.engine, self.tokenizer, self.args, self.model_limit = engine, tokenizer, args, model_limit

    def __call__(self, prompts: Sequence[str], seeds: Sequence[int]) -> list[str]:
        from vllm import SamplingParams

        from verl.utils.torch_functional import tokenize_and_postprocess_data

        responses = []
        for start in range(0, len(prompts), self.args.rollout_batch_size):
            batch = prompts[start : start + self.args.rollout_batch_size]
            token_rows = []
            for prompt in batch:
                chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                ids, mask = tokenize_and_postprocess_data(prompt=chat, tokenizer=self.tokenizer, max_length=self.args.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation="error")
                token_ids = ids[0][mask[0].bool()].tolist()
                if len(token_ids) + self.args.max_new_tokens > self.model_limit:
                    raise ValueError("A rollout prompt plus response exceeds max_model_len; increase the context budget")
                token_rows.append({"prompt_token_ids": token_ids})
            params = [
                SamplingParams(n=1, max_tokens=self.args.max_new_tokens, temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, seed=seed, detokenize=False)
                for seed in seeds[start : start + len(batch)]
            ]
            outputs = self.engine.generate(prompts=token_rows, sampling_params=params, use_tqdm=False)
            if len(outputs) != len(batch) or any(len(output.outputs) != 1 for output in outputs):
                raise ValueError("vLLM returned misaligned continuation responses")
            responses.extend(self.tokenizer.batch_decode([output.outputs[0].token_ids for output in outputs], skip_special_tokens=True))
        return responses


def score_start_pages(engine: Any, tokenizer: Any, pseudo_rollouts: Sequence[Any], *, model_limit: int) -> list[dict[str, Any]]:
    prefixes = build_artificial_group_response_prefixes(pseudo_rollouts, tokenizer)
    scores = score_product_page_grouped_choice_rollouts(engine, pseudo_rollouts, max_model_len=model_limit, assistant_response_prefix_token_ids=[prefix.token_ids for prefix in prefixes])
    if len(scores) != len(pseudo_rollouts) or any(score is None for score in scores):
        raise ValueError("Every starting-page option group must fit the pseudo context budget")
    return [
        {
            "source_page_index": pseudo.source_page_index,
            "group_name": pseudo.option_group.name,
            "correct_options": list(pseudo.option_group.correct_options),
            "none_is_correct": pseudo.none_is_correct,
            "correct_probability": score.correct_probability,
            "prompt": pseudo.prompt,
            "assistant_response_prefix": prefix.text,
            "prompt_token_count": len(pseudo.prompt_token_ids),
            "options": [{"label": choice.label, "action": choice.action, "is_none": choice.action == GROUP_NONE_ACTION, "probability": choice.probability} for choice in score.choices],
        }
        for pseudo, prefix, score in zip(pseudo_rollouts, prefixes, scores)
    ]


def format_markdown_summary(report: dict[str, Any]) -> str:
    def rate(value):
        return "n/a" if value is None else f"{value:.3%}"

    partial = report["configuration"].get("count_partial_reward", False)
    reward_label = "Full or partial reward" if partial else "Full reward"
    pseudo_label = "Pseudo positive reward" if partial else "Pseudo full reward"
    explanation = (
        "The pseudo positive-reward probability sums independent option-combination probabilities whose native purchase reward is greater than zero." if partial else "The pseudo full-reward probability sums independent option-combination probabilities whose native purchase reward is exactly one."
    )
    lines = [
        "# Product-page continuation outcome testbed",
        "",
        f"Mode: `{report['mode']}`. Model: `{report['configuration']['model']}`.",
        "",
        explanation + " Pseudo estimates assume a purchase and independent group choices; they do not predict purchase/exit decisions.",
        "",
        f"| ASIN | Rollouts | No purchase | {reward_label} / all | {reward_label} / purchases | {pseudo_label} |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for page in report["pages"]:
        result = page.get("outcomes", {})
        lines.append(f"| {page['asin']} | {result.get('trajectories', 0)} | {rate(result.get('no_purchase_rate'))} | {rate(result.get('reward_purchase_rate'))} | {rate(result.get('reward_given_purchase_rate'))} | {rate(result.get('pseudo_reward_probability'))} |")
    lines.extend(
        [
            "",
            ("Expected native fractional rewards (0–1)." if partial else "Expected training-style binary rewards (0 or 1); partial purchases contribute zero.") + " Non-purchases contribute zero to the all-rollout mean.",
            "",
            "| ASIN | Empirical E[R] / all | Empirical E[R] / purchases | Pseudo E[R] | Pseudo minus all | Pseudo minus purchases |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for page in report["pages"]:
        result = page.get("outcomes", {})
        values = [result.get(key) for key in ("empirical_expected_reward_all", "empirical_expected_reward_given_purchase", "pseudo_expected_reward", "pseudo_minus_expected_reward_all", "pseudo_minus_expected_reward_given_purchase")]
        lines.append(f"| {page['asin']} | " + " | ".join("n/a" if value is None else f"{value:.6f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "First-turn terminations: counts and percentages of all rollouts for each ASIN. "
            "The first turn is the first sampled action after the product-page setup (training step 3). "
            "Immediate purchases count regardless of reward. Other search exits mean returning to the search bar or issuing a new search.",
            "",
            "| ASIN | Return to results on first turn | Other search exit on first turn | Buy immediately |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for page in report["pages"]:
        result = page.get("outcomes", {})
        cells = [f"{result[name + '_count']} ({rate(result[name + '_rate'])})" if name + "_count" in result else "n/a" for name in ("first_turn_back_to_results", "first_turn_other_search_exit", "first_turn_purchase")]
        lines.append(f"| {page['asin']} | " + " | ".join(cells) + " |")
    lines.extend(["", "JSON includes exact prompts, oracle receipts, actions, final option sets, termination reasons, Wilson 95% rate intervals, native pseudo reward distributions, and probability/expected-reward gaps.", ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"), (output.with_suffix(".md"), format_markdown_summary(report))):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content)
        os.replace(temporary, path)


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in ("num_pages", "samples_per_page", "max_steps", "history_length", "results_size", "num_products", "max_prompt_length", "max_new_tokens", "rollout_batch_size", "tensor_parallel_size"):
        _validate_positive_integer(getattr(args, name), name)
    if args.max_steps > 13:
        raise ValueError("max_steps must be at most 13")
    if not 2 <= args.results_size <= 10:
        raise ValueError("results_size must be between 2 and 10")
    if args.max_model_len is not None:
        _validate_positive_integer(args.max_model_len, "max_model_len")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite for Monte Carlo sampling")
    if not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("top_p and gpu_memory_utilization must be in (0, 1]")
    if args.top_k != -1 and args.top_k < 1:
        raise ValueError("top_k must be -1 or positive")


def run_testbed(args: argparse.Namespace) -> dict[str, Any]:
    _validate_arguments(args)
    server = create_server(args.catalog, args.attributes, args.seed, args.num_products)
    selected = sample_diverse_product_goals(server.all_products, server.goals, args.num_pages, args.seed)
    starts, pages = [], []
    for index, item in enumerate(selected):
        start = construct_start_episode(server, item, seed=args.seed + index, history_length=args.history_length, results_size=args.results_size)
        oracle = validate_full_reward(start, item.goal["goal_options"])
        if not oracle["full_reward"]:
            raise ValueError(f"ASIN {item.product['asin']} does not attain full native reward with the goal options: {oracle}. No environment reward rules were changed.")
        starts.append(start)
        pages.append({"source_page_index": index, "asin": item.product["asin"], "category": item.product.get("category"), "goal": item.goal, "price": server.product_prices[item.product["asin"]], "setup": start.setup, "oracle": oracle})
        logger.info("Validated start %d/%d: %s", index + 1, args.num_pages, item.product["asin"])
    report = {
        "schema_version": 2,
        "status": "running",
        "mode": "validate_only" if args.validate_only else "monte_carlo",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sampling": {"goal_source": "native synthetic WebShop goals, all generated goals as in the grouped testbed", "available_goals": len(server.goals), "setup": "constructed results set followed by a native product click"},
        "pages": pages,
    }
    if args.validate_only:
        report["status"] = "complete"
        write_report(report, args.output)
        return report

    _configure_vllm_engine()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    parts = extract_product_page_contexts([start.prompt for start in starts])
    pseudo_rollouts = prepare_product_page_grouped_choice_rollouts(parts, tokenizer, [item.goal["goal_options"] for item in selected])
    # Singleton groups are necessary here: their selection can matter for reward.
    if {pseudo.source_page_index for pseudo in pseudo_rollouts} != set(range(len(starts))):
        raise ValueError("Every sampled page must contribute option groups")
    kwargs = dict(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        max_logprobs=required_max_logprobs(pseudo_rollouts),
        enable_prefix_caching=True,
    )
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    engine = LLM(**kwargs)
    model_limit = _engine_model_limit(engine)
    report["configuration"].update(effective_max_model_len=model_limit, vllm_engine_version="V0", pseudo_probability_mode="artificial_group_thinking_prefix", truncation="error")
    group_scores = score_start_pages(engine, tokenizer, pseudo_rollouts, model_limit=model_limit)
    policy = VllmPolicy(engine, tokenizer, args, model_limit)
    for index, (start, page) in enumerate(zip(starts, pages)):
        groups = [group for group in group_scores if group["source_page_index"] == index]
        pseudo_probability = math.prod(group["correct_probability"] for group in groups)
        pseudo_rewards = compute_pseudo_rewards(start, groups)
        trajectories = collect_rollouts(start, policy, samples_per_page=args.samples_per_page, max_steps=args.max_steps, seed=args.seed + 10_000 + index * args.samples_per_page * args.max_steps, store_prompts=args.store_trajectory_prompts)
        page.update(groups=groups, pseudo_rewards=pseudo_rewards, trajectories=trajectories, outcomes=summarize_outcomes(trajectories, pseudo_probability, count_partial_reward=args.count_partial_reward, pseudo_rewards=pseudo_rewards))
        report["pages_completed"] = index + 1
        # Checkpoint complete pages so a later inference error doesn't lose them.
        write_report(report, args.output)
        logger.info("Page %d/%d: %s", index + 1, len(pages), page["outcomes"] | {"final_option_sets": "see JSON"})
    all_trajectories = [trajectory for page in pages for trajectory in page["trajectories"]]
    mean_pseudo = float(np.mean([page["outcomes"]["pseudo_all_correct_probability"] for page in pages]))
    # Pooled purchases weight pages differently from the equally sampled starts.
    # Keep calibration gaps per page instead of comparing unlike aggregates.
    report["summary"] = summarize_outcomes(all_trajectories, None, count_partial_reward=args.count_partial_reward)
    report["summary"]["page_mean_pseudo_all_correct_probability"] = mean_pseudo
    report["summary"]["page_mean_pseudo_reward_probability"] = float(np.mean([page["outcomes"]["pseudo_reward_probability"] for page in pages]))
    report["summary"]["page_mean_pseudo_expected_reward"] = float(np.mean([page["outcomes"]["pseudo_expected_reward"] for page in pages]))
    report["summary"]["page_mean_absolute_unconditional_gap"] = float(np.mean([abs(page["outcomes"]["pseudo_minus_reward_purchase_rate"]) for page in pages]))
    conditional_gaps = [abs(page["outcomes"]["pseudo_minus_reward_given_purchase_rate"]) for page in pages if page["outcomes"]["pseudo_minus_reward_given_purchase_rate"] is not None]
    report["summary"]["page_mean_absolute_purchase_conditional_gap"] = float(np.mean(conditional_gaps)) if conditional_gaps else None
    for name in ("all", "given_purchase"):
        gaps = [abs(page["outcomes"][f"pseudo_minus_expected_reward_{name}"]) for page in pages if page["outcomes"][f"pseudo_minus_expected_reward_{name}"] is not None]
        report["summary"][f"page_mean_absolute_expected_reward_gap_{name}"] = float(np.mean(gaps)) if gaps else None
    report["status"] = "complete"
    write_report(report, args.output)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer")
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--attributes", type=Path, default=_DEFAULT_ATTRIBUTES)
    parser.add_argument("--num-products", type=int, default=1000, help="Native WebShop catalog/index size")
    parser.add_argument("--num-pages", type=int, default=20)
    parser.add_argument("--samples-per-page", type=int, default=256)
    parser.add_argument("--count_partial_reward", "--count-partial-reward", action="store_true", help="Count positive native rewards and compute fractional expected rewards; default rewards are training-style binary 0/1")
    parser.add_argument("--max-steps", type=int, default=13, help="Additional actions after the two setup steps; at most 13")
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--results-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--store-trajectory-prompts", action="store_true", help="Also save every continuation prompt (starting and pseudo prompts are always saved)")
    parser.add_argument("--validate-only", action="store_true", help="Construct real starts and verify native full-reward purchases without loading a model")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argument_parser().parse_args()
    report = run_testbed(args)
    print(f"Wrote {args.output}: {len(report['pages'])} pages, mode={report['mode']}")


if __name__ == "__main__":
    main()
