"""Compare search-results pseudo success estimates with native rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_product_page_choices_testbed import _wilson_interval
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import (
    SelectedProductGoal,
    _configure_vllm_engine,
    _engine_model_limit,
    _validate_positive_integer,
    sample_diverse_product_goals,
)
from treehca.pseudo_rollout_product_page_outcomes_testbed import ProductEpisode, VllmPolicy, _webshop_modules, create_server
from treehca.pseudo_rollout_results_page_testbed import VLLMPromptLogProbAdapter
from treehca.webshop_success_probability import WebshopProbabilityState, estimate_search_results_success_probability

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]
_WEBSHOP_ROOT = _ROOT / "agent_system/environments/env_package/webshop/webshop"
_DEFAULT_CATALOG = _WEBSHOP_ROOT / "data/items_shuffle_1000.json"
_DEFAULT_ATTRIBUTES = _WEBSHOP_ROOT / "data/items_ins_v2_1000.json"
_DEFAULT_OUTPUT = _ROOT / "pseudo_prob_test_results/pseudo_rollout_search_results_outcomes_report.json"
_NEXT_PAGE_ACTION = "click[next >]"


def construct_results_start_episode(
    server: Any,
    selected: SelectedProductGoal,
    *,
    seed: int,
    history_length: int = 1,
) -> ProductEpisode:
    """Create a native, forward-pageable results state for one exact goal."""
    _, native = _webshop_modules()
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.memory import SimpleMemory

    private_server = copy.copy(server)
    private_server.user_sessions = {}
    private_server.goals = [copy.deepcopy(selected.goal)]
    private_server.weights = [1]
    private_server.cum_weights = [0, 1]
    env = native.WebAgentTextEnv(observation_mode="text", server=private_server, seed=seed, session=0)
    initial_observation, _ = env.reset(session=0)
    private_server.user_sessions = {env.session: private_server.user_sessions[env.session]}

    # The native <q> route gives stable pagination over every catalog product
    # with the target's query while guaranteeing that the source product occurs.
    query = selected.product["query"]
    keywords = ["<q>", query]
    search_action = f"search[<q> {query}]"
    result_html, result_url, status = private_server.receive(env.session, env.browser.current_url, keywords=keywords)
    if status["done"] or status["reward"] != 0:
        raise ValueError("The setup search unexpectedly terminated or earned reward")
    env.browser.page_source = result_html
    env.browser.current_url = result_url
    results_observation = env.observation
    env.prev_actions.append(search_action)
    env.prev_obs.append(results_observation)

    manager = WebshopEnvironmentManager(None, None, SimpleNamespace(env=SimpleNamespace(history_length=history_length)))
    manager.memory = SimpleMemory()
    manager.memory.reset(batch_size=1)
    manager.tasks = manager.extract_task([initial_observation])
    manager.memory.store({"text_obs": manager.format_obs([initial_observation]), "action": [search_action]})
    manager.pre_text_obs = manager.format_obs([results_observation])
    prompt = manager.build_text_obs(manager.pre_text_obs, [{"available_actions": env.get_available_actions()}])[0]
    episode = ProductEpisode(env=env, manager=manager, prompt=prompt)

    parts = extract_product_page_contexts([prompt])[0]
    target_action = f"click[{selected.product['asin'].lower()}]"
    matching_asins = [product["asin"] for product in private_server.all_products if product.get("query") == query]
    if selected.product["asin"] not in matching_asins:
        raise ValueError("The exact-query results do not contain the selected source product")
    if parts.current_step != 2 or parts.completed_steps != 1 or search_action not in (parts.history_block or ""):
        raise ValueError("Starting prompt did not retain the setup search as training step 1")
    episode.setup = {
        "query": query,
        "search_action": search_action,
        "target_action": target_action,
        "matching_asins": matching_asins,
        "initial_observation": initial_observation,
        "results_observation": results_observation,
        "real_prompt": prompt,
    }
    return episode


def validate_reachable_full_reward(start: ProductEpisode, selected: SelectedProductGoal) -> dict[str, Any]:
    """Follow only forward pages and verify a native full-reward purchase."""
    probe = start.clone()
    target_action = f"click[{selected.product['asin'].lower()}]"
    next_page_clicks = 0
    while target_action[len("click[") : -1] not in probe.env.get_available_actions()["clickables"]:
        if _NEXT_PAGE_ACTION[len("click[") : -1] not in probe.env.get_available_actions()["clickables"]:
            raise ValueError(f"Target {selected.product['asin']} is not reachable through forward results pages")
        probe.advance(_NEXT_PAGE_ACTION)
        next_page_clicks += 1
    probe.advance(target_action)
    for option in selected.goal["goal_options"].values():
        probe.advance(f"click[{option}]")
    receipt = probe.advance("click[buy now]")
    return {**{key: receipt[key] for key in ("raw_reward", "full_reward", "selected_options", "reward_info")}, "next_page_clicks": next_page_clicks}


def forbidden_exit_reason(env: Any, action: str) -> str | None:
    """Identify actions excluded from the pseudo estimator's path space."""
    engine, _ = _webshop_modules()
    action_name, argument = engine.parse_action(action)
    if action_name == "search" and argument:
        return "new_search"
    if action_name != "click" or argument is None:
        return None
    argument = argument.lower()
    if argument not in env.get_available_actions()["clickables"]:
        return None
    if argument == engine.BACK_TO_SEARCH.lower():
        return "back_to_search"
    if argument != engine.PREV_PAGE.lower():
        return None
    page_name = env.server.get_page_name(env.browser.current_url)
    if page_name == "search_results":
        return "previous_results_page"
    if page_name == "item_page":
        return "back_to_results"
    return None


def collect_rollouts(
    start: ProductEpisode,
    sample_responses: Callable[[Sequence[str], Sequence[int]], Sequence[str]],
    *,
    samples_per_page: int,
    max_steps: int = 14,
    seed: int = 0,
    store_prompts: bool = False,
) -> list[dict[str, Any]]:
    """Run independent continuations from one search-results state."""
    from agent_system.environments.env_package.webshop.projection import webshop_projection

    _validate_positive_integer(samples_per_page, "samples_per_page")
    _validate_positive_integer(max_steps, "max_steps")
    if max_steps > 14:
        raise ValueError("max_steps must be at most 14")
    episodes = [start.clone() for _ in range(samples_per_page)]
    records = [
        {
            "sample_index": index,
            "steps": [],
            "termination_reason": None,
            "purchased": False,
            "full_reward": False,
            "raw_reward": 0.0,
            "selected_options": {},
        }
        for index in range(samples_per_page)
    ]
    active = list(range(samples_per_page))
    for step in range(max_steps):
        seeds = [seed + index * max_steps + step for index in active]
        responses = list(sample_responses([episodes[index].prompt for index in active], seeds))
        if len(responses) != len(active):
            raise ValueError("Sampler returned the wrong number of live-trajectory responses")
        actions, valids = webshop_projection(responses.copy())
        next_active = []
        for index, action, valid, step_seed in zip(active, actions, valids, seeds):
            episode, record = episodes[index], records[index]
            reason = forbidden_exit_reason(episode.env, action)
            step_record = {
                "continuation_step": step + 1,
                "training_step": step + 2,
                "seed": step_seed,
                "projected_action": action,
                "format_valid": bool(valid),
                "executed": reason is None,
            }
            if store_prompts:
                step_record["prompt"] = episode.prompt
            if reason is not None:
                record["termination_reason"] = reason
                record["selected_options"] = dict(_active_session(episode)["options"])
            else:
                receipt = episode.advance(action)
                step_record.update(raw_reward=receipt["raw_reward"], selected_options=receipt["selected_options"], observation=receipt["observation"])
                record.update({key: receipt[key] for key in ("purchased", "full_reward", "raw_reward", "selected_options")})
                if receipt["done"]:
                    record.update(
                        termination_reason="purchase" if receipt["purchased"] else "environment_done",
                        purchased_asin=receipt["purchased_asin"],
                        reward_info=receipt["reward_info"],
                    )
                elif step + 1 == max_steps:
                    record["termination_reason"] = "step_limit"
                else:
                    next_active.append(index)
            record["steps"].append(step_record)
        active = next_active
        if not active:
            break
    return records


def _active_session(episode: ProductEpisode) -> dict[str, Any]:
    return episode.env.server.user_sessions[episode.env.session]


def summarize_outcomes(trajectories: Sequence[dict[str, Any]], pseudo_probability: float) -> dict[str, Any]:
    """Compare the pseudo estimate with unconditional native full success."""
    pseudo_probability = float(pseudo_probability)
    if not math.isfinite(pseudo_probability) or not 0 <= pseudo_probability <= 1:
        raise ValueError("pseudo_probability must be finite and in [0, 1]")
    total = len(trajectories)
    successes = sum(record["purchased"] and record["raw_reward"] == 1.0 for record in trajectories)
    purchases = sum(record["purchased"] for record in trajectories)
    empirical_probability = successes / total if total else None
    return {
        "trajectories": total,
        "purchase_count": purchases,
        "no_purchase_count": total - purchases,
        "partial_or_zero_reward_purchase_count": purchases - successes,
        "full_reward_purchase_count": successes,
        "empirical_success_probability": empirical_probability,
        "empirical_success_wilson_95": list(_wilson_interval(successes, total)),
        "pseudo_success_probability": pseudo_probability,
        "pseudo_minus_empirical_success_probability": pseudo_probability - empirical_probability if empirical_probability is not None else None,
        "termination_counts": dict(Counter(record["termination_reason"] for record in trajectories)),
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render the per-start probability comparison and aggregate errors."""

    def probability(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3%}"

    lines = [
        "# Search-results continuation outcome testbed",
        "",
        f"Mode: `{report['mode']}`. Model: `{report['configuration']['model']}`.",
        "",
        "Pseudo paths use forward pagination, correct-product clicks, and grouped correct option choices. Empirical rollouts start at the identical results state and count only native full-reward purchases as success.",
        "",
        "| Target ASIN | Query | Rollouts | Empirical success | Pseudo success | Signed gap |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for page in report["pages"]:
        outcome = page.get("outcomes", {})
        query = page["setup"]["query"].replace("|", "\\|")
        lines.append(f"| {page['asin']} | {query} | {outcome.get('trajectories', 0)} | {probability(outcome.get('empirical_success_probability'))} | {probability(outcome.get('pseudo_success_probability'))} | {probability(outcome.get('pseudo_minus_empirical_success_probability'))} |")
    summary = report.get("summary")
    if summary is not None:
        lines.extend(
            [
                "",
                "## Aggregate comparison",
                "",
                f"- Mean pseudo success: {probability(summary['page_mean_pseudo_success_probability'])}",
                f"- Pooled empirical success: {probability(summary['pooled_empirical_success_probability'])}",
                f"- Mean absolute per-page gap: {probability(summary['page_mean_absolute_gap'])}",
                f"- Root mean squared per-page gap: {probability(summary['page_root_mean_squared_gap'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Rollouts that issue a new search, return to the search page, move to a previous results page, or return from a product to results terminate as out-of-scope failures.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    """Atomically checkpoint JSON and its Markdown companion."""
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in (
        (output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
        (output.with_suffix(".md"), format_markdown_summary(report)),
    ):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content)
        os.replace(temporary, path)


def _required_product_max_logprobs(server: Any) -> int:
    largest_group = max((len(values) + 1 for item in server.all_products for values in item.get("options", {}).values()), default=0)
    return max(1, 2 * largest_group)


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "num_pages",
        "samples_per_page",
        "max_steps",
        "history_length",
        "num_products",
        "max_prompt_length",
        "max_new_tokens",
        "rollout_batch_size",
        "tensor_parallel_size",
    ):
        _validate_positive_integer(getattr(args, name), name)
    if args.max_steps > 14:
        raise ValueError("max_steps must be at most 14")
    if args.max_model_len is not None:
        _validate_positive_integer(args.max_model_len, "max_model_len")
    if not math.isfinite(args.path_probability_threshold) or not 0 <= args.path_probability_threshold <= 1:
        raise ValueError("path_probability_threshold must be finite and in [0, 1]")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("top_p and gpu_memory_utilization must be in (0, 1]")
    if args.top_k != -1 and args.top_k < 1:
        raise ValueError("top_k must be -1 or positive")


def run_testbed(args: argparse.Namespace) -> dict[str, Any]:
    """Construct starts, score them, run continuations, and write the report."""
    _validate_arguments(args)
    server = create_server(args.catalog, args.attributes, args.seed, args.num_products)
    selected = sample_diverse_product_goals(server.all_products, server.goals, args.num_pages, args.seed)
    starts, pages = [], []
    for index, item in enumerate(selected):
        start = construct_results_start_episode(server, item, seed=args.seed + index, history_length=args.history_length)
        oracle = validate_reachable_full_reward(start, item)
        if not oracle["full_reward"]:
            raise ValueError(f"ASIN {item.product['asin']} failed its native forward-path full-reward oracle: {oracle}")
        starts.append(start)
        pages.append(
            {
                "source_page_index": index,
                "asin": item.product["asin"],
                "category": item.product.get("category"),
                "goal": item.goal,
                "price": server.product_prices[item.product["asin"]],
                "setup": start.setup,
                "oracle": oracle,
            }
        )
        logger.info("Validated results start %d/%d: %s", index + 1, len(selected), item.product["asin"])

    report = {
        "schema_version": 1,
        "status": "running",
        "mode": "validate_only" if args.validate_only else "monte_carlo",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sampling": {
            "goal_source": "native synthetic WebShop goals sampled by product diversity",
            "available_goals": len(server.goals),
            "setup": "native exact-catalog-query search results before any product click",
            "success_event": "native purchase with raw reward exactly 1.0",
        },
        "pages": pages,
    }
    if args.validate_only:
        report["status"] = "complete"
        write_report(report, args.output)
        return report

    report["status"] = "initializing_model"
    write_report(report, args.output)
    logger.info("Initializing tokenizer and vLLM engine with tensor_parallel_size=%d", args.tensor_parallel_size)
    model_start = time.perf_counter()
    _configure_vllm_engine()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    engine_kwargs = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "max_logprobs": _required_product_max_logprobs(server),
        # Results action scoring uses prompt_logprobs, which V0 cannot combine
        # with prefix caching. Product grouped-choice scoring does not require it.
        "enable_prefix_caching": False,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    engine = LLM(**engine_kwargs)
    model_limit = _engine_model_limit(engine)
    logger.info("Tokenizer and vLLM engine initialized in %.1f seconds", time.perf_counter() - model_start)
    adapter = VLLMPromptLogProbAdapter(engine)
    policy = VllmPolicy(engine, tokenizer, args, model_limit)
    report["configuration"].update(
        effective_max_model_len=model_limit,
        vllm_engine_version="V0",
        enable_prefix_caching=False,
        product_max_logprobs=engine_kwargs["max_logprobs"],
        pseudo_probability_mode="forward_results_paths_and_grouped_product_choices",
        truncation="error",
    )

    for index, (start, page) in enumerate(zip(starts, pages)):
        report["status"] = "scoring_pseudo_probability"
        report["active_page_index"] = index
        write_report(report, args.output)
        logger.info("Scoring pseudo success probability for page %d/%d", index + 1, len(pages))
        pseudo_start = time.perf_counter()
        probability_state = WebshopProbabilityState(start.env, start.manager, start.prompt)
        pseudo_probability = estimate_search_results_success_probability(
            probability_state,
            tokenizer,
            adapter,
            engine,
            args.path_probability_threshold,
            max_model_len=model_limit,
        )
        logger.info("Pseudo success probability for page %d is %.6g (%.1f seconds)", index + 1, pseudo_probability, time.perf_counter() - pseudo_start)
        report["status"] = "sampling_empirical_rollouts"
        write_report(report, args.output)
        logger.info("Sampling %d empirical rollouts for page %d/%d with max_steps=%d", args.samples_per_page, index + 1, len(pages), args.max_steps)
        rollout_start = time.perf_counter()
        trajectories = collect_rollouts(
            start,
            policy,
            samples_per_page=args.samples_per_page,
            max_steps=args.max_steps,
            seed=args.seed + 10_000 + index * args.samples_per_page * args.max_steps,
            store_prompts=args.store_trajectory_prompts,
        )
        logger.info("Empirical rollouts for page %d completed in %.1f seconds", index + 1, time.perf_counter() - rollout_start)
        page.update(
            pseudo_success_probability=pseudo_probability,
            trajectories=trajectories,
            outcomes=summarize_outcomes(trajectories, pseudo_probability),
        )
        report["pages_completed"] = index + 1
        write_report(report, args.output)
        logger.info("Page %d/%d: %s", index + 1, len(pages), page["outcomes"])

    all_trajectories = [trajectory for page in pages for trajectory in page["trajectories"]]
    total_successes = sum(trajectory["purchased"] and trajectory["raw_reward"] == 1.0 for trajectory in all_trajectories)
    gaps = [page["outcomes"]["pseudo_minus_empirical_success_probability"] for page in pages]
    report["summary"] = {
        "pages": len(pages),
        "trajectories": len(all_trajectories),
        "full_reward_purchase_count": total_successes,
        "pooled_empirical_success_probability": total_successes / len(all_trajectories) if all_trajectories else None,
        "page_mean_pseudo_success_probability": float(np.mean([page["pseudo_success_probability"] for page in pages])),
        "page_mean_absolute_gap": float(np.mean(np.abs(gaps))),
        "page_root_mean_squared_gap": float(np.sqrt(np.mean(np.square(gaps)))),
        "termination_counts": dict(Counter(trajectory["termination_reason"] for trajectory in all_trajectories)),
    }
    report["status"] = "complete"
    report.pop("active_page_index", None)
    write_report(report, args.output)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer")
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--attributes", type=Path, default=_DEFAULT_ATTRIBUTES)
    parser.add_argument("--num-products", type=int, default=1000)
    parser.add_argument("--num-pages", type=int, default=20, help="Number of independent search-results starting states")
    parser.add_argument("--samples-per-page", type=int, default=256)
    parser.add_argument("--path-probability-threshold", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=14, help="Additional actions after the setup search; at most 14")
    parser.add_argument("--history-length", type=int, default=1)
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
    parser.add_argument("--store-trajectory-prompts", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argument_parser().parse_args()
    report = run_testbed(args)
    print(f"Wrote {args.output}: {len(report['pages'])} pages, mode={report['mode']}")


if __name__ == "__main__":
    main()
