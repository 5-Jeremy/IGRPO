"""Compare search-results pseudo success estimates with native rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing
import os
import tempfile
import time
import traceback
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
from treehca.webshop_success_probability import WebshopProbabilityState, _product_can_earn_full_reward, estimate_search_results_probabilities

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


def sample_full_reward_product_goals(server: Any, num_pages: int, seed: int) -> tuple[SelectedProductGoal, ...]:
    """Sample only goals whose stated options earn native full reward."""
    _webshop_modules()
    from web_agent_site.engine.goal import get_reward

    eligible_goals = []
    for goal in server.goals:
        asin = goal.get("asin")
        options = goal.get("goal_options")
        if asin not in server.product_item_dict or not isinstance(options, dict):
            continue
        product = server.product_item_dict[asin]
        price = server.product_prices[asin]
        # Native reward matches option values against (group, value) pairs.
        if get_reward(product, goal, price=price, options=options) == 1.0:
            eligible_goals.append(goal)
    return sample_diverse_product_goals(server.all_products, eligible_goals, num_pages, seed)


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
            "entered_full_reward_product_page": False,
            "first_full_reward_product_asin": None,
            "first_full_reward_product_step": None,
        }
        for index in range(samples_per_page)
    ]
    active = list(range(samples_per_page))
    product_eligibility_cache: dict[str, bool] = {}
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
                previous_page_name = _page_name(episode.env)
                receipt = episode.advance(action)
                step_record.update(raw_reward=receipt["raw_reward"], selected_options=receipt["selected_options"], observation=receipt["observation"])
                record.update({key: receipt[key] for key in ("purchased", "full_reward", "raw_reward", "selected_options")})
                entered_asin = _eligible_product_entry(episode.env, previous_page_name, product_eligibility_cache)
                if entered_asin is not None and not record["entered_full_reward_product_page"]:
                    record.update(entered_full_reward_product_page=True, first_full_reward_product_asin=entered_asin, first_full_reward_product_step=step + 1)
                    step_record["entered_full_reward_product_page"] = True
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


def _page_name(env: Any) -> str | None:
    if not hasattr(env, "server"):
        return None
    return env.server.get_page_name(env.browser.current_url)


def _eligible_product_entry(env: Any, previous_page_name: str | None, eligibility_cache: dict[str, bool]) -> str | None:
    """Recognize the first results-to-item transition to a full-reward product."""
    if previous_page_name != "search_results" or _page_name(env) != "item_page":
        return None
    session = env.server.user_sessions[env.session]
    asin = session.get("asin")
    if asin is None:
        return None
    if asin not in eligibility_cache:
        eligibility_cache[asin] = _product_can_earn_full_reward(SimpleNamespace(env=env), asin)
    return asin if eligibility_cache[asin] else None


def summarize_outcomes(trajectories: Sequence[dict[str, Any]], pseudo_probability: float, pseudo_product_entry_probability: float) -> dict[str, Any]:
    """Compare pseudo and empirical product entry and full purchase success."""
    pseudo_probability = float(pseudo_probability)
    if not math.isfinite(pseudo_probability) or not 0 <= pseudo_probability <= 1:
        raise ValueError("pseudo_probability must be finite and in [0, 1]")
    pseudo_product_entry_probability = float(pseudo_product_entry_probability)
    if not math.isfinite(pseudo_product_entry_probability) or not 0 <= pseudo_product_entry_probability <= 1:
        raise ValueError("pseudo_product_entry_probability must be finite and in [0, 1]")
    total = len(trajectories)
    successes = sum(record["purchased"] and record["raw_reward"] == 1.0 for record in trajectories)
    product_entries = sum(bool(record["entered_full_reward_product_page"]) for record in trajectories)
    purchases = sum(record["purchased"] for record in trajectories)
    empirical_probability = successes / total if total else None
    empirical_product_entry_probability = product_entries / total if total else None
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
        "full_reward_product_entry_count": product_entries,
        "empirical_full_reward_product_entry_probability": empirical_product_entry_probability,
        "empirical_full_reward_product_entry_wilson_95": list(_wilson_interval(product_entries, total)),
        "pseudo_full_reward_product_entry_probability": pseudo_product_entry_probability,
        "pseudo_minus_empirical_full_reward_product_entry_probability": pseudo_product_entry_probability - empirical_product_entry_probability if empirical_product_entry_probability is not None else None,
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
        (
            "Pseudo paths use forward pagination, full-reward-capable product clicks, and grouped correct option choices. "
            "Empirical rollouts start at the identical results state. Product entry is counted on the first qualifying "
            "results-to-product transition; purchase success requires native full reward."
        ),
        "",
        "| Target ASIN | Query | Rollouts | Empirical product entry | Pseudo product entry | Entry gap | Empirical success | Pseudo success | Success gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for page in report["pages"]:
        outcome = page.get("outcomes", {})
        query = page["setup"]["query"].replace("|", "\\|")
        values = (
            page["asin"],
            query,
            outcome.get("trajectories", 0),
            *(probability(outcome.get(key)) for key in (
                "empirical_full_reward_product_entry_probability",
                "pseudo_full_reward_product_entry_probability",
                "pseudo_minus_empirical_full_reward_product_entry_probability",
                "empirical_success_probability",
                "pseudo_success_probability",
                "pseudo_minus_empirical_success_probability",
            )),
        )
        lines.append("| " + " | ".join(map(str, values)) + " |")
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
                f"- Mean pseudo full-reward product entry: {probability(summary['page_mean_pseudo_full_reward_product_entry_probability'])}",
                f"- Pooled empirical full-reward product entry: {probability(summary['pooled_empirical_full_reward_product_entry_probability'])}",
                f"- Mean absolute per-page product-entry gap: {probability(summary['page_mean_absolute_full_reward_product_entry_gap'])}",
                f"- Root mean squared per-page product-entry gap: {probability(summary['page_root_mean_squared_full_reward_product_entry_gap'])}",
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


def _visible_gpu_ids() -> tuple[str, ...]:
    """Return CUDA-visible device identifiers without assuming numeric IDs."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return tuple(device.strip() for device in configured.split(",") if device.strip() and device.strip() != "-1")
    try:
        import torch

        return tuple(str(index) for index in range(torch.cuda.device_count()))
    except Exception:
        return ()


def _resolve_gpu_groups(args: argparse.Namespace, visible_gpu_ids: Sequence[str] | None = None) -> tuple[tuple[str, ...], ...]:
    """Partition visible GPUs into independent tensor-parallel workers."""
    devices = tuple(_visible_gpu_ids() if visible_gpu_ids is None else visible_gpu_ids)
    tensor_parallel_size = args.tensor_parallel_size
    if not devices:
        if args.data_parallel_size not in (None, 1):
            raise ValueError("data_parallel_size greater than one requires visible CUDA devices")
        return ((),)
    if len(devices) < tensor_parallel_size or len(devices) % tensor_parallel_size:
        raise ValueError("The number of visible GPUs must be divisible by tensor_parallel_size")
    available_workers = len(devices) // tensor_parallel_size
    data_parallel_size = available_workers if args.data_parallel_size is None else args.data_parallel_size
    if data_parallel_size > available_workers:
        raise ValueError(f"Requested {data_parallel_size} data-parallel workers, but only {available_workers} tensor-parallel GPU group(s) are available")
    return tuple(tuple(devices[start : start + tensor_parallel_size]) for start in range(0, data_parallel_size * tensor_parallel_size, tensor_parallel_size))


def _engine_and_policy(args: argparse.Namespace, server: Any) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    """Initialize one worker-local tokenizer, vLLM engine, and rollout policy."""
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
    return tokenizer, engine, VLLMPromptLogProbAdapter(engine), model_limit, engine_kwargs


def _evaluate_start_shard(
    args: argparse.Namespace,
    server: Any,
    indexed_starts: Sequence[tuple[int, ProductEpisode]],
    *,
    worker_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run model inference for one worker's disjoint starting-page shard."""
    logger.info("Worker %d initializing tokenizer and vLLM engine with tensor_parallel_size=%d", worker_rank, args.tensor_parallel_size)
    model_start = time.perf_counter()
    tokenizer, engine, adapter, model_limit, engine_kwargs = _engine_and_policy(args, server)
    policy = VllmPolicy(engine, tokenizer, args, model_limit)
    logger.info("Worker %d initialized tokenizer and vLLM engine in %.1f seconds", worker_rank, time.perf_counter() - model_start)

    results = []
    for local_index, (global_index, start) in enumerate(indexed_starts, start=1):
        logger.info("Worker %d scoring pseudo success for shard page %d/%d (global page %d)", worker_rank, local_index, len(indexed_starts), global_index + 1)
        pseudo_start = time.perf_counter()
        probability_state = WebshopProbabilityState(start.env, start.manager, start.prompt)
        probabilities = estimate_search_results_probabilities(
            probability_state,
            tokenizer,
            adapter,
            engine,
            args.path_probability_threshold,
            max_model_len=model_limit,
        )
        logger.info("Worker %d pseudo probabilities for global page %d: product entry %.6g, success %.6g (%.1f seconds)", worker_rank, global_index + 1, probabilities.product_entry, probabilities.success, time.perf_counter() - pseudo_start)
        logger.info("Worker %d sampling %d empirical rollouts for global page %d", worker_rank, args.samples_per_page, global_index + 1)
        rollout_start = time.perf_counter()
        trajectories = collect_rollouts(
            start,
            policy,
            samples_per_page=args.samples_per_page,
            max_steps=args.max_steps,
            seed=args.seed + 10_000 + global_index * args.samples_per_page * args.max_steps,
            store_prompts=args.store_trajectory_prompts,
        )
        logger.info("Worker %d empirical rollouts for global page %d completed in %.1f seconds", worker_rank, global_index + 1, time.perf_counter() - rollout_start)
        results.append(
            {
                "source_page_index": global_index,
                "worker_rank": worker_rank,
                "pseudo_success_probability": probabilities.success,
                "pseudo_full_reward_product_entry_probability": probabilities.product_entry,
                "trajectories": trajectories,
                "outcomes": summarize_outcomes(trajectories, probabilities.success, probabilities.product_entry),
            }
        )
    metadata = {
        "effective_max_model_len": model_limit,
        "vllm_engine_version": "V0",
        "enable_prefix_caching": False,
        "product_max_logprobs": engine_kwargs["max_logprobs"],
        "pseudo_probability_mode": "forward_results_paths_and_grouped_product_choices",
        "truncation": "error",
    }
    return results, metadata


def _data_parallel_worker(
    args: argparse.Namespace,
    indexed_selected: Sequence[tuple[int, SelectedProductGoal]],
    gpu_ids: tuple[str, ...],
    worker_rank: int,
    result_path: str,
) -> None:
    """Spawn entry point: reconstruct native states and evaluate one GPU shard."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s [worker {worker_rank}] %(message)s")
    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        server = create_server(args.catalog, args.attributes, args.seed, args.num_products)
        indexed_starts = []
        for global_index, selected in indexed_selected:
            start = construct_results_start_episode(server, selected, seed=args.seed + global_index, history_length=args.history_length)
            oracle = validate_reachable_full_reward(start, selected)
            if not oracle["full_reward"]:
                raise ValueError(f"ASIN {selected.product['asin']} failed its worker-local native oracle: {oracle}")
            indexed_starts.append((global_index, start))
        pages, metadata = _evaluate_start_shard(args, server, indexed_starts, worker_rank=worker_rank)
        payload = {"status": "complete", "worker_rank": worker_rank, "gpu_ids": list(gpu_ids), "pages": pages, "metadata": metadata}
    except BaseException as error:
        payload = {"status": "error", "worker_rank": worker_rank, "gpu_ids": list(gpu_ids), "error": repr(error), "traceback": traceback.format_exc()}
    Path(result_path).write_text(json.dumps(payload, allow_nan=False))
    if payload["status"] == "error":
        raise RuntimeError(payload["error"])


def _run_data_parallel(
    args: argparse.Namespace,
    selected: Sequence[SelectedProductGoal],
    gpu_groups: Sequence[tuple[str, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Spawn one model replica per GPU group and merge its disjoint results."""
    shards = _shard_selected_starts(selected, len(gpu_groups))
    # More GPUs than pages should not create model replicas with no work.
    active = [(rank, gpu_groups[rank], shard) for rank, shard in enumerate(shards) if shard]
    context = multiprocessing.get_context("spawn")
    original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    with tempfile.TemporaryDirectory(prefix="webshop-search-results-dp-") as temporary_directory:
        processes = []
        result_paths = []
        try:
            for rank, gpu_ids, shard in active:
                result_path = str(Path(temporary_directory) / f"worker-{rank}.json")
                result_paths.append(result_path)
                if gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                process = context.Process(target=_data_parallel_worker, args=(args, shard, gpu_ids, rank, result_path), name=f"webshop-dp-{rank}")
                process.start()
                processes.append(process)
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise
        finally:
            if original_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices

        try:
            for process in processes:
                process.join()
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise

        payloads = []
        for process, result_path in zip(processes, result_paths):
            if not Path(result_path).exists():
                raise RuntimeError(f"Data-parallel worker {process.name} exited with code {process.exitcode} without writing a result")
            payload = json.loads(Path(result_path).read_text())
            if process.exitcode != 0 or payload.get("status") != "complete":
                raise RuntimeError(f"Data-parallel worker {payload.get('worker_rank')} failed: {payload.get('error')}\n{payload.get('traceback', '')}")
            payloads.append(payload)

    pages = sorted((page for payload in payloads for page in payload["pages"]), key=lambda page: page["source_page_index"])
    if len(pages) != len(selected):
        raise RuntimeError(f"Data-parallel workers returned {len(pages)} pages for {len(selected)} starts")
    metadata = payloads[0]["metadata"]
    if any(payload["metadata"] != metadata for payload in payloads[1:]):
        raise RuntimeError("Data-parallel workers reported inconsistent model metadata")
    return pages, metadata


def _shard_selected_starts(selected: Sequence[SelectedProductGoal], worker_count: int) -> list[list[tuple[int, SelectedProductGoal]]]:
    """Assign starts round-robin while retaining their global output order."""
    _validate_positive_integer(worker_count, "worker_count")
    return [[(index, item) for index, item in enumerate(selected) if index % worker_count == rank] for rank in range(worker_count)]


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
    if args.data_parallel_size is not None:
        _validate_positive_integer(args.data_parallel_size, "data_parallel_size")
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
    selected = sample_full_reward_product_goals(server, args.num_pages, args.seed)
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

    gpu_groups = _resolve_gpu_groups(args)
    active_worker_count = min(len(gpu_groups), len(selected))
    report["configuration"].update(
        requested_data_parallel_size=args.data_parallel_size,
        effective_data_parallel_size=active_worker_count,
        gpu_groups=[list(group) for group in gpu_groups[:active_worker_count]],
    )
    report["status"] = "initializing_workers"
    write_report(report, args.output)
    logger.info(
        "Starting %d data-parallel worker(s), tensor_parallel_size=%d, GPU groups=%s",
        active_worker_count,
        args.tensor_parallel_size,
        report["configuration"]["gpu_groups"],
    )
    report["status"] = "running_workers"
    write_report(report, args.output)
    if active_worker_count == 1:
        dynamic_pages, model_metadata = _evaluate_start_shard(args, server, list(enumerate(starts)), worker_rank=0)
    else:
        dynamic_pages, model_metadata = _run_data_parallel(args, selected, gpu_groups[:active_worker_count])
    report["configuration"].update(model_metadata)
    for dynamic_page in dynamic_pages:
        index = dynamic_page.pop("source_page_index")
        pages[index].update(dynamic_page)
        report["pages_completed"] = report.get("pages_completed", 0) + 1
    write_report(report, args.output)

    all_trajectories = [trajectory for page in pages for trajectory in page["trajectories"]]
    total_successes = sum(trajectory["purchased"] and trajectory["raw_reward"] == 1.0 for trajectory in all_trajectories)
    total_product_entries = sum(bool(trajectory["entered_full_reward_product_page"]) for trajectory in all_trajectories)
    gaps = [page["outcomes"]["pseudo_minus_empirical_success_probability"] for page in pages]
    product_entry_gaps = [page["outcomes"]["pseudo_minus_empirical_full_reward_product_entry_probability"] for page in pages]
    report["summary"] = {
        "pages": len(pages),
        "trajectories": len(all_trajectories),
        "full_reward_purchase_count": total_successes,
        "pooled_empirical_success_probability": total_successes / len(all_trajectories) if all_trajectories else None,
        "page_mean_pseudo_success_probability": float(np.mean([page["pseudo_success_probability"] for page in pages])),
        "page_mean_absolute_gap": float(np.mean(np.abs(gaps))),
        "page_root_mean_squared_gap": float(np.sqrt(np.mean(np.square(gaps)))),
        "termination_counts": dict(Counter(trajectory["termination_reason"] for trajectory in all_trajectories)),
        "full_reward_product_entry_count": total_product_entries,
        "pooled_empirical_full_reward_product_entry_probability": total_product_entries / len(all_trajectories) if all_trajectories else None,
        "page_mean_pseudo_full_reward_product_entry_probability": float(np.mean([page["pseudo_full_reward_product_entry_probability"] for page in pages])),
        "page_mean_absolute_full_reward_product_entry_gap": float(np.mean(np.abs(product_entry_gaps))),
        "page_root_mean_squared_full_reward_product_entry_gap": float(np.sqrt(np.mean(np.square(product_entry_gaps)))),
    }
    report["status"] = "complete"
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
    parser.add_argument("--data-parallel-size", type=int, help="Independent model replicas; default uses every visible tensor-parallel GPU group")
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
