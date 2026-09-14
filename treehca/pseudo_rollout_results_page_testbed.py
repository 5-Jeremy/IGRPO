"""Score realistic WebShop result-page actions without running agent rollouts."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from treehca.product_page_parser import ProductPageContextParts, extract_product_page_contexts
from treehca.pseudo_rollout_results_page import compute_results_page_answer_probability

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_WEBSHOP_ROOT = _ROOT / "agent_system/environments/env_package/webshop/webshop"
_RESULTS_ROOT = _ROOT / "pseudo_prob_test_results"
_DEFAULT_CATALOG = _WEBSHOP_ROOT / "data/items_shuffle_1000.json"
_DEFAULT_ATTRIBUTES = _WEBSHOP_ROOT / "data/items_ins_v2_1000.json"
_DEFAULT_OUTPUT = _RESULTS_ROOT / "pseudo_rollout_results_page_report.json"
_NEXT_PAGE_ACTION = "click[next >]"
_PREVIOUS_PAGE_ACTION = "click[< prev]"


@dataclass(frozen=True)
class SearchPageSpec:
    """A real search result set paired with one synthetic goal and page number."""

    goal: dict[str, Any]
    query: str
    page_number: int
    total_results: int
    products: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RenderedResultsPage:
    """One history-bearing WebShop results prompt and its scoring metadata."""

    spec: SearchPageSpec
    prompt: str
    context_parts: ProductPageContextParts
    correct_product_asins: tuple[str, ...]


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_webshop_assets(
    catalog_path: Path,
    attributes_path: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, float], dict[str, set[str]], list[dict[str, Any]], Any]:
    """Load the catalog, reproducible synthetic goals, and real Lucene index."""
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    from web_agent_site.engine.engine import init_search_engine, load_products
    from web_agent_site.engine.goal import get_goals

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        products, product_by_asin, product_prices, attribute_to_asins = load_products(
            str(catalog_path),
            str(attributes_path),
            human_goals=False,
        )
        goals = get_goals(products, product_prices, human_goals=False)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    search_engine = init_search_engine(num_products=len(products))
    return products, product_by_asin, product_prices, attribute_to_asins, goals, search_engine


def sample_search_pages(
    products: Sequence[dict[str, Any]],
    product_by_asin: dict[str, dict[str, Any]],
    attribute_to_asins: dict[str, set[str]],
    goals: Sequence[dict[str, Any]],
    search_engine: Any,
    num_pages: int,
    seed: int,
) -> tuple[SearchPageSpec, ...]:
    """Run real catalog searches and sample nonempty pages reproducibly.

    Even-numbered selections prefer page one; odd-numbered selections prefer a
    later page when one exists. This keeps both first-page and previous-page
    behavior represented without manufacturing any search results.
    """
    _validate_positive_integer(num_pages, "num_pages")
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    from web_agent_site.engine.engine import PRODUCT_WINDOW, get_product_per_page, get_top_n_product_from_keywords

    rng = random.Random(seed)
    candidate_goals = [goal for goal in goals if isinstance(goal.get("query"), str) and goal["query"].strip()]
    rng.shuffle(candidate_goals)
    selected = []
    used_keys = set()
    for goal in candidate_goals:
        query = goal["query"].strip()
        ranked_products = get_top_n_product_from_keywords(
            query.lower().split(),
            search_engine,
            products,
            product_by_asin,
            attribute_to_asins,
        )
        if not ranked_products:
            continue
        total_pages = math.ceil(len(ranked_products) / PRODUCT_WINDOW)
        if len(selected) % 2 == 1 and total_pages > 1:
            page_number = rng.randint(2, total_pages)
        else:
            page_number = 1
        key = (goal["instruction_text"], query, page_number)
        if key in used_keys:
            continue
        page_products = tuple(get_product_per_page(ranked_products, page_number))
        if not page_products:
            continue
        used_keys.add(key)
        selected.append(
            SearchPageSpec(
                goal=goal,
                query=query,
                page_number=page_number,
                total_results=len(ranked_products),
                products=page_products,
            )
        )
        if len(selected) == num_pages:
            return tuple(selected)
    raise ValueError(f"Requested {num_pages} pages, but only {len(selected)} distinct nonempty search pages could be generated")


def _matching_option_selection(product: dict[str, Any], goal: dict[str, Any]) -> dict[str, str] | None:
    """Find a selection receiving full credit from WebShop's option reward."""
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    goal_options = goal.get("goal_options", {})
    if not goal_options:
        return {}
    if not isinstance(goal_options, dict):
        raise ValueError("Synthetic goal_options must be a mapping")

    from web_agent_site.engine.goal import get_option_reward

    product_options = product.get("options", {})
    if not product_options or any(not values for values in product_options.values()):
        return None
    group_names = tuple(product_options)
    # This uses the same goal representation as get_reward. Selecting one value
    # from every group cannot lower option reward, so no empty choice is needed.
    reward_targets = goal_options.items()
    for values in itertools.product(*(product_options[group_name] for group_name in group_names)):
        selection = dict(zip(group_names, values))
        _, num_matches = get_option_reward(selection.values(), reward_targets)
        if num_matches == len(goal_options):
            return selection
    return None


def product_can_earn_full_reward(
    product: dict[str, Any],
    goal: dict[str, Any],
    price: float,
    *,
    reward_function: Callable[..., float] | None = None,
) -> bool:
    """Return whether some available option selection earns WebShop reward 1."""
    selection = _matching_option_selection(product, goal)
    if selection is None:
        return False
    if reward_function is None:
        from web_agent_site.engine.goal import get_reward

        reward_function = get_reward
    reward = reward_function(product, goal, price=price, options=selection)
    return math.isclose(float(reward), 1.0, rel_tol=0.0, abs_tol=1e-12)


def render_results_pages(
    specs: Sequence[SearchPageSpec],
    product_prices: dict[str, float],
    *,
    reward_function: Callable[..., float] | None = None,
) -> tuple[RenderedResultsPage, ...]:
    """Render actual WebShop pages with the search action in prompt history."""
    if str(_WEBSHOP_ROOT) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP_ROOT))
    from web_agent_site.engine import engine
    from web_agent_site.envs import web_agent_text_env

    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.memory import SimpleMemory

    manager = object.__new__(WebshopEnvironmentManager)
    manager.config = SimpleNamespace(env=SimpleNamespace(history_length=1))
    manager.tasks = [spec.goal["instruction_text"] for spec in specs]
    manager.memory = SimpleMemory()
    manager.memory.reset(batch_size=len(specs))
    text_env = object.__new__(web_agent_text_env.WebAgentTextEnv)
    text_env.observation_mode = "text"

    initial_observations = []
    result_observations = []
    result_infos = []
    search_actions = []
    with web_agent_text_env.app.test_request_context("/"):
        for index, spec in enumerate(specs):
            session_id = f"results-page-pseudo-testbed-{index}"
            initial_html = engine.map_action_to_html(
                "start",
                session_id=session_id,
                instruction_text=spec.goal["instruction_text"],
            )
            text_env.browser = SimpleNamespace(page_source=initial_html, current_url=f"/{session_id}")
            text_env.instruction_text = spec.goal["instruction_text"]
            initial_observations.append(text_env.observation)

            result_html = engine.map_action_to_html(
                "search",
                session_id=session_id,
                products=spec.products,
                keywords=spec.query.lower().split(),
                page=spec.page_number,
                total=spec.total_results,
                instruction_text=spec.goal["instruction_text"],
            )
            text_env.browser = SimpleNamespace(
                page_source=result_html,
                current_url=f"/search_results/{session_id}/{spec.query}/{spec.page_number}",
            )
            result_observations.append(text_env.observation)
            result_infos.append({"available_actions": text_env.get_available_actions()})
            search_actions.append(f"search[{spec.query}]")

    formatted_initial = manager.format_obs(initial_observations)
    manager.memory.store({"text_obs": formatted_initial, "action": search_actions})
    formatted_results = manager.format_obs(result_observations)
    prompts = manager.build_text_obs(formatted_results, result_infos)
    context_parts = extract_product_page_contexts(prompts)

    rendered = []
    for spec, prompt, parts, search_action in zip(specs, prompts, context_parts, search_actions):
        if parts.current_step != 2 or parts.history_length != 1 or search_action not in (parts.history_block or ""):
            raise ValueError("Rendered prompt did not retain the actual search as its one-step history")
        correct_asins = tuple(
            product["asin"]
            for product in spec.products
            if product_can_earn_full_reward(
                product,
                spec.goal,
                product_prices[product["asin"]],
                reward_function=reward_function,
            )
        )
        rendered.append(
            RenderedResultsPage(
                spec=spec,
                prompt=prompt,
                context_parts=parts,
                correct_product_asins=correct_asins,
            )
        )
    return tuple(rendered)


class VLLMPromptLogProbAdapter:
    """Expose vLLM chosen-token prompt log-probabilities as ``compute_log_prob``."""

    def __init__(self, inference_engine: Any):
        self.inference_engine = inference_engine

    def compute_log_prob(self, batch: Any) -> Any:
        from vllm import SamplingParams

        from verl import DataProto

        input_ids = batch.batch["input_ids"]
        responses = batch.batch["responses"]
        prompt_length = input_ids.shape[1] - responses.shape[1]
        outputs = self.inference_engine.generate(
            prompts=[{"prompt_token_ids": row.tolist()} for row in input_ids],
            sampling_params=SamplingParams(
                n=1,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                prompt_logprobs=0,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        if len(outputs) != len(input_ids):
            raise ValueError("vLLM returned a misaligned prompt-logprob batch")

        rows = []
        for row_index, output in enumerate(outputs):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != input_ids.shape[1]:
                raise ValueError("vLLM did not return one prompt-logprob entry per input token")
            row = []
            for token_index in range(prompt_length, input_ids.shape[1]):
                token_id = int(input_ids[row_index, token_index])
                token_values = prompt_logprobs[token_index]
                if token_values is None or token_id not in token_values:
                    raise ValueError(f"vLLM omitted token {token_id} at prompt position {token_index}")
                value = getattr(token_values[token_id], "logprob", token_values[token_id])
                row.append(float(value))
            rows.append(row)
        return DataProto.from_dict(tensors={"old_log_probs": torch.tensor(rows, dtype=torch.float32)})


def _action_kind(action: str, displayed_asins: set[str]) -> tuple[str, str | None]:
    if action == _NEXT_PAGE_ACTION:
        return "next_page", None
    if action == _PREVIOUS_PAGE_ACTION:
        return "previous_page", None
    if action == "click[back to search]":
        return "back_to_search", None
    if action.startswith("click[") and action.endswith("]"):
        asin = action[len("click[") : -1].upper()
        if asin in displayed_asins:
            return "product", asin
    return "other", None


def score_page_actions(
    page: RenderedResultsPage,
    tokenizer: Any,
    actor_rollout_wg: Any,
) -> dict[str, Any]:
    """Score every displayed action and serialize one page's raw results."""
    displayed_asins = {product["asin"] for product in page.spec.products}
    correct_asins = set(page.correct_product_asins)
    action_rows = []
    for action in page.context_parts.admissible_actions:
        probability = float(
            compute_results_page_answer_probability(
                page.prompt,
                action,
                tokenizer,
                actor_rollout_wg,
            ).item()
        )
        kind, asin = _action_kind(action, displayed_asins)
        action_rows.append(
            {
                "action": action,
                "kind": kind,
                "asin": asin,
                "is_correct_product": asin in correct_asins if asin is not None else False,
                "joint_probability": probability,
            }
        )

    maximum = max(row["joint_probability"] for row in action_rows)
    tolerance = max(1e-30, maximum * 1e-12)
    top_actions = [row["action"] for row in action_rows if math.isclose(row["joint_probability"], maximum, rel_tol=0.0, abs_tol=tolerance)]
    correct_probability = sum(row["joint_probability"] for row in action_rows if row["is_correct_product"])
    probability_sum = sum(row["joint_probability"] for row in action_rows)
    return {
        "shopping_task": page.spec.goal["instruction_text"],
        "target_asin": page.spec.goal["asin"],
        "query": page.spec.query,
        "search_action": f"search[{page.spec.query}]",
        "page_number": page.spec.page_number,
        "total_results": page.spec.total_results,
        "displayed_asins": [product["asin"] for product in page.spec.products],
        "correct_product_asins": list(page.correct_product_asins),
        "has_correct_product": bool(correct_asins),
        "correct_product_joint_probability": correct_probability if correct_asins else None,
        "sum_of_action_joint_probabilities": probability_sum,
        "top_actions": top_actions,
        "next_page_is_top": _NEXT_PAGE_ACTION in top_actions,
        "previous_page_is_top": _PREVIOUS_PAGE_ACTION in top_actions,
        "prompt": page.prompt,
        "actions": action_rows,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None, "p10": None, "p90": None}
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def summarize_results(page_results: Sequence[dict[str, Any]], requested_pages: int) -> dict[str, Any]:
    """Aggregate action-probability distributions and requested top-action rates."""
    all_actions = [action for page in page_results for action in page["actions"]]
    kinds = sorted({action["kind"] for action in all_actions})
    next_eligible = [page for page in page_results if any(action["kind"] == "next_page" for action in page["actions"])]
    previous_eligible = [page for page in page_results if any(action["kind"] == "previous_page" for action in page["actions"])]
    correct_pages = [page for page in page_results if page["has_correct_product"]]
    pagination_top = [page for page in page_results if page["next_page_is_top"] or page["previous_page_is_top"]]
    return {
        "pages_requested": requested_pages,
        "pages_evaluated": len(page_results),
        "actions_scored": len(all_actions),
        "action_joint_probability_distribution": _distribution([action["joint_probability"] for action in all_actions]),
        "action_joint_probability_distribution_by_kind": {kind: _distribution([action["joint_probability"] for action in all_actions if action["kind"] == kind]) for kind in kinds},
        "next_page_top_count": sum(page["next_page_is_top"] for page in next_eligible),
        "next_page_pages_available": len(next_eligible),
        "next_page_top_percentage_when_available": 100.0 * sum(page["next_page_is_top"] for page in next_eligible) / len(next_eligible) if next_eligible else None,
        "previous_page_top_count": sum(page["previous_page_is_top"] for page in previous_eligible),
        "previous_page_pages_available": len(previous_eligible),
        "previous_page_top_percentage_when_available": 100.0 * sum(page["previous_page_is_top"] for page in previous_eligible) / len(previous_eligible) if previous_eligible else None,
        "pagination_action_top_count": len(pagination_top),
        "pagination_action_top_percentage_all_pages": 100.0 * len(pagination_top) / len(page_results) if page_results else None,
        "pages_with_correct_products": len(correct_pages),
        "average_correct_product_joint_probability": float(np.mean([page["correct_product_joint_probability"] for page in correct_pages])) if correct_pages else None,
        "correct_product_joint_probability_distribution": _distribution([page["correct_product_joint_probability"] for page in correct_pages]),
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Create a concise human-readable companion to the complete JSON report."""
    summary = report["summary"]
    config = report["configuration"]

    def display(value: Any, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}g}"
        return str(value)

    lines = [
        "# Search-results pseudo-probability testbed",
        "",
        f"Model: `{config['model']}`  ",
        f"Seed: `{config['seed']}`  ",
        f"Pages: `{summary['pages_evaluated']}` of `{summary['pages_requested']}` requested  ",
        "Agent rollouts performed: `no`",
        "",
        "## Aggregate analysis",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Actions scored | {summary['actions_scored']} |",
        f"| Next-page top when available | {display(summary['next_page_top_percentage_when_available'])}% ({summary['next_page_top_count']}/{summary['next_page_pages_available']}) |",
        f"| Previous-page top when available | {display(summary['previous_page_top_percentage_when_available'])}% ({summary['previous_page_top_count']}/{summary['previous_page_pages_available']}) |",
        f"| Either pagination action top across all pages | {display(summary['pagination_action_top_percentage_all_pages'])}% ({summary['pagination_action_top_count']}/{summary['pages_evaluated']}) |",
        f"| Pages containing a full-reward product | {summary['pages_with_correct_products']} |",
        f"| Average full-reward-product joint probability | {display(summary['average_correct_product_joint_probability'])} |",
        "",
        "Joint action probabilities are raw sequence probabilities and therefore do not sum to one. Tied top actions are all counted.",
        "",
        "## Probability distribution by action kind",
        "",
        "| Kind | Count | Mean | Median | Min | Max | P10 | P90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, distribution in summary["action_joint_probability_distribution_by_kind"].items():
        lines.append(f"| {kind} | {distribution['count']} | {display(distribution['mean'])} | {display(distribution['median'])} | {display(distribution['minimum'])} | {display(distribution['maximum'])} | {display(distribution['p10'])} | {display(distribution['p90'])} |")
    lines.extend(
        [
            "",
            "## Per-page results",
            "",
            "| Query | Page | Correct products | Correct probability | Top action(s) |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for page in report["pages"]:
        query = page["query"].replace("|", "\\|")
        top_actions = ", ".join(page["top_actions"]).replace("|", "\\|")
        lines.append(f"| {query} | {page['page_number']} | {len(page['correct_product_asins'])} | {display(page['correct_product_joint_probability'])} | {top_actions} |")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_path: Path, markdown_path: Path) -> None:
    """Atomically write JSON details and the Markdown analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_path.with_name(output_path.name + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary_json, output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_markdown = markdown_path.with_name(markdown_path.name + ".tmp")
    temporary_markdown.write_text(format_markdown_summary(report))
    os.replace(temporary_markdown, markdown_path)


def _configure_vllm_engine() -> None:
    if "vllm" in sys.modules:
        raise RuntimeError("Run this testbed as a module so VLLM_USE_V1 can be configured before importing vLLM")
    os.environ["VLLM_USE_V1"] = "0"


def _validate_run_arguments(args: argparse.Namespace) -> None:
    for name in ("num_pages", "tensor_parallel_size"):
        _validate_positive_integer(getattr(args, name), name)
    if args.max_model_len is not None:
        _validate_positive_integer(args.max_model_len, "max_model_len")
    if not math.isfinite(args.gpu_memory_utilization) or not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")


def _build_vllm_engine_kwargs(args: argparse.Namespace, tokenizer_name: str) -> dict[str, Any]:
    """Build engine settings compatible with V0 prompt-logprob scoring."""
    engine_kwargs = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        # vLLM V0 cannot combine prompt_logprobs with prefix caching. Repeated
        # action probes share a long prefix and otherwise crash in get_logprobs.
        "enable_prefix_caching": False,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    return engine_kwargs


def run_testbed(args: argparse.Namespace) -> dict[str, Any]:
    """Generate real result pages, run pseudo scoring only, and write analysis."""
    _validate_run_arguments(args)
    _configure_vllm_engine()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    products, product_by_asin, prices, attribute_to_asins, goals, search_engine = load_webshop_assets(args.catalog, args.attributes, args.seed)
    specs = sample_search_pages(products, product_by_asin, attribute_to_asins, goals, search_engine, args.num_pages, args.seed)
    pages = render_results_pages(specs, prices)

    engine_kwargs = _build_vllm_engine_kwargs(args, tokenizer_name)
    inference_engine = LLM(**engine_kwargs)
    adapter = VLLMPromptLogProbAdapter(inference_engine)

    page_results = []
    for index, page in enumerate(pages, start=1):
        logger.info("Scoring page %d/%d with %d actions", index, len(pages), len(page.context_parts.admissible_actions))
        page_results.append(score_page_actions(page, tokenizer, adapter))

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
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "enable_prefix_caching": False,
            "history_length": 1,
            "search_query_source": "synthetic goal catalog query",
            "correct_product_definition": "some available option selection produces WebShop reward 1.0",
            "probability_definition": "exp(sum(action-token log probabilities)); answer tags excluded unless merged into an overlapping token",
            "agent_rollouts_performed": False,
            "available_training_goals": len(goals),
        },
        "summary": summarize_results(page_results, args.num_pages),
        "pages": page_results,
    }
    markdown_path = args.markdown_output or args.output.with_suffix(".md")
    write_report(report, args.output, markdown_path)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model name or local merged checkpoint accepted by vLLM")
    parser.add_argument("--tokenizer", help="Tokenizer name/path; defaults to --model")
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--attributes", type=Path, default=_DEFAULT_ATTRIBUTES)
    parser.add_argument("--num-pages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, help="Optional vLLM model context limit")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Full JSON report")
    parser.add_argument("--markdown-output", type=Path, help="Markdown analysis; defaults to the JSON path with a .md suffix")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_argument_parser().parse_args()
    report = run_testbed(args)
    summary = report["summary"]
    logger.info(
        "Wrote %d pages; pagination-top=%.2f%%; mean correct-product probability=%s",
        summary["pages_evaluated"],
        summary["pagination_action_top_percentage_all_pages"],
        summary["average_correct_product_joint_probability"],
    )
    logger.info("JSON report: %s", args.output)
    logger.info("Markdown summary: %s", args.markdown_output or args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
