"""CPU-only tests for the search-results pseudo-probability testbed."""

from argparse import Namespace

import pytest
import torch

import treehca.pseudo_rollout_results_page_testbed as testbed
from treehca.product_page_parser import ProductPageContextParts
from treehca.pseudo_rollout_results_page_testbed import (
    RenderedResultsPage,
    SearchPageSpec,
    _action_kind,
    _build_vllm_engine_kwargs,
    format_markdown_summary,
    product_can_earn_full_reward,
    render_results_pages,
    score_page_actions,
    summarize_results,
)


def _product(asin, *, color="red"):
    return {
        "asin": asin,
        "Title": f"Product {asin}",
        "MainImage": "image.jpg",
        "Price": "$10.00",
        "options": {"color": [color]},
    }


def test_full_reward_product_requires_compatible_selectable_options():
    goal = {"goal_options": {"color": "red"}}
    seen_options = []

    def reward(product, goal, *, price, options):
        seen_options.append(options)
        return 1.0 if price < 20 and options == {"color": "red"} else 0.0

    assert product_can_earn_full_reward(_product("GOOD"), goal, 10.0, reward_function=reward)
    assert not product_can_earn_full_reward(_product("BAD", color="blue"), goal, 10.0, reward_function=reward)
    assert seen_options == [{"color": "red"}]


def test_renderer_includes_real_search_action_in_history():
    good = _product("GOOD")
    bad = _product("BAD", color="blue")
    goal = {
        "asin": "GOOD",
        "instruction_text": "Find a red product",
        "goal_options": {"color": "red"},
    }
    spec = SearchPageSpec(goal=goal, query="red product", page_number=2, total_results=12, products=(good, bad))

    pages = render_results_pages(
        [spec],
        {"GOOD": 10.0, "BAD": 10.0},
        reward_function=lambda product, goal, *, price, options: float(product["asin"] == "GOOD" and options == {"color": "red"}),
    )

    assert len(pages) == 1
    page = pages[0]
    assert page.context_parts.current_step == 2
    assert page.context_parts.history_length == 1
    assert "Action 1: 'search[red product]'" in page.context_parts.history_block
    assert "'Page 2 (Total results: 12)'" in page.context_parts.current_observation
    assert page.correct_product_asins == ("GOOD",)
    assert "click[next >]" in page.context_parts.admissible_actions
    assert "click[< prev]" in page.context_parts.admissible_actions
    assert "click[good]" in page.context_parts.admissible_actions


def test_action_kind_recognizes_navigation_and_displayed_products():
    displayed = {"ABC"}
    assert _action_kind("click[next >]", displayed) == ("next_page", None)
    assert _action_kind("click[< prev]", displayed) == ("previous_page", None)
    assert _action_kind("click[abc]", displayed) == ("product", "ABC")
    assert _action_kind("click[missing]", displayed) == ("other", None)


def test_vllm_prompt_logprob_engine_disables_prefix_caching():
    args = Namespace(
        model="model",
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=0.8,
        trust_remote_code=False,
        seed=3,
        max_model_len=4096,
    )

    engine_kwargs = _build_vllm_engine_kwargs(args, "tokenizer")

    assert engine_kwargs["enable_prefix_caching"] is False
    assert engine_kwargs["max_model_len"] == 4096


def test_page_scoring_uses_joint_probability_function_for_every_action(monkeypatch):
    actions = ("click[next >]", "click[good]", "click[bad]")
    probabilities = dict(zip(actions, (0.1, 0.3, 0.2)))
    calls = []

    def fake_probability(prompt, action, tokenizer, worker):
        calls.append((prompt, action, tokenizer, worker))
        return torch.tensor(probabilities[action], dtype=torch.float64)

    monkeypatch.setattr(testbed, "compute_results_page_answer_probability", fake_probability)
    goal = {"asin": "GOOD", "instruction_text": "Find a product", "goal_options": {}}
    spec = SearchPageSpec(goal=goal, query="product", page_number=1, total_results=2, products=(_product("GOOD"), _product("BAD")))
    parts = ProductPageContextParts(
        agent_introduction="intro",
        shopping_task="Find a product",
        history_block="history",
        completed_steps=1,
        history_length=1,
        current_step=2,
        current_observation="results",
        admissible_actions=actions,
        response_instructions="instructions",
    )
    page = RenderedResultsPage(spec=spec, prompt="original prompt", context_parts=parts, correct_product_asins=("GOOD",))

    result = score_page_actions(page, tokenizer="tokenizer", actor_rollout_wg="worker")

    assert [call[1] for call in calls] == list(actions)
    assert result["top_actions"] == ["click[good]"]
    assert result["correct_product_joint_probability"] == pytest.approx(0.3)


def test_summary_reports_requested_argmax_rates_and_correct_probability():
    pages = [
        {
            "next_page_is_top": True,
            "previous_page_is_top": False,
            "has_correct_product": True,
            "correct_product_joint_probability": 0.2,
            "actions": [
                {"kind": "next_page", "joint_probability": 0.3},
                {"kind": "product", "joint_probability": 0.2},
            ],
        },
        {
            "next_page_is_top": False,
            "previous_page_is_top": True,
            "has_correct_product": False,
            "correct_product_joint_probability": None,
            "actions": [
                {"kind": "next_page", "joint_probability": 0.1},
                {"kind": "previous_page", "joint_probability": 0.4},
                {"kind": "product", "joint_probability": 0.05},
            ],
        },
    ]

    summary = summarize_results(pages, requested_pages=2)

    assert summary["next_page_top_percentage_when_available"] == 50.0
    assert summary["previous_page_top_percentage_when_available"] == 100.0
    assert summary["pagination_action_top_percentage_all_pages"] == 100.0
    assert summary["pages_with_correct_products"] == 1
    assert summary["average_correct_product_joint_probability"] == pytest.approx(0.2)
    assert summary["action_joint_probability_distribution"]["count"] == 5


def test_markdown_explicitly_says_no_agent_rollouts():
    page = {
        "query": "red product",
        "page_number": 1,
        "correct_product_asins": [],
        "correct_product_joint_probability": None,
        "top_actions": ["click[next >]"],
    }
    summary = {
        "pages_requested": 1,
        "pages_evaluated": 1,
        "actions_scored": 1,
        "next_page_top_percentage_when_available": 100.0,
        "next_page_top_count": 1,
        "next_page_pages_available": 1,
        "previous_page_top_percentage_when_available": None,
        "previous_page_top_count": 0,
        "previous_page_pages_available": 0,
        "pagination_action_top_percentage_all_pages": 100.0,
        "pagination_action_top_count": 1,
        "pages_with_correct_products": 0,
        "average_correct_product_joint_probability": None,
        "action_joint_probability_distribution_by_kind": {"next_page": {"count": 1, "mean": 0.1, "median": 0.1, "minimum": 0.1, "maximum": 0.1, "p10": 0.1, "p90": 0.1}},
    }

    markdown = format_markdown_summary({"configuration": {"model": "model", "seed": 0}, "summary": summary, "pages": [page]})

    assert "Agent rollouts performed: `no`" in markdown
    assert "Average full-reward-product joint probability" in markdown
