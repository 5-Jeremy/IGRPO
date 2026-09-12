"""CPU-only tests for the pseudo-rollout probability testbed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treehca.product_page_parser import ProductPageContextParts
from treehca.pseudo_rollout import ActionChoiceScore, ProductPagePseudoRollout, ProductPagePseudoRolloutScores
from treehca.pseudo_rollout_testbed import (
    EmpiricalActionSamples,
    RenderedProductPage,
    SampledAgentResponse,
    ThinkingPrefix,
    average_pseudo_rollout_scores,
    build_conditioned_rollout_records,
    build_page_result,
    compare_action_distributions,
    extract_thinking_prefix,
    format_markdown_summary,
    project_completions,
    sample_diverse_products,
    sample_real_prompt_completions,
    summarize_page_results,
    write_report,
)


def _product(asin, action_options, category):
    return {
        "asin": asin,
        "Title": f"Product {asin}",
        "category": category,
        "options": {"size": [f"option-{asin}-{index}" for index in range(action_options)]},
    }


def _goal(asin, suffix=""):
    return {"asin": asin, "instruction_text": f"Training goal for {asin}{suffix}"}


def test_diverse_sampler_is_seeded_and_balances_action_counts():
    products = [
        _product("A1", 0, "alpha"),
        _product("A2", 0, "beta"),
        _product("B1", 1, "alpha"),
        _product("B2", 1, "gamma"),
        _product("C1", 2, "delta"),
        _product("C2", 2, "epsilon"),
    ]

    goals = [_goal(product["asin"]) for product in products]
    first = sample_diverse_products(products, goals, 3, seed=17)
    second = sample_diverse_products(products, goals, 3, seed=17)

    assert [(product["asin"], task) for product, task in first] == [(product["asin"], task) for product, task in second]
    assert {len(product["options"]["size"]) for product, _ in first} == {0, 1, 2}
    assert len({product["category"] for product, _ in first}) == 3


def test_diverse_sampler_excludes_parser_incompatible_products():
    valid = _product("VALID", 1, "alpha")
    no_title = _product("NO-TITLE", 1, "beta")
    no_title["Title"] = ""
    duplicate_option = _product("DUPLICATE", 1, "gamma")
    duplicate_option["options"] = {"size": ["size"]}

    products = [no_title, duplicate_option, valid]
    goals = [_goal(product["asin"]) for product in products]
    selected = sample_diverse_products(products, goals, 1, seed=0)

    assert selected[0][0]["asin"] == "VALID"
    with pytest.raises(ValueError, match="only 1 parser-compatible"):
        sample_diverse_products(products, goals, 2, seed=0)


def test_diverse_sampler_uses_only_generated_training_goals():
    with_goal = _product("WITH-GOAL", 1, "alpha")
    without_goal = _product("WITHOUT-GOAL", 2, "beta")
    goals = [_goal("WITH-GOAL", " first"), _goal("WITH-GOAL", " second")]

    selected = sample_diverse_products([with_goal, without_goal], goals, 1, seed=0)

    assert selected[0][0]["asin"] == "WITH-GOAL"
    assert selected[0][1] in {goal["instruction_text"] for goal in goals}
    with pytest.raises(ValueError, match="only 1 parser-compatible"):
        sample_diverse_products([with_goal, without_goal], goals, 2, seed=0)


def test_projection_counts_environment_actions_and_reports_format_separately():
    completions = ["well formatted", "missing think", "unknown action", "unparseable"]

    def projection(_responses):
        return ["click[first]", "click[second]", "click[unknown]", "tail"], [1, 0, 1, 0]

    result = project_completions(completions, ["click[first]", "click[second]"], projection=projection)

    assert result.action_counts == (1, 1)
    assert result.total_completions == 4
    assert result.format_valid_completions == 2
    assert result.recognized_completions == 2
    assert result.recognized_with_invalid_format == 1
    assert result.invalid_format_completions == 2
    assert result.inadmissible_completions == 2
    assert len(result.invalid_examples) == 3


def test_distribution_comparison_reports_effect_size_and_bootstrap_test():
    exact = compare_action_distributions([0.5, 0.5], [50, 50], bootstrap_replicates=500, seed=3)
    mismatch = compare_action_distributions([0.5, 0.5], [100, 0], bootstrap_replicates=500, seed=3)
    empty = compare_action_distributions([0.5, 0.5], [0, 0], bootstrap_replicates=500, seed=3)

    assert exact["total_variation_distance"] == pytest.approx(0.0)
    assert exact["jensen_shannon_divergence_nats"] == pytest.approx(0.0)
    assert exact["parametric_bootstrap_p_value"] == 1.0
    assert mismatch["total_variation_distance"] == pytest.approx(0.5)
    assert mismatch["parametric_bootstrap_p_value"] < 0.01
    assert empty["conditional_sample_size"] == 0
    assert empty["parametric_bootstrap_p_value"] is None


def test_distribution_comparison_supports_nonidentical_conditioned_null_rows():
    result = compare_action_distributions(
        [0.5, 0.5],
        [1, 1],
        bootstrap_replicates=100,
        seed=3,
        per_sample_expected_probabilities=[[1.0, 0.0], [0.0, 1.0]],
    )

    assert result["bootstrap_null"] == "poisson_multinomial_from_conditioned_rows"
    assert result["parametric_bootstrap_p_value"] == 1.0


class _FakeSamplingEngine:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outputs = []
        for row, params in enumerate(kwargs["sampling_params"]):
            candidates = [SimpleNamespace(token_ids=[row, sample]) for sample in range(params.n)]
            outputs.append(SimpleNamespace(outputs=candidates))
        return outputs


class _FakeTokenizer:
    def batch_decode(self, token_batches, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return ["-".join(str(token_id) for token_id in token_batch) for token_batch in token_batches]


def test_real_prompt_sampling_uses_one_n_way_request_per_page():
    engine = _FakeSamplingEngine()
    completions = sample_real_prompt_completions(
        engine,
        _FakeTokenizer(),
        [(1, 2), (3, 4)],
        samples_per_page=3,
        max_new_tokens=20,
        temperature=1.0,
        seed=10,
        page_batch_size=2,
    )

    assert [[completion.text for completion in page] for page in completions] == [["0-0", "0-1", "0-2"], ["1-0", "1-1", "1-2"]]
    assert [[completion.token_ids for completion in page] for page in completions] == [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)]]
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["prompts"] == [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4]}]
    assert [params.n for params in call["sampling_params"]] == [3, 3]
    assert [params.seed for params in call["sampling_params"]] == [10, 11]
    assert all(params.max_tokens == 20 and params.temperature == 1.0 for params in call["sampling_params"])
    assert all(params.top_p == 1.0 and params.top_k == -1 for params in call["sampling_params"])


class _ThinkingTokenizer:
    fragments = {1: "<think>", 2: "reason", 3: "</think>", 4: "<action>", 5: "click[first]"}

    def decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "".join(self.fragments[token_id] for token_id in token_ids)

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(range(len(text)))


def test_extract_thinking_prefix_replaces_closing_tag_and_action_with_decision_cue():
    completion = SampledAgentResponse("<think>reason</think><action>click[first]", (1, 2, 3, 4, 5))

    prefix = extract_thinking_prefix(completion, _ThinkingTokenizer())

    assert prefix.token_ids == tuple(range(len(prefix.text)))
    assert prefix.text == "<think>reason\nThe best next action is:"
    assert "</think>" not in prefix.text
    assert "<action>" not in prefix.text
    assert prefix.status == "complete_thinking_block"


def test_extract_thinking_prefix_uses_only_decision_cue_for_malformed_thinking():
    completion = SampledAgentResponse("reason<action>click[first]", (2, 4, 5))

    prefix = extract_thinking_prefix(completion, _ThinkingTokenizer())

    assert prefix.token_ids == tuple(range(len(prefix.text)))
    assert prefix.text == "The best next action is:"
    assert prefix.status == "no_complete_thinking_block"


def _page_inputs():
    actions = ("click[first]", "click[second]")
    parts = ProductPageContextParts(
        agent_introduction="Agent.",
        shopping_task="Find a thing",
        history_block=None,
        completed_steps=None,
        history_length=None,
        current_step=None,
        current_observation="[button] first [button] second",
        admissible_actions=actions,
        response_instructions="Respond.",
    )
    page = RenderedProductPage("ASIN", "category", "Find a thing", "real prompt", parts)
    pseudo = ProductPagePseudoRollout("pseudo prompt", (1, 2), ("A", "B"), actions, ((10, 11), (12, 13)))
    scores = ProductPagePseudoRolloutScores(
        choices=(
            ActionChoiceScore("A", actions[0], 0.6, -0.5108256238, -1.0, -1.0),
            ActionChoiceScore("B", actions[1], 0.4, -0.9162907319, -1.0, -1.0),
        )
    )
    empirical = EmpiricalActionSamples((6, 4), 12, 11, 10, 1, 1, 2, ({"response": "bad", "projected_action": "bad", "reason": "inadmissible_action"},))
    return page, pseudo, scores, empirical


def test_average_pseudo_scores_averages_each_aligned_action():
    _, _, first, _ = _page_inputs()
    second = ProductPagePseudoRolloutScores(
        choices=(
            ActionChoiceScore("A", "click[first]", 0.2, -1.6094379124, -1.0, -1.0),
            ActionChoiceScore("B", "click[second]", 0.8, -0.2231435513, -1.0, -1.0),
        )
    )

    average = average_pseudo_rollout_scores([first, second])

    assert average.label_probabilities == pytest.approx({"A": 0.4, "B": 0.6})


def test_conditioned_page_result_retains_every_probe_and_uses_conditioned_bootstrap():
    page, pseudo, first, _ = _page_inputs()
    second = ProductPagePseudoRolloutScores(
        choices=(
            ActionChoiceScore("A", "click[first]", 0.2, -1.6094379124, -1.0, -1.0),
            ActionChoiceScore("B", "click[second]", 0.8, -0.2231435513, -1.0, -1.0),
        )
    )
    empirical = EmpiricalActionSamples(
        (1, 1),
        2,
        2,
        2,
        0,
        0,
        0,
        (),
        projected_actions=("click[first]", "click[second]"),
        format_valids=(1, 1),
    )
    thinking = (
        ThinkingPrefix((101, 102), "<think>one</think>", "complete_thinking_block"),
        ThinkingPrefix((), "", "no_complete_thinking_block"),
    )
    records = build_conditioned_rollout_records(pseudo, [first, second], thinking, empirical)
    average = average_pseudo_rollout_scores([first, second])

    result = build_page_result(
        page,
        pseudo,
        average,
        empirical,
        real_prompt_tokens=7,
        bootstrap_replicates=100,
        seed=2,
        pseudo_probability_mode="mean_conditioned_on_sampled_thinking_for_recognized_actions",
        all_sample_scores=average,
        conditioned_rollouts=records,
        conditioned_comparison_scores=[first, second],
    )

    assert len(result["conditioned_pseudo_rollouts"]) == 2
    assert result["conditioned_pseudo_rollouts"][0]["projected_label"] == "A"
    assert result["conditioned_pseudo_rollouts"][0]["projected_action_probability"] == pytest.approx(0.6)
    assert result["conditioned_pseudo_rollouts"][0]["projected_action_is_pseudo_argmax"] is True
    assert result["complete_thinking_fraction"] == 0.5
    assert result["conditioned_top_action_agreement"] == 1.0
    assert result["actions"][0]["pseudo_probability_all_samples"] == pytest.approx(0.4)
    assert result["comparison_conditional_on_recognized_action"]["bootstrap_null"] == "poisson_multinomial_from_conditioned_rows"


def test_page_and_aggregate_reports_are_serializable_and_human_readable(tmp_path: Path):
    page, pseudo, scores, empirical = _page_inputs()
    page_result = build_page_result(page, pseudo, scores, empirical, real_prompt_tokens=7, bootstrap_replicates=100, seed=2)
    summary = summarize_page_results([page_result], requested_pages=1)
    report = {
        "configuration": {"model": "model", "seed": 2, "samples_per_page": 12, "temperature": 1.0},
        "summary": summary,
        "pages": [page_result],
    }

    assert page_result["actions"][0]["empirical_probability_conditional"] == pytest.approx(0.6)
    assert summary["overall_recognized_action_rate"] == pytest.approx(10 / 12)
    markdown = format_markdown_summary(report)
    assert "Pseudo-rollout action-probability testbed" in markdown
    assert "ASIN" in markdown

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    assert json.loads(json_path.read_text())["summary"]["pages_evaluated"] == 1
    assert markdown_path.read_text() == markdown
