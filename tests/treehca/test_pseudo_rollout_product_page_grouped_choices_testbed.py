"""CPU-only tests for the grouped-choice probability testbed."""

import json
from pathlib import Path

import pytest

from treehca.product_page_parser import ProductOptionGroup, ProductPageContextParts
from treehca.pseudo_rollout_product_page import ActionChoiceScore, ProductOptionGroupPseudoRollout, ProductOptionGroupPseudoRolloutScores
from treehca.pseudo_rollout_product_page_choices_testbed import RenderedProductPage
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import (
    EmpiricalGroupOptionSamples,
    GroupResponsePrefix,
    SelectedProductGoal,
    build_argument_parser,
    build_artificial_group_response_prefixes,
    build_group_result,
    format_markdown_summary,
    project_group_completions,
    sample_diverse_product_goals,
    summarize_group_results,
    write_report,
)


def _product(asin: str, category: str, values: list[str]) -> dict:
    return {
        "asin": asin,
        "Title": f"Product {asin}",
        "category": category,
        "options": {"color": values},
    }


def _goal(asin: str, color: str, suffix: str = "") -> dict:
    return {
        "asin": asin,
        "instruction_text": f"Find {asin} with color: {color}{suffix}",
        "goal_options": {"color": color},
    }


def test_sampler_keeps_exact_goal_options_and_excludes_trivial_groups():
    useful = _product("USEFUL", "alpha", ["red", "blue"])
    trivial = _product("TRIVIAL", "beta", ["only"])
    selected = sample_diverse_product_goals(
        [trivial, useful],
        [_goal("TRIVIAL", "only"), _goal("USEFUL", "red"), _goal("USEFUL", "blue")],
        num_pages=1,
        seed=4,
    )

    assert len(selected) == 1
    assert isinstance(selected[0], SelectedProductGoal)
    assert selected[0].product["asin"] == "USEFUL"
    assert selected[0].goal["goal_options"]["color"] in {"red", "blue"}


def _group_inputs():
    parts = ProductPageContextParts(
        agent_introduction="Agent.",
        shopping_task="Find red",
        history_block=None,
        completed_steps=None,
        history_length=None,
        current_step=None,
        current_observation="[button] red [button] scarlet [button] blue",
        admissible_actions=("click[red]", "click[scarlet]", "click[blue]"),
        response_instructions="Respond.",
    )
    page = RenderedProductPage("ASIN", "category", "Find red", "real prompt", parts)
    option_group = ProductOptionGroup("color", ("red", "scarlet", "blue"), ("red", "scarlet"))
    pseudo = ProductOptionGroupPseudoRollout(
        prompt="pseudo prompt",
        prompt_token_ids=(1, 2),
        labels=("A", "B", "C"),
        actions=("click[red]", "click[scarlet]", "click[blue]"),
        variant_token_ids=((10, 11), (12, 13), (14, 15)),
        source_page_index=2,
        option_group=option_group,
    )
    scores = ProductOptionGroupPseudoRolloutScores(
        choices=(
            ActionChoiceScore("A", "click[red]", 0.40, -0.9162907, -1.0, -1.0),
            ActionChoiceScore("B", "click[scarlet]", 0.35, -1.0498221, -1.0, -1.0),
            ActionChoiceScore("C", "click[blue]", 0.25, -1.3862944, -1.0, -1.0),
        ),
        source_page_index=2,
        option_group=option_group,
    )
    empirical = EmpiricalGroupOptionSamples(
        action_counts=(4, 4, 2),
        total_completions=12,
        format_valid_completions=11,
        recognized_completions=10,
        recognized_with_invalid_format=1,
        invalid_format_completions=1,
        inadmissible_completions=2,
        invalid_examples=(),
        probability_denominator=10,
    )
    return page, pseudo, scores, empirical


def test_optional_denominator_includes_valid_non_options_but_not_other_groups():
    page, pseudo, scores, _ = _group_inputs()
    completions = ["same red", "other size", "buy", "invalid", "same blue"]

    def projection(_responses):
        return ["click[red]", "click[large]", "click[buy now]", "not an action", "click[blue]"], [1, 1, 1, 1, 0]

    kwargs = {
        "admissible_actions": (*pseudo.actions, "click[large]", "click[buy now]"),
        "all_option_actions": (*pseudo.actions, "click[large]"),
        "projection": projection,
    }
    default = project_group_completions(completions, pseudo, **kwargs)
    inclusive = project_group_completions(completions, pseudo, include_valid_non_option_actions_in_denominator=True, **kwargs)

    assert default.action_counts == inclusive.action_counts == (1, 0, 1)
    assert default.probability_denominator == 2
    assert default.included_valid_non_option_completions == 0
    assert inclusive.probability_denominator == 3
    assert inclusive.included_valid_non_option_completions == 1
    assert inclusive.different_group_option_completions == 1
    assert inclusive.inadmissible_completions == 1
    assert inclusive.recognized_with_invalid_format == 1
    result = build_group_result(page, pseudo, scores, inclusive, real_prompt_tokens=20, bootstrap_replicates=100, seed=3)
    assert sum(option["empirical_probability_conditional"] for option in result["options"]) == pytest.approx(2 / 3)
    assert result["empirical_correct_probability_conditional"] == pytest.approx(1 / 3)
    assert result["monte_carlo"]["recognized_group_option_completions"] == 2
    assert result["monte_carlo"]["empirical_option_probability_denominator"] == 3


def test_command_line_flag_enables_inclusive_empirical_denominator():
    args = build_argument_parser().parse_args(["--include-valid-non-option-actions-in-empirical-denominator"])

    assert args.include_valid_non_option_actions_in_empirical_denominator is True


class _PrefixTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return list(range(len(text)))


def test_artificial_prefix_names_each_group_and_retains_replaceable_token_ids():
    _, pseudo, _, _ = _group_inputs()
    tokenizer = _PrefixTokenizer()

    (prefix,) = build_artificial_group_response_prefixes([pseudo], tokenizer)

    assert prefix == GroupResponsePrefix(
        text="<think>The best choice for the color group corresponds to the label:",
        token_ids=tuple(range(len(prefix.text))),
        source="artificial_group_thinking",
    )
    assert tokenizer.calls == [(prefix.text, {"add_special_tokens": False})]


def test_group_result_retains_multiple_correct_options_and_conditional_rates():
    page, pseudo, scores, empirical = _group_inputs()
    response_prefix = GroupResponsePrefix("<think>prefix:", (91, 92), "artificial_group_thinking")

    result = build_group_result(
        page,
        pseudo,
        scores,
        empirical,
        real_prompt_tokens=20,
        bootstrap_replicates=100,
        seed=3,
        response_prefix=response_prefix,
    )

    assert result["source_page_index"] == 2
    assert result["group_name"] == "color"
    assert result["correct_options"] == ["red", "scarlet"]
    assert result["pseudo_correct_probability"] == pytest.approx(0.75)
    assert result["empirical_correct_probability_conditional"] == pytest.approx(0.8)
    assert [row["is_correct"] for row in result["options"]] == [True, True, False]
    assert result["options"][0]["empirical_probability_conditional"] == pytest.approx(0.4)
    assert result["monte_carlo"]["recognized_group_option_rate"] == pytest.approx(10 / 12)
    assert result["pseudo_response_prefix"] == {
        "source": "artificial_group_thinking",
        "text": "<think>prefix:",
        "token_count": 2,
    }


def test_summary_aggregates_correct_options_before_high_probability_checks():
    page, pseudo, scores, empirical = _group_inputs()
    matching = build_group_result(page, pseudo, scores, empirical, real_prompt_tokens=20, bootstrap_replicates=100, seed=3)
    pseudo_only = json.loads(json.dumps(matching))
    pseudo_only["source_page_index"] = 3
    pseudo_only["options"][0]["pseudo_probability"] = 0.35
    pseudo_only["options"][1]["pseudo_probability"] = 0.35
    pseudo_only["options"][2]["pseudo_probability"] = 0.3
    pseudo_only["options"][0]["empirical_probability_conditional"] = 0.25
    pseudo_only["options"][1]["empirical_probability_conditional"] = 0.25
    pseudo_only["options"][2]["empirical_probability_conditional"] = 0.5
    pseudo_only["pseudo_correct_probability"] = 0.7
    pseudo_only["empirical_correct_probability_conditional"] = 0.5

    summary = summarize_group_results([matching, pseudo_only], requested_pages=4, high_probability_threshold=0.6)

    assert summary["pseudo_high_correct_groups"] == 2
    assert summary["empirical_high_correct_groups"] == 1
    assert summary["pseudo_high_correct_also_empirical_high"] == 1
    assert summary["pseudo_high_confirmation_rate"] == pytest.approx(0.5)
    assert summary["empirical_high_confirmation_rate"] == 1.0
    assert summary["bidirectional_high_agreement"] == pytest.approx(0.5)


def test_report_is_serializable_and_markdown_emphasizes_both_directions(tmp_path: Path):
    page, pseudo, scores, empirical = _group_inputs()
    group_result = build_group_result(page, pseudo, scores, empirical, real_prompt_tokens=20, bootstrap_replicates=100, seed=3)
    summary = summarize_group_results([group_result], requested_pages=1, high_probability_threshold=0.6)
    report = {
        "configuration": {"model": "model", "samples_per_page": 12},
        "summary": summary,
        "groups": [group_result],
    }

    markdown = format_markdown_summary(report)
    assert "Pseudo correct mass high implies empirical correct mass high" in markdown
    assert "Empirical correct mass high implies pseudo correct mass high" in markdown

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    assert json.loads(json_path.read_text())["summary"]["groups_evaluated"] == 1
    assert markdown_path.read_text() == markdown
