"""CPU-only tests for the grouped-choice probability testbed."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from treehca import pseudo_rollout_product_page_grouped_choices_testbed as testbed
from treehca.product_page_parser import ProductOptionGroup, ProductPageContextParts
from treehca.pseudo_rollout_product_page import ActionChoiceScore, ProductOptionGroupPseudoRollout, ProductOptionGroupPseudoRolloutScores
from treehca.pseudo_rollout_product_page_choices_testbed import RenderedProductPage, SampledAgentResponse
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import (
    EmpiricalGroupOptionSamples,
    GroupResponsePrefix,
    SelectedProductGoal,
    build_argument_parser,
    build_artificial_group_response_prefixes,
    build_group_result,
    extract_generated_group_thinking_prefix,
    extract_page_thinking_prefix,
    format_markdown_summary,
    prepend_page_thinking_to_completions,
    project_group_completions,
    run_testbed,
    sample_diverse_product_goals,
    score_generated_thinking_group_pseudo_rollouts,
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
    _, pseudo, _, _ = _group_inputs()
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
    assert default.denominator_inclusions == (True, False, False, False, True)
    assert inclusive.denominator_inclusions == (True, False, True, False, True)


def test_command_line_flag_enables_inclusive_empirical_denominator():
    args = build_argument_parser().parse_args(["--include-valid-non-option-actions-in-empirical-denominator", "--condition-pseudo-on-thinking", "--pseudo-score-batch-size", "17"])

    assert args.include_valid_non_option_actions_in_empirical_denominator is True
    assert args.condition_pseudo_on_thinking is True
    assert args.pseudo_score_batch_size == 17


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


def test_generated_thinking_prefix_retains_thought_and_appends_group_specific_cue():
    _, pseudo, _, _ = _group_inputs()
    tokenizer = _PrefixTokenizer()

    page_thinking = extract_page_thinking_prefix(
        SampledAgentResponse("<think>Compare the requested colors. </think><action>click[red]</action>", (1, 2)), tokenizer
    )
    prefix = extract_generated_group_thinking_prefix(
        page_thinking,
        pseudo,
        tokenizer,
    )

    assert page_thinking.text == "<think>Compare the requested colors. </think>"
    assert prefix.text == "<think>Compare the requested colors.\nThe best choice for the color group corresponds to the label:"
    assert prefix.source == "generated_page_thinking"
    assert prefix.status == "complete_thinking_block"
    with pytest.raises(ValueError, match="no complete"):
        extract_page_thinking_prefix(SampledAgentResponse("No closing thought <action>click[red]</action>", (3, 4)), tokenizer)

    continued = prepend_page_thinking_to_completions((SampledAgentResponse("<action>click[blue]</action>", (9,)),), page_thinking)
    assert continued[0].text == "<think>Compare the requested colors. </think><action>click[blue]</action>"
    assert continued[0].token_ids == page_thinking.token_ids + (9,)


def test_generated_thinking_scores_each_group_once_from_shared_page_thought(monkeypatch):
    _, pseudo, scores, _ = _group_inputs()
    tokenizer = _PrefixTokenizer()
    page_thinking = extract_page_thinking_prefix(SampledAgentResponse("<think>first</think><action>click[red]</action>", (1,)), tokenizer)
    calls = []

    def fake_score(_engine, pseudo_batch, **kwargs):
        calls.append((tuple(pseudo_batch), kwargs))
        return [scores] * len(pseudo_batch)

    monkeypatch.setitem(score_generated_thinking_group_pseudo_rollouts.__globals__, "score_product_page_grouped_choice_rollouts", fake_score)
    group_scores, group_prefixes = score_generated_thinking_group_pseudo_rollouts(
        object(),
        tokenizer,
        [pseudo, pseudo],
        {2: page_thinking},
        max_model_len=100,
        batch_size=1,
    )

    assert group_scores == (scores, scores)
    assert group_prefixes[0] == group_prefixes[1]
    assert len(calls) == 2
    assert all(call[0] == (pseudo,) for call in calls)
    assert all(call[1]["max_model_len"] == 100 for call in calls)
    assert calls[0][1]["assistant_response_prefix_token_ids"] == [group_prefixes[0].token_ids]


def test_conditioned_run_shares_one_page_thought_between_empirical_and_pseudo(monkeypatch, tmp_path):
    page, pseudo, scores, _ = _group_inputs()
    pseudo = replace(pseudo, source_page_index=0)
    scores = replace(scores, source_page_index=0)
    tokenizer = _PrefixTokenizer()
    calls = []

    class FakeLLM:
        def __init__(self, **_kwargs):
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1000))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer)))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setattr(testbed, "_configure_vllm_engine", lambda: None)
    monkeypatch.setattr(testbed, "load_webshop_products_and_goals", lambda *_args: ([{}], [{}]))
    monkeypatch.setattr(testbed, "sample_diverse_product_goals", lambda *_args: (SelectedProductGoal({}, {"goal_options": {"color": "red"}, "instruction_text": "Find red"}),))
    monkeypatch.setattr(testbed, "render_product_pages", lambda *_args: (page,))
    monkeypatch.setattr(testbed, "prepare_product_page_grouped_choice_rollouts", lambda *_args: (pseudo,))
    monkeypatch.setattr(testbed, "tokenize_real_prompts", lambda *_args: ((91, 92),))
    monkeypatch.setattr(testbed, "required_max_logprobs", lambda *_args: 3)

    def fake_sample(_engine, _tokenizer, prompt_ids, **kwargs):
        calls.append(("sample", prompt_ids, kwargs["samples_per_page"]))
        if kwargs["samples_per_page"] == 1:
            return ((SampledAgentResponse("<think>shared</think><action>click[red]</action>", (7,)),),)
        return ((SampledAgentResponse("<action>click[red]</action>", (8,)), SampledAgentResponse("<action>click[blue]</action>", (9,))),)

    def fake_score(_engine, rollouts, **kwargs):
        calls.append(("score", tuple(rollouts), kwargs["assistant_response_prefix_token_ids"]))
        return [scores] * len(rollouts)

    def fake_project(completions, *_args, **_kwargs):
        calls.append(("project", tuple(completion.text for completion in completions)))
        return None

    def fake_result(*_args, **kwargs):
        calls.append(("result", kwargs))
        return {"source_page_index": 0}

    monkeypatch.setattr(testbed, "sample_real_prompt_completions", fake_sample)
    monkeypatch.setattr(testbed, "score_product_page_grouped_choice_rollouts", fake_score)
    monkeypatch.setattr(testbed, "project_group_completions", fake_project)
    monkeypatch.setattr(testbed, "build_group_result", fake_result)
    monkeypatch.setattr(testbed, "summarize_group_results", lambda *_args, **_kwargs: {"groups_evaluated": 1})
    monkeypatch.setattr(testbed, "write_report", lambda *_args: None)

    args = build_argument_parser().parse_args(["--condition-pseudo-on-thinking", "--samples-per-page", "2", "--max-new-tokens", "20", "--output", str(tmp_path / "report.json")])
    report = run_testbed(args)

    thought_ids = tuple(range(len("<think>shared</think>")))
    assert calls[0] == ("sample", ((91, 92),), 1)
    assert calls[1] == ("sample", ((91, 92) + thought_ids,), 2)
    assert calls[2][0] == "score" and calls[2][1] == (pseudo,)
    assert calls[2][2] == [tuple(range(len("<think>shared\nThe best choice for the color group corresponds to the label:")))]
    assert calls[3] == ("project", ("<think>shared</think><action>click[red]</action>", "<think>shared</think><action>click[blue]</action>"))
    assert calls[4][1]["pseudo_probability_mode"] == "shared_page_thinking"
    assert report["configuration"]["pseudo_probability_mode"] == "shared_page_thinking"


def test_group_result_records_shared_thought_without_its_text():
    page, pseudo, scores, empirical = _group_inputs()
    prefix = GroupResponsePrefix("<think>private\nGroup cue:", (1, 2, 3), "generated_page_thinking", "complete_thinking_block")
    result = build_group_result(
        page,
        pseudo,
        scores,
        empirical,
        real_prompt_tokens=20,
        bootstrap_replicates=100,
        seed=3,
        pseudo_probability_mode="shared_page_thinking",
        shared_thinking_prefix=prefix,
        shared_thinking_tokens=2,
    )

    assert result["pseudo_correct_probability"] == pytest.approx(0.75)
    assert result["shared_page_thinking"] == {"status": "complete_thinking_block", "empirical_prefix_tokens": 2, "pseudo_prefix_tokens": 3}
    assert "private" not in json.dumps(result)


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


def test_none_is_unobservable_in_one_turn_and_click_comparison_is_conditioned():
    import math
    from dataclasses import replace

    page, pseudo, scores, _ = _group_inputs()
    pseudo = replace(pseudo, labels=(*pseudo.labels, "D"), actions=(*pseudo.actions, "none"), variant_token_ids=(*pseudo.variant_token_ids, (16, 17)), include_none=True)
    scores = replace(scores, choices=tuple(replace(choice, probability=choice.probability / 2) for choice in scores.choices) + (ActionChoiceScore("D", "none", 0.5, math.log(0.5), -1, -1),))
    empirical = project_group_completions(
        ["red", "buy", "blue", "none"],
        pseudo,
        admissible_actions=(*pseudo.actions[:-1], "click[buy now]"),
        all_option_actions=pseudo.actions,
        projection=lambda _: (["click[red]", "click[buy now]", "click[blue]", "none"], [1, 1, 1, 0]),
    )
    assert empirical.action_counts == (1, 0, 1, 0)
    assert empirical.probability_denominator == 2
    result = build_group_result(page, pseudo, scores, empirical, real_prompt_tokens=20, bootstrap_replicates=100, seed=0)
    assert result["none_choice"]["pseudo_probability"] == 0.5
    assert result["none_choice"]["empirical_probability"] is None
    assert len(result["options"]) == 3
    assert sum(row["pseudo_probability"] for row in result["options"]) == pytest.approx(1)
    assert result["pseudo_correct_probability"] == pytest.approx(0.75)
    assert result["empirical_correct_probability_conditional"] == 0.5
    assert all(row["action"] != "none" for row in result["options"])
