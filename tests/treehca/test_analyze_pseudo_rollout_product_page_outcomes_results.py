"""Analysis compatibility for ordinary and shared-thinking outcome reports."""

import copy
import json
import sys

import pytest

from treehca.analyze_pseudo_rollout_product_page_outcomes_results import analyze_report, format_markdown_summary, main


def _report() -> dict:
    def trajectory(reward, *sections):
        return {
            "purchased": reward is not None,
            "raw_reward": 0.0 if reward is None else reward,
            "steps": [{"projected_action": f"click[{section}]", "executed": True} for section in sections],
        }

    return {
        "schema_version": 2,
        "status": "complete",
        "mode": "monte_carlo",
        "configuration": {"pseudo_probability_mode": "artificial_group_thinking_prefix", "count_partial_reward": False},
        "pages": [
            {
                "source_page_index": 0, "asin": "A", "groups": [{"group_name": "color"}, {"group_name": "size"}],
                "trajectories": [trajectory(1.0, "description", "description"), trajectory(1.0, "features"), trajectory(1.0, "description", "features"), trajectory(0.5, "description", "features")],
                "outcomes": {"pseudo_reward_probability": 0.8, "no_purchase_rate": 0.0, "reward_purchase_rate": 0.75, "reward_given_purchase_rate": 0.75},
            },
            {
                "source_page_index": 1, "asin": "B", "groups": [{"group_name": "size"}],
                "trajectories": [trajectory(None, "description", "features"), trajectory(None)],
                "outcomes": {"pseudo_reward_probability": 0.8, "no_purchase_rate": 1.0, "reward_purchase_rate": 0.0, "reward_given_purchase_rate": None},
            },
        ],
    }


@pytest.mark.parametrize("conditioned", [False, True])
def test_analyzer_accepts_both_product_outcome_report_modes(conditioned):
    report = _report()
    if conditioned:
        report["schema_version"] = 3
        report["configuration"]["pseudo_probability_mode"] = "shared_page_thinking"
        report["configuration"]["condition_pseudo_on_thinking"] = True
        report["configuration"]["count_partial_reward"] = True
        for page in report["pages"]:
            page["shared_page_thinking"] = {"status": "complete_thinking_block", "assistant_prefix_token_count": 12}
            for trajectory in page["trajectories"]:
                for step in trajectory["steps"]:
                    section = step["projected_action"][6:-1]
                    step["product_subpage_after_action"] = section if section in {"description", "features"} else None
        report["pages"][0]["outcomes"].update(reward_purchase_rate=1.0, reward_given_purchase_rate=1.0)
    analysis = analyze_report(report)

    assert analysis["source_report_schema_version"] == (3 if conditioned else 2)
    assert analysis["pseudo_probability_mode"] == ("shared_page_thinking" if conditioned else "artificial_group_thinking_prefix")
    assert analysis["reward_event"] == ("positive_native_reward" if conditioned else "full_native_reward")
    assert analysis["comparisons"]["all_rollouts"]["pages_evaluated"] == 2
    assert analysis["comparisons"]["all_rollouts"]["mean_absolute_gap"] == pytest.approx(0.5 if conditioned else 0.425)
    assert analysis["comparisons"]["all_rollouts"]["bidirectional_decisive_agreement"] == 0.5
    assert analysis["comparisons"]["given_purchase"]["pages_excluded"] == 1
    assert analysis["comparisons"]["given_purchase"]["excluded_pages"][0]["asin"] == "B"
    navigation = analysis["page_navigation"]
    assert [row["choice_groups"] for row in navigation] == [2, 1]
    assert navigation[0]["correct_empirical_rollouts"] == (4 if conditioned else 3)
    assert navigation[0]["description_visit_count"] == (3 if conditioned else 2)
    assert navigation[0]["features_visit_count"] == (3 if conditioned else 2)
    assert navigation[0]["description_visit_rate_among_correct"] == pytest.approx(0.75 if conditioned else 2 / 3)
    assert navigation[0]["visit_evidence"] == ("native_post_step_page" if conditioned else "executed_click_inference")
    assert navigation[1]["correct_empirical_rollouts"] == 0
    assert navigation[1]["description_visit_rate_among_correct"] is None
    assert "| A | 2 | 4 |" in format_markdown_summary(analysis)
    assert "Decisive mismatches" in format_markdown_summary(analysis)


def test_analyzer_rejects_incomplete_or_invalid_outcome_reports():
    report = _report()
    report["status"] = "running"
    with pytest.raises(ValueError, match="completed"):
        analyze_report(report)
    report = _report()
    report["pages"][0]["outcomes"]["pseudo_reward_probability"] = float("nan")
    with pytest.raises(ValueError, match="finite probability"):
        analyze_report(report)
    report = copy.deepcopy(_report())
    report["configuration"]["pseudo_probability_mode"] = "unknown"
    with pytest.raises(ValueError, match="Unrecognized"):
        analyze_report(report)


def test_recorded_page_state_takes_precedence_over_a_projected_click():
    report = _report()
    for page in report["pages"]:
        for trajectory in page["trajectories"]:
            for step in trajectory["steps"]:
                step["product_subpage_after_action"] = None

    navigation = analyze_report(report)["page_navigation"]

    assert navigation[0]["description_visit_count"] == 0
    assert navigation[0]["features_visit_count"] == 0
    assert navigation[0]["visit_evidence"] == "native_post_step_page"


def test_analysis_cli_writes_shared_thinking_summary(monkeypatch, tmp_path):
    report = _report()
    report["schema_version"] = 3
    report["configuration"]["pseudo_probability_mode"] = "shared_page_thinking"
    input_path = tmp_path / "outcomes.json"
    input_path.write_text(json.dumps(report))
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["analyzer", str(input_path), "--output-dir", str(output_dir)])

    main()

    summary = json.loads((output_dir / "analysis_summary.json").read_text())
    assert summary["pseudo_probability_mode"] == "shared_page_thinking"
    markdown = (output_dir / "analysis_summary.md").read_text()
    assert "Pseudo mode: `shared_page_thinking`" in markdown
    assert "| ASIN | Choice groups | Rollouts | No purchase | Full reward / all | Full reward / purchases | Pseudo full reward |" in markdown
    assert "| A | 2 | 4 | 0.000% | 75.000% | 75.000% | 80.000% | 3 | 2/3 (66.667%) | 2/3 (66.667%) |" in markdown
