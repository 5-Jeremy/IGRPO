import json
import math

import pytest

from treehca.analyze_pseudo_rollout_product_page_choices_results import analyze_report, write_plots, write_summaries


def _report():
    return {
        "pages": [
            {
                "asin": "match",
                "actions": [
                    {"label": "A", "action": "red", "pseudo_probability": 0.7, "empirical_probability_conditional": 0.6},
                    {"label": "B", "action": "blue", "pseudo_probability": 0.3, "empirical_probability_conditional": 0.4},
                ],
            },
            {
                "asin": "empirical-tie",
                "actions": [
                    {"label": "A", "action": "small", "pseudo_probability": 0.4, "empirical_probability_conditional": 0.5},
                    {"label": "B", "action": "large", "pseudo_probability": 0.6, "empirical_probability_conditional": 0.5},
                ],
            },
            {
                "asin": "no-recognized-actions",
                "actions": [
                    {"label": "A", "action": "buy", "pseudo_probability": 1.0, "empirical_probability_conditional": None},
                ],
            },
        ]
    }


def test_analyze_report_computes_entropies_and_tie_aware_top_action_agreement():
    analysis = analyze_report(_report())

    assert analysis["pages_evaluated"] == 2
    assert analysis["pages_excluded"] == 1
    assert analysis["top_action_agreement"]["matching_pages"] == 2
    assert analysis["top_action_agreement"]["fraction"] == 1.0
    assert len(analysis["action_points"]) == 4
    assert analysis["page_points"][0]["pseudo_entropy_nats"] == pytest.approx(-0.7 * math.log(0.7) - 0.3 * math.log(0.3))
    assert analysis["page_points"][1]["empirical_top_labels"] == ["A", "B"]


def test_analyze_report_rejects_partially_missing_empirical_distribution():
    report = _report()
    report["pages"][0]["actions"][1]["empirical_probability_conditional"] = None

    with pytest.raises(ValueError, match="mixes missing and present"):
        analyze_report(report)


def test_analyze_report_averages_conditioned_rollout_agreement_over_pages():
    report = _report()
    report["pages"][0]["conditioned_pseudo_rollouts"] = [
        {"format_valid": True, "projected_label": "A", "label_probabilities": {"A": 0.8, "B": 0.2}},
        {"format_valid": True, "projected_label": "B", "label_probabilities": {"A": 0.6, "B": 0.4}},
        {"format_valid": True, "projected_label": "A", "label_probabilities": {"A": 0.5, "B": 0.5}},
        {"format_valid": False, "projected_label": "A", "label_probabilities": {"A": 0.95, "B": 0.05}},
    ]
    report["pages"][1]["conditioned_pseudo_rollouts"] = [
        {"format_valid": True, "projected_label": "A", "label_probabilities": {"A": 0.3, "B": 0.7}},
        {"format_valid": True, "projected_label": None, "label_probabilities": {"A": 0.9, "B": 0.1}},
    ]

    analysis = analyze_report(report)
    agreement = analysis["conditioned_rollout_top_action_agreement"]
    by_threshold = {row["minimum_top_probability_exclusive"]: row for row in agreement["thresholds"]}

    assert agreement["pages_with_conditioned_rollouts"] == 2
    # Page rates are 2/3 and 0, so pages have equal weight rather than the four
    # valid rollouts being pooled into a 2/4 rate.
    assert by_threshold[None]["mean_page_match_fraction"] == pytest.approx(1 / 3)
    assert by_threshold[None]["eligible_rollouts"] == 4
    assert by_threshold[None]["matching_rollouts"] == 2
    assert by_threshold[0.5]["mean_page_match_fraction"] == pytest.approx(0.25)
    assert by_threshold[0.7]["mean_page_match_fraction"] == 1.0
    assert by_threshold[0.7]["pages_included"] == 1
    assert by_threshold[0.8]["mean_page_match_fraction"] is None


def test_writes_plots_and_machine_and_human_readable_summaries(tmp_path):
    analysis = analyze_report(_report())
    plot_paths = write_plots(analysis, tmp_path)
    summary = {
        "input_report": "input.json",
        "pages_in_report": analysis["pages_in_report"],
        "pages_evaluated": analysis["pages_evaluated"],
        "pages_excluded": analysis["pages_excluded"],
        "excluded_pages": analysis["excluded_pages"],
        "top_action_agreement": analysis["top_action_agreement"],
        "conditioned_rollout_top_action_agreement": analysis["conditioned_rollout_top_action_agreement"],
        "plots": plot_paths,
        "page_results": analysis["page_points"],
    }
    json_path, markdown_path = write_summaries(summary, tmp_path)

    assert all(tmp_path.joinpath(filename).stat().st_size > 0 for filename in ("probability_scatter_by_label.png", "entropy_scatter.png"))
    assert json.loads(json_path.read_text())["top_action_agreement"]["fraction"] == 1.0
    assert "Top-action agreement: **100.00%** (2/2)" in markdown_path.read_text()
    assert "## Per-rollout sampled-action agreement" in markdown_path.read_text()
