import json

import pytest

from treehca.analyze_pseudo_rollout_product_page_non_option_choices_results import analyze_report, build_summary, write_plots, write_summaries


def _page(asin, pseudo, empirical, counts=None):
    actions = ("click[back to search]", "click[description]", "click[buy now]", "click[red]")
    counts = counts or (10, 20, 30, 40)
    return {
        "asin": asin,
        "actions": [
            {
                "label": chr(ord("A") + index),
                "action": action,
                "pseudo_probability": pseudo[index],
                "empirical_probability_conditional": empirical[index],
                "empirical_count": counts[index],
            }
            for index, action in enumerate(actions)
        ],
    }


def test_analysis_compares_control_mass_and_distribution_within_controls():
    report = {"pages": [_page("match", (0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4))]}

    analysis = analyze_report(report)

    assert len(analysis["action_points"]) == 3
    assert analysis["mean_pseudo_non_option_mass"] == pytest.approx(0.6)
    assert analysis["mean_empirical_non_option_mass"] == pytest.approx(0.6)
    assert analysis["mean_absolute_non_option_mass_error"] == pytest.approx(0.0)
    assert analysis["mean_non_option_conditional_total_variation_distance"] == pytest.approx(0.0)
    assert analysis["top_non_option_action_agreement"]["fraction"] == 1.0
    assert {row["action"] for row in analysis["action_summaries"]} == {"click[back to search]", "click[description]", "click[buy now]"}


def test_analysis_separates_total_control_mass_from_conditional_control_mix():
    report = {"pages": [_page("scaled", (0.1, 0.2, 0.3, 0.4), (0.05, 0.1, 0.15, 0.7))]}

    analysis = analyze_report(report)

    assert analysis["mean_absolute_non_option_mass_error"] == pytest.approx(0.3)
    assert analysis["mean_non_option_conditional_total_variation_distance"] == pytest.approx(0.0)


def test_analysis_excludes_pages_without_recognized_empirical_actions():
    report = {"pages": [_page("empty", (0.1, 0.2, 0.3, 0.4), (None, None, None, None), counts=(0, 0, 0, 0))]}

    analysis = analyze_report(report)

    assert analysis["pages_evaluated"] == 0
    assert analysis["pages_excluded"] == 1


def test_writes_plots_and_summaries(tmp_path):
    analysis = analyze_report({"pages": [_page("match", (0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4))]})
    plot_paths = write_plots(analysis, tmp_path)
    summary = build_summary(analysis, tmp_path / "input.json", plot_paths)
    json_path, markdown_path = write_summaries(summary, tmp_path)

    assert all(tmp_path.joinpath(name).stat().st_size > 0 for name in ("non_option_probability_scatter.png", "non_option_mass_scatter.png"))
    assert json.loads(json_path.read_text())["mean_pseudo_non_option_mass"] == pytest.approx(0.6)
    assert "Mean pseudo non-option mass: **60.00%**" in markdown_path.read_text()
