import json

import pytest

from treehca.analyze_pseudo_rollout_product_page_grouped_choices_results import analyze_report, build_summary, write_plots, write_summaries


def _group(asin, name, pseudo, empirical, correct_indices=(0,)):
    options = []
    for index, (pseudo_probability, empirical_probability) in enumerate(zip(pseudo, empirical)):
        options.append(
            {
                "label": chr(ord("A") + index),
                "option": f"option-{index}",
                "is_correct": index in correct_indices,
                "pseudo_probability": pseudo_probability,
                "empirical_probability_conditional": empirical_probability,
            }
        )
    pseudo_correct = sum(probability for index, probability in enumerate(pseudo) if index in correct_indices)
    empirical_correct = None if any(probability is None for probability in empirical) else sum(probability for index, probability in enumerate(empirical) if index in correct_indices)
    empirical_option_mass = None if any(probability is None for probability in empirical) else sum(empirical)
    denominator = 0 if empirical_option_mass is None else 100
    recognized = 0 if empirical_option_mass is None else round(empirical_option_mass * denominator)
    return {
        "source_page_index": 0,
        "asin": asin,
        "group_name": name,
        "pseudo_correct_probability": pseudo_correct,
        "empirical_correct_probability_conditional": empirical_correct,
        "monte_carlo": {
            "recognized_group_option_completions": recognized,
            "empirical_option_probability_denominator": denominator,
        },
        "options": options,
    }


def _report():
    return {
        "configuration": {"high_probability_threshold": 0.6},
        "groups": [
            _group("both", "color", [0.4, 0.35, 0.25], [0.35, 0.4, 0.25], (0, 1)),
            _group("pseudo-only", "size", [0.35, 0.35, 0.3], [0.25, 0.25, 0.5], (0, 1)),
            _group("empirical-only", "style", [0.3, 0.25, 0.45], [0.4, 0.4, 0.2], (0, 1)),
            _group("even", "material", [0.5, 0.5], [0.5, 0.5]),
            _group("excluded", "pack", [0.9, 0.1], [None, None]),
        ],
    }


def test_analysis_aggregates_correct_options_before_checking_both_directions():
    analysis = analyze_report(_report())

    assert analysis["groups_evaluated"] == 4
    assert analysis["groups_excluded"] == 1
    assert analysis["groups_with_decisive_correct_mass"] == 3
    assert len(analysis["option_points"]) == 11
    assert len(analysis["decisive_group_points"]) == 3
    assert analysis["pseudo_high_confirmation"]["fraction"] == pytest.approx(0.5)
    assert analysis["empirical_high_confirmation"]["fraction"] == pytest.approx(0.5)
    assert analysis["bidirectional_decisive_agreement"]["fraction"] == pytest.approx(1 / 3)
    assert {point["asin"] for point in analysis["decisive_mismatches"]} == {"pseudo-only", "empirical-only"}
    assert not any(point["asin"] == "even" for point in analysis["decisive_group_points"])
    both = next(point for point in analysis["decisive_group_points"] if point["asin"] == "both")
    assert both["pseudo_correct_probability"] == pytest.approx(0.75)
    assert both["empirical_correct_probability"] == pytest.approx(0.75)


def test_threshold_is_strict_and_can_be_overridden():
    report = {"groups": [_group("edge", "color", [0.6, 0.4], [0.6, 0.4])]}

    analysis = analyze_report(report, high_probability_threshold=0.6)

    assert analysis["decisive_group_points"] == []
    with pytest.raises(ValueError, match="greater than 0.5"):
        analyze_report(report, high_probability_threshold=0.5)


def test_analysis_rejects_partially_missing_empirical_distribution():
    report = {"groups": [_group("broken", "color", [0.7, 0.3], [0.8, None])]}

    with pytest.raises(ValueError, match="mixes missing and present"):
        analyze_report(report)


def test_analysis_allows_option_mass_below_one_with_inclusive_denominator():
    report = {"groups": [_group("buy-mass", "color", [0.35, 0.35, 0.3], [0.3, 0.2, 0.1], (0, 1))]}

    analysis = analyze_report(report)

    (point,) = analysis["decisive_group_points"]
    assert point["pseudo_correct_probability"] == pytest.approx(0.7)
    assert point["empirical_correct_probability"] == pytest.approx(0.5)
    assert analysis["pseudo_high_confirmation"]["fraction"] == 0.0


def test_writes_decisive_plots_and_machine_and_human_summaries(tmp_path):
    analysis = analyze_report(_report())
    plot_paths = write_plots(analysis, tmp_path)
    summary = build_summary(analysis, tmp_path / "input.json", plot_paths)
    json_path, markdown_path = write_summaries(summary, tmp_path)

    assert all(tmp_path.joinpath(filename).stat().st_size > 0 for filename in ("grouped_correct_probability_scatter.png", "high_probability_directional_agreement.png"))
    assert json.loads(json_path.read_text())["bidirectional_decisive_agreement"]["fraction"] == pytest.approx(1 / 3)
    markdown = markdown_path.read_text()
    assert "Pseudo correct mass high → empirical correct mass high" in markdown
    assert "Empirical correct mass high → pseudo correct mass high" in markdown
    assert "sums all fuzzy-correct options" in markdown
