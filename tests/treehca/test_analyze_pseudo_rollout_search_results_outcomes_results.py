import json

import pytest

from treehca.analyze_pseudo_rollout_search_results_outcomes_results import analyze_report, build_summary, write_plots, write_summaries


def _page(index, entry, success):
    return {
        "source_page_index": index,
        "asin": f"ITEM-{index}",
        "outcomes": {
            "pseudo_full_reward_product_entry_probability": entry[0],
            "empirical_full_reward_product_entry_probability": entry[1],
            "pseudo_success_probability": success[0],
            "empirical_success_probability": success[1],
        },
    }


def _report():
    return {
        "status": "complete",
        "mode": "monte_carlo",
        "pages": [
            _page(0, (0.8, 0.7), (0.8, 0.4)),
            _page(1, (0.7, 0.3), (0.3, 0.8)),
            _page(2, (0.2, 0.9), (0.8, 0.7)),
            _page(3, (0.4, 0.4), (0.4, 0.4)),
        ],
    }


def test_analysis_keeps_entry_and_success_agreement_independent():
    analysis = analyze_report(_report())

    entry = analysis["events"]["product_entry"]
    success = analysis["events"]["success"]
    assert entry["decisive_pages"] == 3
    assert success["decisive_pages"] == 3
    assert entry["bidirectional_decisive_agreement"]["fraction"] == pytest.approx(1 / 3)
    assert success["bidirectional_decisive_agreement"]["fraction"] == pytest.approx(1 / 3)
    assert {point["asin"] for point in entry["decisive_mismatches"]} == {"ITEM-1", "ITEM-2"}
    assert {point["asin"] for point in success["decisive_mismatches"]} == {"ITEM-0", "ITEM-1"}
    assert entry["pseudo_high_confirmation"]["fraction"] == pytest.approx(0.5)
    assert success["empirical_high_confirmation"]["fraction"] == pytest.approx(0.5)


def test_threshold_is_strict_and_missing_empirical_pages_are_excluded():
    report = _report()
    report["pages"] = [_page(0, (0.6, 0.6), (0.8, None))]

    analysis = analyze_report(report)

    assert analysis["events"]["product_entry"]["decisive_pages"] == 0
    assert analysis["events"]["success"]["pages_excluded"] == 1
    with pytest.raises(ValueError, match="greater than 0.5"):
        analyze_report(report, 0.5)
    with pytest.raises(ValueError, match="completed monte_carlo"):
        analyze_report({**report, "mode": "validate_only"})


def test_writes_separate_scatter_plots_and_summaries(tmp_path):
    analysis = analyze_report(_report())
    paths = write_plots(analysis, tmp_path)
    summary = build_summary(analysis, tmp_path / "input.json", paths)
    json_path, markdown_path = write_summaries(summary, tmp_path)

    for event in ("product_entry", "success"):
        for path in paths[event].values():
            assert tmp_path.joinpath(path).stat().st_size > 0
    assert paths["product_entry"]["probability_scatter"] != paths["success"]["probability_scatter"]
    assert json.loads(json_path.read_text())["events"]["product_entry"]["decisive_pages"] == 3
    assert "Full-reward product entry" in markdown_path.read_text()
    assert "Full-reward purchase success" in markdown_path.read_text()
