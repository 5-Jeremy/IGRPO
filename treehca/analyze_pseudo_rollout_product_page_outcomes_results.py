"""Analyze completed product-page continuation outcome reports.

Accepts both artificial-prefix reports (schema 2) and reports conditioned on
one shared page thought (schema 3).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

_DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "pseudo_prob_test_results/pseudo_rollout_product_page_outcomes_report.json"


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be a finite probability, got {value!r}")
    return float(value)


def _visited_product_subpage(trajectory: dict[str, Any], section: str) -> tuple[bool, bool]:
    """Prefer native post-step page state; infer older reports from executed clicks."""
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Each trajectory must contain a steps list")
    visited = False
    inferred = False
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each trajectory step must be an object")
        if "product_subpage_after_action" in step:
            visited |= step["product_subpage_after_action"] == section
        else:
            inferred = True
            visited |= step.get("executed") is True and step.get("projected_action") == f"click[{section}]"
    return visited, inferred


def _page_navigation_rows(pages: list[dict[str, Any]], *, count_partial_reward: bool) -> list[dict[str, Any]]:
    rows = []
    for index, page in enumerate(pages):
        groups = page.get("groups")
        trajectories = page.get("trajectories")
        outcomes = page["outcomes"]
        if not isinstance(groups, list) or not isinstance(trajectories, list):
            raise ValueError(f"pages[{index}] must contain groups and trajectories lists")
        correct = []
        for trajectory in trajectories:
            if not isinstance(trajectory, dict):
                raise ValueError(f"pages[{index}].trajectories must contain objects")
            reward = _probability(trajectory.get("raw_reward"), f"pages[{index}] trajectory raw_reward")
            if trajectory.get("purchased") and (reward > 0 if count_partial_reward else reward == 1.0):
                correct.append(trajectory)
        description_visits = 0
        features_visits = 0
        inferred = False
        for trajectory in correct:
            description, description_inferred = _visited_product_subpage(trajectory, "description")
            features, features_inferred = _visited_product_subpage(trajectory, "features")
            description_visits += description
            features_visits += features
            inferred |= description_inferred or features_inferred
        count = len(correct)
        purchase_rate = outcomes.get("reward_given_purchase_rate")
        rows.append({
            "source_page_index": page.get("source_page_index", index),
            "asin": page.get("asin"),
            "choice_groups": len(groups),
            "rollouts": len(trajectories),
            "no_purchase_rate": _probability(outcomes.get("no_purchase_rate"), f"pages[{index}].outcomes.no_purchase_rate"),
            "reward_purchase_rate": _probability(outcomes.get("reward_purchase_rate"), f"pages[{index}].outcomes.reward_purchase_rate"),
            "reward_given_purchase_rate": None if purchase_rate is None else _probability(purchase_rate, f"pages[{index}].outcomes.reward_given_purchase_rate"),
            "pseudo_reward_probability": _probability(outcomes.get("pseudo_reward_probability"), f"pages[{index}].outcomes.pseudo_reward_probability"),
            "correct_empirical_rollouts": count,
            "description_visit_count": description_visits,
            "description_visit_rate_among_correct": description_visits / count if count else None,
            "features_visit_count": features_visits,
            "features_visit_rate_among_correct": features_visits / count if count else None,
            "visit_evidence": "executed_click_inference" if inferred else "native_post_step_page",
        })
    return rows


def analyze_report(report: dict[str, Any], *, high_probability_threshold: float = 0.6) -> dict[str, Any]:
    """Compare pseudo reward probability with both empirical denominators."""
    threshold = _probability(high_probability_threshold, "high_probability_threshold")
    if not 0.5 < threshold < 1:
        raise ValueError("high_probability_threshold must be greater than 0.5 and less than 1")
    if report.get("status") != "complete" or report.get("mode") != "monte_carlo":
        raise ValueError("The input must be a completed monte_carlo report")
    if report.get("schema_version") not in (2, 3):
        raise ValueError("Expected a schema-version-2 or schema-version-3 product-page outcome report")
    pages = report.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("The input report must contain a nonempty pages list")
    configuration = report.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("The input report must contain a configuration object")
    pseudo_mode = configuration.get("pseudo_probability_mode", "artificial_group_thinking_prefix")
    if pseudo_mode not in ("artificial_group_thinking_prefix", "shared_page_thinking"):
        raise ValueError(f"Unrecognized pseudo_probability_mode: {pseudo_mode!r}")

    comparisons = {}
    for name, empirical_field in (("all_rollouts", "reward_purchase_rate"), ("given_purchase", "reward_given_purchase_rate")):
        points = []
        excluded = []
        for index, page in enumerate(pages):
            if not isinstance(page, dict) or not isinstance(page.get("outcomes"), dict):
                raise ValueError(f"pages[{index}] must contain an outcomes object")
            outcomes = page["outcomes"]
            pseudo = _probability(outcomes.get("pseudo_reward_probability"), f"pages[{index}].outcomes.pseudo_reward_probability")
            empirical_value = outcomes.get(empirical_field)
            if empirical_value is None:
                excluded.append({"source_page_index": page.get("source_page_index", index), "asin": page.get("asin"), "reason": "no purchases" if name == "given_purchase" else "no empirical rollouts"})
                continue
            empirical = _probability(empirical_value, f"pages[{index}].outcomes.{empirical_field}")
            points.append({
                "source_page_index": page.get("source_page_index", index),
                "asin": page.get("asin"),
                "pseudo_probability": pseudo,
                "empirical_probability": empirical,
                "signed_gap": pseudo - empirical,
                "absolute_gap": abs(pseudo - empirical),
                "pseudo_high": pseudo > threshold,
                "empirical_high": empirical > threshold,
            })
        pseudo_high = [point for point in points if point["pseudo_high"]]
        empirical_high = [point for point in points if point["empirical_high"]]
        both_high = sum(point["pseudo_high"] and point["empirical_high"] for point in points)
        decisive = [point for point in points if point["pseudo_high"] or point["empirical_high"]]
        comparisons[name] = {
            "pages_evaluated": len(points),
            "pages_excluded": len(excluded),
            "excluded_pages": excluded,
            "mean_absolute_gap": sum(point["absolute_gap"] for point in points) / len(points) if points else None,
            "pseudo_high_confirmation_rate": both_high / len(pseudo_high) if pseudo_high else None,
            "empirical_high_confirmation_rate": both_high / len(empirical_high) if empirical_high else None,
            "bidirectional_decisive_agreement": both_high / len(decisive) if decisive else None,
            "decisive_mismatches": [point for point in decisive if point["pseudo_high"] != point["empirical_high"]],
            "points": points,
        }
    count_partial_reward = bool(configuration.get("count_partial_reward", False))
    return {
        "schema_version": 1,
        "source_report_schema_version": report["schema_version"],
        "pseudo_probability_mode": pseudo_mode,
        "reward_event": "positive_native_reward" if count_partial_reward else "full_native_reward",
        "high_probability_threshold": threshold,
        "pages_in_report": len(pages),
        "comparisons": comparisons,
        "page_navigation": _page_navigation_rows(pages, count_partial_reward=count_partial_reward),
    }


def format_markdown_summary(analysis: dict[str, Any]) -> str:
    """Show both denominators and decisive mismatches in a compact report."""
    def display(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3%}"

    lines = [
        "# Product-page outcome probability analysis", "",
        f"Pseudo mode: `{analysis['pseudo_probability_mode']}`. Reward event: `{analysis['reward_event']}`.", "",
        f"High means strictly greater than {analysis['high_probability_threshold']:g}.", "",
        "| Empirical denominator | Evaluated pages | Excluded pages | Mean absolute gap | Pseudo high confirmed | Empirical high confirmed | Decisive agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (("all_rollouts", "All rollouts"), ("given_purchase", "Purchases")):
        comparison = analysis["comparisons"][name]
        cells = (
            label, str(comparison["pages_evaluated"]), str(comparison["pages_excluded"]),
            display(comparison["mean_absolute_gap"]), display(comparison["pseudo_high_confirmation_rate"]),
            display(comparison["empirical_high_confirmation_rate"]), display(comparison["bidirectional_decisive_agreement"]),
        )
        lines.append("| " + " | ".join(cells) + " |")
    reward_label = "Full or partial reward" if analysis["reward_event"] == "positive_native_reward" else "Full reward"
    pseudo_label = "Pseudo positive reward" if analysis["reward_event"] == "positive_native_reward" else "Pseudo full reward"
    lines.extend([
        "", "## Per-page visits among correct empirical rollouts", "",
        "Each description/features rate counts a rollout once if it reached that page at any time. The denominator is correct empirical rollouts under the selected reward event.", "",
        f"| ASIN | Choice groups | Rollouts | No purchase | {reward_label} / all | {reward_label} / purchases | {pseudo_label} | Correct rollouts | Description among correct | Features among correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in analysis["page_navigation"]:
        count = row["correct_empirical_rollouts"]
        description = f"{row['description_visit_count']}/{count} ({display(row['description_visit_rate_among_correct'])})" if count else "n/a"
        features = f"{row['features_visit_count']}/{count} ({display(row['features_visit_rate_among_correct'])})" if count else "n/a"
        cells = (
            str(row["asin"]), str(row["choice_groups"]), str(row["rollouts"]), display(row["no_purchase_rate"]),
            display(row["reward_purchase_rate"]), display(row["reward_given_purchase_rate"]),
            display(row["pseudo_reward_probability"]), str(count), description, features,
        )
        lines.append("| " + " | ".join(cells) + " |")
    if any(row["visit_evidence"] == "executed_click_inference" for row in analysis["page_navigation"]):
        lines.extend(["", "Older reports without recorded post-step page state infer visits from executed description/features clicks."])
    for name, label in (("all_rollouts", "All rollouts"), ("given_purchase", "Purchases")):
        lines.extend(["", f"## Decisive mismatches: {label}", ""])
        mismatches = analysis["comparisons"][name]["decisive_mismatches"]
        if not mismatches:
            lines.append("None.")
        else:
            lines.extend(["| ASIN | Pseudo probability | Empirical probability |", "| --- | ---: | ---: |"])
            lines.extend(f"| {point['asin']} | {display(point['pseudo_probability'])} | {display(point['empirical_probability'])} |" for point in mismatches)
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--high-probability-threshold", type=float, default=0.6)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    analysis = analyze_report(json.loads(args.input.read_text()), high_probability_threshold=args.high_probability_threshold)
    output_dir = args.output_dir or args.input.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "analysis_summary.md").write_text(format_markdown_summary(analysis))
    print(f"Wrote product-page outcome analysis to {output_dir}")


if __name__ == "__main__":
    main()
