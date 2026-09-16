"""Analyze aggregate correct-option probabilities in a grouped-choice report.

Example:
    python -m treehca.analyze_pseudo_rollout_product_page_grouped_choices_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_ROOT = _ROOT / "pseudo_prob_test_results"
_DEFAULT_INPUT = _RESULTS_ROOT / "pseudo_rollout_grouped_choice_calibration_report.json"
_DEFAULT_HIGH_PROBABILITY_THRESHOLD = 0.6


def _finite_probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number, got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a finite probability, got {value!r}")
    return probability


def _resolve_threshold(report: dict[str, Any], explicit_threshold: float | None) -> float:
    value = explicit_threshold
    if value is None:
        configuration = report.get("configuration", {})
        value = configuration.get("high_probability_threshold", _DEFAULT_HIGH_PROBABILITY_THRESHOLD) if isinstance(configuration, dict) else _DEFAULT_HIGH_PROBABILITY_THRESHOLD
    threshold = _finite_probability(value, "high_probability_threshold")
    if not 0.5 < threshold < 1.0:
        raise ValueError("high_probability_threshold must be greater than 0.5 and less than 1")
    return threshold


def analyze_report(report: dict[str, Any], high_probability_threshold: float | None = None) -> dict[str, Any]:
    """Measure high-probability agreement on each group's total correct mass."""
    groups = report.get("groups")
    if not isinstance(groups, list):
        raise ValueError("The input report must contain a 'groups' list")
    threshold = _resolve_threshold(report, high_probability_threshold)

    option_points: list[dict[str, Any]] = []
    correct_probability_points: list[dict[str, Any]] = []
    excluded_groups: list[dict[str, str]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{group_index}] must be an object")
        asin = str(group.get("asin", f"page-{group_index}"))
        group_name = group.get("group_name")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"Group for ASIN {asin!r} has an invalid group_name")
        group_id = f"{asin}:{group_name}"
        options = group.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"Group {group_id!r} must contain at least two options")

        labels: set[str] = set()
        pseudo_probabilities: list[float] = []
        empirical_probabilities: list[float] = []
        empirical_missing = 0
        parsed_options = []
        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                raise ValueError(f"Group {group_id!r} option {option_index} must be an object")
            label = option.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError(f"Group {group_id!r} option {option_index} has an invalid label")
            if label in labels:
                raise ValueError(f"Group {group_id!r} contains duplicate label {label!r}")
            labels.add(label)
            option_name = option.get("option")
            if not isinstance(option_name, str) or not option_name:
                raise ValueError(f"Group {group_id!r} label {label!r} has an invalid option name")
            pseudo_probability = _finite_probability(option.get("pseudo_probability"), f"group {group_id} label {label} pseudo_probability")
            empirical_value = option.get("empirical_probability_conditional")
            if empirical_value is None:
                empirical_missing += 1
                empirical_probability = None
            else:
                empirical_probability = _finite_probability(empirical_value, f"group {group_id} label {label} empirical_probability_conditional")
                empirical_probabilities.append(empirical_probability)
            is_correct = option.get("is_correct")
            if not isinstance(is_correct, bool):
                raise ValueError(f"Group {group_id!r} label {label!r} must have a boolean is_correct field")
            pseudo_probabilities.append(pseudo_probability)
            parsed_options.append((label, option_name, is_correct, pseudo_probability, empirical_probability))

        if not math.isclose(sum(pseudo_probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Pseudo probabilities for group {group_id!r} sum to {sum(pseudo_probabilities):.12g}, not 1")
        if not any(is_correct for _, _, is_correct, _, _ in parsed_options):
            raise ValueError(f"Group {group_id!r} has no correct options")
        if empirical_missing:
            if empirical_missing != len(options):
                raise ValueError(f"Group {group_id!r} mixes missing and present empirical probabilities")
            excluded_groups.append({"asin": asin, "group_name": group_name, "reason": "no recognized empirical group-option selections"})
            continue
        empirical_option_mass = sum(empirical_probabilities)
        monte_carlo = group.get("monte_carlo")
        if isinstance(monte_carlo, dict) and "empirical_option_probability_denominator" in monte_carlo:
            denominator = monte_carlo["empirical_option_probability_denominator"]
            recognized = monte_carlo.get("recognized_group_option_completions")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (denominator, recognized)) or denominator == 0 or recognized > denominator:
                raise ValueError(f"Group {group_id!r} has invalid empirical denominator counts")
            expected_option_mass = recognized / denominator
        else:
            expected_option_mass = 1.0
        if not math.isclose(empirical_option_mass, expected_option_mass, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Empirical option probabilities for group {group_id!r} sum to {empirical_option_mass:.12g}, expected {expected_option_mass:.12g} from its denominator counts")

        for label, option_name, is_correct, pseudo_probability, empirical_probability in parsed_options:
            assert empirical_probability is not None
            option_points.append(
                {
                    "source_page_index": group.get("source_page_index"),
                    "asin": asin,
                    "group_name": group_name,
                    "label": label,
                    "option": option_name,
                    "is_correct": is_correct,
                    "pseudo_probability": pseudo_probability,
                    "empirical_probability": empirical_probability,
                }
            )
        pseudo_correct_probability = sum(pseudo_probability for _, _, is_correct, pseudo_probability, _ in parsed_options if is_correct)
        empirical_correct_probability = sum(empirical_probability for _, _, is_correct, _, empirical_probability in parsed_options if is_correct and empirical_probability is not None)
        for report_field, calculated in (
            ("pseudo_correct_probability", pseudo_correct_probability),
            ("empirical_correct_probability_conditional", empirical_correct_probability),
        ):
            if report_field in group and not math.isclose(_finite_probability(group[report_field], f"group {group_id} {report_field}"), calculated, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Group {group_id!r} {report_field} does not equal the sum over correct options")
        pseudo_high = pseudo_correct_probability > threshold
        empirical_high = empirical_correct_probability > threshold
        decisive = pseudo_high or empirical_high
        correct_probability_points.append(
            {
                "source_page_index": group.get("source_page_index"),
                "asin": asin,
                "group_name": group_name,
                "correct_options": [option_name for _, option_name, is_correct, _, _ in parsed_options if is_correct],
                "pseudo_correct_probability": pseudo_correct_probability,
                "empirical_correct_probability": empirical_correct_probability,
                "absolute_probability_error": abs(pseudo_correct_probability - empirical_correct_probability),
                "pseudo_high": pseudo_high,
                "empirical_high": empirical_high,
                "decisive": decisive,
                "high_status_matches": pseudo_high == empirical_high,
            }
        )

    pseudo_high_points = [point for point in correct_probability_points if point["pseudo_high"]]
    empirical_high_points = [point for point in correct_probability_points if point["empirical_high"]]
    both_high_points = [point for point in correct_probability_points if point["pseudo_high"] and point["empirical_high"]]
    decisive_points = [point for point in correct_probability_points if point["decisive"]]
    mismatches = [point for point in decisive_points if not point["high_status_matches"]]
    return {
        "high_probability_threshold": threshold,
        "groups_in_report": len(groups),
        "groups_evaluated": len(correct_probability_points),
        "groups_excluded": len(excluded_groups),
        "groups_with_decisive_correct_mass": len(decisive_points),
        "excluded_groups": excluded_groups,
        "option_points": option_points,
        "correct_probability_points": correct_probability_points,
        "decisive_group_points": decisive_points,
        "pseudo_high_confirmation": {
            "confirmed": len(both_high_points),
            "total": len(pseudo_high_points),
            "fraction": len(both_high_points) / len(pseudo_high_points) if pseudo_high_points else None,
            "meaning": "Among groups with pseudo-high total correct mass, the fraction whose empirical total correct mass is also high.",
        },
        "empirical_high_confirmation": {
            "confirmed": len(both_high_points),
            "total": len(empirical_high_points),
            "fraction": len(both_high_points) / len(empirical_high_points) if empirical_high_points else None,
            "meaning": "Among groups with empirically high total correct mass, the fraction whose pseudo total correct mass is also high.",
        },
        "bidirectional_decisive_agreement": {
            "matching": len(both_high_points),
            "total": len(decisive_points),
            "fraction": len(both_high_points) / len(decisive_points) if decisive_points else None,
            "set_definition": "Groups whose total correct-option mass is above the threshold in either distribution; a match is above it in both.",
        },
        "mean_decisive_absolute_probability_error": (sum(point["absolute_probability_error"] for point in decisive_points) / len(decisive_points) if decisive_points else None),
        "decisive_mismatches": mismatches,
    }


def _load_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / f"treehca-matplotlib-{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_plots(analysis: dict[str, Any], output_dir: Path, image_format: str = "png") -> dict[str, str]:
    """Plot each group's aggregate correct probability."""
    points = analysis["correct_probability_points"]
    if not points:
        raise ValueError("No groups have empirical probabilities, so there is nothing to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    threshold = analysis["high_probability_threshold"]

    figure, axis = plt.subplots(figsize=(8.0, 7.5), constrained_layout=True)
    nondecisive = [point for point in points if not point["decisive"]]
    decisive_matches = [point for point in points if point["decisive"] and point["high_status_matches"]]
    decisive_mismatches = [point for point in points if point["decisive"] and not point["high_status_matches"]]
    if nondecisive:
        axis.scatter(
            [point["pseudo_correct_probability"] for point in nondecisive],
            [point["empirical_correct_probability"] for point in nondecisive],
            color="#c7c7c7",
            alpha=1.0,
            s=20,
            label="Not decisive",
        )
    if decisive_matches:
        axis.scatter(
            [point["pseudo_correct_probability"] for point in decisive_matches],
            [point["empirical_correct_probability"] for point in decisive_matches],
            color="#238b45",
            alpha=0.82,
            s=48,
            label="High in both",
        )
    if decisive_mismatches:
        axis.scatter(
            [point["pseudo_correct_probability"] for point in decisive_mismatches],
            [point["empirical_correct_probability"] for point in decisive_mismatches],
            color="#cb181d",
            marker="x",
            linewidths=1.8,
            s=62,
            label="High in one only",
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    axis.axvline(threshold, color="#555555", linestyle=":", linewidth=1.0)
    axis.axhline(threshold, color="#555555", linestyle=":", linewidth=1.0)
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.set_xlabel("Pseudo probability summed over correct options")
    axis.set_ylabel("Empirical correct-option probability conditional on group")
    axis.set_title("Aggregate correct-option probabilities by group")
    axis.grid(alpha=0.16)
    axis.legend(loc="lower right")
    scatter_path = output_dir / f"grouped_correct_probability_scatter.{image_format}"
    figure.savefig(scatter_path, dpi=200)
    plt.close(figure)

    directional = (analysis["pseudo_high_confirmation"], analysis["empirical_high_confirmation"])
    rates = [item["fraction"] if item["fraction"] is not None else 0.0 for item in directional]
    figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    bars = axis.bar(["Pseudo high\n→ empirical high", "Empirical high\n→ pseudo high"], rates, color=("#3182bd", "#756bb1"))
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Confirmation rate")
    axis.set_title(f"Directional agreement for aggregate correct mass > {threshold:g}")
    axis.grid(axis="y", alpha=0.2)
    for bar, item in zip(bars, directional):
        label = "n/a" if item["fraction"] is None else f"{item['fraction']:.1%}\n({item['confirmed']}/{item['total']})"
        axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.025, label, ha="center", va="bottom")
    agreement_path = output_dir / f"high_probability_directional_agreement.{image_format}"
    figure.savefig(agreement_path, dpi=200)
    plt.close(figure)
    return {"probability_scatter": str(scatter_path), "directional_agreement": str(agreement_path)}


def build_summary(analysis: dict[str, Any], input_path: Path, plot_paths: dict[str, str]) -> dict[str, Any]:
    """Drop bulky nondecisive plot records from the persisted summary."""
    return {
        "input_report": str(input_path),
        "high_probability_threshold": analysis["high_probability_threshold"],
        "groups_in_report": analysis["groups_in_report"],
        "groups_evaluated": analysis["groups_evaluated"],
        "groups_excluded": analysis["groups_excluded"],
        "groups_with_decisive_correct_mass": analysis["groups_with_decisive_correct_mass"],
        "excluded_groups": analysis["excluded_groups"],
        "pseudo_high_confirmation": analysis["pseudo_high_confirmation"],
        "empirical_high_confirmation": analysis["empirical_high_confirmation"],
        "bidirectional_decisive_agreement": analysis["bidirectional_decisive_agreement"],
        "mean_decisive_absolute_probability_error": analysis["mean_decisive_absolute_probability_error"],
        "decisive_mismatches": analysis["decisive_mismatches"],
        "plots": plot_paths,
    }


def write_summaries(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_summary.json"
    markdown_path = output_dir / "analysis_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def rate(item: dict[str, Any]) -> str:
        return "not available" if item["fraction"] is None else f"{item['fraction']:.2%} ({item['confirmed']}/{item['total']})"

    bidirectional = summary["bidirectional_decisive_agreement"]
    bidirectional_text = "not available" if bidirectional["fraction"] is None else f"{bidirectional['fraction']:.2%} ({bidirectional['matching']}/{bidirectional['total']})"
    error = summary["mean_decisive_absolute_probability_error"]
    lines = [
        "# Grouped-choice aggregate-correct-probability analysis",
        "",
        f"Input report: `{summary['input_report']}`",
        "",
        f"- High means strictly greater than {summary['high_probability_threshold']:g}.",
        f"- Groups evaluated: {summary['groups_evaluated']} of {summary['groups_in_report']}",
        f"- Groups with decisive correct-option mass: {summary['groups_with_decisive_correct_mass']}",
        f"- Pseudo correct mass high → empirical correct mass high: **{rate(summary['pseudo_high_confirmation'])}**",
        f"- Empirical correct mass high → pseudo correct mass high: **{rate(summary['empirical_high_confirmation'])}**",
        f"- Bidirectional decisive agreement: **{bidirectional_text}**",
        f"- Mean absolute error among decisive group aggregates: {'not available' if error is None else f'{error:.4f}'}",
        "",
        "Each point first sums all fuzzy-correct options in its group. The headline denominator is the union of groups whose aggregate correct mass is high in either distribution. Groups below the threshold in both do not inflate it.",
        "",
        "## Plots",
        "",
        f"- Probability scatter: `{summary['plots']['probability_scatter']}`",
        f"- Directional agreement: `{summary['plots']['directional_agreement']}`",
        "",
    ]
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=_DEFAULT_INPUT, help="Grouped-choice testbed JSON report")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <input stem>_analysis beside the input report")
    parser.add_argument("--image-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--high-probability-threshold", type=float, help="Override the report threshold; must be in (0.5, 1)")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.with_suffix("").with_name(f"{input_path.stem}_analysis")).resolve()
    with input_path.open() as input_file:
        report = json.load(input_file)
    analysis = analyze_report(report, args.high_probability_threshold)
    plot_paths = write_plots(analysis, output_dir, args.image_format)
    summary = build_summary(analysis, input_path, plot_paths)
    json_path, markdown_path = write_summaries(summary, output_dir)

    agreement = analysis["bidirectional_decisive_agreement"]
    agreement_text = "not available" if agreement["fraction"] is None else f"{agreement['fraction']:.2%} ({agreement['matching']}/{agreement['total']})"
    print(f"Bidirectional decisive agreement: {agreement_text}")
    print(f"Probability plot: {plot_paths['probability_scatter']}")
    print(f"Directional plot: {plot_paths['directional_agreement']}")
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
