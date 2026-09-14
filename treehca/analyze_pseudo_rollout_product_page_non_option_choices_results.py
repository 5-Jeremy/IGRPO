"""Compare pseudo and empirical probabilities for non-option product-page actions.

Example:
    python -m treehca.analyze_pseudo_rollout_product_page_non_option_choices_results
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
_DEFAULT_INPUT = _RESULTS_ROOT / "pseudo_rollout_calibration_report.json"
_NON_OPTION_ACTIONS = (
    "click[back to search]",
    "click[< prev]",
    "click[description]",
    "click[features]",
    "click[attributes]",
    "click[buy now]",
)


def _finite_probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number, got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a finite probability, got {value!r}")
    return probability


def _maximizers(rows: list[dict[str, Any]], field: str) -> list[str]:
    maximum = max(row[field] for row in rows)
    tolerance = max(1e-12, abs(maximum) * 1e-12)
    return [row["action"] for row in rows if math.isclose(row[field], maximum, rel_tol=0.0, abs_tol=tolerance)]


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    """Analyze control-action mass and the distribution within those controls."""
    pages = report.get("pages")
    if not isinstance(pages, list):
        raise ValueError("The input report must contain a 'pages' list")

    control_set = set(_NON_OPTION_ACTIONS)
    action_points: list[dict[str, Any]] = []
    page_points: list[dict[str, Any]] = []
    excluded_pages: list[dict[str, str]] = []

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"pages[{page_index}] must be an object")
        asin = str(page.get("asin", f"page-{page_index}"))
        actions = page.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Page {asin!r} must contain a nonempty 'actions' list")

        control_rows = []
        empirical_missing = 0
        seen_actions = set()
        for action_index, row in enumerate(actions):
            if not isinstance(row, dict):
                raise ValueError(f"Page {asin!r} action {action_index} must be an object")
            action = row.get("action")
            if not isinstance(action, str) or not action:
                raise ValueError(f"Page {asin!r} action {action_index} has an invalid action")
            normalized_action = action.lower()
            if normalized_action in seen_actions:
                raise ValueError(f"Page {asin!r} contains duplicate action {action!r}")
            seen_actions.add(normalized_action)
            if normalized_action not in control_set:
                continue

            pseudo_probability = _finite_probability(row.get("pseudo_probability"), f"page {asin} action {action} pseudo_probability")
            empirical_value = row.get("empirical_probability_conditional")
            empirical_probability = None if empirical_value is None else _finite_probability(empirical_value, f"page {asin} action {action} empirical_probability_conditional")
            empirical_missing += int(empirical_probability is None)
            count = row.get("empirical_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"Page {asin!r} action {action!r} has an invalid empirical_count")
            control_rows.append(
                {
                    "asin": asin,
                    "label": str(row.get("label", "")),
                    "action": normalized_action,
                    "pseudo_probability": pseudo_probability,
                    "empirical_probability": empirical_probability,
                    "empirical_count": count,
                }
            )

        if not control_rows:
            raise ValueError(f"Page {asin!r} contains no recognized non-option product-page actions")
        if empirical_missing not in (0, len(control_rows)):
            raise ValueError(f"Page {asin!r} mixes missing and present empirical probabilities")

        pseudo_mass = sum(row["pseudo_probability"] for row in control_rows)
        if pseudo_mass > 1.0 + 1e-6:
            raise ValueError(f"Non-option pseudo probabilities for page {asin!r} sum to more than one")
        if empirical_missing:
            excluded_pages.append({"asin": asin, "reason": "no recognized empirical actions"})
            continue

        empirical_mass = sum(row["empirical_probability"] for row in control_rows)
        if empirical_mass > 1.0 + 1e-6:
            raise ValueError(f"Non-option empirical probabilities for page {asin!r} sum to more than one")
        for row in control_rows:
            row["signed_probability_error"] = row["pseudo_probability"] - row["empirical_probability"]
            row["absolute_probability_error"] = abs(row["signed_probability_error"])
            action_points.append(row)

        conditional_tv = None
        top_action_matches = None
        pseudo_top_actions: list[str] = []
        empirical_top_actions: list[str] = []
        if pseudo_mass > 0.0 and empirical_mass > 0.0:
            for row in control_rows:
                row["pseudo_probability_given_non_option"] = row["pseudo_probability"] / pseudo_mass
                row["empirical_probability_given_non_option"] = row["empirical_probability"] / empirical_mass
            conditional_tv = 0.5 * sum(abs(row["pseudo_probability_given_non_option"] - row["empirical_probability_given_non_option"]) for row in control_rows)
            pseudo_top_actions = _maximizers(control_rows, "pseudo_probability_given_non_option")
            empirical_top_actions = _maximizers(control_rows, "empirical_probability_given_non_option")
            top_action_matches = bool(set(pseudo_top_actions) & set(empirical_top_actions))

        page_points.append(
            {
                "asin": asin,
                "pseudo_non_option_mass": pseudo_mass,
                "empirical_non_option_mass": empirical_mass,
                "signed_mass_error": pseudo_mass - empirical_mass,
                "absolute_mass_error": abs(pseudo_mass - empirical_mass),
                "non_option_conditional_total_variation_distance": conditional_tv,
                "pseudo_top_non_option_actions": pseudo_top_actions,
                "empirical_top_non_option_actions": empirical_top_actions,
                "top_non_option_action_matches": top_action_matches,
            }
        )

    action_summaries = []
    for action in _NON_OPTION_ACTIONS:
        points = [point for point in action_points if point["action"] == action]
        if not points:
            continue
        action_summaries.append(
            {
                "action": action,
                "pages": len(points),
                "mean_pseudo_probability": sum(point["pseudo_probability"] for point in points) / len(points),
                "mean_empirical_probability": sum(point["empirical_probability"] for point in points) / len(points),
                "mean_signed_probability_error": sum(point["signed_probability_error"] for point in points) / len(points),
                "mean_absolute_probability_error": sum(point["absolute_probability_error"] for point in points) / len(points),
            }
        )

    conditional_pages = [point for point in page_points if point["non_option_conditional_total_variation_distance"] is not None]
    matching_pages = sum(bool(point["top_non_option_action_matches"]) for point in conditional_pages)
    return {
        "non_option_action_definition": list(_NON_OPTION_ACTIONS),
        "pages_in_report": len(pages),
        "pages_evaluated": len(page_points),
        "pages_excluded": len(excluded_pages),
        "excluded_pages": excluded_pages,
        "action_points": action_points,
        "page_points": page_points,
        "action_summaries": action_summaries,
        "mean_pseudo_non_option_mass": sum(point["pseudo_non_option_mass"] for point in page_points) / len(page_points) if page_points else None,
        "mean_empirical_non_option_mass": sum(point["empirical_non_option_mass"] for point in page_points) / len(page_points) if page_points else None,
        "mean_absolute_non_option_mass_error": sum(point["absolute_mass_error"] for point in page_points) / len(page_points) if page_points else None,
        "mean_non_option_conditional_total_variation_distance": (
            sum(point["non_option_conditional_total_variation_distance"] for point in conditional_pages) / len(conditional_pages) if conditional_pages else None
        ),
        "top_non_option_action_agreement": {
            "matching_pages": matching_pages,
            "evaluated_pages": len(conditional_pages),
            "fraction": matching_pages / len(conditional_pages) if conditional_pages else None,
            "tie_rule": "A match occurs when the pseudo and empirical sets of maximum-probability non-option actions overlap.",
        },
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
    """Plot individual control probabilities and total non-option mass."""
    if not analysis["page_points"]:
        raise ValueError("No pages have empirical probabilities, so there is nothing to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()

    figure, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for action in _NON_OPTION_ACTIONS:
        points = [point for point in analysis["action_points"] if point["action"] == action]
        if points:
            axis.scatter(
                [point["pseudo_probability"] for point in points],
                [point["empirical_probability"] for point in points],
                label=action.removeprefix("click[").removesuffix("]"),
                alpha=0.72,
                s=34,
            )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.set_xlabel("Pseudo probability")
    axis.set_ylabel("Empirical probability conditional on a recognized action")
    axis.set_title("Non-option action probabilities")
    axis.grid(alpha=0.2)
    axis.legend(title="Product-page control")
    probability_path = output_dir / f"non_option_probability_scatter.{image_format}"
    figure.savefig(probability_path, dpi=200)
    plt.close(figure)

    points = analysis["page_points"]
    figure, axis = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    axis.scatter(
        [point["pseudo_non_option_mass"] for point in points],
        [point["empirical_non_option_mass"] for point in points],
        color="#6a3d9a",
        alpha=0.78,
        s=42,
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.set_xlabel("Pseudo probability mass on non-option actions")
    axis.set_ylabel("Empirical mass on non-option actions")
    axis.set_title("Total non-option action mass by page")
    axis.grid(alpha=0.2)
    mass_path = output_dir / f"non_option_mass_scatter.{image_format}"
    figure.savefig(mass_path, dpi=200)
    plt.close(figure)
    return {"non_option_probability_scatter": str(probability_path), "non_option_mass_scatter": str(mass_path)}


def build_summary(analysis: dict[str, Any], input_path: Path, plot_paths: dict[str, str]) -> dict[str, Any]:
    """Remove raw plot points while retaining page and action summaries."""
    return {
        key: value
        for key, value in analysis.items()
        if key not in {"action_points"}
    } | {"input_report": str(input_path), "plots": plot_paths}


def write_summaries(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "non_option_analysis_summary.json"
    markdown_path = output_dir / "non_option_analysis_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def percentage(value: float | None) -> str:
        return "not available" if value is None else f"{value:.2%}"

    agreement = summary["top_non_option_action_agreement"]
    lines = [
        "# Non-option product-page action analysis",
        "",
        f"Input report: `{summary['input_report']}`",
        "",
        f"- Pages evaluated: {summary['pages_evaluated']} of {summary['pages_in_report']}",
        f"- Mean pseudo non-option mass: **{percentage(summary['mean_pseudo_non_option_mass'])}**",
        f"- Mean empirical non-option mass: **{percentage(summary['mean_empirical_non_option_mass'])}**",
        f"- Mean absolute mass error: **{percentage(summary['mean_absolute_non_option_mass_error'])}**",
        f"- Mean TV within non-option choices: **{percentage(summary['mean_non_option_conditional_total_variation_distance'])}**",
        f"- Top non-option action agreement: **{percentage(agreement['fraction'])}** ({agreement['matching_pages']}/{agreement['evaluated_pages']})",
        "",
        "## Per-action comparison",
        "",
        "| Action | Pages | Mean pseudo | Mean empirical | Mean signed error | Mean absolute error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["action_summaries"]:
        lines.append(
            f"| `{row['action']}` | {row['pages']} | {row['mean_pseudo_probability']:.2%} | {row['mean_empirical_probability']:.2%} | "
            f"{row['mean_signed_probability_error']:+.2%} | {row['mean_absolute_probability_error']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Non-option mass is measured within the full forced-choice distribution. TV and top-action agreement renormalize over non-option actions only, separating the choice among controls from the total probability assigned to controls.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=_DEFAULT_INPUT, help="Pseudo-rollout testbed JSON report")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <input stem>_non_option_analysis beside the input report")
    parser.add_argument("--image-format", choices=("png", "pdf", "svg"), default="png")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.with_suffix("").with_name(f"{input_path.stem}_non_option_analysis")).resolve()
    with input_path.open() as input_file:
        report = json.load(input_file)
    analysis = analyze_report(report)
    plot_paths = write_plots(analysis, output_dir, args.image_format)
    summary = build_summary(analysis, input_path, plot_paths)
    json_path, markdown_path = write_summaries(summary, output_dir)

    print(f"Mean pseudo non-option mass: {summary['mean_pseudo_non_option_mass']:.2%}")
    print(f"Mean empirical non-option mass: {summary['mean_empirical_non_option_mass']:.2%}")
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
