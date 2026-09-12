"""Analyze probability calibration in a pseudo-rollout testbed JSON report.

Example:
    python -m treehca.analyze_pseudo_rollout_product_page_choices_results \
        outputs/pseudo_rollout_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_INPUT = Path("outputs/pseudo_rollout_calibration.json")


def _finite_probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number, got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a finite probability, got {value!r}")
    return probability


def _entropy(probabilities: list[float]) -> float:
    return -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)


def _maximizers(labels: list[str], probabilities: list[float]) -> list[str]:
    maximum = max(probabilities)
    tolerance = max(1e-12, abs(maximum) * 1e-12)
    return [label for label, probability in zip(labels, probabilities, strict=True) if math.isclose(probability, maximum, rel_tol=0.0, abs_tol=tolerance)]


def _label_sort_key(label: str) -> tuple[int, int | str]:
    """Sort spreadsheet-style labels A..Z, AA.. before arbitrary labels."""
    if label.isalpha() and label.isupper():
        value = 0
        for character in label:
            value = value * 26 + ord(character) - ord("A") + 1
        return (0, value)
    return (1, label)


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract plot records and top-action agreement from a testbed report."""
    pages = report.get("pages")
    if not isinstance(pages, list):
        raise ValueError("The input report must contain a 'pages' list")

    action_points: list[dict[str, Any]] = []
    page_points: list[dict[str, Any]] = []
    excluded_pages: list[dict[str, str]] = []
    matching_pages = 0

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"pages[{page_index}] must be an object")
        asin = str(page.get("asin", f"page-{page_index}"))
        actions = page.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Page {asin!r} must contain a nonempty 'actions' list")

        labels: list[str] = []
        pseudo_probabilities: list[float] = []
        empirical_probabilities: list[float] = []
        empirical_missing = 0
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"Page {asin!r} action {action_index} must be an object")
            label = action.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError(f"Page {asin!r} action {action_index} has an invalid label")
            if label in labels:
                raise ValueError(f"Page {asin!r} contains duplicate label {label!r}")
            labels.append(label)
            pseudo_probabilities.append(_finite_probability(action.get("pseudo_probability"), f"page {asin} label {label} pseudo_probability"))

            empirical = action.get("empirical_probability_conditional")
            if empirical is None:
                empirical_missing += 1
            else:
                empirical_probabilities.append(_finite_probability(empirical, f"page {asin} label {label} empirical_probability_conditional"))

        if not math.isclose(sum(pseudo_probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Pseudo probabilities for page {asin!r} sum to {sum(pseudo_probabilities):.12g}, not 1")
        if empirical_missing:
            if empirical_missing != len(actions):
                raise ValueError(f"Page {asin!r} mixes missing and present empirical probabilities")
            excluded_pages.append({"asin": asin, "reason": "no recognized empirical actions"})
            continue
        if not math.isclose(sum(empirical_probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Empirical probabilities for page {asin!r} sum to {sum(empirical_probabilities):.12g}, not 1")

        pseudo_top_labels = _maximizers(labels, pseudo_probabilities)
        empirical_top_labels = _maximizers(labels, empirical_probabilities)
        top_action_matches = bool(set(pseudo_top_labels) & set(empirical_top_labels))
        matching_pages += int(top_action_matches)

        pseudo_entropy = _entropy(pseudo_probabilities)
        empirical_entropy = _entropy(empirical_probabilities)
        page_points.append(
            {
                "asin": asin,
                "pseudo_entropy_nats": pseudo_entropy,
                "empirical_entropy_nats": empirical_entropy,
                "pseudo_top_labels": pseudo_top_labels,
                "empirical_top_labels": empirical_top_labels,
                "top_action_matches": top_action_matches,
            }
        )
        for action, label, pseudo_probability, empirical_probability in zip(actions, labels, pseudo_probabilities, empirical_probabilities, strict=True):
            action_points.append(
                {
                    "asin": asin,
                    "label": label,
                    "action": str(action.get("action", "")),
                    "pseudo_probability": pseudo_probability,
                    "empirical_probability": empirical_probability,
                }
            )

    evaluated_pages = len(page_points)
    return {
        "pages_in_report": len(pages),
        "pages_evaluated": evaluated_pages,
        "pages_excluded": len(excluded_pages),
        "excluded_pages": excluded_pages,
        "action_points": action_points,
        "page_points": page_points,
        "top_action_agreement": {
            "matching_pages": matching_pages,
            "evaluated_pages": evaluated_pages,
            "fraction": matching_pages / evaluated_pages if evaluated_pages else None,
            "tie_rule": "A match occurs when the pseudo and empirical sets of maximum-probability labels overlap.",
        },
    }


def _load_matplotlib():
    # Some cluster home directories are read-only. Set this before importing matplotlib.
    cache_dir = Path(tempfile.gettempdir()) / f"treehca-matplotlib-{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def write_plots(analysis: dict[str, Any], output_dir: Path, image_format: str = "png") -> dict[str, str]:
    """Write the requested scatter plots and return their paths."""
    if not analysis["page_points"]:
        raise ValueError("No pages have empirical probabilities, so there is nothing to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, np = _load_matplotlib()

    action_points = analysis["action_points"]
    labels = sorted({point["label"] for point in action_points}, key=_label_sort_key)
    label_indices = {label: index for index, label in enumerate(labels)}
    colors = plt.colormaps["turbo"](np.linspace(0.02, 0.98, max(len(labels), 2)))

    figure, axis = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    for label in labels:
        points = [point for point in action_points if point["label"] == label]
        axis.scatter(
            [point["pseudo_probability"] for point in points],
            [point["empirical_probability"] for point in points],
            color=colors[label_indices[label]],
            label=label,
            alpha=0.72,
            edgecolors="none",
            s=31,
        )
    probability_limit = max(
        1.0,
        max(max(point["pseudo_probability"], point["empirical_probability"]) for point in action_points),
    )
    axis.plot([0.0, probability_limit], [0.0, probability_limit], color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    axis.set(xlim=(0.0, probability_limit), ylim=(0.0, probability_limit))
    axis.set_xlabel("Pseudo probability")
    axis.set_ylabel("Empirical probability (conditional on a recognized action)")
    axis.set_title("Pseudo vs. empirical action probabilities")
    axis.grid(alpha=0.2)
    legend_columns = max(1, math.ceil(len(labels) / 18))
    axis.legend(title="Label", bbox_to_anchor=(1.02, 1.0), loc="upper left", ncol=legend_columns, fontsize=7, markerscale=0.9)
    probability_path = output_dir / f"probability_scatter_by_label.{image_format}"
    figure.savefig(probability_path, dpi=200)
    plt.close(figure)

    page_points = analysis["page_points"]
    pseudo_entropies = [point["pseudo_entropy_nats"] for point in page_points]
    empirical_entropies = [point["empirical_entropy_nats"] for point in page_points]
    entropy_limit = max(pseudo_entropies + empirical_entropies + [1e-9]) * 1.04
    figure, axis = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    axis.scatter(pseudo_entropies, empirical_entropies, color="#2468b4", alpha=0.8, s=42)
    axis.plot([0.0, entropy_limit], [0.0, entropy_limit], color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    axis.set(xlim=(0.0, entropy_limit), ylim=(0.0, entropy_limit))
    axis.set_xlabel("Pseudo-distribution entropy (nats)")
    axis.set_ylabel("Empirical-distribution entropy (nats)")
    axis.set_title("Pseudo vs. empirical action entropy")
    axis.grid(alpha=0.2)
    entropy_path = output_dir / f"entropy_scatter.{image_format}"
    figure.savefig(entropy_path, dpi=200)
    plt.close(figure)

    return {"probability_scatter": str(probability_path), "entropy_scatter": str(entropy_path)}


def _summary_without_plot_records(analysis: dict[str, Any], input_path: Path, plot_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "input_report": str(input_path),
        "pages_in_report": analysis["pages_in_report"],
        "pages_evaluated": analysis["pages_evaluated"],
        "pages_excluded": analysis["pages_excluded"],
        "excluded_pages": analysis["excluded_pages"],
        "top_action_agreement": analysis["top_action_agreement"],
        "plots": plot_paths,
        "page_results": analysis["page_points"],
    }


def write_summaries(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "analysis_summary.json"
    markdown_path = output_dir / "analysis_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    agreement = summary["top_action_agreement"]
    fraction = agreement["fraction"]
    fraction_text = "not available" if fraction is None else f"{fraction:.2%}"
    lines = [
        "# Pseudo-rollout probability analysis",
        "",
        f"Input report: `{summary['input_report']}`",
        "",
        f"- Pages evaluated: {summary['pages_evaluated']} of {summary['pages_in_report']}",
        f"- Pages excluded: {summary['pages_excluded']}",
        f"- Top-action agreement: **{fraction_text}** ({agreement['matching_pages']}/{agreement['evaluated_pages']})",
        f"- Tie handling: {agreement['tie_rule']}",
        "",
        "## Plots",
        "",
        f"- Probability scatter, colored by action label: `{summary['plots']['probability_scatter']}`",
        f"- Entropy scatter: `{summary['plots']['entropy_scatter']}`",
        "",
        "Entropies use natural logarithms and are reported in nats. Empirical probabilities are conditional on the model producing a recognized admissible action.",
        "",
    ]
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=_DEFAULT_INPUT, help="Pseudo-rollout testbed JSON report")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <input stem>_analysis beside the input report")
    parser.add_argument("--image-format", choices=("png", "pdf", "svg"), default="png")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.with_suffix("").with_name(f"{input_path.stem}_analysis")).resolve()
    with input_path.open() as input_file:
        report = json.load(input_file)
    analysis = analyze_report(report)
    plot_paths = write_plots(analysis, output_dir, args.image_format)
    summary = _summary_without_plot_records(analysis, input_path, plot_paths)
    json_path, markdown_path = write_summaries(summary, output_dir)

    agreement = analysis["top_action_agreement"]
    fraction = agreement["fraction"]
    agreement_text = "not available" if fraction is None else f"{fraction:.2%} ({agreement['matching_pages']}/{agreement['evaluated_pages']})"
    print(f"Top-action agreement: {agreement_text}")
    print(f"Probability plot: {plot_paths['probability_scatter']}")
    print(f"Entropy plot: {plot_paths['entropy_scatter']}")
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
