"""Analyze pseudo versus empirical search-results outcome probabilities.

Example:
    python -m treehca.analyze_pseudo_rollout_search_results_outcomes_results
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
_DEFAULT_INPUT = _ROOT / "pseudo_prob_test_results/pseudo_rollout_search_results_outcomes_report.json"
_DEFAULT_HIGH_PROBABILITY_THRESHOLD = 0.6
_EVENTS = {
    "product_entry": ("full_reward_product_entry", "Full-reward product entry"),
    "success": ("success", "Full-reward purchase success"),
}


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a finite probability, got {value!r}")
    return probability


def analyze_report(report: dict[str, Any], high_probability_threshold: float = _DEFAULT_HIGH_PROBABILITY_THRESHOLD) -> dict[str, Any]:
    """Compare both outcome rates across completed search-results starts."""
    threshold = _finite_probability(high_probability_threshold, "high_probability_threshold")
    if not 0.5 < threshold < 1.0:
        raise ValueError("high_probability_threshold must be greater than 0.5 and less than 1")
    if report.get("status") != "complete" or report.get("mode") != "monte_carlo":
        raise ValueError("The input must be a completed monte_carlo report")
    pages = report.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("The input report must contain a nonempty 'pages' list")

    events: dict[str, dict[str, Any]] = {}
    for event, (field_part, _) in _EVENTS.items():
        points = []
        excluded_pages = []
        for index, page in enumerate(pages):
            if not isinstance(page, dict) or not isinstance(page.get("outcomes"), dict):
                raise ValueError(f"pages[{index}] must contain an outcomes object")
            outcomes = page["outcomes"]
            pseudo = _finite_probability(outcomes.get(f"pseudo_{field_part}_probability"), f"pages[{index}] pseudo {event} probability")
            empirical_value = outcomes.get(f"empirical_{field_part}_probability")
            if empirical_value is None:
                excluded_pages.append({"source_page_index": page.get("source_page_index", index), "asin": page.get("asin"), "reason": "no empirical rollouts"})
                continue
            empirical = _finite_probability(empirical_value, f"pages[{index}] empirical {event} probability")
            pseudo_high = pseudo > threshold
            empirical_high = empirical > threshold
            points.append(
                {
                    "source_page_index": page.get("source_page_index", index),
                    "asin": page.get("asin"),
                    "pseudo_probability": pseudo,
                    "empirical_probability": empirical,
                    "absolute_probability_error": abs(pseudo - empirical),
                    "pseudo_high": pseudo_high,
                    "empirical_high": empirical_high,
                    "decisive": pseudo_high or empirical_high,
                    "high_status_matches": pseudo_high == empirical_high,
                }
            )
        pseudo_high_points = [point for point in points if point["pseudo_high"]]
        empirical_high_points = [point for point in points if point["empirical_high"]]
        both_high = sum(point["pseudo_high"] and point["empirical_high"] for point in points)
        decisive = [point for point in points if point["decisive"]]
        events[event] = {
            "pages_evaluated": len(points),
            "pages_excluded": len(excluded_pages),
            "excluded_pages": excluded_pages,
            "points": points,
            "decisive_pages": len(decisive),
            "pseudo_high_confirmation": {
                "confirmed": both_high,
                "total": len(pseudo_high_points),
                "fraction": both_high / len(pseudo_high_points) if pseudo_high_points else None,
            },
            "empirical_high_confirmation": {
                "confirmed": both_high,
                "total": len(empirical_high_points),
                "fraction": both_high / len(empirical_high_points) if empirical_high_points else None,
            },
            "bidirectional_decisive_agreement": {
                "matching": both_high,
                "total": len(decisive),
                "fraction": both_high / len(decisive) if decisive else None,
            },
            "mean_decisive_absolute_probability_error": sum(point["absolute_probability_error"] for point in decisive) / len(decisive) if decisive else None,
            "decisive_mismatches": [point for point in decisive if not point["high_status_matches"]],
        }
    return {"high_probability_threshold": threshold, "pages_in_report": len(pages), "events": events}


def _load_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / f"treehca-matplotlib-{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_plots(analysis: dict[str, Any], output_dir: Path, image_format: str = "png") -> dict[str, dict[str, str]]:
    """Write a separate probability scatter and agreement chart for each event."""
    if any(not details["points"] for details in analysis["events"].values()):
        raise ValueError("No pages have empirical probabilities for both events, so there is nothing to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    threshold = analysis["high_probability_threshold"]
    paths = {}
    for event, (_, title) in _EVENTS.items():
        details = analysis["events"][event]
        points = details["points"]
        figure, axis = plt.subplots(figsize=(8.0, 7.5), constrained_layout=True)
        for selected, color, marker, label in (
            ([point for point in points if not point["decisive"]], "#c7c7c7", "o", "Not decisive"),
            ([point for point in points if point["decisive"] and point["high_status_matches"]], "#238b45", "o", "High in both"),
            ([point for point in points if point["decisive"] and not point["high_status_matches"]], "#cb181d", "x", "High in one only"),
        ):
            if selected:
                axis.scatter(
                    [point["pseudo_probability"] for point in selected],
                    [point["empirical_probability"] for point in selected],
                    color=color,
                    marker=marker,
                    s=62 if marker == "x" else 48,
                    label=label,
                )
        axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0, alpha=0.55)
        axis.axvline(threshold, color="#555555", linestyle=":", linewidth=1.0)
        axis.axhline(threshold, color="#555555", linestyle=":", linewidth=1.0)
        axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        axis.set_xlabel("Pseudo probability")
        axis.set_ylabel("Empirical rate")
        axis.set_title(f"{title} by search-results start")
        axis.grid(alpha=0.16)
        axis.legend(loc="lower right")
        scatter_path = output_dir / f"{event}_probability_scatter.{image_format}"
        figure.savefig(scatter_path, dpi=200)
        plt.close(figure)

        directional = (details["pseudo_high_confirmation"], details["empirical_high_confirmation"])
        rates = [item["fraction"] if item["fraction"] is not None else 0.0 for item in directional]
        figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        bars = axis.bar(["Pseudo high\n→ empirical high", "Empirical high\n→ pseudo high"], rates, color=("#3182bd", "#756bb1"))
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Confirmation rate")
        axis.set_title(f"{title}: directional agreement above {threshold:g}")
        axis.grid(axis="y", alpha=0.2)
        for bar, item in zip(bars, directional, strict=True):
            label = "n/a" if item["fraction"] is None else f"{item['fraction']:.1%}\n({item['confirmed']}/{item['total']})"
            axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.025, label, ha="center", va="bottom")
        directional_path = output_dir / f"{event}_directional_agreement.{image_format}"
        figure.savefig(directional_path, dpi=200)
        plt.close(figure)
        paths[event] = {"probability_scatter": str(scatter_path), "directional_agreement": str(directional_path)}
    return paths


def build_summary(analysis: dict[str, Any], input_path: Path, plot_paths: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Keep decisive mismatches while omitting all routine per-page points."""
    return {
        "input_report": str(input_path),
        "high_probability_threshold": analysis["high_probability_threshold"],
        "pages_in_report": analysis["pages_in_report"],
        "events": {event: {key: value for key, value in details.items() if key != "points"} for event, details in analysis["events"].items()},
        "plots": plot_paths,
    }


def write_summaries(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_summary.json"
    markdown_path = output_dir / "analysis_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def rate(item: dict[str, Any], numerator: str) -> str:
        return "not available" if item["fraction"] is None else f"{item['fraction']:.2%} ({item[numerator]}/{item['total']})"

    lines = [
        "# Search-results outcome probability analysis",
        "",
        f"Input report: `{summary['input_report']}`",
        "",
        f"High means strictly greater than {summary['high_probability_threshold']:g}. Decisive pages are high in at least one distribution; pages below the threshold in both do not enter the agreement denominator.",
        "",
    ]
    for event, (_, title) in _EVENTS.items():
        details = summary["events"][event]
        plots = summary["plots"][event]
        error = details["mean_decisive_absolute_probability_error"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Pages evaluated: {details['pages_evaluated']} of {summary['pages_in_report']}",
                f"- Decisive pages: {details['decisive_pages']}",
                f"- Pseudo high → empirical high: **{rate(details['pseudo_high_confirmation'], 'confirmed')}**",
                f"- Empirical high → pseudo high: **{rate(details['empirical_high_confirmation'], 'confirmed')}**",
                f"- Bidirectional decisive agreement: **{rate(details['bidirectional_decisive_agreement'], 'matching')}**",
                f"- Mean absolute error among decisive pages: {'not available' if error is None else f'{error:.4f}'}",
                f"- Probability scatter: `{plots['probability_scatter']}`",
                f"- Directional agreement: `{plots['directional_agreement']}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=_DEFAULT_INPUT, help="Search-results outcomes testbed JSON report")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <input stem>_analysis beside the input report")
    parser.add_argument("--image-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--high-probability-threshold", type=float, default=_DEFAULT_HIGH_PROBABILITY_THRESHOLD, help="Must be in (0.5, 1); default: 0.6")
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
    for event in _EVENTS:
        agreement = analysis["events"][event]["bidirectional_decisive_agreement"]
        agreement_text = "not available" if agreement["fraction"] is None else f"{agreement['fraction']:.2%} ({agreement['matching']}/{agreement['total']})"
        print(f"{event} bidirectional decisive agreement: {agreement_text}")
        print(f"{event} probability plot: {plot_paths[event]['probability_scatter']}")
        print(f"{event} directional plot: {plot_paths[event]['directional_agreement']}")
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
