#!/usr/bin/env python3
"""Paired significance analysis over the common white-paper topics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from evaluation_io import METRIC_GROUPS, METHODS, load_method_scores


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def bootstrap_ci(
    differences: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        size = min(1000, iterations - start)
        indices = generator.integers(0, len(differences), size=(size, len(differences)))
        means[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def plot_results(results: list[dict], group_differences: dict, output: Path) -> None:
    baselines = [row["baseline"] for row in results]
    means = np.asarray([row["mean_difference"] for row in results])
    lows = np.asarray([row["ci_low"] for row in results])
    highs = np.asarray([row["ci_high"] for row in results])
    groups = list(METRIC_GROUPS)
    heatmap = np.asarray(
        [[group_differences[baseline][group] for group in groups] for baseline in baselines]
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": (1.05, 1.0)},
        constrained_layout=True,
    )
    positions = np.arange(len(baselines))
    left.errorbar(
        means,
        positions,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="o",
        color="#1F4E79",
        ecolor="#6F8FAF",
        capsize=3,
        linewidth=1.2,
    )
    left.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    left.set_yticks(positions, baselines)
    left.invert_yaxis()
    left.set_xlabel("Mean paired difference (TWP-Gen - baseline)")
    left.set_title("(a) Overall difference and 95% CI", loc="left", fontweight="bold")
    left.grid(axis="x", color="#E1E1E1", linewidth=0.5)
    left.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)

    image = right.imshow(heatmap, cmap="Blues", aspect="auto")
    right.set_xticks(np.arange(len(groups)), ["Content", "Adaptability", "Evidence"], rotation=20, ha="right")
    right.set_yticks(np.arange(len(baselines)), baselines)
    right.set_title("(b) Improvement by metric group", loc="left", fontweight="bold")
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            right.text(column, row, f"{heatmap[row, column]:.3f}", ha="center", va="center", fontsize=7.4)
    figure.colorbar(image, ax=right, fraction=0.047, pad=0.03)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/significance"))
    parser.add_argument("--backbone", choices=("14b", "32b"), default="32b")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    scores = {
        name: load_method_scores(args.results_root / folder, args.backbone)
        for name, folder in METHODS.items()
    }
    common = set.intersection(*(set(records) for records in scores.values()))
    if not common:
        raise ValueError("No topic has complete results for all methods")
    topics = sorted(common)
    twp = np.asarray([scores["TWP-Gen"][topic]["overall"] for topic in topics])

    results = []
    p_values = {}
    group_differences: dict[str, dict[str, float]] = {}
    for baseline in METHODS:
        if baseline == "TWP-Gen":
            continue
        base = np.asarray([scores[baseline][topic]["overall"] for topic in topics])
        differences = twp - base
        statistic, p_value = wilcoxon(
            twp,
            base,
            zero_method="wilcox",
            correction=True,
            alternative="two-sided",
            method="approx",
        )
        ci_low, ci_high = bootstrap_ci(
            differences, args.bootstrap_iterations, args.seed
        )
        row = {
            "baseline": baseline,
            "n_topics": len(topics),
            "twpgen_mean": float(twp.mean()),
            "baseline_mean": float(base.mean()),
            "mean_difference": float(differences.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "wilcoxon_statistic": float(statistic),
            "p_value": float(p_value),
        }
        results.append(row)
        p_values[baseline] = float(p_value)
        group_differences[baseline] = {
            group: float(
                np.mean(
                    [
                        scores["TWP-Gen"][topic]["groups"][group]
                        - scores[baseline][topic]["groups"][group]
                        for topic in topics
                    ]
                )
            )
            for group in METRIC_GROUPS
        }

    adjusted = holm_adjust(p_values)
    for row in results:
        row["p_holm"] = adjusted[row["baseline"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_significance.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "protocol": {
            "paired_unit": "topic",
            "n_common_topics": len(topics),
            "document_score": "unweighted mean of the ten metrics",
            "test": "two-sided Wilcoxon signed-rank test",
            "multiple_comparison_correction": "Holm",
            "confidence_interval": f"95% paired bootstrap ({args.bootstrap_iterations} resamples)",
            "backbone": args.backbone,
        },
        "comparisons": results,
        "group_mean_differences": group_differences,
    }
    (args.output_dir / "paired_significance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(results, group_differences, args.output_dir / "statistical_significance")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
