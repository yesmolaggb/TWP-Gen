#!/usr/bin/env python3
"""Original-style, publication-ready DuEE representation comparison."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


STYLE = {
    "竞赛行为-胜负": ("Competition—Result", "#E69F00", "s"),
    "竞赛行为-夺冠": ("Competition—Championship", "#F3B33D", "s"),
    "竞赛行为-退赛": ("Competition—Withdrawal", "#F6CF71", "s"),
    "司法行为-拘捕": ("Legal—Detention", "#0072B2", "o"),
    "司法行为-起诉": ("Legal—Prosecution", "#56B4E9", "o"),
    "司法行为-约谈": ("Legal—Summons", "#2A6F97", "o"),
    "组织关系-加盟": ("Organization—Joining", "#7B2CBF", "^"),
    "组织关系-退出": ("Organization—Leaving", "#9D4EDD", "^"),
    "组织关系-辞/离职": ("Organization—Resignation", "#C77DFF", "^"),
    "组织行为-开幕": ("Organization—Opening", "#5A189A", "^"),
    "产品行为-发布": ("Product—Release", "#009E73", "D"),
    "产品行为-上映": ("Product—Premiere", "#66C2A5", "D"),
    "人生-死亡": ("Life—Death", "#D55E00", "P"),
    "人生-结婚": ("Life—Marriage", "#CC79A7", "P"),
    "灾害/意外-车祸": ("Accident—Traffic accident", "#4D4D4D", "X"),
}


def project(matrix: np.ndarray, seed: int) -> np.ndarray:
    matrix = normalize(np.asarray(matrix, dtype=np.float32), norm="l2")
    n_components = min(50, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    kwargs = {
        "n_components": 2,
        "perplexity": min(35.0, max(5.0, (len(matrix) - 1) / 3.0)),
        "init": "pca",
        "learning_rate": "auto",
        "metric": "cosine",
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 1500
    else:
        kwargs["n_iter"] = 1500
    points = TSNE(**kwargs).fit_transform(reduced)
    points -= points.mean(axis=0, keepdims=True)
    points /= np.maximum(points.std(axis=0, keepdims=True), 1e-12)
    return points


def score_line(metrics: dict, method: str) -> str:
    values = metrics["matched_comparison"][method]
    return (
        f"ARI {values['ARI']['mean']:.1f}   "
        f"NMI {values['NMI']['mean']:.1f}   "
        f"ACC {values['ACC']['mean']:.1f}   "
        f"B³ F1 {values['BCubed_F1']['mean']:.1f}"
    )


def draw(axis, points: np.ndarray, labels: np.ndarray, ordered_labels: list) -> None:
    for label in ordered_labels:
        display, color, marker = STYLE[label]
        mask = labels == label
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=12 if marker != "X" else 16,
            c=color,
            marker=marker,
            alpha=0.64,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            label=display,
        )
    axis.set_xlabel("t-SNE 1", labelpad=2)
    axis.set_ylabel("t-SNE 2", labelpad=2)
    axis.grid(True, linestyle=(0, (2, 3)), linewidth=0.45, color="#D7DCE2")
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors="#555555", labelsize=6.2, length=2.4)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#6E747B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arrays = np.load(args.input, allow_pickle=False)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    labels = arrays["labels"].astype(str)
    counts = Counter(labels.tolist())
    ordered_labels = sorted(counts, key=lambda label: (-counts[label], STYLE[label][0]))

    print("Projecting semantic-only representation...", flush=True)
    semantic_points = project(arrays["semantic_embeddings"], args.seed)
    print("Projecting multidimensional representation...", flush=True)
    gesi_points = project(arrays["gesi_embeddings"], args.seed)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(7.35, 3.22), dpi=180)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.61],
        left=0.062,
        right=0.992,
        top=0.82,
        bottom=0.17,
        wspace=0.18,
    )
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    draw(left, semantic_points, labels, ordered_labels)
    draw(right, gesi_points, labels, ordered_labels)
    right.set_ylabel("")

    left.set_title("(a) Semantic-only baseline", loc="left", pad=21, fontsize=8.6, fontweight="bold")
    right.set_title("(b) Multidimensional representation (ours)", loc="left", pad=21, fontsize=8.6, fontweight="bold")
    left.text(
        0.0,
        1.035,
        score_line(metrics, "ConvergeWriter_semantic_only"),
        transform=left.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#444A50",
    )
    right.text(
        0.0,
        1.035,
        score_line(metrics, "GESI_multidimensional_latent"),
        transform=right.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#444A50",
    )

    handles = []
    for label in ordered_labels:
        display, color, marker = STYLE[label]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker=marker,
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor="none",
                label=f"{display}  ({counts[label]})",
            )
        )
    legend_axis.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=5.65,
        title="DuEE event type  (n)",
        title_fontsize=6.6,
        handletextpad=0.55,
        borderaxespad=0,
        labelspacing=0.62,
    )

    n_types = len(set(labels.tolist()))
    figure.text(
        0.062,
        0.055,
        f"Gold event labels are used only for visualization and external evaluation; both panels use the same {len(labels):,} aligned instances.",
        ha="left",
        fontsize=6.25,
        color="#4B5258",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"duee_top{n_types}_clustering_comparison"
    figure.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    np.savez_compressed(
        args.output_dir / f"duee_top{n_types}_clustering_projection.npz",
        labels=labels,
        semantic_points=semantic_points,
        gesi_points=gesi_points,
    )
    print(f"Saved v2 figure under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
