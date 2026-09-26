#!/usr/bin/env python3
"""Run the white-paper cluster-number sensitivity experiment.

This experiment uses topic-specific TWP-Gen representations, not DuEE.  For each
topic and each requested k it reports Silhouette, Davies-Bouldin and the ratio of
clusters containing fewer than five knowledge units.  The paper's quality,
evidence-support and balance scores are then calculated from the macro means.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize


DEFAULT_K = (20, 30, 40, 50, 60, 70, 80)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def selected_checkpoints(dataset_root: Path, topics_file: Path | None) -> list[Path]:
    paths = sorted(dataset_root.glob("*/clusters_/embed_0.pt"))
    if topics_file is None:
        return paths
    wanted = {
        line.strip()
        for line in topics_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    found = [path for path in paths if path.parent.parent.name in wanted]
    missing = sorted(wanted.difference(path.parent.parent.name for path in found))
    if missing:
        raise FileNotFoundError(f"Topics without embed_0.pt: {missing}")
    return found


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], output: Path) -> None:
    k = np.asarray([row["k"] for row in rows], dtype=float)
    balance = np.asarray([row["balance_score"] for row in rows], dtype=float)
    small = 100.0 * np.asarray([row["small_cluster_ratio_mean"] for row in rows])
    selected = int(np.argmax(balance))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(6.85, 3.75), constrained_layout=True)
    points = axis.scatter(
        k,
        balance,
        c=small,
        cmap="viridis_r",
        s=42,
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )
    axis.plot(k, balance, color="#1F4E79", linewidth=1.5, zorder=2)
    axis.annotate(
        rf"Selected setting: $k={int(k[selected])}$",
        xy=(k[selected], balance[selected]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8.2,
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
    )
    axis.set_xlabel(r"Number of evidence clusters, $k$")
    axis.set_ylabel(r"Balance score, $B(k)$ (%)")
    axis.set_xticks(k.astype(int))
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("Small clusters (<5 units), %")
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/k_sensitivity"))
    parser.add_argument("--topics-file", type=Path, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--small-cluster-threshold", type=int, default=5)
    parser.add_argument("--silhouette-sample-size", type=int, default=10000)
    args = parser.parse_args()

    checkpoints = selected_checkpoints(args.dataset_root, args.topics_file)
    if not checkpoints:
        raise FileNotFoundError(
            f"No <topic>/clusters_/embed_0.pt files under {args.dataset_root}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for topic_index, checkpoint_path in enumerate(checkpoints, 1):
        topic_dir = checkpoint_path.parent.parent
        matrix = normalize(
            np.asarray(load_checkpoint(checkpoint_path)["embed"], dtype=np.float32),
            norm="l2",
        )
        feature_path = topic_dir / "po_tuple_features_all_svos.pk"
        sample_weight = None
        if feature_path.is_file():
            with feature_path.open("rb") as handle:
                features = pickle.load(handle)
            weights = np.asarray(features.get("tuple_freq", []), dtype=np.float64)
            if len(weights) == len(matrix):
                sample_weight = weights
        print(f"[{topic_index}/{len(checkpoints)}] {topic_dir.name}", flush=True)
        for k in args.k:
            if k >= len(matrix):
                print(f"  skip k={k}: only {len(matrix)} units", flush=True)
                continue
            model = KMeans(
                n_clusters=k,
                random_state=args.random_state,
                n_init=args.n_init,
                max_iter=300,
            )
            labels = model.fit_predict(matrix, sample_weight=sample_weight)
            sizes = np.bincount(labels, minlength=k)
            sample_size = min(args.silhouette_sample_size, len(matrix))
            rows.append(
                {
                    "topic": topic_dir.name,
                    "n_units": int(len(matrix)),
                    "k": int(k),
                    "silhouette": float(
                        silhouette_score(
                            matrix,
                            labels,
                            sample_size=sample_size if sample_size < len(matrix) else None,
                            random_state=args.random_state,
                        )
                    ),
                    "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
                    "small_cluster_ratio": float(
                        np.mean(sizes < args.small_cluster_threshold)
                    ),
                    "mean_cluster_size": float(sizes.mean()),
                    "min_cluster_size": int(sizes.min()),
                    "max_cluster_size": int(sizes.max()),
                }
            )

    write_csv(args.output_dir / "per_topic_results.csv", rows)
    summary = []
    for k in args.k:
        selected = [row for row in rows if row["k"] == k]
        if not selected:
            continue
        entry = {"k": k, "n_topics": len(selected)}
        for metric in ("silhouette", "davies_bouldin", "small_cluster_ratio", "mean_cluster_size"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)

    silhouettes = np.asarray([row["silhouette_mean"] for row in summary])
    dbi = np.asarray([row["davies_bouldin_mean"] for row in summary])
    quality = 50.0 * (silhouettes / silhouettes.max() + dbi.min() / dbi)
    support = 100.0 * (1.0 - np.asarray([row["small_cluster_ratio_mean"] for row in summary]))
    denominator = quality + support
    balance = np.divide(
        2.0 * quality * support,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    for index, row in enumerate(summary):
        row["quality_score"] = float(quality[index])
        row["evidence_support_score"] = float(support[index])
        row["balance_score"] = float(balance[index])
    write_csv(args.output_dir / "summary.csv", summary)
    payload = {
        "protocol": {
            "dataset_root": str(args.dataset_root.resolve()),
            "topics": len(checkpoints),
            "k_values": args.k,
            "representation": "L2-normalized topic-specific embed_0.pt",
            "weighted_kmeans": True,
            "random_state": args.random_state,
            "n_init": args.n_init,
            "small_cluster_threshold": args.small_cluster_threshold,
            "quality_score": "50 * (Silhouette/max(Silhouette) + min(DBI)/DBI)",
            "evidence_support_score": "100 * (1 - small-cluster ratio)",
            "balance_score": (
                "2 * quality score * evidence-support score / "
                "(quality score + evidence-support score)"
            ),
        },
        "summary": summary,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_summary(summary, args.output_dir / "cluster_number_sensitivity")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
