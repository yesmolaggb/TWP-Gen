#!/usr/bin/env python3
"""Compare K-Means, GMM and Bisecting K-Means on fixed DuEE embeddings.

The input is the ``top10_comparison_arrays.npz`` produced by the DuEE
representation experiment.  It must contain ``labels`` and
``gesi_embeddings``.  Gold labels determine only the controlled number of
clusters and the external metrics; they are never passed to a clusterer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, normalize

from clustering_metrics import evaluate


METRICS = ("ARI", "NMI", "ACC", "B3_F1")


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, object]]:
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1))
            if len(rows) > 1
            else 0.0,
            "values": [float(row[metric]) for row in rows],
        }
        for metric in METRICS
    }


def predictions_for(
    algorithm: str,
    matrix: np.ndarray,
    n_clusters: int,
    runs: int,
    n_init: int,
) -> list[np.ndarray]:
    predictions = []
    for seed in range(runs):
        if algorithm == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=n_init,
                max_iter=300,
            )
        elif algorithm == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                random_state=seed,
                n_init=n_init,
                max_iter=500,
                reg_covar=1e-6,
            )
        elif algorithm == "bisecting":
            model = BisectingKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=n_init,
                max_iter=300,
                bisecting_strategy="biggest_inertia",
            )
        else:  # guarded by argparse
            raise ValueError(algorithm)
        predictions.append(model.fit_predict(matrix).astype(np.int32))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("kmeans", "gmm", "bisecting"),
        default=("kmeans", "gmm", "bisecting"),
    )
    args = parser.parse_args()

    arrays = np.load(args.input, allow_pickle=False)
    missing = {"labels", "gesi_embeddings"}.difference(arrays.files)
    if missing:
        raise KeyError(f"{args.input} is missing {sorted(missing)}")
    labels_text = arrays["labels"].astype(str)
    y_true = LabelEncoder().fit_transform(labels_text)
    matrix = normalize(
        np.asarray(arrays["gesi_embeddings"], dtype=np.float32), norm="l2"
    )
    n_clusters = len(np.unique(y_true))

    display = {
        "kmeans": "TWP-Gen (K-Means)",
        "gmm": "GMM",
        "bisecting": "Bisecting K-Means",
    }
    result = {
        "protocol": {
            "input": str(args.input.resolve()),
            "representation": "fixed L2-normalized TWP-Gen multidimensional embeddings",
            "n_samples": int(len(y_true)),
            "n_clusters": int(n_clusters),
            "runs": args.runs,
            "random_seeds": list(range(args.runs)),
            "n_init_per_seed": args.n_init,
            "labels_used_by_clustering": False,
        },
        "algorithms": {},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for algorithm in args.algorithms:
        predictions = predictions_for(
            algorithm, matrix, n_clusters, args.runs, args.n_init
        )
        metrics = summarize([evaluate(y_true, prediction) for prediction in predictions])
        result["algorithms"][display[algorithm]] = metrics
        np.savez_compressed(
            args.output_dir / f"{algorithm}_predictions.npz",
            labels=labels_text,
            predictions=np.stack(predictions),
        )
        csv_rows.append(
            {
                "algorithm": display[algorithm],
                **{
                    f"{metric}_{stat}": metrics[metric][stat]
                    for metric in METRICS
                    for stat in ("mean", "std")
                },
            }
        )

    (args.output_dir / "clustering_algorithm_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "clustering_algorithm_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
