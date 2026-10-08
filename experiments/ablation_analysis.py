#!/usr/bin/env python3
"""Aggregate completed end-to-end feature-ablation evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation_io import METRIC_GROUPS, METRIC_ORDER, load_method_scores


DEFAULT_VARIANTS = (
    "TWP-Gen=own",
    "w/o Event=ablation/-Event",
    "w/o Graph=ablation/-Graph",
    "w/o Entity-Verb=ablation/-PO",
    "Semantic-only=ablation/semantic-only",
)


def parse_variants(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Variant must use LABEL=RELATIVE_PATH: {value}")
        label, path = value.split("=", 1)
        result.append((label.strip(), Path(path.strip())))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ablation"))
    parser.add_argument("--backbone", choices=("14b", "32b"), default="32b")
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="repeat LABEL=RELATIVE_PATH; defaults reproduce the paper directory layout",
    )
    args = parser.parse_args()

    variants = parse_variants(args.variant or list(DEFAULT_VARIANTS))
    output_rows = []
    topic_rows = []
    for label, relative_path in variants:
        records = load_method_scores(args.results_root / relative_path, args.backbone)
        if not records:
            raise ValueError(f"No scores for {label}")
        row = {"variant": label, "n_topics": len(records)}
        for metric in METRIC_ORDER:
            row[metric] = sum(item["metrics"][metric] for item in records.values()) / len(records)
        for group in METRIC_GROUPS:
            row[group] = sum(item["groups"][group] for item in records.values()) / len(records)
        row["overall"] = sum(item["overall"] for item in records.values()) / len(records)
        output_rows.append(row)
        for item in records.values():
            topic_rows.append(
                {
                    "variant": label,
                    "topic": item["topic"],
                    **item["metrics"],
                    "overall": item["overall"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("ablation_summary.csv", output_rows),
        ("ablation_per_topic.csv", topic_rows),
    ):
        with (args.output_dir / filename).open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "protocol": {
            "backbone": args.backbone,
            "overall": "unweighted mean of the ten metrics",
            "variants": {label: str(path) for label, path in variants},
        },
        "summary": output_rows,
    }
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
