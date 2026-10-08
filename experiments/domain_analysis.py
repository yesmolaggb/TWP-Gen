#!/usr/bin/env python3
"""Aggregate the ten evaluation metrics by the eight topic domains."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from evaluation_io import METRIC_ORDER, METHODS, canonical_topic, load_method_scores


def load_domains(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = data.get("topics") if isinstance(data, dict) else data
    if not isinstance(topics, list):
        raise ValueError(f"{path}: expected a topic list")
    result = {}
    for row in topics:
        key = canonical_topic(str(row["title"]))
        result[key] = {
            "id": row.get("id"),
            "title": row["title"],
            "domain": row["domain"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--topics",
        type=Path,
        default=Path("dataset/whitepaper_topics.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/domain_analysis"))
    parser.add_argument("--backbone", choices=("14b", "32b"), default="32b")
    args = parser.parse_args()

    domains = load_domains(args.topics)
    method_scores = {
        name: load_method_scores(args.results_root / folder, args.backbone)
        for name, folder in METHODS.items()
    }
    common = set.intersection(*(set(records) for records in method_scores.values()))
    unknown = sorted(topic for topic in common if topic not in domains)
    if unknown:
        raise ValueError(f"Evaluation topics missing from {args.topics}: {unknown}")

    domain_order = list(dict.fromkeys(row["domain"] for row in domains.values()))
    rows = []
    twpgen_metrics = []
    for method, records in method_scores.items():
        for domain in domain_order:
            selected = [
                records[topic]
                for topic in common
                if domains[topic]["domain"] == domain
            ]
            if not selected:
                continue
            rows.append(
                {
                    "domain": domain,
                    "n": len(selected),
                    "method": method,
                    "mean_score": sum(row["overall"] for row in selected) / len(selected),
                }
            )
            if method == "TWP-Gen":
                twpgen_metrics.append(
                    {
                        "domain": domain,
                        "n": len(selected),
                        **{
                            metric: sum(row["metrics"][metric] for row in selected) / len(selected)
                            for metric in METRIC_ORDER
                        },
                        "overall": sum(row["overall"] for row in selected) / len(selected),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method_by_domain.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "twpgen_domain_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(twpgen_metrics[0]))
        writer.writeheader()
        writer.writerows(twpgen_metrics)

    counts = Counter(domains[topic]["domain"] for topic in common)
    payload = {
        "protocol": {
            "backbone": args.backbone,
            "n_common_topics": len(common),
            "domain_source": str(args.topics.resolve()),
        },
        "domain_counts": {domain: counts[domain] for domain in domain_order},
        "method_by_domain": rows,
        "twpgen_domain_metrics": twpgen_metrics,
    }
    (args.output_dir / "domain_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
