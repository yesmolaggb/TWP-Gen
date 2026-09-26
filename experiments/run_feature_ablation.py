#!/usr/bin/env python3
"""Run the five clustering feature variants for one prepared topic.

The topic must already contain ``po_tuple_features_all_svos.pk`` and the other
pre-clustering artifacts.  Each run uses the same hyperparameters and writes to
its own ``clusters_<variant>`` directory.  The full model retains the historical
``clusters_`` directory name.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = (
    "full",
    "without_event",
    "without_graph",
    "without_entity_verb",
    "semantic_only",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="prepared dataset directory name")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/ablation_runs.json"))
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runner = repo / "outline_generator" / "run_twpgen.py"
    topic_dir = args.dataset_root / args.topic
    feature_bank = topic_dir / "po_tuple_features_all_svos.pk"
    if not feature_bank.is_file():
        raise FileNotFoundError(feature_bank)

    records = []
    for variant in args.variant:
        environment = os.environ.copy()
        environment["TWPGEN_DATASET"] = args.topic
        environment["TWPGEN_DATASET_ROOT"] = str(args.dataset_root.resolve())
        environment["TWPGEN_FEATURE_VARIANT"] = variant
        print(f"Running {variant} for {args.topic}", flush=True)
        process = subprocess.run(
            [sys.executable, str(runner)],
            cwd=str(repo),
            env=environment,
            text=True,
            check=False,
        )
        records.append({"variant": variant, "returncode": process.returncode})
        if process.returncode:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"topic": args.topic, "runs": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if any(row["returncode"] for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
