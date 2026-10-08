#!/usr/bin/env python3
"""Export the classified table-of-contents corpus as a single JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tables = sorted(args.input_root.glob("*/toc_table.csv"))
    if not tables:
        raise FileNotFoundError(f"No toc_table.csv files found under {args.input_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, table in enumerate(tables, 1):
            with table.open("r", encoding="utf-8-sig", newline="") as source:
                headings = list(csv.DictReader(source))
            record = {
                "record_id": f"TOC{index:04d}",
                "source_name": table.parent.name,
                "headings": headings,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"exported={len(tables)} output={args.output}")


if __name__ == "__main__":
    main()
