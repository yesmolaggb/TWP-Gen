"""Summarise existing evaluation result files into the reported table.

Example
-------
    python evaluation_dataset/evaluation/run_summary.py \
        --results-root path/to/eval --backbone 32b

The result directory must contain one subdirectory per method
(``rag``, ``storm``, ``omni``, ``conver``, ``own``) with the three metric-group
files produced by the scoring stage. The script prints the table and writes a JSON,
a CSV and a Markdown copy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from aggregate import BACKBONES, METHODS, summarize  # noqa: E402
from config import default_results_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="directory containing one subdirectory per method",
    )
    parser.add_argument(
        "--backbone",
        choices=sorted(BACKBONES),
        default="32b",
        help="which generated-document set to summarise (default: 32b)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where the summaries are written (default: <output_dir>/evaluation)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.results_root
    if not root.is_dir():
        print(f"results directory not found: {root}", file=sys.stderr)
        return 2

    missing = [name for name in METHODS if not (root / name).is_dir()]
    if missing:
        print(f"missing method directories: {', '.join(missing)}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or default_results_dir()
    results = summarize(root, args.backbone, output_dir)

    headers = ["Framework", *[f"{m}" for m in results]]
    print(" | ".join(headers))
    for framework, result in results.items():
        print(f"{framework}: overall {result.overall():.2f}")

    print(f"summaries written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
