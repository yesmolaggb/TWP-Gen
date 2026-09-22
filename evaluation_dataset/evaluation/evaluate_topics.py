"""Score a directory of generated white papers with the ten-metric protocol.

Example
-------
    python evaluation_dataset/evaluation/evaluate_topics.py \
        --article-dir output/article \
        --output      output/evaluation/scores.json \
        --repeats     3

Two or more runs are merged per metric into the reported average. Method names are
removed, documents are reordered, and the reference white paper is never passed to
the evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import default_results_dir, resolve_settings, topics_file  # noqa: E402
from metrics import METRIC_ORDER  # noqa: E402
from scorer import (  # noqa: E402
    DEFAULT_METHOD_NAMES,
    Evaluator,
    load_documents,
    randomly_reorder,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--article-dir",
        type=Path,
        required=True,
        help="directory holding the generated white papers (*.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result JSON path (default: <output_dir>/evaluation/scores.json)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="independent scoring runs per document (default: 3)",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=None,
        help=f"task list JSON (default: {topics_file()})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only score the first N documents (0 = all)",
    )
    parser.add_argument("--model", default=None, help="override the evaluator model")
    parser.add_argument("--base-url", default=None, help="override the API base URL")
    parser.add_argument("--api-key", default=None, help="override the API key")
    parser.add_argument("--env-file", default=None, help="extra .env file to load")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip documents already present in the output file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the presentation order",
    )
    return parser.parse_args(argv)


def load_existing(path: Path) -> dict[str, dict]:
    """Read a previous run so ``--resume`` can skip finished documents."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    records = payload.get("documents", payload if isinstance(payload, list) else [])
    return {record.get("topic"): record for record in records if record.get("topic")}


def serialize(document, method_names: list[str]) -> dict:
    return {
        "topic": document.topic,
        "article_file": str(document.path),
        "method_names_removed": method_names,
        "status": "success",
        "scores": {
            metric: [round(value, 6) for value in document.scores.get(metric, [])]
            for metric in METRIC_ORDER
        },
        "mean": {
            metric: (
                round(document.mean(metric), 6)
                if document.mean(metric) is not None
                else None
            )
            for metric in METRIC_ORDER
        },
        "overall": (
            round(document.overall(), 6)
            if document.overall() is not None
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = resolve_settings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        env_file=args.env_file,
    )

    article_dir = args.article_dir
    if not article_dir.is_dir():
        print(f"article directory not found: {article_dir}", file=sys.stderr)
        return 2

    output = args.output or (default_results_dir() / "scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    documents = randomly_reorder(
        load_documents(article_dir, DEFAULT_METHOD_NAMES), seed=args.seed
    )
    if args.limit:
        documents = documents[: args.limit]

    done = load_existing(output) if args.resume else {}
    records = list(done.values())
    evaluator = Evaluator(settings=settings, method_names=DEFAULT_METHOD_NAMES)

    print(
        f"scoring {len(documents)} document(s) with {settings.model} "
        f"at {settings.base_url}, {args.repeats} run(s) each"
    )

    for index, document in enumerate(documents, 1):
        if document.topic in done:
            print(f"[{index}/{len(documents)}] skip (already scored): {document.topic}")
            continue
        print(f"[{index}/{len(documents)}] scoring: {document.topic}")
        scored = evaluator.score_document(
            document.topic, document.text, repeats=args.repeats
        )
        scored.path = document.path
        record = serialize(scored, DEFAULT_METHOD_NAMES)
        records.append(record)
        output.write_text(
            json.dumps({"documents": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
