#!/usr/bin/env python3
"""Generate a single topic (used for parallel execution).

Example
-------
python run_one_topic.py --topic "示例技术白皮书" \
    --outline-root dataset/outline --source-root dataset/sources \
    --output-root output/post_outline/generated

Credentials come from the repository ``.env`` (see ``llm_settings``); use
``--env-file`` only when the file lives somewhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_postoutline_experiment as m
from llm_settings import build_client, resolve_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="topic index in TOPICS")
    parser.add_argument("--topic", default=None, help="topic name (overrides --index)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--outline-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None,
                        help="defaults to <repo root>/.env")
    parser.add_argument("--model", default=None, help="overrides the resolved default model")
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--base-url", default=None, help="overrides OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="overrides OPENAI_API_KEY")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    settings = resolve_settings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_thinking=not args.no_thinking,
        env_file=args.env_file,
    )
    client = build_client(settings)

    if args.topic is None and args.index is None:
        parser.error("either --topic or --index is required")
    topic = args.topic if args.topic else m.TOPICS[args.index]
    print(f"START topic={topic} model={settings.model}", flush=True)
    result = m.generate_topic(
        client=client,
        model=settings.model,
        planner_model=args.planner_model or settings.model,
        topic=topic,
        outline_root=args.outline_root,
        source_root=args.source_root,
        output_root=args.output_root,
        optimized=True,
        enable_thinking=not args.no_thinking,
        overwrite=True,
    )
    print(
        f"DONE topic={topic} status={result.get('status')} "
        f"chars={result.get('characters')} cov={result.get('citation_coverage')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
