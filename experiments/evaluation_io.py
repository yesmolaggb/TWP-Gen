"""Shared readers for per-topic LLM evaluation results."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


METRIC_GROUPS = {
    "Content Quality": (
        "relevance",
        "coverage",
        "depth",
        "novelty",
    ),
    "White-Paper Adaptability": (
        "technical_specificity",
        "understandability",
        "structurality",
    ),
    "Evidence Credibility": (
        "citation_sufficiency",
        "citation_validity",
        "factual_consistency",
    ),
}
METRIC_ORDER = tuple(metric for group in METRIC_GROUPS.values() for metric in group)

METHODS = {
    "Direct RAG": "rag",
    "STORM": "storm",
    "OmniThink": "omni",
    "ConvergeWriter": "conver",
    "TWP-Gen": "own",
}


def canonical_topic(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list or {{'results': [...]}}")
    return [row for row in data if row.get("status", "success") == "success"]


def _index(records: list[dict], path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in records:
        topic = str(row.get("topic", "")).strip()
        if not topic:
            raise ValueError(f"{path}: record without topic")
        key = canonical_topic(topic)
        if key in result:
            raise ValueError(f"{path}: duplicate normalized topic {topic!r}")
        result[key] = row
    return result


def _score(row: dict, container: str, metric: str, path: Path) -> float:
    try:
        value = float(row[container][metric]["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid {container}.{metric}.score") from error
    if not 0.0 <= value <= 5.0:
        raise ValueError(f"{path}: score outside [0, 5]: {value}")
    return value


def _new_result_candidates(method_dir: Path, backbone: str) -> tuple[Path, ...]:
    """Possible locations of the unified evaluator output, in priority order."""
    return (
        method_dir / f"scores_{backbone}.json",
        method_dir / "scores.json",
        method_dir / "evaluation" / f"scores_{backbone}.json",
        method_dir / "evaluation" / "scores.json",
    )


def _load_unified_scores(path: Path) -> dict[str, dict]:
    """Read ``evaluate_topics.py`` output into the shared analysis schema."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected {{'documents': [...]}}")

    result: dict[str, dict] = {}
    for row in records:
        if row.get("status", "success") != "success":
            continue
        topic = str(row.get("topic", "")).strip()
        if not topic:
            raise ValueError(f"{path}: record without topic")
        key = canonical_topic(topic)
        if key in result:
            raise ValueError(f"{path}: duplicate normalized topic {topic!r}")

        means = row.get("mean")
        raw_runs = row.get("scores")
        metrics: dict[str, float] = {}
        for metric in METRIC_ORDER:
            value = means.get(metric) if isinstance(means, dict) else None
            if value is None and isinstance(raw_runs, dict):
                runs = raw_runs.get(metric)
                if isinstance(runs, list) and runs:
                    value = sum(float(item) for item in runs) / len(runs)
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid mean score for {metric}") from error
            if not 0.0 <= value <= 5.0:
                raise ValueError(f"{path}: score outside [0, 5]: {value}")
            metrics[metric] = value

        groups = {
            group: sum(metrics[name] for name in names) / len(names)
            for group, names in METRIC_GROUPS.items()
        }
        result[key] = {
            "topic": topic,
            "metrics": metrics,
            "groups": groups,
            "overall": sum(metrics.values()) / len(METRIC_ORDER),
        }
    if not result:
        raise ValueError(f"{path}: no successful evaluation records")
    return result


def load_method_scores(method_dir: Path, backbone: str = "32b") -> dict[str, dict]:
    """Return ``canonical topic -> {topic, metrics, groups, overall}``."""
    for candidate in _new_result_candidates(method_dir, backbone):
        if candidate.is_file():
            return _load_unified_scores(candidate)

    # Backward-compatible fallback for the original three-file evaluator output.
    suffix = "" if backbone == "32b" else f"_{backbone}"
    specs = (
        (method_dir / f"eval_results{suffix}.json", "rubric_grading", METRIC_GROUPS["Content Quality"]),
        (method_dir / f"eval_adaptability_{backbone}.json", "white_paper_adaptability", METRIC_GROUPS["White-Paper Adaptability"]),
        (method_dir / f"eval_evidence_{backbone}.json", "evidence_credibility", METRIC_GROUPS["Evidence Credibility"]),
    )
    indexed = [(_index(_load_records(path), path), path, container, metrics) for path, container, metrics in specs]
    topics = set(indexed[0][0])
    for records, path, _, _ in indexed[1:]:
        if set(records) != topics:
            raise ValueError(f"Topic mismatch between evaluation groups in {method_dir}; see {path}")

    result: dict[str, dict] = {}
    for topic_key in sorted(topics):
        metrics: dict[str, float] = {}
        display_topic = indexed[0][0][topic_key]["topic"]
        for records, path, container, names in indexed:
            for name in names:
                metrics[name] = _score(records[topic_key], container, name, path)
        groups = {
            group: sum(metrics[name] for name in names) / len(names)
            for group, names in METRIC_GROUPS.items()
        }
        result[topic_key] = {
            "topic": display_topic,
            "metrics": metrics,
            "groups": groups,
            "overall": sum(metrics.values()) / len(METRIC_ORDER),
        }
    return result
