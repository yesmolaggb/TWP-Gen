"""Aggregate per-document scores into the reported summary.

Reads the three metric-group result files produced by the scoring stage for one
.method directory, computes the mean of every metric, the three group averages and
the unweighted overall average, and writes JSON, CSV and Markdown summaries.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from config import default_results_dir
from metrics import (
    DISPLAY_NAMES,
    METRIC_GROUPS,
    METRIC_ORDER,
    SHORT_NAMES,
    validate_score,
)

# Result file for each metric group, and the container key inside it.
GROUP_FILES: dict[str, tuple[str, str]] = {
    "Content Quality": ("eval_results.json", "rubric_grading"),
    "White-Paper Adaptability": (
        "eval_adaptability_32b.json",
        "white_paper_adaptability",
    ),
    "Evidence Credibility": (
        "eval_evidence_32b.json",
        "evidence_credibility",
    ),
}

METHODS: dict[str, str] = {
    "rag": "DirectRAG",
    "storm": "STORM",
    "omni": "OmniThink",
    "conver": "ConvergeWriter",
    "own": "TWP-Gen",
}

BACKBONES: dict[str, str] = {
    "14b": "Qwen3-14B without Reasoning",
    "32b": "Qwen3-32B with Reasoning",
}


@dataclass
class MethodResult:
    """Per-metric means for one method, plus how many documents they cover."""

    metrics: dict[str, float]
    counts: dict[str, int]

    def group_average(self, group: str) -> float:
        values = [
            self.metrics[metric]
            for metric in METRIC_GROUPS[group]
            if metric in self.metrics
        ]
        return sum(values) / len(values)

    def overall(self) -> float:
        values = [
            self.metrics[metric] for metric in METRIC_ORDER if metric in self.metrics
        ]
        return sum(values) / len(values)


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    records = [r for r in payload if r.get("status", "success") == "success"]
    if not records:
        raise ValueError(f"{path}: no successful records")
    return records


def evaluate_method(method_dir: Path) -> MethodResult:
    """Average each metric over every document of one method."""
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}

    for group, (filename, container_key) in GROUP_FILES.items():
        records = load_records(method_dir / filename)
        for metric in METRIC_GROUPS[group]:
            if metric not in METRIC_DEFINITIONS_KEYS:
                raise KeyError(f"{filename}: unknown metric '{metric}'")
            scores = []
            for index, record in enumerate(records):
                container = record.get(container_key)
                if not isinstance(container, dict):
                    raise KeyError(
                        f"{filename}: record {index} has no '{container_key}' block"
                    )
                entry = container.get(metric)
                if not isinstance(entry, dict) or "score" not in entry:
                    raise KeyError(
                        f"{filename}: record {index} has no score for '{metric}'"
                    )
                scores.append(
                    validate_score(entry["score"], f"{filename}: record {index} {metric}")
                )
            metrics[metric] = sum(scores) / len(scores)
            counts[metric] = len(scores)

    return MethodResult(metrics=metrics, counts=counts)


METRIC_DEFINITIONS_KEYS = frozenset(METRIC_ORDER)


def evaluate_all(root: Path, backbone: str) -> dict[str, MethodResult]:
    """Evaluate every method for one generation backbone."""
    results = {}
    for method_dir, framework in METHODS.items():
        results[framework] = evaluate_method(root / method_dir)
    return results


def render_table(results: dict[str, MethodResult]) -> str:
    """Plain-text table, best value bolded and runner-up underlined."""
    headers = ["Framework", *[SHORT_NAMES[m] for m in METRIC_ORDER], "Average"]

    columns = {metric: [r.metrics[metric] for r in results.values()] for metric in METRIC_ORDER}
    columns["average"] = [r.overall() for r in results.values()]

    best: dict[str, float] = {}
    second: dict[str, float] = {}
    for key, values in columns.items():
        ordered = sorted({round(v, 12) for v in values}, reverse=True)
        best[key] = ordered[0]
        second[key] = ordered[1] if len(ordered) > 1 else ordered[0]

    def cell(key: str, value: float) -> str:
        text = f"{value:.2f}"
        rounded = round(value, 12)
        if rounded == best[key]:
            return f"**{text}**"
        if rounded == second[key] and second[key] != best[key]:
            return f"<u>{text}</u>"
        return text

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for framework, result in results.items():
        row = [framework]
        for metric in METRIC_ORDER:
            row.append(cell(metric, result.metrics[metric]))
        row.append(cell("average", result.overall()))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_json(results: dict[str, MethodResult], output_path: Path, backbone: str) -> None:
    payload = {
        "generated_model": BACKBONES.get(backbone, backbone),
        "metrics": [DISPLAY_NAMES[m] for m in METRIC_ORDER],
        "methods": {},
    }
    for framework, result in results.items():
        payload["methods"][framework] = {
            "metrics": {m: round(result.metrics[m], 6) for m in METRIC_ORDER},
            "counts": result.counts,
            "groups": {
                group: round(result.group_average(group), 6)
                for group in METRIC_GROUPS
            },
            "overall": round(result.overall(), 6),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_csv(results: dict[str, MethodResult], output_path: Path, backbone: str) -> None:
    fieldnames = [
        "generated_model",
        "framework",
        *METRIC_ORDER,
        "content_quality_avg",
        "adaptability_avg",
        "evidence_avg",
        "overall_avg",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for framework, result in results.items():
            row = {
                "generated_model": BACKBONES.get(backbone, backbone),
                "framework": framework,
                "content_quality_avg": round(result.group_average("Content Quality"), 6),
                "adaptability_avg": round(
                    result.group_average("White-Paper Adaptability"), 6
                ),
                "evidence_avg": round(
                    result.group_average("Evidence Credibility"), 6
                ),
                "overall_avg": round(result.overall(), 6),
            }
            for metric in METRIC_ORDER:
                row[metric] = round(result.metrics[metric], 6)
            writer.writerow(row)


def save_markdown(results: dict[str, MethodResult], output_path: Path, backbone: str) -> None:
    body = render_table(results)
    text = (
        f"# Evaluation summary\n\n"
        f"Generated by: {BACKBONES.get(backbone, backbone)}\n\n"
        f"Best results are bold, second-best underlined.\n\n{body}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def summarize(root: Path, backbone: str, output_dir: Path | None = None) -> dict:
    """Read results, print the table and write the three summary files."""
    results = evaluate_all(root, backbone)
    target = output_dir or default_results_dir()
    save_json(results, target / f"summary_{backbone}.json", backbone)
    save_csv(results, target / f"summary_{backbone}.csv", backbone)
    save_markdown(results, target / f"summary_{backbone}.md", backbone)
    return results
