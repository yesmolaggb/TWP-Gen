"""LLM scoring protocol.

For one generated document the evaluator is called once per metric group. Each call
receives the definitions of its own metrics plus the group-specific 0--5 rubric and
returns one independent score per metric; no group-level aggregate is requested.
Every document is scored ``--repeats`` times and the final value of each metric is
the arithmetic mean of the runs.

Documents are anonymised (method names removed) and randomly reordered before
scoring, and the reference white paper is never passed to the evaluator.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import EvalSettings, build_client, resolve_settings
from metrics import (
    GROUP_RUBRICS,
    METRIC_DEFINITIONS,
    METRIC_ORDER,
    format_rubric,
)

SYSTEM_PROMPT = "You are a strict and fair evaluator."

USER_TEMPLATE = """[{instruction}]

Response:
[{response}]

Rubric:
{rubric}

Provide your evaluation in the following format:
[RESULT] Score: <score from 0 to 5>
<your detailed feedback>"""

QUESTION_TEMPLATE = (
    "You are a strict and fair evaluator. Below is a technical white paper written "
    "for the topic '{topic}'. Score each metric listed below independently on a 0-5 "
    "scale, using the definition of the metric and the shared rubric.\n\n"
    "Metrics:\n{metric_block}"
)

SCORE_RE = re.compile(r"\[RESULT\]\s*Score:\s*([0-5](?:\.\d+)?)", re.IGNORECASE)
SCORE_FALLBACK_RE = re.compile(r"Score:\s*([0-5](?:\.\d+)?)", re.IGNORECASE)

MAX_ARTICLE_CHARS = 120_000


@dataclass
class Document:
    """One generated white paper to be scored."""

    topic: str
    path: Path
    method: str = ""
    text: str = ""
    scores: dict[str, list[float]] = field(default_factory=dict)
    feedback: dict[str, list[str]] = field(default_factory=dict)

    def mean(self, metric: str) -> float | None:
        values = self.scores.get(metric) or []
        if not values:
            return None
        return sum(values) / len(values)

    def overall(self) -> float | None:
        values = [self.mean(metric) for metric in METRIC_ORDER]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return sum(values) / len(values)


def anonymize(text: str, method_names: list[str]) -> str:
    """Remove system-related identifiers so the evaluator cannot recognise a method."""
    for name in method_names:
        if not name:
            continue
        text = re.sub(re.escape(name), "System", text, flags=re.IGNORECASE)
    return text


def read_document(path: Path, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Read a generated white paper, trimming the middle if it is very long."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n[middle part omitted for length]\n\n" + tail


def build_group_prompt(group: str, topic: str, article: str) -> str:
    """Build the prompt for one metric group."""
    metric_lines = []
    for metric in group_metrics(group):
        info = METRIC_DEFINITIONS[metric]
        line = f"- {metric}: {info['definition']}"
        if info.get("question"):
            line += f" Question: {info['question']}"
        metric_lines.append(line)

    instruction = QUESTION_TEMPLATE.format(
        topic=topic,
        metric_block="\n".join(metric_lines),
    )
    return USER_TEMPLATE.format(
        instruction=instruction,
        response=article,
        rubric=format_rubric(GROUP_RUBRICS[group]),
    )


def group_metrics(group: str) -> tuple[str, ...]:
    from metrics import METRIC_GROUPS

    return METRIC_GROUPS[group]


def parse_scores(output: str, group: str) -> dict[str, float]:
    """Extract one score per metric from a group response.

    The evaluator is asked to return ``[RESULT] Score: <n>`` per metric. Scores are
    matched in the order the metrics were listed; when the model labels them, the
    label is used instead.
    """
    metrics = group_metrics(group)
    results: dict[str, float] = {}

    for metric in metrics:
        pattern = re.compile(
            rf"{metric}[^\n]*?\[RESULT\]\s*Score:\s*([0-5](?:\.\d+)?)",
            re.IGNORECASE,
        )
        match = pattern.search(output)
        if match:
            results[metric] = float(match.group(1))

    if len(results) == len(metrics):
        return results

    plain = [float(m.group(1)) for m in SCORE_RE.finditer(output)]
    if not plain:
        plain = [float(m.group(1)) for m in SCORE_FALLBACK_RE.finditer(output)]

    for metric, value in zip(metrics, plain):
        results.setdefault(metric, value)
    return results


class Evaluator:
    """Call the evaluator model for one document."""

    def __init__(
        self,
        settings: EvalSettings | None = None,
        *,
        retries: int = 3,
        method_names: list[str] | None = None,
    ) -> None:
        self.settings = settings or resolve_settings()
        self.client = build_client(self.settings)
        self.retries = retries
        self.method_names = list(method_names or DEFAULT_METHOD_NAMES)

    def score_group(self, group: str, topic: str, article: str) -> dict[str, float]:
        prompt = build_group_prompt(group, topic, article)
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.settings.max_output_tokens,
                    temperature=self.settings.temperature,
                )
                output = response.choices[0].message.content or ""
                scores = parse_scores(output, group)
                if scores:
                    return scores
                last_error = ValueError("no score found in the response")
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
            time.sleep(min(2**attempt, 8))

        raise RuntimeError(f"scoring failed for group {group}: {last_error}")

    def score_document(self, topic: str, text: str, repeats: int = 3) -> Document:
        document = Document(topic=topic, path=Path(""), text=text)
        for _ in range(repeats):
            for group in GROUP_RUBRICS:
                scores = self.score_group(group, topic, text)
                for metric, value in scores.items():
                    document.scores.setdefault(metric, []).append(value)
        return document


DEFAULT_METHOD_NAMES = [
    "TWP-Gen",
    "TWP-GEN",
    "DirectRAG",
    "Direct RAG",
    "STORM",
    "OmniThink",
    "ConvergeWriter",
]


def load_documents(article_dir: Path, method_names: list[str] | None = None) -> list[Document]:
    """Load every ``*.md`` article in a directory as an anonymised document."""
    names = method_names or DEFAULT_METHOD_NAMES
    documents = []
    for path in sorted(article_dir.glob("*.md")):
        text = anonymize(read_document(path), names)
        documents.append(Document(topic=path.stem, path=path, text=text))
    return documents


def randomly_reorder(documents: list[Document], seed: int | None = None) -> list[Document]:
    """Randomise presentation order before scoring."""
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    return shuffled
