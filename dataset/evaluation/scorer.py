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

Return one independent score for every metric. Use the metric keys and order
shown below, do not omit any metric, and do not return a group-level aggregate:
{score_format}

After all score lines, provide brief supporting feedback."""

QUESTION_TEMPLATE = (
    "You are a strict and fair evaluator. Below is a technical white paper written "
    "for the topic '{topic}'. Score each metric listed below independently on a 0-5 "
    "scale, using the definition of the metric and the shared rubric.\n\n"
    "Metrics:\n{metric_block}"
)

SCORE_RE = re.compile(r"\[RESULT\]\s*Score:\s*([0-5](?:\.\d+)?)", re.IGNORECASE)
SCORE_FALLBACK_RE = re.compile(r"Score:\s*([0-5](?:\.\d+)?)", re.IGNORECASE)

# Keep the complete prompt within a 32k-class context window. Chinese technical
# prose is substantially denser in tokens than English prose, so character limits
# are deliberately conservative.
MAX_ARTICLE_CHARS = 18_000
MAX_EVIDENCE_CHARS = 12_000


@dataclass
class Document:
    """One generated white paper to be scored."""

    topic: str
    path: Path
    method: str = ""
    text: str = ""
    evidence: dict = field(default_factory=dict)
    reference_path: Path | None = None
    diagnostic_path: Path | None = None
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


def build_group_prompt(
    group: str,
    topic: str,
    article: str,
    evidence: dict | None = None,
) -> str:
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
    response = article
    if group == "Evidence Credibility":
        if evidence:
            evidence_json = compact_evidence_json(evidence, MAX_EVIDENCE_CHARS)
            response += (
                "\n\nEvidence package for citation and factuality checking:\n"
                + evidence_json
            )
        else:
            response += "\n\nEvidence package: unavailable."
    score_format = "\n".join(
        f"{metric} [RESULT] Score: <score from 0 to 5>"
        for metric in group_metrics(group)
    )
    return USER_TEMPLATE.format(
        instruction=instruction,
        response=response,
        rubric=format_rubric(GROUP_RUBRICS[group]),
        score_format=score_format,
    )


def _read_json(path: Path | None, default):
    if path is None or not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def compact_evidence_json(evidence: dict, max_chars: int) -> str:
    """Fit evidence into the prompt without dropping cited source records.

    All actually cited source metadata remains present. If compaction is needed,
    only long article passages, reference excerpts, and uncited excerpts are
    shortened. This preserves citation coverage while preventing context overflow.
    """
    payload = json.loads(json.dumps(evidence, ensure_ascii=False))

    def render() -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    text = render()
    if len(text) <= max_chars:
        return text

    for passage_limit, reference_limit, uncited_limit in (
        (320, 420, 240),
        (180, 240, 120),
        (100, 140, 80),
    ):
        for pair in payload.get("cited_passage_reference_pairs", []):
            pair["article_passage"] = str(pair.get("article_passage", ""))[:passage_limit]
            pair["reference_excerpt"] = str(pair.get("reference_excerpt", ""))[:reference_limit]
        payload["uncited_article_excerpts"] = [
            str(value)[:uncited_limit]
            for value in payload.get("uncited_article_excerpts", [])
        ]
        text = render()
        if len(text) <= max_chars:
            return text

    # The list of cited sources is never removed. At the final fallback, uncited
    # excerpts are omitted and each cited pair keeps a short alignment sample.
    payload["uncited_article_excerpts"] = []
    for pair in payload.get("cited_passage_reference_pairs", []):
        pair["article_passage"] = str(pair.get("article_passage", ""))[:80]
        pair["reference_excerpt"] = str(pair.get("reference_excerpt", ""))[:100]
    return render()


def build_evidence_package(
    article: str,
    reference_path: Path | None,
    diagnostic_path: Path | None,
) -> dict:
    """Build the evidence input described by the paper's scoring protocol.

    ``run_postoutline_experiment.py`` writes one reference JSON and one diagnostic
    JSON per article.  This function turns those artifacts into document-level
    citation statistics, cited passage/reference pairs, source metadata and
    uncited article excerpts for the Evidence Credibility scoring call.
    """
    references = _read_json(reference_path, [])
    if isinstance(references, dict):
        references = references.get("references", references.get("sources", []))
    references = references if isinstance(references, list) else []
    by_id = {
        int(item["id"]): item
        for item in references
        if isinstance(item, dict) and str(item.get("id", "")).isdigit()
    }

    paragraphs = []
    for block in re.split(r"\n\s*\n", article):
        text = " ".join(
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if text and not text.startswith("|"):
            paragraphs.append(text)

    citation_ids = [int(value) for value in re.findall(r"\[\^(\d+)\]", article)]
    cited_pairs_by_id: dict[int, dict] = {}
    uncited_excerpts = []
    for paragraph in paragraphs:
        ids = [int(value) for value in re.findall(r"\[\^(\d+)\]", paragraph)]
        if not ids:
            if len(re.sub(r"\s+", "", paragraph)) >= 45:
                uncited_excerpts.append(paragraph[:1200])
            continue
        for ref_id in dict.fromkeys(ids):
            source = by_id.get(ref_id, {})
            cited_pairs_by_id.setdefault(
                ref_id,
                {
                    "citation_id": ref_id,
                    "article_passage": paragraph[:800],
                    "reference_excerpt": str(source.get("content", ""))[:1000],
                    "source_title": source.get("title", ""),
                    "source_url": source.get("url", ""),
                },
            )

    diagnostics = _read_json(diagnostic_path, {})
    if isinstance(diagnostics, list):
        diagnostics = {"section_diagnostics": diagnostics}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    cited_paragraphs = sum(bool(re.search(r"\[\^\d+\]", text)) for text in paragraphs)
    statistics = {
        "citation_mentions": len(citation_ids),
        "unique_citations": len(set(citation_ids)),
        "source_records": len(references),
        "invalid_citation_ids": sorted(set(citation_ids) - set(by_id)),
        "cited_paragraphs": cited_paragraphs,
        "eligible_paragraphs": len(paragraphs),
        "paragraph_citation_coverage": (
            round(cited_paragraphs / len(paragraphs), 4) if paragraphs else 0.0
        ),
    }
    for key in (
        "citation_coverage",
        "sentence_citation_coverage",
        "citations",
        "unique_citations",
        "source_documents",
    ):
        if key in diagnostics:
            statistics[f"generator_{key}"] = diagnostics[key]

    cited_reference_ids = list(dict.fromkeys(citation_ids))
    cited_references = [by_id[ref_id] for ref_id in cited_reference_ids if ref_id in by_id]
    return {
        "citation_statistics": statistics,
        "cited_passage_reference_pairs": [
            cited_pairs_by_id[ref_id]
            for ref_id in cited_reference_ids
            if ref_id in cited_pairs_by_id
        ],
        "sources": [
            {
                "citation_id": item.get("id"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
            }
            for item in cited_references
            if isinstance(item, dict)
        ],
        "uncited_article_excerpts": uncited_excerpts[:8],
    }


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

    def score_group(
        self,
        group: str,
        topic: str,
        article: str,
        evidence: dict | None = None,
    ) -> dict[str, float]:
        prompt = build_group_prompt(group, topic, article, evidence)
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
                if len(scores) == len(group_metrics(group)):
                    return scores
                last_error = ValueError(
                    f"expected {len(group_metrics(group))} scores, got {len(scores)}"
                )
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
            time.sleep(min(2**attempt, 8))

        raise RuntimeError(f"scoring failed for group {group}: {last_error}")

    def score_document(
        self,
        topic: str,
        text: str,
        repeats: int = 3,
        evidence: dict | None = None,
    ) -> Document:
        document = Document(topic=topic, path=Path(""), text=text, evidence=evidence or {})
        for _ in range(repeats):
            for group in GROUP_RUBRICS:
                scores = self.score_group(group, topic, text, document.evidence)
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


def load_documents(
    article_dir: Path,
    method_names: list[str] | None = None,
    reference_dir: Path | None = None,
    diagnostic_dir: Path | None = None,
) -> list[Document]:
    """Load every ``*.md`` article in a directory as an anonymised document."""
    names = method_names or DEFAULT_METHOD_NAMES
    documents = []
    for path in sorted(article_dir.glob("*.md")):
        text = anonymize(read_document(path), names)
        reference_path = reference_dir / f"{path.stem}.json" if reference_dir else None
        diagnostic_path = diagnostic_dir / f"{path.stem}.json" if diagnostic_dir else None
        evidence = build_evidence_package(text, reference_path, diagnostic_path)
        documents.append(
            Document(
                topic=path.stem,
                path=path,
                text=text,
                evidence=evidence,
                reference_path=reference_path,
                diagnostic_path=diagnostic_path,
            )
        )
    return documents


def randomly_reorder(documents: list[Document], seed: int | None = None) -> list[Document]:
    """Randomise presentation order before scoring."""
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    return shuffled
