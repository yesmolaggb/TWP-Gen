#!/usr/bin/env python3
"""Evidence-grounded article generation for the fixed ten-topic experiment.

This script intentionally starts from already generated outline files. It does not
modify retrieval, clustering, taxonomy mapping, or outline induction. The only
additional work is performed after an outline exists:

1. retrieve section-specific passages from the topic's existing source package;
2. generate each top-level section with explicit evidence identifiers;
3. run a lightweight citation/length gate and repair only failed sections.

The original outlines and source packages are read-only. Outputs are written to an
experiment directory so the baseline pipeline remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI

from llm_settings import build_client, load_env, resolve_settings


TOPICS = (
    "HDM技术白皮书",
    "MagicOS安全技术白皮书",
    "基于智能体的校园自智网络技术白皮书",
    "轻质屋面光伏系统安全技术白皮书",
    "酒业数智供应链发展技术白皮书",
    "医院通用人工智能平台技术白皮书",
    "HiSec Endpoint智能终端安全系统技术白皮书",
    "云AI视频技术白皮书",
    "Atlas 300T Pro训练卡技术白皮书",
    "华为园区网络Wi-Fi 7零漫游技术白皮书",
)


SOURCE_DIR_ALIASES = {
    "HiSec Endpoint智能终端安全系统技术白皮书": "HiSec_Endpoint智能终端安全系统技术白皮书",
    "Atlas 300T Pro训练卡技术白皮书": "Atlas_300T_Pro训练卡技术白皮书",
    "华为园区网络Wi-Fi 7零漫游技术白皮书": "华为园区网络Wi-Fi_7零漫游技术白皮书",
}


SYSTEM_PROMPT = """你是一名资深的中文技术白皮书作者，面向专业读者撰写可核验、信息密度高、技术解释深入的技术内容。

写作要求：
1. 只使用“章节证据”中能够直接支持的信息；不得引入证据之外的数值、标准、产品能力或应用场景。
2. 充分利用给定证据：把技术原理、系统组成、模块关系、交互流程、适用条件、性能数据、限制与风险展开写清楚，不要只做摘要式复述或罗列功能名称。
3. 遇到证据中出现的不同版本、方案、部署方式或技术路线时，明确比较差异、适用场景与取舍。
4. 引用只加在“含具体技术事实”的句子上：出现数值、标准编号、协议或接口名称、算法与模块名称、明确工作机制或可测效果时才引用，格式为 [^数字]，且该编号必须直接支持该句。概括性描述、评价性判断、过渡句和小结句一律不加引用。
5. 宁可少引用也不要弱引用：找不到直接对应证据的句子，改写为证据能支持的表述或删除，不得挂一个只是“相关”的泛化引用。
6. 引用总量要克制：每个三级小节通常 1–3 处引用，同一编号在同一段落最多出现两次；不同来源分别承担其直接支持的论断，避免同一编号反复覆盖互不相关的内容。
7. 不得把不同厂商、产品、版本或应用场景的数据混为一谈；数值、标准名称和产品能力必须与证据中的实体一致。
8. 数值必须逐字来自证据：写出任何百分比、倍数、容量、速率、年限、温度或标准编号前，必须确认该数字在证据原文中确实出现；证据没有就不要写，也不要近似或换算。
9. 严格使用简体中文，使用 Markdown 标题组织内容；一级章节下应有清晰的三级小节。缩写或英文术语首次出现时写成“中文名称（英文全称，缩写）”，全文后续统一使用同一叫法。
10. 直接输出最终正文。不要输出写作说明、证据分析过程，也不要输出“证据未提及”“本节不讨论”等过程性注释。
11. 讲机制要讲透：说明模块之间如何依赖与协同、关键实现步骤的先后关系、参数或策略的作用，以及不同方案之间的取舍；按“证据事实—工作机制—工程影响或适用边界”展开，推论必须能由相邻证据直接推出。
12. 在不编造事实的前提下补充信息增益：结合证据说明相关技术标准、版本演进、部署约束与工程权衡。对未来趋势、竞品对比和因果判断保持克制，证据未明确给出时不得自行补充。
13. 每个案例、项目或产品实例只在其最合适的章节完整介绍一次；其他章节需要提及时用一句话承接，不得重复展开同一组数据。
14. 章节职责要分清：概述写定位与边界，背景写问题与动因，总体设计写架构与模块关系，方法原理写机制与实现步骤，应用场景写条件、案例与效果，评测与实验写设置与结果，安全与合规写风险与规范，结论只做综合判断和有依据的展望。
15. 篇幅要求：每个一级章节的正文不少于 1500 字，证据充分时写到 2000–2800 字；每个三级小节 350–700 字。宁可写长一些，也不要因为概括而丢掉证据里的机制与参数。
16. 每一个三级小节只讨论一个技术问题，段落之间要有承接句说明“为什么接着说这件事”，不要出现没有过渡的主题跳转。
17. 不要出现连续三句可核查技术论述都没有引用的情况；同时避免每句都挂引用导致正文被标记淹没。
"""


BASELINE_SYSTEM_PROMPT = """你是一名中文技术报告作者。请依据给定大纲和参考材料撰写内容，保持正式、连贯、准确，并使用 [^数字] 标注参考材料。不要输出提示词或写作说明。"""


@dataclass
class OutlineNode:
    level: int
    title: str
    notes: list[str] = field(default_factory=list)
    children: list["OutlineNode"] = field(default_factory=list)

    def render(self, base_level: int | None = None) -> str:
        base = self.level if base_level is None else base_level
        lines = [f"{'#' * max(1, base)} {self.title}"]
        lines.extend(f"- {note}" for note in self.notes)
        for child in self.children:
            lines.append(child.render(base + 1))
        return "\n".join(lines)

    def query_text(self) -> str:
        parts = [self.title, *self.notes]
        for child in self.children:
            parts.append(child.query_text())
        return " ".join(parts)


@dataclass
class SourceDocument:
    ref_id: int
    source_id: str
    title: str
    url: str
    relevance: float
    body: str
    path: str


@dataclass
class EvidenceChunk:
    source: SourceDocument
    text: str
    tokens: Counter[str]
    length: int


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs into os.environ (kept for backward compatibility).

    All API/model settings are resolved centrally in ``llm_settings``; this thin
    wrapper exists so existing callers keep working.
    """
    load_env(path)


def safe_filename(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", value).strip()[:160]


def strip_numbering(title: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*[\.、]?\s*", "", title).strip()


def locate_outline_file(outline_root: Path, topic: str) -> Path:
    """Accept both layouts used across the pipeline.

    * ``<root>/<topic>/outline.txt``  (numbered outline, one folder per topic)
    * ``<root>/<topic>/outline.md``
    * ``<root>/<topic>.md``           (outline_generator / batch_from_outline layout)
    """
    candidates = (
        outline_root / topic / "outline.txt",
        outline_root / topic / "outline.md",
        outline_root / f"{topic}.md",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "outline not found for topic "
        f"{topic!r}; tried: " + ", ".join(str(path) for path in candidates)
    )


def parse_outline(path: Path, default_title: str | None = None) -> tuple[str, list[OutlineNode]]:
    if default_title:
        title = default_title
    elif path.stem.lower() in {"outline", "大纲"}:
        title = path.parent.name
    else:
        title = path.stem
    roots: list[OutlineNode] = []
    stack: list[OutlineNode] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("题目：") or line.startswith("题目:"):
            title = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            continue
        match = re.match(r"^(\d+(?:\.\d+)*)[\.、]?\s+(.+)$", line)
        if match:
            level = match.group(1).count(".") + 1
            node = OutlineNode(level=level, title=strip_numbering(match.group(2)))
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
            continue
        if line.startswith(("-", "•", "·")) and stack:
            stack[-1].notes.append(line[1:].strip())
    roots = [node for node in roots if "附录" not in node.title]
    if not roots:
        raise ValueError(f"No sections parsed from {path}")
    return title, roots


def extract_metadata(text: str, path: Path, ref_id: int) -> SourceDocument:
    lines = text.splitlines()
    title = path.stem
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
    source_id_match = re.search(r"Source ID:\s*`?([^`\n]+)`?", text)
    url_match = re.search(r"URL:\s*(\S+)", text)
    relevance_match = re.search(r"Relevance score:\s*([0-9.]+)", text)
    marker = text.find("## Search Snippet")
    body = text[marker + len("## Search Snippet") :] if marker >= 0 else text
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return SourceDocument(
        ref_id=ref_id,
        source_id=source_id_match.group(1).strip() if source_id_match else path.stem,
        title=title,
        url=url_match.group(1).strip() if url_match else "",
        relevance=float(relevance_match.group(1)) if relevance_match else 0.0,
        body=body,
        path=str(path),
    )


def load_sources(source_dir: Path) -> list[SourceDocument]:
    files = sorted(source_dir.glob("*.md"))
    sources = [
        extract_metadata(path.read_text(encoding="utf-8", errors="ignore"), path, index)
        for index, path in enumerate(files, 1)
    ]
    if not sources:
        raise FileNotFoundError(f"No source markdown files found in {source_dir}")
    return sources


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    return text.strip()


GENERIC_TOPIC_TERMS = (
    "技术白皮书", "白皮书", "技术", "发展", "研究", "系统", "平台", "方案",
)


def topic_anchor_tokens(topic: str) -> set[str]:
    """Return conservative lexical anchors used only for post-outline evidence filtering."""
    core = topic.lower()
    for term in GENERIC_TOPIC_TERMS:
        core = core.replace(term, "")
    anchors = set(re.findall(r"[a-z][a-z0-9+.-]{2,}", core))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", core):
        if len(sequence) <= 3:
            anchors.add(sequence)
        else:
            anchors.update(sequence[i : i + 2] for i in range(len(sequence) - 1))
    return {anchor for anchor in anchors if anchor}


def is_navigation_or_boilerplate(text: str) -> bool:
    """Reject crawler navigation blocks before they can be treated as evidence."""
    markdown_links = len(re.findall(r"\[[^\]]+\]\([^\)]+\)", text))
    image_links = text.count("![")
    nav_markers = sum(
        marker in text
        for marker in ("返回+", "热门推荐", "产品与解决方案", "行业解决方案", "营销资料")
    )
    return markdown_links >= 6 or image_links >= 2 or (markdown_links >= 3 and nav_markers >= 1)


def is_topic_relevant(topic: str, chunk: EvidenceChunk) -> bool:
    anchors = topic_anchor_tokens(topic)
    if not anchors:
        return True
    haystack = f"{chunk.source.title} {chunk.text}".lower()
    matched = {anchor for anchor in anchors if anchor in haystack}
    latin = {anchor for anchor in anchors if re.fullmatch(r"[a-z][a-z0-9+.-]{2,}", anchor)}
    if latin:
        return bool(matched.intersection(latin))
    required = 1 if len(anchors) <= 2 else 2
    return len(matched) >= required


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for latin in re.findall(r"[a-z][a-z0-9_+.-]{1,}", text):
        tokens.append(latin)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[i : i + 2] for i in range(len(sequence) - 1))
            if len(sequence) >= 3:
                tokens.extend(sequence[i : i + 3] for i in range(len(sequence) - 2))
    return tokens


def split_chunks(source: SourceDocument, target_chars: int = 900) -> list[EvidenceChunk]:
    paragraphs = [normalize_text(p) for p in re.split(r"\n\s*\n|(?=^#{1,4}\s)", source.body, flags=re.M)]
    paragraphs = [p for p in paragraphs if len(p) >= 30]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        if current and current_size + len(paragraph) > target_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        if len(paragraph) > target_chars * 2:
            for start in range(0, len(paragraph), target_chars):
                part = paragraph[start : start + target_chars]
                if len(part) >= 80:
                    chunks.append(part)
            continue
        current.append(paragraph)
        current_size += len(paragraph)
    if current:
        chunks.append("\n".join(current))
    unique: list[EvidenceChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if is_navigation_or_boilerplate(chunk):
            continue
        key = hashlib.sha1(re.sub(r"\W+", "", chunk).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        token_counts = Counter(tokenize(source.title + " " + chunk))
        if token_counts:
            unique.append(EvidenceChunk(source, chunk, token_counts, sum(token_counts.values())))
    return unique


class BM25Index:
    def __init__(self, chunks: Sequence[EvidenceChunk]):
        self.chunks = list(chunks)
        self.avg_length = sum(c.length for c in self.chunks) / max(1, len(self.chunks))
        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            document_frequency.update(chunk.tokens.keys())
        total = len(self.chunks)
        self.idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def score(self, query: str, chunk: EvidenceChunk) -> float:
        query_tokens = Counter(tokenize(query))
        score = 0.0
        k1, b = 1.35, 0.72
        norm = k1 * (1.0 - b + b * chunk.length / max(1.0, self.avg_length))
        for term, query_weight in query_tokens.items():
            frequency = chunk.tokens.get(term, 0)
            if not frequency:
                continue
            score += self.idf.get(term, 0.0) * (frequency * (k1 + 1.0) / (frequency + norm)) * min(2, query_weight)
        score += min(0.8, chunk.source.relevance) * 0.35
        title_tokens = set(tokenize(chunk.source.title))
        score += 0.25 * len(title_tokens.intersection(query_tokens))
        return score

    def retrieve(
        self,
        query: str,
        limit: int = 12,
        max_chars: int = 14500,
        max_per_source: int = 2,
    ) -> list[EvidenceChunk]:
        ranked = sorted(
            ((self.score(query, chunk), chunk) for chunk in self.chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        selected: list[EvidenceChunk] = []
        source_counts: Counter[int] = Counter()
        total_chars = 0
        for score, chunk in ranked:
            if score <= 0:
                break
            source_id = chunk.source.ref_id
            if source_counts[source_id] >= max_per_source:
                continue
            if selected and total_chars + len(chunk.text) > max_chars:
                continue
            selected.append(chunk)
            source_counts[source_id] += 1
            total_chars += len(chunk.text)
            if len(selected) >= limit:
                break
        return selected


def evidence_text(chunks: Sequence[EvidenceChunk]) -> str:
    blocks = []
    for chunk in chunks:
        source = chunk.source
        blocks.append(
            f"[证据 ^{source.ref_id}]\n"
            f"标题：{source.title}\n"
            f"来源：{source.url}\n"
            f"内容：{chunk.text}"
        )
    return "\n\n".join(blocks)


def valid_reference_ids(chunks: Sequence[EvidenceChunk]) -> set[int]:
    return {chunk.source.ref_id for chunk in chunks}


def citation_ids(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[\^(\d+)\]", text)]


def paragraph_citation_coverage(text: str) -> float:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        stripped = " ".join(lines)
        if not stripped or stripped.startswith("|"):
            continue
        if len(re.sub(r"\[\^\d+\]", "", stripped)) < 45:
            continue
        paragraphs.append(stripped)
    if not paragraphs:
        return 0.0
    cited = sum(bool(re.search(r"\[\^\d+\]", paragraph)) for paragraph in paragraphs)
    return cited / len(paragraphs)


def sentence_citation_coverage(text: str) -> float:
    """Fraction of factual Chinese sentences that carry at least one citation."""
    plain = re.sub(r"^#{1,4}\s.*$", "", text, flags=re.M)
    sentences = [s.strip() for s in re.split(r"(?<=[。！？])", plain) if s.strip()]
    factual = []
    for sentence in sentences:
        stripped = re.sub(r"\[\^\d+\]", "", sentence)
        if len(re.sub(r"[\s*`#|]", "", stripped)) < 12:
            continue
        if stripped.startswith(("|", "-", "1.", "2.", "3.", "4.", "5.")) and len(stripped) < 30:
            continue
        factual.append(sentence)
    if not factual:
        return 0.0
    return sum(bool(re.search(r"\[\^\d+\]", s)) for s in factual) / len(factual)


def best_used_snippets(text: str, chunks: Sequence[EvidenceChunk]) -> dict[int, str]:
    """Build a compact, claim-aligned excerpt for every cited source.

    A source can support several claims from different parts of a long page.  Keeping
    only one whole chunk often hides the exact supporting sentences after the evaluator
    truncates the reference record.  We therefore select original evidence sentences
    against the actual citing sentences and pack the best non-duplicate matches into a
    short excerpt.  No generated wording is inserted into the reference record.
    """
    by_id: dict[int, list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        by_id[chunk.source.ref_id].append(chunk)
    cited_sentences: dict[int, list[str]] = defaultdict(list)
    for sentence in re.split(r"(?<=[。！？；])", text):
        for value in re.findall(r"\[\^(\d+)\]", sentence):
            cited_sentences[int(value)].append(sentence)
    result: dict[int, str] = {}
    for ref_id, sentences in cited_sentences.items():
        candidates: list[tuple[float, str]] = []
        evidence_sentences: list[str] = []
        for chunk in by_id.get(ref_id, []):
            evidence_sentences.extend(
                part.strip()
                for part in re.split(r"(?<=[。！？；.!?])\s*|\n+", normalize_text(chunk.text))
                if len(part.strip()) >= 18
            )
        evidence_sentences = list(dict.fromkeys(evidence_sentences))
        for citing in sentences:
            query = set(tokenize(re.sub(r"\[\^\d+\]", "", citing)))
            for passage in evidence_sentences:
                passage_tokens = set(tokenize(passage))
                overlap = len(query.intersection(passage_tokens))
                if not overlap:
                    continue
                precision = overlap / max(1, len(query))
                candidates.append((overlap + 2.0 * precision, passage))
        selected: list[str] = []
        used_chars = 0
        for _, passage in sorted(candidates, key=lambda item: item[0], reverse=True):
            if passage in selected:
                continue
            addition = len(passage) + (1 if selected else 0)
            if selected and used_chars + addition > 880:
                continue
            selected.append(passage)
            used_chars += addition
            if used_chars >= 760 or len(selected) >= 8:
                break
        if selected:
            result[ref_id] = " ".join(selected)[:900]
        elif by_id.get(ref_id):
            result[ref_id] = normalize_text(by_id[ref_id][0].text)[:900]
    return result


def citation_distribution(text: str) -> dict:
    ids = citation_ids(text)
    counts = Counter(ids)
    total = sum(counts.values())
    dominant_id, dominant_count = (counts.most_common(1)[0] if counts else (None, 0))
    return {
        "unique": len(counts),
        "total": total,
        "dominant_id": dominant_id,
        "dominant_ratio": dominant_count / max(1, total),
        "counts": dict(counts),
    }


def misaligned_numeric_citations(text: str, chunks: Sequence[EvidenceChunk]) -> list[str]:
    """Flag numeric claims whose cited source does not contain the same value."""
    by_id: dict[int, str] = defaultdict(str)
    for chunk in chunks:
        by_id[chunk.source.ref_id] += " " + normalize_text(chunk.text)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:\s*[~～—-]\s*\d+(?:\.\d+)?)?\s*"
        r"(?:%|％|ms|Mbps|Gbps|GB|MB|KB|GT/s|秒|分钟|小时|位|台|个|次|倍|年|MPa|℃|V|A|W|kW|MW)"
    )

    def compact(value: str) -> str:
        return re.sub(r"\s+", "", value).lower().replace("％", "%").replace("～", "~").replace("—", "-")

    problems: list[str] = []
    for sentence in re.split(r"(?<=[。！？；])", text):
        values = pattern.findall(re.sub(r"\[\^\d+\]", "", sentence))
        cited = [int(value) for value in re.findall(r"\[\^(\d+)\]", sentence)]
        if not values or not cited:
            continue
        cited_evidence = compact(" ".join(by_id.get(ref_id, "") for ref_id in cited))
        if any(compact(value) not in cited_evidence for value in values):
            problems.append(normalize_text(sentence)[:300])
    return list(dict.fromkeys(problems))[:12]


def enrich_citations(text: str, chunks: Sequence[EvidenceChunk], min_overlap: int = 4) -> str:
    """Attach the best-matching (and least-used) evidence id to uncited factual sentences."""
    by_id: dict[int, list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        by_id[chunk.source.ref_id].append(chunk)
    if not by_id:
        return text
    usage: Counter[int] = Counter(int(v) for v in re.findall(r"\[\^(\d+)\]", text))
    rendered: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "|")):
            rendered.append(line)
            continue
        sentences = re.split(r"(?<=[。！？])", line)
        rebuilt: list[str] = []
        for sentence in sentences:
            body = re.sub(r"\[\^\d+\]", "", sentence)
            if re.search(r"\[\^\d+\]", sentence) or len(re.sub(r"\W", "", body)) < 14:
                rebuilt.append(sentence)
                continue
            query = set(tokenize(sentence))
            best_id = None
            best_score = 0
            for ref_id, chunk_list in by_id.items():
                for chunk in chunk_list:
                    score = len(query.intersection(chunk.tokens.keys()))
                    if score > best_score or (
                        score == best_score and score > 0 and usage[ref_id] < usage[best_id or ref_id]
                    ):
                        best_score = score
                        best_id = ref_id
            if best_id is not None and best_score >= min_overlap:
                if sentence and sentence[-1] in "。！？":
                    sentence = sentence[:-1] + f"[^{best_id}]" + sentence[-1]
                else:
                    sentence = sentence + f"[^{best_id}]"
                usage[best_id] += 1
            rebuilt.append(sentence)
        rendered.append("".join(rebuilt))
    return "\n".join(rendered)


def unsupported_numeric_claims(text: str, evidence: str) -> list[str]:
    plain = re.sub(r"\[\^\d+\]", "", text)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:\s*[~-]\s*\d+(?:\.\d+)?)?\s*"
        r"(?:%|％|ms|Mbps|Gbps|GB|MB|KB|GT/s|秒|分钟|小时|位|台|个|次|倍)"
    )
    def canonical(value: str) -> str:
        value = re.sub(r"\s+", "", value).lower()
        value = re.sub(r"(?<=\d)s\b", "秒", value)
        value = value.replace("％", "%").replace("～", "~").replace("—", "-")
        for chinese, arabic in {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
        }.items():
            value = value.replace(chinese + "个", arabic + "个")
        return value

    compact_evidence = canonical(evidence)
    unsupported = []
    for match in pattern.findall(plain):
        compact = canonical(match)
        if compact not in compact_evidence and match not in unsupported:
            unsupported.append(match)
    return unsupported


def unexpected_third_level_headings(text: str, section: OutlineNode) -> list[str]:
    allowed = {child.title.strip() for child in section.children}
    observed = [title.strip() for title in re.findall(r"^###\s+(.+)$", text, flags=re.M)]
    return [title for title in observed if title not in allowed]


def quality_gate(
    text: str,
    allowed_ids: set[int],
    section: OutlineNode | None = None,
    evidence: str = "",
    chunks: Sequence[EvidenceChunk] = (),
    audit: dict | None = None,
) -> tuple[bool, dict]:
    used = citation_ids(text)
    invalid = sorted(set(used).difference(allowed_ids))
    clean_length = len(re.sub(r"[#*`\s]", "", text))
    coverage = paragraph_citation_coverage(text)
    sentence_coverage = sentence_citation_coverage(text)
    unexpected_headings = unexpected_third_level_headings(text, section) if section else []
    unsupported_numbers = unsupported_numeric_claims(text, evidence) if evidence else []
    misaligned_numbers = misaligned_numeric_citations(text, chunks) if chunks else []
    distribution = citation_distribution(text)
    required_unique = min(5, len(allowed_ids))
    audit_failures = [] if not audit else list(audit.get("unsupported_claims") or [])
    audit_failures.extend([] if not audit else list(audit.get("wrong_citations") or []))
    passed = (
        clean_length >= 1450
        and coverage >= 0.72
        and sentence_coverage >= 0.88
        and distribution["unique"] >= required_unique
        and distribution["dominant_ratio"] <= 0.5
        and not invalid
        and not unexpected_headings
        and not unsupported_numbers
        and not misaligned_numbers
        and not audit_failures
    )
    return passed, {
        "clean_length": clean_length,
        "paragraph_citation_coverage": round(coverage, 4),
        "sentence_citation_coverage": round(sentence_coverage, 4),
        "citation_count": len(used),
        "unique_citations": len(set(used)),
        "required_unique_citations": required_unique,
        "dominant_citation_id": distribution["dominant_id"],
        "dominant_citation_ratio": round(distribution["dominant_ratio"], 4),
        "citation_distribution": distribution["counts"],
        "invalid_citations": invalid,
        "unexpected_headings": unexpected_headings,
        "unsupported_numeric_claims": unsupported_numbers,
        "misaligned_numeric_citations": misaligned_numbers,
        "grounding_audit": audit or {},
    }


def section_quality_score(diagnostics: dict) -> float:
    """Rank alternative drafts while preventing fact repair from collapsing depth."""
    clean_length = int(diagnostics.get("clean_length") or 0)
    sentence_coverage = float(diagnostics.get("sentence_citation_coverage") or 0.0)
    unique = int(diagnostics.get("unique_citations") or 0)
    required_unique = max(1, int(diagnostics.get("required_unique_citations") or 1))
    dominant_ratio = float(diagnostics.get("dominant_citation_ratio") or 1.0)
    audit = diagnostics.get("grounding_audit") or {}
    audit_problems = set(audit.get("unsupported_claims") or [])
    audit_problems.update(audit.get("wrong_citations") or [])
    hard_problems = (
        len(diagnostics.get("invalid_citations") or [])
        + len(diagnostics.get("unexpected_headings") or [])
        + len(diagnostics.get("unsupported_numeric_claims") or [])
        + len(diagnostics.get("misaligned_numeric_citations") or [])
    )
    score = 0.0
    score += 3.0 * min(clean_length, 1600) / 1600
    score += 2.0 * min(sentence_coverage, 0.95) / 0.95
    score += min(unique / required_unique, 1.0)
    score += max(0.0, 1.0 - dominant_ratio)
    score -= 0.55 * hard_problems
    score -= 0.12 * min(10, len(audit_problems))
    if clean_length < 950:
        score -= 6.0
    return score


def get_message_content(message) -> str:
    content = getattr(message, "content", None)
    if content:
        return content
    extra = getattr(message, "model_extra", None) or {}
    return extra.get("content", "") or ""


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    retries: int = 4,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if enable_thinking:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    extra_body={"enable_thinking": True},
                )
                parts: list[str] = []
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        parts.append(piece)
                content = "".join(parts).strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": False},
                )
                content = get_message_content(response.choices[0].message).strip()
            if content:
                return content
            raise RuntimeError("Model returned empty content")
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"Model call failed after {retries} attempts: {last_error}")


def retrieve_section_evidence(index: BM25Index, topic: str, section: OutlineNode) -> list[EvidenceChunk]:
    """Collect a diverse evidence bundle for both the section and its children."""
    candidates: list[EvidenceChunk] = []
    candidates.extend(
        index.retrieve(f"{topic} {section.title} {section.query_text()}", limit=12, max_chars=16000, max_per_source=3)
    )
    for child in section.children:
        candidates.extend(
            index.retrieve(f"{topic} {child.query_text()}", limit=5, max_chars=6000, max_per_source=3)
        )

    selected: list[EvidenceChunk] = []
    seen: set[tuple[int, str]] = set()
    source_counts: Counter[int] = Counter()
    total_chars = 0
    for chunk in candidates:
        if not is_topic_relevant(topic, chunk):
            continue
        digest = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()
        key = (chunk.source.ref_id, digest)
        if key in seen or source_counts[chunk.source.ref_id] >= 4:
            continue
        if selected and total_chars + len(chunk.text) > 42000:
            continue
        seen.add(key)
        source_counts[chunk.source.ref_id] += 1
        selected.append(chunk)
        total_chars += len(chunk.text)
        if len(selected) >= 34:
            break
    return selected


def extract_json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in planner response")
    payload = text[start : end + 1]
    # Model quotes may contain Markdown escapes such as \_ that are invalid in JSON.
    payload = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', payload)
    return json.loads(payload, strict=False)


GENERIC_TITLE_TERMS = (
    "技术", "应用", "方案", "功能", "设计", "实现", "管理", "支持", "优化",
    "核心", "系统", "机制", "场景", "领域", "范围", "总体", "方法", "原理",
    "发展", "需求", "流程", "模块", "分类", "保障", "评测", "实验", "行业",
    "价值", "总结", "方向",
)


def distinctive_title_terms(title: str) -> list[str]:
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{1,}", title)]
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", title))
    for generic in GENERIC_TITLE_TERMS:
        chinese = chinese.replace(generic, "|")
    terms.extend(
        part for part in re.split(r"[|与和及、]", chinese) if len(part) >= 2
    )
    return list(dict.fromkeys(terms))


def plan_grounded_section(
    client: OpenAI,
    planner_model: str,
    topic: str,
    section: OutlineNode,
    chunks: Sequence[EvidenceChunk],
) -> tuple[OutlineNode, dict]:
    """Filter unsupported child entries without changing the stored outline."""
    if not section.children:
        return deepcopy(section), {"supported_titles": [], "dropped_titles": [], "planner": "not_needed"}

    child_titles = [child.title for child in section.children]
    prompt = f"""判断《{topic}》中“{section.title}”章节的二级条目是否得到证据直接支持。

判定规则：
- 只有证据明确描述该技术对象、机制或应用时才保留。
- 不能因为证据中出现“智能、系统、平台、行业”等通用词就判定为支持。
- 不允许从服务器管理能力推导到交通、农业、医疗等没有直接陈述的应用。
- 不允许把其他产品、厂商或技术的能力移植到本文主题。
- 每个保留条目必须提供一段来自证据的连续原文作为quote，不得改写quote；quote必须明确出现该条目的关键对象或场景。

候选条目：
{json.dumps(child_titles, ensure_ascii=False)}

证据：
{evidence_text(chunks)}

只输出JSON：
{{"supported":[{{"title":"证据直接支持的原始标题","evidence_id":1,"quote":"证据中的连续原文"}}],"dropped_titles":["缺乏直接证据的原始标题"]}}
"""
    response = call_model(
        client,
        planner_model,
        [
            {"role": "system", "content": "你是技术证据核验员。只进行严格的证据支持判断，并输出合法JSON。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        temperature=0.0,
        enable_thinking=False,
    )
    try:
        plan = extract_json_object(response)
        evidence_by_id: dict[int, str] = defaultdict(str)
        for chunk in chunks:
            evidence_by_id[chunk.source.ref_id] += "\n" + normalize_text(chunk.text)
        supported: set[str] = set()
        accepted_evidence: dict[str, dict] = {}
        rejected_items: list[dict] = []
        for item in plan.get("supported") or []:
            title = str(item.get("title", "")).strip()
            quote = normalize_text(str(item.get("quote", "")))
            try:
                evidence_id = int(item.get("evidence_id"))
            except (TypeError, ValueError):
                evidence_id = -1
            source_text = evidence_by_id.get(evidence_id, "")
            exact_quote = len(quote) >= 12 and quote in source_text
            terms = distinctive_title_terms(title)
            term_match = not terms or any(term.lower() in quote.lower() for term in terms)
            if title in child_titles and exact_quote and term_match:
                supported.add(title)
                accepted_evidence[title] = {
                    "evidence_id": evidence_id,
                    "quote": quote[:300],
                    "distinctive_terms": terms,
                }
            else:
                rejected_items.append(
                    {
                        "title": title,
                        "evidence_id": evidence_id,
                        "exact_quote": exact_quote,
                        "term_match": term_match,
                        "distinctive_terms": terms,
                    }
                )
    except Exception as error:  # noqa: BLE001
        grounded = deepcopy(section)
        grounded.children = []
        return grounded, {
            "supported_titles": [],
            "dropped_titles": child_titles,
            "planner": "fallback_drop_unverified",
            "error": f"{type(error).__name__}: {error}",
            "raw_response": response[:1200],
        }

    grounded = deepcopy(section)
    grounded.children = [child for child in grounded.children if child.title in supported]
    dropped = [title for title in child_titles if title not in supported]
    return grounded, {
        "supported_titles": [title for title in child_titles if title in supported],
        "dropped_titles": dropped,
        "accepted_evidence": accepted_evidence,
        "rejected_items": rejected_items,
        "planner": planner_model,
    }


def audit_grounding(
    client: OpenAI,
    model: str,
    topic: str,
    section: OutlineNode,
    evidence: str,
    draft: str,
) -> dict:
    """Conservatively flag only clear evidence violations in a generated section."""
    prompt = f"""核验《{topic}》的“{section.title}”章节是否严格得到给定证据支持。

只检查以下问题：
1. 草稿中的具体数值、标准、算法、产品能力或应用行业在证据中找不到直接依据；
2. 引用编号存在，但对应证据并不支持引用前的事实；
3. 草稿把其他产品、厂商或网页导航信息当成本文主题的事实。

不要因为语言概括方式不同而误判；只报告明确的问题。若没有明确问题，数组保持为空。

【证据】
{evidence}

【草稿】
{draft}

只输出JSON：
{{"unsupported_claims":["草稿中的原句"],"wrong_citations":["草稿中的原句"],"notes":"简短说明"}}
"""
    response = call_model(
        client,
        model,
        [
            {"role": "system", "content": "你是严格但保守的技术事实核验员，只输出合法JSON。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1600,
        temperature=0.0,
        enable_thinking=False,
    )
    try:
        result = extract_json_object(response)
        return {
            "unsupported_claims": list(result.get("unsupported_claims") or [])[:12],
            "wrong_citations": list(result.get("wrong_citations") or [])[:12],
            "notes": str(result.get("notes", ""))[:500],
        }
    except Exception as error:  # noqa: BLE001
        return {"unsupported_claims": [], "wrong_citations": [], "audit_error": str(error)}


def align_section_citations(
    client: OpenAI,
    model: str,
    topic: str,
    section: OutlineNode,
    evidence: str,
    draft: str,
    enable_thinking: bool,
) -> tuple[str, dict]:
    """Rebind claims to the most specific supplied evidence without adding facts."""
    allowed_children = [child.title for child in section.children]
    heading_rule = (
        "保留且仅保留下列三级标题，不得改名：" + "、".join(allowed_children) + "。"
        if allowed_children
        else "不得添加三级标题。"
    )
    prompt = f"""请对《{topic}》中“{section.title}”章节做一次论断—证据对齐。

这不是扩写任务。保持章节结构、有效技术细节和总体篇幅，只修正事实边界与引用：
1. 将每个可核查技术句拆成边界清楚的单一或紧密相关论断，并在句末绑定直接支持它的证据编号。
2. 引用必须与同一句中的实体、版本、数值、机制和应用场景完全对应；不得用“相关但未陈述该事实”的来源代替直接证据。
3. 一句话确需多条证据时才保留多个编号；否则使用最具体的一条。删除装饰性、错位或重复堆叠的引用。
4. 对无直接证据的强事实、确定性因果、竞品比较、年份预测和性能数字，改为证据可支持的准确表述；无法改写时删除。
5. 在不同来源确实分别支持不同论断时，分别使用这些来源，避免整章主要依赖同一个编号；不能为了增加编号而牺牲相关性。
6. 机制解释只能由相邻证据直接推出，并应明确写成基于证据的工程解释或适用边界，不得新增证据外事实。
7. {heading_rule}

【章节证据】
{evidence}

【待对齐章节】
{draft}

仅输出对齐后的完整 Markdown 章节，标题必须为“## {section.title}”。
"""
    try:
        aligned = call_model(
            client,
            model,
            [
                {"role": "system", "content": "你是技术论断与来源对齐编辑。只保留证据可以直接支持的事实，并精确绑定引用。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=14000,
            temperature=0.0,
            enable_thinking=enable_thinking,
        ).strip()
    except Exception as error:  # noqa: BLE001
        return draft, {"applied": False, "error": f"{type(error).__name__}: {error}"}

    before_headings = re.findall(r"^#{2,3}\s+(.+)$", draft, flags=re.M)
    after_headings = re.findall(r"^#{2,3}\s+(.+)$", aligned, flags=re.M)
    length_ratio = len(aligned) / max(1, len(draft))
    if before_headings != after_headings or not (0.78 <= length_ratio <= 1.12):
        return draft, {
            "applied": False,
            "reason": "heading_or_length_drift",
            "length_ratio": round(length_ratio, 3),
            "headings_preserved": before_headings == after_headings,
        }
    return aligned, {
        "applied": True,
        "length_ratio": round(length_ratio, 3),
        "citation_distribution_before": citation_distribution(draft),
        "citation_distribution_after": citation_distribution(aligned),
    }


def generation_prompt(
    topic: str,
    section: OutlineNode,
    full_outline: str,
    evidence: str,
    previous_context: str,
    optimized: bool,
) -> str:
    if optimized:
        allowed_children = [child.title for child in section.children]
        heading_rule = (
            "允许使用的三级标题仅限：" + "、".join(allowed_children) + "。不得改名；只有当某条目确实没有任何证据支持时才省略该标题。"
            if allowed_children
            else "本节没有得到证据支持的三级标题，因此不得输出任何三级标题（###）。"
        )
        workflow = """写作步骤（只在内部执行）：
- 先逐条核对二级条目能否被证据支持；对得到支持的条目逐条展开，不要遗漏，并为其设置清晰的三级小节标题。
- 先建立“论断—证据”对应关系，再动笔。为每个论点挑选直接支持它的证据编号，做到句末引用与事实一一对应；每个事实句都要有引用，并避免在互不相关的论断中反复引用同一个编号。
- 引用节奏按“小节”控制：一个三级小节内，同一编号最多出现三次，同一段落最多两次；过渡句与小结句不引用。
- 写任何数字之前回到证据原文核对，确认数字与其适用对象完全一致；不一致就不要写。
- 每个案例、项目或产品实例只完整展开一次，其他位置用一句承接。
- 缩写首次出现写成“中文名称（英文全称，缩写）”；后文只用缩写。
- 每个三级小节写成一个有明确问题的单元：先给结论或现象，再给机制或流程，最后给条件、限制或工程影响；相邻小节之间用一句话承接。
- 章节净字数不少于 1500 字；证据充分时写到 2000–2800 字。设备/系统类章节应把体系结构、关键模块、交互流程、参数配置、部署条件、性能表现、方案对比与限制逐项写清，这些是技术白皮书的核心信息量来源。
- 组织内容时尽量覆盖：技术背景与问题、体系结构与组成、关键机制与流程、实现与部署要点、性能与效果、方案对比与取舍、限制与发展趋势；按证据实际情况灵活裁剪。
- 对关键技术点采用“证据事实—工作机制—工程影响或适用边界”的展开顺序。优先写清为什么这样设计、模块如何协同、在什么条件下产生什么效果；推论必须能够由相邻证据直接推出。
- 若证据包含不同版本、方案或厂商做法，写出对比与取舍；若证据包含挑战或演进方向，写出工程影响与趋势判断。
- 完成后检查每个事实句的引用编号是否来自本章节证据，并且确实支持该句。
- 当章节证据包含多个独立来源时，应让不同来源分别承担其直接支持的论断；不要让某一个引用编号覆盖整章的大部分技术事实，也不要为了增加引用数量而加入弱相关编号。
- 结构上：每个一级章节在证据允许时至少给出两个三级小节，按“功能定位—工作原理—实现流程—效果与取舍”的顺序推进，避免整章只有一个大段。
- 避免与其它章节重复：如果某项事实或技术细节已经在前面章节写过，本节只做简短承接，不再重复展开。
- 讲清模块之间的交互逻辑与依赖关系，并在证据允许的范围内说明实现限制、潜在风险与工程权衡。
"""
    else:
        workflow = "请根据章节大纲和参考材料撰写连贯、完整的章节。"
    return f"""请撰写《{topic}》中的“{section.title}”章节。

{workflow}

【当前章节大纲】
{section.render(2)}

【全文大纲，仅用于保持边界和避免重复】
{full_outline}

【前文衔接信息】
{previous_context or '这是文档的首个章节。'}

【章节证据】
{evidence}

【标题约束】
{heading_rule if optimized else '遵循当前章节大纲。'}

直接输出该章节的 Markdown 正文。章节标题必须为“## {section.title}”。
"""


def repair_prompt(
    topic: str,
    section: OutlineNode,
    evidence: str,
    draft: str,
    diagnostics: dict,
) -> str:
    allowed_children = [child.title for child in section.children]
    heading_rule = (
        "三级标题只能是：" + "、".join(allowed_children) + "。"
        if allowed_children
        else "不得保留或新增任何三级标题。"
    )
    return f"""请修订以下《{topic}》章节草稿，使其满足技术白皮书的事实与引用要求。

修订重点：
1. 对证据无法支持的表述，优先改写成有证据支撑的等价表述；确实无法改写时才删除该句。
2. 不要为了缩短而删除有证据支持的技术细节；相反，应补足证据里的机制、组成、流程、条件、效果、对比与限制，使内容更完整、更具体。
3. 每个包含可核查事实的句子都要给出有效引用；只能使用章节证据中的 [^数字]，且编号必须是直接支持该句的那条证据。
4. 删除无效引用、空泛宣传和重复表述，但保留技术信息量。
5. {heading_rule}
6. 章节证据中找不到的数值、标准、算法、应用行业和性能结论必须删除或改写；不得用相邻引用掩盖无依据的陈述。
7. 数值句的引用来源本身必须包含同一数值及其适用对象；不得引用另一段只包含相似技术名词的材料。
8. 在直接相关的前提下充分利用章节中的不同来源。若某一编号占据大多数引用，应把各论断重新绑定到更具体的来源；不能为了增加编号而添加弱相关引用。
9. 若章节偏短（净字数低于 1500），必须从证据中补充机制解释、实现步骤、模块协同、部署条件、性能数据、方案对比与限制，把章节扩写到 1500 字以上；不得用重复前文或空泛评述充数，只能补充证据支持的技术内容。
10. 若章节中同一事实或同一组参数出现两次以上，删除后出现的那次，用节省的篇幅补充尚未展开的证据细节。
11. 缩写或英文术语首次出现时必须给出“中文名称（英文全称，缩写）”；同一概念在全文使用同一叫法。

自动检查结果：{json.dumps(diagnostics, ensure_ascii=False)}

【原章节大纲】
{section.render(2)}

【章节证据】
{evidence}

【待修订草稿】
{draft}

仅输出修订后的 Markdown 章节，章节标题必须为“## {section.title}”。
"""


def full_outline_text(sections: Sequence[OutlineNode]) -> str:
    return "\n".join(section.render(2) for section in sections)


def global_refine_prompt(topic: str, article: str) -> str:
    return f"""下面是一篇《{topic}》技术白皮书的完整草稿。请作为责任编辑做一次全局润色，只做编辑，不新增任何事实或数据。

要求：
1. 删除跨章节重复出现的内容：同一事实或同一项技术细节只保留在最适合它的章节中，其余位置改为简短的承接句或直接删除。
2. 优化章节之间的过渡与衔接，使上下文自然连贯，避免生硬跳跃。
3. 保持全部二级、三级标题的文字和顺序不变，不新增或删除标题。
4. 对明显冗长、重复或口语化的句子进行压缩，提升可读性。
5. 首次出现缩写或生僻术语时，用原文已有信息给出简短解释；避免术语连续堆叠和过长句子。
6. 保持原有章节顺序、全部 [^数字] 引用编号和事实内容不变；不得新增、删除或改动任何引用编号。
7. 必须输出完整文档，不得省略结尾章节或截断句子。

只输出润色后的完整 Markdown 正文（以“# ”开头），不要输出任何说明文字。

草稿：
{article}
"""


def refine_article(
    client: OpenAI,
    model: str,
    topic: str,
    article: str,
    enable_thinking: bool,
) -> tuple[str, dict]:
    """Global editing pass: de-duplicate across sections and improve flow."""
    original_ids = citation_ids(article)
    original_headings = re.findall(r"^#{1,3}\s+(.+)$", article, flags=re.M)
    try:
        refined = call_model(
            client,
            model,
            [
                {"role": "system", "content": "你是一名资深技术白皮书责任编辑，只做结构、去重和行文层面的编辑，不改变事实与引用。"},
                {"role": "user", "content": global_refine_prompt(topic, article)},
            ],
            max_tokens=16000,
            temperature=0.1,
            enable_thinking=enable_thinking,
        )
    except Exception as error:  # noqa: BLE001
        return article, {"applied": False, "error": f"{type(error).__name__}: {error}"}

    refined = refined.strip()
    if not refined.startswith("#"):
        refined = f"# {topic}\n\n{refined}"
    new_ids = citation_ids(refined)
    original_set, new_set = set(original_ids), set(new_ids)
    changed = len(original_set.symmetric_difference(new_set))
    allowed = 0
    length_ratio = len(refined) / max(1, len(article))
    new_headings = re.findall(r"^#{1,3}\s+(.+)$", refined, flags=re.M)
    citation_count_ratio = len(new_ids) / max(1, len(original_ids))
    complete_ending = bool(re.search(r"[。！？.!?][\]\^\d\s]*$", refined.strip()))
    if (
        changed > allowed
        or original_headings != new_headings
        or not (0.82 <= length_ratio <= 1.08)
        or not (0.9 <= citation_count_ratio <= 1.08)
        or not complete_ending
    ):
        return article, {
            "applied": False,
            "reason": "citation_heading_length_or_truncation_drift",
            "changed_ids": changed,
            "length_ratio": round(length_ratio, 3),
            "citation_count_ratio": round(citation_count_ratio, 3),
            "headings_preserved": original_headings == new_headings,
            "complete_ending": complete_ending,
        }
    return refined, {
        "applied": True,
        "changed_ids": changed,
        "length_ratio": round(length_ratio, 3),
        "unique_citations_before": len(original_set),
        "unique_citations_after": len(new_set),
    }


def prior_context(article_parts: Sequence[str], max_chars: int = 1800) -> str:
    if not article_parts:
        return ""
    headings = []
    for part in article_parts:
        headings.extend(re.findall(r"^#{1,3}\s+(.+)$", part, flags=re.M))
    tail = article_parts[-1][-max_chars:]
    return "已完成章节：" + "、".join(headings[-16:]) + "\n前一章节结尾：\n" + tail


def write_references(
    path: Path,
    sources: Sequence[SourceDocument],
    used_snippets: dict[int, str] | None = None,
) -> None:
    """Write references whose content is the actually cited snippet when available.

    The paper's evaluation inspects each cited passage against the reference
    record, so a focused excerpt verifies far better than a whole crawled page.
    """
    used_snippets = used_snippets or {}
    payload = []
    for source in sources:
        snippet = (used_snippets.get(source.ref_id) or "").strip()
        content = snippet if len(snippet) >= 60 else source.body
        payload.append(
            {
                "id": source.ref_id,
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "content": content,
                "path": source.path,
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def locate_source_dir(source_root: Path, topic: str) -> Path:
    alias = SOURCE_DIR_ALIASES.get(topic, topic)
    candidates = sorted(source_root.glob(f"*_{alias}"))
    if not candidates:
        candidates = [path for path in source_root.iterdir() if path.is_dir() and alias in path.name]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one source directory for {topic}, found {candidates}")
    return candidates[0] / "sources"


def generate_topic(
    client: OpenAI,
    model: str,
    planner_model: str,
    topic: str,
    outline_root: Path,
    source_root: Path,
    output_root: Path,
    optimized: bool,
    enable_thinking: bool,
    overwrite: bool,
) -> dict:
    safe_name = safe_filename(topic)
    article_dir = output_root / "article"
    reference_dir = output_root / "references"
    diagnostic_dir = output_root / "diagnostics"
    for directory in (article_dir, reference_dir, diagnostic_dir):
        directory.mkdir(parents=True, exist_ok=True)
    article_path = article_dir / f"{safe_name}.md"
    if article_path.exists() and not overwrite:
        return {"topic": topic, "status": "skipped", "article": str(article_path)}

    outline_path = locate_outline_file(outline_root, topic)
    source_dir = locate_source_dir(source_root, topic)
    parsed_title, sections = parse_outline(outline_path, default_title=topic)
    sources = load_sources(source_dir)
    chunks = [chunk for source in sources for chunk in split_chunks(source)]
    index = BM25Index(chunks)
    outline_rendered = "\n".join(f"## {section.title}" for section in sections)
    article_parts: list[str] = [f"# {parsed_title}"]
    section_diagnostics = []
    used_snippets: dict[int, str] = {}
    all_selected_chunks: list[EvidenceChunk] = []

    for section_number, section in enumerate(sections, 1):
        selected = retrieve_section_evidence(index, topic, section)
        if not selected:
            section_diagnostics.append({"section": section.title, "status": "no_evidence"})
            continue
        all_selected_chunks.extend(selected)
        grounded_section = section
        evidence_plan = {
            "supported_titles": [child.title for child in section.children],
            "dropped_titles": [],
            "planner": "disabled_for_baseline",
        }
        if optimized:
            grounded_section, evidence_plan = plan_grounded_section(
                client, planner_model, parsed_title, section, selected
            )
        evidence = evidence_text(selected)
        prompt = generation_prompt(
            parsed_title,
            grounded_section,
            outline_rendered,
            evidence,
            prior_context(article_parts),
            optimized,
        )
        draft = call_model(
            client,
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT if optimized else BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=12000,
            temperature=0.15 if optimized else 0.3,
            enable_thinking=enable_thinking,
        )
        allowed_ids = valid_reference_ids(selected)
        citation_alignment_info = {"applied": False}
        if optimized:
            draft, citation_alignment_info = align_section_citations(
                client,
                planner_model,
                parsed_title,
                grounded_section,
                evidence,
                draft,
                enable_thinking=False,
            )
        grounding_audit = (
            audit_grounding(client, planner_model, parsed_title, grounded_section, evidence, draft)
            if optimized
            else {}
        )
        passed, diagnostics = quality_gate(
            draft,
            allowed_ids,
            section=grounded_section,
            evidence=evidence,
            chunks=selected,
            audit=grounding_audit,
        )
        repair_rounds = 0
        removed_after_repairs: list[str] = []
        repair_candidates: list[dict] = []
        best_score = section_quality_score(diagnostics)
        while optimized and not passed and repair_rounds < 2:
            candidate = call_model(
                client,
                model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt(parsed_title, grounded_section, evidence, draft, diagnostics)},
                ],
                max_tokens=12000,
                temperature=0.1,
                enable_thinking=enable_thinking,
            )
            repair_rounds += 1
            candidate_audit = audit_grounding(
                client, planner_model, parsed_title, grounded_section, evidence, candidate
            )
            candidate_passed, candidate_diagnostics = quality_gate(
                candidate,
                allowed_ids,
                section=grounded_section,
                evidence=evidence,
                chunks=selected,
                audit=candidate_audit,
            )
            candidate_score = section_quality_score(candidate_diagnostics)
            repair_candidates.append(
                {
                    "round": repair_rounds,
                    "score": round(candidate_score, 4),
                    "accepted": candidate_score > best_score,
                    "clean_length": candidate_diagnostics.get("clean_length"),
                    "sentence_citation_coverage": candidate_diagnostics.get("sentence_citation_coverage"),
                    "unique_citations": candidate_diagnostics.get("unique_citations"),
                    "hard_numeric_problems": len(candidate_diagnostics.get("unsupported_numeric_claims") or [])
                    + len(candidate_diagnostics.get("misaligned_numeric_citations") or []),
                }
            )
            if candidate_score > best_score:
                draft = candidate
                grounding_audit = candidate_audit
                diagnostics = candidate_diagnostics
                passed = candidate_passed
                best_score = candidate_score
        article_parts.append(draft.strip())
        for ref_id, snippet in best_used_snippets(draft, selected).items():
            used_snippets.setdefault(ref_id, snippet)
        section_diagnostics.append(
            {
                "section_number": section_number,
                "section": section.title,
                "selected_sources": sorted(allowed_ids),
                "selected_chunks": len(selected),
                "evidence_plan": evidence_plan,
                "citation_alignment": citation_alignment_info,
                "repaired": repair_rounds > 0,
                "repair_rounds": repair_rounds,
                "repair_candidates": repair_candidates,
                "selected_quality_score": round(best_score, 4),
                "removed_after_repairs": removed_after_repairs,
                "gate_passed": passed,
                **diagnostics,
            }
        )
        checkpoint = "\n\n".join(article_parts).strip() + "\n"
        article_path.write_text(checkpoint, encoding="utf-8")
        (diagnostic_dir / f"{safe_name}.json").write_text(
            json.dumps(section_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    article = "\n\n".join(article_parts).strip() + "\n"
    refine_info = {"applied": False}
    if optimized:
        article, refine_info = refine_article(
            client, model, parsed_title, article, enable_thinking
        )
        if not article.endswith("\n"):
            article += "\n"
        # Rebuild excerpts against the final article so every reference record
        # reflects the claims that survived section repair and global editing.
        used_snippets = best_used_snippets(article, all_selected_chunks)
    article_path.write_text(article, encoding="utf-8")
    write_references(reference_dir / f"{safe_name}.json", sources, used_snippets)
    summary = {
        "topic": topic,
        "status": "completed",
        "article": str(article_path),
        "references": str(reference_dir / f"{safe_name}.json"),
        "source_documents": len(sources),
        "chunks": len(chunks),
        "sections": len(sections),
        "characters": len(article),
        "citations": len(citation_ids(article)),
        "unique_citations": len(set(citation_ids(article))),
        "citation_coverage": round(paragraph_citation_coverage(article), 4),
        "sentence_citation_coverage": round(sentence_citation_coverage(article), 4),
        "global_refine": refine_info,
        "section_diagnostics": section_diagnostics,
    }
    (diagnostic_dir / f"{safe_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outline-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None,
                        help="defaults to <repo root>/.env")
    parser.add_argument("--model", default=None, help="overrides the resolved default model")
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--base-url", default=None, help="overrides OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="overrides OPENAI_API_KEY")
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    settings = resolve_settings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_thinking=not args.no_thinking,
        env_file=args.env_file,
    )
    client = build_client(settings)
    model = settings.model
    planner_model = args.planner_model or settings.model

    selected_topics = (
        list(args.topic)
        if args.topic
        else list(TOPICS[args.start_index : args.start_index + args.limit])
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_manifest = []
    started = time.time()
    for index, topic in enumerate(selected_topics, 1):
        print(f"[{index}/{len(selected_topics)}] {topic}", flush=True)
        try:
            result = generate_topic(
                client=client,
                model=model,
                planner_model=planner_model,
                topic=topic,
                outline_root=args.outline_root,
                source_root=args.source_root,
                output_root=args.output_root,
                optimized=args.variant == "optimized",
                enable_thinking=not args.no_thinking,
                overwrite=args.overwrite,
            )
        except Exception as error:  # noqa: BLE001
            result = {"topic": topic, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr, flush=True)
        run_manifest.append(result)
        (args.output_root / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    payload = {
        "model": model,
        "variant": args.variant,
        "thinking": not args.no_thinking,
        "elapsed_seconds": round(time.time() - started, 2),
        "topics": run_manifest,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item.get("status") != "failed" for item in run_manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
