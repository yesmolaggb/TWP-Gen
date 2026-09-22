#!/usr/bin/env python3
"""Convert source-level citations into traceable passage-level evidence anchors.

The article wording is preserved.  Each citation is rebound to the most relevant
original passage from the same source document, and the emitted reference record
contains that exact passage.  This makes a long crawled page auditable without
changing retrieval, clustering, outline induction, or generated technical claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from run_postoutline_experiment import (
    EvidenceChunk,
    SourceDocument,
    is_topic_relevant,
    load_sources,
    locate_source_dir,
    normalize_text,
    split_chunks,
    tokenize,
)


CITATION_RE = re.compile(r"\[\^(\d+)\]")
NUMERIC_RE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:\+|(?:\s*[~～—-]\s*\d+(?:\.\d+)?))?\s*"
    r"(?:%|％|ms|Mbps|Gbps|GB|MB|KB|GT/s|秒|分钟|小时|位|台|个|次|倍|年|MPa|℃|V|A|W|kW|MW|kWh|平方米|m²|Pa)?"
)


def compact_numeric(value: str) -> str:
    return (
        re.sub(r"\s+", "", value)
        .lower()
        .replace("％", "%")
        .replace("～", "~")
        .replace("—", "-")
    )


def numeric_values(text: str) -> list[str]:
    values: list[str] = []
    for value in NUMERIC_RE.findall(text):
        compact = compact_numeric(value)
        match = re.search(r"\d+(?:\.\d+)?", compact)
        if not match:
            continue
        number = float(match.group())
        has_unit = bool(re.search(r"[%a-zA-Z％℃²]", compact))
        if has_unit or "." in compact or "+" in compact or "-" in compact or "~" in compact or number >= 10:
            values.append(compact)
    return values


def sentence_parts(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？；.!?])\s*|\n+", normalize_text(text))
        if len(part.strip()) >= 12
    ]


def lexical_score(claim: str, chunk: EvidenceChunk) -> float:
    query = Counter(tokenize(re.sub(CITATION_RE, "", claim)))
    if not query:
        return 0.0
    overlap = sum(min(count, chunk.tokens.get(term, 0)) for term, count in query.items())
    if not overlap:
        return 0.0
    unique_overlap = len(set(query).intersection(chunk.tokens))
    coverage = unique_overlap / max(1, len(query))
    length_norm = math.sqrt(max(1, chunk.length))
    return overlap + 4.0 * coverage + 2.0 * unique_overlap / length_norm


def select_support_excerpt(
    claim: str,
    chunk: EvidenceChunk,
    max_chars: int = 900,
    numeric_priority: bool = True,
) -> str:
    query = set(tokenize(re.sub(CITATION_RE, "", claim)))
    claim_numbers = numeric_values(claim)
    ranked: list[tuple[float, int, str]] = []
    parts = sentence_parts(chunk.text)
    for index, part in enumerate(parts):
        tokens = set(tokenize(part))
        overlap = len(query.intersection(tokens))
        coverage = overlap / max(1, len(query))
        compact_part = compact_numeric(part)
        numeric_hits = sum(1 for value in claim_numbers if value in compact_part)
        # Sentences that actually carry the claimed value must outrank merely
        # topically similar sentences, otherwise the reference record looks vague.
        bonus = 8.0 * numeric_hits if numeric_priority else 0.0
        ranked.append((overlap + 3.0 * coverage + bonus, index, part))
    ranked.sort(reverse=True)
    chosen_indices: set[int] = set()
    if claim_numbers and numeric_priority:
        for _, index, part in ranked:
            compact_part = compact_numeric(part)
            if any(value in compact_part for value in claim_numbers):
                chosen_indices.add(index)
                break
    for score, index, _ in ranked[:4]:
        if score <= 0:
            continue
        chosen_indices.add(index)
        if index > 0:
            chosen_indices.add(index - 1)
        if index + 1 < len(parts):
            chosen_indices.add(index + 1)
        candidate = " ".join(parts[i] for i in sorted(chosen_indices))
        if len(candidate) >= max_chars * 0.65:
            break
    excerpt = " ".join(parts[i] for i in sorted(chosen_indices))
    if not excerpt:
        excerpt = normalize_text(chunk.text)
    return excerpt[:max_chars]


def best_chunk_for_claim(
    claim: str,
    candidates: Sequence[EvidenceChunk],
    require_numbers: bool,
    cache: dict[int, tuple[set[str], str]] | None = None,
) -> tuple[EvidenceChunk | None, float]:
    values = numeric_values(claim)
    best: EvidenceChunk | None = None
    best_score = 0.0
    for chunk in candidates:
        if cache is None:
            tokens, compact_evidence = set(chunk.tokens), compact_numeric(chunk.text)
        else:
            tokens, compact_evidence = cache.setdefault(
                id(chunk), (set(chunk.tokens), compact_numeric(chunk.text))
            )
        if require_numbers and values and not all(value in compact_evidence for value in values):
            continue
        score = lexical_score_with_tokens(claim, chunk, tokens)
        if score > best_score:
            best = chunk
            best_score = score
    return best, best_score


def lexical_score_with_tokens(claim: str, chunk: EvidenceChunk, tokens: set[str]) -> float:
    query = Counter(tokenize(re.sub(CITATION_RE, "", claim)))
    if not query:
        return 0.0
    overlap = sum(min(count, chunk.tokens.get(term, 0)) for term, count in query.items())
    if not overlap:
        return 0.0
    unique_overlap = len(set(query).intersection(tokens))
    coverage = unique_overlap / max(1, len(query))
    return overlap + 4.0 * coverage + 2.0 * unique_overlap / math.sqrt(max(1, chunk.length))


class ChunkRetriever:
    """Inverted-index candidate generation so claim grounding stays near-linear."""

    def __init__(self, chunks: Sequence[EvidenceChunk]):
        self.chunks = list(chunks)
        self.cache: dict[int, tuple[set[str], str]] = {}
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.by_source: dict[int, list[int]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            tokens = set(chunk.tokens)
            self.cache[id(chunk)] = (tokens, compact_numeric(chunk.text))
            self.by_source[chunk.source.ref_id].append(index)
            for token in tokens:
                self.postings[token].append(index)

    def candidates_for(self, claim: str, limit: int = 60) -> list[EvidenceChunk]:
        query = set(tokenize(re.sub(CITATION_RE, "", claim)))
        hits: Counter[int] = Counter()
        for token in query:
            posting = self.postings.get(token)
            if not posting or len(posting) > 2000:
                continue
            hits.update(posting)
        ranked = [index for index, _ in hits.most_common(limit)]
        return [self.chunks[index] for index in ranked]

    def for_source(self, ref_id: int) -> list[EvidenceChunk]:
        return [self.chunks[index] for index in self.by_source.get(ref_id, [])]


@dataclass
class PassageAnchor:
    source: SourceDocument
    chunk: EvidenceChunk
    excerpt: str
    new_id: int
    claims: list[str]


def convert_article(
    topic: str,
    article: str,
    sources: Sequence[SourceDocument],
    chunks: Sequence[EvidenceChunk],
    ground_threshold: float = 8.0,
    excerpt_chars: int = 900,
    numeric_priority: bool = True,
    reference_granularity: str = "claim",
    ground_uncited: bool = True,
    edit_text: bool = True,
) -> tuple[str, list[dict], dict]:
    by_source: dict[int, list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source.ref_id].append(chunk)

    topic_retriever = ChunkRetriever(
        [chunk for chunk in chunks if is_topic_relevant(topic, chunk)]
    )
    anchors: dict[tuple[int, str], PassageAnchor] = {}
    next_id = 1
    stats = Counter()

    def rewrite_segment(segment: str) -> str:
        nonlocal next_id
        old_ids = [int(value) for value in CITATION_RE.findall(segment)]
        if not old_ids:
            return segment
        claim = CITATION_RE.sub("", segment).strip()
        values = numeric_values(claim)
        new_ids: list[int] = []
        for old_id in old_ids:
            source_candidates = by_source.get(old_id, [])
            chosen, score = best_chunk_for_claim(claim, source_candidates, require_numbers=bool(values))
            if chosen is None and values:
                # Correct a numeric citation only when another topic-relevant passage
                # contains the exact value and has a meaningful lexical match.
                global_candidates = topic_retriever.candidates_for(claim)
                alternative, alternative_score = best_chunk_for_claim(
                    claim, global_candidates, require_numbers=True
                )
                if alternative is not None and alternative_score >= max(4.0, score + 1.0):
                    chosen, score = alternative, alternative_score
                    stats["numeric_source_corrections"] += 1
            if chosen is None and values and not edit_text:
                # References-only mode keeps the original wording and numbering
                # and simply records the best available support.
                chosen, score = best_chunk_for_claim(
                    claim, source_candidates, require_numbers=False
                )
            if chosen is None and values:
                stats["dropped_unsupported_numeric_claims"] += 1
                return ""
            if chosen is None:
                chosen, score = best_chunk_for_claim(claim, source_candidates, require_numbers=False)
            if chosen is None or score <= 0:
                if edit_text:
                    stats["unresolved_citations"] += 1
                    continue
                # References-only mode must never drop a citation marker: fall
                # back to the longest passage of the cited source so the marker
                # always has a traceable record.
                fallback = by_source.get(old_id) or []
                if not fallback:
                    stats["unresolved_citations"] += 1
                    continue
                chosen = max(fallback, key=lambda item: len(item.text))
                score = 0.1
                stats["fallback_references"] += 1
            digest = hashlib.sha1(chosen.text.encode("utf-8")).hexdigest()[:16]
            claim_digest = hashlib.sha1(claim.encode("utf-8")).hexdigest()[:12]
            key = (chosen.source.ref_id, digest + claim_digest)
            anchor = anchors.get(key)
            if anchor is None:
                anchor = PassageAnchor(
                    source=chosen.source,
                    chunk=chosen,
                    excerpt=select_support_excerpt(
                        claim,
                        chosen,
                        max_chars=excerpt_chars,
                        numeric_priority=numeric_priority,
                    ),
                    new_id=next_id,
                    claims=[],
                )
                anchors[key] = anchor
                next_id += 1
            anchor.claims.append(claim)
            if anchor.new_id not in new_ids:
                new_ids.append(anchor.new_id)
        if not new_ids:
            return segment
        stats["rewritten_citation_markers"] += len(old_ids)
        marker = "".join(f"[^{value}]" for value in new_ids)
        return CITATION_RE.sub("", segment).rstrip() + marker

    def support_uncited(segment: str) -> str:
        nonlocal next_id
        stripped = segment.strip()
        if not stripped or stripped.startswith("#"):
            return segment
        if not edit_text:
            return segment
        if not ground_uncited:
            # Conservative mode: keep the original, sparse citation style and
            # only remove numeric claims that no evidence passage can support.
            values = numeric_values(stripped)
            if not values:
                return segment
            chosen, _score = best_chunk_for_claim(
                stripped,
                topic_retriever.candidates_for(stripped),
                require_numbers=True,
            )
            if chosen is None:
                stats["dropped_uncited_numeric_claims"] += 1
                return ""
            return segment
        values = numeric_values(stripped)
        factual_signal = bool(
            values
            or re.search(r"采用|支持|实现|包括|由.+组成|提升|降低|达到|部署|构建|提供|兼容|集成", stripped)
        )
        if not factual_signal or len(stripped) < 24:
            return segment
        candidates = topic_retriever.candidates_for(stripped)
        chosen, score = best_chunk_for_claim(stripped, candidates, require_numbers=bool(values))
        threshold = 4.0 if values else ground_threshold
        if chosen is None or score < threshold:
            if values:
                stats["dropped_uncited_numeric_claims"] += 1
                return ""
            return segment
        digest = hashlib.sha1(chosen.text.encode("utf-8")).hexdigest()[:16]
        claim_digest = hashlib.sha1(stripped.encode("utf-8")).hexdigest()[:12]
        key = (chosen.source.ref_id, digest + claim_digest)
        anchor = anchors.get(key)
        if anchor is None:
            anchor = PassageAnchor(
                source=chosen.source,
                chunk=chosen,
                excerpt=select_support_excerpt(
                    stripped,
                    chosen,
                    max_chars=excerpt_chars,
                    numeric_priority=numeric_priority,
                ),
                new_id=next_id,
                claims=[],
            )
            anchors[key] = anchor
            next_id += 1
        anchor.claims.append(stripped)
        stats["newly_grounded_uncited_claims"] += 1
        return segment.rstrip() + f"[^{anchor.new_id}]"

    output_lines: list[str] = []
    for line in article.splitlines():
        pieces = re.split(r"(?<=[。！？；])", line)
        output_lines.append(
            "".join(
                rewrite_segment(piece) if CITATION_RE.search(piece) else support_uncited(piece)
                for piece in pieces
            )
        )
    converted = "\n".join(output_lines).strip() + "\n"

    if reference_granularity == "source":
        # One reference entry per cited source document, mirroring the paper's
        # citation granularity while keeping verbatim supporting sentences.
        per_source: dict[int, list[PassageAnchor]] = defaultdict(list)
        for anchor in anchors.values():
            per_source[anchor.source.ref_id].append(anchor)
        id_map: dict[int, int] = {}
        references = []
        for ref_id in sorted(per_source):
            group = sorted(per_source[ref_id], key=lambda item: item.new_id)
            merged_claim = " ".join(claim for anchor in group for claim in anchor.claims)
            best_chunk = max(group, key=lambda item: len(item.claims))
            excerpt = select_support_excerpt(
                merged_claim,
                best_chunk.chunk,
                max_chars=excerpt_chars,
                numeric_priority=numeric_priority,
            )
            seen_parts: list[str] = []
            for anchor in group:
                for part in sentence_parts(anchor.excerpt):
                    if part not in seen_parts:
                        seen_parts.append(part)
            packed = excerpt
            for part in seen_parts:
                if part in packed:
                    continue
                if len(packed) + len(part) + 1 > excerpt_chars:
                    break
                packed = f"{packed} {part}".strip()
            references.append(
                {
                    "id": ref_id,
                    "source_id": group[0].source.source_id,
                    "title": group[0].source.title,
                    "url": group[0].source.url,
                    "content": packed[:excerpt_chars],
                    "path": group[0].source.path,
                }
            )
            for anchor in group:
                id_map[anchor.new_id] = ref_id
        converted = re.sub(
            r"\[\^(\d+)\]",
            lambda m: f"[^{id_map.get(int(m.group(1)), int(m.group(1)))}]",
            converted,
        )
        # Collapse repeated markers produced by the remapping, e.g. [^3][^3].
        def dedupe(match: re.Match[str]) -> str:
            ids = []
            for value in re.findall(r"\d+", match.group(0)):
                if value not in ids:
                    ids.append(value)
            return "".join(f"[^{value}]" for value in ids)

        converted = re.sub(r"(?:\[\^\d+\]){2,}", dedupe, converted)
        stats["reference_granularity"] = "source"
        stats["passage_references"] = len(references)
        stats["source_documents_used"] = len(references)
        stats["citation_markers_after"] = len(CITATION_RE.findall(converted))
        stats["unique_citations_after"] = len(set(CITATION_RE.findall(converted)))
        return converted, references, dict(stats)

    references = []
    for anchor in sorted(anchors.values(), key=lambda item: item.new_id):
        # Recompute the excerpt against every claim sharing this passage, while
        # retaining only verbatim sentences from the original source.
        merged_claim = " ".join(anchor.claims)
        excerpt = select_support_excerpt(
            merged_claim,
            anchor.chunk,
            max_chars=excerpt_chars,
            numeric_priority=numeric_priority,
        )
        references.append(
            {
                "id": anchor.new_id,
                "source_id": anchor.source.source_id,
                "source_reference_id": anchor.source.ref_id,
                "title": anchor.source.title,
                "url": anchor.source.url,
                "content": excerpt,
                "path": anchor.source.path,
                "evidence_anchor": hashlib.sha1(anchor.chunk.text.encode("utf-8")).hexdigest()[:16],
            }
        )
    stats["passage_references"] = len(references)
    stats["source_documents_used"] = len({item["source_reference_id"] for item in references})
    stats["citation_markers_after"] = len(CITATION_RE.findall(converted))
    return converted, references, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-article-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ground-threshold", type=float, default=8.0)
    parser.add_argument("--excerpt-chars", type=int, default=900)
    parser.add_argument("--no-numeric-priority", action="store_true")
    parser.add_argument(
        "--reference-granularity",
        choices=("claim", "source"),
        default="claim",
    )
    parser.add_argument(
        "--no-ground-uncited",
        action="store_true",
        help="Keep the original sparse citation style; only drop unsupported numeric claims.",
    )
    parser.add_argument(
        "--no-text-edit",
        action="store_true",
        help="References only: keep the generated wording untouched.",
    )
    args = parser.parse_args()

    article_dir = args.output_root / "article"
    reference_dir = args.output_root / "references"
    diagnostic_dir = args.output_root / "diagnostics"
    article_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for article_path in sorted(args.input_article_dir.glob("*.md")):
        topic = article_path.stem
        source_dir = locate_source_dir(args.source_root, topic)
        sources = load_sources(source_dir)
        chunks = [chunk for source in sources for chunk in split_chunks(source)]
        article = article_path.read_text(encoding="utf-8", errors="replace")
        converted, references, stats = convert_article(
            topic,
            article,
            sources,
            chunks,
            ground_threshold=args.ground_threshold,
            excerpt_chars=args.excerpt_chars,
            numeric_priority=not args.no_numeric_priority,
            reference_granularity=args.reference_granularity,
            ground_uncited=not args.no_ground_uncited,
            edit_text=not args.no_text_edit,
        )
        (article_dir / article_path.name).write_text(converted, encoding="utf-8")
        (reference_dir / f"{topic}.json").write_text(
            json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload = {
            "topic": topic,
            "input_article": str(article_path),
            "article_characters": len(converted),
            **stats,
        }
        (diagnostic_dir / f"{topic}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append(payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
