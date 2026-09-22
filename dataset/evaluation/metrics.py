"""The ten evaluation metrics, their groups and the 0--5 rubrics.

This is the single definition of what the evaluation measures: the scoring scripts
import these constants, and the appendix of the paper reports the same definitions
and rubrics.
"""

from __future__ import annotations

MIN_SCORE = 0.0
MAX_SCORE = 5.0

# ---------------------------------------------------------------- metric groups
METRIC_GROUPS: dict[str, tuple[str, ...]] = {
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

METRIC_ORDER: tuple[str, ...] = (
    "relevance",
    "coverage",
    "depth",
    "novelty",
    "technical_specificity",
    "understandability",
    "structurality",
    "citation_sufficiency",
    "citation_validity",
    "factual_consistency",
)

# ------------------------------------------------------- human-readable labels
DISPLAY_NAMES: dict[str, str] = {
    "relevance": "Relevance",
    "coverage": "Breadth",
    "depth": "Depth",
    "novelty": "Novelty",
    "technical_specificity": "Technical Specificity",
    "understandability": "Understandability",
    "structurality": "Structurality",
    "citation_sufficiency": "Citation Sufficiency",
    "citation_validity": "Citation Validity",
    "factual_consistency": "Factual Consistency",
}

SHORT_NAMES: dict[str, str] = {
    "relevance": "Rel.",
    "coverage": "Breadth",
    "depth": "Depth",
    "novelty": "Novelty",
    "technical_specificity": "Tech.Spec.",
    "understandability": "Understand.",
    "structurality": "Struct.",
    "citation_sufficiency": "Cit.Suff.",
    "citation_validity": "Cit.Valid.",
    "factual_consistency": "Fact.Cons.",
}

# --------------------------------------------------------- metric definitions
METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "relevance": {
        "group": "Content Quality",
        "question": "Does the article remain consistently focused on the requested topic?",
        "definition": (
            "Whether the document remains consistently aligned with the input topic "
            "and avoids irrelevant or generic digressions."
        ),
    },
    "coverage": {
        "group": "Content Quality",
        "question": "Does the article provide broad coverage of the topic?",
        "definition": (
            "Whether it covers the major background, principles, components, "
            "applications, challenges, evaluation aspects, and implications expected "
            "for the topic."
        ),
    },
    "depth": {
        "group": "Content Quality",
        "question": (
            "How thoroughly does the article explore the topic and its related areas?"
        ),
        "definition": (
            "Whether it explains mechanisms, implementation details, module "
            "relationships, trade-offs, limitations, and technical implications beyond "
            "surface-level description."
        ),
    },
    "novelty": {
        "group": "Content Quality",
        "question": (
            "Does the article cover novel aspects that relate to the intent but are "
            "not directly derived from it?"
        ),
        "definition": (
            "Whether it provides useful additional perspectives, comparisons, "
            "challenges, trends, or future directions beyond directly restating the "
            "topic."
        ),
    },
    "technical_specificity": {
        "group": "White-Paper Adaptability",
        "definition": (
            "Whether core principles, system architecture, components, workflows, "
            "advantages, limitations, and deployment considerations are discussed with "
            "professional technical detail."
        ),
    },
    "understandability": {
        "group": "White-Paper Adaptability",
        "definition": (
            "Whether the document is clear, coherent, readable, and uses technical "
            "terminology accurately and consistently."
        ),
    },
    "structurality": {
        "group": "White-Paper Adaptability",
        "definition": (
            "Whether the section hierarchy, content order, section functions, "
            "transitions, and overall organization conform to technical-white-paper "
            "conventions."
        ),
    },
    "citation_sufficiency": {
        "group": "Evidence Credibility",
        "definition": (
            "Whether key technical concepts, mechanisms, data, conclusions, and "
            "application claims receive adequate citation support across the document."
        ),
    },
    "citation_validity": {
        "group": "Evidence Credibility",
        "definition": (
            "Whether citation identifiers can be matched to accessible references and "
            "whether each cited source directly supports its associated claim."
        ),
    },
    "factual_consistency": {
        "group": "Evidence Credibility",
        "definition": (
            "Whether technical concepts, entities, relationships, mechanisms, and "
            "numerical statements agree with the retrieved evidence without "
            "contradiction or unsupported additions."
        ),
    },
}

# ------------------------------------------------------------------- rubrics
CONTENT_QUALITY_RUBRIC: dict[int, str] = {
    0: (
        "The document is essentially unrelated to the topic, provides almost no "
        "meaningful coverage, is extremely superficial, and contains no meaningful "
        "additional insight."
    ),
    1: (
        "The document has only weak relevance, covers one or two narrow aspects, "
        "provides shallow descriptions with little explanation, and adds very little "
        "new perspective."
    ),
    2: (
        "The document is partially relevant and covers some major aspects, but "
        "substantial gaps remain. Technical explanations are generally undeveloped, "
        "and the additional ideas are limited, generic, or weakly connected."
    ),
    3: (
        "The document is mostly relevant and covers the main aspects at an acceptable "
        "level. It provides moderate depth on several key points and introduces some "
        "useful additional perspectives, although noticeable omissions remain."
    ),
    4: (
        "The document remains clearly focused on the topic, covers most important "
        "aspects, explains them with clear detail, and provides several relevant "
        "insights, examples, or connections."
    ),
    5: (
        "The document is consistently aligned with the topic, provides comprehensive "
        "and well-balanced coverage, offers deep and rigorous explanations, and "
        "introduces valuable, well-integrated perspectives beyond the obvious scope."
    ),
}

ADAPTABILITY_RUBRIC: dict[int, str] = {
    0: (
        "The document contains almost no concrete technical content, is largely "
        "unreadable or incoherent, and lacks an effective organizational structure."
    ),
    1: (
        "The document mentions only a few technical terms without explaining their "
        "principles or mechanisms. It is difficult to understand, and its section "
        "hierarchy, order, and functions are unclear."
    ),
    2: (
        "The document provides some technical detail, but the content is shallow, "
        "scattered, or generic. Parts are understandable, while the overall structure "
        "remains incomplete or poorly organized."
    ),
    3: (
        "The document explains several relevant technical principles or components at "
        "an acceptable level and has recognizable sections in a mostly reasonable "
        "order. However, some mechanisms, transitions, or structural relationships "
        "remain underdeveloped."
    ),
    4: (
        "The document presents clear technical principles, components, workflows, "
        "benefits, and some limitations. It is coherent and readable and follows a "
        "clear and logical white-paper structure with only minor weaknesses."
    ),
    5: (
        "The document systematically explains its technical principles, mechanisms, "
        "architecture, workflows, benefits, limitations, and implementation issues. It "
        "is highly clear and readable and has an excellent white-paper structure with "
        "coherent hierarchy and natural transitions."
    ),
}

EVIDENCE_RUBRIC: dict[int, str] = {
    0: (
        "The document contains no identifiable citations or evidence support. Its "
        "citations are invalid or cannot be matched to the reference materials, and it "
        "contains numerous contradictory or unverifiable technical claims."
    ),
    1: (
        "The document contains very few citations, leaving most key claims unsupported. "
        "Most citations are invalid or untraceable, and serious factual inconsistencies "
        "or unsupported technical conclusions are present."
    ),
    2: (
        "The document provides some citations, but their coverage is clearly "
        "insufficient. Some citations can be matched, although numbering, evidence "
        "alignment, or traceability problems remain, and several technical claims are "
        "exaggerated, mismatched, or unverifiable."
    ),
    3: (
        "Citations generally cover the main content, and most of them can be matched to "
        "relevant reference materials. The document is factually credible overall, "
        "although some important claims or details remain weakly supported or slightly "
        "inconsistent."
    ),
    4: (
        "Citations adequately cover most key technical claims and major sections. They "
        "are generally valid, traceable, and relevant, and most technical concepts, "
        "data, mechanisms, and application scenarios are supported by the reference "
        "materials."
    ),
    5: (
        "Citations comprehensively support the key concepts, mechanisms, data, "
        "conclusions, and application scenarios. They are clearly matched, complete, "
        "highly relevant, and traceable, while the document's technical facts are fully "
        "consistent with the reference materials."
    ),
}

GROUP_RUBRICS: dict[str, dict[int, str]] = {
    "Content Quality": CONTENT_QUALITY_RUBRIC,
    "White-Paper Adaptability": ADAPTABILITY_RUBRIC,
    "Evidence Credibility": EVIDENCE_RUBRIC,
}

# ------------------------------------------------------ human evaluation labels
HUMAN_ASPECTS: tuple[str, ...] = (
    "Structural completeness",
    "Technical clarity",
    "Content coherence and readability",
    "Evidence grounding",
)

HUMAN_LABELS: tuple[tuple[str, str], ...] = (
    (
        "Directly Acceptable",
        "The document can be used without substantive revision.",
    ),
    (
        "Acceptable with Minor Revisions",
        "The document is generally usable but requires localized corrections, "
        "supplementary explanations, or minor structural adjustments.",
    ),
    (
        "Unacceptable",
        "The document has major deficiencies in technical content, organization, "
        "readability, or evidence support and requires substantial rewriting.",
    ),
)


def format_rubric(rubric: dict[int, str]) -> str:
    """Render a 0--5 rubric as the text block sent to the evaluator."""
    return "\n".join(f"{score}: {text}" for score, text in sorted(rubric.items()))


def validate_score(value, context: str) -> float:
    """Check that a score is numeric and inside ``[MIN_SCORE, MAX_SCORE]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: score must be numeric, got {value!r}")
    score = float(value)
    if not MIN_SCORE <= score <= MAX_SCORE:
        raise ValueError(
            f"{context}: score {score} is outside [{MIN_SCORE}, {MAX_SCORE}]"
        )
    return score
