#!/usr/bin/env python3
"""Two-stage editorial pass for clarity and document structure.

The pass first audits a completed article and then applies only presentation-level
changes.  It is deliberately forbidden from adding technical facts, numbers, claims,
sections, or citations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from llm_settings import build_client, resolve_settings
from run_postoutline_experiment import call_model, extract_json_object


def headings(text: str) -> list[str]:
    return re.findall(r"^#{1,3}\s+.+$", text, flags=re.M)


def citation_ids(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[\^(\d+)\]", text)]


def audit_prompt(topic: str, article: str) -> str:
    return f"""请以技术白皮书责任编辑身份审阅《{topic}》，只诊断表达与组织问题，不评价事实正确性，也不建议增加文章中不存在的技术内容。

重点检查：
1. 同一事实、案例或技术功能是否在不同章节重复展开；
2. 缩写或专业术语首次出现时是否缺少简短说明，后续叫法是否一致；
3. 是否存在过长句、术语连续堆叠、一个段落承担多个不同主题的情况；
4. 章节和小节之间是否缺少承接，是否出现结论先于依据或顺序跳跃；
5. 小节是否围绕一个明确问题展开，结论与展望是否重复前文细节；
6. 是否存在残句、截断、空泛宣传语或过程性写作说明。

只输出合法 JSON：
{{"redundancy":["具体位置与问题"],"undefined_terms":["术语及首次位置"],"long_or_dense_sentences":["原句"],"transition_issues":["具体位置与问题"],"structure_issues":["具体位置与问题"],"truncation_or_noise":["具体位置与问题"]}}

文章：
{article}
"""


READABILITY_RULES = """允许的修改：
- 拆分超长句：一句话包含三个以上并列信息时，拆成两到三个短句，保持原意不变；
- 理顺段落：一个段落只讲一个技术问题，把混在一起的内容分成相邻段落；
- 补足上下文：缩写或专业术语首次出现时，用文章已有文字补一句简短说明（例如它属于哪一层、解决什么问题），不得引入外部知识；
- 统一叫法：同一对象在全文使用同一名称，避免“该模块/该组件/该单元”混用造成歧义；
- 补衔接句：在相邻小节之间添加简短承接，说明为什么接着说下一件事；
- 精简冗余：删除同一段落内重复表述，但不得删除技术信息。
"""

DEDUP_RULES = """允许的修改：
- 删除跨章节重复的解释，只在最合适的位置保留完整表述，不要仅仅换一种说法重复同一事实；
- 将过长句拆成逻辑清楚的短句，把主题不同的内容分成相邻段落；
- 在不引入外部知识的前提下，用文章已有文字说明缩写或术语，并统一全文叫法；
- 使用简短、非事实性的承接句明确“问题—设计—机制—应用—验证—结论”的推进关系；
- 按章节职责重新压缩重复内容：概述只保留定位、范围和核心价值的摘要；背景只保留问题、需求和演进动因；总体设计只描述架构与模块关系；方法原理集中解释机制；应用场景集中呈现案例、使用条件和效果；评测与实验集中呈现设置与结果；安全与合规集中处理风险和规范；结论只保留综合结论与有依据的展望；
- 删除残句、空泛宣传和写作过程说明。
"""


def rewrite_prompt(topic: str, article: str, audit: dict, focus: str = "dedup") -> str:
    rules = READABILITY_RULES if focus == "readability" else DEDUP_RULES
    return f"""请根据编辑审查结果，完成《{topic}》的最终责任编辑修订。

{rules}

硬性约束：
1. 不得新增任何技术事实、数字、标准、产品能力、案例、比较结论或未来预测。
2. 所有一级、二级、三级标题的文字与顺序必须原样保留；不得新增、删除或改名。
3. 不得修改引用编号，不得把引用移到不能直接支持的句子；删除重复句时可以连同该句引用删除。
4. 只做小幅精修：修订后篇幅保持在原文的 90%–103%，不得大幅删减。去重时删掉“重复的解释句”，保留每个技术点的完整表述一次即可。
5. 长句拆分、术语解释和衔接句都只能用原文已有信息，不得引入外部知识，也不得新增事实性内容。
5. 每段只围绕一个中心展开，避免整段术语罗列；首次出现的关键缩写尽量使用“中文名称（英文全称，缩写）”或文章已有的等价说明。
6. 只输出完整 Markdown 正文，从原来的一级标题开始，不要输出审查说明。

编辑审查：
{json.dumps(audit, ensure_ascii=False, indent=2)}

原文：
{article}
"""


def polish_article(
    client: OpenAI,
    model: str,
    topic: str,
    article: str,
    focus: str = "dedup",
) -> tuple[str, dict]:
    raw_audit = call_model(
        client,
        model,
        [
            {"role": "system", "content": "你是严谨的技术白皮书责任编辑，只诊断可读性和组织问题。"},
            {"role": "user", "content": audit_prompt(topic, article)},
        ],
        max_tokens=3500,
        temperature=0.0,
        enable_thinking=False,
    )
    try:
        audit = extract_json_object(raw_audit)
    except Exception as error:  # noqa: BLE001
        audit = {"audit_parse_error": str(error), "raw": raw_audit[:2000]}

    polished = call_model(
        client,
        model,
        [
            {
                "role": "system",
                "content": "你是技术白皮书最终责任编辑。只改善表达、术语一致性、去重和衔接，绝不添加技术事实。",
            },
            {"role": "user", "content": rewrite_prompt(topic, article, audit, focus)},
        ],
        max_tokens=16000,
        temperature=0.05,
        enable_thinking=False,
    ).strip()
    if not polished.startswith("#"):
        polished = f"# {topic}\n\n{polished}"

    original_headings = headings(article)
    new_headings = headings(polished)
    old_citations = citation_ids(article)
    new_citations = citation_ids(polished)
    new_ids = set(new_citations)
    old_ids = set(old_citations)
    length_ratio = len(polished) / max(1, len(article))
    citation_ratio = len(new_citations) / max(1, len(old_citations))
    complete = bool(re.search(r"[。！？.!?]\s*$", polished))
    valid = (
        original_headings == new_headings
        and new_ids.issubset(old_ids)
        and 0.88 <= length_ratio <= 1.05
        and citation_ratio >= 0.85
        and complete
    )
    diagnostics = {
        "accepted": valid,
        "audit": audit,
        "headings_preserved": original_headings == new_headings,
        "new_citation_ids": sorted(new_ids - old_ids),
        "length_ratio": round(length_ratio, 4),
        "citation_count_ratio": round(citation_ratio, 4),
        "complete_ending": complete,
    }
    return (polished if valid else article), diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-article", type=Path, required=True)
    parser.add_argument("--output-article", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None,
                        help="defaults to <repo root>/.env")
    parser.add_argument("--model", default=None, help="overrides the resolved default model")
    parser.add_argument("--base-url", default=None, help="overrides OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="overrides OPENAI_API_KEY")
    parser.add_argument("--focus", choices=("dedup", "readability"), default="dedup")
    args = parser.parse_args()

    settings = resolve_settings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        env_file=args.env_file,
    )
    client = build_client(settings)
    article = args.input_article.read_text(encoding="utf-8", errors="replace")
    topic = args.input_article.stem
    polished, diagnostics = polish_article(
        client, settings.model, topic, article, focus=args.focus
    )
    args.output_article.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostic.parent.mkdir(parents=True, exist_ok=True)
    args.output_article.write_text(polished.strip() + "\n", encoding="utf-8")
    args.diagnostic.write_text(
        json.dumps({"topic": topic, **diagnostics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"topic": topic, **diagnostics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
