#!/usr/bin/env python3
import re
import os
import json
import argparse
from collections import defaultdict
from openai import OpenAI
import twpgen_config as project_args

# ──────────────────────────────────────────────
# 客户端配置
# ──────────────────────────────────────────────
DEFAULT_LLM_MODEL = os.environ.get("TWPGEN_LLM_MODEL", "deepseek-v3")

def build_llm_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and configure your local API key.")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

client = build_llm_client()

# ──────────────────────────────────────────────
# 固定一级目录
# ──────────────────────────────────────────────
WHITEPAPER_CHAPTERS = [
    {
        "id": 1,
        "title": "概述",
        "keywords": "摘要、引言、前言、简介、总览",
        "keyword_list": ["摘要", "引言", "前言", "简介", "总览"],
        "description": "对技术做概述介绍",
        "max_subsections": None,
        "force_generate": True,
        "classifiable": False,
    },
    {
        "id": 2,
        "title": "背景",
        "keywords": "背景、现状、挑战、痛点、需求、动机、问题定义",
        "keyword_list": ["背景", "现状", "挑战", "痛点", "需求", "动机", "问题定义"],
        "description": "解释'为什么写这篇'，描述问题产生的土壤和驱动力。",
        "max_subsections": 2,
        "force_generate": True,
        "classifiable": False,
    },
    {
        "id": 3,
        "title": "总体设计",
        "keywords": "方案、总体方案、方案概述、架构、总体架构、模块组成、系统结构、整体流程",
        "keyword_list": ["总体方案", "方案概述", "架构", "总体架构", "模块组成", "系统结构", "整体流程"],
        "description": "某项技术的总体设计，技术的总体框架，只要说的都是技术上的一些总体的描述即可，虽然可能提到了很多的技术",
        "max_subsections": None,
        "force_generate": False,
        "classifiable": True,
    },
    {
        "id": 4,
        "title": "方法原理",
        "keywords": "原理、方法、算法、模型、机制、协议、关键技术、训练、推理、公式推导",
        "keyword_list": ["原理", "方法", "算法", "模型", "机制", "协议", "关键技术", "训练", "推理", "公式推导"],
        "description": "深入技术层面说明核心方法如何工作，只要说的都是技术上的一些核心方法即可，虽然可能提到了很多的技术",
        "max_subsections": None,
        "force_generate": False,
        "classifiable": True,
    },
    {
        "id": 5,
        "title": "应用场景",
        "keywords": "应用、场景、案例、业务、任务、落地",
        "keyword_list": ["应用", "场景", "案例", "业务", "任务", "落地"],
        "description": "主要描述技术用在什么业务、任务、场景、对象或案例中，强调'在哪里用、谁来用、怎么用'。不应主要讲算法原理或工程实现。",
        "max_subsections": None,
        "force_generate": False,
        "classifiable": True,
    },
    {
        "id": 6,
        "title": "技术实现",
        "keywords": "实现、接口、模块、集成、部署、配置、监控、运维、日志、优化、SDK、API",
        "keyword_list": ["实现", "接口", "模块", "集成", "部署", "配置", "监控", "运维", "日志", "优化", "SDK", "API"],
        "description": "主要描述如何工程落地，包括接口、模块实现、集成、部署、配置、监控、运维、日志、性能优化。重点是'系统怎么做出来、怎么接入、怎么运行，只要说的都是技术上的一些具体实现即可，虽然可能提到了很多的技术'。",
        "max_subsections": None,
        "force_generate": False,
        "classifiable": True,
    },
    {
        "id": 7,
        "title": "评测与实验",
        "keywords": "实验、测试、benchmark、指标、对比、结果、分析",
        "keyword_list": ["实验", "测试", "benchmark", "指标", "对比", "结果", "分析"],
        "description": "描述测试、实验、benchmark、指标、对比、实验设置、结果分析、案例验证等。只有当簇主要内容明确在讲测试、实验时才归入。",
        "max_subsections": None,
        "force_generate": False,
        "classifiable": True,
    },
    {
        "id": 8,
        "title": "安全与合规",
        "keywords": "安全、隐私、合规、风险、威胁模型、攻击与防护、权限控制、加密、审计、治理",
        "keyword_list": ["安全", "隐私", "合规", "风险", "威胁模型", "攻击与防护", "权限控制", "加密", "审计", "治理"],
        "description": "说明'风险怎么控'，从安全与合法合规视角展开。",
        "max_subsections": None,
        "force_generate": False,
        "classifiable": True,
    },
    {
        "id": 9,
        "title": "结论与展望",
        "keywords": "总结、结论、讨论、局限、展望、未来工作",
        "keyword_list": ["总结", "结论", "讨论", "局限", "展望", "未来工作"],
        "description": "文章收尾章节，总结已有成果并指向未来。",
        "max_subsections": None,
        "force_generate": True,
        "classifiable": False,
    },
]

CHAPTER_BY_ID = {ch["id"]: ch for ch in WHITEPAPER_CHAPTERS}
CLASSIFIABLE_CHAPTERS = [ch for ch in WHITEPAPER_CHAPTERS if ch["classifiable"]]
CLASSIFIABLE_IDS = {ch["id"] for ch in CLASSIFIABLE_CHAPTERS}

# ──────────────────────────────────────────────
# 读取数据
# ──────────────────────────────────────────────
def load_matching_results(filepath):
    """读取 output_by_matching.txt，清洗句子标签，返回 {topic_name: [句子, ...]}"""
    bracket_prefix_re = re.compile(r'^(\s*\[[^\]]*\])+\s*')
    topics = {}
    current_topic = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if re.match(r'^Topic\s+\d+', line):
                current_topic = line.rstrip(':').strip()
                topics[current_topic] = []
            elif line.strip().startswith('[sent_id=') and current_topic:
                sentence = bracket_prefix_re.sub('', line).strip()
                if sentence:
                    topics[current_topic].append(sentence)

    return topics

# ──────────────────────────────────────────────
# 工具函数：提取 JSON
# ──────────────────────────────────────────────
def extract_json_from_text(raw_text):
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    m = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if m:
        return json.loads(m.group())

    raise ValueError("未找到有效 JSON")

# ──────────────────────────────────────────────
# 提示词公共约束
# ──────────────────────────────────────────────
def build_title_grounding_rules(doc_title):
    return f"""【文章题目】
{doc_title}

【题目使用规则（必须严格遵守）】
1. 所有总结、分类和大纲生成都必须围绕这个题目展开，理解"文章主要围绕什么对象、问题或方向写"。
2. 题目只用于确定写作重心、章节命名语气和内容组织方向。
3. 具体内容只能来自用户提供的句子或摘要，不能因为题目而补充任何未出现的新事实、新方法、新实验、新场景。
4. 如果题目很大，但材料里只覆盖其中一部分，只能写材料真实覆盖到的部分。
5. 禁止捏造；禁止根据常识脑补；禁止扩写没有证据支撑的内容。

【重要：最终大纲格式要求】
6. 最终输出的大纲最后必须是"附录"章节，附录不需要展开二级小节，只输出一级标题。
7. 所有章节序号必须连续排列，最终格式为：1. X、2. X、3. X、...、N. 附录。"""

# ──────────────────────────────────────────────
# 第1步A：仅做摘要
# ──────────────────────────────────────────────
def summarize_topic(topic_name, sentences, model, doc_title):
    sentences_text = "\n".join(f"- {s}" for s in sentences)
    title_rules = build_title_grounding_rules(doc_title)

    prompt = f"""你现在只需要做一件事：对一个主题簇进行摘要。

{title_rules}

【主题簇句子】（{topic_name}）
{sentences_text}

要求：
1. 只根据上面的句子生成摘要，不得引入任何句子之外的新信息。
2. 摘要必须准确概括这个簇主要在讲什么。
3. 输出 2~3 句核心摘要，尽量紧凑。
4. 不要做章节分类，不要解释，不要输出多余内容。

严格按以下 JSON 格式输出：
{{
  "summary": "摘要内容（2~3句）"
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严格的文本摘要工具，只输出符合格式要求的 JSON，不输出任何其他内容。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        data = extract_json_from_text(raw)
        summary = str(data.get("summary", "")).strip()
        return summary
    except Exception as e:
        print(f"  [摘要失败] {topic_name}: {e}")
        return ""

# ──────────────────────────────────────────────
# 第1步B：仅根据原始簇句子做分类
# ──────────────────────────────────────────────
def classify_topic_by_cluster(topic_name, sentences, model, doc_title):
    chapters_desc = "\n".join(
        f"  {ch['id']}. {ch['title']}：{ch['description']}"
        for ch in CLASSIFIABLE_CHAPTERS
    )

    sentences_text = "\n".join(f"- {s}" for s in sentences)
    title_rules = build_title_grounding_rules(doc_title)

    prompt = f"""你现在只需要做一件事：根据一个主题簇的原始句子内容，判断它是否应该归入技术白皮书的某一章节。

{title_rules}

【重要原则】
1. 你必须直接依据"主题簇原始句子"进行判断，不能依据摘要进行判断。
2. 只能在第 3~8 章之间选择；概述(1)、背景(2)、结论与展望(9)不参与这里的分类。
3. 每个 Topic 最多只能归入一个章节。
4. 只有当这个簇里的"主要内容"明显都在讲某一章节对应的内容时，才允许归类。主要内容即可，也就是簇里的主要信息60%以上符合某一章既可以归类。
5. 如果这个簇内容混杂、重心不明确、或者不能稳定归入某一章，chapter_id 必须填 null。
6. 不允许为了强行归类而勉强匹配；宁可不分，也不要错分。
7. reason 必须明确说明：这个簇主要在讲什么，为什么属于该章；若不归类，也要说明为什么不够集中或不够匹配。

【章节判定说明（仅依据description描述）】
请严格根据各章节的 description 描述来判断类别，不要参考 keywords：
{chapters_desc}

【主题簇原始句子】（{topic_name}）
{sentences_text}

请严格判断：
- 先看这个簇的主要内容是不是集中讲同一类东西；
- 再判断是否与某一章的 description 定义高度一致；
- 如果不是高度一致，就不要分类。

严格按以下 JSON 格式输出，不要任何其他文字：
{{
  "chapter_id": 整数或null,
  "reason": "一句话说明分类依据；若不分类，要说明为什么这个簇不足以稳定归入某章"
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严格的文本分类工具。你只能依据给定的主题簇原始句子进行分类，禁止依据摘要推断，禁止强行归类。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        raw = resp.choices[0].message.content.strip()
        data = extract_json_from_text(raw)

        cid = data.get("chapter_id")
        reason = str(data.get("reason", "")).strip()

        if not (isinstance(cid, int) and cid in CLASSIFIABLE_IDS):
            cid = None

        return cid, reason

    except Exception as e:
        print(f"  [分类失败] {topic_name}: {e}")
        return None, ""

# ──────────────────────────────────────────────
# 第1步：对所有 Topic 先摘要，再按原始簇分类
# ──────────────────────────────────────────────
def classify_all_topics(topics, model, doc_title):
    chapter_to_summaries = defaultdict(list)
    all_summaries = []
    classification_details = []

    total = len(topics)
    for idx, (topic_name, sentences) in enumerate(topics.items()):
        if not sentences:
            continue

        print(f"  [{idx+1}/{total}] {topic_name} ({len(sentences)} 句)...")

        summary = summarize_topic(topic_name, sentences, model, doc_title)
        if summary:
            all_summaries.append((topic_name, summary))
            print(f"      摘要完成")

        cid, reason = classify_topic_by_cluster(
            topic_name, sentences, model, doc_title
        )

        if cid is not None:
            chapter_title = CHAPTER_BY_ID[cid]["title"]
            print(f"      → 分类到 {cid}. {chapter_title} | 原因: {reason}")
            chapter_to_summaries[cid].append((topic_name, summary))
            classification_details.append({
                "topic_name": topic_name,
                "summary": summary,
                "chapter_id": cid,
                "chapter_title": chapter_title,
                "reason": reason,
            })
        else:
            print(f"      → 未匹配 3~8 章节 | 原因: {reason or '无'}")
            classification_details.append({
                "topic_name": topic_name,
                "summary": summary,
                "chapter_id": None,
                "chapter_title": None,
                "matched_keyword": None,
                "reason": reason,
            })

    return chapter_to_summaries, all_summaries, classification_details

# ──────────────────────────────────────────────
# 强制生成章节：概述 / 背景 / 结论与展望
# ──────────────────────────────────────────────
def generate_overview(all_summaries, model, doc_title):
    summaries_text = "\n".join(f"【{name}】{summary}" for name, summary in all_summaries)
    title_rules = build_title_grounding_rules(doc_title)

    prompt = f"""下方是一篇技术文章各主题摘要的集合。
请根据这些摘要，生成技术白皮书「1. 概述」章节的二级小节大纲。

{title_rules}

【概述章节的定位】
概述是文章的"名片"，向读者回答"这篇文章写了什么"。
职责是：
1. 点明文章的核心对象/主题是什么；
2. 概括文章涵盖的主要内容范围（有哪些方面）；
3. 引导读者了解文章的整体结构。

【概述与背景的本质区别】
- 概述：回答"写了什么"（what）
- 背景：回答"为什么写"（why）
- ★ 概述必须只讲"是什么、覆盖哪些方面"，禁止讲原因、问题、挑战、动机，以及应用场景。
- ★ 概述禁止与背景章节内容重叠

【各主题摘要】
{summaries_text}

生成规则：
1. 二级小节：2~3 个。
2. 每个二级小节：标题（≤15字）+ 1~2 句要点（≤40字/句）。
3. 要点只提炼摘要中的信息，不引入额外内容。
4. 内容只能包含：是什么、包含什么、涵盖哪些方面。
5. 禁止出现"由于...问题"、"面临...挑战"、"为了解决..."等因果/动机类表述。

严格按以下格式输出，不要任何说明：
1. 概述
   1.1 子节标题
       - 要点
   1.2 子节标题
       - 要点"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严格的大纲生成器，只使用给定摘要中的信息，不得引入任何外部知识。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[1. 概述 生成失败: {e}]"


def generate_background(all_summaries, model, doc_title):
    summaries_text = "\n".join(f"【{name}】{summary}" for name, summary in all_summaries)
    title_rules = build_title_grounding_rules(doc_title)

    prompt = f"""下方是一篇技术文章各主题摘要的集合。
请根据这些摘要，生成技术白皮书「2. 背景」章节的二级小节大纲。

{title_rules}

【背景章节的定位】
背景是"导火索"，向读者回答"为什么写这篇"，描述问题产生的土壤和驱动力。
职责是：
1. 说明题目相关领域的现状是什么；
2. 揭示现有问题、痛点或挑战；
3. 阐明推动研究/建设的需求与动机。

【背景与概述的本质区别】
- 概述：回答"写了什么"（what），讲是什么、覆盖哪些方面
- 背景：回答"为什么写"（why），讲问题、挑战、动机
- ★ 背景必须只讲"为什么"，禁止重复概述中的内容
- ★ 背景禁止描述文章本身包含什么、覆盖哪些方面

【各主题摘要】
{summaries_text}

生成规则：
1. 二级小节：最多 2 个。
2. 每个二级小节：标题（≤15字）+ 1~2 句要点（≤40字/句）。
3. 要点只提炼摘要中的信息，不引入额外内容。
4. 内容必须包含"现状问题"或"挑战动机"类表述。
5. 禁止出现"本文提出"、"本白皮书介绍"、"包括...技术方案"等描述文章本身的表述。

严格按以下格式输出，不要任何说明：
2. 背景
   2.1 子节标题
       - 要点
   2.2 子节标题
       - 要点"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严格的大纲生成器，只使用给定摘要中的信息，不得引入任何外部知识。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[2. 背景 生成失败: {e}]"


def generate_conclusion(all_summaries, model, doc_title):
    summaries_text = "\n".join(f"【{name}】{summary}" for name, summary in all_summaries)
    title_rules = build_title_grounding_rules(doc_title)

    prompt = f"""下方是一篇技术文章各主题摘要的集合。
请根据这些摘要，生成技术白皮书「9. 结论与展望」章节的二级小节大纲。

{title_rules}

【结论与展望章节的定位】
这是文章收尾章节，用于：
1. 总结全文已经覆盖的核心内容与主要结论；
2. 提炼与题目相关的工作价值、意义或效果；
3. 给出未来可继续推进的方向。

注意：
- 必须保留本章；
- 不能引入摘要中没有出现的新技术细节；
- 应以总结和展望为主，不重复展开中间章节细节；
- "展望"只能做方向性表述，不能杜撰未来方案细节。

【各主题摘要】
{summaries_text}

生成规则：
1. 二级小节：2~3 个。
2. 每个二级小节：标题（≤15字）+ 1~2 句要点（≤40字/句）。
3. 要点只提炼摘要中的信息，不引入额外内容。
4. 内容必须符合"总结 + 展望"的语义。

严格按以下格式输出，不要任何说明：
9. 结论与展望
   9.1 子节标题
       - 要点
   9.2 子节标题
       - 要点"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是严格的大纲生成器，只使用给定摘要中的信息，不得引入任何外部知识。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=450
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[9. 结论与展望 生成失败: {e}]"

# ──────────────────────────────────────────────
# 第2步：每个保留章节调用一次模型生成二级目录
# ──────────────────────────────────────────────
def generate_chapter_outline(chapter, summary_list, model, doc_title):
    cid = chapter['id']
    title = chapter['title']
    description = chapter['description']
    max_sub = chapter.get('max_subsections')

    content_text = "\n".join(f"【{name}】{summary}" for name, summary in summary_list)
    title_rules = build_title_grounding_rules(doc_title)

    limit_rule = (
        f"4. 二级小节数量上限：最多 {max_sub} 个，优先保留最重要的内容。"
        if max_sub else
        "4. 二级小节数量：按子主题自然拆分，有几个写几个，不强行凑数也不合并。"
    )

    prompt = f"""你是大纲生成器。
下方摘要已归入「{cid}. {title}」章节，请根据它们生成该章的二级小节大纲。

{title_rules}

【本章定位（必须严格遵守）】
{description}

【归入本章的摘要】
{content_text}

生成规则：
1. 所有二级小节的标题和要点，必须与本章定位一致。
2. 要结合题目理解这一章在整篇文章中的位置，使标题表述更贴合题目主线。
3. 具体内容只能来自归入本章的摘要，禁止引入任何未出现的信息。
4. 按摘要涉及的不同子主题拆分二级小节。
5. 每个二级小节：标题（≤15字）+ 1~2 句要点（≤40字/句）。
{limit_rule}

严格按以下格式输出，不要任何说明：
{cid}. {title}
   {cid}.1 子节标题
       - 要点
   {cid}.2 子节标题（若有更多子主题）
       - 要点

【最终大纲格式提醒】
请注意，最终完整大纲的章节顺序为：1. X、2. X、3. X、...、N. 附录（附录不展开二级小节）。"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"你是严格的大纲生成器。当前章节是「{title}」，定位为：{description}。输出必须符合章节定位、贴合文章题目，但所有事实只能来自给定摘要。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=700
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[{cid}. {title} 生成失败: {e}]"

# ──────────────────────────────────────────────
# 解析与重编号
# ──────────────────────────────────────────────
def parse_outline_fragment(fragment):
    lines = [line.rstrip() for line in fragment.strip().splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0].strip()
    m1 = re.match(r'^(\d+)\.\s*(.+)$', first)
    if not m1:
        return None

    chapter_num = int(m1.group(1))
    chapter_title = m1.group(2).strip()
    result = {
        "chapter_num": chapter_num,
        "chapter_title": chapter_title,
        "subsections": []
    }

    current_sub = None
    for line in lines[1:]:
        stripped = line.strip()

        m2 = re.match(r'^(\d+)\.(\d+)\s+(.+)$', stripped)
        if m2:
            current_sub = {
                "num": f"{m2.group(1)}.{m2.group(2)}",
                "title": m2.group(3).strip(),
                "bullets": []
            }
            result["subsections"].append(current_sub)
            continue

        mb = re.match(r'^-\s+(.+)$', stripped)
        if mb and current_sub is not None:
            current_sub["bullets"].append(mb.group(1).strip())
            continue

    return result


def merge_outline(chapter_outlines, doc_title=None):
    """合并章节大纲，附录追加在最后，不参与重编号"""
    parsed = []
    appendix_line = None

    for frag in chapter_outlines:
        frag_stripped = frag.strip()
        # 检测附录行（只有一级标题，没有二级小节）
        if re.match(r'^\d+\.\s*附录$', frag_stripped):
            appendix_line = frag_stripped
            continue
        item = parse_outline_fragment(frag)
        if item:
            parsed.append(item)

    final_lines = []

    if doc_title:
        final_lines.append(f"题目：{doc_title}")
        final_lines.append("")

    # 有内容章节重编号
    for new_idx, chapter in enumerate(parsed, start=1):
        chapter_title = chapter["chapter_title"]
        final_lines.append(f"{new_idx}. {chapter_title}")
        for sub_idx, sub in enumerate(chapter["subsections"], start=1):
            final_lines.append(f"   {new_idx}.{sub_idx} {sub['title']}")
            for bullet in sub["bullets"]:
                final_lines.append(f"      - {bullet}")

    # 附录追加在最后（不参与重编号，保持原编号）
    if appendix_line:
        final_lines.append(appendix_line)

    return "\n".join(final_lines)

# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def process_single_topic(topic_name, model=DEFAULT_LLM_MODEL):
    input_file = f"./dataset/{topic_name}/output_by_matching.txt"
    output_file = f"./dataset/{topic_name}/outline.txt"
    classification_file = f"./dataset/{topic_name}/classification_details.json"

    if not os.path.exists(input_file):
        print(f"  [跳过] {topic_name}: 找不到 {input_file}")
        return False

    print(f"\n{'=' * 60}")
    print(f"处理题目: {topic_name}")
    print(f"{'=' * 60}")

    doc_title = topic_name
    print(f"题目内容: {doc_title}\n")

    topics = load_matching_results(input_file)
    print(f"加载句子: 共 {len(topics)} 个 Topic\n")

    if not topics:
        print(f"  [跳过] {topic_name}: 没有找到任何句子")
        return False

    # 阶段1
    print("=== 阶段1：逐 Topic 摘要 + 分类 ===")
    chapter_to_summaries, all_summaries, classification_details = classify_all_topics(
        topics, model, doc_title
    )
    matched_ids = sorted(chapter_to_summaries.keys())
    print(f"\n有内容的分类章节 id：{matched_ids}\n")

    # 保存分类详情
    classification_output = {
        "doc_title": doc_title,
        "total_topics": len(classification_details),
        "classified_topics": len([d for d in classification_details if d["chapter_id"] is not None]),
        "unclassified_topics": len([d for d in classification_details if d["chapter_id"] is None]),
        "classification_details": classification_details
    }
    with open(classification_file, 'w', encoding='utf-8') as f:
        json.dump(classification_output, f, ensure_ascii=False, indent=2)
    print(f"分类详情已保存到: {classification_file}\n")

    # 阶段2
    print("=== 阶段2：逐章节生成二级目录 ===")
    chapter_outlines = []

    for chapter in WHITEPAPER_CHAPTERS:
        cid = chapter["id"]

        if cid == 1:
            print(f"  1. 概述 → 强制保留，汇总全部 {len(all_summaries)} 个摘要，生成中...")
            chapter_outlines.append(generate_overview(all_summaries, model, doc_title))
            continue

        if cid == 2:
            print(f"  2. 背景 → 强制保留，汇总全部 {len(all_summaries)} 个摘要，生成中...")
            chapter_outlines.append(generate_background(all_summaries, model, doc_title))
            continue

        if 3 <= cid <= 8:
            if cid not in chapter_to_summaries:
                print(f"  {cid}. {chapter['title']} → 无内容，跳过")
                continue

            summary_list = chapter_to_summaries[cid]
            max_sub = chapter.get('max_subsections')
            limit_note = f"（上限{max_sub}个二级节）" if max_sub else ""
            print(f"  {cid}. {chapter['title']} → {len(summary_list)} 个摘要{limit_note}，生成中...")
            chapter_outlines.append(generate_chapter_outline(chapter, summary_list, model, doc_title))
            continue

        if cid == 9:
            print(f"  9. 结论与展望 → 强制保留，汇总全部 {len(all_summaries)} 个摘要，生成中...")
            chapter_outlines.append(generate_conclusion(all_summaries, model, doc_title))
            continue

    # 阶段3：汇总整合
    print("\n=== 阶段3：汇总整合（连续重编号） ===")

    # 直接追加"附录"章节（不经过分类，也不单独调模型）
    appendix_block = f"{len(chapter_outlines) + 1}. 附录"
    chapter_outlines.append(appendix_block)
    print(f"  附录 → 直接追加（不参与分类，也不单独调模型）")

    final_outline = merge_outline(chapter_outlines, doc_title=doc_title)

    print("\n" + "=" * 60)
    print("生成的技术白皮书大纲")
    print("=" * 60)
    print(final_outline)

    # 保存大纲
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_outline)
    print(f"\n大纲已保存到: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description="生成技术白皮书大纲")
    parser.add_argument(
        "--input", type=str,
        default=f"./dataset/{project_args.dataset}/output_by_matching.txt",
        help="输入文件路径（output_by_matching.txt）"
    )
    parser.add_argument(
        "--title_file", type=str,
        default="/workspace/TWP-Gen/knowledge_collector/topic.txt",
        help="题目文件路径"
    )
    parser.add_argument(
        "--output", type=str,
        default=f"./dataset/{project_args.dataset}/outline.txt",
        help="输出大纲文件路径"
    )
    parser.add_argument("--model", type=str, default="deepseek-v3", help="使用的模型")
    parser.add_argument(
        "--all", action="store_true",
        help="处理 topic.txt 中的所有题目，依次生成大纲"
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="指定单个题目目录名称（如：算力运维体系技术白皮书）"
    )
    args = parser.parse_args()

    if args.all:
        print("=" * 60)
        print("批量模式：处理所有题目")
        print("=" * 60)

        with open(args.title_file, 'r', encoding='utf-8') as f:
            all_topics = [line.strip() for line in f if line.strip()]

        print(f"共找到 {len(all_topics)} 个题目\n")

        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

        success_count = 0
        fail_count = 0

        for idx, topic in enumerate(all_topics, 1):
            topic_dir = os.path.join(dataset_dir, topic)

            if not os.path.isdir(topic_dir):
                print(f"[{idx}/{len(all_topics)}] 跳过: {topic} (目录不存在)")
                fail_count += 1
                continue

            success = process_single_topic(topic, args.model)
            if success:
                success_count += 1
            else:
                fail_count += 1

        print("\n" + "=" * 60)
        print(f"批量处理完成！成功: {success_count}, 跳过: {fail_count}")
        print("=" * 60)

    elif args.topic:
        print("=" * 60)
        print(f"单题模式：{args.topic}")
        print("=" * 60)
        process_single_topic(args.topic, args.model)

    else:
        process_single_topic(project_args.dataset, args.model)


if __name__ == "__main__":
    main()
