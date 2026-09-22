"""
自定义大纲报告生成器 - 完整版
OutlineArticleWriter - 支持读取本地大纲文件生成报告

与 run.py 的区别：
  - run.py: 用户输入主题 → 自动生成大纲 → 深度学习 → 生成报告
  - 本脚本: 用户指定大纲 → 深度学习（可跳过）→ 生成报告

用法:
    # 只指定大纲，使用自动深度搜索
    python src/run_from_outline.py --outline /path/to/outline.md

    # 指定大纲 + 预搜索结果（跳过章节级搜索）
    python src/run_from_outline.py --outline /path/to/outline.md --search /path/to/search.json

    # 跳过搜索，直接基于大纲生成（知识为空）
    python src/run_from_outline.py --outline /path/to/outline.md --no-search
"""

import argparse
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:  # central configuration: paths, keys, topics
    import twpgen_settings as repo_config
except Exception:  # pragma: no cover
    repo_config = None


def _default_article_dir() -> str:
    return repo_config.article_dir if repo_config else "output/article"


def _default_references_dir() -> str:
    return repo_config.references_dir if repo_config else "output/references"
import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

# 确保 src 模块可以被导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from src.llms.llm import llm
from src.agent.message import ReportState, Chapter
from src.agent.deepsearch import DeepSearch, DeepSearchResult
from src.agent.learning import get_all_search_results, get_real_reference_ids
from src.prompts import apply_prompt_template
from src.utils.parse_model_res import extract_xml_content
from src.utils.print_util import colored_print
from src.tools.md2html import markdown2html
from src.tools.search import SearchResult
from src.config.workflow_config import workflow_configs


# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# 大纲解析（复用 src/agent/outline.py 的逻辑）
# ============================================================================

_markdown_re = re.compile(r'(?s)```\s*markdown\n(.*)```')
_title_re = re.compile(r'^(#+)\s+(.*)')


def parse_outline(outline_str: str) -> Chapter:
    """解析大纲字符串为 Chapter 结构（与 src/agent/outline.py 一致）"""
    m = _markdown_re.search(outline_str)
    if m:
        outline_str = m.group(1)

    root = Chapter(id=0, level=0)
    current = root
    id_counter = 0
    doc_title = None
    skipped_first_level1 = False

    for line in outline_str.split('\n'):
        line = line.strip()
        if not line:
            continue
        tm = _title_re.match(line)
        if tm:
            id_counter += 1
            level = len(tm.group(1))
            title = tm.group(2)
            # 第一个 level-1 标题是大纲文件的第一行（如 "题目：" 或 "1. 概述"），不是真正的章节标题
            if level == 1 and not skipped_first_level1:
                doc_title = title
                skipped_first_level1 = True
                current = root
                continue
            while current and current.level >= level:
                current = current.parent_chapter
            if not current:
                current = root
            new_ch = Chapter(id=id_counter, level=level, title=title, parent_chapter=current)
            current.sub_chapter.append(new_ch)
            current = new_ch
        else:
            sm = extract_xml_content(line, 'summary')
            if sm:
                current.summary = sm[0]
            thm = extract_xml_content(line, 'thinking')
            if thm:
                current.thinking = thm[0]

    if not root.sub_chapter:
        raise ValueError('no valid chapter found')

    # 第一个 level-1 标题是大纲标题，其下的子章节才是真正的顶级章节
    if skipped_first_level1 and doc_title:
        # root.title = doc_title
        # root.sub_chapter 里的都是 level-2 节点（原本的 1.1、1.2 等）
        # 它们现在应该变成 level-1 的顶级章节
        root_chapter = Chapter(id=0, level=1, title=doc_title)
        for ch in root.sub_chapter:
            ch.level = 1
            ch.parent_chapter = root_chapter
            root_chapter.sub_chapter.append(ch)
        root_chapter.parent_chapter = None
        _clear_parents(root_chapter.sub_chapter)
    else:
        root_chapter = root.sub_chapter[0]
        root_chapter.parent_chapter = None
        _clear_parents(root_chapter.sub_chapter)

    return root_chapter


def _convert_numbered_to_markdown(text: str) -> str:
    """
    将编号大纲格式转换为 Markdown 标题格式。

    输入示例:
        1. 概述
           1.1 算力运维体系构成
              - 涵盖异构算力统一调度...
    输出示例:
        # 概述
        ## 1.1 算力运维体系构成
           - 涵盖异构算力统一调度...
    """
    lines = text.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append('')
            continue

        # 匹配顶级编号: "1. 标题" 或 "1、标题"
        top_match = re.match(r'^(\d+)[\.、]\s+(.+)$', stripped)
        if top_match:
            level = 1
            title = top_match.group(2)
            result.append(f"{'#' * level} {title}")
            continue

        # 匹配子级编号: "1.1 标题", "1.1.1 标题" 等
        sub_match = re.match(r'^(\d+(?:\.\d+)+)\s+(.+)$', stripped)
        if sub_match:
            nums = sub_match.group(1)
            title = sub_match.group(2)
            level = nums.count('.') + 1
            result.append(f"{'#' * level} {title}")
            continue

        # 普通内容行（bullet、纯文本等）直接保留缩进
        result.append(line)

    return '\n'.join(result)


def _clear_parents(chapters):
    for ch in chapters:
        ch.parent_chapter = None
        _clear_parents(ch.sub_chapter)


def load_outline_from_file(file_path: str) -> Chapter:
    """从文件加载大纲，支持编号格式和 Markdown 标题格式"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"大纲文件不存在: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 支持两种格式：
    # 1. Markdown 标题格式: # 标题, ## 标题
    # 2. 编号格式: 1. 标题,    1.1 标题,       - 内容
    #    转换为 Markdown 标题格式后解析
    # has_markdown_headers = bool(_markdown_re.search(content))  # 太宽泛
    has_markdown_headers = bool(re.search(r'^#{1,6}\s+\S', content, re.MULTILINE))
    has_numbered_outline = bool(re.search(r'^\d+[\.、]\s+\S', content, re.MULTILINE))

    if has_numbered_outline and not has_markdown_headers:
        content = _convert_numbered_to_markdown(content)
        colored_print(f"[INFO] 大纲格式: 编号格式 → 已转换为 Markdown 标题格式", color='cyan')

    colored_print(f"[INFO] 已加载大纲文件: {file_path}", color='cyan')
    return parse_outline(content)


def load_search_results_from_file(file_path: str) -> List[List[Dict]]:
    """从文件加载搜索结果（与 src/agent/outline.py 一致）"""
    path = Path(file_path)
    if not path.exists():
        colored_print(f"[WARNING] 搜索结果文件不存在: {file_path}，将跳过", color='yellow')
        return []

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                return data
            else:
                return [data]
        if isinstance(data, dict):
            if 'results' in data:
                return data['results']
            if 'search_results' in data:
                return data['search_results']
        return []
    except json.JSONDecodeError:
        colored_print(f"[WARNING] 搜索结果文件格式错误: {file_path}", color='yellow')
        return []


def outline_knowledge_2_str(outline_knowledge, max_length=100000) -> str:
    """将搜索结果转换为字符串（与 src/agent/outline.py 一致）"""
    if not outline_knowledge:
        return "[]"

    max_col = 0
    for knowledge in outline_knowledge:
        max_col = max(max_col, len(knowledge))
    result = []
    total_length = 0
    for i in range(max_col):
        for knowledge in outline_knowledge:
            if i < len(knowledge):
                content = knowledge[i].get('content', '')
                if total_length + len(content) > max_length:
                    break
                result.append({'content': content, 'id': knowledge[i].get('id', 0)})
                total_length += len(content)
    return json.dumps(result)


# ============================================================================
# Learning 流程（复用 src/agent/learning.py 的逻辑）
# ============================================================================

def learning_node(outline: Chapter, search_depth: int = 3, top_n: int = 5) -> tuple:
    """
    对每个章节进行深度搜索（与 src/agent/learning.py 的 learning_node 完全一致）
    
    Returns:
        tuple: (knowledge, outline) - 搜索结果和大纲
    """
    knowledge = []
    search_id = 1

    for chapter in outline.sub_chapter:
        colored_print(f"\n[DEEP SEARCH] 正在搜索章节: {chapter.title}", color='purple')

        # 注意：参数名必须与 src/agent/deepsearch.py 中的 DeepSearch.__init__ 一致
        ds = DeepSearch(
            title=outline.title,
            chapter=chapter.title,
            sub_chapter=[sub_ch.title for sub_ch in chapter.sub_chapter],
            chapter_outline=chapter.summary if chapter.summary else '',
            max_depth=search_depth,
            search_top_n=top_n
        )
        results = ds.deep_search()
        search_results = get_all_search_results(results)

        for key, value in search_results.items():
            knowledge += [
                {"id": search_id + i, "content": result.content, "url": result.url}
                for i, result in enumerate(value)
            ]
            search_id += len(value)

        chapter.learning_knowledge = [
            {"insight": re_knowledge.insight,
             "real_reference": get_real_reference_ids(knowledge, re_knowledge.references)}
            for re_knowledge in results.re_knowledge
        ]

        colored_print(f"  获取到 {len(results.re_knowledge)} 条知识", color='cyan')

    return knowledge, outline


# ============================================================================
# 报告生成（复用 src/agent/generate.py 的逻辑）
# ============================================================================

class OutputStatus:
    NormalContentStatus = 0
    ToolsStartMatch = 1
    ToolsOutput = 2
    MaybeReferenceInEnd = 3


class ContentProcessor:
    """与 src/agent/generate.py 中的 ContentProcessor 完全一致"""
    def __init__(self, knowledge: str):
        self.tools = ["table", "chart"]
        self.buffer = ""
        self.current_tool = ""
        self.report = ""
        self.result = []
        self.status = OutputStatus.NormalContentStatus
        self.max_tool_name_len = 0
        self.knowledge = knowledge
        for tool in self.tools:
            self.max_tool_name_len = max(self.max_tool_name_len, len(tool))

    def process_content(self, content: str):
        self.report = self.report + content
        for char in content:
            self._process_char(char)
        if self.status == OutputStatus.NormalContentStatus \
                or self.status == OutputStatus.MaybeReferenceInEnd:
            if check_reference_end(self.buffer):
                self.status = OutputStatus.MaybeReferenceInEnd
            else:
                self.status = OutputStatus.NormalContentStatus
        if self.status == OutputStatus.NormalContentStatus and self.buffer:
            self.result.append(self.buffer)
            self.buffer = ""
        if self.result:
            result, self.result = self.result, []
            return result
        return None

    def clear_buf(self):
        if self.buffer:
            final, self.buffer = self.buffer, ""
            return [final]
        return None

    def _process_char(self, char: str):
        if self.status == OutputStatus.NormalContentStatus \
                or self.status == OutputStatus.MaybeReferenceInEnd:
            if char == "<":
                self.status = OutputStatus.ToolsStartMatch
                if self.buffer:
                    self.result.append(self.buffer)
                self.buffer = char
            else:
                self.buffer += char
        elif self.status == OutputStatus.ToolsStartMatch:
            self.buffer += char
            if len(self.buffer) > self.max_tool_name_len + 2:
                self.status = OutputStatus.NormalContentStatus
            elif char == ">":
                for tool in self.tools:
                    if f"<{tool}>" == self.buffer.lower():
                        self.current_tool = tool
                        self.status = OutputStatus.ToolsOutput
                        break
                if self.status != OutputStatus.ToolsOutput:
                    self.status = OutputStatus.NormalContentStatus
        elif self.status == OutputStatus.ToolsOutput:
            self.buffer += char
            if char == ">":
                if self.buffer.lower().endswith(f"</{self.current_tool}>"):
                    tool_result = self._process_tool(self.buffer, self.current_tool)
                    if tool_result:
                        self.result.append(tool_result)
                    self.buffer = ""
                    self.status = OutputStatus.NormalContentStatus

    def _process_tool(self, tool_content: str, tool: str) -> str:
        if tool == "table":
            table = extract_xml_content(tool_content, "markdown")
            if table:
                return table[0]
            else:
                return ""
        elif tool == "chart":
            description = extract_xml_content(tool_content, "description")
            description = description[0] if description else ""
            index = self.report.rfind("###")
            above = self.report[index:] if index > 0 else self.report
            chart = llm(llm_type="report", messages=apply_prompt_template(
                prompt_name="generate/chart",
                state={
                    "above": above,
                    "description": description,
                    "reference": self.knowledge
                }
            ), stream=False)

            input_schema = extract_xml_content(chart, "input_schema")
            if not input_schema:
                input_schema = extract_xml_content(chart, "echarts")
            if input_schema:
                input_schema = input_schema[0]
                chart_id = str(int(time.time() * 1000))
                return f"""``` custom_html
<div id="{chart_id}" class="chart-container" style="width:800px; height:600px; "></div>
<script>
var chartDom = document.getElementById('{chart_id}');
var myChart = echarts.init(chartDom);
var option;
chartDom.style.display = 'block';
option = {input_schema};
myChart.setOption(option);
</script>
```"""
            else:
                return ""
        return ""


def _tail_text(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[-max_chars:]


def _head_text(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars]


def _build_generation_state(domain: str, now: str, topic: str, chapter_outline: str, outline: str, reference: str, above: str) -> dict:
    # Keep prompt input below the 32K context window. The generated report can be long,
    # but each chapter only needs nearby previous text plus its own evidence.
    return {
        "domain": domain,
        "now": now,
        "query": topic,
        "chapter_outline": _head_text(chapter_outline, 3000),
        "outline": _head_text(outline, 4000),
        "reference": _head_text(reference, 12000),
        "above": _tail_text(above, 3000),
    }


def check_reference_end(sb: str) -> bool:
    """与 src/agent/generate.py 一致"""
    last_bracket_open = sb.rfind('[')
    last_bracket_close = sb.rfind(']')
    if last_bracket_open >= 0 and last_bracket_close < last_bracket_open:
        return True
    trim_right = sb.rstrip(' ')
    if len(trim_right) > 0 and trim_right[-1] == ']':
        return True
    return False


def generate_node(outline: Chapter, topic: str, domain: str) -> str:
    """
    生成报告（与 src/agent/generate.py 的 generate_node 一致）
    """
    final_report = ""
    colored_print(f"\n{'#' * outline.level} {outline.title}\n", color="green", end="")
    final_report += f"{'#' * outline.level} {outline.title}\n"

    for level2_chapter in outline.sub_chapter:
        def ref_replace(s: str) -> str:
            all_id = re.findall(r'\d+', s)
            m: List[int] = []
            for s2 in all_id:
                try:
                    id = int(s2)
                    if 0 <= id < len(level2_chapter.learning_knowledge):
                        ref_ids = level2_chapter.learning_knowledge[id]["real_reference"]
                        m.extend(ref_ids)
                except ValueError:
                    continue
            m.sort()
            result = []
            prev = None
            for num in m:
                if num != prev:
                    result.append(f"[^%d]" % num)
                    prev = num
            return ''.join(result)

        chapter_title = f"{'#' * level2_chapter.level} {level2_chapter.title}"
        colored_print(f"{chapter_title}\n", color="green", end="")

        prev_report = final_report + f'\n{chapter_title}\n'
        chapter_report = ''

        # 准备知识内容
        level2_chapter.merge_knowledge()
        knowledge_str = level2_chapter.get_knowledge_str()

        content_processor = ContentProcessor(knowledge_str)

        for thinking, content in llm(llm_type="report", messages=apply_prompt_template(
                prompt_name="generate/generate",
                state=_build_generation_state(
                    domain=domain,
                    now=time.strftime("%a %b %d %Y"),
                    topic=topic,
                    chapter_outline=level2_chapter.get_outline(),
                    outline=outline.get_outline(),
                    reference=knowledge_str,
                    above=prev_report,
                )
        ), stream=True):
            if thinking:
                colored_print(thinking, color="orange", end="")
            if content:
                output_strs = content_processor.process_content(content)
                if output_strs:
                    for output_str in output_strs:
                        pattern = re.compile(r"(\[\^[^\[\]]+\] *)+")
                        output_str = pattern.sub(lambda m: ref_replace(m.group(0)), output_str)
                        chapter_report += output_str
                        colored_print(output_str, color="green", end="")

        colored_print('\n', color="green")
        if chapter_report.count(chapter_title):
            final_report = final_report + '\n' + chapter_report
        else:
            final_report = prev_report + chapter_report

    return final_report


def save_local_node(report: str, outline: Chapter, knowledge: List[Dict],
                    article_dir: str = None, references_dir: str = None,
                    file_base_name: str = None, save_html: bool = True):
    """
    保存报告到本地，文章和参考文献分开保存。

    Args:
        article_dir: 文章输出目录（若为 None 则使用默认值）
        references_dir: 参考文献输出目录（若为 None 则不保存参考文献文件）
        file_base_name: 文件基础名（若为 None 则用时间戳）
        save_html: 是否保存 HTML 版本
    """
    if article_dir is None:
        article_dir = _default_article_dir()

    os.makedirs(article_dir, exist_ok=True)

    if file_base_name:
        file_base = file_base_name
    else:
        file_base = f"report_{int(time.time() * 1000)}"

    # 参考文献保存为 JSON：标号、url、content
    if knowledge and references_dir:
        os.makedirs(references_dir, exist_ok=True)
        ref_path = os.path.join(references_dir, f"{file_base}.json")
        ref_data = []
        for k in knowledge:
            if k.get('url'):
                ref_data.append({
                    "id": k['id'],
                    "url": k['url'],
                    "content": k.get('content') or k.get('insight', ''),
                })
        with open(ref_path, 'w', encoding='utf-8') as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)
        colored_print(f"[SUCCESS] 参考文献已保存: {ref_path}", color="cyan")

    # 文章不含参考文献
    md_path = os.path.join(article_dir, f"{file_base}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    colored_print(f"[SUCCESS] Markdown报告已保存: {md_path}", color="green", bold=True)

    if save_html:
        html = markdown2html(outline.title, report)
        html_path = os.path.join(article_dir, f"{file_base}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        colored_print(f"[SUCCESS] HTML报告已保存: {html_path}", color="green", bold=True)


# ============================================================================
# 主函数
# ============================================================================

async def generate_from_outline(
    outline_path: str,
    search_path: str = None,
    topic: str = None,
    domain: str = "Industry Research",
    output_dir: str = None,
    save_html: bool = True,
    skip_search: bool = False,
    search_depth: int = 3,
    references_dir: str = None,
    file_base_name: str = None,
):
    output_dir = output_dir or _default_article_dir()
    references_dir = references_dir or _default_references_dir()
    """从大纲文件生成报告"""

    colored_print("=" * 60, color="blue")
    colored_print("OutlineArticleWriter - 自定义大纲报告生成器", color="blue", bold=True)
    colored_print("=" * 60, color="blue")

    # 1. 加载大纲
    colored_print("\n[Step 1/4] 加载大纲文件...", color="cyan")
    outline = load_outline_from_file(outline_path)

    if not topic:
        topic = outline.title

    colored_print(f"  标题: {outline.title}", color="white")
    colored_print(f"  章节数: {len(outline.sub_chapter)}", color="white")
    colored_print(f"  领域: {domain}", color="white")

    # 2. 知识搜索/加载
    knowledge = []
    if skip_search:
        colored_print("\n[Step 2/4] 跳过搜索（--no-search 模式）", color="yellow")
        # 为每个章节初始化空的 learning_knowledge
        for chapter in outline.sub_chapter:
            chapter.learning_knowledge = []
    elif search_path:
        colored_print("\n[Step 2/4] 加载预搜索结果...", color="cyan")
        search_results = load_search_results_from_file(search_path)
        colored_print(f"  已加载 {len(search_results)} 轮搜索结果", color="white")
        total_items = sum(len(r) for r in search_results)
        colored_print(f"  共 {total_items} 条搜索内容", color="white")

        # 将预搜索结果转换为 knowledge 格式
        search_id = 1
        for search_round in search_results:
            for item in search_round:
                knowledge.append({
                    "id": search_id,
                    "content": item.get('content', ''),
                    "url": item.get('url', '')
                })
                search_id += 1

        # 分配给每个章节（简化处理，平均分配）
        total_chapters = len(outline.sub_chapter)
        if total_chapters > 0:
            items_per_chapter = len(knowledge) // total_chapters
            for i, chapter in enumerate(outline.sub_chapter):
                start = i * items_per_chapter
                end = start + items_per_chapter if i < total_chapters - 1 else len(knowledge)
                chapter.knowledge = knowledge[start:end]
                chapter.learning_knowledge = [
                    {"insight": item['content'], "real_reference": [item['id']]}
                    for item in knowledge[start:end]
                ]
    else:
        colored_print("\n[Step 2/4] 执行章节深度搜索...", color="cyan")
        colored_print(f"  搜索深度: {search_depth}", color="white")
        colored_print(f"  每章节Top结果: {workflow_configs.get('search', {}).get('topN', 5)}", color="white")

        knowledge, outline = learning_node(
            outline=outline,
            search_depth=search_depth,
            top_n=workflow_configs.get("search", {}).get("topN", 5)
        )
        colored_print(f"\n  总计获取知识: {len(knowledge)} 条", color="green")

    # 3. 生成报告
    colored_print("\n[Step 3/4] 生成报告...", color="cyan")
    colored_print("-" * 40, color="purple")

    report = generate_node(
        outline=outline,
        topic=topic,
        domain=domain
    )

    # 4. 保存报告
    colored_print("\n[Step 4/4] 保存报告...", color="cyan")
    save_local_node(
        report=report,
        outline=outline,
        knowledge=knowledge,
        article_dir=output_dir,
        references_dir=references_dir,
        file_base_name=file_base_name,
        save_html=save_html
    )

    colored_print("\n" + "=" * 60, color="green")
    colored_print("报告生成完成!", color="green", bold=True)
    colored_print("=" * 60, color="green")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='OutlineArticleWriter - 从自定义大纲生成报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
与 run.py 的区别：
  - run.py: 用户输入主题 → 自动生成大纲 → 深度学习 → 生成报告
  - 本脚本: 用户指定大纲 → 深度学习（可跳过）→ 生成报告

示例:
  # 基本用法 - 只指定大纲（会自动对每个章节进行深度搜索）
  python src/run_from_outline.py --outline ./my_outline.md

  # 指定预搜索结果（跳过章节级搜索，直接生成报告）
  python src/run_from_outline.py --outline ./outline.md --search ./search_results.json

  # 跳过搜索，直接生成（知识为空，依赖模型泛化能力）
  python src/run_from_outline.py --outline ./outline.md --no-search

  # 指定领域和输出目录
  python src/run_from_outline.py --outline ./outline.md --domain "Company Research" --output ./output

大纲文件格式:
  # 报告标题
  ## 一、第一章标题
  ### 1.1 小节标题
  ### 1.2 小节标题
  ## 二、第二章标题
  ...

搜索结果文件格式:
  [
    [
      {"id": 1, "content": "搜索内容1", "url": "https://..."},
      {"id": 2, "content": "搜索内容2", "url": "https://..."}
    ],
    [
      {"id": 3, "content": "第二轮搜索内容", "url": "https://..."}
    ]
  ]
        """
    )

    parser.add_argument('--outline', '-o', required=True,
                       help='大纲文件路径 (Markdown格式)')
    parser.add_argument('--search', '-s',
                       help='搜索结果文件路径 (JSON格式，可选)')
    parser.add_argument('--topic', '-t',
                       help='报告标题 (默认使用大纲中的标题)')
    parser.add_argument('--domain', '-d', default='Industry Research',
                       choices=['Industry Research', 'Company Research', 'Comprehensive Analysis'],
                       help='领域类型 (默认: Industry Research)')
    parser.add_argument('--output', '-out', default=_default_article_dir(),
                       help='文章输出目录（默认取仓库配置 article.article_dir）')
    parser.add_argument('--references-output', default=_default_references_dir(),
                       help='参考文献输出目录（默认取仓库配置 article.references_dir）')
    parser.add_argument('--no-html', action='store_true',
                       help='不生成HTML文件')
    parser.add_argument('--no-search', action='store_true',
                       help='跳过搜索，直接生成报告（知识为空）')
    parser.add_argument('--depth', type=int, default=3,
                       help='搜索深度 (默认: 3)')

    args = parser.parse_args()

    asyncio.run(generate_from_outline(
        outline_path=args.outline,
        search_path=args.search,
        topic=args.topic,
        domain=args.domain,
        output_dir=args.output,
        references_dir=args.references_output,
        save_html=not args.no_html,
        skip_search=args.no_search,
        search_depth=args.depth
    ))


if __name__ == '__main__':
    main()
