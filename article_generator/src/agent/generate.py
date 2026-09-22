import json
import logging
import os
from dataclasses import dataclass
from typing import List
import re

from langchain_core.runnables import RunnableConfig

from .message import ReportState
from src.llms.llm import llm
from datetime import datetime
import time

from src.prompts import apply_prompt_template
from src.utils.parse_model_res import extract_xml_content
from src.utils.print_util import colored_print
from src.tools.md2html import markdown2html

logger = logging.getLogger(__name__)

try:  # central configuration: paths, keys, topics
    import sys as _sys
    from pathlib import Path as _Path

    _REPO_ROOT = _Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT))
    import twpgen_settings as _repo_config

    _DEFAULT_ARTICLE_DIR = _repo_config.article_dir
    _DEFAULT_REFERENCES_DIR = _repo_config.references_dir
except Exception:  # pragma: no cover
    _DEFAULT_ARTICLE_DIR = "output/article"
    _DEFAULT_REFERENCES_DIR = "output/references"

def generate_node(state: ReportState):

    outline = state.get("outline")
    final_report = ""
    colored_print(f"{'#' * outline.level} {outline.title}\n", color="green", end="")
    final_report += f"{'#' * outline.level} {outline.title}\n"
    for level2_chapter in outline.sub_chapter:
        def ref_replace(s: str) -> str:
            """
            Replace the reference IDs in the string with the corresponding actual reference IDs

            :param s: Original string
            :param chapter: Chapter object containing the LearningKnowledge attribute
            :return: Replaced string
            """
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

        level2_chapter.merge_knowledge()
        knowledge = level2_chapter.merge_knowledge().get_knowledge_str()
        content_processor = ContentProcessor(knowledge)
        for thinking, content in llm(llm_type="report", messages=apply_prompt_template(
                prompt_name="generate/generate",
                state=_build_generation_state(
                    domain=state.get("domain"),
                    now=datetime.now().strftime("%a %b %d %Y"),
                    topic=state.get("topic"),
                    chapter_outline=level2_chapter.get_outline(),
                    outline=outline.get_outline(),
                    reference=knowledge,
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

    return {
                "final_report": final_report,
                "output": {
                    "message": final_report,
                }
            }


def save_report_local(state: ReportState, config:RunnableConfig):
    """
    whether to save report locally
    """
    need_html = config.get("configurable", {}).get("save_as_html", True)
    if need_html:
        return "save_local_node"
    else:
        return "__end__"


def save_local_node(state: ReportState, config: RunnableConfig):
    """
    Save article and references under /workspace/TWP-Gen/output by default.
    """
    configurable = config.get("configurable", {})
    article_dir = configurable.get("save_path", _DEFAULT_ARTICLE_DIR)
    references_dir = configurable.get("references_path", _DEFAULT_REFERENCES_DIR)
    file_base = f"report_{str(int(time.time() * 1000))}"
    try:
        os.makedirs(article_dir, exist_ok=True)
        os.makedirs(references_dir, exist_ok=True)
    except OSError:
        logger.error(f"Failed to create output directories: {article_dir}, {references_dir}")

    report = state.get("final_report")
    knowledge = state.get("knowledge") or []

    if knowledge:
        ref_data = [
            {
                "id": k.get("id"),
                "url": k.get("url", ""),
                "content": k.get("content") or k.get("insight", ""),
            }
            for k in knowledge
            if k.get("url")
        ]
        ref_path = os.path.join(references_dir, f"{file_base}.json")
        with open(ref_path, 'w', encoding='utf-8') as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(article_dir, f"{file_base}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)

    outline = state.get("outline")
    need_html = config.get("configurable", {}).get("save_as_html", True)
    if need_html:
        html = markdown2html(outline.title, report)
        file_path = os.path.join(article_dir, f"{file_base}.html")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)
            colored_print("The report is saved in ", color="red", bg_color="black", end="")
            colored_print(file_path, color="red", bg_color="black", bold=True, underline=True)
        except Exception:
            logger.error(f"Failed to save report: {file_path}")


@dataclass
class OutputStatus:
    NormalContentStatus = 0
    ToolsStartMatch = 1
    ToolsOutput = 2
    MaybeReferenceInEnd = 3


class ContentProcessor:
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
            if description:
                description = description[0]
            else:
                description = ""
            index = self.report.rfind("###")
            if index > 0:
                above = self.report[index:]
            else:
                above = self.report
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
    """
    Check if the reference has ended and determine if caching needs to continue
        params sb: Content in string builder (passed in string form)
        return: If the reference has ended, return True; otherwise, return False
    """
    last_bracket_open = sb.rfind('[')
    last_bracket_close = sb.rfind(']')
    if last_bracket_open >= 0 and last_bracket_close < last_bracket_open:
        return True

    trim_right = sb.rstrip(' ')
    if len(trim_right) > 0 and trim_right[-1] == ']':
        return True

    return False


if __name__ == '__main__':
    cp = ContentProcessor("")
    contents = ["This is a test case [^",
                "1].",
                "I will test whether multiple",
                " consecutive references [^2]",
                "[^3] can be correctly parsed and called by the tool",
                "For example:",
                "<Table><markdown>", "| h1 | h2 |\n|---|---|\n| v1 | v2 |",
                "</markdown>",
                "</Table>Can it be parsed.\n",
                "This<Tool>",
                "</Tool>is not"
                " an effective tool"]
    for content in contents:
        results = cp.process_content(content)
        print("=" * 30)
        if results:
            for result in results:
                print(result)
    results = cp.clear_buf()
    print("=" * 30)
    if results:
        for result in results:
            print(result)
    print("=" * 30)
