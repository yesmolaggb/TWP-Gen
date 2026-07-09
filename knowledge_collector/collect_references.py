"""
collect_references.py

TWP-Gen information collection stage.

Flow:
1. Read information_collector/topic.txt, or accept --topic for a single topic.
2. Generate search queries with the configured OpenAI-compatible LLM.
3. Search with Tavily, optionally combined with a local Google Alerts service.
4. Fetch and clean web pages concurrently.
5. Save the raw collection report and write corpus.txt under TWPGEN_DATASET_ROOT.
"""

import re
import datetime
import urllib.parse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import trafilatura
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import os
from openai import OpenAI
from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────

from typing import List, Dict

COLLECTOR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("TWPGEN_ROOT", COLLECTOR_ROOT.parent)).resolve()


def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
LLM_MODEL = os.environ.get("TWPGEN_LLM_MODEL", "qwen3-max")


def make_llm_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for reference collection")
    if not OPENAI_BASE_URL:
        raise RuntimeError("OPENAI_BASE_URL or OPENAI_API_BASE is required for reference collection")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


class TavilyKeyPool:
    """Tavily API key pool loaded from env vars or an optional key file."""

    def __init__(self, key_file: str | None = None):
        self._keys: list[str] = []
        self._idx: int = 0
        self._failed: set[int] = set()
        self._load(key_file)

    def _load(self, key_file: str | None):
        raw_keys = os.environ.get("TAVILY_API_KEYS") or os.environ.get("TAVILY_API_KEY", "")
        if raw_keys:
            pieces = re.split(r"[\n,;]+", raw_keys)
            self._keys = [item.strip() for item in pieces if item.strip()]
        elif key_file:
            path = Path(key_file).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Tavily key file not found: {key_file}")
            lines = path.read_text(encoding="utf-8").splitlines()
            self._keys = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        if not self._keys:
            raise ValueError("No Tavily API key found. Set TAVILY_API_KEYS, TAVILY_API_KEY, or TWPGEN_TAVILY_KEY_FILE.")
        log(f"  -> Loaded {len(self._keys)} Tavily API key(s)")

    def _advance(self):
        original = self._idx
        while True:
            if self._idx not in self._failed and self._keys[self._idx]:
                return
            self._idx = (self._idx + 1) % len(self._keys)
            if self._idx == original:
                break
        if len(self._failed) >= len(self._keys):
            raise RuntimeError("All Tavily API keys have failed")

    def get_key(self) -> str:
        self._advance()
        return self._keys[self._idx]

    def rotate(self):
        self._idx = (self._idx + 1) % len(self._keys)
        self._advance()

    def report_failure(self, key: str, is_auth_error: bool = False):
        try:
            failed_idx = self._keys.index(key)
        except ValueError:
            return
        if is_auth_error and failed_idx not in self._failed:
            self._failed.add(failed_idx)
            log(f"  -> Tavily key [{key[:12]}...] failed, switching ({len(self._failed)}/{len(self._keys)})")
            self._advance()


_tavily_pool: TavilyKeyPool | None = None


def get_tavily_pool() -> TavilyKeyPool:
    global _tavily_pool
    if _tavily_pool is None:
        _tavily_pool = TavilyKeyPool(os.environ.get("TWPGEN_TAVILY_KEY_FILE"))
    return _tavily_pool


def get_tavily_client() -> tuple["TavilyClient", "TavilyKeyPool"]:
    """获取一个 TavilyClient 实例及对应的 key 池"""
    pool = get_tavily_pool()
    key = pool.get_key()
    return TavilyClient(api_key=key), pool



# Google Alerts API service
GOOGLE_ALERTS_API_URL = os.environ.get("TWPGEN_GOOGLE_ALERTS_API_URL", "http://localhost:5000")

PROMPT_PATH = COLLECTOR_ROOT / "prompt" / "search.txt"
EXTRACT_PROMPT_PATH = COLLECTOR_ROOT / "prompt" / "extract.txt"

TOPIC_FILE = Path(os.environ.get("TWPGEN_TOPIC_FILE", COLLECTOR_ROOT / "topic.txt")).expanduser()
RESULT_DIR = Path(os.environ.get("TWPGEN_COLLECTOR_RESULT_DIR", COLLECTOR_ROOT / "result")).expanduser()
DATASET_DIR = Path(os.environ.get("TWPGEN_DATASET_ROOT", PROJECT_ROOT / "dataset")).expanduser()

TOP_N = int(os.environ.get("TWPGEN_SEARCH_TOP_N", "10"))

# ─────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────

@dataclass
class SearchResult:
    id: int
    title: str
    url: str
    content: str
    source: str = "tavily"  # "tavily" or "google_alerts"

# ─────────────────────────────────────────────────────
# 切块器
# ─────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n", "\n", " ", ".", ",",
        "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", "",
    ],
)

# ─────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """去除完全重复的 chunk，保留首次出现的顺序"""
    seen = set()
    result = []
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def deduplicate_lines_in_corpus(chunks: List[str]) -> List[str]:
    """
    对 corpus.txt 中的所有行进行去重。
    场景：多个关键词/多篇文章切块后，
    同一句话可能散落在不同 chunk 中导致重复。
    """
    all_lines = []
    for chunk in chunks:
        # chunk 内部用换行拆分（每行一条）
        lines = chunk.splitlines()
        all_lines.extend(lines)

    seen = set()
    result = []
    for line in all_lines:
        key = line.strip()
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            result.append(line)
    return result


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_extract_prompt() -> str:
    """加载文章清洗提示词模板"""
    raw = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    # 找到 PROMPT = ''' 之后的完整内容
    match = re.search(r"PROMPT\s*=\s*'''(.+?)'''", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def deduplicate_lines(text: str) -> str:
    """去除文本中完全重复的行（strip后比较），保留首次出现的顺序"""
    lines = text.splitlines()
    seen = set()
    result = []
    for line in lines:
        key = line.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(line)
    return "\n".join(result)


def extract_article(text: str) -> str:
    """
    将爬取的原始文本交给 LLM 清洗：
    - 删除废话、空话、套话、重复内容
    - 保留核心信息，适度精简但不改变原意
    - 超过500字符自动拆条
    - 每行一条，不编号不加标题
    """
    if not text or len(text.strip()) < 50:
        return ""

    prompt_template = load_extract_prompt()
    prompt = prompt_template.replace("{article}", text)

    try:
        client = make_llm_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        extracted = resp.choices[0].message.content.strip()
        # 清洗 LLM 输出中自身残留的完全重复行
        return deduplicate_lines(extracted) if extracted else text
    except Exception as e:
        log(f"    → LLM 清洗失败，回退原始文本: {e}")
        return text


def load_topics(topic_file: Path) -> List[str]:
    """
    读取 topic.txt，每一行作为一个题目
    自动跳过空行
    """
    if not topic_file.exists():
        raise FileNotFoundError(f"题目文件不存在: {topic_file}")

    lines = topic_file.read_text(encoding="utf-8").splitlines()
    topics = [line.strip() for line in lines if line.strip()]
    return topics


def extract_search_queries(text: str) -> List[str]:
    queries = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return [q.strip() for q in queries if 3 <= len(q.strip()) <= 500]


def safe_filename(name: str, max_len: int = 120) -> str:
    """
    清洗题目，避免非法文件名字符
    """
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    return name[:max_len]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# 需要登录或 JS 渲染，直接跳过
SKIP_DOMAINS = {
    "reddit.com", "x.com", "twitter.com",
    "zhihu.com", "bilibili.com",
}


def fetch_page(url: str) -> str:
    """用 Tavily extract API 抓取 URL 正文（同时覆盖 Tavily 和 Google Alerts 的 URL）"""
    pool = get_tavily_pool()
    key = pool.get_key()

    if not key:
        log(f"    -> 警告: Tavily Key 池为空，尝试 trafilatura 兜底")
        return fetch_page_trafilatura(url)

    domain = urllib.parse.urlparse(url).netloc.lstrip("www.")
    if any(domain.endswith(d) for d in SKIP_DOMAINS):
        log(f"    -> 跳过（需要登录: {domain}）")
        return ""

    if url.lower().endswith(".pdf"):
        return _fetch_pdf(url)

    try:
        client = TavilyClient(api_key=key)
        resp = client.extract(urls=[url], include_images=False)
        results = resp.get("results", [])
        failed = resp.get("failed_results", [])
        if failed:
            log(f"    -> Tavily extract 部分 URL 失败，尝试 trafilatura 兜底: {failed}")
            return fetch_page_trafilatura(url)
        raw = results[0].get("raw_content", "") if results else ""
        if raw and len(raw.strip()) > 50:
            return raw
        log(f"    -> Tavily extract 返回空，尝试 trafilatura 兜底")
        return fetch_page_trafilatura(url)
    except Exception as e:
        is_auth = any(code in str(e) for code in ["401", "403", "429", "Forbidden", "Unauthorized", "quota"])
        pool.report_failure(key, is_auth_error=is_auth)
        log(f"    -> Tavily extract 异常{'（认证/额度错误）' if is_auth else '（临时错误）'}: {e}")
        return fetch_page_trafilatura(url)


def fetch_page_trafilatura(url: str) -> str:
    """trafilatura + requests 兜底抓取（PDF 走 PDF 专用逻辑）"""
    domain = urllib.parse.urlparse(url).netloc.lstrip("www.")
    if any(domain.endswith(d) for d in SKIP_DOMAINS):
        log(f"    -> 跳过（需要登录: {domain}）")
        return ""

    if url.lower().endswith(".pdf"):
        return _fetch_pdf(url)

    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 50:
            return text
    except Exception:
        pass

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 50:
            return text
    except Exception:
        pass

    return ""


def _fetch_pdf(url: str) -> str:
    """下载 PDF 并提取文字"""
    try:
        import io
        import pdfplumber
        import requests

        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:20]]

        return "\n".join(pages).strip()
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """清洗文章文本：去序号、去数字点占比高的行、去空行、压成一行"""
    cleaned_lines = []

    def is_cjk(c):
        cp = ord(c)
        return (
            0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x3000 <= cp <= 0x303F or
            0xFF00 <= cp <= 0xFFEF or
            0x3040 <= cp <= 0x30FF or
            0xAC00 <= cp <= 0xD7AF
        )

    def is_chinese(c):
        cp = ord(c)
        return (
            0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF
        )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^\d+[、．.]\s*", "", line)

        if len(line) < 20:
            continue

        digit_dot_count = sum(1 for c in line if c.isdigit() or c in ".．。")
        if len(line) > 0 and digit_dot_count / len(line) > 0.4:
            continue

        ascii_count = sum(1 for c in line if c.isascii() and c.strip())
        if len(line) > 0 and ascii_count / len(line) > 0.8:
            continue

        visible_chars = [c for c in line if not c.isspace()]
        chinese_count = sum(1 for c in visible_chars if is_chinese(c))
        if visible_chars and chinese_count / len(visible_chars) < 0.6:
            continue

        garbage_count = sum(1 for c in line if not c.isascii() and not is_cjk(c))
        if len(line) > 0 and garbage_count / len(line) > 0.6:
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()

# ═══════════════════════════════════════════════════════
# Google Alerts API 检测与调用
# ═══════════════════════════════════════════════════════

def check_google_alerts_api() -> bool:
    """检测 Google Alerts API 服务是否启动"""
    try:
        resp = requests.get(f"{GOOGLE_ALERTS_API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def search_via_api(keywords: List[str], target_count: int = 10) -> Dict[str, List[str]]:
    """
    通过 API 搜索，返回 {keyword: [urls]} 字典
    """
    try:
        resp = requests.post(
            f"{GOOGLE_ALERTS_API_URL}/search",
            json={"keywords": keywords, "target_count": target_count},
            timeout=600
        )
        if resp.status_code == 200:
            return resp.json().get("results", {})
        return {}
    except (Exception, KeyboardInterrupt, SystemExit) as e:
        log(f"  [API] 调用失败: {e}")
        return {}


def run_search_google_alerts(queries: List[str]) -> Dict[str, List[SearchResult]]:
    """
    通过 API 执行 Google Alerts 搜索
    """
    all_results: Dict[str, List[SearchResult]] = {}
    search_id = 1

    # 批量调用 API
    api_results = search_via_api(queries, TOP_N)

    for query in queries:
        urls = api_results.get(query, [])
        items = []
        for url in urls:
            items.append(SearchResult(
                id=search_id,
                title="",  # Google Alerts 不返回标题
                url=url,
                content="",
                source="google_alerts",
            ))
            search_id += 1
        all_results[query] = items
        log(f"  [Google Alerts] {query[:50]} → {len(items)} 条")

    return all_results


def fetch_article_google_alerts(url: str) -> str:
    """Google Alerts 方式：统一走 fetch_page（Tavily extract API 兜底）"""
    return fetch_page(url)


# ─────────────────────────────────────────────────────
# Qwen3：生成搜索关键词
# ─────────────────────────────────────────────────────

def generate_queries(user_question: str) -> List[str]:
    system_prompt = load_prompt()
    client = make_llm_client()

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        extra_body={"enable_thinking": True},
        stream=True,
    )

    answer_buf = []
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            answer_buf.append(delta.content)

    return extract_search_queries("".join(answer_buf))

# ─────────────────────────────────────────────────────
# Tavily：执行搜索
# ─────────────────────────────────────────────────────

def run_search(queries: List[str]) -> Dict[str, List[SearchResult]]:
    pool = get_tavily_pool()
    all_results: Dict[str, List[SearchResult]] = {}
    search_id = 1

    for query in queries:
        items = []
        dead_keys = 0  # 累计死掉的 key 数，用于判断池是否耗尽
        rotated_this_query = False  # 本 query 内是否已经 rotate 过

        while True:
            try:
                key = pool.get_key()
            except RuntimeError:
                # 所有 key 都死透了
                log(f"  ⚠ [{query[:50]}] 所有 Key 均已失效，放弃搜索")
                break

            try:
                client = TavilyClient(api_key=key)
                resp = client.search(
                    query,
                    max_results=TOP_N,
                    include_raw_content=False,
                    country="china"
                )
                results_list = resp.get("results", [])

                if results_list:
                    for raw in results_list:
                        items.append(SearchResult(
                            id=search_id,
                            title=raw.get("title", ""),
                            url=raw.get("url", ""),
                            content="",
                            source="tavily",
                        ))
                        search_id += 1
                    log(f"  [{query[:50]}] → {len(items)} 条")
                    break  # 有结果，退出本 query，进入下一个
                else:
                    # 静默限流：返回空，换 key 重试
                    log(f"  ⚠ [{query[:50]}] Key [{key[:12]}...] 返回 0 条，切换 Key 重试")
                    pool.rotate()
                    rotated_this_query = True
                    continue

            except Exception as e:
                is_auth = any(code in str(e) for code in ["401", "403", "429", "Forbidden", "Unauthorized", "quota", "usage limit", "upgrade", "exceed"])
                if is_auth:
                    pool.report_failure(key, is_auth_error=True)
                    dead_keys += 1
                    log(f"  ⚠ [{query[:50]}] Key [{key[:12]}...] 额度耗尽（{dead_keys}/10），切换下一个")
                    try:
                        pool.get_key()  # 验证是否还有可用 key
                    except RuntimeError:
                        log(f"  ⚠ [{query[:50]}] 所有 Key 均已失效，放弃搜索")
                        break
                else:
                    # 临时错误，换 key 重试
                    log(f"  ⚠ [{query[:50]}] 临时错误，切换 Key: {e}")
                    pool.rotate()
                    rotated_this_query = True
                continue

        all_results[query] = items
        if not items:
            log(f"  [{query[:50]}] → 0 条")

    return all_results

# ─────────────────────────────────────────────────────
# 结果明细文本
# ─────────────────────────────────────────────────────

def build_result_report(user_question: str, crawled_results: Dict[str, List[Dict]], use_dual_search: bool) -> str:
    """
    生成保存到 information_collector/result/题目.txt 的文本
    """
    lines = []
    lines.append(user_question)
    lines.append("")
    lines.append(f"搜索模式：{'双重搜索' if use_dual_search else '单一搜索（Tavily）'}")
    lines.append("")
    lines.append("关键词拆分：")

    if crawled_results:
        for idx, query in enumerate(crawled_results.keys(), 1):
            lines.append(f"{idx}. {query}")
    else:
        lines.append("无")

    lines.append("")

    for query, articles in crawled_results.items():
        lines.append("=" * 80)
        lines.append(f"关键词：{query}")
        lines.append("=" * 80)
        lines.append("")

        if not articles:
            lines.append("无搜索结果")
            lines.append("")
            continue

        for idx, article in enumerate(articles, 1):
            source_tag = f"[{article.get('source', 'unknown')}]"
            lines.append(f"[文章 {idx}] {article.get('title', '')} {source_tag}")
            lines.append(f"URL: {article.get('url', '')}")
            lines.append("正文：")
            lines.append(article.get("content", "") or "抓取失败或内容为空")
            lines.append("")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────
# 保存 & 多线程 URL 爬取
# ─────────────────────────────────────────────────────


def _process_single_url(args):
    """线程函数：抓取一个 URL，走完整处理流程，返回 article 字典或 None"""
    query, item, article_id_ref, lq = args

    with article_id_ref["lock"]:
        aid = article_id_ref["next_id"]
        article_id_ref["next_id"] += 1

    source_tag = "Google Alerts" if item.source == "google_alerts" else "Tavily"
    lq.put(f"  抓取 [{aid}] {item.url[:70]}... [{source_tag}]")

    text = fetch_page(item.url)

    if not text:
        lq.put("    → 空内容，跳过")
        return None

    text = clean_text(text)
    if not text:
        lq.put("    → 清洗后为空，跳过")
        return None

    text = extract_article(text)
    if not text:
        lq.put("    → LLM清洗后为空，跳过")
        return None

    lq.put(f"    → 成功，{len(text)} 字符")
    chunks = text_splitter.split_text(text)

    return {
        "query": query,
        "article": {
            "title": item.title or "无标题",
            "url": item.url,
            "content": text,
            "source": item.source,
        },
        "chunks": chunks,
    }


def _drain_log_queue(lq: queue.Queue):
    """把日志队列中的消息全部输出"""
    while True:
        try:
            msg = lq.get_nowait()
            if msg is None:
                break
            log(msg)
        except queue.Empty:
            break


def save_results(user_question: str, all_results: Dict[str, List[SearchResult]], use_dual_search: bool = False):
    """
    每个题目保存两份内容：
    1. information_collector/result/题目.txt
       - 保存题目、关键词、文章原文
    2. TWPGEN_DATASET_ROOT/题目/corpus.txt
       - 保存切块后的 chunk

    逻辑：
    - 10 线程并发爬取所有 URL（fetch_page + clean_text + extract_article）
    - Google Alerts 和 Tavily 的 URL 统一走 fetch_page（基于 Tavily extract API）
    """
    crawled_results: Dict[str, List[Dict]] = {}
    all_chunks: List[str] = []

    for query in all_results.keys():
        crawled_results[query] = []

    # 线程安全的计数器和锁
    article_id_ref = {"next_id": 1, "lock": threading.Lock()}
    chunks_lock = threading.Lock()
    lq = queue.Queue()

    total_urls = sum(len(v) for v in all_results.values())
    log(f"  → 启动 10 线程并发爬取，共 {total_urls} 个 URL")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for query, items in all_results.items():
            for item in items:
                future = executor.submit(_process_single_url, (query, item, article_id_ref, lq))
                futures[future] = (query, item)

        for future in as_completed(futures):
            result = future.result()
            _drain_log_queue(lq)

            if result is not None:
                query = result["query"]
                article = result["article"]
                chunks = result["chunks"]

                with chunks_lock:
                    all_chunks.extend(chunks)

                crawled_results[query].append(article)

    # 把队列里剩余的日志全部输出
    _drain_log_queue(lq)

    # 输出结果
    safe_name = safe_filename(user_question)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULT_DIR / f"{safe_name}.txt"
    report_text = build_result_report(user_question, crawled_results, use_dual_search)
    result_file.write_text(report_text, encoding="utf-8")
    log(f"  → 结果明细已写入 {result_file}")

    topic_dir = DATASET_DIR / safe_name
    topic_dir.mkdir(parents=True, exist_ok=True)

    final_chunks = deduplicate_chunks(all_chunks)
    final_lines = deduplicate_lines_in_corpus(all_chunks)
    chunk_file = topic_dir / "corpus.txt"
    chunk_file.write_text("\n".join(final_lines), encoding="utf-8")
    log(f"  → 共 {len(final_lines)} 行（chunk去重后 {len(final_chunks)}，原始 {len(all_chunks)}）")
    log(f"  → chunk 已写入 {chunk_file}")

# ─────────────────────────────────────────────────────
# 单题目处理
# ─────────────────────────────────────────────────────

def process_topic(user_question: str):
    log("=" * 100)
    log(f"开始处理题目：{user_question}")

    # 检测 Google Alerts API 是否可用
    use_dual_search = check_google_alerts_api()
    if use_dual_search:
        log("  → Google Alerts API 已启动，使用双重搜索模式")
    else:
        log("  → Google Alerts API 未启动，使用单一搜索模式（Tavily）")

    log("Step 1/3 - Qwen3 生成搜索关键词...")
    queries = generate_queries(user_question)
    if not queries:
        log("✗ 未提取到关键词，跳过")
        return

    log(f"  → {len(queries)} 个关键词:")
    for q in queries:
        log(f"     • {q}")

    log("Step 2/3 - 执行搜索...")
    all_results = run_search(queries)

    # 如果 Google Alerts API 可用，执行双重搜索
    if use_dual_search:
        log("  [Google Alerts API] 搜索...")
        alerts_results = run_search_google_alerts(queries)

        # 合并结果，按 URL 去重
        for query in queries:
            existing_urls = {item.url for item in all_results.get(query, [])}
            for item in alerts_results.get(query, []):
                if item.url not in existing_urls:
                    all_results.setdefault(query, []).append(item)
                    existing_urls.add(item.url)
    else:
        log("  [跳过] Google Alerts API 未启动")

    log("Step 3/3 - 抓取页面内容并保存...")
    save_results(user_question, all_results, use_dual_search)

    log(f"完成：{user_question}")
    log("=" * 100)

# ─────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="collect_references: 搜索并生成语料")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="指定单个题目（不传则遍历 topic.txt 所有题目）",
    )
    args = parser.parse_args()

    if args.topic:
        # 单题目模式：只处理指定题目
        try:
            process_topic(args.topic)
        except (Exception, KeyboardInterrupt, SystemExit) as e:
            log(f"✗ 处理失败：{args.topic}")
            log(f"  错误信息：{e}")
    else:
        # 全量模式：遍历 topic.txt 所有题目
        topics = load_topics(TOPIC_FILE)

        if not topics:
            log(f"✗ {TOPIC_FILE} 中没有可处理的题目")
            return

        log(f"共读取到 {len(topics)} 个题目")

        for idx, topic in enumerate(topics, 1):
            log(f"[{idx}/{len(topics)}] 准备处理")
            try:
                process_topic(topic)
            except Exception as e:
                log(f"✗ 处理失败：{topic}")
                log(f"  错误信息：{e}")

        log("全部处理完成 ✓")


if __name__ == "__main__":
    main()
