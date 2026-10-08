import os
import re
import sys
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========= 配置 =========
INPUT_ROOT = Path(os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy"))
SOURCE_MD_NAME = "toc_lines.md"                  # 优先用这个
FALLBACK_MD_NAME = "combined.md"                 # 没有就用这个
TOC_NAME = "toc.md"

MODEL_NAME = os.getenv("TWPGEN_TAXONOMY_MODEL", "qwen3-max")
MAX_REGEN = 3
MAX_WORKERS = 5

# 生成/审查输出 token
MAX_TOKENS_GEN = 1200
MAX_TOKENS_AUDIT = 200

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ======= 正则 =======
RE_L1 = re.compile(r"^[一二三四五六七八九十]+、\S+")
RE_L2 = re.compile(r"^  \d+\.\d+ .+")
RE_L3 = re.compile(r"^(\s*)(\d+)\.(\d+)\.\d+(.*)$")   # 新增：三级标题检测
RE_BAD = re.compile(
    r"(第[一二三四五六七八九十0-9]+节)|(\d+\.\d+\.\d+)|(^\s*[一二三四五六七八九十]+、\s*)|（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\)"
)

# ======= 新增：强制压平三级标题 =======
def flatten_three_level(toc_text: str) -> str:
    """把 X.Y.Z 三级标题强制改写成 X.Y 二级，避免因三级标题耗尽重试次数"""
    lines = toc_text.splitlines()
    new_lines = []
    for line in lines:
        m = RE_L3.match(line)
        if m:
            major, mid, title = m.group(2), m.group(3), m.group(4)
            new_lines.append(f"  {major}.{mid}{title}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

# ======= 小工具 =======
def make_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

def call_llm(client: OpenAI, messages, max_tokens=800, temperature=0.0) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()

def read_source_text(book_dir: Path) -> str:
    p1 = book_dir / SOURCE_MD_NAME
    p2 = book_dir / FALLBACK_MD_NAME
    if p1.exists():
        return p1.read_text(encoding="utf-8", errors="ignore")
    if p2.exists():
        return p2.read_text(encoding="utf-8", errors="ignore")
    return ""

def is_already_processed(book_dir: Path) -> bool:
    """
    已处理判断：
    toc.md 存在且非空，就认为已处理，直接跳过
    """
    toc_path = book_dir / TOC_NAME
    if not toc_path.exists():
        return False
    try:
        content = toc_path.read_text(encoding="utf-8", errors="ignore").strip()
        return bool(content)
    except Exception:
        return False

# ======= 生成提示词（严格两级、严格编号） =======
def build_gen_messages(book_name: str, source_text: str, extra_fix):
    system = (
        "你是一个目录整理器。用户提供的是目录页/候选目录行抽取文本。\n"
        "你只能基于用户给定文本整理目录，禁止总结正文，禁止编造不存在的条目。\n\n"
        "输出必须严格符合以下格式（仅两级）：\n"
        "1) 第一行：书名（与用户提供一致）\n"
        "2) 第二行：空行\n"
        "3) 第三行：######\n"
        "4) 一级行：一、标题 / 二、标题 / 三、标题 ...（中文大写数字 + 顿号）\n"
        "5) 二级行：两空格开头 + X.Y + 空格 + 标题，例如：  1.1 标题、  2.3 标题\n"
        "6) 严禁出现三级及以后：不要 1.1.1，不要第X节，不要（一），不要 一、 出现在二级\n"
        "7) 去掉页码/点线/装饰符号，只保留标题文字\n"
        "8) 只输出目录，不要任何解释\n"
    )
    if extra_fix:
        system += f"\n【上次审查不合格原因】{extra_fix}\n请修正后输出。"

    user = (
        f"书名：{book_name}\n"
        f"来源文件优先：{SOURCE_MD_NAME}（没有则用 {FALLBACK_MD_NAME}）。\n"
        "请基于下面文本整理目录：\n\n"
        f"{source_text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ======= 审查提示词：只返 PASS 或 FAIL:原因 =======
def build_audit_messages(book_name: str, toc_text: str):
    system = (
        "你是目录格式审查器，只输出 PASS 或 FAIL:原因。\n"
        "合格标准：\n"
        "A) 第一行必须是书名（包含用户书名关键词即可）\n"
        "B) 第三行必须是 ######\n"
        "C) 一级行必须形如：一、... / 二、... / 三、...\n"
        "D) 二级行必须形如：两空格 + 数字.数字 + 空格 + 标题（如：  1.1 xxx）\n"
        "E) 禁止三级及以后：禁止 1.1.1、(一)/(二)、（一）（二）、第X节、以及二级里出现一、二、\n"
        "只输出 PASS 或 FAIL:...\n"
    )
    user = f"书名：{book_name}\n\n这是 toc.md 内容：\n\n{toc_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ======= 本地快速审查 =======
def quick_check(toc_text: str):
    lines = [x.rstrip() for x in toc_text.splitlines()]
    if len(lines) < 3:
        return False, "内容行数太少"
    if "######" not in lines[2]:
        return False, "第三行不是######"

    for ln in lines[3:]:
        if not ln.strip():
            continue
        if RE_BAD.search(ln) and not RE_L1.match(ln):
            return False, f"包含不允许的层级/节/三级: {ln[:60]}"
        if not (RE_L1.match(ln) or RE_L2.match(ln)):
            return False, f"行格式不符合一级或二级: {ln[:60]}"
    return True, "OK"

# ======= 单本书处理 =======
def process_one_book(book_dir_str: str) -> dict:
    """
    子进程执行函数
    """
    book_dir = Path(book_dir_str)
    book_name = book_dir.name
    toc_path = book_dir / TOC_NAME

    try:
        # 已处理过则跳过
        if is_already_processed(book_dir):
            return {"book": book_name, "status": "skipped", "reason": "toc.md already exists and is non-empty"}

        source_text = read_source_text(book_dir)
        if not source_text.strip():
            return {"book": book_name, "status": "skipped", "reason": "source text is empty"}

        client = make_client()
        extra_fix = None

        for attempt in range(1, MAX_REGEN + 1):
            # 生成
            gen_msgs = build_gen_messages(book_name, source_text, extra_fix)
            toc_text = call_llm(client, gen_msgs, max_tokens=MAX_TOKENS_GEN, temperature=0.0)

            # 新增：生成后立即压平三级标题
            toc_text = flatten_three_level(toc_text)

            toc_path.write_text(toc_text + "\n", encoding="utf-8")

            # 本地快速审查
            ok, reason = quick_check(toc_text)
            if not ok:
                extra_fix = reason
                continue

            # 保存后读取，再调用模型审查
            saved = toc_path.read_text(encoding="utf-8", errors="ignore")
            audit_msgs = build_audit_messages(book_name, saved)
            verdict = call_llm(client, audit_msgs, max_tokens=MAX_TOKENS_AUDIT, temperature=0.0)

            if verdict.strip() == "PASS":
                return {"book": book_name, "status": "done", "attempt": attempt}

            if verdict.startswith("FAIL:"):
                extra_fix = verdict[5:].strip()
            else:
                extra_fix = f"审查输出不规范：{verdict[:80]}"

        return {"book": book_name, "status": "failed", "reason": f"max regen reached: {extra_fix or 'unknown'}"}

    except Exception as e:
        return {"book": book_name, "status": "error", "reason": str(e)}

# ======= 主流程 =======
def main():
    if API_KEY in ("", "your_api_key_here", None):
        raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY，避免把 API Key 硬编码在代码里")

    book_dirs = sorted([p for p in INPUT_ROOT.iterdir() if p.is_dir()])

    # 先读取并筛选：只保留未处理且有源文本的目录
    pending = []
    skipped_count = 0

    for book_dir in book_dirs:
        if is_already_processed(book_dir):
            skipped_count += 1
            continue

        source_text = read_source_text(book_dir)
        if not source_text.strip():
            skipped_count += 1
            continue

        pending.append(str(book_dir))

    print(f"总目录数: {len(book_dirs)}")
    print(f"跳过数: {skipped_count}")
    print(f"待处理数: {len(pending)}")

    if not pending:
        print("没有需要处理的目录。")
        return

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_book, book_dir_str) for book_dir_str in pending]

        for future in tqdm(as_completed(futures), total=len(futures), desc="TOC Progress", unit="book", dynamic_ncols=True, file=sys.stdout):
            result = future.result()
            results.append(result)

    done = sum(1 for x in results if x["status"] == "done")
    skipped = sum(1 for x in results if x["status"] == "skipped")
    failed = sum(1 for x in results if x["status"] == "failed")
    error = sum(1 for x in results if x["status"] == "error")

    print("\n===== 处理完成 =====")
    print(f"成功: {done}")
    print(f"跳过: {skipped}")
    print(f"失败: {failed}")
    print(f"异常: {error}")

    if failed or error:
        print("\n===== 失败/异常详情 =====")
        failed_log = INPUT_ROOT.parent / "failed_books.txt"
        with open(failed_log, "w", encoding="utf-8") as f:
            for item in results:
                if item["status"] in ("failed", "error"):
                    path = str(INPUT_ROOT / item["book"])
                    reason = item.get("reason", "")
                    print(f"[{item['status']}] {item['book']}: {reason}")
                    f.write(f"{path}\t{item['status']}\t{reason}\n")
        print(f"\n失败路径已保存至: {failed_log}")

if __name__ == "__main__":
    main()
