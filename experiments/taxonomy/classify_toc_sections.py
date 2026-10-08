"""
add_category.py
---------------
功能：
  遍历 BASE_DIR 下所有子文件夹中的 toc_table.csv，
  用大模型对每本白皮书的一级标题做分类，
  将分类结果写入新列 category，并覆盖保存原文件。

特性：
  - 5 线程并发处理
  - 校验失败自动重试（最多3次），第2次起开启深度思考模式
  - 连接超时自动重试（最多3次），指数退避
  - 所有失败记录写入 failed_log.csv
"""

import os
import csv
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI

# ========== 配置 ==========
BASE_DIR    = os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy")
TOC_CSV     = "toc_table.csv"
FAILED_LOG  = os.path.join(BASE_DIR, "failed_log.csv")
API_KEY     = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL    = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL       = os.getenv("TWPGEN_TAXONOMY_MODEL", "qwen3-max")
MAX_WORKERS = 5      # 并发线程数
MAX_RETRIES = 3      # 最大重试次数（校验失败 & 超时各自独立计数，但总轮次共用）
RETRY_DELAY = 2      # 重试基础间隔（秒），指数退避
TIMEOUT     = 60     # 单次请求超时（秒）
# ==========================

CATEGORIES = [
    "概述",
    "背景",
    "方案与目标",
    "架构设计",
    "方法原理",
    "场景",
    "实现",
    "评测与实验",
    "安全与合规",
    "结论与展望",
    "附录",
]

SYSTEM_PROMPT = """\
你是一个技术白皮书目录分类专家。
你的任务是：将给定的白皮书一级标题列表，逐一归类到下面 11 个大类之一。

大类及其包含的典型关键词（仅供参考，请综合理解语义）：
1. 概述         —— 摘要、引言、前言、简介、总览、Executive Summary、文档结构
2. 背景         —— 背景、现状、挑战、痛点、需求、动机、问题定义
3. 方案与目标   —— 方案、总体方案、方案概述、目标、范围、设计原则、约束、关键特性
4. 架构设计     —— 架构、总体架构、技术架构、系统设计、模块设计、组件、分层、数据流、流程设计
5. 方法原理     —— 原理、方法、算法、模型、机制、协议、关键技术、训练、推理、公式推导
6. 场景         —— 场景、应用场景、使用场景、业务场景、典型场景、适用场景、落地场景、应用案例
7. 实现         —— 实现、工程实现、接口、API、集成、部署、运维、配置、监控、日志、可扩展性、性能优化
8. 评测与实验   —— 评测、测试、实验、性能、对比、Benchmark、指标、实验设置、结果分析、案例验证
9. 安全与合规   —— 安全、隐私、合规、风险、威胁模型、攻击与防护、权限控制、加密、审计、治理
10. 结论与展望   —— 总结、结论、讨论、局限、展望、未来工作
11. 附录        —— 参考文献、引用、附录、术语、名词解释、缩写表、致谢

输出要求（严格遵守）：
- 只输出一个合法的 JSON 对象，不要有任何多余文字、markdown 代码块、解释或思考过程。
- JSON 格式：{"seq编号": "大类名称", ...}
- seq编号 与输入一一对应，大类名称必须是上面 11 个大类之一的完整名称。

示例输入：
{"1": "发展背景", "2": "IPv6+ 概念", "6": "IPv6+ 技术体系", "11": "应用领域", "18": "IP产业代际和愿景", "19": "总结与展望"}

示例输出：
{"1": "背景", "2": "概述", "6": "架构设计", "11": "背景", "18": "结论与展望", "19": "结论与展望"}
"""

# 线程安全锁
_log_lock      = threading.Lock()
_progress_lock = threading.Lock()
failed_records = []
_done_count    = 0


def _update_progress(total: int, stats: dict):
    global _done_count
    with _progress_lock:
        _done_count += 1
        done   = _done_count
        bar_w  = 40
        filled = int(bar_w * done / total)
        bar    = "█" * filled + "░" * (bar_w - filled)
        print(
            f"\r进度 [{bar}] {done}/{total}  "
            f"✅{stats['success']} ❌{stats['failed']} ⚠{stats['error']}",
            end="", flush=True,
        )


def safe_print(msg: str):
    pass  # 静默，只保留进度条


def record_failure(folder_name: str, reason: str, detail: str = ""):
    with _log_lock:
        failed_records.append({
            "folder_name": folder_name,
            "reason":      reason,
            "detail":      detail[:300],
            "time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


def call_api_once(seq_title_map: dict, enable_thinking: bool) -> dict:
    """
    单次 API 调用，含超时设置。
    成功返回 {seq: category}；失败抛出异常。
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=TIMEOUT,
    )
    user_content = json.dumps(seq_title_map, ensure_ascii=False)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        extra_body={"enable_thinking": enable_thinking},
        stream=False,
    )
    raw = completion.choices[0].message.content.strip()

    # 去掉偶发的 markdown 代码块包裹
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def classify_titles(seq_title_map: dict, folder_name: str) -> dict | None:
    """
    带完整重试逻辑的分类函数：

    每轮尝试流程：
      1. 发起 API 请求
         - 若超时/网络异常 → 记录超时次数，等待后重试（最多 MAX_RETRIES 轮）
         - 若成功返回 → 进入校验
      2. 校验大类合法性
         - 若存在非法大类 → 记录校验失败次数，等待后重试（下次开启深度思考）
         - 若全部合法 → 返回结果

    第 1 次：关闭深度思考（快）
    第 2、3 次：开启深度思考（准）
    三次全部失败 → 返回 None，由调用方记录日志
    """
    for attempt in range(1, MAX_RETRIES + 1):
        enable_thinking = (attempt > 1)
        mode_tag = "🧠深度思考" if enable_thinking else "⚡普通"
        safe_print(f"  [{folder_name}] 第 {attempt}/{MAX_RETRIES} 次 {mode_tag}…")

        # ── Step 1: 请求 API ────────────────────────────────────────
        try:
            result = call_api_once(seq_title_map, enable_thinking)
        except Exception as e:
            err_str = str(e)
            is_timeout = any(k in err_str.lower() for k in (
                "timeout", "timed out", "connection", "network",
                "read error", "connect error", "remote disconnected",
            ))
            err_type = "超时/连接异常" if is_timeout else "API异常"
            safe_print(f"  [{folder_name}] ❌ {err_type}（第{attempt}次）: {err_str[:120]}")

            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                safe_print(f"  [{folder_name}] ⏳ {wait}s 后重试…")
                time.sleep(wait)
            else:
                record_failure(folder_name, f"三次全部{err_type}", err_str[:300])
            continue

        # ── Step 2: 校验大类 ────────────────────────────────────────
        invalid_items = {
            seq: cat for seq, cat in result.items()
            if cat not in CATEGORIES
        }

        if invalid_items:
            safe_print(
                f"  [{folder_name}] ⚠ 校验失败（第{attempt}次），"
                f"非法大类: {invalid_items}"
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                safe_print(f"  [{folder_name}] ⏳ {wait}s 后重试（将开启深度思考）…")
                time.sleep(wait)
            else:
                record_failure(
                    folder_name,
                    "三次校验全部失败",
                    f"最后一次非法大类: {invalid_items}",
                )
            continue

        # ── 成功 ────────────────────────────────────────────────────
        safe_print(f"  [{folder_name}] ✅ 分类成功（第 {attempt} 次）")
        return result

    return None  # 三次全部失败


def process_folder(folder_info: tuple) -> str:
    """处理单个子文件夹，供线程池调用。返回状态字符串。"""
    idx, total, folder_name = folder_info
    csv_path = os.path.join(BASE_DIR, folder_name, TOC_CSV)
    tag = f"[{idx}/{total}] {folder_name}"

    safe_print(f"\n{tag} ─ 开始")

    # ── 文件检查 ────────────────────────────────────────────────────
    if not os.path.isfile(csv_path):
        safe_print(f"{tag} [SKIP] 无 {TOC_CSV}")
        record_failure(folder_name, "文件缺失", csv_path)
        return "skip"

    # ── 读取 CSV ────────────────────────────────────────────────────
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        safe_print(f"{tag} [ERROR] 读取失败: {e}")
        record_failure(folder_name, "读取CSV失败", str(e))
        return "error"

    if not rows:
        safe_print(f"{tag} [SKIP] 空文件")
        record_failure(folder_name, "CSV为空", "")
        return "skip"

    # ── 收集一级标题 ────────────────────────────────────────────────
    l1_map = {
        str(row["seq"]): row["title"]
        for row in rows
        if str(row["level"]) == "1"
    }
    if not l1_map:
        safe_print(f"{tag} [SKIP] 无一级标题")
        record_failure(folder_name, "无一级标题", "")
        return "skip"

    safe_print(f"{tag} → {len(l1_map)} 个一级标题待分类")

    # ── 调用大模型 ──────────────────────────────────────────────────
    seq_to_category = classify_titles(l1_map, folder_name)

    if seq_to_category is None:
        safe_print(f"{tag} ❌ 三次全部失败，已记录到日志")
        return "failed"

    # ── 写入 category 列（顺序单遍历，L2 直接继承当前 L1 的分类）────
    fieldnames = list(rows[0].keys())
    if "category" not in fieldnames:
        fieldnames.append("category")

    current_cat = "未分类"
    for row in rows:
        lvl = str(row.get("level", "")).strip()
        if lvl == "1":
            current_cat = seq_to_category.get(str(row["seq"]), "未分类")
            row["category"] = current_cat
        elif lvl == "2":
            row["category"] = current_cat  # 直接跟着上一个 L1 走
        else:
            row["category"] = "未分类"

    # ── 写回 CSV ────────────────────────────────────────────────────
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        safe_print(f"{tag} ✅ 完成")
        return "success"
    except Exception as e:
        safe_print(f"{tag} [ERROR] 写入失败: {e}")
        record_failure(folder_name, "写入CSV失败", str(e))
        return "error"


def write_failed_log():
    if not failed_records:
        print("\n✅ 无任何失败记录。")
        return
    with open(FAILED_LOG, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["folder_name", "reason", "detail", "time"]
        )
        writer.writeheader()
        writer.writerows(failed_records)
    print(f"\n⚠  失败记录（{len(failed_records)} 条）→ {FAILED_LOG}")


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] 目录不存在: {BASE_DIR}")
        return

    subfolders = sorted([
        n for n in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, n))
    ])
    if not subfolders:
        print("[WARN] 未找到任何子文件夹。")
        return

    total = len(subfolders)
    print(f"共 {total} 个子文件夹，{MAX_WORKERS} 线程并发处理\n" + "=" * 55)

    tasks  = [(i, total, name) for i, name in enumerate(subfolders, start=1)]
    stats  = {"success": 0, "skip": 0, "failed": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_folder, t): t for t in tasks}
        for future in as_completed(futures):
            status = future.result()
            stats[status] = stats.get(status, 0) + 1
            _update_progress(total, stats)

    print("\n" + "=" * 55)
    print(
        f"✅ 成功: {stats['success']}  "
        f"⏭  跳过: {stats['skip']}  "
        f"❌ 失败: {stats['failed']}  "
        f"⚠  错误: {stats['error']}"
    )
    print("=" * 55)

    write_failed_log()


if __name__ == "__main__":
    main()
