"""
parse_toc.py
------------
功能：
  1. 扫描 BASE_DIR 下所有子文件夹，编号后保存 folder_index.csv 到 BASE_DIR。
  2. 对每个子文件夹中的 toc.md，解析目录结构，
     生成含 doc_id / level / seq / title / parent_l1 五列的 toc_table.csv，
     保存在对应子文件夹中。

toc.md 格式约定：
  - 第 1 行：标题（白皮书名称，用作 doc_id 的来源，也可直接用文件夹名）
  - 第 2-4 行：元数据（忽略）
  - 第 5 行起：目录正文
      * 无缩进（或仅 "一、二、三" 这类汉字编号）→ 一级标题 (level=1)
      * 有缩进（2~4 个空格或 \t 开头）        → 二级标题 (level=2)
      * 空行 / "#" 分隔行                       → 跳过
"""

import os
import csv
import re

# ========== 配置 ==========
BASE_DIR = os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy")
TOC_FILENAME = "toc.md"
INDEX_FILENAME = "folder_index.csv"
OUTPUT_FILENAME = "toc_table.csv"
# ==========================


def clean_title(text: str) -> str:
    """去掉前后空白、编号前缀（如 '1.1 '、'一、'、'(一)'）后返回纯标题文字。"""
    text = text.strip()
    # 去掉形如 "1.1 " "2.3.4 " 的数字编号
    text = re.sub(r"^\d+(\.\d+)*[\s\.\、]+", "", text)
    # 去掉形如 "一、" "二、" 的汉字编号
    text = re.sub(r"^[一二三四五六七八九十百]+[、\.\s]+", "", text)
    # 去掉形如 "(一)" "(1)" 的括号编号
    text = re.sub(r"^[\(（][一二三四五六七八九十\d]+[\)）]\s*", "", text)
    return text.strip()


def is_skip_line(line: str) -> bool:
    """判断是否是需要跳过的行（空行、纯 '#' 分隔行等）。"""
    stripped = line.strip()
    if not stripped:
        return True
    if re.fullmatch(r"#+", stripped):
        return True
    return False


def detect_level(line: str) -> int:
    """
    根据行首缩进判断层级：
      - 无缩进  → 1
      - 有缩进  → 2
    """
    if line.startswith((" ", "\t")):
        return 2
    return 1


def parse_toc(filepath: str, doc_id: str) -> list[dict]:
    """解析单个 toc.md，返回行列表（每行是一个字典）。"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 第 5 行（index=4）起是目录正文
    toc_lines = lines[4:]

    rows = []
    seq = 0
    current_l1 = None   # 当前一级标题（用于填 parent_l1）

    for line in toc_lines:
        if is_skip_line(line):
            continue

        level = detect_level(line)
        title = clean_title(line)

        if not title:
            continue

        seq += 1

        if level == 1:
            current_l1 = title
            parent_l1 = ""
        else:
            parent_l1 = current_l1 if current_l1 else ""

        rows.append({
            "doc_id":    doc_id,
            "level":     level,
            "seq":       seq,
            "title":     title,
            "parent_l1": parent_l1,
        })

    return rows


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] 目录不存在: {BASE_DIR}")
        return

    # ── 1. 扫描子文件夹并编号 ──────────────────────────────────────────
    subfolders = sorted([
        name for name in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, name))
    ])

    if not subfolders:
        print("[WARN] 未找到任何子文件夹。")
        return

    index_path = os.path.join(BASE_DIR, INDEX_FILENAME)
    with open(index_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["folder_no", "folder_name"])
        writer.writeheader()
        for i, name in enumerate(subfolders, start=1):
            writer.writerow({"folder_no": i, "folder_name": name})

    print(f"[OK] 文件夹索引已保存 → {index_path}  ({len(subfolders)} 个文件夹)")

    # ── 2. 逐文件夹解析 toc.md ────────────────────────────────────────
    success, skipped = 0, 0

    for folder_no, folder_name in enumerate(subfolders, start=1):
        folder_path = os.path.join(BASE_DIR, folder_name)
        toc_path    = os.path.join(folder_path, TOC_FILENAME)

        if not os.path.isfile(toc_path):
            print(f"[SKIP] 未找到 toc.md: {folder_path}")
            skipped += 1
            continue

        # doc_id 使用 folder_index.csv 中的编号
        doc_id = str(folder_no)

        try:
            rows = parse_toc(toc_path, doc_id)
        except Exception as e:
            print(f"[ERROR] 解析失败 {toc_path}: {e}")
            skipped += 1
            continue

        if not rows:
            print(f"[WARN] 解析结果为空: {toc_path}")
            skipped += 1
            continue

        out_path = os.path.join(folder_path, OUTPUT_FILENAME)
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["doc_id", "level", "seq", "title", "parent_l1"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"[OK] {folder_name}  →  {len(rows)} 条记录  →  {out_path}")
        success += 1

    print(f"\n完成：{success} 个成功，{skipped} 个跳过。")


if __name__ == "__main__":
    main()
