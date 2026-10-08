import os
import re
import sys
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = Path(os.getenv("TWPGEN_TAXONOMY_OCR_DIR", "data/ocr_output"))
OUTPUT_ROOT = Path(os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy"))

# ====== 判定规则：一行同时包含“字”和“数字” ======
RE_HAS_DIGIT = re.compile(r"\d")
RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")     # 中文
RE_HAS_ALPHA = re.compile(r"[A-Za-z]")         # 英文（可选）

def line_has_text_and_number(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    has_digit = bool(RE_HAS_DIGIT.search(line))
    has_text = bool(RE_HAS_CJK.search(line)) or bool(RE_HAS_ALPHA.search(line))
    return has_digit and has_text

def list_md_files(folder: Path):
    mds = sorted(folder.glob("*.md"))
    # 排除你可能之前生成过的文件
    mds = [p for p in mds if p.name not in ("combined.md", "toc.md") and not p.name.endswith("_toc.md")]
    return mds

def read_lines(p: Path):
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()

def find_first_dir_index(mds):
    """返回第一个包含'目录'的md下标；没有则返回 None"""
    for i, p in enumerate(mds):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "目录" in txt:
            return i
    return None

def extract_lines_from_files(files):
    out = []
    for p in files:
        for ln in read_lines(p):
            if line_has_text_and_number(ln):
                out.append(ln.strip())
    return out

def write_result(out_dir: Path, lines, picked_files, mode_desc: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "toc_lines.md"

    # 简单去重（保序）
    seen = set()
    uniq = []
    for x in lines:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    # 写入
    header = []
    header.append(f"# toc_lines")
    header.append("")
    header.append(f"- mode: {mode_desc}")
    header.append(f"- files_used: {len(picked_files)}")
    for p in picked_files:
        header.append(f"  - {p.name}")
    header.append("")
    header.append("----")
    header.append("")

    out_path.write_text("\n".join(header + uniq) + "\n", encoding="utf-8")

def process_one_folder(folder: Path):
    mds = list_md_files(folder)
    if not mds:
        return

    idx = find_first_dir_index(mds)
    if idx is not None:
        # 命中“目录”：当前md + 后3个 = 共4个（越界就取到末尾）
        picked = mds[idx: idx + 4]
        lines = extract_lines_from_files(picked)
        mode = "matched '目录' -> current md + next 3 md (total 4)"
    else:
        # 没命中：全量md
        picked = mds
        lines = extract_lines_from_files(picked)
        mode = "no '目录' found -> all md files"

    out_dir = OUTPUT_ROOT / folder.name
    write_result(out_dir, lines, picked, mode)

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    folders = sorted([p for p in INPUT_ROOT.iterdir() if p.is_dir()])

    for folder in tqdm(folders, desc="Extract TOC Lines", unit="folder", dynamic_ncols=True, file=sys.stdout):
        process_one_folder(folder)

if __name__ == "__main__":
    main()
