"""
analyze_category.py
-------------------
针对 BASE_DIR 下所有子文件夹的 toc_table.csv，统计三项指标：

A. 出现率  —— 每个大类在多少比例的书里出现过（每本书只算一次）
B. 首次出现位置  —— 每个大类在每本书中【所有出现位置】的相对位置均值，
                   再汇总成跨书的中位数 / 均值
C. 覆盖深度  —— 每本书该大类下的 L2 数量，汇总均值 / P50 / P90

结果保存到 BASE_DIR/category_stats.csv
"""

import os
import csv
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib import rcParams

rcParams["axes.unicode_minus"] = False

# 直接用字体文件路径，绕开系统字体缓存，确保中文正常显示
def _find_cjk_font() -> str:
    """自动查找可用的中文字体文件路径，找不到返回 None。"""
    candidates = [
        # matplotlib 自带
        "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/NotoSansCJKsc.ttf",
        "/usr/local/lib/python3.11/dist-packages/matplotlib/mpl-data/fonts/ttf/NotoSansCJKsc.ttf",
        "/usr/local/lib/python3.10/dist-packages/matplotlib/mpl-data/fonts/ttf/NotoSansCJKsc.ttf",
        # 系统 Noto
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto/NotoSansCJKsc-Regular.otf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        # Windows（WSL）
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simsun.ttc",
    ]
    # 也从 matplotlib 自带目录里搜
    import glob, os
    mpl_font_dir = os.path.join(os.path.dirname(_fm.__file__), "mpl-data", "fonts", "ttf")
    candidates += glob.glob(os.path.join(mpl_font_dir, "*CJK*.ttf"))
    candidates += glob.glob(os.path.join(mpl_font_dir, "*CJK*.otf"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

_FONT_PATH = _find_cjk_font()
if _FONT_PATH:
    _FP   = _fm.FontProperties(fname=_FONT_PATH)
    _FP_B = _fm.FontProperties(fname=_FONT_PATH, size=13, weight="bold")
else:
    # 找不到中文字体，回退到默认（图表会显示方块但不报错）
    print("[WARN] 未找到中文字体，图表标签可能显示为方块。")
    _FP   = _fm.FontProperties()
    _FP_B = _fm.FontProperties(size=13, weight="bold")


def _apply_fp(ax, fp=None, fp_title=None):
    """统一把字体属性应用到坐标轴上所有文字元素。"""
    fp = fp or _FP
    fp_t = fp_title or _FP_B
    if ax.title.get_text():
        ax.title.set_fontproperties(fp_t)
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_fontproperties(fp)
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_fontproperties(fp)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)

# ========== 配置 ==========
BASE_DIR   = os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy")
TOC_CSV    = "toc_table.csv"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "category_stats.csv")
# ==========================

CATEGORIES = [
    "概述", "背景", "方案与目标", "架构设计", "方法原理", "场景",
    "实现", "评测与实验", "安全与合规", "结论与展望", "附录",
]


# ──────────────────────────────────────────────────────────────────────────────
# 算法说明
# ──────────────────────────────────────────────────────────────────────────────
# 数据结构（每本书独立处理）：
#   rows = 该书所有行，每行含 seq / level / title / category
#   total_rows = 该书总标题数（L1 + L2 都算，seq 的最大值）
#
# A. 出现率
#   - 对每本书，收集"出现过的大类集合"（去重）
#   - 出现率 = 该大类出现的书数 / 总书数
#   - 例：背景在同一本书出现 3 次 → 该书只贡献 1 次计数
#
# B. 首次出现位置（相对位置）
#   - 对每本书、每个大类，找出【所有 L1 行】的 seq 列表
#   - 相对位置 = seq / total_rows（0~1 之间，越大越靠后）
#   - 该书该大类的"代表位置" = 所有出现位置的均值
#     （例如背景出现在 seq=2 和 seq=8，共 10 行 → 位置均值 = (0.2+0.8)/2 = 0.5）
#   - 跨书汇总：收集所有书的代表位置，计算中位数 & 均值
#
# C. 覆盖深度（L2 数量）
#   - 对每本书、每个大类，统计 level=2 且 category=该大类 的行数
#   - 跨书汇总（只统计"出现过该大类"的书）：均值 / P50 / P90
# ──────────────────────────────────────────────────────────────────────────────


def percentile(data: list[float], p: int) -> float:
    """计算百分位数（线性插值）。"""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    subfolders = sorted([
        n for n in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, n))
    ])

    total_books = 0  # 成功读取的书总数

    # 跨书累计容器
    # A: {cat: set of book_ids that have this cat}
    cat_book_set: dict[str, set] = defaultdict(set)

    # B: {cat: [per-book 代表相对位置, ...]}
    cat_positions: dict[str, list[float]] = defaultdict(list)

    # C: {cat: [per-book L2数量, ...]}  仅出现过该大类的书
    cat_l2_counts: dict[str, list[int]] = defaultdict(list)

    for folder_name in subfolders:
        csv_path = os.path.join(BASE_DIR, folder_name, TOC_CSV)
        if not os.path.isfile(csv_path):
            continue

        try:
            rows = load_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] 读取失败 {csv_path}: {e}")
            continue

        if not rows:
            continue

        book_id = folder_name
        total_books += 1

        # 该书总标题数（用 seq 最大值）
        try:
            total_rows = max(int(r["seq"]) for r in rows)
        except Exception:
            continue

        if total_rows == 0:
            continue

        # ── 按大类分组 ────────────────────────────────────────────────
        # cat → L1行列表, L2行列表
        cat_l1_rows: dict[str, list[dict]] = defaultdict(list)
        cat_l2_rows: dict[str, list[dict]] = defaultdict(list)

        for row in rows:
            cat = row.get("category", "").strip()
            if not cat:
                continue
            lvl = str(row.get("level", "")).strip()
            if lvl == "1":
                cat_l1_rows[cat].append(row)
            elif lvl == "2":
                cat_l2_rows[cat].append(row)

        # ── A & B & C ─────────────────────────────────────────────────
        appeared_cats = set(cat_l1_rows.keys())

        for cat in appeared_cats:
            # A: 标记该书出现了此大类
            cat_book_set[cat].add(book_id)

            # B: 所有 L1 出现位置的均值作为该书代表位置
            seqs = [int(r["seq"]) for r in cat_l1_rows[cat]]
            rel_positions = [s / total_rows for s in seqs]
            book_rep_pos = sum(rel_positions) / len(rel_positions)
            cat_positions[cat].append(book_rep_pos)

            # C: 该大类下 L2 数量
            cat_l2_counts[cat].append(len(cat_l2_rows[cat]))

    if total_books == 0:
        print("[ERROR] 未读取到任何有效书籍。")
        return

    print(f"共读取 {total_books} 本书，正在汇总…")

    # ── 汇总输出 ──────────────────────────────────────────────────────
    result_rows = []
    for cat in CATEGORIES:
        # A
        book_count   = len(cat_book_set.get(cat, set()))
        appear_rate  = book_count / total_books

        # B
        positions = cat_positions.get(cat, [])
        pos_median = percentile(positions, 50) if positions else None
        pos_mean   = (sum(positions) / len(positions)) if positions else None

        # C
        l2_counts = cat_l2_counts.get(cat, [])
        l2_mean   = (sum(l2_counts) / len(l2_counts)) if l2_counts else None
        l2_p50    = percentile(l2_counts, 50) if l2_counts else None
        l2_p90    = percentile(l2_counts, 90) if l2_counts else None

        def fmt(v, digits=4):
            return round(v, digits) if v is not None else ""

        result_rows.append({
            "category":          cat,
            "A_book_count":      book_count,
            "A_appear_rate":     fmt(appear_rate, 4),
            "B_pos_median":      fmt(pos_median, 4),
            "C_l2_mean":         fmt(l2_mean, 2),
        })

    fieldnames = [
        "category",
        "A_book_count", "A_appear_rate",
        "B_pos_median",
        "C_l2_mean",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"\n结果已保存 → {OUTPUT_CSV}\n")

    # 控制台预览
    col_w   = [14, 12, 14, 14, 12]
    headers = ["category", "书数(A)", "出现率(A)", "位置中位(B)", "L2均值(C)"]
    sep = "  "
    header_line = sep.join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for r in result_rows:
        vals = [
            r["category"],
            str(r["A_book_count"]),
            f"{float(r['A_appear_rate']):.1%}" if r["A_appear_rate"] != "" else "-",
            str(r["B_pos_median"]),
            str(r["C_l2_mean"]),
        ]
        print(sep.join(v.ljust(col_w[i]) for i, v in enumerate(vals)))

    # ── 画图 ──
    draw_charts(result_rows, total_books, os.path.dirname(__file__))


def draw_charts(result_rows: list[dict], total_books: int, out_dir: str):
    """生成三张图并保存到 out_dir。"""
    cats        = [r["category"]      for r in result_rows]
    book_counts = [r["A_book_count"]  for r in result_rows]
    appear_rate = [float(r["A_appear_rate"]) if r["A_appear_rate"] != "" else 0
                   for r in result_rows]
    pos_median  = [float(r["B_pos_median"]) if r["B_pos_median"] != "" else 0
                   for r in result_rows]
    l2_mean     = [float(r["C_l2_mean"]) if r["C_l2_mean"] != "" else 0
                   for r in result_rows]

    BAR_COLOR  = "#4C72B0"
    LINE_COLOR = "#DD8452"
    POS_CMAP   = plt.cm.RdYlGn_r

    # ── 图A：出现书数（柱）+ 出现率（折线）双轴 ─────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(len(cats))
    ax1.bar(x, book_counts, color=BAR_COLOR, alpha=0.8, label="出现书数")
    ax1.set_ylabel("出现书数", color=BAR_COLOR, fontproperties=_FP)
    ax1.tick_params(axis="y", labelcolor=BAR_COLOR)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(cats, rotation=25, ha="right", fontproperties=_FP)
    ax1.set_ylim(0, total_books * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(list(x), [v * 100 for v in appear_rate],
             color=LINE_COLOR, marker="o", linewidth=2, label="出现率(%)")
    ax2.set_ylabel("出现率 (%)", color=LINE_COLOR, fontproperties=_FP)
    ax2.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.set_ylim(0, 115)

    for i, (b, r) in enumerate(zip(book_counts, appear_rate)):
        ax1.text(i, b + total_books * 0.01, f"{r:.0%}",
                 ha="center", va="bottom", fontsize=8.5, color=LINE_COLOR)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    for t in leg.get_texts():
        t.set_fontproperties(_FP)
    ax1.set_title("A · 各大类出现率", fontproperties=_FP_B, pad=10)
    fig.tight_layout()
    path_a = os.path.join(out_dir, "chart_A_appear_rate.png")
    fig.savefig(path_a, dpi=150)
    plt.close(fig)
    print(f"[图A] 已保存 → {path_a}")

    # ── 图B：水平甘特风格 ────────────────────────────────────────────
    sorted_b = sorted(zip(pos_median, cats), key=lambda t: t[0], reverse=True)
    b_pos  = [t[0] for t in sorted_b]
    b_cats = [t[1] for t in sorted_b]
    colors = [POS_CMAP(v) for v in b_pos]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(b_cats))
    ax.barh(list(y), [1] * len(b_cats), color="#EEEEEE", height=0.6)
    ax.barh(list(y), b_pos, color=colors, height=0.6, alpha=0.9)
    for i, v in enumerate(b_pos):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlim(0, 1.12)
    ax.set_yticks(list(y))
    ax.set_yticklabels(b_cats, fontproperties=_FP)
    ax.set_xlabel("相对位置（0 = 目录最前，1 = 目录最末）", fontproperties=_FP)
    ax.set_title("B · 各大类在目录中的典型位置（中位数）", fontproperties=_FP_B, pad=10)
    sm = plt.cm.ScalarMappable(cmap=POS_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("相对位置", fontproperties=_FP)
    fig.tight_layout()
    path_b = os.path.join(out_dir, "chart_B_position.png")
    fig.savefig(path_b, dpi=150)
    plt.close(fig)
    print(f"[图B] 已保存 → {path_b}")

    # ── 图C：水平条形图，覆盖深度 ────────────────────────────────────
    sorted_c = sorted(zip(l2_mean, cats), key=lambda t: t[0], reverse=True)
    c_vals = [t[0] for t in sorted_c]
    c_cats = [t[1] for t in sorted_c]
    max_v  = max(c_vals) if max(c_vals) > 0 else 1
    c_colors = [plt.cm.Blues(0.35 + 0.55 * (v / max_v)) for v in c_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(c_cats))
    ax.barh(list(y), c_vals, color=c_colors, height=0.6, alpha=0.9)
    for i, v in enumerate(c_vals):
        ax.text(v + max_v * 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(c_cats, fontproperties=_FP)
    ax.set_xlabel("平均 L2 子章节数量", fontproperties=_FP)
    ax.set_title("C · 各大类覆盖深度（L2 均值）", fontproperties=_FP_B, pad=10)
    ax.set_xlim(0, max_v * 1.18)
    fig.tight_layout()
    path_c = os.path.join(out_dir, "chart_C_depth.png")
    fig.savefig(path_c, dpi=150)
    plt.close(fig)
    print(f"[图C] 已保存 → {path_c}")


if __name__ == "__main__":
    main()
