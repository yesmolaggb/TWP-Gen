"""
analyze_sequence.py
-------------------
C1: 每本书生成模块顺序 module_seq（只用 L1，按 seq 排序后拼接）
C2: 压扁连续重复模块 → module_seq_compact
C3: Top 10 最常见完整目录套路
C4: 相邻模块转移 Top 统计（"架构设计后面最常接什么"）

输出文件（均保存到 BASE_DIR）：
  book_sequences.csv      —— 每本书一行，含 C1/C2 结果
  top_templates.csv       —— C3 Top 10 套路
  top_transitions.csv     —— C4 相邻转移 Top
"""

import os
import csv
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
from matplotlib import rcParams

rcParams["axes.unicode_minus"] = False

def _find_cjk_font() -> str:
    import glob
    candidates = [
        "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/NotoSansCJKsc.ttf",
        "/usr/local/lib/python3.11/dist-packages/matplotlib/mpl-data/fonts/ttf/NotoSansCJKsc.ttf",
        "/usr/local/lib/python3.10/dist-packages/matplotlib/mpl-data/fonts/ttf/NotoSansCJKsc.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
    ]
    mpl_dir = os.path.join(os.path.dirname(_fm.__file__), "mpl-data", "fonts", "ttf")
    candidates += glob.glob(os.path.join(mpl_dir, "*CJK*.ttf"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

_FONT_PATH = _find_cjk_font()
if _FONT_PATH:
    _FP   = _fm.FontProperties(fname=_FONT_PATH, size=10)
    _FP_S = _fm.FontProperties(fname=_FONT_PATH, size=8.5)
    _FP_B = _fm.FontProperties(fname=_FONT_PATH, size=13, weight="bold")
else:
    print("[WARN] 未找到中文字体")
    _FP = _FP_S = _fm.FontProperties()
    _FP_B = _fm.FontProperties(size=13, weight="bold")

# ========== 配置 ==========
BASE_DIR   = os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy")
TOC_CSV    = "toc_table.csv"
OUT_DIR    = os.path.dirname(__file__)
SEQ_OUT    = os.path.join(OUT_DIR, "book_sequences.csv")
TPL_OUT    = os.path.join(OUT_DIR, "top_templates.csv")
TRANS_OUT  = os.path.join(OUT_DIR, "top_transitions.csv")
TOP_N      = 10   # 仅用于图，CSV 和控制台输出全部
# ==========================


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compact(seq: list[str]) -> list[str]:
    """C2：去重，每个大类在整本书内只保留第一次出现（不限相邻）。"""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main():
    subfolders = sorted([
        n for n in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, n))
    ])

    # ── C1 / C2：逐书构建模块序列 ────────────────────────────────────
    book_rows   = []   # 最终写入 book_sequences.csv
    all_compact    = []   # 每本书的去重序列，供 C4 用（含附录）
    all_compact_c3 = []   # 供 C3 用（去附录）

    total_books = 0
    for folder_name in subfolders:
        csv_path = os.path.join(BASE_DIR, folder_name, TOC_CSV)
        if not os.path.isfile(csv_path):
            continue
        try:
            rows = load_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] {folder_name}: {e}")
            continue
        if not rows:
            continue

        # 只取 L1 行，按 seq 升序
        l1_rows = sorted(
            [r for r in rows if str(r.get("level", "")).strip() == "1"],
            key=lambda r: int(r.get("seq", 0)),
        )

        # 过滤掉没有 category 的行
        cats = [r.get("category", "").strip() for r in l1_rows]
        cats = [c for c in cats if c]

        if not cats:
            continue

        total_books += 1
        doc_id = rows[0].get("doc_id", folder_name)

        # C1：原始序列
        module_seq = " → ".join(cats)

        # C2：去重（全局），保留附录供 C4 相邻统计用
        cats_compact = compact(cats)
        module_seq_compact = " → ".join(cats_compact)

        # 去附录版本（C3 统计 & book_sequences 主列用）
        cats_for_c3 = [c for c in cats_compact if c != "附录"]
        module_seq_compact_no_appendix = " → ".join(cats_for_c3)

        book_rows.append({
            "doc_id":                    doc_id,
            "folder_name":               folder_name,
            "module_seq":                module_seq,
            "module_seq_compact":        module_seq_compact_no_appendix,  # 去附录，与 C3 模板对齐
            "module_seq_compact_full":   module_seq_compact,              # 含附录（供参考）
        })
        all_compact.append(cats_compact)          # C4 用（含附录）
        all_compact_c3.append(cats_for_c3)        # C3 用（不含附录）

    print(f"共读取 {total_books} 本书")

    # ── 写 C1/C2 结果 ────────────────────────────────────────────────
    with open(SEQ_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc_id", "folder_name", "module_seq", "module_seq_compact", "module_seq_compact_full"]
        )
        writer.writeheader()
        writer.writerows(book_rows)
    print(f"[C1/C2] book_sequences.csv → {SEQ_OUT}")

    # ── C3：Top 模板（完整压扁序列计数）────────────────────────────
    template_counter: Counter = Counter()
    for cats_for_c3 in all_compact_c3:
        key = " → ".join(cats_for_c3)
        template_counter[key] += 1

    top_templates = template_counter.most_common()   # 全部输出，不截断
    tpl_rows = []
    for rank, (template, count) in enumerate(top_templates, start=1):
        pct = count / total_books if total_books else 0
        tpl_rows.append({
            "rank":      rank,
            "count":     count,
            "pct":       f"{pct:.1%}",
            "template":  template,
        })

    with open(TPL_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "count", "pct", "template"])
        writer.writeheader()
        writer.writerows(tpl_rows)
    print(f"[C3]    top_templates.csv  → {TPL_OUT}")

    # ── C4：相邻转移统计 ─────────────────────────────────────────────
    # 统计每个"from"节点的总出度（用于算占比）
    from_total: Counter = Counter()
    pair_counter: Counter = Counter()

    for cats_compact in all_compact:
        for i in range(len(cats_compact) - 1):
            src, dst = cats_compact[i], cats_compact[i + 1]
            pair_counter[(src, dst)] += 1
            from_total[src] += 1

    top_pairs = pair_counter.most_common()
    trans_rows = []
    for rank, ((src, dst), count) in enumerate(top_pairs, start=1):
        # 条件占比 = 该对出现次数 / src 的总出度
        cond_pct = count / from_total[src] if from_total[src] else 0

        trans_rows.append({
            "rank":      rank,
            "from":      src,
            "to":        dst,
            "count":     count,
            "cond_pct":  f"{cond_pct:.4f}",
        })

    with open(TRANS_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "from", "to", "count", "cond_pct"]
        )
        writer.writeheader()
        writer.writerows(trans_rows)
    print(f"[C4]    top_transitions.csv → {TRANS_OUT}")

    # ── 控制台预览 ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("C3: 模板套路（全部）")
    print("-" * 80)
    for r in tpl_rows:
        print(f"  #{r['rank']:3d}  {r['pct']:>6}  ({r['count']:>4}本)  {r['template']}")

    print("\n" + "=" * 80)
    print("C4: 相邻转移（全部）")
    print("-" * 80)
    print(f"  {'#':>3}  {'from':^14} → {'to':^14}  {'次数':>6}  {'全局占比':>10}  {'条件占比':>10}")
    print("  " + "-" * 70)
    for r in trans_rows:
        print(
            f"  #{r['rank']:3d}  {r['from']:^14} → {r['to']:^14}"
            f"  {r['count']:>6}  {r['global_pct']:>10}  {r['cond_pct']:>10}"
        )

    print("\n完成。统计文件已保存到:")
    print(f"  - {SEQ_OUT}")
    print(f"  - {TPL_OUT}")
    print(f"  - {TRANS_OUT}")

    # ── 画图 ──────────────────────────────────────────────────────────
    draw_c3(tpl_rows, total_books, OUT_DIR)
    draw_c4_flow(all_compact, OUT_DIR)



def draw_c3(tpl_rows: list[dict], total_books: int, out_dir: str):
    """C3：Top10 模板水平柱状图，数据与 top_templates.csv 前10行完全对应。"""
    rows10 = tpl_rows[:10]
    if not rows10:
        return

    labels = [r["template"] for r in rows10][::-1]
    counts = [r["count"]    for r in rows10][::-1]
    pcts   = [float(r["pct"].rstrip("%")) for r in rows10][::-1]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(16, max(5, n * 0.75 + 1.5)))
    max_c   = max(counts) if counts else 1
    colors  = [plt.cm.Blues(0.4 + 0.5 * (c / max_c)) for c in counts]
    ax.barh(range(n), counts, color=colors, height=0.6, alpha=0.9)

    for i, (c, p) in enumerate(zip(counts, pcts)):
        ax.text(c + max_c * 0.005, i, f"{c}本  ({p:.1f}%)",
                va="center", fontproperties=_FP_S)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontproperties=_FP_S)
    ax.set_xlabel("出现书数", fontproperties=_FP)
    ax.set_title("C3 · Top10 目录套路（已去除附录）", fontproperties=_FP_B, pad=10)
    ax.set_xlim(0, max_c * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "chart_C3_templates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图C3] 已保存 → {path}")


def draw_c4_flow(all_compact: list[list[str]], out_dir: str):
    """
    C4：模块流向网络图。
    节点两列排布，箭头粗细+颜色=转移频率，每节点只画 Top3 出边，避免杂乱。
    """
    import math
    if not all_compact:
        return

    # ── 统计转移 ──────────────────────────────────────────────────────
    pair_counter: Counter = Counter()
    from_total:   Counter = Counter()
    node_order: dict[str, int] = {}
    for cats in all_compact:
        for c in cats:
            if c not in node_order:
                node_order[c] = len(node_order)
        for i in range(len(cats) - 1):
            pair_counter[(cats[i], cats[i+1])] += 1
            from_total[cats[i]] += 1

    node_list = sorted(node_order.keys(), key=lambda c: node_order[c])
    n = len(node_list)

    # ── 节点位置：两列，按顺序从上到下交错 ───────────────────────────
    NODE_R = 0.30
    GAP_Y  = 1.15
    pos: dict[str, tuple] = {}
    for i, node in enumerate(node_list):
        col = i % 2
        row = i // 2
        pos[node] = (col * 3.2, -(row * GAP_Y))

    # ── 每个 from 节点取 Top3 出边 ────────────────────────────────────
    from_grouped: dict[str, list] = {}
    for (src, dst), cnt in pair_counter.items():
        from_grouped.setdefault(src, []).append((cnt, dst))
    top_edges = []
    for src, pairs in from_grouped.items():
        for cnt, dst in sorted(pairs, reverse=True)[:3]:
            top_edges.append((src, dst, cnt))

    max_cnt = max(c for _, _, c in top_edges) if top_edges else 1

    # ── 画布 ──────────────────────────────────────────────────────────
    half  = (n + 1) // 2
    fig_h = max(9, half * GAP_Y + 2.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    # ── 画箭头 ────────────────────────────────────────────────────────
    for src, dst, cnt in top_edges:
        sx, sy = pos[src]
        dx, dy = pos[dst]
        ratio  = cnt / max_cnt
        lw     = 1.2 + ratio * 6.5
        alpha  = 0.28 + ratio * 0.62
        color  = plt.cm.YlOrRd(0.25 + ratio * 0.70)

        angle = math.atan2(dy - sy, dx - sx)
        x1 = sx + NODE_R * math.cos(angle)
        y1 = sy + NODE_R * math.sin(angle)
        x2 = dx - NODE_R * 1.3  * math.cos(angle)
        y2 = dy - NODE_R * 1.3  * math.sin(angle)

        # 同列节点之间用弧线，跨列用直线
        rad = 0.22 if (node_list.index(src) % 2 == node_list.index(dst) % 2) else 0.06
        hw = 0.10 + ratio * 0.18   # 箭头头部宽度
        hl = 0.12 + ratio * 0.20   # 箭头头部长度
        tw = 0.02 + ratio * 0.06   # 箭杆宽度
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle=f"fancy,head_width={hw:.2f},head_length={hl:.2f},tail_width={tw:.2f}",
                        color=color, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}",
                    ))
        # 箭头中点标注次数
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.08, str(cnt), ha="center", va="bottom",
                fontsize=7.5, color="#444",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

    # ── 画节点 ────────────────────────────────────────────────────────
    for node, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), NODE_R, color="#4C72B0", zorder=3, alpha=0.92))
        ax.text(x, y, node, ha="center", va="center",
                fontproperties=_FP_S, color="white", zorder=4)

    # ── 图例 ──────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    leg_items = [
        Line2D([0],[0], color=plt.cm.YlOrRd(0.95), lw=6,   label="高频转移"),
        Line2D([0],[0], color=plt.cm.YlOrRd(0.60), lw=3.0, label="中频转移"),
        Line2D([0],[0], color=plt.cm.YlOrRd(0.30), lw=1.2, label="低频转移"),
    ]
    leg = ax.legend(handles=leg_items, loc="lower right", framealpha=0.88)
    for t in leg.get_texts():
        t.set_fontproperties(_FP_S)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - 0.9, max(all_x) + 0.9)
    ax.set_ylim(min(all_y) - 0.9, max(all_y) + 0.9)
    ax.set_title("C4 · 模块转移流向图（每节点 Top3，箭头粗细 = 频率）",
                 fontproperties=_FP_B, pad=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_C4_flow.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图C4] 已保存 → {path}")


if __name__ == "__main__":
    main()
