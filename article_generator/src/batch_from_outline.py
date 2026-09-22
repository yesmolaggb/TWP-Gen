#!/usr/bin/env python3
"""
OutlineArticleWriter 批量大纲报告生成器

用法:
    python src/batch_from_outline.py

大纲文件目录: ./topic/  （每个 .md 文件即一个主题，文件名=题目名）

输出结构:
    ./output/article/       — 文章（.md + .html，文件名=大纲文件名）
    ./output/references/    — 参考文献（.json，含标号/id、url、content）
"""

import asyncio
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_from_outline import generate_from_outline
from src.utils.print_util import colored_print

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:  # central configuration: paths, keys, topics
    import twpgen_settings as repo_config
except Exception:  # pragma: no cover
    repo_config = None

DEFAULT_OUTPUT_DIR = repo_config.output_dir if repo_config else "output"


def sanitize(name: str) -> str:
    illegal = '<>:"/\\|?*'
    for ch in illegal:
        name = name.replace(ch, '_')
    return name


async def batch_generate(topic_dir: str, output_dir: str,
                         domain: str, save_html: bool):
    article_dir = os.path.join(output_dir, "article")
    refs_dir = os.path.join(output_dir, "references")

    pattern = os.path.join(topic_dir, "*.md")
    outline_files = glob.glob(pattern)

    if not outline_files:
        colored_print(f"[错误] 在 {topic_dir} 中未找到大纲文件 (*.md)", color="red")
        return

    topics = [(os.path.splitext(os.path.basename(f))[0], f) for f in outline_files]

    colored_print("=" * 60, color="blue")
    colored_print("OutlineArticleWriter - 批量大纲报告生成器", color="blue", bold=True)
    colored_print("=" * 60, color="blue")
    colored_print(f"  大纲目录 : {topic_dir}", color="white")
    colored_print(f"  文章输出 : {article_dir}", color="white")
    colored_print(f"  参考文献 : {refs_dir}", color="white")
    colored_print(f"  领域     : {domain}", color="white")
    colored_print(f"  生成HTML : {save_html}", color="white")
    colored_print(f"  主题数量 : {len(topics)}", color="white")
    colored_print("=" * 60, color="blue")

    ok, fail, skip = 0, 0, 0

    for idx, (topic, path) in enumerate(topics, 1):
        safe_name = sanitize(topic)
        article_file = os.path.join(article_dir, f"{safe_name}.md")

        # 自动断点续传：文章已存在则跳过
        if os.path.exists(article_file):
            colored_print(f"[{idx}/{len(topics)}] 跳过(已完成): {topic}", color="yellow")
            skip += 1
            continue

        colored_print(f"\n[{idx}/{len(topics)}] 开始: {topic}", color="purple")
        start = time.time()

        try:
            await generate_from_outline(
                outline_path=path,
                topic=topic,
                domain=domain,
                output_dir=article_dir,
                save_html=save_html,
                references_dir=refs_dir,
                file_base_name=safe_name,
            )
            elapsed = time.time() - start
            colored_print(f"[完成] {topic} ({elapsed:.0f}s)", color="green", bold=True)
            ok += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            colored_print(f"[失败] {topic}: {e}", color="red")
            fail += 1

    colored_print("\n" + "=" * 60, color="blue")
    colored_print("批量生成完成", color="blue", bold=True)
    colored_print("=" * 60, color="blue")
    colored_print(f"  成功: {ok}  |  失败: {fail}  |  跳过: {skip}", color="white")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OutlineArticleWriter 批量大纲报告生成器')
    parser.add_argument('--topic-dir', default='./topic',
                       help='大纲文件目录 (默认: ./topic)')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help='输出根目录 (默认: ./output)')
    parser.add_argument('--domain', '-d', default='Industry Research',
                       choices=['Industry Research', 'Company Research', 'Comprehensive Analysis'],
                       help='领域类型 (默认: Industry Research)')

    args = parser.parse_args()

    asyncio.run(batch_generate(
        topic_dir=args.topic_dir,
        output_dir=args.output_dir,
        domain=args.domain,
        save_html=True,
    ))
