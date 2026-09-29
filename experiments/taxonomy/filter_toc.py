import os
from openai import OpenAI

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("TWPGEN_TAXONOMY_FILTER_MODEL", "qwen3-235b-a22b")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

BASE_DIR = os.getenv("TWPGEN_TAXONOMY_DATA_DIR", "data/taxonomy")


def ai_judge(toc_content: str, folder_name: str) -> tuple:
    prompt = f"""你是一个专业的文档目录质量审核员。请分析以下目录（TOC）内容，按照下面三个规则判断是否应该保留：

规则1：必须有二级目录（即在一级标题下还有更细的子标题层级），如果整个目录只有一个层级、没有任何子级内容，则删除,只要整个目录有耳机目录就行，而不是必须每一个一级目录下面都要有二级目录。注意：不要以编号是中文还是阿拉伯数字作为判断依据。
规则2：一级标题数量必须不少于3个，少于3个则删除。注意：不要以编号是中文还是阿拉伯数字作为判断依据。
规则3：目录不能有明显错误。

文件夹名称：{folder_name}

目录内容：
{toc_content}

请先简要说明你的判断理由（一句话），再给出最终结论。
最后一行必须只输出数字：1 表示保留，2 表示删除。"""

    messages = [{"role": "user", "content": prompt}]

    # 深度思考模式，流式接收
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=True,
        max_tokens=1000,
    )

    full_content = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            full_content += delta.content

    result = full_content.strip()
    lines = [l.strip() for l in result.splitlines() if l.strip()]
    verdict = 1
    for line in reversed(lines):
        if line in ("1", "2"):
            verdict = int(line)
            break

    reason_lines = [l for l in lines if l not in ("1", "2")]
    reason = " ".join(reason_lines)[:80] if reason_lines else "无"

    return verdict, reason


def preview():
    if not os.path.exists(BASE_DIR):
        print(f"[错误] 目录不存在: {BASE_DIR}")
        return

    folders = sorted([f for f in os.listdir(BASE_DIR)
                      if os.path.isdir(os.path.join(BASE_DIR, f))])[:10]

    print(f"预览前 {len(folders)} 个文件夹（不做任何删除）\n")

    for i, folder in enumerate(folders, 1):
        toc_path = os.path.join(BASE_DIR, folder, "toc.md")

        if not os.path.exists(toc_path):
            print(f"[{i:02d}] {folder}")
            print(f"      结果: 2  原因: 无toc.md文件\n")
            continue

        with open(toc_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        try:
            verdict, reason = ai_judge(content, folder)
            tag = "保留 ✅" if verdict == 1 else "删除 ❌"
            print(f"[{i:02d}] {folder}")
            print(f"      结果: {verdict} ({tag})")
            print(f"      理由: {reason}\n")
        except Exception as e:
            print(f"[{i:02d}] {folder}")
            print(f"      结果: 1  AI调用失败，默认保留 ({e})\n")


if __name__ == "__main__":
    preview()
