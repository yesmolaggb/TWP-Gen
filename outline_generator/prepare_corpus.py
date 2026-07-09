import os
import pickle as pk
from tqdm import tqdm


def create_corpus_info_without_labels(input_file, output_file):
    """
    从 corpus.txt 创建无标注的 corpus_info.pk。
    字段结构与 graph_embedding_dataset.py 和 evaluate_topic_clusters.py 所需完全兼容：
      - id       : 句子索引（与列表下标一致）
      - type     : 占位标签 'unknown'（无真实标签时使用）
      - sentence : 原始句子文本（gae_dataloader.getFeature() 读取此字段生成 BERT 特征）
      - trigger_word : 占位空字符串
      - arguments    : 占位空列表
    """
    print(f"正在从 {input_file} 读取语料库...")

    lines = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:  # 跳过空行
                lines.append(line)

    print(f"读取了 {len(lines)} 个句子")

    corpus_info = []
    for sent_id, sentence in tqdm(enumerate(lines), total=len(lines), desc="处理句子"):
        entry = {
            "id": sent_id,          # 索引与列表位置保持一致
            "type": "unknown",      # evaluate_topic_clusters.py 读此字段作为真实标签
            "sentence": sentence,   # gae_dataloader.getFeature() 读此字段
            "trigger_word": "",    # 占位，保持结构一致
            "arguments": [],        # 占位，保持结构一致
        }
        corpus_info.append(entry)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # 写入前删除旧文件
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧文件: {output_file}")

    with open(output_file, "wb") as f:
        pk.dump(corpus_info, f)

    print(f"成功创建 corpus_info.pk，包含 {len(corpus_info)} 个条目")
    return corpus_info


def verify_corpus_info(corpus_txt_path, corpus_info_path):
    """验证 corpus.txt 与 corpus_info.pk 条目数量一致，并抽查字段完整性。"""
    with open(corpus_txt_path, "r", encoding="utf-8") as f:
        txt_lines = [l.strip() for l in f if l.strip()]

    with open(corpus_info_path, "rb") as f:
        corpus_info = pk.load(f)

    print("\n===== 验证结果 =====")
    print(f"corpus.txt   行数: {len(txt_lines)}")
    print(f"corpus_info.pk 条目数: {len(corpus_info)}")

    mismatches = 0
    for i in range(min(len(txt_lines), len(corpus_info))):
        if txt_lines[i] != corpus_info[i]["sentence"]:
            mismatches += 1
            if mismatches <= 3:
                print(f"不匹配 [id={i}]:")
                print(f"  corpus.txt     : {txt_lines[i][:80]}")
                print(f"  corpus_info.pk : {corpus_info[i]['sentence'][:80]}")

    if len(txt_lines) == len(corpus_info) and mismatches == 0:
        print("验证通过: 两个文件完全一致")
    else:
        print(f"验证失败: 行数差异={len(txt_lines) - len(corpus_info)}, 内容不匹配数={mismatches}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 corpus.txt 创建无标注的 corpus_info.pk")
    parser.add_argument("--dataset-dir", default=None,
                        help="数据集目录，默认为 ./dataset/{TWPGEN_DATASET}")
    args = parser.parse_args()

    dataset_name = os.environ.get("TWPGEN_DATASET", "topic")
    TWPGEN_ROOT = os.path.dirname(os.path.abspath(__file__))  # /workspace/TWP-Gen
    DATASET_DIR = args.dataset_dir or os.path.join(TWPGEN_ROOT, "dataset", dataset_name)

    corpus_txt_path  = os.path.join(DATASET_DIR, "corpus.txt")
    corpus_info_path = os.path.join(DATASET_DIR, "corpus_info.pk")

    # 生成 corpus_info.pk
    create_corpus_info_without_labels(corpus_txt_path, corpus_info_path)

    # 验证生成结果
    verify_corpus_info(corpus_txt_path, corpus_info_path)
