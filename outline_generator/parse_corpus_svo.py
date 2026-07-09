import argparse
import os
import pickle as pk

import spacy
from tqdm import tqdm

import twpgen_config as args
from extract_svo_cn import findSVOs  # 使用针对中文的SVO提取函数

### Parse Corpus and Extract Subject-Verb-Object Triplets for Chinese text
# args example:
# python3 parse_corpus_svo.py \
#     --is_sentence 1 \
#     --input_file ./covid19/corpus.txt \
#     --save_path ./covid19/corpus_parsed_svo.pk


def filterEntity(entitys: list) -> list:
    """
    根据白名单过滤实体类型
    """
    entitys_filter = []
    for entity in entitys:
        label = entity[3]
        # 使用args中定义的标签白名单
        if label in args.label_whitelist:
            entitys_filter.append(entity)
    return entitys_filter


def main(input_file, spacy_model, save_path):  # 对中文使用zh_core_web_lg模型

    print("正在加载Spacy模型...")
    nlp = spacy.load(spacy_model)

    lines = []  # 事件句子
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                lines.append(line)
    save_dict_data = {}
    print(f"总句子数: {len(lines)}")
    print("开始解析和提取...")
    
    entitys_count = []
    for sent_id, sent in tqdm(enumerate(lines)):
        doc = nlp(sent)  # 解析句子
        token_list = [token.text for token in doc]
        # 中文文本不需要空格连接
        raw_sent = sent  # 保持原始句子不变，确保与输入完全一致
        
        # 提取主谓宾三元组，确保索引是基于token而非字符
        svos = findSVOs(doc)
        
        # 提取实体，索引是基于token的
        # 在spaCy中，ent.start和ent.end是基于token的索引，而非字符位置
        entitys = [[ent.text, ent.start, ent.end, ent.label_] for ent in doc.ents]
        entitys = filterEntity(entitys)
        entitys_count.append(len(entitys))

        save_dict_data[sent_id] = {
            "sent_id": sent_id + 1,  # 从1开始
            "raw_sentence": raw_sent,
            "token_list": token_list,
            "svos": svos,
            "entitys": entitys,
        }

    print(f"处理完成，共处理 {len(save_dict_data)} 个句子")
    
    # 避免文件部分覆写
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, "wb") as f:
        pk.dump(save_dict_data, f)
    print(f"结果已保存至 {save_path}")


if __name__ == "__main__":
    # 确保使用zh_core_web_lg模型处理中文
    main(args.corpus, "zh_core_web_lg", args.parsed_corpus)
