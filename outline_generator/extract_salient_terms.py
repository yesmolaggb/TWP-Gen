"""
请先运行parse_corpus_svo.py处理中文语料库，得到包含SVO三元组的.pk文件。
"""
import argparse
import json
import pickle as pk
from collections import defaultdict
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
import string
import os
import twpgen_config as args

### 选择显著的动词词元和宾语中心词
# 使用示例:
# python3 extract_salient_terms.py \
#     --corpus_w_svo_pickle ./zhcorpus/corpus_parsed_svo.pk \
#     --min_verb_freq 3 \
#     --min_obj_freq 3

# 中文停用词集合
chinese_stop_words = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", 
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "这个", "那个", "这些", "那些", "这样", "那样", "这么", "那么", 
    "什么", "怎么", "如何", "何", "啊", "呀", "吧", "吗", "呢", "哦", "哪", "哎", "喂", "嗯", "这里", "那里", "但是", "因为",
    "所以", "如果", "虽然", "然后", "而且", "并且", "或者", "不过", "可是", "然而", "只是", "之", "的话", "让", "得", "地", 
    "被", "但", "并", "可", "却", "把", "来", "从", "对", "给", "向", "和", "与", "为", "以", "及", "像", "由"
])

# 扩展中文停用词：添加标点符号
for c in string.punctuation:
    chinese_stop_words.add(c)
# 扩展中文停用词：添加中文标点
for c in "，。！？；：""''（）【】《》「」『』〈〉…—·、":
    chinese_stop_words.add(c)
# 扩展中文停用词：添加数字
for c in string.digits:
    chinese_stop_words.add(c)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def get_salience(item2local_freq, item2global_freq):
    import math

    N_sent_bkg = 100000000  # 背景语料句子数量估计
    item2salience = {}
    for item, local_freq in item2local_freq.items():
        if item not in item2global_freq:
            # 如果在全局频率文件中找不到，尝试去除可能的引号
            clean_item = item.strip('"')
            if clean_item in item2global_freq:
                item2salience[item] = local_freq * math.log(item2global_freq[clean_item], 10)
            else:
                item2salience[item] = -1
        else:
            salience = local_freq * math.log(item2global_freq[item], 10)
            item2salience[item] = salience
    return item2salience


def get_salient_frequent_verb_lemmas(
    verb2local_freq, verb2global_freq, top_ratio=0.8, min_freq=5
):
    verb2salience = get_salience(verb2local_freq, verb2global_freq)
    # 中文功能动词停用词
    stopword_verbs = chinese_stop_words | {
        "是", "有", "来", "去", "做", "用", "能", "会", "要", "可以", "可能", "应该", "应当", "必须", "需要",
        "获得", "进行", "使", "让", "认为", "表示", "说", "指出", "称", "告诉", "回应", "提到", "提出"
    }

    # 过滤掉salience为-1的动词（在全局频率中找不到的）
    filtered_verb2salience = {k: v for k, v in verb2salience.items() if v > 0}
    if not filtered_verb2salience:
        print("警告：所有动词的显著性都是负值，可能是全局频率文件不匹配")
        filtered_verb2salience = verb2salience

    V = max(1, int(len(filtered_verb2salience) * top_ratio))  # 至少选择1个
    salient_verbs = {}
    for ele in sorted(filtered_verb2salience.items(), key=lambda x: -x[1]):
        if ele[0] not in stopword_verbs:
            salient_verbs[ele[0]] = ele[1]
        if len(salient_verbs) == V:
            break

    print(f"选择 {len(salient_verbs)} 个显著动词")

    frequent_salient_verbs = {}
    for verb, saliency in salient_verbs.items():
        if verb2local_freq[verb] >= min_freq:
            frequent_salient_verbs[verb] = saliency

    print(f"选择 {len(frequent_salient_verbs)} 个高频显著动词")
    return frequent_salient_verbs


def get_salient_frequent_object_heads(
    oh2local_freq, oh2global_freq, top_ratio=0.8, min_freq=3
):
    oh2salience = get_salience(oh2local_freq, oh2global_freq)

    # 中文功能名词停用词
    stopword_nouns = chinese_stop_words | {""}
    
    # 过滤掉salience为-1的宾语头（在全局频率中找不到的）
    filtered_oh2salience = {k: v for k, v in oh2salience.items() if v > 0}
    if not filtered_oh2salience:
        print("警告：所有宾语头的显著性都是负值，可能是全局频率文件不匹配")
        filtered_oh2salience = oh2salience
    
    V = max(1, int(len(filtered_oh2salience) * top_ratio))  # 至少选择1个
    salient_oh = {}
    for ele in sorted(filtered_oh2salience.items(), key=lambda x: -x[1]):
        if ele[0] not in stopword_nouns:
            salient_oh[ele[0]] = ele[1]
        if len(salient_oh) == V:
            break

    print(f"选择 {len(salient_oh)} 个显著宾语中心词")

    frequent_salient_ohs = {}
    for head, saliency in salient_oh.items():
        if oh2local_freq[head] >= min_freq:
            frequent_salient_ohs[head] = saliency

    print(f"选择 {len(frequent_salient_ohs)} 个高频显著宾语中心词")
    return frequent_salient_ohs


def main(
    corpus_w_svo_pickle,
    verb_freq_file,
    all_lemma_freq_file,
    spacy_model,
    min_verb_freq,
    top_verb_ratio,
    min_obj_freq,
    top_obj_ratio,
):

    print("加载Spacy模型")
    nlp = spacy.load(spacy_model)
    # 对于中文，不使用WhitespaceTokenizer，使用SpaCy的默认分词器
    # 对于英文，使用WhitespaceTokenizer
    if spacy_model.startswith("en_"):
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    print("加载语料库")
    with open(corpus_w_svo_pickle, "rb") as f:
        corpus = pk.load(f)

    print("加载词频文件")
    try:
        with open(verb_freq_file, "r", encoding="utf-8") as f:
            verb2global_freq = json.load(f)
        print(f"动词频率文件加载成功，包含 {len(verb2global_freq)} 个条目")
    except Exception as e:
        print(f"加载动词频率文件时出错: {e}")
        verb2global_freq = {}

    try:
        with open(all_lemma_freq_file, "rb") as f:
            lemma2global_freq = json.load(f)
        print(f"词元频率文件加载成功，包含 {len(lemma2global_freq)} 个条目")
    except Exception as e:
        print(f"加载词元频率文件时出错: {e}")
        lemma2global_freq = {}
    
    # 统计动词和宾语的频率
    verb2local_freq = defaultdict(int)
    obj2local_freq = defaultdict(int)
    for doc in corpus.values():
        for em in doc["svos"]:
            # 获取动词部分
            if em[1] is not None:
                verb = em[1][0]  # 取词元部分
                if verb.startswith("!"):
                    verb = verb[1:]  # 去除否定标记
                if verb.startswith("被"):
                    verb = verb[2:]  # 去除被动标记
                verb2local_freq[verb] += 1
            
            # 获取宾语部分
            if em[2] is not None:
                obj = em[2][0]  # 取词元部分
                obj2local_freq[obj] += 1

    verb2local_freq = dict(verb2local_freq)
    obj2local_freq = dict(obj2local_freq)
    print(f"语料库中共有 {len(verb2local_freq)} 个唯一动词和 {len(obj2local_freq)} 个唯一宾语")
    
    # 检查动词是否在全局频率文件中
    found_verbs = sum(1 for v in verb2local_freq if v in verb2global_freq)
    print(f"在全局频率文件中找到 {found_verbs}/{len(verb2local_freq)} 个动词 ({found_verbs/len(verb2local_freq)*100:.2f}%)")
    
    # 提取宾语中心词
    print("提取宾语中心词")
    obj_head2local_freq = defaultdict(int)
    obj2obj_head_info = {}
    for obj, local_freq in tqdm(obj2local_freq.items()):
        # 解析宾语
        parsed_obj = nlp(obj)
        obj_head_lemma = None
        obj_head_relative_index = 0
        
        # 寻找ROOT节点作为中心词
        for i, tok in enumerate(parsed_obj):
            if tok.dep_ == "ROOT":
                # 确保使用有效的词元
                obj_head_lemma = tok.text if not tok.lemma_ or tok.lemma_ == '' else tok.lemma_
                obj_head_relative_index = i
                break
        
        # 如果没有找到ROOT，尝试找名词
        if obj_head_lemma is None:
            for i, tok in enumerate(parsed_obj):
                if tok.pos_ in ["NOUN", "PROPN"]:
                    # 确保使用有效的词元
                    obj_head_lemma = tok.text if not tok.lemma_ or tok.lemma_ == '' else tok.lemma_
                    obj_head_relative_index = i
                    break
        
        # 如果还是没找到，使用最后一个词（中文中通常最后一个词是中心词）
        if obj_head_lemma is None and len(parsed_obj) > 0:
            last_token = parsed_obj[len(parsed_obj) - 1]
            # 确保使用有效的词元
            obj_head_lemma = last_token.text if not last_token.lemma_ or last_token.lemma_ == '' else last_token.lemma_
            obj_head_relative_index = len(parsed_obj) - 1
        
        # 如果对象为空，跳过
        if obj_head_lemma is None:
            continue
        
        # 如果词元为空，使用原始文本
        if not obj_head_lemma or obj_head_lemma.strip() == '':
            obj_head_lemma = obj
        
        obj2obj_head_info[obj] = {
            "obj_head_lemma": obj_head_lemma,
            "obj_head_relative_index": obj_head_relative_index,
        }
        obj_head2local_freq[obj_head_lemma] += local_freq
    
    obj_head2local_freq = dict(obj_head2local_freq)
    print(f"提取了 {len(obj_head2local_freq)} 个唯一宾语中心词")
    
    # 检查宾语中心词是否在全局频率文件中
    found_obj_heads = sum(1 for oh in obj_head2local_freq if oh in lemma2global_freq)
    print(f"在全局频率文件中找到 {found_obj_heads}/{len(obj_head2local_freq)} 个宾语中心词 ({found_obj_heads/len(obj_head2local_freq)*100:.2f}%)")
    
    # 计算动词和宾语中心词的显著性
    frequent_salient_verbs = get_salient_frequent_verb_lemmas(
        verb2local_freq,
        verb2global_freq,
        top_ratio=top_verb_ratio,
        min_freq=min_verb_freq,
    )

    frequent_salient_object_heads = get_salient_frequent_object_heads(
        obj_head2local_freq,
        lemma2global_freq,
        top_ratio=top_obj_ratio,
        min_freq=min_obj_freq,
    )

    # 避免文件部分覆写
    if os.path.exists(f"{corpus_w_svo_pickle[:-3]}_salient_verbs.pk"):
        os.remove(f"{corpus_w_svo_pickle[:-3]}_salient_verbs.pk")
    if os.path.exists(f"{corpus_w_svo_pickle[:-3]}_salient_obj_heads.pk"):
        os.remove(f"{corpus_w_svo_pickle[:-3]}_salient_obj_heads.pk")
    if os.path.exists(f"{corpus_w_svo_pickle[:-3]}_obj2obj_head_info.pk"):
        os.remove(f"{corpus_w_svo_pickle[:-3]}_obj2obj_head_info.pk")

    print("保存选择的显著术语")
    with open(f"{corpus_w_svo_pickle[:-3]}_salient_verbs.pk", "wb") as f:
        pk.dump(frequent_salient_verbs, f)
    with open(f"{corpus_w_svo_pickle[:-3]}_salient_obj_heads.pk", "wb") as f:
        pk.dump(frequent_salient_object_heads, f)
    with open(f"{corpus_w_svo_pickle[:-3]}_obj2obj_head_info.pk", "wb") as f:
        pk.dump(obj2obj_head_info, f)


if __name__ == "__main__":
    # 从args中获取参数
    main(
        args.parsed_corpus,
        args.verb_freq_file,
        args.all_lemma_freq_file,
        args.spacy_model,  # 使用中文模型
        args.min_verb_freq,
        args.top_verb_ratio,
        args.min_obj_freq,
        args.top_obj_ratio,
    )               