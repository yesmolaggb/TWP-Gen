import argparse
import pickle as pk
import json
import rbo
import numpy as np
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math
from tqdm import tqdm
import os
import twpgen_config as args
import sys
import re

lemmatizer = WordNetLemmatizer()

# 定义常量，避免魔法数字
# 错误状态码
ERROR_NOT_IN_DICT = -1       # 词不在词典中
ERROR_NON_SALIENT = -2       # 非显著谓词提及，没有嵌入特征
ERROR_OTHER = -3             # 其他错误
ERROR_NO_EXPANSION = -4      # 无扩展结果
ERROR_ZERO_SCORE = -5        # 所有得分为0
# 特殊状态码
SINGLE_SENSE_VERB = -10      # 单义词标记，使用-10避免与词义ID冲突


def cosine_similarity_embedding(emb_a, emb_b):
    """
    计算两个向量之间的余弦相似度
    
    Args:
        emb_a: 第一个向量，可以是numpy数组或列表
        emb_b: 第二个向量，可以是numpy数组或列表
        
    Returns:
        余弦相似度值
    """
    # 检查None值
    if emb_a is None or emb_b is None:
        return 0.0
    
    # 检查数据类型，确保是数值数据
    if isinstance(emb_a, dict) or isinstance(emb_b, dict):
        return 0.0
        
    # 将列表转换为numpy数组
    if isinstance(emb_a, list):
        emb_a = np.array(emb_a)
    if isinstance(emb_b, list):
        emb_b = np.array(emb_b)
    
    # 计算余弦相似度
    try:
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(emb_a, emb_b) / (norm_a * norm_b)
    except Exception as e:
        print(f"计算余弦相似度时出错: {e}")
        return 0.0


def normalize_token(token):
    """
    标准化token，去除特殊字符和标点符号
    
    Args:
        token: 输入的token
        
    Returns:
        标准化后的token
    """
    if not token:
        return ""
    
    # 如果是非字符串类型，尝试转换
    if not isinstance(token, str):
        try:
            token = str(token)
        except:
            return ""
    
    # 去除标点符号和特殊字符
    token = re.sub(r'[^\w\s]', '', token)
    
    # 去除[UNK]标记
    token = re.sub(r'\[UNK\]', '', token)
    
    # 去除空格
    token = token.strip()
    
    return token


def string_overlap_score(str1, str2):
    """
    计算两个字符串的字符重叠度
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
        
    Returns:
        重叠度得分，范围为0到1
    """
    # 检查输入类型
    if not isinstance(str1, str):
        try:
            str1 = str(str1)
        except:
            return 0.0
    
    if not isinstance(str2, str):
        try:
            str2 = str(str2)
        except:
            return 0.0
            
    if not str1 or not str2:
        return 0.0
    
    # 将字符串转换为字符集合
    set1 = set(str1)
    set2 = set(str2)
    
    # 计算交集大小
    intersection = len(set1.intersection(set2))
    
    # 计算并集大小
    union = len(set1.union(set2))
    
    # 计算Jaccard相似度
    if union == 0:
        return 0.0
    
    return intersection / union


def improved_rbo_similarity(list1, list2, p=0.9):
    """
    改进的RBO相似度计算，处理空列表和无交集情况
    
    Args:
        list1: 第一个列表
        list2: 第二个列表
        p: RBO权重参数
        
    Returns:
        RBO相似度
    """
    # 检查空列表
    if not list1 or not list2:
        return 0.0
    
    # 确保列表中的元素可以比较
    try:
        # 深度复制列表避免修改原始数据
        list1_safe = [str(item) for item in list1]
        list2_safe = [str(item) for item in list2]
        
        # 计算RBO相似度
        rbo_score = rbo.RankingSimilarity(list1_safe, list2_safe, verbose=False).rbo()
        return rbo_score
    except Exception as e:
        print(f"计算RBO相似度时出错: {e}")
        return 0.0


def extract_expansion_from_mention(verb_info):
    """
    从提及中提取扩展结果，处理不同格式的数据
    
    Args:
        verb_info: 提及信息，可能包含不同格式的扩展结果
        
    Returns:
        处理后的扩展结果列表
    """
    # 检查不同可能的字段名
    expansion_fields = ["verb_expansion_results", "expanded", "expansion_results", "expd_results", "expd"]
    
    expansion_data = None
    for field in expansion_fields:
        if field in verb_info and verb_info[field] is not None:
            expansion_data = verb_info[field]
            break
    
    if expansion_data is None or len(expansion_data) == 0:
        return []
    
    expd_lemma2score = defaultdict(float)
    
    # 处理不同格式的扩展结果
    for rank, ele in enumerate(expansion_data):
        # 处理列表格式 [token, score]
        if isinstance(ele, list) and len(ele) >= 1:
            token = normalize_token(ele[0])
            if not token:
                continue
                
            score = ele[1] if len(ele) > 1 else 1.0 / math.log(2 + rank)
            expd_lemma = lemmatizer.lemmatize(token, "v")
            expd_lemma2score[expd_lemma] += score
        # 处理元组格式 (token, score)
        elif isinstance(ele, tuple) and len(ele) >= 1:
            token = normalize_token(ele[0])
            if not token:
                continue
                
            score = ele[1] if len(ele) > 1 else 1.0 / math.log(2 + rank)
            expd_lemma = lemmatizer.lemmatize(token, "v")
            expd_lemma2score[expd_lemma] += score
        # 处理字典格式 {"token_str": token, "score": score}
        elif isinstance(ele, dict):
            # 尝试不同的键名
            token = None
            for key in ["token_str", "token", "word", "text"]:
                if key in ele and ele[key]:
                    token = normalize_token(ele[key])
                    if token:
                        break
                        
            if not token:
                continue
                
            score = None
            for key in ["score", "weight", "probability", "prob"]:
                if key in ele:
                    try:
                        score = float(ele[key])
                        break
                    except:
                        pass
                        
            if score is None:
                score = 1.0 / math.log(2 + rank)
                
            expd_lemma = lemmatizer.lemmatize(token, "v")
            expd_lemma2score[expd_lemma] += score
        # 处理字符串格式
        elif isinstance(ele, str):
            token = normalize_token(ele)
            if not token:
                continue
                
            score = 1.0 / math.log(2 + rank)
            expd_lemma = lemmatizer.lemmatize(token, "v")
            expd_lemma2score[expd_lemma] += score
    
    # 排序
    sorted_expd = [
        ele[0] for ele in sorted(expd_lemma2score.items(), key=lambda x: -x[1])
    ]
    
    return sorted_expd


def aggregate_sense_feature_by_ranking(bert_expanded_verbnet):
    """Return lemma -> sense with features
    {
        lemma 1: {
            sense1: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
            sense2: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
        },
        lemma 2: {
            sense1: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
            sense2: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
        }
    }
    """
    lemmatizer = WordNetLemmatizer()
    lemma2sense_features = {}
    
    try:
        # 处理不同格式的词典
        if isinstance(bert_expanded_verbnet, list):
            # 列表格式的词典
            for entry in tqdm(bert_expanded_verbnet, desc="生成词义特征"):
                if 'lemma' in entry and 'senses' in entry:
                    lemma = entry['lemma']
                    senses = entry['senses']
                    
                    if not isinstance(senses, list):
                        continue
                        
                    if lemma not in lemma2sense_features:
                        lemma2sense_features[lemma] = {}
                        
                    for sense_id, sense in enumerate(senses):
                        sense_feature = defaultdict(float)
                        
                        # 获取词向量
                        sense_embed = None
                        if "sense_embed" in sense and sense["sense_embed"] is not None:
                            sense_embed = sense["sense_embed"]
                        elif "features" in sense and "embedding" in sense["features"]:
                            sense_embed = sense["features"]["embedding"]
                        
                        # 处理例句扩展
                        examples_w_expansion = sense.get("examples_w_expansion", [])
                        if not examples_w_expansion:
                            lemma2sense_features[lemma][sense_id] = [{}, sense_embed]
                            continue
                            
                        for example_w_expansion in examples_w_expansion:
                            if len(example_w_expansion) < 2:
                                continue
                                
                            for ele in example_w_expansion[1]:
                                if "expansion_results" not in ele:
                                    continue
                                    
                                expd_results = ele["expansion_results"]
                                expd_lemma2rank = {}
                                for expd_lemma_rank, term in enumerate(expd_results):
                                    if isinstance(term, dict) and "token_str" in term:
                                        # 标准化token
                                        token = normalize_token(term["token_str"])
                                        if not token:
                                            continue
                                        
                                        expd_lemma = lemmatizer.lemmatize(token, "v")
                                    else:
                                        continue
                                    if expd_lemma not in expd_lemma2rank:
                                        expd_lemma2rank[expd_lemma] = 1 + expd_lemma_rank

                                for expd_lemma, expd_lemma_rank in expd_lemma2rank.items():
                                    sense_feature[expd_lemma] += 1.0 / math.log(1 + expd_lemma_rank)
                        
                        sense_feature = dict(sense_feature)
                        sorted_features = {
                            ele[0]: ele[1]
                            for ele in sorted(sense_feature.items(), key=lambda x: -x[1])
                        }
                        lemma2sense_features[lemma][sense_id] = [sorted_features, sense_embed]
        else:
            # 字典格式的词典
            for lemma, info in tqdm(bert_expanded_verbnet.items(), desc="生成词义特征"):
                # 检查lemma是否有效
                if not lemma or not isinstance(lemma, str):
                    continue
                    
                # 适应实际词典结构
                if isinstance(info, dict) and 'senses' in info:
                    senses = info['senses']
                    lemma_str = info.get('lemma', lemma)  # 使用词典中的lemma字段，如果存在
                    if isinstance(lemma_str, str) and lemma_str.strip():
                        lemma = lemma_str
                else:
                    senses = info  # 兼容旧格式
                    
                # 初始化词义特征词典
                if lemma not in lemma2sense_features:
                    lemma2sense_features[lemma] = {}
                    
                if not isinstance(senses, list):
                    continue
                    
                for sense_id, sense in enumerate(senses):
                    sense_feature = defaultdict(float)
                    
                    # 获取词向量
                    sense_embed = None
                    if "sense_embed" in sense and sense["sense_embed"] is not None:
                        sense_embed = sense["sense_embed"]
                    elif "features" in sense and "embedding" in sense["features"]:
                        sense_embed = sense["features"]["embedding"]
                    
                    # 处理例句扩展
                    examples_w_expansion = sense.get("examples_w_expansion", [])
                    if not examples_w_expansion:
                        lemma2sense_features[lemma][sense_id] = [{}, sense_embed]
                        continue
                        
                    for example_w_expansion in examples_w_expansion:
                        if len(example_w_expansion) < 2:
                            continue
                            
                        for ele in example_w_expansion[1]:
                            if "expansion_results" not in ele:
                                continue
                                
                            expd_results = ele["expansion_results"]
                            expd_lemma2rank = {}
                            for expd_lemma_rank, term in enumerate(expd_results):
                                if isinstance(term, dict) and "token_str" in term:
                                    # 标准化token
                                    token = normalize_token(term["token_str"])
                                    if not token:
                                        continue
                                    
                                    expd_lemma = lemmatizer.lemmatize(token, "v")
                                else:
                                    continue
                                if expd_lemma not in expd_lemma2rank:
                                    expd_lemma2rank[expd_lemma] = 1 + expd_lemma_rank

                            for expd_lemma, expd_lemma_rank in expd_lemma2rank.items():
                                sense_feature[expd_lemma] += 1.0 / math.log(1 + expd_lemma_rank)
                    
                    sense_feature = dict(sense_feature)
                    sorted_features = {
                        ele[0]: ele[1]
                        for ele in sorted(sense_feature.items(), key=lambda x: -x[1])
                    }
                    lemma2sense_features[lemma][sense_id] = [sorted_features, sense_embed]
    except Exception as e:
        print(f"生成词义特征时出错: {e}")
                
    return lemma2sense_features


def disambiguate_word_sense(verb_info, lemma2sense_features, options):
    """
    词义消歧函数，能够处理不同格式的扩展结果，并结合多种相似度指标
    
    Args:
        verb_info: 动词提及信息，格式为字典 {
            'verb': str,
            'verb_embed': list or np.array,
            'verb_expansion_results': list
        }
        lemma2sense_features: 词义特征字典
        options: 消歧选项
        
    Returns:
        sense_id: 消歧结果，正数表示成功消歧的词义ID，负数表示不同的错误状态
            -1: 词不在词典中
            -2: 非显著谓词提及，没有嵌入特征
            -3: 其他错误
            -4: 无扩展结果
            -5: 所有得分为0
            -10: 单义词（只有一个词义的词）
    """
    try:
        sense_top_k = options.get("sense_top_k", 10)
        rbo_p = options.get("rbo_p", 0.9)
        embed_weight = options.get("embed_weight", 0.6)
        rbo_weight = options.get("rbo_weight", 0.3)
        char_overlap_weight = options.get("char_overlap_weight", 0.1)
        # 新增参数：是否将单义词转换为多义处理（默认为True）
        convert_single_sense = options.get("convert_single_sense", True)

        # 获取动词lemma
        lemma = None
        if "verb" in verb_info:
            lemma = verb_info["verb"]
            
        if not lemma or not isinstance(lemma, str):
            return ERROR_OTHER
            
        # 获取嵌入向量
        lemma_embed = None
        for field in ["verb_embed", "embedding", "emb", "vector"]:
            if field in verb_info and verb_info[field] is not None:
                lemma_embed = verb_info[field]
                break
        
        if lemma_embed is None:  # 非显著谓词提及，没有嵌入特征
            return ERROR_NON_SALIENT
            
        if lemma not in lemma2sense_features:  # 词不在词典中
            return ERROR_NOT_IN_DICT
            
        # 提取提及的扩展结果
        sorted_expd_lemma = extract_expansion_from_mention(verb_info)
        
        # 修复单义词判断逻辑
        if len(lemma2sense_features[lemma]) == 1 and not convert_single_sense:
            # 使用新的单义词标记-10，避免与词义ID=0冲突
            return SINGLE_SENSE_VERB
        
        # 对于有扩展结果的情况，总是进行消歧计算，避免将多义词误判为单义词
        if not sorted_expd_lemma:
            # 如果确实没有扩展结果
            if len(lemma2sense_features[lemma]) == 1:
                # 且只有一个词义，可以视为单义词
                return SINGLE_SENSE_VERB
            else:
                # 否则，标记为无扩展结果错误
                return ERROR_NO_EXPANSION
        
        # 开始消歧
        sense_id2scores = {}
        
        for sense_id, sense_feature in lemma2sense_features[lemma].items():
            # 检查数据格式
            if not isinstance(sense_feature, list) or len(sense_feature) < 2:
                continue
                
            # 提取词义的扩展特征
            sorted_features_dict = sense_feature[0]
            sense_embed = sense_feature[1]
            
            if not isinstance(sorted_features_dict, dict):
                continue
                
            ranked_sense_feature = list(sorted_features_dict.keys())[:sense_top_k]
            
            # 计算RBO相似度
            rbo_score = improved_rbo_similarity(sorted_expd_lemma, ranked_sense_feature, rbo_p)
            
            # 计算嵌入相似度
            embed_score = cosine_similarity_embedding(sense_embed, lemma_embed)
            
            # 计算字符重叠度
            char_overlap_scores = []
            for expd in sorted_expd_lemma[:5]:  # 只考虑前5个扩展词
                for sense_expd in ranked_sense_feature[:5]:  # 只考虑前5个词义扩展词
                    overlap = string_overlap_score(expd, sense_expd)
                    char_overlap_scores.append(overlap)
            
            char_overlap_score = max(char_overlap_scores) if char_overlap_scores else 0.0
            
            # 计算最终得分
            final_score = (
                embed_weight * embed_score + 
                rbo_weight * rbo_score + 
                char_overlap_weight * char_overlap_score
            )
            
            sense_id2scores[sense_id] = {
                "final_score": final_score,
                "rbo_score": rbo_score,
                "embed_score": embed_score,
                "char_overlap_score": char_overlap_score
            }
        
        # 按最终得分排序
        sorted_senses = sorted(sense_id2scores.items(), key=lambda x: -x[1]["final_score"])
        
        # 如果没有得分，返回-5
        if not sorted_senses:
            return ERROR_ZERO_SCORE
            
        # 如果最高得分为0，返回-5
        if sorted_senses[0][1]["final_score"] == 0.0:
            return ERROR_ZERO_SCORE
        
        # 确保返回的是整数类型的sense_id
        best_sense_id = sorted_senses[0][0]
        
        # 处理不同类型的sense_id
        try:
            return int(best_sense_id)
        except (ValueError, TypeError):
            # 如果无法转换为整数，则返回一个正数表示成功消歧
            return 1
    except Exception as e:
        print(f"词义消歧过程中出错: {e}")
        return ERROR_OTHER


def main(mention_file, save_path, dict_file):
    """
    主函数，加载数据并进行词义消歧
    
    Args:
        mention_file: 提及特征文件路径
        save_path: 保存路径
        dict_file: 词典文件路径
    """
    try:
        print(f"正在加载词典文件: {dict_file}")
        with open(dict_file, "r", encoding="utf-8") as read_file:
            verb_sense_dict = json.load(read_file)

        print(f"正在加载提及特征文件: {mention_file}")
        with open(mention_file, "rb") as f:
            parsed_data = pk.load(f)

        print("正在提取词典特征...")
        lemma2sense_features = aggregate_sense_feature_by_ranking(verb_sense_dict)
        print(f"成功提取 {len(lemma2sense_features)} 个动词的词义特征")

        parsed_data_wsd = {}
        options = {
            "sense_top_k": 10,
            "rbo_p": 0.9,
            "embed_weight": 0.6,
            "rbo_weight": 0.3, 
            "char_overlap_weight": 0.1,
            "convert_single_sense": False  # 先保持与旧版代码一致的行为
        }
        
        # 统计消歧结果
        stats = {
            "total": 0,
            "non_salient": 0,        # 非显著谓词 (-2)
            "not_in_dict": 0,        # 不在词典中 (-1)
            "single_sense": 0,       # 单义词 (-10)
            "successful": 0,         # 成功消歧 (>0)
            "no_expansion": 0,       # 无扩展结果 (-4)
            "zero_score": 0,         # 得分为0 (-5)
            "other_errors": 0        # 其他错误 (-3)
        }
        
        print("正在进行词义消歧...")
        for svo_id, svo in tqdm(parsed_data.items(), desc="消歧动词"):
            stats["total"] += 1
            
            sense_id = disambiguate_word_sense(svo, lemma2sense_features, options)
            parsed_data_wsd[svo_id] = sense_id
            
            # 统计不同结果
            if isinstance(sense_id, (int, float)):
                if sense_id > 0:
                    stats["successful"] += 1
                elif sense_id == SINGLE_SENSE_VERB:
                    stats["single_sense"] += 1
                elif sense_id == ERROR_NOT_IN_DICT:
                    stats["not_in_dict"] += 1
                elif sense_id == ERROR_NON_SALIENT:
                    stats["non_salient"] += 1
                elif sense_id == ERROR_NO_EXPANSION:
                    stats["no_expansion"] += 1
                elif sense_id == ERROR_ZERO_SCORE:
                    stats["zero_score"] += 1
                else:
                    stats["other_errors"] += 1
            else:
                # 非数值类型，记为其他错误
                stats["other_errors"] += 1
                parsed_data_wsd[svo_id] = ERROR_OTHER

        # 第二步：如果启用自动修复单义词，将所有标记为-10的单义词改为其实际词义ID（通常是0）
        if options.get("auto_fix_single_sense", False):
            print("正在修复单义词标记...")
            fixed_count = 0
            for svo_id, sense_id in parsed_data_wsd.items():
                if sense_id == SINGLE_SENSE_VERB:
                    parsed_data_wsd[svo_id] = 0  # 将单义词改为第一个义项（索引为0）
                    fixed_count += 1
                    
            if fixed_count > 0:
                # 更新统计信息
                stats["successful"] += fixed_count
                stats["single_sense"] -= fixed_count
                print(f"已将 {fixed_count} 个单义词标记修复为第一个义项")

        print(f"正在保存消歧结果到: {save_path}")
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, "wb") as f:
            pk.dump(parsed_data_wsd, f)
        
        # 打印统计信息
        print("\n消歧统计:")
        print(f"总提及数量: {stats['total']}")
        print(f"成功消歧 (>0): {stats['successful']} ({stats['successful']/stats['total']*100:.2f}%)")
       
        
        return {
            "stats": stats,
            "saved_path": save_path
        }
    except Exception as e:
        print(f"主函数执行出错: {e}")
        return {
            "error": str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='中文动词词义消歧')
    parser.add_argument('--mention_file', default='./dataset/DuEE/parsed_corpus_salient_po_mention_features.pk',
                        help='提及特征文件路径')
    parser.add_argument('--save_path', default='./dataset/DuEE/po_mention_disambiguated.pk',
                        help='保存路径')
    parser.add_argument('--dict_file', default='./resources/verb_sense_dict_w_features.json',
                        help='词典文件路径')
    
    args_parsed = parser.parse_args()
    
    # 如果使用命令行参数
    if len(sys.argv) > 1:
        main(args_parsed.mention_file, args_parsed.save_path, args_parsed.dict_file)
    else:
        # 使用配置文件中的参数
        main(args.mention_file, args.save_disambiguated_path, args.dict_file)
