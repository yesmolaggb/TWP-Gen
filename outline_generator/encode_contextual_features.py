"""
Please run parse_corpus_svo.py first to obtain the .pk sentence files before running this code.
"""
import argparse
import json
import math
import os
import pickle as pk
import random
import re

import torch
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from tqdm import tqdm
import numpy as np


import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:  # central configuration: model paths, dictionary, dataset paths
    import twpgen_settings as _cfg
except Exception:  # pragma: no cover
    _cfg = None


def _model_dir(name: str, default: str) -> str:
    if _cfg is None:
        return default
    return _cfg.language_model_paths.get(name, default)


MODELS = {
    'blu': (BertForMaskedLM, BertTokenizer, _model_dir('blu', '/workspace/model/bert-large-uncased-whole-word-masking')),
    'macbert': (BertForMaskedLM, BertTokenizer, _model_dir('macbert', '/workspace/model/chinese-macbert-base')),
    'chinese-bert-wwm': (BertForMaskedLM, BertTokenizer, _model_dir('chinese-bert-wwm', '/workspace/model/chinese-bert-wwm')),
}

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def convert_embedding_dimension(embedding, target_dim=768):
    """
    将嵌入向量转换为目标维度
    
    Args:
        embedding: 原始嵌入向量，numpy数组
        target_dim: 目标维度，默认为768
        
    Returns:
        转换后的嵌入向量
    """
    if embedding is None:
        return None
        
    original_dim = embedding.shape[0]
    
    if original_dim == target_dim:
        return embedding
    
    if original_dim > target_dim:
        # 方法1: 直接截取前target_dim个维度
        # return embedding[:target_dim]
        
        # 方法2: 使用PCA的思想，通过线性变换降维
        # 创建一个随机正交矩阵(但固定种子以确保一致性)
        np.random.seed(42)
        projection_matrix = np.random.randn(original_dim, target_dim)
        q, r = np.linalg.qr(projection_matrix)
        projection_matrix = q[:, :target_dim]
        
        # 使用这个矩阵进行线性变换
        result = np.dot(embedding, projection_matrix)
        
        # 归一化以保持向量长度
        norm_ratio = np.linalg.norm(embedding) / np.linalg.norm(result)
        result = result * norm_ratio
        
        return result
    else:
        # 如果原始维度小于目标维度，通过零填充扩展
        result = np.zeros(target_dim)
        result[:original_dim] = embedding
        return result


def prepare_sentence(token_list, tokenizer, is_chinese=False):
    """
    Inputs:
        token_list: a list of tokens
        tokenizer: a TokenizerClass object from HuggingFace
        is_chinese: whether the model is for Chinese
    Return:
        tokenized_text: A list of tokens obtained from basic tokenizer (e.g., WhiteSpaceTokenizer)
        tokenized_to_id_indicies: A list of (tokenids_chunks_index, token_id_start_index, token_id_end_index)
        tokenids_chunks: A list of token_id_end_index
    """
    # setting for BERT
    model_max_tokens = 512
    has_sos_eos = True
    ############## ########
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2
    sliding_window_size = max_tokens // 2

    if not hasattr(prepare_sentence, "sos_id"):
        prepare_sentence.sos_id, prepare_sentence.eos_id = tokenizer.encode("", add_special_tokens=True)

    # tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
    tokenized_text = token_list
    tokenized_to_id_indicies = []

    tokenids_chunks = []  # useful only if the sentence is longer than max_tokens
    tokenids_chunk = []

    for index, token in enumerate(tokenized_text + [None]):
        if token is not None:
            if is_chinese:
                # 对于中文，直接使用tokenizer的tokenize方法
                tokens = tokenizer.tokenize(token)
            else:
                tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
            tokenids_chunks.append([prepare_sentence.sos_id] + tokenids_chunk + [prepare_sentence.eos_id])
            if sliding_window_size > 0:
                tokenids_chunk = tokenids_chunk[-sliding_window_size:]
            else:
                tokenids_chunk = []
        if token is not None:
            tokenized_to_id_indicies.append((len(tokenids_chunks),
                                             len(tokenids_chunk),
                                             len(tokenids_chunk) + len(tokens)))
            tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

    return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


def sentence_encode(tokens_id, model, layer):
    input_ids = torch.tensor([tokens_id], device=model.device)

    with torch.no_grad():
        last_hidden_states = model(input_ids).last_hidden_state
        
    layer_embedding = tensor_to_numpy(last_hidden_states.squeeze(0))[1: -1]
    return layer_embedding


def sentence_to_wordtoken_embeddings(layer_embeddings, tokenized_text, tokenized_to_id_indicies):
    word_embeddings = []
    for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
    assert len(word_embeddings) == len(tokenized_text)
    return np.array(word_embeddings)


def handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
    layer_embeddings = [
        sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks
    ]
    word_embeddings = sentence_to_wordtoken_embeddings(layer_embeddings,
                                                       tokenized_text,
                                                       tokenized_to_id_indicies)
    return word_embeddings

def process_sentence(tokenizer, model, sentence, word=None, layer=-1, is_chinese=True):
    """
    处理句子并提取特征
    
    Args:
        tokenizer: BERT分词器
        model: BERT模型
        sentence: 输入句子
        word: 可选，句子中的特定词语，默认为None
        layer: 使用的BERT层，默认为-1（最后一层）
        is_chinese: 是否为中文模型，默认为True
    
    Returns:
        句子或词语的向量表示
    """
    # 兼容旧版本的调用方式
    # 检测第一个参数是否为token_list
    if isinstance(tokenizer, list):
        # 旧的调用方式: process_sentence(token_list, tokenizer, model, layer, is_chinese)
        token_list, tokenizer, model, layer, is_chinese = tokenizer, model, sentence, word, layer
        word = None
    else:
        # 处理输入句子
        if isinstance(sentence, str):
            if is_chinese:
                # 对于中文，按字符分词
                token_list = list(sentence)
            else:
                # 对于英文，按空格分词
                token_list = sentence.split()
        else:
            # 如果已经是token列表，直接使用
            token_list = sentence
    
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(token_list, tokenizer, is_chinese)
    contextualized_word_representations = handle_sentence(
        model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
    
    # 如果指定了特定词语，尝试找到并返回该词语的表示
    if word is not None and isinstance(word, str):
        word_found = False
        if is_chinese:
            word_chars = list(word)
            # 尝试在token_list中找到连续的子序列匹配word_chars
            for i in range(len(token_list) - len(word_chars) + 1):
                if token_list[i:i+len(word_chars)] == word_chars:
                    # 返回词语的平均表示
                    word_found = True
                    return np.mean(contextualized_word_representations[i:i+len(word_chars)], axis=0)
        else:
            if word in token_list:
                word_found = True
                idx = token_list.index(word)
                return contextualized_word_representations[idx]
        
        # 如果没有找到词语，返回整个句子的平均表示，而不是返回整个二维数组
        if not word_found:
            return np.mean(contextualized_word_representations, axis=0)
    
    # 如果没有指定词语，返回所有token的表示
    return contextualized_word_representations
    
def predict_masked_words(sentence_w_mask, model, tokenizer, top_k=50, is_chinese=False) -> list:
    """
    sentence_w_mask: `str`, a single sentence with [MASK] token
    model: a BertForMaskedLM model
    """
    inputs = tokenizer(sentence_w_mask, return_tensors="pt").to(model.device)
    masked_position_indice = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], axis=1)

    word_prob, word_index = torch.topk(probs[masked_position_indice], top_k, dim=-1)
    word_prob = tensor_to_numpy(word_prob)
    word_index = tensor_to_numpy(word_index)

    predicted_words = []
    for i in range(len(word_prob)):    
        pred_word = []
        for r, j in enumerate(word_index[i]):
            if is_chinese:
                # 对于中文，使用convert_ids_to_tokens方法
                token = tokenizer.convert_ids_to_tokens(int(j))
            else:
                token = tokenizer.ids_to_tokens[j]
            pred_word.append([token, float(word_prob[i][r])])
        predicted_words.append(pred_word)
    return predicted_words

def predict_masked_words_for_span(sentence, word_to_mask, model, tokenizer, start_pos=None, top_k=50, is_chinese=True) -> list:
    """
    对句子中的整个词进行掩码预测
    
    Args:
        sentence: 原始句子
        word_to_mask: 要掩码的词
        model: BERT模型
        tokenizer: BERT分词器
        start_pos: 要掩码的词在句子中的起始位置，如果为None则替换所有出现的词
        top_k: 返回的预测结果数量
        is_chinese: 是否为中文模型
    
    Returns:
        预测结果列表
    """
    # 计算需要多少个[MASK]来替换词语
    if is_chinese:
        # 对于中文，每个字符替换为一个[MASK]
        mask_tokens = tokenizer.mask_token * len(word_to_mask)
    else:
        mask_tokens = "[MASK]" * len(word_to_mask)
    
    # 构建掩码句子
    if start_pos is not None:
        # 只替换指定位置的词
        before = sentence[:start_pos]
        after = sentence[start_pos + len(word_to_mask):]
        masked_sent = before + mask_tokens + after
    else:
        # 替换所有出现的词
        masked_sent = sentence.replace(word_to_mask, mask_tokens)
    
    # 使用tokenizer处理掩码句子
    inputs = tokenizer(masked_sent, return_tensors="pt").to(model.device)
    
    # 找到所有[MASK]的位置
    mask_token_id = tokenizer.mask_token_id
    masked_positions = torch.where(inputs.input_ids[0] == mask_token_id)[0]
    
    # 如果没有找到掩码位置，返回空列表
    if len(masked_positions) == 0:
        return []
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 对每个掩码位置进行预测
    all_predictions = []
    for pos_idx, pos in enumerate(masked_positions):
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            token = tokenizer.convert_ids_to_tokens(int(idx))
            predictions.append([token, float(prob)])
        
        all_predictions.append(predictions)
    
    # 合并多个[MASK]的预测结果
    if len(all_predictions) > 1:
        # 尝试组合预测结果生成完整词语
        combined_predictions = []
        for i in range(min(top_k, len(all_predictions[0]))):
            combined_word = ""
            combined_score = 1.0
            for pos_preds in all_predictions:
                if i < len(pos_preds):
                    # 对于中文模型，去掉可能的##前缀
                    token = pos_preds[i][0]
                    if token.startswith('##'):
                        token = token[2:]
                    combined_word += token
                    combined_score *= pos_preds[i][1]
            
            # 计算几何平均分数
            combined_score = combined_score ** (1.0 / len(all_predictions))
            combined_predictions.append([combined_word, combined_score])
        
        # 按分数排序
        combined_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return combined_predictions[:top_k]
    else:
        return all_predictions[0]

def load_verb_sense_dict(dict_path):
    """
    加载中文动词词典
    """
    print(f"Loading verb sense dictionary from {dict_path}")
    with open(dict_path, 'r', encoding='utf-8') as f:
        verb_dict = json.load(f)
    
    # 构建lemma到sense的映射
    lemma_to_senses = {}
    for entry in verb_dict:
        lemma = entry['lemma']
        # 移除词性标记，例如 "称霸-v" -> "称霸"
        if '-' in lemma:
            lemma = lemma.split('-')[0]
        lemma_to_senses[lemma] = entry['senses']
    
    print(f"Loaded {len(lemma_to_senses)} verb entries")
    return lemma_to_senses

def main(input_file,
         lm_type,
         layer,
         top_k,
         gpu_id,
         dict_file=None):

    print("loading Transformer model")
    model_class, tokenizer_class, pretrained_weights = MODELS[lm_type]
    need_lower = (lm_type.endswith("u"))
    is_chinese = (lm_type in ['macbert', 'chinese-bert-wwm'])

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    # get model for mlm prediction
    mlm_model = model_class.from_pretrained(pretrained_weights)
    mlm_model.eval()
    mlm_model = mlm_model.to(torch.device(f'cuda:{gpu_id}'))

    # get model for embedding extraction
    model = mlm_model.bert

    # 加载中文动词词典（如果提供）
    verb_dict = None
    if dict_file and os.path.exists(dict_file):
        verb_dict = load_verb_sense_dict(dict_file)

    print("loading corpus")
    with open(input_file, "rb") as f:
        parsed_data = pk.load(f)

    save_dict_data = {}
    for sent_id in tqdm(parsed_data):
        sent_info = parsed_data[sent_id]
        if len(sent_info['svos']) == 0:
            continue

        token_list = sent_info['token_list']
        if need_lower:
            token_list = [token.lower() for token in token_list]
        token_embeds = process_sentence(tokenizer, model, token_list, None, layer, is_chinese)
        for svo_id, svo in enumerate(sent_info['svos']):
            svo_index = f"{sent_id}_{svo_id}"
            if svo[0] is not None:
                subj = svo[0][0]
                subj_range = svo[0][1]
                subj_embed = np.average(token_embeds[subj_range,:], axis=0)
                if is_chinese:
                    # 对于中文，使用整个词掩码预测
                    original_sent = "".join(token_list)
                    subj_text = "".join(token_list[subj_range[0]:subj_range[-1]+1])
                    subj_expand_res = predict_masked_words_for_span(original_sent, subj_text, model, tokenizer, subj_range[0], top_k, is_chinese)
                else:
                    # 对于英文，继续使用原来的方法
                    subj_masked_sent = " ".join(token_list[:subj_range[0]] + ["[MASK]"] + token_list[subj_range[-1]+1:])
                    subj_expand_res = predict_masked_words(subj_masked_sent, mlm_model, tokenizer, top_k, is_chinese)[0]
            else:
                subj = None
                subj_embed = None
                subj_expand_res = []
            
            verb = svo[1][0]
            verb_index = svo[1][1]
            verb_embed = token_embeds[verb_index]
            
            if is_chinese:
                # 对于中文，使用整个词掩码预测
                original_sent = "".join(token_list)
                verb_text = token_list[verb_index]
                verb_expand_res = predict_masked_words_for_span(original_sent, verb_text, model, tokenizer, verb_index, top_k, is_chinese)
            else:
                # 对于英文，继续使用原来的方法
                verb_masked_sent = " ".join(token_list[:verb_index] + ["[MASK]"] + token_list[verb_index+1:])
                verb_expand_res = predict_masked_words(verb_masked_sent, mlm_model, tokenizer, top_k, is_chinese)[0]
            
            # 如果有动词词典，添加词典中的义项信息
            verb_senses = []
            if verb_dict and verb in verb_dict:
                verb_senses = verb_dict[verb]
            
            if svo[2] is not None:
                obj = svo[2][0]
                obj_range = svo[2][1]
                obj_embed = np.average(token_embeds[obj_range,:], axis=0)
                if is_chinese:
                    # 对于中文，使用整个词掩码预测
                    original_sent = "".join(token_list)
                    obj_text = "".join(token_list[obj_range[0]:obj_range[-1]+1])
                    obj_expand_res = predict_masked_words_for_span(original_sent, obj_text, model, tokenizer, obj_range[0], top_k, is_chinese)
                else:
                    # 对于英文，继续使用原来的方法
                    obj_masked_sent = " ".join(token_list[:obj_range[0]] + ["[MASK]"] + token_list[obj_range[-1]+1:])
                    obj_expand_res = predict_masked_words(obj_masked_sent, mlm_model, tokenizer, top_k, is_chinese)[0]
            else:
                obj = None
                obj_embed = None
                obj_expand_res = []

            save_dict_data[svo_index] = {
                "subj": subj,
                "subj_embed": subj_embed,
                "subj_expansion_results": subj_expand_res,
                "verb": verb,
                "verb_embed": verb_embed,
                "verb_expansion_results": verb_expand_res,
                "verb_senses": verb_senses,  # 添加动词义项信息
                "obj": obj,
                "obj_embed": obj_embed,
                "obj_expansion_results": obj_expand_res,
            }

    save_path = f"{input_file[:-3]}_{lm_type}_expanded_l{layer}_topk_{top_k}_embeded.pk"
    with open(save_path, "wb") as f:
        pk.dump(save_dict_data, f)


if __name__ == '__main__':
    # python encode_contextual_features.py --gpu_id 7
    # python encode_contextual_features.py --input_file ./2659docs_cleaned_parsed_svo_0415.pk --gpu_id 2
    # python encode_contextual_features.py --lm_type macbert --input_file ./2659docs_cleaned_parsed_svo_0415.pk --dict_file /workspace/TWP-Gen/resources/verb_sense_dict.json --gpu_id 0
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="./20docs_cleaned_parsed_svo_0415.pk")
    parser.add_argument("--lm_type", default="blu", help="language model name")
    parser.add_argument("--lm_layer", default=-1, help="language model layer for features")
    parser.add_argument("--top_k", default=10, type=int, help="top_k expansion results")
    parser.add_argument("--gpu_id", default=0, type=int, help="gpu id for bert model")
    parser.add_argument("--dict_file", default=None, help="path to the verb sense dictionary")
    args = parser.parse_args()
    print(vars(args))
    main(args.input_file, args.lm_type, args.lm_layer, args.top_k, args.gpu_id, args.dict_file)
