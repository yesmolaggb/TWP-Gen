import argparse
import itertools
import math
import os
import pickle as pk
from collections import defaultdict
import twpgen_config as args
import json

import numpy as np
import torch
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

from graph_embedding_dataset import GAEDataLoader

lemmatizer = WordNetLemmatizer()

# 添加加载简化后句子嵌入向量的函数
def load_summarized_embeddings(summarized_embeddings_path):
    """加载简化后句子的嵌入向量"""
    print(f"加载简化后句子嵌入向量: {summarized_embeddings_path}")
    if not os.path.exists(summarized_embeddings_path):
        print(f"警告: 简化后句子嵌入向量文件不存在: {summarized_embeddings_path}")
        return None
    
    with open(summarized_embeddings_path, 'rb') as f:
        summarized_data = pk.load(f)
    
    return summarized_data['summarized_embeddings']

# 添加加载事件嵌入向量的函数
def load_event_embeddings(event_embeddings_path):
    """加载事件的嵌入向量"""
    print(f"加载事件嵌入向量: {event_embeddings_path}")
    if not os.path.exists(event_embeddings_path):
        print(f"警告: 事件嵌入向量文件不存在: {event_embeddings_path}")
        return None
    
    with open(event_embeddings_path, 'rb') as f:
        event_data = pk.load(f)
    
    return event_data['event_embeddings'], event_data['sent_id_to_events']


def tfidf(item_list, item2features, item2freq):
    item_doc = []
    for item in item_list:
        features = item2features[item]
        doc = ""
        for k, v in features.items():
            cnt = math.ceil(v * item2freq[item])
            doc += (k + " ") * cnt
        item_doc.append(doc)

    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X = vectorizer.fit_transform(item_doc)  # dim: [item_len, 6500]
    return X


def pca(X, pca_dim=50):
    pca_dim = min(pca_dim, X.shape[1] - 1)
    svd = TruncatedSVD(pca_dim)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print(
        "Explained variance of the SVD step: {}%".format(int(explained_variance * 100))
    )
    return X


def tfidf_pca(item_list, item2features, item2freq, pca_dim=50):
    X_init = tfidf(item_list, item2features, item2freq)
    X = pca(X_init, pca_dim=pca_dim)
    return X


def kmean_clustering_w_init_feature(item_list, X, n_clusters=10):
    km = KMeans(
        n_clusters=n_clusters, init="k-means++", max_iter=100, n_init=5, random_state=42
    )
    km.fit(X)
    cluster_id2items = {}
    item2center_distance = km.transform(X)
    for row_id, item in enumerate(item_list):
        cluster_id = km.labels_[row_id]
        if cluster_id not in cluster_id2items:
            cluster_id2items[cluster_id] = {}
        distance = item2center_distance[row_id, cluster_id]
        cluster_id2items[cluster_id][item] = distance
    return km, cluster_id2items


def main(
    mention_file,
    sense_mapping,
    save_file,
    use_all_svos,
    pca_dim,
    entity_strategy,
    gae_feature_path,
    ve_graph_path,
    corpus_info_path,
    feature_path,
    summarized_embeddings_path=None,  # 简化后句子嵌入向量路径
    event_embeddings_path=None,  # 新增参数：事件嵌入向量路径
):

    print("=== Loading Data ===")
    with open(mention_file, "rb") as f:
        po_mentions = pk.load(f)
    with open(sense_mapping, "rb") as f:
        svo_id2sense_id = pk.load(f)
    gae_feature = np.loadtxt(gae_feature_path, delimiter=",")
    gae_dataloader = GAEDataLoader(ve_graph_path, corpus_info_path, feature_path)
    
    # 加载简化后句子嵌入向量（如果提供了路径）
    summarized_embeddings = None
    if summarized_embeddings_path and os.path.exists(summarized_embeddings_path):
        summarized_embeddings = load_summarized_embeddings(summarized_embeddings_path)
        print(f"成功加载简化后句子嵌入向量，共 {len(summarized_embeddings)} 个")
        
    # 加载事件嵌入向量（如果提供了路径）
    event_embeddings = None
    sent_id_to_events = None
    if event_embeddings_path and os.path.exists(event_embeddings_path):
        event_embeddings, sent_id_to_events = load_event_embeddings(event_embeddings_path)
        print(f"成功加载事件嵌入向量，共 {len(event_embeddings)} 个")

    print("=== Collecting P-O Mention Features ===")
    sense2embed_list = defaultdict(list)
    sense2expd_lemma_w_weight = {}
    sense2freq = defaultdict(int)
    obj_head2embed_list = defaultdict(list)
    obj_head2expd_lemma_w_weight = {}
    obj_head2freq = defaultdict(int)
    ents2embed_list = defaultdict(list)
    ents2expd_lemma_w_weight = {}
    ents2freq = defaultdict(int)

    processed_svo_cnt = 0
    for svo_id, svo_info in tqdm(po_mentions.items()):
        if not use_all_svos and (not svo_info["all_salient_flag"]):
            continue
        verb = svo_info["verb"]
        verb_embed = svo_info["verb_embed"]
        verb_expd_results = svo_info["verb_expansion_results"]
        obj_head = svo_info["obj_head"]
        obj_head_embed = svo_info["obj_head_embed"]
        obj_head_expd_results = svo_info["obj_head_expansion_results"]
        entitys_embed = svo_info["entitys_embed"]
        # process verb
        if verb_embed is not None:  # not salient verb
            expd_lemma2weight = defaultdict(float)
            for ele in verb_expd_results:
                expd_lemma = lemmatizer.lemmatize(ele[0], "v")
                expd_lemma2weight[expd_lemma] += ele[1]
            expd_lemma2weight = dict(expd_lemma2weight)

            sense_id = svo_id2sense_id[svo_id]
            sense = f"{verb}_{sense_id}"  # e.g. warn_0
            sense2embed_list[sense].append(verb_embed)
            sense2freq[sense] += 1
            if sense not in sense2expd_lemma_w_weight:
                sense2expd_lemma_w_weight[sense] = defaultdict(float)
            for expd_lemma, weight in expd_lemma2weight.items():
                sense2expd_lemma_w_weight[sense][expd_lemma] += weight

        # process obj (head)
        if obj_head_embed is not None:  # not salient object head

            expd_lemma2weight = defaultdict(float)
            for ele in obj_head_expd_results:
                expd_lemma = lemmatizer.lemmatize(ele[0])
                expd_lemma2weight[expd_lemma] += ele[1]
            expd_lemma2weight = dict(expd_lemma2weight)

            obj_head2embed_list[obj_head].append(obj_head_embed)
            obj_head2freq[obj_head] += 1
            if obj_head not in obj_head2expd_lemma_w_weight:
                obj_head2expd_lemma_w_weight[obj_head] = defaultdict(float)
            for expd_lemma, weight in expd_lemma2weight.items():
                obj_head2expd_lemma_w_weight[obj_head][expd_lemma] += weight

        # process ents
        if len(entitys_embed) != 0:
            for ent_embed_info in entitys_embed:
                ent_name = ent_embed_info["ent_name"]
                ent_word = ent_embed_info["ent_words"]
                # ent_label = ent_embed_info['ent_label']
                ent_embed = ent_embed_info["ent_embed"]
                ent_expand_res = ent_embed_info["ent_expand_res"]
                # 找ent word中每个expd的词元，相同词元累加，相当于去重一下
                # 每个实体可能由多个单词组成，遍历每个单词的expand res
                ent2expd_lemma_w_weight = defaultdict(list)
                for i, word_expand_res in enumerate(ent_expand_res):
                    expd_lemma2weight = defaultdict(float)
                    for exp in word_expand_res:
                        expd_lemma = lemmatizer.lemmatize(exp[0])
                        expd_lemma2weight[expd_lemma] += exp[1]
                    expd_lemma2weight = dict(expd_lemma2weight)
                    ent2expd_lemma_w_weight[ent_word[i]] = expd_lemma2weight

                ent2expd_lemma_w_weight = dict(ent2expd_lemma_w_weight)
                if ent_name not in ents2expd_lemma_w_weight:
                    ents2expd_lemma_w_weight[ent_name] = {}
                for word, word2expd_lemma_w_weight in ent2expd_lemma_w_weight.items():
                    if word not in ents2expd_lemma_w_weight[ent_name]:
                        ents2expd_lemma_w_weight[ent_name][word] = defaultdict(float)
                    for expd_lemma, weight in word2expd_lemma_w_weight.items():
                        ents2expd_lemma_w_weight[ent_name][word][expd_lemma] += weight

                ents2embed_list[ent_name].append(ent_embed)
                ents2freq[ent_name] += 1
        processed_svo_cnt += 1

    print(f"Processed {processed_svo_cnt} SVO triplets")
    # 获取verb with sense的均值embed 和 对应预测词元权重
    sense2freq = dict(sense2freq)
    sense2embed = {
        k: np.average(v, axis=0)
        for k, v in sense2embed_list.items()
        if k in sense2embed_list
    }
    sense2expd_lemma_w_weight = {
        k: dict(v)
        for k, v in sense2expd_lemma_w_weight.items()
        if k in sense2expd_lemma_w_weight
    }
    # 获取obj_head 的均值embed 和 对应预测词元权重
    obj_head2freq = dict(obj_head2freq)
    obj_head2embed = {
        k: np.average(v, axis=0)
        for k, v in obj_head2embed_list.items()
        if k in obj_head2freq
    }
    obj_head2expd_lemma_w_weight = {
        k: dict(v)
        for k, v in obj_head2expd_lemma_w_weight.items()
        if k in obj_head2expd_lemma_w_weight
    }
    # 获取每个entity的均值embed和对应预测词元权重
    ents2embed_list = dict(ents2embed_list)
    ents2freq = dict(ents2freq)
    ents2embed = defaultdict()
    for k, v in ents2embed_list.items():
        v = np.average(v, axis=0)
        v = np.average(v, axis=0)
        # nan_mask = np.isnan(v)
        # if nan_mask.max == True:
        #     print(1)
        ents2embed[k] = v
    ents2embed = dict(ents2embed)
    # embed = np.average(, axis=0)
    # token_arr = np.array(token_embeds)
    # nan_mask = np.isnan(token_embeds)
    # nan_rows = np.any(nan_mask, axis=1)
    # if nan_rows.max == True:
    #         print(sent_id)

    for ent_name, ent2expd_lemma_w_weight in ents2expd_lemma_w_weight.items():
        for word, word_expd_lemma_w_weight in ent2expd_lemma_w_weight.items():
            if word in ent2expd_lemma_w_weight:
                ent2expd_lemma_w_weight[word] = dict(ent2expd_lemma_w_weight[word])
        if ent_name in ents2expd_lemma_w_weight:
            ents2expd_lemma_w_weight[ent_name] = dict(
                ents2expd_lemma_w_weight[ent_name]
            )

    print("=== Getting Seperate Features for Predicate Senses and Object Heads ===")
    # 获取所有verb with sense以及inversed verb with sense，便于索引
    selected_vs_list = list(sense2freq.keys())
    inv_vs_vocab = {vs: i for i, vs in enumerate(selected_vs_list)}  # debug. len = 368
    selected_oh_list = list(obj_head2freq.keys())
    inv_oh_vocab = {oh: i for i, oh in enumerate(selected_oh_list)}  # debug. len = 121
    selected_ent_list = list(ents2freq.keys())
    inv_ent_list = {ent: i for i, ent in enumerate(selected_ent_list)}

    # graph_embeds = []

    selected_vs_expd_lemma_w_weight = {
        vs: sense2expd_lemma_w_weight[vs] for vs in selected_vs_list
    }
    print("Get verb sense context expansion embedding")
    vs_context_embed = tfidf_pca(
        selected_vs_list, selected_vs_expd_lemma_w_weight, sense2freq, pca_dim=pca_dim
    )
    print("Get verb sense BERT embedding")
    selected_vs_bert_embed = np.array([sense2embed[vs] for vs in selected_vs_list])
    vs_bert_embed = pca(selected_vs_bert_embed, pca_dim=pca_dim)
    print("Merge and get verb sense final embedding")
    # vs_final_embed = pca(np.concatenate([vs_bert_embed, vs_context_embed], axis=1), math.ceil(pca_dim*1.5))
    vs_final_embed = np.concatenate(
        [vs_bert_embed, vs_context_embed], axis=1
    )  # text encoder embedding

    selected_oh_expd_lemma_w_weight = {
        oh: obj_head2expd_lemma_w_weight[oh] for oh in selected_oh_list
    }
    print("Get object head context expansion embedding")
    oh_context_embed = tfidf_pca(
        selected_oh_list,
        selected_oh_expd_lemma_w_weight,
        obj_head2freq,
        pca_dim=pca_dim,
    )
    print("Get object head BERT embedding")
    selected_oh_bert_embed = np.array([obj_head2embed[oh] for oh in selected_oh_list])
    oh_bert_embed = pca(selected_oh_bert_embed, pca_dim=pca_dim)
    print("Merge and get object head final embedding")
    # oh_final_embed = pca(np.concatenate([oh_bert_embed, oh_context_embed], axis=1), math.ceil(pca_dim*1.5))
    oh_final_embed = np.concatenate([oh_bert_embed, oh_context_embed], axis=1)

    # select entity
    ents2expd_lemma_w_weight_concat = defaultdict()
    for ent, ent_expd_lemma_w_weight in ents2expd_lemma_w_weight.items():
        ent_expd_lemma_w_weight_concat = defaultdict(float)
        words_weight_list = []
        words_list = []
        spacer = " "
        if len(ent.split(" ")) == 1:
            spacer = ""

        # 限制最大组合数
        MAX_COMBINATIONS = 100  # 设置最大组合数量
        total_combinations = 1  # 初始组合数

        for word, word_expd_lemma_w_weight in ent_expd_lemma_w_weight.items():
            word_list = []
            weight_list = []
            # 限制每个词的扩展数量，优先保留高权重的扩展
            if len(word_expd_lemma_w_weight) > 5:
                # 按权重排序并只保留前5个
                sorted_items = sorted(word_expd_lemma_w_weight.items(), key=lambda x: x[1], reverse=True)
                sorted_items = sorted_items[:5]
            else:
                sorted_items = word_expd_lemma_w_weight.items()
                
            for expd_lemma, weight in sorted_items:
                if len(word_list) < 3 or weight >= 1:
                    word_list.append(expd_lemma)
                    weight_list.append(weight)
                    
            if len(words_list) == 0:
                words_list = word_list
                words_weight_list = weight_list
            else:
                # 计算笛卡尔积后的组合数量
                potential_combinations = len(words_list) * len(word_list)
                
                # 如果组合数量过大，选择性减少
                if potential_combinations > MAX_COMBINATIONS:
                    print(f"实体 '{ent}' 的组合数量 {potential_combinations} 超过限制 {MAX_COMBINATIONS}，将进行截断")
                    # 减少word_list或words_list的大小
                    if len(word_list) > 3:
                        word_list = word_list[:3]
                        weight_list = weight_list[:3]
                    if len(words_list) > 3 and len(word_list) > 3:
                        words_list = words_list[:3]
                        words_weight_list = words_weight_list[:3]
                
                # 安全地计算笛卡尔积
                new_words_list = []
                new_weights_list = []
                for i, w1 in enumerate(words_list):
                    for j, w2 in enumerate(word_list):
                        if len(new_words_list) >= MAX_COMBINATIONS:
                            break
                        new_words_list.append(spacer.join([w1, w2]))
                        new_weights_list.append(words_weight_list[i] + weight_list[j])
                    if len(new_words_list) >= MAX_COMBINATIONS:
                        break
                        
                # 更新列表，确保不超过最大组合数
                words_list = new_words_list[:MAX_COMBINATIONS]
                words_weight_list = new_weights_list[:MAX_COMBINATIONS]
                
        # 限制最终结果的大小
        if len(words_list) > MAX_COMBINATIONS:
            # 按权重排序并只保留最重要的组合
            sorted_items = sorted(zip(words_list, words_weight_list), 
                                 key=lambda x: x[1], reverse=True)
            words_list = [item[0] for item in sorted_items[:MAX_COMBINATIONS]]
            words_weight_list = [item[1] for item in sorted_items[:MAX_COMBINATIONS]]
            
        for i, word in enumerate(words_list):
            ent_expd_lemma_w_weight_concat[word] = words_weight_list[i]
        ent_expd_lemma_w_weight_concat = dict(ent_expd_lemma_w_weight_concat)
        ents2expd_lemma_w_weight_concat[ent] = ent_expd_lemma_w_weight_concat

    ents2expd_lemma_w_weight_concat = dict(ents2expd_lemma_w_weight_concat)
    selected_ent_expd_lemma_w_weight = {
        ent: ents2expd_lemma_w_weight_concat[ent] for ent in selected_ent_list
    }
    print("Get entitys context expansion embedding")
    ent_context_embed = tfidf_pca(
        selected_ent_list,
        selected_ent_expd_lemma_w_weight,
        ents2freq,
        pca_dim=pca_dim,
    )
    print("Get entitys BERT embedding")
    selected_ent_bert_embed = np.array([ents2embed[ent] for ent in selected_ent_list])
    selected_ent_bert_embed = np.nan_to_num(selected_ent_bert_embed)
    # nan_mask = np.isnan(selected_ent_bert_embed)
    # nan_rows = np.any(nan_mask, axis=1)
    # selected_ent_bert_embed = selected_ent_bert_embed[~nan_rows, :]
    # if nan_rows.max == True:
    #     print(1)
    # nan_rows = np.any(nan_mask, axis=1)
    # selected_ent_bert_embed = selected_ent_bert_embed[~nan_rows, :]

    ent_bert_embed = pca(selected_ent_bert_embed, pca_dim=pca_dim)
    print("Merge and get entitys final embedding")
    ent_final_embed = np.concatenate([ent_bert_embed, ent_context_embed], axis=1)

    # get sent nodes
    nodes = gae_dataloader.words_graph["nodes"]
    name2id = gae_dataloader.name2id
    base_nodes_id = gae_dataloader.base_nodes_id

    sent2nodes = defaultdict(list)
    for node_name, sentid_list in nodes.items():
        for sid in sentid_list:
            if sid < 0:
                continue
            sent2nodes[sid].append(name2id[node_name])

    print("=== Getting P-O Tuple Features ===")
    vocab = {}
    inv_vocab = {}
    vs_emb = []
    oh_emb = []
    ents_emb = []
    sents_id = []
    sents_graph_emd = []
    ents_vocab = []
    vs_w_oh2freq = defaultdict(int)
    
    # 创建节点ID到GAE特征索引的安全映射
    node_to_gae_index = {}
    for i, node_id in enumerate(base_nodes_id):
        if i < len(gae_feature):
            node_to_gae_index[node_id] = i
    
    # 计算所有有效GAE特征的平均值，用作后备特征
    if len(gae_feature) > 0:
        mean_feature = np.mean(gae_feature, axis=0)
    else:
        # 如果没有特征，使用零向量
        print("警告: GAE特征为空，使用零向量替代")
        mean_feature = np.zeros(256)  # 假设GAE特征维度为256，根据实际情况调整
    
    print(f"共有 {len(node_to_gae_index)} 个节点ID映射到GAE特征索引（共 {len(gae_feature)} 个特征向量）")
    
    for svo_id, svo_info in tqdm(po_mentions.items()):
        sent_id = svo_id.split("_")[0]
        sense_id = svo_id2sense_id[svo_id]  # svo_id: {sent_id}_{svo_id}
        vs = f"{svo_info['verb']}_{sense_id}"  # contain_0
        oh = svo_info["obj_head"]
        entitys_embed = svo_info["entitys_embed"]
        # 过滤vs和oh不全的
        if vs not in selected_vs_list or oh not in selected_oh_list:
            continue

        vs_oh_tuple = (vs, oh)
        if vs_oh_tuple not in vocab and len(entitys_embed) != 0:
            index = len(vocab)
            vocab[vs_oh_tuple] = index
            inv_vocab[index] = vs_oh_tuple
            vs_emb.append(vs_final_embed[inv_vs_vocab[vs], :])
            oh_emb.append(oh_final_embed[inv_oh_vocab[oh], :])
            sents_id.append(sent_id)

            ## 获取图嵌入
            sent_nodes = sent2nodes[int(sent_id)]
            sent_graph_embed = []
            found_embeddings = 0
            
            # 遍历句子中的所有节点
            for node_id in sent_nodes:
                # 如果节点在映射中，直接使用对应特征
                if node_id in node_to_gae_index:
                    feat_idx = node_to_gae_index[node_id]
                    sent_graph_embed.append(gae_feature[feat_idx])
                    found_embeddings += 1
            
            # 如果找不到任何特征，使用均值特征
            if not sent_graph_embed:
                if int(sent_id) < 100:  # 只对部分句子输出日志，避免过多输出
                    print(f"信息: 句子ID {sent_id} 没有找到任何匹配的GAE特征，使用平均特征")
                sent_graph_embed = [mean_feature]
            
            # 对收集的特征求平均
            sent_graph_embed = np.array(sent_graph_embed).astype(np.float32)
            sent_graph_embed = np.average(sent_graph_embed, axis=0)
            sents_graph_emd.append(sent_graph_embed)

            ent_emb = []
            ent_vocab = []
            for ent_info in entitys_embed:
                ent_name = ent_info["ent_name"]
                ent_label = ent_info["ent_label"]
                ent_vocab.append(ent_name + "[" + ent_label + "]")
                emb = ent_final_embed[inv_ent_list[ent_name], :]
                ent_emb.extend(emb)
            # ent_emb = np.array(ent_emb).astype(np.float32)
            ents_emb.append(ent_emb)
            ents_vocab.append(ent_vocab)

        vs_w_oh2freq[vs_oh_tuple] += 1

    # pca降维 ents_emb
    print("Using PAC to ents_emb")
    max_len = max(len(l) for l in ents_emb)
    for ent_emb in tqdm(ents_emb):
        ent_emb.extend([0] * (max_len - len(ent_emb)))
    ents_emb = np.array(ents_emb).astype(np.float32)
    ents_emb = pca(ents_emb, pca_dim=1000)

    vs_w_oh2freq = dict(vs_w_oh2freq)
    vs_emb = np.array(vs_emb).astype(np.float32)
    oh_emb = np.array(oh_emb).astype(np.float32)
    oh_emb = np.concatenate((oh_emb, ents_emb), axis=1)
    sents_graph_emd = np.array(sents_graph_emd).astype(np.float32)
    
    # 添加简化后句子嵌入向量
    summarized_emb = []
    if summarized_embeddings:
        print("添加简化后句子嵌入向量...")
        missing_count = 0
        found_count = 0
        
        for sent_id in tqdm(sents_id):
            if sent_id in summarized_embeddings:
                # 找到对应的简化句子嵌入向量
                embedding = summarized_embeddings[sent_id]
                summarized_emb.append(embedding)
                found_count += 1
            else:
                # 如果没有找到，使用零向量
                embedding_dim = 768  # BERT嵌入维度
                summarized_emb.append(np.zeros(embedding_dim))
                missing_count += 1
        
        print(f"找到 {found_count} 个简化句子嵌入向量")
        print(f"缺失 {missing_count} 个简化句子嵌入向量")
        
        # 转换为numpy数组并降维
        summarized_emb = np.array(summarized_emb).astype(np.float32)
        print(f"简化句子嵌入向量原始维度: {summarized_emb.shape}")
        
        # 对简化句子嵌入向量进行PCA降维，与其他特征保持一致
        summarized_emb = pca(summarized_emb, pca_dim=pca_dim)
        print(f"简化句子嵌入向量降维后维度: {summarized_emb.shape}")
    else:
        print("未提供简化后句子嵌入向量，使用零向量代替")
        # 创建与句子数量相同的零向量
        summarized_emb = np.zeros((len(sents_id), pca_dim)).astype(np.float32)
        
    # 添加事件嵌入向量
    event_emb = []
    if event_embeddings and sent_id_to_events:
        print("添加事件嵌入向量...")
        missing_count = 0
        found_count = 0
        
        for sent_id in tqdm(sents_id):
            if sent_id in sent_id_to_events and sent_id_to_events[sent_id]:
                # 如果句子包含多个事件，取平均
                event_vectors = []
                for event_id in sent_id_to_events[sent_id]:
                    if event_id in event_embeddings:
                        event_vectors.append(event_embeddings[event_id])
                
                if event_vectors:
                    # 对该句子的所有事件嵌入向量取平均
                    avg_event_emb = np.mean(event_vectors, axis=0)
                    event_emb.append(avg_event_emb)
                    found_count += 1
                else:
                    # 如果找不到事件嵌入向量，使用零向量
                    embedding_dim = 768  # BERT嵌入维度
                    event_emb.append(np.zeros(embedding_dim))
                    missing_count += 1
            else:
                # 如果句子没有事件，使用零向量
                embedding_dim = 768  # BERT嵌入维度
                event_emb.append(np.zeros(embedding_dim))
                missing_count += 1
        
        print(f"找到 {found_count} 个事件嵌入向量")
        print(f"缺失 {missing_count} 个事件嵌入向量")
        
        # 转换为numpy数组并降维
        event_emb = np.array(event_emb).astype(np.float32)
        print(f"事件嵌入向量原始维度: {event_emb.shape}")
        
        # 对事件嵌入向量进行PCA降维，与其他特征保持一致
        event_emb = pca(event_emb, pca_dim=pca_dim)
        print(f"事件嵌入向量降维后维度: {event_emb.shape}")
    else:
        print("未提供事件嵌入向量，使用零向量代替")
        # 创建与句子数量相同的零向量
        event_emb = np.zeros((len(sents_id), pca_dim)).astype(np.float32)

    tuple_freq = []
    for i in range(len(inv_vocab)):
        tuple_freq.append(vs_w_oh2freq[inv_vocab[i]])
    print(f"Total number of distinct (verb sense, object head) pairs: {len(vocab)}")

    if os.path.exists(save_file):
        os.remove(save_file)
    with open(save_file, "wb") as f:
        pk.dump(
            {
                "vocab": vocab,
                "inv_vocab": inv_vocab,
                "vs_emb": vs_emb,
                "oh_emb": oh_emb,
                "tuple_freq": tuple_freq,
                "sents_id": sents_id,
                "ents_emb": ents_emb,
                "ents_vocab": ents_vocab,
                "sents_graph_emb": sents_graph_emd,
                "summarized_emb": summarized_emb,  # 简化后句子嵌入向量
                "event_emb": event_emb,  # 添加事件嵌入向量
            },
            f,
        )
    
    # 打印各个特征的维度信息
    print("\n特征维度信息:")
    print(f"vs_emb: {vs_emb.shape}")
    print(f"oh_emb: {oh_emb.shape}")
    print(f"ents_emb: {ents_emb.shape}")
    print(f"sents_graph_emb: {sents_graph_emd.shape}")
    print(f"summarized_emb: {summarized_emb.shape}")
    print(f"event_emb: {event_emb.shape}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--mention_file", help="input <predicate, object head> mentions pickle file"
    # )
    # parser.add_argument(
    #     "--sense_mapping", help="input predicate sense disambiguation result file"
    # )
    # parser.add_argument("--save_file", help="output file name")
    # parser.add_argument(
    #     "--use_all_svos",
    #     default=False,
    #     action="store_true",
    #     help="""whether to use all po mentions (i.e., those contain either a salient predicate or an objct head 
    #             or a strict set of po mentions (i.e., those contain both a salient predicate and an object head)""",
    # )
    # parser.add_argument("--pca_dim", default=500, help="reduced dimensionality")
    # parser.add_argument(
    #     "--entity_strategy",
    #     default="avg",
    #     help="strategy of compute entity bert embedding",
    # )
    # parser.add_argument("--ve_graph_path", default="./dataset/ACE_2005/ve_graph.pk")
    # parser.add_argument(
    #     "--corpus_info_path", default="./dataset/ACE_2005/corpus_info.pk"
    # )
    # parser.add_argument("--feature_path", default=args.feature_path)
    # parser.add_argument("--gae_feature_path", default=args.gae_feature_path)
    # args = parser.parse_args()
    # print(vars(args))

    # debug
    # python3 build_tuple_feature_bank.py \
    # --mention_file ./covid19/corpus_parsed_svo_salient_po_mention_features.pk \
    # --sense_mapping ./covid19/po_mention_disambiguated.pk \
    # --save_file ./covid19/po_tuple_features_all_svos.pk \
    # --use_all_svos
    # main(
    #     "./covid19/corpus_parsed_svo_salient_po_mention_features.pk",
    #     "./covid19/po_mention_disambiguated.pk",
    #     "./covid19/po_tuple_features_all_svos.pk",
    #     True,
    #     500,
    # )

    # 添加简化后句子嵌入向量路径
    dataset = args.dataset
    summarized_embeddings_path = f"./dataset/{dataset}/summarized_sentences_embeddings.pk"
    
    # 检查文件是否存在
    if os.path.exists(summarized_embeddings_path):
        print(f"找到简化后句子嵌入向量文件: {summarized_embeddings_path}")
    else:
        print(f"警告: 简化后句子嵌入向量文件不存在: {summarized_embeddings_path}")
        print("请先运行 encode_sentence_summaries.py 生成简化后句子嵌入向量")
        summarized_embeddings_path = None
        
    # 添加事件嵌入向量路径
    event_embeddings_path = f"./dataset/{dataset}/event_embeddings.pk"
    
    # 检查文件是否存在
    if os.path.exists(event_embeddings_path):
        print(f"找到事件嵌入向量文件: {event_embeddings_path}")
    else:
        print(f"警告: 事件嵌入向量文件不存在: {event_embeddings_path}")
        print("请先运行 encode_event_features.py 生成事件嵌入向量")
        event_embeddings_path = None

    main(
        args.mention_file,
        args.save_disambiguated_path,
        args.save_po_tuple_feature_path,
        args.use_all_svos,
        500,
        "avg",
        args.gae_feature_path,
        args.graph_path,
        args.corpus_info_path,
        args.feature_path,
        summarized_embeddings_path,  # 简化后句子嵌入向量路径
        event_embeddings_path,  # 添加事件嵌入向量路径
    )
