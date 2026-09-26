import json
import pickle as pk
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# 定义文件路径
import twpgen_config as project_args
dataset = project_args.dataset
summarized_file = f"./dataset/{dataset}/summarized_sentences.json"
output_file = f"./dataset/{dataset}/summarized_sentences_embeddings.pk"

# 加载简化后的句子数据
print("加载简化后的句子数据...")
with open(summarized_file, 'r', encoding='utf-8') as f:
    summarized_data = json.load(f)

# 使用与原项目相同的模型 - chinese-bert-wwm
print("加载BERT模型...")
model_path = project_args.language_model_paths.get(
    'chinese-bert-wwm', './models/chinese-bert-wwm'
)
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 获取句子嵌入的函数
def get_sentence_embedding(sentence, model, tokenizer, device):
    """获取句子的BERT嵌入向量"""
    # 对句子进行分词和编码
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取BERT输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 使用[CLS]令牌的最后一层隐藏状态作为句子嵌入
    sentence_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return sentence_embedding[0]  # 返回形状为(768,)的numpy数组

# 处理所有简化后的句子
print("处理简化后的句子...")
summarized_embeddings = {}
original_sentences = {}
summarized_sentences = {}

total_sentences = len(summarized_data['sentences'])
for sent_id, info in tqdm(summarized_data['sentences'].items(), total=total_sentences):
    original_sentence = info['original']
    summarized_sentence = info['summarized']
    
    # 获取简化后句子的嵌入向量
    embedding = get_sentence_embedding(summarized_sentence, model, tokenizer, device)
    
    # 保存结果
    summarized_embeddings[sent_id] = embedding
    original_sentences[sent_id] = original_sentence
    summarized_sentences[sent_id] = summarized_sentence

# 保存结果
print(f"保存结果到 {output_file}...")
with open(output_file, 'wb') as f:
    pk.dump({
        'summarized_embeddings': summarized_embeddings,
        'original_sentences': original_sentences,
        'summarized_sentences': summarized_sentences
    }, f)

print("完成！")

# 显示一些统计信息
print(f"处理了 {len(summarized_embeddings)} 个句子")
print(f"嵌入向量维度: {list(summarized_embeddings.values())[0].shape}") 
