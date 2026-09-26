import json
import pickle as pk
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# 命令行参数解析
parser = argparse.ArgumentParser(description="生成事件嵌入向量")
parser.add_argument("--dataset", type=str, default=os.environ.get("TWPGEN_DATASET", "topic"),
                    help="数据集名称（对应 dataset/ 下的子目录名）")
args = parser.parse_args()
dataset = args.dataset

# 定义文件路径
events_file = f"./dataset/{dataset}/extracted_events.json"
output_file = f"./dataset/{dataset}/event_embeddings.pk"

# 加载事件数据
print("加载事件数据...")
with open(events_file, 'r', encoding='utf-8') as f:
    events_data = json.load(f)

# 使用与原项目相同的模型 - chinese-bert-wwm
print("加载BERT模型...")
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:  # central configuration: model paths, dictionary, dataset paths
    import twpgen_settings as _cfg
except Exception:  # pragma: no cover
    _cfg = None

model_path = _cfg.language_model_paths.get('chinese-bert-wwm', './models/chinese-bert-wwm') if _cfg else './models/chinese-bert-wwm'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 获取事件文本嵌入的函数
def get_event_embedding(event, model, tokenizer, device):
    """获取事件的BERT嵌入向量"""
    parts = []

    # 添加事件类别名称（章节大类）
    if "category_name" in event and event["category_name"] != "未分类":
        parts.append(event["category_name"])

    # 添加事件类型
    if "type" in event:
        parts.append(event["type"])

    event_text = " ".join(parts) if parts else "未知事件"

    # 对事件文本进行分词和编码
    inputs = tokenizer(event_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 获取BERT输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 使用[CLS]令牌的最后一层隐藏状态作为事件嵌入
    event_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return event_embedding[0]  # 返回形状为(768,)的numpy数组

# 处理所有事件
print("处理事件数据...")
event_embeddings = {}
event_texts = {}
sent_id_to_events = {}

# 遍历所有句子及其事件
for sent_id, info in tqdm(events_data.items()):
    raw_sentence = info.get("raw_sentence", "")
    events = info.get("events", [])
    
    # 如果句子有事件，处理每个事件
    if events:
        sent_events = []
        for i, event in enumerate(events):
            # 生成事件的唯一ID
            event_id = f"{sent_id}_{i}"
            
            # 获取事件嵌入向量
            embedding = get_event_embedding(event, model, tokenizer, device)
            
            # 构建事件文本表示用于后续参考
            event_text = f"[{event.get('category_name', '')}] {event.get('type', '')}"
            
            # 保存结果
            event_embeddings[event_id] = embedding
            event_texts[event_id] = event_text
            sent_events.append(event_id)
        
        # 将句子ID映射到其所有事件ID
        sent_id_to_events[sent_id] = sent_events

# 保存结果
print(f"保存结果到 {output_file}...")
with open(output_file, 'wb') as f:
    pk.dump({
        'event_embeddings': event_embeddings,
        'event_texts': event_texts,
        'sent_id_to_events': sent_id_to_events
    }, f)

print("完成！")

# 显示一些统计信息
print(f"处理了 {len(event_embeddings)} 个事件")
print(f"涉及 {len(sent_id_to_events)} 个句子")
print(f"嵌入向量维度: {list(event_embeddings.values())[0].shape if event_embeddings else None}") 
