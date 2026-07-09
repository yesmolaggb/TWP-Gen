import os
import pickle as pk
import json
import time
from tqdm import tqdm
from openai import OpenAI
import argparse
import signal
import sys
import concurrent.futures
from threading import Lock
import twpgen_config as project_args

# 初始化OpenAI客户端（请替换为有效API密钥）
DEFAULT_LLM_MODEL = os.environ.get("TWPGEN_LLM_MODEL", "deepseek-v3")

def build_llm_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and configure your local API key.")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

client = build_llm_client()

interrupted = False
checkpoint_data = None
results_lock = Lock()  # 添加锁以保护共享结果字典

def signal_handler(sig, frame):
    """处理中断信号，设置中断标志"""
    global interrupted
    print("\n检测到中断信号，将在当前句子处理完成后保存断点...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_parsed_corpus(file_path):
    """加载解析后的语料库"""
    print(f"加载解析后的语料库: {file_path}")
    with open(file_path, 'rb') as f:
        return pk.load(f)

def extract_events_from_sentence(sentence, svos=None, model=DEFAULT_LLM_MODEL):
    """使用大模型从句子中提取一个主要事件"""

    events = []

    # 按章节类别约束的事件抽取提示词（每句只抽一个）
    prompt = f"""请从以下句子中提取一个最核心的事件。

句子: "{sentence}"

请根据句子语义，判断它触发的事件属于以下哪个类别：
  1. 总体设计：架构、方案、模块、系统结构等设计相关事件
  2. 方法原理：算法、模型、机制、协议、公式、训练推理等原理相关事件
  3. 应用场景：业务、任务、落地、案例等应用相关事件
  4. 技术实现：接口、部署、运维、SDK、API、集成、优化等实现相关事件
  5. 评测与实验：测试、benchmark、指标、对比等评测相关事件
  6. 安全与合规：安全、隐私、合规、风险、加密、审计等安全相关事件

返回格式必须是严格的JSON格式，不要添加任何额外说明：
{{
  "event": {{
    "category": "类别编号(1-6)",
    "category_name": "类别名称",
    "type": "具体事件类型（根据句子语义自由描述）",
    "participants": [
      {{"entity": "参与者1"}},
      {{"entity": "参与者2"}}
    ]
  }}
}}

规则：
1. category 必须填入 1-6 的数字
2. category_name 必须与编号对应
3. 如果动词触发的语义与上述6类都不匹配，category 填 0，category_name 填"未分类"
4. type 应根据句子具体语境灵活描述，不是固定的标签
5. participants 只保留与事件直接相关的实体（entity），不要 role 字段"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的事件抽取助手，擅长分析文本中的核心事件。你必须始终返回严格的JSON格式，不添加任何额外的解释或Markdown标记。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        result = response.choices[0].message.content

        json_result = None

        try:
            json_result = json.loads(result)
        except json.JSONDecodeError:
            import re
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, result)
            if match:
                try:
                    json_result = json.loads(match.group(1))
                except:
                    pass

            if not json_result:
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, result)
                if match:
                    try:
                        json_result = json.loads(match.group(1))
                    except:
                        pass

            if not json_result:
                cleaned_result = result.replace('\\"', '"').replace('\\n', ' ')
                start_idx = cleaned_result.find('{')
                end_idx = cleaned_result.rfind('}')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = cleaned_result[start_idx:end_idx+1]
                    try:
                        json_result = json.loads(json_str)
                    except:
                        try:
                            json_str = json_str.replace("'", '"')
                            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
                            json_result = json.loads(json_str)
                        except:
                            pass

        if json_result and json_result.get("event"):
            event_data = json_result["event"]

            if "category" not in event_data:
                event_data["category"] = 0
                event_data["category_name"] = "未分类"

            if "category_name" not in event_data:
                if event_data["category"] in [1, 2, 3, 4, 5, 6]:
                    category_names = {1: "总体设计", 2: "方法原理", 3: "应用场景",
                                    4: "技术实现", 5: "评测与实验", 6: "安全与合规"}
                    event_data["category_name"] = category_names.get(event_data["category"], "未分类")
                else:
                    event_data["category_name"] = "未分类"

            if "type" not in event_data:
                event_data["type"] = "未知事件"

            if "participants" not in event_data or not isinstance(event_data["participants"], list):
                event_data["participants"] = []

            events.append(event_data)
        else:
            basic_event = {
                "category": 0,
                "category_name": "未分类",
                "type": "未知事件",
                "participants": [],
                "parsing_error": "自动创建的基本结构"
            }
            events.append(basic_event)

    except Exception as e:
        print(f"API调用或处理错误: {e}")
        basic_event = {
            "category": 0,
            "category_name": "未分类",
            "type": "API错误",
            "participants": [],
            "error": str(e)
        }
        events.append(basic_event)

    return {"events": events}

def process_sentence_task(args):
    """处理单个句子的任务函数，用于多线程执行"""
    sent_id, sentence, svos = args
    try:
        events = extract_events_from_sentence(sentence, svos)
        return sent_id, {
            'raw_sentence': sentence,
            'events': events.get('events', [])
        }
    except Exception as e:
        print(f"处理句子 {sent_id} 时出错: {e}")
        return sent_id, {
            'raw_sentence': sentence,
            'events': [],
            'error': str(e)
        }

def process_corpus(corpus, output_file, checkpoint_file=None, start_idx=0, max_sentences=None, num_threads=5):
    """处理语料库，提取事件（使用多线程）"""
    global interrupted, checkpoint_data
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pk.load(f)
            start_idx = checkpoint_data.get('current_idx', 0)
            results = checkpoint_data.get('results', {})
            print(f"从断点恢复，从索引 {start_idx} 继续处理")
    else:
        results = {}
    
    sentences = []
    for sent_id, sent_data in corpus.items():
        if 'raw_sentence' in sent_data and 'svos' in sent_data:
            sentences.append((sent_id, sent_data['raw_sentence'], sent_data.get('svos', [])))
    
    total = len(sentences)
    if max_sentences is not None and max_sentences > 0:
        total = min(total, max_sentences)
        sentences = sentences[:total]
    
    print(f"总共有 {total} 个句子需要处理，使用 {num_threads} 个线程")
    
    # 分批处理，每批次处理一定数量的句子
    batch_size = min(100, total)  # 每批次最多处理100个句子
    
    with tqdm(total=total-start_idx, desc="提取事件") as pbar:
        current_idx = start_idx
        while current_idx < total and not interrupted:
            end_idx = min(current_idx + batch_size, total)
            batch_sentences = sentences[current_idx:end_idx]
            
            # 使用线程池并行处理句子
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_sentence = {executor.submit(process_sentence_task, args): args for args in batch_sentences}
                
                for future in concurrent.futures.as_completed(future_to_sentence):
                    if interrupted:
                        break
                    
                    sent_id, result = future.result()
                    with results_lock:
                        results[sent_id] = result
                        pbar.update(1)
            
            # 更新当前索引
            current_idx = end_idx
            
            # 保存断点
            if checkpoint_file and not interrupted:
                checkpoint_data = {'current_idx': current_idx, 'results': results}
                with open(checkpoint_file, 'wb') as f:
                    pk.dump(checkpoint_data, f)
            
            # 检查是否中断
            if interrupted:
                checkpoint_data = {'current_idx': current_idx, 'results': results}
                with open(checkpoint_file, 'wb') as f:
                    pk.dump(checkpoint_data, f)
                print(f"\n已保存断点到 {checkpoint_file}，处理到索引 {current_idx}/{total}")
                break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已保存结果到 {output_file}")
    
    if checkpoint_file and os.path.exists(checkpoint_file) and not interrupted:
        os.remove(checkpoint_file)
        print(f"处理完成，已删除断点文件")

def test_extraction():
    """测试事件抽取功能（仅使用项目内句子）"""
    try:
        corpus_path = '/workspace/TWP-Gen/dataset/DuEE/parsed_corpus.pk'
        if not os.path.exists(corpus_path):
            print(f"错误：语料库文件 {corpus_path} 不存在")
            return
        
        corpus = load_parsed_corpus(corpus_path)
        test_sentences = []
        test_svos = []
        
        for sent_id, sent_data in list(corpus.items())[:5]:
            if 'raw_sentence' in sent_data and 'svos' in sent_data:
                test_sentences.append(sent_data['raw_sentence'])
                test_svos.append(sent_data.get('svos', []))
        
        if len(test_sentences) < 1:
            print("错误：语料库中无有效句子用于测试")
            return
        
        print("测试事件抽取功能，使用以下句子：")
        for i, (sentence, svos) in enumerate(zip(test_sentences, test_svos)):
            print(f"\n句子 {i+1}: {sentence}")
            if svos:
                print(f"SVO信息: {svos}")
            events = extract_events_from_sentence(sentence, svos)
            print(f"提取结果: {json.dumps(events, ensure_ascii=False, indent=2)}")
    
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

def process_five_sentences():
    """处理五个项目内句子并保存结果"""
    print("开始处理五个句子...")
    corpus_path = '/workspace/TWP-Gen/dataset/DuEE/parsed_corpus.pk'
    if not os.path.exists(corpus_path):
        print(f"错误：语料库文件 {corpus_path} 不存在")
        return
    
    corpus = load_parsed_corpus(corpus_path)
    sentences = []
    
    for sent_id, sent_data in list(corpus.items())[:5]:
        if 'raw_sentence' in sent_data and 'svos' in sent_data:
            sentences.append((sent_id, sent_data['raw_sentence'], sent_data.get('svos', [])))
    
    if len(sentences) < 1:
        print("错误：语料库中无有效句子用于处理")
        return
    
    if len(sentences) < 5:
        print(f"警告：语料库中仅找到 {len(sentences)} 个句子，将处理这些句子")
    
    results = {}
    stats = {
        "total_events": 0,
        "successful_events": 0,
        "fallback_events": 0,
        "api_errors": 0
    }
    
    print(f"正在处理 {len(sentences)} 个句子...")
    for sent_id, sentence, svos in sentences:
        print(f"\n处理句子 {sent_id}: {sentence}")
        print(f"SVO三元组数量: {len(svos)}")
        
        events = extract_events_from_sentence(sentence, svos)
        results[sent_id] = {
            'raw_sentence': sentence,
            'events': events.get('events', [])
        }
        
        # 统计事件处理结果
        for event in events.get('events', []):
            stats["total_events"] += 1
            if "parsing_error" in event:
                stats["fallback_events"] += 1
                print(f"  - 使用了基本事件结构: {event['trigger']}")
            elif "error" in event:
                stats["api_errors"] += 1
                print(f"  - API错误: {event['error']}")
            else:
                stats["successful_events"] += 1
                print(f"  - 成功提取事件: {event['event_type']} (触发词: {event['trigger']})")
    
    # 输出统计信息
    print("\n统计信息:")
    print(f"总事件数: {stats['total_events']}")
    print(f"成功提取事件数: {stats['successful_events']} ({stats['successful_events']/stats['total_events']*100:.1f}% 成功率)")
    print(f"使用基本结构事件数: {stats['fallback_events']} ({stats['fallback_events']/stats['total_events']*100:.1f}% 失败率)")
    print(f"API错误事件数: {stats['api_errors']} ({stats['api_errors']/stats['total_events']*100:.1f}% 错误率)")
    
    output_file = '/workspace/TWP-Gen/dataset/DuEE/project_test_events.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存结果到 {output_file}")
    
    # 保存详细的日志文件
    log_file = '/workspace/TWP-Gen/dataset/DuEE/extraction_log.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)
    
    print(f"详细日志已保存到 {log_file}")

def main():
    parser = argparse.ArgumentParser(description="从文本中提取事件")
    parser.add_argument("--topic", type=str, default=None, help="指定dataset下的题目目录名称（如：算力运维体系技术白皮书）")
    parser.add_argument("--input", type=str, default=None, help="输入语料库文件路径（不指定时自动根据--topic生成）")
    parser.add_argument("--output", type=str, default=None, help="输出事件文件路径（不指定时自动根据--topic生成）")
    parser.add_argument("--checkpoint", type=str, default=None, help="断点文件路径（不指定时自动根据--topic生成）")
    parser.add_argument("--start_idx", type=int, default=0, help="开始处理的索引")
    parser.add_argument("--max_sentences", type=int, default=-1, help="处理的最大句子数，-1表示处理所有句子")
    parser.add_argument("--test", action="store_true", help="测试事件抽取功能")
    parser.add_argument("--five", action="store_true", help="处理五个句子并保存结果")
    parser.add_argument("--threads", type=int, default=10, help="并行处理的线程数")
    
    args = parser.parse_args()
    
    # 处理 --topic 参数，自动生成路径
    if args.topic:
        dataset = args.topic
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", dataset)
        if args.input is None:
            args.input = os.path.join(base_dir, "parsed_corpus.pk")
        if args.output is None:
            args.output = os.path.join(base_dir, "extracted_events.json")
        if args.checkpoint is None:
            args.checkpoint = os.path.join(base_dir, "event_extraction_checkpoint.pk")
        print(f"指定题目: {dataset}")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")

    if args.five:
        process_five_sentences()
    elif args.test:
        test_extraction()
    else:
        # 如果没有指定 input，使用默认配置
        input_path = args.input if args.input else project_args.parsed_corpus
        output_path = args.output if args.output else os.path.join(
            os.path.dirname(project_args.parsed_corpus), "extracted_events.json"
        )
        checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(
            os.path.dirname(project_args.parsed_corpus), "event_extraction_checkpoint.pk"
        )
        corpus = load_parsed_corpus(input_path)
        max_sentences = None if args.max_sentences == -1 else args.max_sentences
        process_corpus(corpus, output_path, checkpoint_path, args.start_idx, max_sentences, args.threads)

if __name__ == "__main__":
    main()