import json
import time
from openai import OpenAI
import os
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import argparse
import signal
import sys
import pickle as pk
import twpgen_config as project_args

# 阿里云API配置
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

# 全局变量
interrupted = False
results_lock = Lock()  # 添加锁以保护共享结果字典

def signal_handler(sig, frame):
    """处理中断信号，设置中断标志"""
    global interrupted
    print("\n检测到中断信号，将在当前处理完成后保存结果...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def summarize_sentence_task(args):
    """处理单个句子精简的任务函数，用于多线程执行"""
    sent_id, sentence = args
    max_retries = 3

    for attempt in range(max_retries):
        try:
            prompt = f"请将以下句子精简，保留关键信息，动词必须全部保留，实体也要保留，使其尽可能简短：\n\n{sentence}\n\n精简后的句子："

            response = client.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本精简助手，擅长提取句子中的关键信息，去除冗余内容，使句子更加简洁。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2  # 降低温度以获得更确定性的输出
            )

            summarized = response.choices[0].message.content.strip()

            # 基本验证：确保精简后的句子不为空
            if not summarized or summarized.strip() == "":
                summarized = sentence  # 如果精简结果为空，使用原句

            return sent_id, {
                "original": sentence,
                "summarized": summarized
            }

        except Exception as e:
            if attempt < max_retries - 1:
                # 重试前等待一段时间
                time.sleep(1 * (attempt + 1))
                continue
            else:
                # 最后一次尝试失败，返回原句
                return sent_id, {
                    "original": sentence,
                    "summarized": sentence,
                    "error": str(e),
                    "api_failed": True
                }

def process_sentences_multithread(unique_sentences, output_file, num_threads=10, batch_size=50):
    """使用多线程处理句子精简"""
    global interrupted

    results = {}
    sentence_items = list(unique_sentences.items())
    total = len(sentence_items)

    print(f"总共有 {total} 个句子需要处理，使用 {num_threads} 个线程")

    # 分批处理，避免内存占用过大
    batch_size = min(batch_size, total)

    with tqdm(total=total, desc="精简句子") as pbar:
        current_idx = 0
        while current_idx < total and not interrupted:
            end_idx = min(current_idx + batch_size, total)
            batch_sentences = sentence_items[current_idx:end_idx]

            # 使用线程池并行处理句子
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_sentence = {executor.submit(summarize_sentence_task, args): args for args in batch_sentences}

                for future in concurrent.futures.as_completed(future_to_sentence):
                    if interrupted:
                        break

                    try:
                        sent_id, result = future.result()
                        with results_lock:
                            results[sent_id] = result
                            pbar.update(1)
                    except Exception as e:
                        print(f"处理句子时出错: {e}")
                        pbar.update(1)

            # 更新当前索引
            current_idx = end_idx

            # 定期保存中间结果
            if current_idx % 100 == 0 or current_idx == total:
                with results_lock:
                    save_results(results, output_file, len(unique_sentences))

            # 在批次间添加小延迟，避免API限制
            if current_idx < total and not interrupted:
                time.sleep(0.5)  # 减少延迟时间，因为多线程已经分散了请求

            # 检查是否中断
            if interrupted:
                print(f"\n已中断处理，保存当前结果...")
                break

    # 保存最终结果
    save_results(results, output_file, len(unique_sentences))
    return results

def save_results(results, output_file, total_sentences):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_sentences": total_sentences,
            "processed_sentences": len(results),
            "sentences": results
        }, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="使用多线程精简句子")
    parser.add_argument("--input", type=str, default=project_args.parsed_corpus, help="输入文件路径（parsed_corpus.pk）")
    parser.add_argument("--output", type=str, default="./dataset/{}/summarized_sentences.json".format(project_args.dataset), help="输出文件路径")
    parser.add_argument("--threads", type=int, default=10, help="并行处理的线程数")
    parser.add_argument("--batch_size", type=int, default=10, help="每批处理的句子数")
    parser.add_argument("--use_sent_id", action="store_true", help="使用sent_id作为键（默认使用corpus_id确保完全对应）")

    args = parser.parse_args()

    # 文件路径
    input_file = args.input
    output_file = args.output

    # 加载parsed_corpus.pk数据
    print("加载parsed_corpus.pk数据...")
    try:
        with open(input_file, 'rb') as f:
            corpus_data = pk.load(f)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"错误：加载pickle文件时出错 {e}")
        return

    # 从parsed_corpus.pk中提取句子，确保完全对应
    print("从parsed_corpus.pk中提取句子...")
    unique_sentences = {}

    # 验证数据完整性
    missing_sentences = 0
    duplicate_sentences = 0

    for corpus_id, corpus_item in corpus_data.items():
        if isinstance(corpus_item, dict) and 'raw_sentence' in corpus_item:
            # 根据用户选择使用不同的ID策略
            if args.use_sent_id:
                # 使用sent_id（可能有不连续的问题）
                key_id = corpus_item.get('sent_id', corpus_id)
                print(f"使用sent_id作为键，sent_id={key_id}, corpus_id={corpus_id}")
            else:
                # 使用corpus_id确保完全对应（推荐）
                key_id = corpus_id

            sentence = corpus_item['raw_sentence']

            # 确保句子不为空
            if sentence and sentence.strip():
                if key_id in unique_sentences:
                    duplicate_sentences += 1
                    print(f"警告: 发现重复ID {key_id}")
                else:
                    unique_sentences[key_id] = sentence.strip()
            else:
                missing_sentences += 1
                print(f"警告: corpus_id {corpus_id} 的句子为空")

    print(f"从parsed_corpus.pk中提取完成:")
    print(f"  - 总条目数: {len(corpus_data)}")
    print(f"  - 有效句子数: {len(unique_sentences)}")
    print(f"  - 空句子数: {missing_sentences}")
    print(f"  - 重复ID数: {duplicate_sentences}")

    # 验证ID连续性和完整性
    if not args.use_sent_id:
        expected_ids = set(range(len(corpus_data)))
        actual_ids = set(unique_sentences.keys())
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids

        print(f"ID完整性检查:")
        print(f"  - 期望ID范围: 0 到 {len(corpus_data)-1}")
        print(f"  - 实际ID数量: {len(actual_ids)}")
        print(f"  - 缺失ID数: {len(missing_ids)}")
        print(f"  - 额外ID数: {len(extra_ids)}")

        if missing_ids:
            print(f"  - 缺失的ID: {sorted(list(missing_ids))[:10]}...")
        if extra_ids:
            print(f"  - 额外的ID: {sorted(list(extra_ids))[:10]}...")

    if len(unique_sentences) == 0:
        print("没有找到需要处理的句子")
        return

    # 使用多线程处理句子
    print("开始多线程精简句子...")
    results = process_sentences_multithread(unique_sentences, output_file, args.threads, args.batch_size)

    # 输出统计信息
    successful_count = sum(1 for r in results.values() if 'error' not in r)
    error_count = len(results) - successful_count

    print(f"\n处理完成!")
    print(f"总句子数: {len(unique_sentences)}")
    print(f"成功处理: {successful_count}")
    print(f"处理错误: {error_count}")
    if error_count > 0:
        print(f"错误率: {error_count/len(results)*100:.1f}%")
    print(f"结果已保存到: {output_file}")

    # 最终验证：确保输出文件与原始数据完全对应
    print("\n=== 最终验证 ===")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            import json
            output_data = json.load(f)

        output_sentences = output_data.get('sentences', {})

        # 验证每个ID是否对应
        correspondence_errors = 0
        for key_id in unique_sentences.keys():
            str_key = str(key_id)  # JSON中的键是字符串
            if str_key not in output_sentences:
                correspondence_errors += 1
                if correspondence_errors <= 3:
                    print(f"对应错误: ID {key_id} 在输出文件中缺失")
            else:
                # 验证原始句子是否匹配
                original_in_output = output_sentences[str_key].get('original', '')
                original_in_corpus = unique_sentences[key_id]
                if original_in_output != original_in_corpus:
                    correspondence_errors += 1
                    if correspondence_errors <= 3:
                        print(f"内容不匹配: ID {key_id}")
                        print(f"  原始: {original_in_corpus[:50]}...")
                        print(f"  输出: {original_in_output[:50]}...")

        if correspondence_errors == 0:
            print("✅ 验证成功: 输出文件与parsed_corpus.pk完全对应")
        else:
            print(f"❌ 验证失败: 发现 {correspondence_errors} 个对应错误")

    except Exception as e:
        print(f"验证过程中出错: {e}")

if __name__ == "__main__":
    main()
