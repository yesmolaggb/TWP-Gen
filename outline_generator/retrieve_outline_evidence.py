import re
import twpgen_config as project_args


def extract_topic_details(doc1_content):
    """从cluster.txt文件中提取每个topic的详细信息"""
    topics = {}
    current_topic = None

    for line in doc1_content.split('\n'):
        if line.startswith('Topic'):
            topic_match = re.match(r'Topic (\d+)', line)
            if topic_match:
                current_topic = f"Topic {topic_match.group(1)}"
                topics[current_topic] = []

            parts = line.split(': ', 1)
            if len(parts) < 2:
                continue

            pairs = parts[1].split(', ')
            for pair in pairs:
                # 格式: "114/动词_方向/宾语/实体[TYPE]/.../sent_id"
                pair_parts = pair.split('/')
                if len(pair_parts) < 3:
                    continue
                try:
                    sent_id = int(pair_parts[-1])  # 最后一个数字是 sent_id / corpus 行号

                    # 提取动词（去掉方向数字）和宾语
                    verb_raw = pair_parts[1] if len(pair_parts) > 1 else ''
                    verb = verb_raw.split('_')[0] if '_' in verb_raw else verb_raw
                    obj = pair_parts[2] if len(pair_parts) > 2 else ''

                    # 提取实体关键词用于验证
                    entities = []
                    for part in pair_parts[3:-1]:  # 跳过动词、宾语字段和最后的sent_id
                        if '[' in part and ']' in part:
                            entity = part.split('[')[0].strip()
                            if entity and len(entity) > 1:
                                entities.append(entity)

                    topics[current_topic].append({
                        'sent_id': sent_id,
                        'verb': verb,
                        'obj': obj,
                        'entities': entities,
                        'raw': pair
                    })
                except (ValueError, IndexError):
                    continue

    return topics


def load_corpus(corpus_file):
    """加载corpus.txt文件，返回列表（行号即为sent_id）"""
    with open(corpus_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def match_sentences(topics, corpus):
    """双保险：按 sent_id 直接索引 + 实体关键词验证"""
    topic_sentences = {}

    for topic_name, entries in topics.items():
        topic_sentences[topic_name] = []
        seen_sent_ids = set()  # 每个 topic 内去重

        for entry in entries:
            sent_id = entry['sent_id']
            entities = entry['entities']
            verb = entry.get('verb', '')
            obj = entry.get('obj', '')

            # 主匹配：直接用 sent_id 作为行索引
            if 0 <= sent_id < len(corpus):
                sentence = corpus[sent_id]
            else:
                if sent_id not in seen_sent_ids:
                    seen_sent_ids.add(sent_id)
                    topic_sentences[topic_name].append({
                        'sent_id': sent_id,
                        'verb': verb,
                        'obj': obj,
                        'sentence': f"[索引越界: {sent_id}，语料库共{len(corpus)}行]",
                        'verify': 'INDEX_OUT_OF_RANGE'
                    })
                continue

            # 去重：同一 topic 内相同 sent_id 只保留一次
            if sent_id in seen_sent_ids:
                continue
            seen_sent_ids.add(sent_id)

            # 验证：检查实体关键词是否出现在取到的句子里
            matched_keywords = [kw for kw in entities if kw in sentence]
            total_keywords = len(entities)

            if total_keywords == 0:
                verify_status = 'NO_KEYWORDS'
            elif len(matched_keywords) == total_keywords:
                verify_status = 'FULL_MATCH'
            elif len(matched_keywords) > 0:
                verify_status = f'PARTIAL_MATCH({len(matched_keywords)}/{total_keywords})'
            else:
                # 关键词一个都没命中，尝试全文搜索找最佳句子作为备选
                best_sentence = sentence
                best_score = 0
                fallback_id = sent_id
                for i, s in enumerate(corpus):
                    score = sum(1 for kw in entities if kw in s)
                    if score > best_score:
                        best_score = score
                        best_sentence = s
                        fallback_id = i

                if best_score > 0 and fallback_id != sent_id:
                    topic_sentences[topic_name].append({
                        'sent_id': sent_id,
                        'verb': verb,
                        'obj': obj,
                        'sentence': sentence,
                        'verify': 'NO_MATCH_BY_INDEX',
                        'fallback_sent_id': fallback_id,
                        'fallback_sentence': best_sentence,
                        'fallback_score': f'{best_score}/{total_keywords}'
                    })
                    continue
                else:
                    verify_status = 'NO_MATCH'

            topic_sentences[topic_name].append({
                'sent_id': sent_id,
                'verb': verb,
                'obj': obj,
                'sentence': sentence,
                'verify': verify_status
            })

    return topic_sentences


def main():
    dataset = project_args.dataset
    cluster_file = f'./dataset/{dataset}/clusters_/cluster.txt'
    corpus_file = f'./dataset/{dataset}/corpus.txt'
    output_file = f'./dataset/{dataset}/output_by_matching.txt'

    print("读取cluster.txt文件...")
    with open(cluster_file, 'r', encoding='utf-8') as f:
        doc1_content = f.read()

    print("提取Topic详细信息...")
    topics = extract_topic_details(doc1_content)

    print("加载corpus.txt文件...")
    corpus = load_corpus(corpus_file)
    print(f"语料库共 {len(corpus)} 行")

    print("双保险匹配句子...")
    topic_sentences = match_sentences(topics, corpus)

    # 统计验证结果
    stats = {'FULL_MATCH': 0, 'PARTIAL_MATCH': 0, 'NO_MATCH': 0,
             'NO_MATCH_BY_INDEX': 0, 'NO_KEYWORDS': 0, 'INDEX_OUT_OF_RANGE': 0}
    for entries in topic_sentences.values():
        for e in entries:
            v = e.get('verify', '')
            if v.startswith('PARTIAL_MATCH'):
                stats['PARTIAL_MATCH'] += 1
            elif v in stats:
                stats[v] += 1

    print("\n验证统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n输出结果...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for topic_name, entries in topic_sentences.items():
            f.write(f"{topic_name}:\n")
            for entry in entries:
                verify = entry.get('verify', '')
                verb = entry.get('verb', '')
                obj = entry.get('obj', '')
                f.write(f"  [sent_id={entry['sent_id']}][{verify}][{verb}/{obj}] {entry['sentence']}\n")
                if 'fallback_sentence' in entry:
                    f.write(f"  [备选sent_id={entry['fallback_sent_id']}][score={entry['fallback_score']}][{verb}/{obj}] {entry['fallback_sentence']}\n")
            f.write("\n")

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
