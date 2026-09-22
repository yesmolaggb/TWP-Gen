import argparse
import pickle as pk
import torch
from transformers import BertForMaskedLM, BertTokenizer
from transformers import logging as hf_logging
from tqdm import tqdm
from encode_contextual_features import (
    predict_masked_words,
    predict_masked_words_for_span,
    process_sentence,
    convert_embedding_dimension,
)
import os
import traceback
import math
from multiprocessing import get_context
from queue import Empty

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


def split_list(lst, n):
    if n <= 0:
        return [lst]
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_one_sentence(
    sent_id,
    corpus,
    salient_verbs,
    salient_obj_heads,
    obj2obj_head_infos,
    tokenizer,
    mlm_model,
    model,
    top_k,
    need_lower,
    is_chinese,
):
    sent_results = {}
    all_salient_cnt = 0

    sent_info = corpus[sent_id]
    if len(sent_info['svos']) == 0:
        return sent_results, all_salient_cnt

    to_processed_svo_id_list = []
    for svo_id, svo in enumerate(sent_info['svos']):
        all_salient_flag = True
        any_salient_flag = False

        verb_lemma = svo[1][0]
        if verb_lemma.startswith("!"):
            verb_lemma = verb_lemma[1:]

        if verb_lemma in salient_verbs:
            any_salient_flag = True
        else:
            all_salient_flag = False

        if svo[2] is not None:
            obj = svo[2][0]
            obj_head_info = obj2obj_head_infos.get(obj, {})
            if len(obj_head_info) != 0:
                obj_head_lemma = obj_head_info['obj_head_lemma']
                if obj_head_lemma in salient_obj_heads:
                    any_salient_flag = True
                else:
                    all_salient_flag = False
            else:
                all_salient_flag = False
        else:
            all_salient_flag = False

        if any_salient_flag:
            to_processed_svo_id_list.append([svo_id, all_salient_flag])

        all_salient_cnt += int(all_salient_flag)

    if len(to_processed_svo_id_list) == 0:
        return sent_results, all_salient_cnt

    token_list = sent_info['token_list']
    if need_lower:
        token_list = [token.lower() for token in token_list]

    try:
        with torch.no_grad():
            token_embeds = process_sentence(token_list, tokenizer, model, -1, is_chinese)

        if token_embeds is None or not hasattr(token_embeds, 'shape'):
            print(f"警告: 句子 {sent_id} 的token_embeds无效，跳过")
            return sent_results, all_salient_cnt

        if len(token_embeds.shape) != 2 or token_embeds.shape[0] != len(token_list):
            print(
                f"警告: 句子 {sent_id} 的token_embeds形状不符合预期: "
                f"{token_embeds.shape}, 预期: ({len(token_list)}, ?)"
            )
            return sent_results, all_salient_cnt
    except Exception as e:
        print(f"处理句子 {sent_id} 时 process_sentence 出错: {e}")
        traceback.print_exc()
        return sent_results, all_salient_cnt

    for ele in to_processed_svo_id_list:
        try:
            svo_id = ele[0]
            all_salient_flag = ele[1]
            svo_index = f"{sent_id}_{svo_id}"
            svo = sent_info['svos'][svo_id]

            # verb embed
            verb_lemma = svo[1][0]
            verb_index = svo[1][1]
            if verb_lemma.startswith("!"):
                verb_lemma = verb_lemma[1:]

            if verb_lemma in salient_verbs:
                if 0 <= verb_index < len(token_embeds):
                    original_verb_embed = token_embeds[verb_index]
                    verb_embed = convert_embedding_dimension(original_verb_embed, target_dim=768)

                    with torch.no_grad():
                        if is_chinese:
                            original_sent = "".join(token_list)
                            verb_text = token_list[verb_index]
                            verb_expand_res = predict_masked_words_for_span(
                                original_sent, verb_text, mlm_model, tokenizer, verb_index, top_k, is_chinese
                            )
                        else:
                            verb_masked_sent = " ".join(
                                token_list[:verb_index] + ["[MASK]"] + token_list[verb_index + 1:]
                            )
                            verb_expand_res = predict_masked_words(
                                verb_masked_sent, mlm_model, tokenizer, top_k, is_chinese
                            )[0]
                else:
                    print(f"警告: 句子 {sent_id} 的动词索引 {verb_index} 超出范围 [0, {len(token_embeds)})")
                    verb_embed = None
                    verb_expand_res = None
            else:
                verb_embed = None
                verb_expand_res = None

            # entitys embed
            entitys = sent_info['entitys']
            entitys_embed = []
            for entity in entitys:
                try:
                    ent_name = entity[0]
                    ent_start = entity[1]
                    ent_end = entity[2]
                    ent_label = entity[3]
                    ent_embed = []
                    ent_expand_res = []
                    ent_words = []

                    for i in range(ent_start, ent_end):
                        if 0 <= i < len(token_list):
                            ent_words.append(token_list[i])

                            if 0 <= i < len(token_embeds):
                                original_ent_embed = token_embeds[i]
                                converted_ent_embed = convert_embedding_dimension(
                                    original_ent_embed, target_dim=768
                                )
                                ent_embed.append(converted_ent_embed)

                                with torch.no_grad():
                                    if is_chinese:
                                        original_sent = "".join(token_list)
                                        ent_text = token_list[i]
                                        ent_expand_res.append(
                                            predict_masked_words_for_span(
                                                original_sent, ent_text, mlm_model, tokenizer, i, top_k, is_chinese
                                            )
                                        )
                                    else:
                                        ent_masked_sent = " ".join(
                                            token_list[:i] + ["[MASK]"] + token_list[i + 1:]
                                        )
                                        ent_expand_res.append(
                                            predict_masked_words(
                                                ent_masked_sent, mlm_model, tokenizer, top_k, is_chinese
                                            )[0]
                                        )
                            else:
                                print(f"警告: 实体索引 {i} 超出嵌入向量范围 [0, {len(token_embeds)})")

                    entitys_embed.append({
                        "ent_name": ent_name,
                        "ent_label": ent_label,
                        "ent_embed": ent_embed,
                        "ent_expand_res": ent_expand_res,
                        "ent_words": ent_words,
                    })
                except Exception as e:
                    print(f"处理实体时出错: {e}")
                    continue

            # obj embed
            obj = svo[2]
            if obj is None:
                obj = None
                obj_head = None
                obj_head_index = -1
                obj_head_embed = None
                obj_head_expand_res = None
            else:
                try:
                    obj = svo[2][0]
                    obj_range = svo[2][1]
                    obj_head_info = obj2obj_head_infos.get(obj, {})

                    if len(obj_head_info) == 0:
                        obj_head = None
                        obj_head_index = None
                        obj_head_embed = None
                        obj_head_expand_res = None
                    else:
                        obj_head = obj_head_info.get("obj_head_lemma")

                        if obj_head is None:
                            obj_head_index = None
                            obj_head_embed = None
                            obj_head_expand_res = None
                        elif obj_head not in salient_obj_heads:
                            obj_head_index = None
                            obj_head_embed = None
                            obj_head_expand_res = None
                        else:
                            obj_head_index = None
                            obj_head_text = obj_head_info.get("obj_head_text", obj_head)

                            if isinstance(obj_range, list):
                                for idx in obj_range:
                                    if 0 <= idx < len(token_list) and token_list[idx] == obj_head_text:
                                        obj_head_index = idx
                                        break

                            if obj_head_index is None and isinstance(obj_range, list):
                                for idx in obj_range:
                                    if (
                                        0 <= idx < len(token_list)
                                        and (obj_head_text in token_list[idx] or token_list[idx] in obj_head_text)
                                    ):
                                        obj_head_index = idx
                                        break

                            if obj_head_index is None:
                                for i, token in enumerate(token_list):
                                    if token == obj_head_text:
                                        obj_head_index = i
                                        break

                            if (
                                obj_head_index is None
                                and "obj_head_relative_index" in obj_head_info
                                and isinstance(obj_range, list)
                            ):
                                rel_idx = obj_head_info["obj_head_relative_index"]
                                if rel_idx < len(obj_range):
                                    idx = obj_range[rel_idx]
                                    if 0 <= idx < len(token_list):
                                        obj_head_index = idx

                            if obj_head_index is None and isinstance(obj_range, list) and len(obj_range) > 0:
                                obj_head_index = obj_range[-1]
                                if obj_head_index >= len(token_list):
                                    obj_head_index = None

                            if obj_head_index is None or obj_head_index >= len(token_embeds):
                                obj_head_index = None
                                obj_head_embed = None
                                obj_head_expand_res = None
                            else:
                                original_obj_head_embed = token_embeds[obj_head_index]
                                obj_head_embed = convert_embedding_dimension(
                                    original_obj_head_embed, target_dim=768
                                )

                                with torch.no_grad():
                                    if is_chinese:
                                        original_sent = "".join(token_list)
                                        obj_head_text_for_mask = token_list[obj_head_index]
                                        obj_head_expand_res = predict_masked_words_for_span(
                                            original_sent,
                                            obj_head_text_for_mask,
                                            mlm_model,
                                            tokenizer,
                                            obj_head_index,
                                            top_k,
                                            is_chinese,
                                        )
                                    else:
                                        obj_head_masked_sent = " ".join(
                                            token_list[:obj_head_index] + ["[MASK]"] + token_list[obj_head_index + 1:]
                                        )
                                        obj_head_expand_res = predict_masked_words(
                                            obj_head_masked_sent, mlm_model, tokenizer, top_k, is_chinese
                                        )[0]
                except Exception as e:
                    print(f"处理宾语时出错: {e}")
                    obj = None
                    obj_head = None
                    obj_head_index = -1
                    obj_head_embed = None
                    obj_head_expand_res = None

            sent_results[svo_index] = {
                "verb": verb_lemma,
                "verb_embed": verb_embed,
                "verb_expansion_results": verb_expand_res,
                "obj_head": obj_head,
                "obj_head_embed": obj_head_embed,
                "obj_head_expansion_results": obj_head_expand_res,
                "entitys_embed": entitys_embed,
                "all_salient_flag": all_salient_flag,
            }

        except Exception as e:
            print(f"处理 SVO {svo_index} 时出错: {e}")
            traceback.print_exc()
            continue

    return sent_results, all_salient_cnt


def worker_main(
    worker_id,
    sent_id_chunk,
    corpus_w_svo_pickle,
    lm_type,
    top_k,
    gpu_ids,
    output_dir,
    progress_queue,
):
    hf_logging.set_verbosity_error()
    torch.set_num_threads(1)

    assigned_gpu = gpu_ids[worker_id % len(gpu_ids)]
    device = torch.device(f"cuda:{assigned_gpu}")

    model_class, tokenizer_class, pretrained_weights = MODELS[lm_type]
    need_lower = lm_type.endswith("u")
    is_chinese = lm_type in ['macbert', 'chinese-bert-wwm']

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    mlm_model = model_class.from_pretrained(pretrained_weights)
    mlm_model.eval()
    mlm_model = mlm_model.to(device)
    model = mlm_model.bert

    with open(corpus_w_svo_pickle, "rb") as f:
        corpus = pk.load(f)
    with open(f"{corpus_w_svo_pickle[:-3]}_salient_verbs.pk", "rb") as f:
        salient_verbs = pk.load(f)
    with open(f"{corpus_w_svo_pickle[:-3]}_salient_obj_heads.pk", "rb") as f:
        salient_obj_heads = pk.load(f)
    with open(f"{corpus_w_svo_pickle[:-3]}_obj2obj_head_info.pk", "rb") as f:
        obj2obj_head_infos = pk.load(f)

    svo_id2features = {}
    all_salient_cnt = 0
    error_count = 0

    for sent_id in sent_id_chunk:
        if error_count > 20:
            break

        try:
            sent_results, sent_all_salient_cnt = process_one_sentence(
                sent_id=sent_id,
                corpus=corpus,
                salient_verbs=salient_verbs,
                salient_obj_heads=salient_obj_heads,
                obj2obj_head_infos=obj2obj_head_infos,
                tokenizer=tokenizer,
                mlm_model=mlm_model,
                model=model,
                top_k=top_k,
                need_lower=need_lower,
                is_chinese=is_chinese,
            )
            svo_id2features.update(sent_results)
            all_salient_cnt += sent_all_salient_cnt
        except Exception as e:
            print(f"[Worker {worker_id} | GPU {assigned_gpu}] 处理句子 {sent_id} 时出错: {e}")
            traceback.print_exc()
            error_count += 1
        finally:
            progress_queue.put(1)

    partial_path = os.path.join(output_dir, f"partial_worker_{worker_id}.pk")
    with open(partial_path, "wb") as f:
        pk.dump({
            "svo_id2features": svo_id2features,
            "all_salient_cnt": all_salient_cnt,
            "worker_id": worker_id,
            "gpu_id": assigned_gpu,
        }, f)


def merge_partial_results(output_dir, final_save_path, num_workers):
    merged = {}
    total_all_salient_cnt = 0

    for worker_id in range(num_workers):
        partial_path = os.path.join(output_dir, f"partial_worker_{worker_id}.pk")
        if not os.path.exists(partial_path):
            continue

        with open(partial_path, "rb") as f:
            part = pk.load(f)

        merged.update(part["svo_id2features"])
        total_all_salient_cnt += part["all_salient_cnt"]

    with open(final_save_path, "wb") as f:
        pk.dump(merged, f)

    for worker_id in range(num_workers):
        partial_path = os.path.join(output_dir, f"partial_worker_{worker_id}.pk")
        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except Exception as e:
                print(f"删除临时文件失败: {partial_path}, 错误: {e}")

    try:
        if os.path.isdir(output_dir) and len(os.listdir(output_dir)) == 0:
            os.rmdir(output_dir)
    except Exception as e:
        print(f"删除临时目录失败: {output_dir}, 错误: {e}")

    print(f"共有 {total_all_salient_cnt} 个SVO三元组的谓词和宾语中心词都是显著词")
    print(f"最终结果保存到 {final_save_path}")


def main_parallel(
    corpus_w_svo_pickle,
    lm_type,
    top_k,
    gpu_ids,
    num_workers,
):
    hf_logging.set_verbosity_error()

    print("加载语料索引以切分任务")
    with open(corpus_w_svo_pickle, "rb") as f:
        corpus = pk.load(f)

    sent_ids = list(corpus.keys())
    total_sent_num = len(sent_ids)
    sent_id_chunks = split_list(sent_ids, num_workers)

    output_dir = f"{corpus_w_svo_pickle[:-3]}_parallel_tmp"
    os.makedirs(output_dir, exist_ok=True)

    ctx = get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []

    for worker_id, sent_id_chunk in enumerate(sent_id_chunks):
        p = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                sent_id_chunk,
                corpus_w_svo_pickle,
                lm_type,
                top_k,
                gpu_ids,
                output_dir,
                progress_queue,
            ),
        )
        p.start()
        processes.append(p)

    finished = 0
    with tqdm(total=total_sent_num, desc="总进度", dynamic_ncols=True) as pbar:
        while finished < total_sent_num:
            try:
                inc = progress_queue.get(timeout=1.0)
                finished += inc
                if finished > total_sent_num:
                    inc -= (finished - total_sent_num)
                    finished = total_sent_num
                if inc > 0:
                    pbar.update(inc)
            except Empty:
                alive = any(p.is_alive() for p in processes)
                if not alive:
                    break
            except Exception:
                alive = any(p.is_alive() for p in processes)
                if not alive:
                    break

    for p in processes:
        p.join()

    final_save_path = f"{corpus_w_svo_pickle[:-3]}_salient_po_mention_features.pk"
    merge_partial_results(output_dir, final_save_path, len(sent_id_chunks))


if __name__ == "__main__":
    import twpgen_config as _args_module

    _default_dataset = _args_module.dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_type", default=_default_dataset, help="corpus type")
    parser.add_argument("--lm_type", default="chinese-bert-wwm", help="language model name")
    parser.add_argument("--top_k_expand_result", default=10, type=int, help="top_k expansion results")
    parser.add_argument("--gpu_ids", nargs="+", default=[0,1], type=int, help="gpu ids, e.g. 0 1")
    parser.add_argument("--num_workers", default=16, type=int, help="number of parallel workers")
    args = parser.parse_args()

    args.parsed_corpus = f"./dataset/{args.corpus_type}/parsed_corpus.pk"

    print(vars(args))
    main_parallel(
        args.parsed_corpus,
        args.lm_type,
        args.top_k_expand_result,
        args.gpu_ids,
        args.num_workers,
    )
