import itertools
import os
import pickle as pk
from dataclasses import dataclass

import torch
import spherical_topic_clustering as latent_space_clustering
import baseline_topic_clustering as latent_space_clustering_baseline
from evaluate_topic_clusters import Evaluator
import twpgen_config as ARGS
import numpy as np

enumArgs = {
    "n_clusters": [60],
    "batch_size": [512],
    # "agg_method": ["concat", "sum", "multi"],
    "agg_method": ["sum"],
    "pretrain_epoch": [1000],
    "sep_decode": [0],
    "lr": [5e-4],
    # "sort_method": ["generative", "discriminative"],
    "sort_method": ["discriminative"],
    # "distribution": ["softmax", "student"],
    "distribution": ["student"],
    # "use_freq": [False, True],
    "use_freq": [True],
    "gamma": [5],  # weight of clustering loss
    "update_interval": [100],
    "tol": [0.001],  # tolerance threshold
    "temperature": [0.1],
}


@dataclass(init=True)
class Arg:
    n_clusters: int
    batch_size: int
    agg_method: str
    pretrain_epoch: str
    sep_decode: int
    lr: float
    sort_method: str
    distribution: str
    use_freq: bool
    gamma: int
    update_interval: int
    tol: float
    temperature: float

    dataset = ARGS.dataset
    input_emb_name = ARGS.save_po_tuple_feature_path

    # 默认值只作为兜底，实际会在读取特征后自动覆盖
    input_dim1 = 1000
    input_dim2 = 2000
    input_dim3 = 256
    input_dim4 = 500
    input_dim5 = 500

    hidden_dims = "[1000, 1000, 1000, 100]"
    load_pretrain = False
    suffix = ""
    # model = "Baseline"
    model = "MyModel"


@dataclass(init=True)
class Metrics:
    ari: float = 0
    nmi: float = 0
    acc: float = 0
    bcubed_f1: float = 0

    def isEqual(self, var2):
        if var2.ari != self.ari:
            return False
        if var2.nmi != self.nmi:
            return False
        if var2.acc != self.acc:
            return False
        if var2.bcubed_f1 != self.bcubed_f1:
            return False
        return True


keys = [
    lambda x: x.ari,
    lambda x: x.nmi,
    lambda x: x.acc,
    lambda x: x.bcubed_f1,
]

keysName = ["ari", "nmi", "acc", "f1"]
nonevalue = [None] * len(enumArgs)
best_args = [Arg(*nonevalue) for _ in range(len(keys))]
best_metrics = [Metrics() for _ in range(len(keys))]
all_metrics = []


def adjust_input_dims_by_current_emb(args, emb_dict):
    """
    根据当前特征文件里的真实维度，自动调整 input_dim1~5
    映射关系：
      input_dim1 <- vs_emb
      input_dim2 <- oh_emb
      input_dim3 <- sents_graph_emb
      input_dim4 <- summarized_emb
      input_dim5 <- event_emb
    缺失时保留原默认值。
    """
    def get_dim(key, old_dim):
        if key not in emb_dict:
            print(f"警告: {key} 不存在，保留默认维度 {old_dim}")
            return old_dim

        value = emb_dict[key]
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)

        if value.ndim != 2:
            raise ValueError(f"{key} 不是二维特征，当前 shape={value.shape}")

        return value.shape[1]

    args.input_dim1 = get_dim("vs_emb", args.input_dim1)
    args.input_dim2 = get_dim("oh_emb", args.input_dim2)
    args.input_dim3 = get_dim("sents_graph_emb", args.input_dim3)
    args.input_dim4 = get_dim("summarized_emb", args.input_dim4)
    args.input_dim5 = get_dim("event_emb", args.input_dim5)

    print(
        f"自动调整后的输入维度: "
        f"input_dim1={args.input_dim1}, "
        f"input_dim2={args.input_dim2}, "
        f"input_dim3={args.input_dim3}, "
        f"input_dim4={args.input_dim4}, "
        f"input_dim5={args.input_dim5}"
    )


def infer_num_samples(emb_dict):
    if "sents_id" in emb_dict:
        return len(emb_dict["sents_id"])

    for _, value in emb_dict.items():
        if isinstance(value, np.ndarray):
            return value.shape[0]

    raise ValueError("无法从 emb_dict 中推断样本数")


for enum_args in itertools.product(*enumArgs.values()):
    args = Arg(*enum_args)
    # print(enum_args)

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    print(f"使用特征文件: {args.input_emb_name}")

    with open(args.input_emb_name, "rb") as fin:
        emb_dict = pk.load(fin)

    if not isinstance(emb_dict, dict):
        raise ValueError("特征文件读取结果不是 dict，无法处理")

    # 先按当前文件自动调整输入维度
    adjust_input_dims_by_current_emb(args, emb_dict)

    num_samples = infer_num_samples(emb_dict)

    # 检查是否包含简化后句子嵌入向量
    if "summarized_emb" not in emb_dict:
        print("警告: 特征文件中不包含简化后句子嵌入向量，将创建零向量替代")
        emb_dict["summarized_emb"] = np.zeros((num_samples, args.input_dim4)).astype(np.float32)

    # 检查是否包含事件嵌入向量
    if "event_emb" not in emb_dict:
        print("警告: 特征文件中不包含事件嵌入向量，将创建零向量替代")
        emb_dict["event_emb"] = np.zeros((num_samples, args.input_dim5)).astype(np.float32)

    # 再补一次，确保缺失补零之后 input_dim 仍正确
    adjust_input_dims_by_current_emb(args, emb_dict)

    # 打印特征维度
    print("\n特征维度信息:")
    for key, value in emb_dict.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")

    X = None
    if args.model == "Baseline":
        X = latent_space_clustering_baseline.train(args, emb_dict)
    else:
        X = latent_space_clustering.train(args, emb_dict)

    evaluator = Evaluator(args.dataset, "clusters_/cluster.txt", args.n_clusters)
    # evaluator = Evaluator('MAVEN_ERE', 'clusters_/35.txt')
    metrics = Metrics(*evaluator.evaluate(["None", "max"]))
    all_metrics.append(metrics)
    # evaluator.cluster_visual("true")
    # evaluator.cluster_visual("pred")
    evaluator.export(evaluator.clusters_labels, "clusters_labels.json")
    # print("[Current Arg]: ", enum_args)
    # print("[Current Metrics]: ", metrics)

    for i, met in enumerate(best_metrics):
        best_metrics[i] = max(met, metrics, key=keys[i])
        if best_metrics[i].isEqual(metrics):
            best_args[i] = Arg(*enum_args)

for i in range(len(keys)):
    print(f"[Best Arg of {keysName[i]}]: {best_args[i]}")
    print(f"[Best Metrics of {keysName[i]}]: {best_metrics[i]}")

# 移除可视化代码以避免matplotlib和字体警告
# if len(enumArgs["n_clusters"]) > 1:
#     print("Plot the performance of different number of clusters")
#     ari_data = [m.ari for m in all_metrics]
#     nmi_data = [m.nmi for m in all_metrics]
#     acc_data = [m.acc for m in all_metrics]
#     f1_data = [m.bcubed_f1 for m in all_metrics]
#     print(f"ARI data: {ari_data}")
#     print(f"NMI data: {nmi_data}")
#     print(f"ACC data: {acc_data}")
#     print(f"F1 data: {f1_data}")