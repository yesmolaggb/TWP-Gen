# 1. 聚类性能评估-ARI（调兰德指数）https://zhuanlan.zhihu.com/p/145856959
# 2. 归一化互信息(NMI)评价指标 https://blog.csdn.net/hang916/article/details/88783931
# 3. 准确率（accuracy）
# 4. 聚类评价指标BCubed Metric https://zhuanlan.zhihu.com/p/57897957

import pickle as pk
from tqdm import tqdm
import json
import os
from sklearn import metrics
import numpy as np
import itertools
import twpgen_config as args
from collections import defaultdict


class Evaluator:
    def __init__(self, dataset, cluster_file, n_clusters) -> None:
        self.prefix = "./dataset/{}/".format(dataset)
        self.n_clusters = n_clusters
        # result_files = [
        #     'corpus_parsed_svo.pk', # Parse Corpus and Extract Subject-Verb-Object Triplets
        #     'corpus_parsed_svo_salient_verbs.pk', # Select Salient Verb Lemmas
        #     'corpus_parsed_svo_salient_obj_heads.pk', # Select Salient Object Heads
        #     'corpus_parsed_svo_obj2obj_head_info.pk', # Select Salient Object to Object Heads
        #     'corpus_parsed_svo_salient_po_mention_features.pk', # Generate Features for Each Salient <Predicate Lemma, Object Head> Mention
        #     'po_mention_disambiguated.pk', # Disambiguate Predicate Senses
        #     'po_tuple_features_all_svos.pk', # Generate Features for Each Salient <Predicate Sense, Object Head> Tuples
        # ]
        self.po_mention_file = "parsed_corpus_salient_po_mention_features.pk"
        self.po_sense_file = "po_mention_disambiguated.pk"
        self.corpus_parsed_file = "parsed_corpus.pk"
        self.cluster_file = cluster_file
        self.labeled_file = "corpus_info.pk"  # labeled data
        self.embeds_path = "./dataset/{}/clusters_/embed_45.pt".format(dataset)

        print("readFile...")
        with tqdm(total=4) as pbar:
            self.po_mention = self.readPickle(
                self.prefix + self.po_mention_file, "rb", pbar
            )
            self.po_sense = self.readPickle(
                self.prefix + self.po_sense_file, "rb", pbar
            )
            self.parsed_svo = self.readPickle(
                self.prefix + self.corpus_parsed_file, "rb", pbar
            )
            self.labeled = self.readPickle(
                self.prefix + self.labeled_file, "rb", pbar
            )

        self.clusters_labels = []  # 每个簇的标签
        self.clusters_size = []  # 每个簇的大小
        self.labels_true = []  # 真实标签
        self.labels_pred = []  # 聚类结果的标签
        self.labels_true_label = []  # 真实标签(标签名)
        self.labels_pred_label = []  # 预测标签(标签名)
        self.labels_index = {}  # 真实标签的索引
        self.index_labels = {}  # 索引对应的真实标签
        self.labels_count = []  # 每个簇中每个类别的数量
        self.labels_freq = defaultdict(int)  # 真实标签中每个类别的频率
        self.pred_labels_freq = defaultdict(int)  # 预测标签中每个类别的频率
        self.strategy = "None"
        self.selected_embed_index = []  # 选中的嵌入向量的索引

        self.getPOsTrueLabel(self.prefix + self.cluster_file)

    def readPickle(self, path: str, mode: str, pbar) -> None:
        with open(path, mode) as f:
            res = pk.load(f)
            pbar.update(1)
            return res

    def getLabelBySentId(self, sent_id: str) -> str:
        try:
            sent_id_int = int(sent_id)
            # 增加范围检查
            if not isinstance(self.labeled, list):
                print(f"Warning: self.labeled is not a list, but {type(self.labeled)}")
                return "UNKNOWN"
            
            if sent_id_int < 0 or sent_id_int >= len(self.labeled):
                print(f"Warning: Sentence ID '{sent_id}' is out of range (0-{len(self.labeled)-1})")
                return "UNKNOWN"
                
            if "type" not in self.labeled[sent_id_int]:
                print(f"Warning: No 'type' key in labeled data for sentence ID '{sent_id}'")
                return "UNKNOWN"
                
            return self.labeled[sent_id_int]["type"]
        except (ValueError, IndexError, KeyError, TypeError) as e:
            print(f"Warning: Error accessing label for sentence ID '{sent_id}': {e}")
            return "UNKNOWN"

    def findDiffType(self, verb: str, obj: str, verb_sense: str, label: str) -> bool:
        same_po_count = 0
        hasDiffType = False
        for k, v in self.po_mention.items():
            if (
                v["verb"] == verb
                and v["obj_head"] == obj
                and str(self.po_sense[k]) == verb_sense
            ):
                same_po_count += 1
                try:
                    sent_id = int(k.split("_")[0])
                    if sent_id < 0 or sent_id >= len(self.labeled):
                        continue
                    sent_type = self.labeled[sent_id]["type"]
                    # print('sent_type: ', sent_type)
                    if sent_type != label:
                        hasDiffType = True
                except (ValueError, IndexError, KeyError, TypeError) as e:
                    continue
        # print("count: ", same_po_count)
        if same_po_count > 1 and hasDiffType:
            return True
        else:
            return False

    # get po's true label
    def getPOsTrueLabel(self, cluster_filepath: str) -> None:
        print("getPOsTrueLabel...")
        try:
            with open(cluster_filepath, "r") as f:
                for line in tqdm(f.readlines()):
                    # line: Topic 0 (10): produce_1 tide 1, produce_1 impact 2
                    line = line.strip("\n")
                    if not ":" in line:
                        print(f"Warning: Invalid line format: {line}")
                        continue
                        
                    parts = line.split(":")
                    if len(parts) < 2:
                        print(f"Warning: Invalid line format after split: {line}")
                        continue
                        
                    pairs_text = parts[1].strip()
                    if not pairs_text:
                        print(f"Warning: No pairs found in line: {line}")
                        continue
                        
                    pairs = pairs_text.split(", ")
                    if not pairs:
                        print(f"Warning: No pairs after split: {line}")
                        continue
                        
                    if pairs[0].startswith(" "):
                        pairs[0] = pairs[0][1:]
                    
                    labels = []
                    for pair in pairs:
                        pair_parts = pair.split("/")
                        if len(pair_parts) < 4:  # 确保至少有index, verb_sense, obj, sent_id
                            print(f"Warning: Invalid pair format: {pair}")
                            continue
                            
                        try:
                            selected_embed_index = int(pair_parts[0])
                        except ValueError:
                            print(f"Warning: Skipping invalid index value: {pair_parts[0]}")
                            continue
                            
                        verb_with_sense = pair_parts[1].split("_")
                        if len(verb_with_sense) < 2:
                            print(f"Warning: Invalid verb_with_sense format: {pair_parts[1]}")
                            continue
                            
                        verb = verb_with_sense[0]
                        sense = verb_with_sense[1]
                        obj = pair_parts[2]
                        sent_id = pair_parts[-1]
                        
                        # 搜索sent_id对应句子的真实标签
                        label = self.getLabelBySentId(sent_id)
                        if label != "UNKNOWN":
                            labels.append(label)
                            self.selected_embed_index.append(selected_embed_index)
                            
                    if len(labels) != 0:
                        self.clusters_labels.append(labels)
                        self.clusters_size.append(len(labels))
                    # print("\033[1;33m ---------------------------------\033[0m")
        except FileNotFoundError:
            print(f"Warning: Cluster file not found: {cluster_filepath}")
        except Exception as e:
            print(f"Error processing cluster file: {e}")

    # get labels_true
    def getLabelsTrue(self) -> None:
        index_true = 0
        for cluster in self.clusters_labels:
            for label in cluster:
                if label not in self.labels_index:
                    self.labels_index[label] = index_true
                    self.index_labels[index_true] = label
                    index_true += 1
        for cluster in self.clusters_labels:
            for label in cluster:
                self.labels_true.append(self.labels_index[label])
                self.labels_true_label.append(label)
                self.labels_freq[label] = self.labels_freq[label] + 1
        # sort labels_freq
        self.labels_freq = dict(
            sorted(self.labels_freq.items(), key=lambda d: d[1], reverse=True)
        )

    # get optimal labels_pred
    def getLabelsPred(self, strategy: str) -> None:
        self.labels_count = []
        for cluster in self.clusters_labels:
            cluster_count = {}
            for item in cluster:
                if item in cluster_count:
                    cluster_count[item] += 1
                else:
                    cluster_count[item] = 1
            self.labels_count.append(cluster_count)
        # max
        if strategy == "max":
            for i in range(len(self.labels_count)):
                max_label = ""
                max_count = 0
                for label, count in self.labels_count[i].items():
                    if count > max_count:
                        max_count = count
                        max_label = label
                self.labels_pred.extend(
                    [self.labels_index[max_label]] * self.clusters_size[i]
                )
        elif strategy == "None":
            index = 0
            for cluster in self.clusters_labels:
                self.labels_pred.extend([index] * len(cluster))
                self.labels_pred_label.extend([index] * len(cluster))
                self.pred_labels_freq[index] = len(cluster)
                index += 1
            self.pred_labels_freq = dict(
                sorted(self.labels_freq.items(), key=lambda d: d[1], reverse=True)
            )
        elif strategy == "permutation":
            # 0 0 0 1 1 2
            # 0 0 0 2 2 1
            # 1 1 1 0 0 2
            # 1 1 1 2 2 0
            # 2 2 2 0 0 1
            # 2 2 2 1 1 0
            labels_pred_pmt = []
            indexs = [i for i in range(len(self.clusters_labels))]
            lens = [len(cluster) for cluster in self.clusters_labels]
            for p in itertools.permutations(indexs):
                for i in range(len(p)):
                    labels_pred_pmt.extend([p[i]] * lens[i])
                self.labels_pred.append(labels_pred_pmt)

    def generateLabel(self, strategy: str) -> None:
        self.labels_pred = []
        self.labels_true = []
        self.labels_true_label = []
        self.getLabelsTrue()
        self.getLabelsPred(strategy)

    def evaluate(self, strategy: list) -> list:
        ari, nmi, acc, bcubed_f1 = self.evaluate_nosupervision(strategy[0])
        # acc = self.evaluate_supervision(strategy[1])
        return ari, nmi, acc, bcubed_f1

    def best_acc(self, labels_true, labels_pred):
        best_acc = 0
        best_permute = None
        print("get best_acc...")
        
        # 使用集合存储已生成的排列，以实现去重
        permutes_set = set()
        permutes = []
        
        # 计算总共可能的排列数
        import math
        total_possible_permutations = math.factorial(self.n_clusters)
        max_permutations = min(args.acc_permutation_times, total_possible_permutations)
        
        print(f"尝试生成 {max_permutations} 个不重复的排列(总可能排列数: {total_possible_permutations})")
        
        # 尝试生成不重复的排列
        attempts = 0
        max_attempts = args.acc_permutation_times * 2  # 设置最大尝试次数，避免无限循环
        
        while len(permutes) < max_permutations and attempts < max_attempts:
            attempts += 1
            # 生成一个随机排列
            new_permute = tuple(np.random.permutation(self.n_clusters).tolist())
            
            # 检查是否已存在
            if new_permute not in permutes_set:
                permutes_set.add(new_permute)
                permutes.append(np.array(new_permute))
                
        print(f"生成了 {len(permutes)} 个不重复的排列，共尝试 {attempts} 次")
        
        # 评估每个排列的准确率
        for permute in tqdm(permutes):
            labels_pred_permute = [permute[label] for label in labels_pred]
            acc = metrics.accuracy_score(labels_true, labels_pred_permute)
            if acc > best_acc:
                best_acc = acc
                best_permute = permute
        return best_acc

    def evaluate_nosupervision(self, strategy) -> list:
        self.generateLabel(strategy)
        
        # 如果没有足够的标签用于评估，则给出警告并返回默认值
        if len(self.labels_true) < 2 or len(self.labels_pred) < 2:
            print("警告: 没有足够的有效标签用于评估")
            return 0, 0, 0, 0
            
        try:
            ari = metrics.adjusted_rand_score(self.labels_true, self.labels_pred)
            nmi = metrics.normalized_mutual_info_score(
                self.labels_true, self.labels_pred, average_method="min"
            )
            precision = metrics.homogeneity_score(self.labels_true, self.labels_pred)
            recall = metrics.completeness_score(self.labels_true, self.labels_pred)
            bcubed_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            acc = self.best_acc(self.labels_true, self.labels_pred)

            ari = (ari + 1) / 2 * 100
            nmi = nmi * 100
            precision = precision * 100
            recall = recall * 100
            acc = acc * 100
            bcubed_f1 = bcubed_f1 * 100
            print("ari(%): {:.2f}%".format(ari))
            print("nmi(%): {:.2f}%".format(nmi))
            print("precision(%): {:.2f}%".format(precision))
            print("recall(%): {:.2f}%".format(recall))
            print("acc(%): {:.2f}%".format(acc))
            print("bcubed_f1(%): {:.2f}%".format(bcubed_f1))
            return ari, nmi, acc, bcubed_f1
        except Exception as e:
            print(f"评估过程中出错: {e}")
            return 0, 0, 0, 0

    def evaluate_supervision(self, strategy) -> list:
        self.generateLabel(strategy)
        
        if len(self.labels_true) < 2 or len(self.labels_pred) < 2:
            print("警告: 没有足够的有效标签用于评估")
            return 0
            
        try:
            acc = metrics.accuracy_score(self.labels_true, self.labels_pred)
            acc = acc * 100
            print("acc(%): {:.2f}%".format(acc))
            return acc
        except Exception as e:
            print(f"评估过程中出错: {e}")
            return 0

    def evaluate_supervision_perm(self, strategy) -> None:
        self.generateLabel(strategy)
        
        if len(self.labels_true) < 2 or not self.labels_pred:
            print("警告: 没有足够的有效标签用于评估")
            return
            
        try:
            max_acc = 0
            labels_pred_perm = None
            
            if isinstance(self.labels_pred[0], list):
                for pred in self.labels_pred:
                    acc = metrics.accuracy_score(self.labels_true, pred)
                    if acc > max_acc:
                        max_acc = acc
                        labels_pred_perm = pred
            else:
                labels_pred_perm = self.labels_pred
                max_acc = metrics.accuracy_score(self.labels_true, labels_pred_perm)
                
            precision_score = metrics.precision_score(
                self.labels_true, labels_pred_perm, average="weighted"
            )
            f1_score = metrics.f1_score(
                self.labels_true, labels_pred_perm, average="weighted"
            )
            recall_score = metrics.recall_score(
                self.labels_true, labels_pred_perm, average="weighted"
            )

            print("acc(%): {:.2f}%".format(max_acc * 100))
            print("precision_score(%): {:.2f}%".format(precision_score * 100))
            print("f1_score(%): {:.2f}%".format(f1_score * 100))
            print("recall_score(%): {:.2f}%".format(recall_score * 100))
        except Exception as e:
            print(f"评估排列过程中出错: {e}")

    def export(self, data: any, filename: str) -> None:
        if os.path.exists(self.prefix + filename):
            os.remove(self.prefix + filename)
        with open(self.prefix + filename, "w") as f:
            f.write(json.dumps(data))

    def cluster_visual(self, trueOfPred: str) -> None:
        """可视化函数已禁用以避免字体和matplotlib警告"""
        print(f"跳过 {trueOfPred} 聚类可视化（已禁用以避免警告）")
        return
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import torch
        from sklearn.manifold import TSNE
        
        # 添加中文字体支持
        try:
            from matplotlib import font_manager
            
            # 从self.prefix中提取dataset名称
            dataset = self.prefix.strip('./dataset/').strip('/')
            
            # 检查系统是否有中文字体
            chinese_fonts = []
            for font in font_manager.fontManager.ttflist:
                if 'SimHei' in font.name or 'SimSun' in font.name or 'WenQuanYi' in font.name or 'DroidSansFallback' in font.name or 'Noto Sans CJK' in font.name:
                    chinese_fonts.append(font.name)
            
            # 如果找到了中文字体，使用第一个找到的字体
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
                print(f"使用中文字体: {chinese_fonts[0]}")
            else:
                # 尝试查找系统中可能存在的其他中文字体
                import subprocess
                try:
                    # 直接尝试设置通用字体
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
                    # 添加默认替代字体
                    plt.rcParams['axes.unicode_minus'] = False
                    print("使用DejaVu Sans字体，中文可能不能正确显示")
                except Exception as e:
                    print(f"设置字体失败: {e}")
                    print("将使用matplotlib内置字体，中文可能显示为方块")
            
            # 设置matplotlib正常显示中文
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        except Exception as font_err:
            print(f"设置中文字体时出错: {font_err}")
        
        if not os.path.exists(self.embeds_path):
            print(f"警告: 嵌入向量文件不存在: {self.embeds_path}")
            return
            
        if not self.selected_embed_index:
            print("警告: 没有选择的嵌入向量索引")
            return

        try:
            cluster_embeds = torch.load(self.embeds_path)
            embed = cluster_embeds["embed"]
            labels = (
                self.labels_true_label if trueOfPred == "true" else self.labels_pred_label
            )
            
            if not labels:
                print(f"警告: 没有有效的标签用于可视化")
                return
                
            z = []
            for index in self.selected_embed_index:
                if index < len(embed):
                    z.append(embed[index])
                else:
                    print(f"警告: 嵌入向量索引越界: {index}")
                    
            if not z:
                print("警告: 没有有效的嵌入向量用于可视化")
                return

            filter_labels = []
            filter_z = []

            label_map = {
                "Personnel": 0,
                "Conflict": 1,
                "Contact": 2,
                "Life": 3,
                "Transaction": 4,
                "Movement": 5,
                "Justice": 6,
                "Business": 7,
                # "Transaction": 4,
                # "Movement": 5,
                # "Conflict:Attack": 0,
                # "Life:Die": 1,
                # "Contact:Phone-Write": 2,
                # "Contact:Meet": 3,
                # "Transaction:Transfer-Money": 4,
                # "Transaction:Transfer-Ownership": 5,
                # "Movement:Transport": 6,
                # "Personnel:End-Position": 7,
                # "Personnel:Elect": 8,
                # "Justice:Charge-Indict": 9,
                # "Justice:Trial-Hearing": 10,
                # "Business:Declare-Bankruptcy": 11,
                # "Business:Start-Org": 12,
            }
            # label_map = {}
            # for i, label in enumerate(self.labels_freq.keys()):
            #     if i > 10:
            #         break
            #     label_map[label] = i

            # for label, index in label_map.items():
            #     for i, lb in enumerate(labels):
            #         if lb == label or label in lb:
            #             filter_labels.append(label)
            #             filter_z.append(z[i])

            # tSNE dimension reduction
            # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=15)
            tsne_z = tsne.fit_transform(torch.tensor(z))
            # set font size of labels on matplotlib plots
            plt.rc("font", size=16)
            # plot the result
            data = pd.DataFrame({"x": tsne_z[:, 0], "y": tsne_z[:, 1], "label": labels})
            sns.set_theme(style="white")
            # 保存图片而不是直接显示
            plt.figure(figsize=(12, 10))
            # plot data with seaborn
            g = sns.scatterplot(data=data, x="x", y="y", hue="label", palette="Set2")
            g.set_title("Cluster visualization")
            # 隐藏坐标轴
            plt.xticks([])
            plt.yticks([])
            g.set(xlabel=None, ylabel=None)
            plt.tight_layout()
            # 保存图片
            save_dir = os.path.join("figs", dataset)
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, f"cluster_{trueOfPred}.png")
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            print(f"图片已保存至: {fig_path}")
            plt.close()
        except Exception as e:
            print(f"可视化过程中出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    evaluator = Evaluator(args.dataset, "clusters_/cluster.txt", 30)
    evaluator.evaluate(["None", "max"])
    evaluator.cluster_visual("true")
    evaluator.cluster_visual("pred")
    evaluator.export(evaluator.clusters_labels, "clusters_labels.json")
