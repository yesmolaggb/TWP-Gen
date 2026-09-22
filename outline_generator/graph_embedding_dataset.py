# from util import util
import os
import pickle as pk
from pathlib import Path

# import pandas as pd
from collections import Counter, defaultdict

import networkx as nx
import scipy.sparse as sp
import spacy
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertModel, BertTokenizer

from encode_contextual_features import process_sentence
from util import util

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:  # central configuration: model paths, dictionary, dataset paths
    import twpgen_settings as _cfg
except Exception:  # pragma: no cover
    _cfg = None


class GAEDataLoader:
    def __init__(
        self, graph_path: str, corpus_info_path: str, feature_path: str = ""
    ) -> None:
        self.words_graph = {}
        self.corpus_info = []
        self.dataset = {}
        self.name2id = defaultdict(int)
        self.id2name = defaultdict(str)
        self.incrementId = 0
        self.incrementYId = 0
        self.base_nodes_id = []
        self.base_nodes_label = []
        self.label2id = defaultdict(int)
        self.Y = []

        # func
        self.loadWordGraph(graph_path)
        self.loadCorpusInfo(corpus_info_path)
        self.getName2id()
        self.G = self.trans2dataset()
        self.A = nx.adjacency_matrix(self.G)
        self.X = self.getFeature(strategy="bert", feature_path=feature_path)
        # get cluster truth label
        self.removeDerivedNode()
        self.labelTrans2Num()
        print("data loading finish...")

    def labelTrans2Num(self) -> None:
        for label in self.base_nodes_label:
            if label not in self.label2id:
                self.label2id[label] = self.incrementYId
                self.incrementYId += 1
        for label in self.base_nodes_label:
            self.Y.append(self.label2id[label])

    def findFreqMaxBaseNode(self, sent_id_list: list) -> list:
        postive_sent_ids = [sent_id for sent_id in sent_id_list if sent_id >= 0]
        if len(postive_sent_ids) == 0:
            return []
        count = Counter(postive_sent_ids)
        freq = count.most_common()
        most = [f[0] for f in freq if f[1] == freq[0][1]]
        return most

    def removeDerivedNode(self) -> None:
        # 只遍历图 G 中的节点，与 getName2id 保持完全一致
        for gid in self.G.nodes():
            word = self.id2name[gid]
            sent_id_list = self.words_graph["nodes"][word]
            maxfreq_nodes = self.findFreqMaxBaseNode(sent_id_list)
            if len(maxfreq_nodes) == 0:
                continue
            elif len(maxfreq_nodes) == 1:
                self.base_nodes_id.append(gid)
                event_type = self.corpus_info[maxfreq_nodes[0]]["type"]
                self.base_nodes_label.append(event_type)
                continue
            else:
                adj_nodes_id = [k for k in self.G[gid].keys()]  # find adjacent nodes
                adj_nodes_sentid_list = []
                for adj_node_id in adj_nodes_id:
                    node_name = self.id2name[adj_node_id]
                    adj_nodes_sentid_list.extend(self.words_graph["nodes"][node_name])
                maxfreq_sentid = Counter(adj_nodes_sentid_list).most_common(1)
                if not maxfreq_sentid:
                    continue
                self.base_nodes_id.append(gid)
                event_type = self.corpus_info[maxfreq_sentid[0][0]]["type"]
                self.base_nodes_label.append(event_type)

    def filteBaseNodeSentenceId(self, nodeid_list: list) -> list:
        sentid_list = []
        selected_nodeid_list = []
        for nodeid in nodeid_list:
            node_name = self.id2name[nodeid]
            sent_id_list = self.words_graph["nodes"][node_name]
            maxfreq_nodes = self.findFreqMaxBaseNode(sent_id_list)
            if len(maxfreq_nodes) != 0:
                sentid_list.append(maxfreq_nodes[0])
                selected_nodeid_list.append(nodeid)
        return [sentid_list, selected_nodeid_list]

    def getFeature(
        self,
        strategy: str = "bert",
        feature_path: str = "",
    ) -> torch.Tensor:
        if strategy == "eye":
            feature_dim = self.A.shape[0]
            feature = torch.eye(feature_dim, feature_dim)
            sparse_tensor = sp.csr_matrix(feature)
            lil_matrix = sparse_tensor.tolil()
            return lil_matrix
        elif strategy == "bert":
            # using exist feature (last extracted, avoid double counting)
            feature = None
            if feature_path != "" and os.path.exists(feature_path):
                print(
                    "detected exist {} file, loading exist feature...".format(
                        feature_path
                    )
                )
                feature = torch.load(feature_path)
            else:
                print(
                    "not detected {} file, generate new nodes feature by BERT".format(
                        feature_path
                    )
                )
                print("loading Transformer model")
                pretrained_weights = _cfg.language_model_paths.get(
                    "macbert", "/workspace/model/chinese-macbert-base"
                ) if _cfg else "/workspace/model/chinese-macbert-base"
                pretrained_weights_path = Path(pretrained_weights)
                model_class, tokenizer_class = BertForMaskedLM, BertTokenizer

                # 用 Path 对象绕过 huggingface_hub 的字符串路径格式校验
                self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights_path, local_files_only=True)
                mlm_model = model_class.from_pretrained(pretrained_weights_path, local_files_only=True)
                mlm_model.eval()
                mlm_model = mlm_model.to(torch.device(f"cuda:{0}"))
                # get model for embedding extraction
                self.model = mlm_model.bert
                # 加载spacy
                self.nlp = spacy.load("en_core_web_lg")

                feature_list = []
                for word, sent_id_list in tqdm(self.words_graph["nodes"].items()):
                    # 只处理图 G 中存在的节点
                    if word not in self.name2id:
                        continue
                    gid = self.name2id[word]
                    if gid not in self.G:
                        continue
                    maxfreq_nodes = self.findFreqMaxBaseNode(sent_id_list)
                    sentence = ""
                    # derive node
                    if len(maxfreq_nodes) == 0:
                        # generate virtual sentence
                        adj_nodes_id = [k for k in self.G[gid].keys()]
                        [
                            sentid_list,
                            selected_nodeid_list,
                        ] = self.filteBaseNodeSentenceId(adj_nodes_id)
                        if not sentid_list:
                            continue
                        sentence = self.corpus_info[sentid_list[0]]["sentence"]
                        derive_word = word.split("/")[-1]
                        origin_word = self.id2name[selected_nodeid_list[0]].split("/")[-1]

                        startindex = sentence.find(origin_word)
                        if startindex == -1:
                            sentence = derive_word + "," + sentence
                        else:
                            sentence = (
                                sentence[:startindex]
                                + derive_word
                                + sentence[startindex + len(origin_word) :]
                            )
                    elif len(maxfreq_nodes) >= 1:
                        sent_id = maxfreq_nodes[0]
                        sentence = self.corpus_info[sent_id]["sentence"]

                    # bert encoder
                    with torch.no_grad():
                        doc = self.nlp(sentence)
                        token_list = [token.text for token in doc]
                        token_embeds = process_sentence(
                            token_list, self.tokenizer, self.model, -1
                        )
                        word_index = 0
                        word = word.split("/")[-1]
                        for i, token in enumerate(token_list):
                            if word in token:
                                word_index = i
                                break
                        word_tensor = torch.from_numpy(token_embeds[word_index])
                        feature_list.append(word_tensor)

                feature = torch.stack(feature_list)
                # 替换 NaN/Inf 为 0，避免下游训练崩溃
                feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
                torch.save(feature, feature_path)

            # return feature
            sparse_tensor = sp.csr_matrix(feature)
            lil_matrix = sparse_tensor.tolil()
            return lil_matrix

    def trans2dataset(self) -> nx.Graph:
        link_data = []
        G = nx.Graph()
        for link in self.words_graph["links"]:
            source = self.name2id[link["source"]]
            target = self.name2id[link["target"]]
            link_data.append([source, target])
            G.add_edge(source, target)

        d = dict(nx.degree(G))
        print("Graph average degree: ", sum(d.values()) / len(G.nodes))
        return G

    def getName2id(self) -> None:
        # 只给出现在 links 里的节点分配 ID，与图 G 保持一致
        # 避免孤立节点（在 nodes 里但不在 links 里）导致 G[gid] KeyError
        nodes_in_links = set()
        for link in self.words_graph["links"]:
            nodes_in_links.add(link["source"])
            nodes_in_links.add(link["target"])
        for k in self.words_graph["nodes"].keys():
            if k in nodes_in_links:
                self.name2id[k] = self.incrementId
                self.id2name[self.incrementId] = k
                self.incrementId += 1

    def loadWordGraph(self, graph_path: str) -> None:
        with open(graph_path, "rb") as f:
            self.words_graph = pk.load(f)

    def loadCorpusInfo(self, corpus_info_path: str) -> None:
        with open(corpus_info_path, "rb") as f:
            self.corpus_info = pk.load(f)
