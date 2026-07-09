import os
import pickle as pk
from collections import defaultdict

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

import twpgen_config as args
from util import util

from functools import lru_cache
import transaction


class ConceptNetRequest:
    def __init__(
        self, word_list: list, weight_threshold: int, weight_v2n: int, save_path: str
    ) -> None:
        """_summary_

        Args:
            word_dict (dict): k is word, v is id
        """
        self.word_list = word_list
        self.weight_threshold = weight_threshold
        self.weight_v2n = weight_v2n
        self.save_path = save_path
        self.frag = self.save_path.split("/")
        self.save_path_dir = "/".join(self.frag[:-1])
        self.savepoint = -1  # value is word index
        self.savepoint_path = "{}/savepoint.point".format(self.save_path_dir)
        self.nodes = defaultdict()
        self.links = []

        # 确保保存目录存在
        os.makedirs(self.save_path_dir, exist_ok=True)
        
        # func
        self.readSavePoint()

    def request(self) -> dict:
        """request ConceptNet WebAPI

        Returns:
            dict: _description_
        """

        @lru_cache(maxsize=None)
        @retry(wait=wait_random_exponential(min=5, max=30), stop=stop_after_attempt)
        def try_task(req_url):
            return requests.get(req_url, proxies={}, timeout=30).json()

        cur_verb_id = ""  # record current verb id
        with tqdm(total=len(self.word_list) - self.savepoint - 1) as t:
            for i in range(self.savepoint + 1, len(self.word_list)):
                word = self.word_list[i]["word"]
                sent_id = self.word_list[i]["sent_id"]
                word_type = self.word_list[i]["word_type"]
                root_id = "/c/zh/{}".format(word)
                req_url = "http://10.146.130.132:22482{}".format(root_id)
                if word_type == "v":
                    cur_verb_id = root_id

                obj = try_task(req_url)

                if obj is None:
                    print("obj is None, request: ", req_url)
                    continue
                edges = obj["edges"]
                if edges is None:
                    print("edges is None, request: ", req_url)
                if len(edges) == 0:
                    print("edges is empty, request: ", req_url)

                self.buildGraph(root_id, edges, sent_id, cur_verb_id, word_type)

                # after every request, save data
                try:
                    self.saveData()
                    self.writeSavePoint(i)
                    transaction.commit()
                except Exception as e:
                    print(e)
                    transaction.abort()

                t.set_postfix(links=len(self.links), nodes=len(self.nodes))
                t.update(1)
            # all request is done, save data
            self.saveData()
            self.writeSavePoint(len(self.word_list) - 1)

    def buildGraph(
        self, root_id: str, edges: list, sent_id: list, cur_verb_id: str, word_type: str
    ) -> None:
        """extend nodes and links

        Args:
            root_id (str): conceptnet query word
            edges (list): conceptnet edges info
            sent_id (list): current word's sentence id
            cur_verb_id (str): current verb id, use to create link
            word_type (str): current word type, v(verb) or n(noun/entity)
        """
        # insert root node
        if root_id not in self.nodes:
            self.nodes[root_id] = [sent_id]
        else:
            self.nodes[root_id].append(sent_id)
        # insert link of v to o
        if word_type == "o" or word_type in args.label_whitelist:
            self.links.append(
                {
                    "source": cur_verb_id,
                    "target": root_id,
                    "relation": word_type,
                    "surfaceText": "",
                    "weight": self.weight_v2n,  # it can be modified later (before train stage)
                }
            )

        # insert child node and link to root
        for edge in edges:
            target_dir = ""
            if edge["start"]["@id"] != root_id:
                target_dir = "start"
            else:
                target_dir = "end"
            target = edge[target_dir]
            if "language" not in target or target["language"] != "zh":
                continue
            if "@id" not in target:
                continue
            weight = edge["weight"]
            # ignore low weight
            if weight < self.weight_threshold:
                continue
            if target["@id"] not in self.nodes:
                self.nodes[target["@id"]] = [-sent_id]
            else:
                self.nodes[target["@id"]].append(-sent_id)
            relation = ""
            surfaceText = ""
            if "rel" in edge and "@id" in edge["rel"]:
                relation = edge["rel"]["@id"]
            if "surfaceText" in edge:
                surfaceText = edge["surfaceText"]
            self.links.append(
                {
                    "source": root_id,
                    "target": target["@id"],
                    "relation": relation,
                    "surfaceText": surfaceText,
                    "weight": edge["weight"],
                }
            )

    def writeSavePoint(self, point: int) -> None:
        if os.path.exists(self.savepoint_path):
            os.remove(self.savepoint_path)
        with open(self.savepoint_path, "w") as f:
            f.write(str(point))

    def readSavePoint(self) -> int:
        if not os.path.exists(self.savepoint_path):
            self.savepoint = -1
            return

        # load savepoint
        with open(self.savepoint_path, "r") as f:
            self.savepoint = int(f.read())
        
        # 只有当save_path文件存在时才加载数据
        if os.path.exists(self.save_path):
            # load pre save data
            with open(self.save_path, "rb") as f:
                data = pk.load(f)
                self.nodes = data["nodes"]
                self.links = data["links"]
        else:
            # 如果文件不存在，重置savepoint
            self.savepoint = -1
            if os.path.exists(self.savepoint_path):
                os.remove(self.savepoint_path)

    def saveData(self) -> None:
        data = {"nodes": self.nodes, "links": self.links}
        util.savePk(path=self.save_path, data=data)

    # if you want restart building, you have to call this
    def clearData(self) -> None:
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        if os.path.exists(self.savepoint_path):
            os.remove(self.savepoint_path)
