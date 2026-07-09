import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _config_path() -> Path:
    raw = os.environ.get("TWPGEN_CONFIG")
    if not raw:
        return PROJECT_ROOT / "twpgen_config.example.json"
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_config() -> dict:
    path = _config_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_CONFIG = _load_config()


def _int_list(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _path(raw: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _cfg(section: str, key: str, default=None):
    value = _CONFIG.get(section, {}).get(key, default)
    return value


dataset = os.environ.get("TWPGEN_DATASET") or _CONFIG.get("dataset_name", "topic")
dataset_root = os.environ.get("TWPGEN_DATASET_ROOT") or _CONFIG.get("dataset_root", "./dataset")
resource_dir = os.environ.get("TWPGEN_RESOURCE_DIR") or _CONFIG.get("resource_dir", "./resources")
output_dir = os.environ.get("TWPGEN_OUTPUT_DIR") or _CONFIG.get("output_dir", "./output")
dataset_root = _path(dataset_root)
resource_dir = _path(resource_dir)
output_dir = _path(output_dir)

feature_cfg = _CONFIG.get("feature_fusion", {})
graph_cfg = _CONFIG.get("graph_model", {})

label_whitelist = [
    "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG",
    "PERSON", "PRODUCT", "WORK_OF_ART", "DATE", "TIME", "MONEY",
    "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL",
]

corpus = f"{dataset_root}/{dataset}/corpus.txt"
parsed_corpus = f"{dataset_root}/{dataset}/parsed_corpus.pk"
spacy_model = os.environ.get("TWPGEN_SPACY_MODEL", _CONFIG.get("spacy_model", "zh_core_web_lg"))
verb_freq_file = f"{resource_dir}/verb_freq.json"
all_lemma_freq_file = f"{resource_dir}/all_lemma_freq.json"
min_verb_freq = int(feature_cfg.get("min_verb_freq", 3))
top_verb_ratio = float(feature_cfg.get("top_verb_ratio", 0.8))
min_obj_freq = int(feature_cfg.get("min_obj_freq", 3))
top_obj_ratio = float(feature_cfg.get("top_obj_ratio", 0.8))
lm_type = os.environ.get("TWPGEN_LM_TYPE", _CONFIG.get("language_model", "chinese-bert-wwm"))
top_k_expand_result = int(feature_cfg.get("top_k_expand_result", 50))
gpu_id = _int_list(os.environ.get("TWPGEN_GPU_IDS"), _CONFIG.get("gpu_ids", [0, 1]))

word_graph = f"{dataset_root}/{dataset}/word_graph.pk"
weight_threshold = 0
weight_v2n = 9
mention_file = f"{dataset_root}/{dataset}/parsed_corpus_salient_po_mention_features.pk"
save_disambiguated_path = f"{dataset_root}/{dataset}/po_mention_disambiguated.pk"
dict_file = f"{resource_dir}/verb_sense_dict_w_features.json"
save_po_tuple_feature_path = f"{dataset_root}/{dataset}/po_tuple_features_all_svos.pk"
use_all_svos = True

graph_model = graph_cfg.get("name", "GAE")
input_dim = int(graph_cfg.get("input_dim", 768))
hidden1_dim = int(graph_cfg.get("hidden1_dim", 512))
hidden2_dim = int(graph_cfg.get("hidden2_dim", 256))
num_epoch = int(graph_cfg.get("num_epoch", 100))
learning_rate = float(graph_cfg.get("learning_rate", 0.001))

graph_path = f"{dataset_root}/{dataset}/ve_graph.pk"
corpus_info_path = f"{dataset_root}/{dataset}/corpus_info.pk"
feature_path = f"{dataset_root}/{dataset}/nodes_feature.pt"
gae_feature_path = f"{dataset_root}/{dataset}/gae_feature.csv"
acc_permutation_times = 10
