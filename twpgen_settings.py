"""Single source of truth for every path, key and tunable in TWP-Gen.

Edit values in ``<repo>/twpgen_config.json`` (copy it from
``twpgen_config.example.json``) or override them through ``<repo>/.env``.
No other module in this repository should hard-code a path, an API key or a
topic list.

Resolution order for every setting
----------------------------------
1. environment variable (``TWPGEN_*`` / ``ARTICLE_LLM_*`` / ``OPENAI_*``)
2. ``twpgen_config.json`` (override the file location with ``TWPGEN_CONFIG``)
3. built-in defaults defined in this file

Relative paths in the config file are resolved against the repository root.
Every stage reads this module:

* ``outline_generator/*`` via ``outline_generator/twpgen_config.py`` (thin re-export)
* ``article_generator/src/post_outline/*`` via ``post_outline/llm_settings.py``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILENAMES = ("twpgen_config.json", "twpgen_config.example.json")
LEGACY_CONFIG = Path("outline_generator") / "twpgen_config.example.json"
ENV_FILE = PROJECT_ROOT / ".env"


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def load_env_file(path: Path = ENV_FILE) -> None:
    """Load ``KEY=VALUE`` pairs from ``.env`` (already-set variables win)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def config_path() -> Path:
    raw = os.environ.get("TWPGEN_CONFIG")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    for name in CONFIG_FILENAMES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            return candidate
    return PROJECT_ROOT / LEGACY_CONFIG


def load_config() -> dict[str, Any]:
    path = config_path()
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


load_env_file()
_CONFIG: dict[str, Any] = load_config()


def _section(name: str) -> dict[str, Any]:
    value = _CONFIG.get(name, {})
    return value if isinstance(value, dict) else {}


def _cfg(path: str, default: Any = None) -> Any:
    """Read ``"section.key"`` (or ``"key"``) from the JSON config."""
    node: Any = _CONFIG
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node in ("", None) else node


def _get(env_names: Iterable[str], cfg_path: str, default: Any) -> Any:
    for name in env_names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return _cfg(cfg_path, default)


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a configured path (relative paths are repository-root relative)."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def path_setting(env_names: Iterable[str], cfg_path: str, default: str) -> str:
    return str(resolve_path(_get(env_names, cfg_path, default)))


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_list(value: Any, default: list[int]) -> list[int]:
    if value in (None, ""):
        return default
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


# --------------------------------------------------------------------------- #
# project / dataset / resources / output
# --------------------------------------------------------------------------- #
project_name = _cfg("project_name", "TWP-Gen")
paper_title = _cfg("paper_title", "")

dataset = _get(("TWPGEN_DATASET",), "dataset_name", "topic")
dataset_root = path_setting(("TWPGEN_DATASET_ROOT",), "dataset_root", "./dataset")
duee_dir = path_setting(("TWPGEN_DUEE_DIR",), "datasets.duee_dir", "./dataset/DuEE")
resource_dir = path_setting(("TWPGEN_RESOURCE_DIR",), "resource_dir", "./resources")
output_dir = path_setting(("TWPGEN_OUTPUT_DIR",), "output_dir", "./output")

article_dir = path_setting(("TWPGEN_ARTICLE_DIR",), "article.article_dir", "./output/article")
references_dir = path_setting(
    ("TWPGEN_REFERENCES_DIR",), "article.references_dir", "./output/references"
)

topic_file = path_setting(
    ("TWPGEN_TOPIC_FILE",), "topics.topic_file", "./knowledge_collector/topic.txt"
)
outline_dir = path_setting(("TWPGEN_OUTLINE_DIR",), "paths.outline_dir", "./dataset")
article_source_dir = path_setting(
    ("TWPGEN_ARTICLE_SOURCE_DIR",), "paths.article_source_dir", "./knowledge_collector/result"
)

# Evaluation dataset: the 60 generation tasks and the evaluation protocol.
evaluation_dataset_dir = path_setting(
    ("TWPGEN_EVAL_DATASET_DIR",), "evaluation_dataset.dataset_dir", "./数据集"
)
topic_dataset_file = path_setting(
    ("TWPGEN_TOPIC_DATASET",),
    "evaluation_dataset.topics_json",
    "./数据集/whitepaper_topics.json",
)
topic_dataset_txt = path_setting(
    ("TWPGEN_TOPIC_DATASET_TXT",),
    "evaluation_dataset.topics_txt",
    "./数据集/whitepaper_topics.txt",
)
evaluation_protocol_file = path_setting(
    ("TWPGEN_EVAL_PROTOCOL",),
    "evaluation_dataset.protocol_json",
    "./数据集/evaluation_method.json",
)
evaluation_protocol_doc = path_setting(
    ("TWPGEN_EVAL_PROTOCOL_MD",),
    "evaluation_dataset.protocol_md",
    "./数据集/evaluation_method.md",
)


# --------------------------------------------------------------------------- #
# dictionary and derived resource files
# --------------------------------------------------------------------------- #
dict_file = path_setting(
    ("TWPGEN_VERB_SENSE_DICT",),
    "resources.verb_sense_dict",
    "./resources/verb_sense_dict_w_features.json",
)
verb_freq_file = f"{resource_dir}/verb_freq.json"
all_lemma_freq_file = f"{resource_dir}/all_lemma_freq.json"


# --------------------------------------------------------------------------- #
# NLP / fusion / graph settings
# --------------------------------------------------------------------------- #
spacy_model = _get(("TWPGEN_SPACY_MODEL",), "spacy_model", "zh_core_web_lg")
lm_type = _get(("TWPGEN_LM_TYPE",), "language_model", "chinese-bert-wwm")
label_whitelist = list(
    _cfg(
        "label_whitelist",
        [
            "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG",
            "PERSON", "PRODUCT", "WORK_OF_ART", "DATE", "TIME", "MONEY",
            "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL",
        ],
    )
)

_feature = _section("feature_fusion")
min_verb_freq = _int(_feature.get("min_verb_freq"), 3)
top_verb_ratio = _float(_feature.get("top_verb_ratio"), 0.8)
min_obj_freq = _int(_feature.get("min_obj_freq"), 3)
top_obj_ratio = _float(_feature.get("top_obj_ratio"), 0.8)
top_k_expand_result = _int(_feature.get("top_k_expand_result"), 50)

_graph = _section("graph_model")
graph_model = _graph.get("name", "GAE")
input_dim = _int(_graph.get("input_dim"), 768)
hidden1_dim = _int(_graph.get("hidden1_dim"), 512)
hidden2_dim = _int(_graph.get("hidden2_dim"), 256)
num_epoch = _int(_graph.get("num_epoch"), 100)
learning_rate = _float(_graph.get("learning_rate"), 0.001)

gpu_id = _int_list(os.environ.get("TWPGEN_GPU_IDS"), list(_cfg("gpu_ids", [0, 1])))

# conda / virtualenv names used by scripts/run_twpgen_pipeline.sh
docgen_env = str(_get(("TWPGEN_DOCGEN_ENV",), "runtime.docgen_env", "/workspace/conda/docgen"))
pipeline_env = str(_get(("TWPGEN_PIPELINE_ENV",), "runtime.pipeline_env", "/workspace/conda/GESI"))
article_env = str(
    _get(("TWPGEN_ARTICLE_ENV",), "runtime.article_env", pipeline_env)
)


# --------------------------------------------------------------------------- #
# pretrained model locations
# --------------------------------------------------------------------------- #
model_root = str(_get(("TWPGEN_MODEL_ROOT",), "models.model_root", "/workspace/model"))
_model_names = _section("models").get(
    "language_models",
    {
        "blu": "bert-large-uncased-whole-word-masking",
        "macbert": "chinese-macbert-base",
        "chinese-bert-wwm": "chinese-bert-wwm",
    },
)
language_model_paths: dict[str, str] = {}
for _key, _value in _model_names.items():
    _raw = str(_value)
    language_model_paths[_key] = (
        str(resolve_path(_raw)) if _raw.startswith(".") else (
            _raw if _raw.startswith("/") else f"{model_root}/{_raw}"
        )
    )
pretrained_weights = language_model_paths.get(lm_type, f"{model_root}/{lm_type}")


# --------------------------------------------------------------------------- #
# derived dataset file paths
# --------------------------------------------------------------------------- #
corpus = f"{dataset_root}/{dataset}/corpus.txt"
duee_parsed_corpus = f"{duee_dir}/parsed_corpus.pk"
duee_projection_json = f"{duee_dir}/project_test_events.json"
duee_extraction_log = f"{duee_dir}/extraction_log.json"
parsed_corpus = f"{dataset_root}/{dataset}/parsed_corpus.pk"
word_graph = f"{dataset_root}/{dataset}/word_graph.pk"
mention_file = f"{dataset_root}/{dataset}/parsed_corpus_salient_po_mention_features.pk"
save_disambiguated_path = f"{dataset_root}/{dataset}/po_mention_disambiguated.pk"
save_po_tuple_feature_path = f"{dataset_root}/{dataset}/po_tuple_features_all_svos.pk"
graph_path = f"{dataset_root}/{dataset}/ve_graph.pk"
corpus_info_path = f"{dataset_root}/{dataset}/corpus_info.pk"
feature_path = f"{dataset_root}/{dataset}/nodes_feature.pt"
gae_feature_path = f"{dataset_root}/{dataset}/gae_feature.csv"

weight_threshold = 0
weight_v2n = 9
use_all_svos = True
acc_permutation_times = 10


# --------------------------------------------------------------------------- #
# LLM settings — used by every stage that calls a model
# --------------------------------------------------------------------------- #
class LLMSettings:
    """Resolved model connection settings."""

    __slots__ = ("api_key", "base_url", "model", "timeout", "enable_thinking")

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 360.0,
        enable_thinking: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Set OPENAI_API_KEY (or ARTICLE_LLM_API_KEY) "
                f"in {ENV_FILE}, or put it in twpgen_config.json."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "enable_thinking": self.enable_thinking,
        }


llm = LLMSettings(
    api_key=str(_get(("ARTICLE_LLM_API_KEY", "OPENAI_API_KEY"), "llm.api_key", "")),
    base_url=str(
        _get(
            ("ARTICLE_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
            "llm.base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    ),
    model=str(_get(("ARTICLE_LLM_MODEL", "TWPGEN_LLM_MODEL"), "llm.model", "qwen3-32b")),
    timeout=_float(_get(("TWPGEN_LLM_TIMEOUT",), "llm.timeout", 360.0), 360.0),
    enable_thinking=_bool(_get(("ARTICLE_ENABLE_THINKING",), "llm.enable_thinking", False)),
)


# --------------------------------------------------------------------------- #
# topics
# --------------------------------------------------------------------------- #
default_topics: list[str] = list(_cfg("topics.default", []) or [])
source_dir_aliases: dict[str, str] = dict(_cfg("topics.source_dir_aliases", {}) or {})


def load_topics(path: str | os.PathLike[str] | None = None) -> list[str]:
    """Read one topic per line; falls back to ``topics.default`` from the config."""
    target = resolve_path(path) if path else resolve_path(topic_file)
    if target.is_file():
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        topics = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        if topics:
            return topics
    return list(default_topics)


topics = load_topics()


# --------------------------------------------------------------------------- #
# inspection helpers
# --------------------------------------------------------------------------- #
def as_dict() -> dict[str, Any]:
    """All resolved settings, for logging or printing."""
    return {
        "project_root": str(PROJECT_ROOT),
        "config_file": str(config_path()),
        "dataset": dataset,
        "dataset_root": dataset_root,
        "duee_dir": duee_dir,
        "resource_dir": resource_dir,
        "output_dir": output_dir,
        "article_dir": article_dir,
        "references_dir": references_dir,
        "topic_file": topic_file,
        "outline_dir": outline_dir,
        "article_source_dir": article_source_dir,
        "evaluation_dataset_dir": evaluation_dataset_dir,
        "topic_dataset_file": topic_dataset_file,
        "topic_dataset_txt": topic_dataset_txt,
        "evaluation_protocol_file": evaluation_protocol_file,
        "evaluation_protocol_doc": evaluation_protocol_doc,
        "dict_file": dict_file,
        "model_root": model_root,
        "language_model": lm_type,
        "spacy_model": spacy_model,
        "gpu_ids": gpu_id,
        "llm": llm.as_dict(),
        "topics": topics,
        "source_dir_aliases": source_dir_aliases,
    }


def export_env() -> str:
    """``export KEY=VALUE`` lines, so shell scripts can reuse the same values."""
    values = {
        "TWPGEN_ROOT": str(PROJECT_ROOT),
        "TWPGEN_CONFIG": str(config_path()),
        "TWPGEN_DATASET": dataset,
        "TWPGEN_DATASET_ROOT": dataset_root,
        "TWPGEN_DUEE_DIR": duee_dir,
        "TWPGEN_RESOURCE_DIR": resource_dir,
        "TWPGEN_OUTPUT_DIR": output_dir,
        "TWPGEN_ARTICLE_DIR": article_dir,
        "TWPGEN_REFERENCES_DIR": references_dir,
        "TWPGEN_TOPIC_FILE": topic_file,
        "TWPGEN_OUTLINE_DIR": outline_dir,
        "TWPGEN_ARTICLE_SOURCE_DIR": article_source_dir,
        "TWPGEN_EVAL_DATASET_DIR": evaluation_dataset_dir,
        "TWPGEN_TOPIC_DATASET": topic_dataset_file,
        "TWPGEN_TOPIC_DATASET_TXT": topic_dataset_txt,
        "TWPGEN_EVAL_PROTOCOL": evaluation_protocol_file,
        "TWPGEN_EVAL_PROTOCOL_MD": evaluation_protocol_doc,
        "TWPGEN_VERB_SENSE_DICT": dict_file,
        "TWPGEN_MODEL_ROOT": model_root,
        "TWPGEN_DOCGEN_ENV": docgen_env,
        "TWPGEN_PIPELINE_ENV": pipeline_env,
        "TWPGEN_ARTICLE_ENV": article_env,
    }
    return "\n".join(f"export {key}={value!r}" for key, value in values.items())


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Print the resolved TWP-Gen configuration.")
    _parser.add_argument(
        "--shell",
        action="store_true",
        help="print export KEY=VALUE lines for shell scripts instead of JSON",
    )
    _args = _parser.parse_args()
    print(export_env() if _args.shell else json.dumps(as_dict(), ensure_ascii=False, indent=2))
