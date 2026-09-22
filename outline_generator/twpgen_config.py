"""Backward-compatible view of the central configuration.

Every path, key and tunable now lives in ``<repo>/twpgen_settings.py`` (values in
``<repo>/twpgen_config.json``). This module only re-exports them so existing
``import twpgen_config as args`` statements keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twpgen_settings import *  # noqa: F401,F403,E402
from twpgen_settings import (  # noqa: F401  (explicit, helps editors/tools)
    PROJECT_ROOT,
    acc_permutation_times,
    all_lemma_freq_file,
    article_dir,
    article_source_dir,
    as_dict,
    config_path,
    corpus,
    corpus_info_path,
    dataset,
    dataset_root,
    dict_file,
    duee_dir,
    evaluation_dataset_dir,
    evaluation_protocol_doc,
    evaluation_protocol_file,
    export_env,
    feature_path,
    gae_feature_path,
    graph_model,
    graph_path,
    gpu_id,
    hidden1_dim,
    hidden2_dim,
    input_dim,
    label_whitelist,
    language_model_paths,
    learning_rate,
    llm,
    lm_type,
    load_config,
    load_env_file,
    load_topics,
    mention_file,
    min_obj_freq,
    min_verb_freq,
    model_root,
    num_epoch,
    outline_dir,
    output_dir,
    paper_title,
    parsed_corpus,
    pretrained_weights,
    project_name,
    references_dir,
    resolve_path,
    resource_dir,
    save_disambiguated_path,
    save_po_tuple_feature_path,
    spacy_model,
    topic_file,
    top_k_expand_result,
    top_obj_ratio,
    top_verb_ratio,
    topics,
    topic_dataset_file,
    topic_dataset_txt,
    use_all_svos,
    verb_freq_file,
    weight_threshold,
    weight_v2n,
    word_graph,
)
