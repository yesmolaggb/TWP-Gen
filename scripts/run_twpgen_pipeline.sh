#!/usr/bin/env bash
set -Eeuo pipefail

# ── 工作目录配置 ──
# 所有路径、API Key、模型名、主题清单等统一在 <repo>/twpgen_settings.py 定义
# （取值来自 twpgen_config.json 与 .env）。这里只解析一次，其余变量全部从它导出。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWPGEN_ROOT="${TWPGEN_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$TWPGEN_ROOT/.env" ]]; then
  set -a
  source "$TWPGEN_ROOT/.env"
  set +a
fi

CONFIG_PY="${TWPGEN_CONFIG_PY:-$TWPGEN_ROOT/twpgen_settings.py}"
if [[ -f "$CONFIG_PY" ]] && command -v python3 >/dev/null 2>&1; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && eval "$line"
  done < <(python3 "$CONFIG_PY" --shell)
fi

# Central config lives at the repository root. Create it on first run so that
# every stage below reads the same file.
ROOT_CONFIG="${TWPGEN_CONFIG:-$TWPGEN_ROOT/twpgen_config.json}"
if [[ ! -f "$ROOT_CONFIG" && -f "$TWPGEN_ROOT/twpgen_config.example.json" ]]; then
  cp "$TWPGEN_ROOT/twpgen_config.example.json" "$ROOT_CONFIG"
fi
export TWPGEN_CONFIG="$ROOT_CONFIG"

SEARCH_REF_ROOT="${TWPGEN_KNOWLEDGE_ROOT:-${TWPGEN_SEARCH_ROOT:-$TWPGEN_ROOT/knowledge_collector}}"
OUTLINE_ROOT="${TWPGEN_OUTLINE_ROOT:-$TWPGEN_ROOT/outline_generator}"
ARTICLE_ROOT="${TWPGEN_ARTICLE_ROOT:-$TWPGEN_ROOT/article_generator}"

BASE_STATE_DIR="$TWPGEN_ROOT/.pipeline_state"
BASE_LOG_DIR="$TWPGEN_ROOT/logs"
DATASET_BASE_DIR="$TWPGEN_ROOT/dataset"

TOPIC_FILE="${TWPGEN_TOPIC_FILE:-$SEARCH_REF_ROOT/topic.txt}"

if [[ "$SEARCH_REF_ROOT" != /* ]]; then
  SEARCH_REF_ROOT="$TWPGEN_ROOT/${SEARCH_REF_ROOT#./}"
fi
if [[ "$TOPIC_FILE" != /* ]]; then
  TOPIC_FILE="$TWPGEN_ROOT/${TOPIC_FILE#./}"
fi
if [[ "$OUTLINE_ROOT" != /* ]]; then
  OUTLINE_ROOT="$TWPGEN_ROOT/${OUTLINE_ROOT#./}"
fi
if [[ "$ARTICLE_ROOT" != /* ]]; then
  ARTICLE_ROOT="$TWPGEN_ROOT/${ARTICLE_ROOT#./}"
fi

mkdir -p "$BASE_STATE_DIR" "$BASE_LOG_DIR" "$DATASET_BASE_DIR"

CURRENT_STAGE=""

# ── 当前题目的 state 目录（每个题目独立）──
STATE_DIR=""
FAILED_FILE=""
MAIN_STATE_FILE=""

print_stage() {
  printf '%s\n' "$1"
}

save_main_state() {
  printf '%s\n' "$1" > "$MAIN_STATE_FILE"
}

get_main_state() {
  if [[ -f "$MAIN_STATE_FILE" ]]; then
    cat "$MAIN_STATE_FILE"
  else
    printf '%s\n' ""
  fi
}

save_failed_state() {
  printf '%s\n' "$1" > "$FAILED_FILE"
}

get_failed_state() {
  if [[ -f "$FAILED_FILE" ]]; then
    cat "$FAILED_FILE"
  else
    printf '%s\n' ""
  fi
}

clear_failed_state() {
  rm -f "$FAILED_FILE"
}

clear_all_state() {
  rm -f "$MAIN_STATE_FILE" "$FAILED_FILE"
  rm -f "$STATE_DIR"/branch1.state "$STATE_DIR"/branch2.state "$STATE_DIR"/branch3.state
  rm -f "$STATE_DIR"/branch1.done "$STATE_DIR"/branch2.done "$STATE_DIR"/branch3.done
  rm -f "$STATE_DIR"/branch1.failed "$STATE_DIR"/branch2.failed "$STATE_DIR"/branch3.failed
}

on_error() {
  local exit_code=$?
  if [[ -n "${CURRENT_STAGE:-}" ]]; then
    save_failed_state "$CURRENT_STAGE"
  fi
  exit "$exit_code"
}
trap on_error ERR

init_conda() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      source "$conda_base/etc/profile.d/conda.sh"
      return
    fi
  fi

  if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    return
  fi

  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    return
  fi
}

activate_env() {
  local env_path="$1"
  init_conda
  if command -v conda >/dev/null 2>&1; then
    conda activate "$env_path" >/dev/null 2>&1 || source activate "$env_path" >/dev/null 2>&1
  else
    source activate "$env_path" >/dev/null 2>&1
  fi
}

run_quiet() {
  local log_file="$1"
  shift
  "$@" >>"$log_file" 2>&1
}

# 将题目名转换为安全的文件夹名（与 collect_references.py 中 safe_filename 逻辑一致）
safe_name() {
  local name="$1"
  python3 - "$name" <<'PYEOF'
import sys, re
name = sys.argv[1]
name = re.sub(r'[\\/:*?"<>|]+', '_', name)
name = re.sub(r'\s+', ' ', name).strip()
if not name:
    name = 'untitled'
print(name[:120])
PYEOF
}

run_step() {
  local step_id="$1"
  local stage_name="$2"
  local log_name="$3"
  shift 3

  local main_state
  main_state="$(get_main_state)"
  if [[ "$main_state" == "$step_id" ]]; then
    return 0
  fi

  CURRENT_STAGE="$step_id"
  print_stage "$stage_name"
  run_quiet "$LOG_DIR/$log_name" "$@"
  save_main_state "$step_id"
  clear_failed_state
}

run_python_step() {
  local step_id="$1"
  local stage_name="$2"
  local log_name="$3"
  local pyfile="$4"
  shift 4
  run_step "$step_id" "$stage_name" "$log_name" python "$pyfile" "$@"
}

run_branch_step() {
  local branch_name="$1"
  local step_id="$2"
  local stage_name="$3"
  local log_name="$4"
  shift 4

  local branch_state_file="$STATE_DIR/${branch_name}.state"
  local branch_failed_file="$STATE_DIR/${branch_name}.failed"
  local current_state=""

  if [[ -f "$branch_state_file" ]]; then
    current_state="$(cat "$branch_state_file")"
  fi

  if [[ "$current_state" == "$step_id" ]]; then
    return 0
  fi

  CURRENT_STAGE="${branch_name}:${step_id}"
  print_stage "$stage_name"
  if ! run_quiet "$LOG_DIR/$log_name" "$@"; then
    printf '%s\n' "$step_id" > "$branch_failed_file"
    save_failed_state "${branch_name}:${step_id}"
    return 1
  fi

  printf '%s\n' "$step_id" > "$branch_state_file"
  rm -f "$branch_failed_file"
  clear_failed_state
}

run_branch_python_step() {
  local branch_name="$1"
  local step_id="$2"
  local stage_name="$3"
  local log_name="$4"
  local pyfile="$5"
  run_branch_step "$branch_name" "$step_id" "$stage_name" "$log_name" python "$pyfile"
}

wait_until_not_running() {
  local keyword="$1"
  while pgrep -af "$keyword" >/dev/null 2>&1; do
    sleep 5
  done
}

mark_branch_done() {
  local branch_name="$1"
  : > "$STATE_DIR/${branch_name}.done"
  rm -f "$STATE_DIR/${branch_name}.failed"
}

branch_is_done() {
  local branch_name="$1"
  [[ -f "$STATE_DIR/${branch_name}.done" ]]
}

branch1() {
  if branch_is_done "branch1"; then
    return 0
  fi

  run_branch_python_step "branch1" "extract_salient_terms" "阶段：路线一 - extract_salient_terms" "branch1_extract_salient_terms.log" "$OUTLINE_ROOT/extract_salient_terms.py"
  run_branch_python_step "branch1" "fuse_multidimensional_features" "阶段：路线一 - fuse_multidimensional_features" "branch1_fuse_multidimensional_features.log" "$OUTLINE_ROOT/fuse_multidimensional_features.py"
  run_branch_python_step "branch1" "resolve_semantic_senses" "阶段：路线一 - resolve_semantic_senses" "branch1_resolve_semantic_senses.log" "$OUTLINE_ROOT/resolve_semantic_senses.py"
  run_branch_step "branch1" "remove_savepoint" "阶段：路线一 - 删除 savepoint.point" "branch1_remove_savepoint.log" rm -f "$DATASET_BASE_DIR/${TWPGEN_DATASET}/savepoint.point"
  run_branch_python_step "branch1" "construct_evidence_graph" "阶段：路线一 - construct_evidence_graph" "branch1_construct_evidence_graph.log" "$OUTLINE_ROOT/construct_evidence_graph.py"
  run_branch_python_step "branch1" "train_graph_autoencoder" "阶段：路线一 - train_graph_autoencoder" "branch1_train_graph_autoencoder.log" "$OUTLINE_ROOT/train_graph_autoencoder.py"

  mark_branch_done "branch1"
}

branch2() {
  if branch_is_done "branch2"; then
    return 0
  fi

  run_branch_python_step "branch2" "compress_evidence_sentences" "阶段：路线二 - compress_evidence_sentences" "branch2_compress_evidence_sentences.log" "$OUTLINE_ROOT/compress_evidence_sentences.py"
  CURRENT_STAGE="branch2:wait_fuse_multidimensional_features"
  print_stage "阶段：路线二 - 等待 fuse_multidimensional_features 结束"
  wait_until_not_running "fuse_multidimensional_features.py"
  run_branch_python_step "branch2" "encode_sentence_summaries" "阶段：路线二 - encode_sentence_summaries" "branch2_encode_sentence_summaries.log" "$OUTLINE_ROOT/encode_sentence_summaries.py"

  mark_branch_done "branch2"
}

branch3() {
  if branch_is_done "branch3"; then
    return 0
  fi

  run_branch_python_step "branch3" "extract_event_features" "阶段：路线三 - extract_event_features" "branch3_extract_event_features.log" "$OUTLINE_ROOT/extract_event_features.py"
  CURRENT_STAGE="branch3:wait_fuse_multidimensional_features"
  print_stage "阶段：路线三 - 等待 fuse_multidimensional_features 结束"
  wait_until_not_running "fuse_multidimensional_features.py"
  run_branch_python_step "branch3" "encode_event_features" "阶段：路线三 - encode_event_features" "branch3_encode_event_features.log" "$OUTLINE_ROOT/encode_event_features.py"

  mark_branch_done "branch3"
}

run_parallel_branches() {
  local pid1 pid2 pid3
  local rc=0

  print_stage "阶段：开始并行三条路线"

  branch1 &
  pid1=$!

  branch2 &
  pid2=$!

  branch3 &
  pid3=$!

  wait "$pid1" || rc=1
  wait "$pid2" || rc=1
  wait "$pid3" || rc=1

  if [[ "$rc" -ne 0 ]]; then
    exit 1
  fi
}

# ── 读取 topic.txt，返回题目列表（跳过空行）──
load_topics() {
  local topic_file="$1"
  if [[ ! -f "$topic_file" ]]; then
    printf '✗ 题目文件不存在: %s\n' "$topic_file"
    exit 1
  fi
  grep -v '^[[:space:]]*$' "$topic_file" || true
}

# ── 检查题目是否已完全处理过（根据最终产出物是否存在）──
is_topic_processed() {
  local topic="$1"
  local sname
  sname="$(safe_name "$topic")"
  local topic_dir="$DATASET_BASE_DIR/$sname"
  # 必须 corpus.txt、parsed_corpus.pk、po_tuple_features_all_svos.pk、outline.txt 都存在才算完成
  [[ -f "$topic_dir/corpus.txt"          &&
     -f "$topic_dir/parsed_corpus.pk"    &&
     -f "$topic_dir/po_tuple_features_all_svos.pk" &&
     -f "$topic_dir/outline.txt"         &&
     -f "$TWPGEN_ROOT/output/post_outline/article/$sname.md" ]]
}

# ── 针对单个题目，询问是否从断点继续 ──
prepare_topic_session() {
  local failed_step
  failed_step="$(get_failed_state)"

  if [[ -n "$failed_step" ]]; then
    printf '检测到上次报错位置：%s，是否从上次报错的地方继续运行？[y/N]\n' "$failed_step"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      clear_failed_state
      return 0
    fi
  fi

  clear_all_state
}

# ── 单个题目完整 pipeline ──
run_topic_pipeline() {
  local topic="$1"
  local sname
  sname="$(safe_name "$topic")"

  # 设置环境变量，所有 TWP-Gen 脚本通过 twpgen_config.py 读取
  export TWPGEN_DATASET="$sname"
  export TWPGEN_DATASET_DIR="$DATASET_BASE_DIR/$sname"

  # 每个题目独立 state 目录和日志目录
  STATE_DIR="$BASE_STATE_DIR/$sname"
  MAIN_STATE_FILE="$STATE_DIR/main.state"
  FAILED_FILE="$STATE_DIR/failed.state"
  LOG_DIR="$BASE_LOG_DIR/$sname"
  mkdir -p "$STATE_DIR" "$LOG_DIR"

  printf '\n%s\n' "══════════════════════════════════════════════════"
  printf '题目：%s\n' "$topic"
  printf '数据目录：%s\n' "$TWPGEN_DATASET_DIR"
  printf '%s\n' "══════════════════════════════════════════════════"

  prepare_topic_session

  # Step 1: 获取新闻（只处理当前题目）
activate_env "${TWPGEN_DOCGEN_ENV:-$TWPGEN_ROOT/.venv-docgen}"
  cd "$SEARCH_REF_ROOT"
  export TWPGEN_ROOT="$TWPGEN_ROOT"
  export TWPGEN_DATASET_ROOT="$DATASET_BASE_DIR"
  export TWPGEN_TOPIC_FILE="$TOPIC_FILE"
  export TWPGEN_COLLECTOR_RESULT_DIR="${TWPGEN_COLLECTOR_RESULT_DIR:-$SEARCH_REF_ROOT/result}"
  run_python_step "collect_references" "阶段：获取新闻" "collect_references.log" \
    "$SEARCH_REF_ROOT/collect_references.py" --topic "$topic"

  # Step 2: TWP-Gen 流程
activate_env "${TWPGEN_PIPELINE_ENV:-$TWPGEN_ROOT/.venv}"
  cd "$TWPGEN_ROOT"

  run_python_step "topic_init_prepare_corpus" "阶段：初始化语料" "topic_init_prepare_corpus.log" \
    "$OUTLINE_ROOT/prepare_corpus.py"
  run_python_step "twpgen_parse_corpus_svo" "阶段：解析语料并抽取 SVO" "twpgen_parse_corpus_svo.log" \
    "$OUTLINE_ROOT/parse_corpus_svo.py"

  run_parallel_branches

  run_python_step "final_build_tuple_feature_bank" "阶段：生成 po tuple features" "final_build_tuple_feature_bank.log" \
    "$OUTLINE_ROOT/build_tuple_feature_bank.py"
  run_python_step "final_run_twpgen" "阶段：运行主流程" "final_run_twpgen.log" \
    "$OUTLINE_ROOT/run_twpgen.py"
  run_python_step "final_retrieve_outline_evidence" "阶段：句子匹配检索" "final_retrieve_outline_evidence.log" \
    "$OUTLINE_ROOT/retrieve_outline_evidence.py"
  run_python_step "final_generate_whitepaper_outline" "阶段：生成大纲" "final_generate_whitepaper_outline.log" \
    "$OUTLINE_ROOT/generate_whitepaper_outline.py"

  # Step 3: Evidence-grounded article generation from the generated outline.
  # Use the same citation-bound path documented in the README so the one-command
  # runner and the step-by-step workflow execute the same implementation.
  activate_env "${TWPGEN_ARTICLE_ENV:-${TWPGEN_PIPELINE_ENV:-$TWPGEN_ROOT/.venv}}"
  cd "$ARTICLE_ROOT"
  mkdir -p "$TWPGEN_ROOT/output/post_outline"
  run_python_step "final_generate_article" "Article generation from outline" "final_generate_article.log" \
    "$ARTICLE_ROOT/src/post_outline/run_postoutline_experiment.py" \
    --outline-root "$DATASET_BASE_DIR" \
    --source-root "$TWPGEN_COLLECTOR_RESULT_DIR" \
    --output-root "$TWPGEN_ROOT/output/post_outline" \
    --topic "$topic"

  clear_all_state
  printf '题目完成：%s\n' "$topic"
}

main() {
  local topics
  mapfile -t topics < <(load_topics "$TOPIC_FILE")

  if [[ ${#topics[@]} -eq 0 ]]; then
    printf '✗ %s 中没有可处理的题目\n' "$TOPIC_FILE"
    exit 1
  fi

  printf '共读取到 %d 个题目\n' "${#topics[@]}"

  local idx=1
  local skipped=0
  for topic in "${topics[@]}"; do
    if is_topic_processed "$topic"; then
      printf '[%d/%d] 跳过（已处理）：%s\n' "$idx" "${#topics[@]}" "$topic"
      skipped=$((skipped + 1))
      idx=$((idx + 1))
      continue
    fi

    printf '[%d/%d] 开始处理：%s\n' "$idx" "${#topics[@]}" "$topic"
    run_topic_pipeline "$topic"
    idx=$((idx + 1))
  done

  printf '\n全部题目处理完成 ✓ (跳过 %d 个已处理题目)\n' "$skipped"
}

main "$@"
