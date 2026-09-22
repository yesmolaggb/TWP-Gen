#!/usr/bin/env bash
# End-to-end reproduction of the post-outline generation pipeline.
#
#   bash article_generator/src/post_outline/run_pipeline.sh \
#       --topics-file  topics.txt \
#       --outline-root dataset/outline \
#       --source-root  dataset/sources \
#       --env-file     .env \
#       --output-root  output/post_outline
#
# Stages: (1) generate sections from existing outlines
#         (2) build traceable, passage-level references
set -euo pipefail

PY="${PY:-python3}"
OUTLINE_ROOT=""
SOURCE_ROOT=""
ENV_FILE=""
OUTPUT_ROOT="out/run"
TOPICS=1
TOPICS_FILE=""
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outline-root) OUTLINE_ROOT="$2"; shift 2 ;;
    --source-root)  SOURCE_ROOT="$2";  shift 2 ;;
    --env-file)     ENV_FILE="$2";     shift 2 ;;
    --output-root)  OUTPUT_ROOT="$2";  shift 2 ;;
    --topics)       TOPICS="$2";       shift 2 ;;
    --topics-file)  TOPICS_FILE="$2";  shift 2 ;;
    --model)        MODEL="$2";        shift 2 ;;
    --python)       PY="$2";           shift 2 ;;
    -h|--help)      sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTLINE_ROOT" || -z "$SOURCE_ROOT" ]]; then
  echo "error: --outline-root and --source-root are required" >&2
  exit 2
fi

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_DIR="$OUTPUT_ROOT/generated"
REF_DIR="$OUTPUT_ROOT/final"
mkdir -p "$GEN_DIR" "$REF_DIR"

echo "=== stage 1/2: section generation -> $GEN_DIR"
if [[ -n "$TOPICS_FILE" ]]; then
  mapfile -t TOPIC_LIST < <(grep -v '^\s*$' "$TOPICS_FILE")
else
  TOPIC_LIST=()
  for ((i = 0; i < TOPICS; i++)); do TOPIC_LIST+=("__INDEX__$i"); done
fi

pids=()
for entry in "${TOPIC_LIST[@]}"; do
  if [[ "$entry" == __INDEX__* ]]; then
    selector=(--index "${entry#__INDEX__}")
    log_name="${entry#__INDEX__}"
  else
    selector=(--topic "$entry")
    log_name="$(echo "$entry" | tr ' /' '__')"
  fi
  extra_env=()
  if [[ -n "$ENV_FILE" ]]; then
    extra_env+=(--env-file "$ENV_FILE")
  fi
  extra_model=()
  if [[ -n "$MODEL" ]]; then
    extra_model+=(--model "$MODEL")
  fi
  "$PY" "$SRC_DIR/run_one_topic.py" \
    "${selector[@]}" \
    --outline-root "$OUTLINE_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --output-root "$GEN_DIR" \
    ${extra_env[@]+"${extra_env[@]}"} \
    ${extra_model[@]+"${extra_model[@]}"} > "$OUTPUT_ROOT/topic_${log_name}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "one or more topics failed; see $OUTPUT_ROOT/topic_*.log" >&2
fi

echo "=== stage 2/2: passage-level references -> $REF_DIR"
"$PY" "$SRC_DIR/passage_level_citations.py" \
  --input-article-dir "$GEN_DIR/article" \
  --source-root "$SOURCE_ROOT" \
  --output-root "$REF_DIR" \
  --reference-granularity source \
  --excerpt-chars 900 \
  --no-ground-uncited \
  --no-text-edit

echo
echo "done. articles:  $REF_DIR/article"
echo "      records:   $REF_DIR/references"
