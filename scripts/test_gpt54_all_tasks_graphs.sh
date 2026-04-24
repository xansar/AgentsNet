#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt-5.4}"
GRAPH_SIZE="${GRAPH_SIZE:-4}"
ROUNDS="${ROUNDS:-1}"
SAMPLES_PER_GRAPH_MODEL="${SAMPLES_PER_GRAPH_MODEL:-1}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/260424-gpt54-nodes${GRAPH_SIZE}-rounds${ROUNDS}}"
MAX_PARALLEL_EXPERIMENTS="${MAX_PARALLEL_EXPERIMENTS:-4}"
START_FROM_SAMPLE="${START_FROM_SAMPLE:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_NUM="${RUN_NUM:-1}"
REASONING_EFFORT="${REASONING_EFFORT:-}"

TASKS=(${TASKS:-coloring leader_election matching vertex_cover consensus})
GRAPH_MODELS=(${GRAPH_MODELS:-ws ba dt})

WEBHOOK="${FEISHU_WEBHOOK_URL-https://www.feishu.cn/flow/api/trigger-webhook/b0d9ebf1d95a32009be1b67d0d6ef493}"
SCRIPT_NAME="$(basename "$0")"
BASE_NAME="${SCRIPT_NAME%.*}"
START_TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"

remaining_samples=$((SAMPLES_PER_GRAPH_MODEL - START_FROM_SAMPLE))
if [[ "$remaining_samples" -lt 0 ]]; then
    remaining_samples=0
fi
RUNS_PER_TASK=$((${#GRAPH_MODELS[@]} * remaining_samples))
TOTAL_RUNS=$((${#TASKS[@]} * RUNS_PER_TASK))
completed_count=0

json_escape() {
    python -c 'import json, sys; print(json.dumps(sys.argv[1], ensure_ascii=False)[1:-1])' "$1"
}

send_finish_notification() {
    local status="$1"
    local end_ts="$2"
    local exit_code="$3"
    local payload

    if [[ -z "$WEBHOOK" ]]; then
        echo "[WARN] FEISHU_WEBHOOK_URL is empty; skip Feishu notification" >&2
        return 0
    fi

    payload=$(printf '{"msg_type":"text","status":"%s","script_name":"%s","base_name":"%s","model":"%s","run_num":%s,"reasoning_effort":"%s","completed":%s,"total_runs":%s,"start_ts":"%s","end_ts":"%s","exit_code":%s,"tasks":"%s","graph_models":"%s","graph_size":%s,"rounds":%s,"samples_per_graph_model":%s,"start_from_sample":%s,"output_dir":"%s"}' \
        "$(json_escape "$status")" \
        "$(json_escape "$SCRIPT_NAME")" \
        "$(json_escape "$BASE_NAME")" \
        "$(json_escape "$MODEL")" \
        "$RUN_NUM" \
        "$(json_escape "$REASONING_EFFORT")" \
        "$completed_count" \
        "$TOTAL_RUNS" \
        "$(json_escape "$START_TS")" \
        "$(json_escape "$end_ts")" \
        "$exit_code" \
        "$(json_escape "${TASKS[*]}")" \
        "$(json_escape "${GRAPH_MODELS[*]}")" \
        "$GRAPH_SIZE" \
        "$ROUNDS" \
        "$SAMPLES_PER_GRAPH_MODEL" \
        "$START_FROM_SAMPLE" \
        "$(json_escape "$OUTPUT_DIR")")

    if [[ "${FEISHU_NOTIFY_DRY_RUN:-0}" == "1" ]]; then
        printf '%s\n' "$payload"
        return 0
    fi

    curl -sS -X POST "$WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "$payload" >/dev/null || echo "[WARN] Failed to send Feishu notification" >&2
}

on_exit() {
    local exit_code=$?
    local end_ts
    end_ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"

    if [[ "$exit_code" -eq 0 ]]; then
        send_finish_notification "success" "$end_ts" "0"
    else
        send_finish_notification "failed" "$end_ts" "$exit_code"
    fi
}
trap on_exit EXIT

export USE_AZURE_OPENAI_AAD="${USE_AZURE_OPENAI_AAD:-1}"
export GPT_ENDPOINT="${GPT_ENDPOINT:-https://societalllm.openai.azure.com/}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"

mkdir -p "$OUTPUT_DIR"

for task in "${TASKS[@]}"; do
    echo "Running task=${task}, graph_models=${GRAPH_MODELS[*]}, graph_size=${GRAPH_SIZE}, rounds=${ROUNDS}"

    uv run main.py \
        --graph_size "$GRAPH_SIZE" \
        --task "$task" \
        --rounds "$ROUNDS" \
        --samples_per_graph_model "$SAMPLES_PER_GRAPH_MODEL" \
        --graph_models "${GRAPH_MODELS[@]}" \
        --model "$MODEL" \
        --seed "$SEED" \
        --start_from_sample "$START_FROM_SAMPLE" \
        --disable_chain_of_thought \
        --output_dir "$OUTPUT_DIR" \
        --max_parallel_experiments "$MAX_PARALLEL_EXPERIMENTS" \
        --log_level "$LOG_LEVEL"

    completed_count=$((completed_count + RUNS_PER_TASK))
done
