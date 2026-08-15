#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PYTHON=${BENCHCOE_VLM_PYTHON:-/home/sm5/anaconda3/envs/Factory/bin/python}
OUTPUT_ROOT=${BENCHCOE_AUTONOMOUS_OUTPUT_ROOT:-outputs/model_benchmarks/autonomous_remaining_full_20260802}
STATE_ROOT="$OUTPUT_ROOT/state"
LOG_ROOT="$OUTPUT_ROOT/logs"
MAX_ATTEMPTS=${BENCHCOE_MAX_ATTEMPTS:-2}
POLL_SECONDS=${BENCHCOE_POLL_SECONDS:-20}
IGNORE_ATTEMPT_LIMIT=${BENCHCOE_IGNORE_ATTEMPT_LIMIT:-0}
FINAL_PREFIX=${BENCHCOE_FINAL_PREFIX:-}
SCHEDULER_LOG=${BENCHCOE_SCHEDULER_LOG:-$LOG_ROOT/scheduler.log}
ATTN_IMPLEMENTATION=${BENCHCOE_ATTN_IMPLEMENTATION:-}
mkdir -p "$STATE_ROOT" "$LOG_ROOT"

ATTN_ARGS=()
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  ATTN_ARGS=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi

if [[ -n "${BENCHCOE_MODELS:-}" ]]; then
  read -r -a MODELS <<<"$BENCHCOE_MODELS"
else
  MODELS=(
    Qwen2.5-VL-7B-Instruct
    InternVL3_5-14B
    gemma-3-12b-it
    Qwen3-VL-8B-Thinking
  )
fi
read -r -a GPUS <<<"${BENCHCOE_GPUS:-0 1 2 3}"
BENCHMARKS=(cmmmu mmmu mmmu_pro mathvista gaokao_mm)

declare -A GPU_PIDS GPU_TASKS TASK_PIDS

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$SCHEDULER_LOG"
}

task_key() {
  printf '%s__%s' "$1" "$2"
}

gpu_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v wanted="$gpu" '
      {gsub(/ /, "", $0)}
      ($1 + 0) == wanted {found=1; exit !(($2 + 0) < 2048 && ($3 + 0) < 10)}
      END {if (!found) exit 1}
    '
}

run_task() {
  local gpu="$1" model="$2" benchmark="$3" log_file="$4"
  if [[ "$benchmark" == gaokao_mm ]]; then
    "$PYTHON" -m bench_coe.run_gaokao_mm_babyvision_models \
      --models-dir models_v --models "$model" --gpu-devices "$gpu" \
      --parallel-workers 1 --output-dir "$OUTPUT_ROOT/vision/gaokao_mm" \
      "${ATTN_ARGS[@]}" \
      >"$log_file" 2>&1
  else
    "$PYTHON" -m bench_coe.run_multimodal_babyvision_models \
      --models-dir models_v --models "$model" --benchmarks "$benchmark" \
      --gpu-devices "$gpu" --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/vision/$benchmark" \
      "${ATTN_ARGS[@]}" \
      >"$log_file" 2>&1
  fi
}

reap_jobs() {
  local gpu pid key status
  for gpu in "${GPUS[@]}"; do
    pid="${GPU_PIDS[$gpu]:-}"
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    key="${GPU_TASKS[$gpu]}"
    wait "$pid"; status=$?
    if [[ "$status" -eq 0 ]]; then
      date '+%F %T' >"$STATE_ROOT/$key.completed"
      rm -f "$STATE_ROOT/$key.failed"
      log "GPU$gpu completed $key"
    else
      date '+%F %T' >"$STATE_ROOT/$key.failed"
      log "GPU$gpu failed $key with status $status"
    fi
    unset 'GPU_PIDS[$gpu]' 'GPU_TASKS[$gpu]' 'TASK_PIDS[$key]'
  done
}

next_task() {
  local model benchmark key attempt
  for model in "${MODELS[@]}"; do
    [[ -f "models_v/$model/.benchcoe_modelscope_complete.json" ]] || continue
    for benchmark in "${BENCHMARKS[@]}"; do
      key="$(task_key "$model" "$benchmark")"
      [[ -f "$STATE_ROOT/$key.completed" ]] && continue
      [[ -n "${TASK_PIDS[$key]:-}" ]] && continue
      attempt=0
      [[ -f "$STATE_ROOT/$key.attempt" ]] && attempt=$(<"$STATE_ROOT/$key.attempt")
      [[ "$IGNORE_ATTEMPT_LIMIT" == 1 ]] || (( attempt < MAX_ATTEMPTS )) || continue
      printf '%s|%s|%s\n' "$model" "$benchmark" "$key"
      return 0
    done
  done
  return 1
}

remaining_count() {
  local model benchmark key attempt count=0
  for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
      key="$(task_key "$model" "$benchmark")"
      [[ -f "$STATE_ROOT/$key.completed" ]] && continue
      attempt=0
      [[ -f "$STATE_ROOT/$key.attempt" ]] && attempt=$(<"$STATE_ROOT/$key.attempt")
      if [[ "$IGNORE_ATTEMPT_LIMIT" == 1 ]] || (( attempt < MAX_ATTEMPTS )); then
        count=$((count + 1))
      fi
    done
  done
  printf '%s\n' "$count"
}

log "autonomous per-GPU vision scheduler started"
while :; do
  reap_jobs
  launched=0
  for gpu in "${GPUS[@]}"; do
    [[ -z "${GPU_PIDS[$gpu]:-}" ]] || continue
    gpu_idle "$gpu" || continue
    task=$(next_task) || break
    IFS='|' read -r model benchmark key <<<"$task"
    attempt=0
    [[ -f "$STATE_ROOT/$key.attempt" ]] && attempt=$(<"$STATE_ROOT/$key.attempt")
    attempt=$((attempt + 1))
    printf '%s\n' "$attempt" >"$STATE_ROOT/$key.attempt"
    mkdir -p "$LOG_ROOT/$model"
    log_file="$LOG_ROOT/$model/${benchmark}.attempt${attempt}.log"
    run_task "$gpu" "$model" "$benchmark" "$log_file" &
    pid=$!
    GPU_PIDS[$gpu]="$pid"
    GPU_TASKS[$gpu]="$key"
    TASK_PIDS[$key]="$pid"
    launched=1
    log "GPU$gpu started $key attempt $attempt"
    sleep 3
  done

  active=${#GPU_PIDS[@]}
  remaining=$(remaining_count)
  if [[ "$active" -eq 0 && "$remaining" -eq 0 ]]; then
    break
  fi
  [[ "$launched" -eq 1 ]] || sleep "$POLL_SECONDS"
done

failed=0
for model in "${MODELS[@]}"; do
  for benchmark in "${BENCHMARKS[@]}"; do
    key="$(task_key "$model" "$benchmark")"
    if [[ ! -f "$STATE_ROOT/$key.completed" ]]; then
      failed=$((failed + 1))
      date '+%F %T' >"$STATE_ROOT/$key.failed_final"
    fi
  done
done
if [[ -n "$FINAL_PREFIX" ]]; then
  printf '%s\n' "$failed" >"$STATE_ROOT/${FINAL_PREFIX}_final_failed_count"
  date '+%F %T' >"$STATE_ROOT/${FINAL_PREFIX}.completed"
else
  printf '%s\n' "$failed" >"$STATE_ROOT/final_failed_count"
  date '+%F %T' >"$STATE_ROOT/scheduler.completed"
fi
log "scheduler finished with $failed permanently failed tasks"
exit 0
