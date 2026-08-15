#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm

VLLM_PYTHON=${BENCHCOE_VLLM_PYTHON:-/home/sm5/anaconda3/envs/VLM/bin/python}
VLM_PYTHON=${BENCHCOE_VLM_PYTHON:-/home/sm5/anaconda3/envs/Factory/bin/python}
OUTPUT_ROOT=${BENCHCOE_EVAL_OUTPUT_ROOT:-outputs/model_benchmarks/scale_extension_full_20260731}
LOG_ROOT="$OUTPUT_ROOT/logs"
STATE_ROOT="$OUTPUT_ROOT/state"
mkdir -p "$LOG_ROOT" "$STATE_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_ROOT/queue.log"
}

front4_idle() {
  local idle_count
  idle_count=$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F, '{gsub(/ /, "", $0); if (($1 + 0) < 4 && ($2 + 0) < 2048 && ($3 + 0) < 10) count++} END {print count + 0}'
  )
  [[ "$idle_count" -eq 4 ]]
}

wait_for_front4() {
  until front4_idle; do
    log "waiting for GPUs 0-3 to become idle"
    sleep 60
  done
}

run_logged() {
  local log_file="$1"
  shift
  "$@" > "$log_file" 2>&1
}

wait_for_jobs() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

run_text_model() {
  local model_name="$1"
  local model_path="models/$model_name"
  local model_log="$LOG_ROOT/$model_name"
  mkdir -p "$model_log"

  log "$model_name: starting full text evaluation on GPUs 0-3"
  run_logged "$model_log/gpu0_bbh_gpqa_mmstar.log" \
    "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
      --models-dir models \
      --models "$model_name" \
      --benchmarks bbh,gpqa,mmstar_text_only \
      --gpqa-configs all \
      --gpu-devices 0 \
      --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/text/official" &
  local pid0=$!

  run_logged "$model_log/gpu1_mmlu_pro_test.log" \
    "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
      --model-root models \
      --models "$model_name" \
      --validation-file MMLU-Pro/data/test-00000-of-00001.parquet \
      --gpu-id 1 \
      --output-root "$OUTPUT_ROOT/text/mmlu_pro_test" &
  local pid1=$!

  run_logged "$model_log/gpu2_gaokao_2010_2022.log" \
    env CUDA_VISIBLE_DEVICES=2 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2010_2022 \
      --model-path "$model_path" \
      --model-name "$model_name" \
      --max-examples-per-task 0 \
      --output-dir "$OUTPUT_ROOT/text/gaokao" &
  local pid2=$!

  run_logged "$model_log/gpu3_gaokao_2023_2024.log" \
    env CUDA_VISIBLE_DEVICES=3 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2023_2024 \
      --model-path "$model_path" \
      --model-name "$model_name" \
      --max-examples-per-task 0 \
      --output-dir "$OUTPUT_ROOT/text/gaokao" &
  local pid3=$!

  wait_for_jobs "$pid0" "$pid1" "$pid2" "$pid3"
}

run_vision_benchmark() {
  local gpu="$1"
  local model_name="$2"
  local benchmark="$3"
  local output_dir="$OUTPUT_ROOT/vision/$benchmark"
  run_logged "$LOG_ROOT/$model_name/gpu${gpu}_${benchmark}.log" \
    "$VLM_PYTHON" -m bench_coe.run_multimodal_babyvision_models \
      --models-dir models_v \
      --models "$model_name" \
      --benchmarks "$benchmark" \
      --gpu-devices "$gpu" \
      --parallel-workers 1 \
      --output-dir "$output_dir"
}

run_vision_model() {
  local model_name="$1"
  local model_log="$LOG_ROOT/$model_name"
  mkdir -p "$model_log"

  log "$model_name: starting full vision evaluation on GPUs 0-3"
  run_vision_benchmark 0 "$model_name" cmmmu &
  local pid0=$!
  run_vision_benchmark 1 "$model_name" mmmu &
  local pid1=$!
  run_vision_benchmark 2 "$model_name" mmmu_pro &
  local pid2=$!
  (
    run_vision_benchmark 3 "$model_name" mathvista && \
    run_logged "$model_log/gpu3_gaokao_mm.log" \
      "$VLM_PYTHON" -m bench_coe.run_gaokao_mm_babyvision_models \
        --models-dir models_v \
        --models "$model_name" \
        --gpu-devices 3 \
        --parallel-workers 1 \
        --output-dir "$OUTPUT_ROOT/vision/gaokao_mm"
  ) &
  local pid3=$!

  wait_for_jobs "$pid0" "$pid1" "$pid2" "$pid3"
}

process_model() {
  local kind="$1"
  local model_name="$2"
  local model_root="$3"
  local complete_marker="$model_root/$model_name/.benchcoe_modelscope_complete.json"
  local success_marker="$STATE_ROOT/$model_name.completed"
  local failure_marker="$STATE_ROOT/$model_name.failed"

  if [[ -f "$success_marker" ]]; then
    log "$model_name: evaluation already completed"
    return
  fi

  while [[ ! -f "$complete_marker" ]]; do
    if [[ "${BENCHCOE_ALLOW_EXISTING_MODELS:-0}" == 1 && -f "$model_root/$model_name/config.json" ]]; then
      log "$model_name: using existing verified local model"
      break
    fi
    log "$model_name: waiting for completed download marker"
    sleep 60
  done

  wait_for_front4
  rm -f "$failure_marker"
  printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$model_name.started"
  if [[ "$kind" == text ]]; then
    run_text_model "$model_name"
  else
    run_vision_model "$model_name"
  fi
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '%s\n' "$(date '+%F %T')" > "$success_marker"
    log "$model_name: all full evaluations completed"
  else
    printf '%s\n' "$(date '+%F %T') status=$status" > "$failure_marker"
    log "$model_name: one or more evaluations failed; continuing to next model"
  fi
}

MODEL_QUEUE=(
  "text|Qwen3-1.7B|models"
  "text|Qwen2.5-3B-Instruct|models"
  "text|Qwen3-14B|models"
  "text|Qwen2.5-14B-Instruct|models"
  "vision|InternVL3_5-8B|models_v"
  "vision|Qwen2.5-VL-7B-Instruct|models_v"
  "vision|InternVL3_5-14B|models_v"
)

if [[ -n "${BENCHCOE_MODEL_QUEUE_FILE:-}" ]]; then
  mapfile -t MODEL_QUEUE < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$BENCHCOE_MODEL_QUEUE_FILE")
fi

log "download-triggered full evaluation queue started"
for entry in "${MODEL_QUEUE[@]}"; do
  IFS='|' read -r kind model_name model_root <<< "$entry"
  process_model "$kind" "$model_name" "$model_root"
done
log "download-triggered full evaluation queue completed"
