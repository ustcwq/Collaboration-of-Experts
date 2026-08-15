#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm

VLLM_PYTHON=/home/sm5/anaconda3/envs/VLM/bin/python
VLM_PYTHON=/home/sm5/anaconda3/envs/Factory/bin/python
TEXT_MODEL=internlm2_5-1_8b-chat
VISION_MODEL=Phi-4-reasoning-vision-15B
OUTPUT_ROOT=outputs/model_benchmarks/family_scale_expansion_full_20260731
LOG_ROOT=$OUTPUT_ROOT/logs/front4_dynamic_gap
STATE_ROOT=$OUTPUT_ROOT/state/front4_dynamic_gap
mkdir -p "$LOG_ROOT/$TEXT_MODEL" "$LOG_ROOT/$VISION_MODEL" "$STATE_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_ROOT/queue.log"
}

gpu_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v wanted="$gpu" '
        {gsub(/ /, "", $0)}
        ($1 + 0) == wanted {exit !(($2 + 0) < 2048 && ($3 + 0) < 10)}
      '
}

wait_for_gpu() {
  local gpu="$1"
  until gpu_idle "$gpu"; do
    sleep 15
  done
}

run_task() {
  local gpu="$1"
  local task="$2"
  shift 2
  local completed="$STATE_ROOT/$task.completed"
  local failed="$STATE_ROOT/$task.failed"
  mkdir -p "$(dirname "$completed")" "$(dirname "$LOG_ROOT/$task.log")"
  [[ -f "$completed" ]] && return 0
  wait_for_gpu "$gpu"
  log "GPU $gpu starting $task"
  if "$@" > "$LOG_ROOT/$task.log" 2>&1; then
    rm -f "$failed"
    date '+%F %T' > "$completed"
    log "GPU $gpu completed $task"
    return 0
  fi
  date '+%F %T' > "$failed"
  log "GPU $gpu failed $task"
  return 1
}

run_vision_benchmark() {
  local gpu="$1"
  local benchmark="$2"
  run_task "$gpu" "$VISION_MODEL/$benchmark" \
    "$VLM_PYTHON" -m bench_coe.run_multimodal_babyvision_models \
      --models-dir models_v --models "$VISION_MODEL" --benchmarks "$benchmark" \
      --gpu-devices "$gpu" --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/vision/$benchmark"
}

gpu0_chain() {
  run_task 0 "$TEXT_MODEL/official_bbh_gpqa_mmstar" \
    "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
      --models-dir models --models "$TEXT_MODEL" \
      --benchmarks bbh,gpqa,mmstar_text_only --gpqa-configs all \
      --gpu-devices 0 --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/text/official"
  run_vision_benchmark 0 cmmmu
}

gpu1_chain() {
  run_task 1 "$TEXT_MODEL/mmlu_pro_test" \
    "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
      --model-root models --models "$TEXT_MODEL" \
      --validation-file MMLU-Pro/data/test-00000-of-00001.parquet \
      --gpu-id 1 --output-root "$OUTPUT_ROOT/text/mmlu_pro_test"
  run_vision_benchmark 1 mmmu
}

gpu2_chain() {
  run_task 2 "$TEXT_MODEL/gaokao_2010_2022" \
    env CUDA_VISIBLE_DEVICES=2 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2010_2022 \
      --model-path "models/$TEXT_MODEL" --model-name "$TEXT_MODEL" \
      --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao"
  run_vision_benchmark 2 mmmu_pro
}

gpu3_chain() {
  run_task 3 "$TEXT_MODEL/gaokao_2023_2024" \
    env CUDA_VISIBLE_DEVICES=3 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2023_2024 \
      --model-path "models/$TEXT_MODEL" --model-name "$TEXT_MODEL" \
      --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao"
  run_vision_benchmark 3 mathvista
  run_task 3 "$VISION_MODEL/gaokao_mm" \
    "$VLM_PYTHON" -m bench_coe.run_gaokao_mm_babyvision_models \
      --models-dir models_v --models "$VISION_MODEL" \
      --gpu-devices 3 --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/vision/gaokao_mm"
}

log "front4 dynamic gap scheduler started"
gpu0_chain & pid0=$!
gpu1_chain & pid1=$!
gpu2_chain & pid2=$!
gpu3_chain & pid3=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  wait "$pid" || failed=1
done

if [[ "$failed" -eq 0 ]]; then
  date '+%F %T' > "$STATE_ROOT/all.completed"
  log "front4 dynamic gap scheduler completed"
else
  log "front4 dynamic gap scheduler finished with failures"
fi
exit "$failed"
