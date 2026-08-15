#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm

MODEL=DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH="models/$MODEL"
OUTPUT_ROOT=outputs/model_benchmarks/missing_leaderboard_family_scales_full_20260801
LOG_ROOT="$OUTPUT_ROOT/logs/$MODEL/opportunistic_gpu01"
STATE_ROOT="$OUTPUT_ROOT/state"
VLLM_PYTHON=/home/sm5/anaconda3/envs/VLM/bin/python

mkdir -p "$LOG_ROOT" "$STATE_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_ROOT/queue.log"
}

gpu_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v wanted="$gpu" '
      {gsub(/ /, "", $0)}
      ($1 + 0) == wanted {exit !(($2 + 0) < 2048 && ($3 + 0) < 10)}
    '
}

wait_for_gpu() {
  local gpu="$1"
  until gpu_idle "$gpu"; do sleep 20; done
}

if [[ -f "$STATE_ROOT/$MODEL.completed" ]]; then
  log "already completed"
  exit 0
fi

while [[ ! -f "$MODEL_PATH/.benchcoe_modelscope_complete.json" ]]; do
  log "waiting for completed download marker"
  sleep 60
done

wait_for_gpu 0
wait_for_gpu 1
printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$MODEL.started"
log "starting split evaluation on GPU0 and GPU1"

(
  "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
    --models-dir models --models "$MODEL" \
    --benchmarks bbh,gpqa,mmstar_text_only --gpqa-configs all \
    --gpu-devices 0 --parallel-workers 1 \
    --output-dir "$OUTPUT_ROOT/text/official" \
    > "$LOG_ROOT/gpu0_bbh_gpqa_mmstar.log" 2>&1 &&
  env CUDA_VISIBLE_DEVICES=0 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
    --dataset gaokao_2023_2024 --model-path "$MODEL_PATH" --model-name "$MODEL" \
    --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao" \
    > "$LOG_ROOT/gpu0_gaokao_2023_2024.log" 2>&1
) & pid0=$!

(
  "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
    --model-root models --models "$MODEL" \
    --validation-file MMLU-Pro/data/test-00000-of-00001.parquet \
    --gpu-id 1 --output-root "$OUTPUT_ROOT/text/mmlu_pro_test" \
    > "$LOG_ROOT/gpu1_mmlu_pro_test.log" 2>&1 &&
  env CUDA_VISIBLE_DEVICES=1 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
    --dataset gaokao_2010_2022 --model-path "$MODEL_PATH" --model-name "$MODEL" \
    --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao" \
    > "$LOG_ROOT/gpu1_gaokao_2010_2022.log" 2>&1
) & pid1=$!

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1

if [[ "$failed" -eq 0 ]]; then
  rm -f "$STATE_ROOT/$MODEL.failed"
  printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$MODEL.completed"
  log "all evaluations completed"
else
  printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$MODEL.failed"
  log "one or more evaluations failed"
fi

exit "$failed"
