#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm

VLLM_PYTHON=/home/sm5/anaconda3/envs/VLM/bin/python
MODEL=Qwen2.5-3B-Instruct
MODEL_PATH=models/$MODEL
OUTPUT_ROOT=outputs/model_benchmarks/scale_extension_full_20260731
LOG_ROOT=$OUTPUT_ROOT/logs/$MODEL/opportunistic
STATE_ROOT=$OUTPUT_ROOT/state/opportunistic_$MODEL
mkdir -p "$LOG_ROOT" "$STATE_ROOT"

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
    sleep 20
  done
}

run_task() {
  local gpu="$1"
  local task="$2"
  shift 2
  local marker="$STATE_ROOT/$task.completed"
  [[ -f "$marker" ]] && return
  wait_for_gpu "$gpu"
  log "GPU $gpu: starting $task"
  if "$@" > "$LOG_ROOT/gpu${gpu}_${task}.log" 2>&1; then
    date '+%F %T' > "$marker"
    log "GPU $gpu: completed $task"
  else
    date '+%F %T' > "$STATE_ROOT/$task.failed"
    log "GPU $gpu: failed $task"
    return 1
  fi
}

if [[ ! -f "$MODEL_PATH/.benchcoe_modelscope_complete.json" ]]; then
  log "waiting for completed model $MODEL"
  while [[ ! -f "$MODEL_PATH/.benchcoe_modelscope_complete.json" ]]; do sleep 30; done
fi

log "opportunistic front4 queue started for $MODEL"

run_task 2 official_bbh_gpqa_mmstar \
  "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
    --models-dir models --models "$MODEL" \
    --benchmarks bbh,gpqa,mmstar_text_only --gpqa-configs all \
    --gpu-devices 2 --parallel-workers 1 \
    --output-dir "$OUTPUT_ROOT/text/official" & pid2=$!

run_task 3 gaokao_2010_2022 \
  env CUDA_VISIBLE_DEVICES=3 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
    --dataset gaokao_2010_2022 --model-path "$MODEL_PATH" --model-name "$MODEL" \
    --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao" & pid3=$!

run_task 0 gaokao_2023_2024 \
  env CUDA_VISIBLE_DEVICES=0 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
    --dataset gaokao_2023_2024 --model-path "$MODEL_PATH" --model-name "$MODEL" \
    --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao" & pid0=$!

run_task 1 mmlu_pro_test \
  "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
    --model-root models --models "$MODEL" \
    --validation-file MMLU-Pro/data/test-00000-of-00001.parquet \
    --gpu-id 1 --output-root "$OUTPUT_ROOT/text/mmlu_pro_test" & pid1=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then failed=1; fi
done

if [[ "$failed" -eq 0 ]]; then
  date '+%F %T' > "$STATE_ROOT/all.completed"
  log "opportunistic front4 queue completed for $MODEL"
else
  log "opportunistic front4 queue finished with failures for $MODEL"
fi
exit "$failed"
