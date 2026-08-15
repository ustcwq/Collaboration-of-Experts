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
OUTPUT_ROOT=outputs/model_benchmarks/scale_extension_existing_gap_full_20260731
LOG_ROOT="$OUTPUT_ROOT/logs"
STATE_ROOT="$OUTPUT_ROOT/state"
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
    log "GPU $gpu busy; waiting"
    sleep 60
  done
}

run_logged() {
  local log_file="$1"
  shift
  "$@" > "$log_file" 2>&1
}

run_gpu0_text() {
  local gpu=0
  local model=Qwen3-4B-Instruct-2507
  wait_for_gpu "$gpu"
  log "GPU $gpu: $model full BBH/GPQA/MMStar"
  run_logged "$LOG_ROOT/gpu0_${model}_official.log" \
    "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
      --models-dir models \
      --models "$model" \
      --benchmarks bbh,gpqa,mmstar_text_only \
      --gpqa-configs all \
      --gpu-devices "$gpu" \
      --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/text/official" || return 1

  wait_for_gpu "$gpu"
  log "GPU $gpu: $model full MMLU-Pro test"
  run_logged "$LOG_ROOT/gpu0_${model}_mmlu_pro_test.log" \
    "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
      --model-root models \
      --models "$model" \
      --validation-file MMLU-Pro/data/test-00000-of-00001.parquet \
      --gpu-id "$gpu" \
      --output-root "$OUTPUT_ROOT/text/mmlu_pro_test"
}

run_vlm_benchmark() {
  local gpu="$1"
  local model="$2"
  local benchmark="$3"
  wait_for_gpu "$gpu"
  log "GPU $gpu: $model full $benchmark"
  run_logged "$LOG_ROOT/gpu${gpu}_${model}_${benchmark}.log" \
    "$VLM_PYTHON" -m bench_coe.run_multimodal_babyvision_models \
      --models-dir models_v \
      --models "$model" \
      --benchmarks "$benchmark" \
      --gpu-devices "$gpu" \
      --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/vision/$benchmark"
}

run_gpu2_queue() {
  local gpu=2
  local text_model=Qwen3-4B-Instruct-2507
  local vision_model=Qwen3-VL-8B-Instruct
  wait_for_gpu "$gpu"
  log "GPU $gpu: $text_model full GAOKAO-Bench-2010-2022"
  run_logged "$LOG_ROOT/gpu2_${text_model}_gaokao_2010_2022.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2010_2022 \
      --model-path "models/$text_model" \
      --model-name "$text_model" \
      --max-examples-per-task 0 \
      --output-dir "$OUTPUT_ROOT/text/gaokao" || return 1

  run_vlm_benchmark "$gpu" "$vision_model" cmmmu || return 1
  run_vlm_benchmark "$gpu" "$vision_model" mmmu_pro
}

run_gpu3_queue() {
  local gpu=3
  local text_model=Qwen3-4B-Instruct-2507
  local vision_model=Qwen3-VL-8B-Instruct
  wait_for_gpu "$gpu"
  log "GPU $gpu: $text_model full GAOKAO-Bench-2023-2024"
  run_logged "$LOG_ROOT/gpu3_${text_model}_gaokao_2023_2024.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2023_2024 \
      --model-path "models/$text_model" \
      --model-name "$text_model" \
      --max-examples-per-task 0 \
      --output-dir "$OUTPUT_ROOT/text/gaokao" || return 1

  run_vlm_benchmark "$gpu" "$vision_model" mmmu || return 1
  run_vlm_benchmark "$gpu" "$vision_model" mathvista || return 1

  wait_for_gpu "$gpu"
  log "GPU $gpu: $vision_model full GAOKAO-MM"
  run_logged "$LOG_ROOT/gpu3_${vision_model}_gaokao_mm.log" \
    "$VLM_PYTHON" -m bench_coe.run_gaokao_mm_babyvision_models \
      --models-dir models_v \
      --models "$vision_model" \
      --gpu-devices "$gpu" \
      --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/vision/gaokao_mm"
}

log "existing 4B language and 8B vision full-evaluation queue started"
run_gpu0_text & pid0=$!
run_gpu2_queue & pid2=$!
run_gpu3_queue & pid3=$!

failed=0
for pid in "$pid0" "$pid2" "$pid3"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -eq 0 ]]; then
  printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/all.completed"
  log "existing-scale full-evaluation queue completed"
else
  printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/all.failed"
  log "existing-scale full-evaluation queue finished with failures"
fi
exit "$failed"
