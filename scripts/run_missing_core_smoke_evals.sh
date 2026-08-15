#!/usr/bin/env bash
set -euo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

VLLM_PYTHON=/home/sm5/anaconda3/envs/VLM/bin/python
export BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm

OUTPUT_ROOT=outputs/model_benchmarks/core_missing_smoke_20260731
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_ROOT/queue.log"
}

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /, "", $0); if (($1 + 0) < 4 && ($2 + 0) < 2048 && ($3 + 0) < 10) print $1}'
}

wait_for_gpus() {
  local needed="$1"
  while true; do
    mapfile -t available < <(idle_gpus)
    if (( ${#available[@]} >= needed )); then
      local joined
      joined=$(IFS=,; echo "${available[*]:0:needed}")
      printf '%s\n' "$joined"
      return
    fi
    log "waiting for $needed idle GPU(s); currently ${#available[@]}" >&2
    sleep 60
  done
}

run_existing_models() {
  local gpu_csv
  gpu_csv=$(wait_for_gpus 2)
  local text_gpu=${gpu_csv%%,*}
  local vision_gpu=${gpu_csv##*,}
  log "testing existing Qwen2.5-Coder-7B-Instruct on GPU $text_gpu"
  (
    "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
      --models-dir models_v \
      --models Qwen2.5-Coder-7B-Instruct \
      --benchmarks bbh,gpqa,mmstar_text_only \
      --gpqa-configs diamond \
      --max-examples 8 \
      --gpu-devices "$text_gpu" \
      --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/text_existing"
    CUDA_VISIBLE_DEVICES="$text_gpu" "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2010_2022 \
      --model-path models_v/Qwen2.5-Coder-7B-Instruct \
      --output-dir "$OUTPUT_ROOT/gaokao_text_existing"
    CUDA_VISIBLE_DEVICES="$text_gpu" "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2023_2024 \
      --model-path models_v/Qwen2.5-Coder-7B-Instruct \
      --output-dir "$OUTPUT_ROOT/gaokao_text_existing"
  ) > "$LOG_ROOT/text_existing.log" 2>&1 &
  local text_pid=$!

  log "testing existing Qwen2.5-VL-3B-Instruct on GPU $vision_gpu"
  (
    python -m bench_coe.run_multimodal_babyvision_models \
      --models-dir models_v \
      --models Qwen2.5-VL-3B-Instruct \
      --benchmarks cmmmu,mmmu,mmmu_pro,mathvista \
      --max-examples-per-benchmark 8 \
      --gpu-devices "$vision_gpu" \
      --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/vision_existing"
    python -m bench_coe.run_gaokao_mm_babyvision_models \
      --models-dir models_v \
      --models Qwen2.5-VL-3B-Instruct \
      --gpu-devices "$vision_gpu" \
      --parallel-workers 1 \
      --limit-per-task 8 \
      --output-dir "$OUTPUT_ROOT/gaokao_mm_existing"
  ) > "$LOG_ROOT/vision_existing.log" 2>&1 &
  local vision_pid=$!

  wait "$text_pid"
  wait "$vision_pid"

  gpu_csv=$(wait_for_gpus 1)
  log "testing existing coder on MMLU-Pro validation using GPU $gpu_csv"
  "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
    --model-root models_v \
    --models Qwen2.5-Coder-7B-Instruct \
    --gpu-id "$gpu_csv" \
    --output-root "$OUTPUT_ROOT/mmlu_pro_existing" \
    > "$LOG_ROOT/mmlu_pro_existing.log" 2>&1
}

wait_for_downloaded_models() {
  local models=(
    qwen3_1_7b qwen3_4b qwen3_14b qwen25_math_7b
    internvl35_4b internvl35_8b internvl35_14b qwen25_vl_7b
  )
  while true; do
    local missing=()
    local model
    for model in "${models[@]}"; do
      [[ -f "benchcoe_assets/models/$model/.benchcoe_ready.json" ]] || missing+=("$model")
    done
    if (( ${#missing[@]} == 0 )); then
      return
    fi
    if ! pgrep -f 'tools/modelscope_assets.py download --profile core' >/dev/null; then
      log "model download stopped before completion; missing: ${missing[*]}"
      return 1
    fi
    log "waiting for model downloads: ${missing[*]}"
    sleep 120
  done
}

run_downloaded_models() {
  local gpu_csv
  gpu_csv=$(wait_for_gpus 4)
  log "testing downloaded text models on BBH, GPQA-Diamond, and MMStar text-only"
  "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
    --models-dir benchcoe_assets/models \
    --models qwen3_1_7b qwen3_4b qwen3_14b qwen25_math_7b \
    --benchmarks bbh,gpqa,mmstar_text_only \
    --gpqa-configs diamond \
    --max-examples 8 \
    --gpu-devices "$gpu_csv" \
    --parallel-workers 4 \
    --output-dir "$OUTPUT_ROOT/text_downloaded" \
    > "$LOG_ROOT/text_downloaded.log" 2>&1

  gpu_csv=$(wait_for_gpus 1)
  log "testing downloaded text models on MMLU-Pro validation"
  "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
    --model-root benchcoe_assets/models \
    --models qwen3_1_7b,qwen3_4b,qwen3_14b,qwen25_math_7b \
    --gpu-id "$gpu_csv" \
    --output-root "$OUTPUT_ROOT/mmlu_pro_downloaded" \
    > "$LOG_ROOT/mmlu_pro_downloaded.log" 2>&1

  gpu_csv=$(wait_for_gpus 4)
  IFS=, read -r -a gaokao_gpus <<< "$gpu_csv"
  local text_models=(qwen3_1_7b qwen3_4b qwen3_14b qwen25_math_7b)
  local gaokao_pids=()
  local index
  log "testing downloaded text models on both GAOKAO-Bench datasets"
  for index in "${!text_models[@]}"; do
    (
      CUDA_VISIBLE_DEVICES="${gaokao_gpus[$index]}" "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
        --dataset gaokao_2010_2022 \
        --model-path "benchcoe_assets/models/${text_models[$index]}" \
        --output-dir "$OUTPUT_ROOT/gaokao_text_downloaded"
      CUDA_VISIBLE_DEVICES="${gaokao_gpus[$index]}" "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
        --dataset gaokao_2023_2024 \
        --model-path "benchcoe_assets/models/${text_models[$index]}" \
        --output-dir "$OUTPUT_ROOT/gaokao_text_downloaded"
    ) > "$LOG_ROOT/gaokao_${text_models[$index]}.log" 2>&1 &
    gaokao_pids+=("$!")
  done
  for index in "${gaokao_pids[@]}"; do wait "$index"; done

  gpu_csv=$(wait_for_gpus 4)
  log "testing downloaded vision models on CMMMU, MMMU, MMMU-Pro, and MathVista"
  python -m bench_coe.run_multimodal_babyvision_models \
    --models-dir benchcoe_assets/models \
    --models internvl35_4b internvl35_8b internvl35_14b qwen25_vl_7b \
    --benchmarks cmmmu,mmmu,mmmu_pro,mathvista \
    --max-examples-per-benchmark 8 \
    --gpu-devices "$gpu_csv" \
    --parallel-workers 4 \
    --output-dir "$OUTPUT_ROOT/vision_downloaded" \
    > "$LOG_ROOT/vision_downloaded.log" 2>&1

  gpu_csv=$(wait_for_gpus 4)
  log "testing downloaded vision models on GAOKAO-MM"
  python -m bench_coe.run_gaokao_mm_babyvision_models \
    --models-dir benchcoe_assets/models \
    --models internvl35_4b internvl35_8b internvl35_14b qwen25_vl_7b \
    --gpu-devices "$gpu_csv" \
    --parallel-workers 4 \
    --limit-per-task 8 \
    --output-dir "$OUTPUT_ROOT/gaokao_mm_downloaded" \
    > "$LOG_ROOT/gaokao_mm_downloaded.log" 2>&1
}

log "queue started"
run_existing_models
wait_for_downloaded_models
run_downloaded_models
log "queue completed"
