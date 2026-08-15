#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm

OUTPUT_ROOT=outputs/model_benchmarks/missing_leaderboard_family_scales_full_20260801
STATE_ROOT="$OUTPUT_ROOT/state"
VLLM_PYTHON=/home/sm5/anaconda3/envs/VLM/bin/python
MODELS=(
  DeepSeek-R1-Distill-Qwen-1.5B
  DeepSeek-R1-Distill-Qwen-14B
  Yi-1.5-6B-Chat
  Mistral-Nemo-Instruct-2407
)

mkdir -p "$STATE_ROOT"

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

run_model() {
  local model="$1"
  local model_path="models/$model"
  local log_root="$OUTPUT_ROOT/logs/$model/opportunistic_gpu01"
  mkdir -p "$log_root"

  wait_for_gpu 0
  wait_for_gpu 1
  wait_for_gpu 2
  printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$model.started"
  printf '%s %s\n' "$(date '+%F %T')" "starting $model on GPU0, GPU1, and GPU2" | tee -a "$log_root/queue.log"

  (
    "$VLLM_PYTHON" -m bench_coe.run_official_model_benchmarks \
      --models-dir models --models "$model" \
      --benchmarks bbh,gpqa,mmstar_text_only --gpqa-configs all \
      --gpu-devices 0 --parallel-workers 1 \
      --output-dir "$OUTPUT_ROOT/text/official" \
      > "$log_root/gpu0_bbh_gpqa_mmstar.log" 2>&1
  ) & local pid0=$!

  (
    "$VLLM_PYTHON" -m bench_coe.evaluate_mmlu_pro_validation_models \
      --model-root models --models "$model" \
      --validation-file MMLU-Pro/data/test-00000-of-00001.parquet \
      --gpu-id 1 --output-root "$OUTPUT_ROOT/text/mmlu_pro_test" \
      > "$log_root/gpu1_mmlu_pro_test.log" 2>&1
  ) & local pid1=$!

  (
    env CUDA_VISIBLE_DEVICES=2 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2010_2022 --model-path "$model_path" --model-name "$model" \
      --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao" \
      > "$log_root/gpu2_gaokao_2010_2022.log" 2>&1 &&
    env CUDA_VISIBLE_DEVICES=2 "$VLLM_PYTHON" -m bench_coe.run_gaokao_text_smoke \
      --dataset gaokao_2023_2024 --model-path "$model_path" --model-name "$model" \
      --max-examples-per-task 0 --output-dir "$OUTPUT_ROOT/text/gaokao" \
      > "$log_root/gpu2_gaokao_2023_2024.log" 2>&1
  ) & local pid2=$!

  local failed=0
  wait "$pid0" || failed=1
  wait "$pid1" || failed=1
  wait "$pid2" || failed=1
  if [[ "$failed" -eq 0 ]]; then
    rm -f "$STATE_ROOT/$model.failed"
    printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$model.completed"
    printf '%s %s\n' "$(date '+%F %T')" "$model completed" | tee -a "$log_root/queue.log"
  else
    printf '%s\n' "$(date '+%F %T')" > "$STATE_ROOT/$model.failed"
    printf '%s %s\n' "$(date '+%F %T')" "$model failed; continuing queue" | tee -a "$log_root/queue.log"
  fi
}

while :; do
  remaining=0
  started=0
  for model in "${MODELS[@]}"; do
    [[ -f "$STATE_ROOT/$model.completed" ]] && continue
    remaining=$((remaining + 1))
    if [[ -f "models/$model/.benchcoe_modelscope_complete.json" ]]; then
      run_model "$model"
      started=1
      break
    fi
  done
  [[ "$remaining" -eq 0 ]] && exit 0
  [[ "$started" -eq 0 ]] && sleep 60
done
