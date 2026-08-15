#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PYTHON=/home/sm5/anaconda3/envs/Factory/bin/python
MODEL=Llama-3.1-Nemotron-Nano-VL-8B-V1
OUTPUT_ROOT=outputs/model_benchmarks/family_scale_expansion_full_20260731
LOG_ROOT=$OUTPUT_ROOT/logs/$MODEL/opportunistic
mkdir -p "$LOG_ROOT"

gpu_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v wanted="$gpu" '{gsub(/ /,"",$0)} ($1+0)==wanted {exit !(($2+0)<2048 && ($3+0)<10)}'
}

wait_gpu() {
  until gpu_idle "$1"; do sleep 20; done
}

run_benchmark() {
  local gpu="$1" benchmark="$2"
  wait_gpu "$gpu"
  printf '%s GPU %s: %s %s\n' "$(date '+%F %T')" "$gpu" "$MODEL" "$benchmark" | tee -a "$LOG_ROOT/queue.log"
  "$PYTHON" -m bench_coe.run_multimodal_babyvision_models \
    --models-dir models_v --models "$MODEL" --benchmarks "$benchmark" \
    --gpu-devices "$gpu" --parallel-workers 1 \
    --output-dir "$OUTPUT_ROOT/vision/$benchmark" \
    > "$LOG_ROOT/gpu${gpu}_${benchmark}.log" 2>&1
}

run_gpu1() {
  run_benchmark 1 cmmmu && run_benchmark 1 mmmu_pro
}

run_gpu2() {
  run_benchmark 2 mmmu && run_benchmark 2 mathvista
  wait_gpu 2
  "$PYTHON" -m bench_coe.run_gaokao_mm_babyvision_models \
    --models-dir models_v --models "$MODEL" --gpu-devices 2 --parallel-workers 1 \
    --output-dir "$OUTPUT_ROOT/vision/gaokao_mm" \
    > "$LOG_ROOT/gpu2_gaokao_mm.log" 2>&1
}

run_gpu1 & pid1=$!
run_gpu2 & pid2=$!
failed=0
wait "$pid1" || failed=1
wait "$pid2" || failed=1
exit "$failed"
