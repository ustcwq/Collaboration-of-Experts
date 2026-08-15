#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
CONFIG=configs/innovation/blind_falsification_jury_dev_v2.yaml
AUDITOR=Qwen2.5-7B-Instruct
RUN_ROOT=outputs/bench_coe/innovation/blind_falsification_jury/dev_v2_20260815
LOG_DIR="$ROOT/$RUN_ROOT/queue_logs"
LOG_PATH="$LOG_DIR/qwen_v2_smoke_n3.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

gpu_is_idle() {
  local gpu=$1
  local used
  local utilization
  IFS=, read -r used utilization < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used=${used// /}
  utilization=${utilization// /}
  [[ "$used" -le 1024 && "$utilization" -le 10 ]]
}

while true; do
  for gpu in 0 1 2 3; do
    if ! gpu_is_idle "$gpu"; then
      continue
    fi
    sleep 20
    if ! gpu_is_idle "$gpu"; then
      continue
    fi
    sleep 20
    if ! gpu_is_idle "$gpu"; then
      continue
    fi
    {
      date '+BFJ smoke start %F %T'
      echo "physical_gpu=$gpu"
    } >> "$LOG_PATH"
    if env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONHASHSEED=20260815 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        "$PYTHON_BIN" -m bench_coe.innovation.run_bfj_audits \
          --config "$CONFIG" \
          --auditor "$AUDITOR" \
          --physical-gpu "$gpu" \
          --smoke-questions 3 >> "$LOG_PATH" 2>&1; then
      date '+BFJ smoke finish %F %T' >> "$LOG_PATH"
      exit 0
    fi
    date '+BFJ smoke attempt failed; returning to idle queue %F %T' >> "$LOG_PATH"
    sleep 60
    break
  done
  sleep 30
done
