#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="outputs/bench_coe/mmlu_pro_validation_single_models"
LOG_ROOT="outputs/bench_coe/clean_mmlu_pro_validation_logs"
PYTHON_BIN="${PYTHON_BIN:-/home/sm5/anaconda3/envs/VLM/bin/python}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

MODELS=(
  Baichuan2-7B-Chat
  DeepSeek-R1-0528-Qwen3-8B
  General-Reasoner-7B-preview
  Llama-3.1-8B-Instruct
  MAmmoTH2-8B-Plus
  Ministral-8B-Instruct-2410
  Nemotron-H-8B-Reasoning-128K
  Qwen2.5-7B-Instruct
  Yi-1.5-9B-Chat
  Yi-9B
  aya-expanse-8b
  gemma-2-9b-it
  glm-4-9b-chat
  granite-3.3-8b-instruct
  internlm3-8b-instruct
)

GPUS=(0 1 2 3)
MAX_UTIL="${MAX_UTIL:-45}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
GPU_UTIL="${GPU_UTIL:-0.45}"

model_done() {
  local model="$1"
  "$PYTHON_BIN" - "$OUT_ROOT" "$model" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
model = sys.argv[2]
result_dir = root / model / "CoT" / "validation"
if not result_dir.is_dir():
    raise SystemExit(1)
total = 0
for path in result_dir.glob("*.json"):
    total += len(json.loads(path.read_text(encoding="utf-8")))
raise SystemExit(0 if total == 70 else 1)
PY
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local line used total util free
    line="$(nvidia-smi --id="$gpu" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r used total util <<<"$line"
    used="${used// /}"
    total="${total// /}"
    util="${util// /}"
    free=$((total - used))
    if [[ "$util" -le "$MAX_UTIL" && "$free" -ge "$MIN_FREE_MB" ]]; then
      echo "gpu=$gpu ready util=$util free_mb=$free"
      return 0
    fi
    echo "gpu=$gpu busy util=$util free_mb=$free; waiting ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
  done
}

worker() {
  local worker_idx="$1"
  local gpu="$2"
  local gpu_count="${#GPUS[@]}"
  local i
  for ((i=worker_idx; i<${#MODELS[@]}; i+=gpu_count)); do
    local model="${MODELS[$i]}"
    if model_done "$model"; then
      echo "[$model] validation cache exists, skip"
      continue
    fi
    wait_for_gpu "$gpu"
    echo "[$model] start on gpu=$gpu"
    if PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench_coe/evaluate_mmlu_pro_validation_models.py \
      --models "$model" \
      --gpu-id "$gpu" \
      --gpu-util "$GPU_UTIL" \
      --output-root "$OUT_ROOT" \
      2>&1 | tee "$LOG_ROOT/${model}.log"; then
      echo "[$model] finished"
    else
      echo "[$model] failed; continuing with next assigned model"
    fi
  done
}

if [[ "${1:-}" == "worker" ]]; then
  worker "$2" "$3"
  exit 0
fi

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  session="clean_mmlu_val_g${gpu}"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"
  tmux new-session -d -s "$session" \
    "PYTHON_BIN='$PYTHON_BIN' GPU_UTIL='$GPU_UTIL' MAX_UTIL='$MAX_UTIL' MIN_FREE_MB='$MIN_FREE_MB' SLEEP_SECONDS='$SLEEP_SECONDS' bash '$ROOT/scripts/launch_clean_mmlu_pro_validation_front4.sh' worker '$idx' '$gpu'"
  echo "started tmux session $session"
done

echo "logs: $LOG_ROOT"
