#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="outputs/bench_coe/mmlu_pro_validation_single_models"
LOG_ROOT="outputs/bench_coe/clean_mmlu_pro_validation_logs"
RESULT_ROOT="outputs/bench_coe/clean_mmlu_pro_validation_to_test"
PYTHON_BIN="${PYTHON_BIN:-/home/sm5/anaconda3/envs/VLM/bin/python}"
mkdir -p "$LOG_ROOT"

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

while true; do
  done_count=0
  for model in "${MODELS[@]}"; do
    if model_done "$model"; then
      done_count=$((done_count + 1))
    fi
  done
  echo "$(date '+%F %T') validation cache progress: ${done_count}/${#MODELS[@]}"
  if [[ "$done_count" -eq "${#MODELS[@]}" ]]; then
    break
  fi
  sleep 120
done

echo "$(date '+%F %T') running clean MMLU-Pro validation-to-test"
"$PYTHON_BIN" bench_coe/clean_mmlu_pro_innovation_experiments.py \
  --output-dir "$RESULT_ROOT" \
  2>&1 | tee "$LOG_ROOT/clean_mmlu_pro_validation_to_test.log"

echo "$(date '+%F %T') clean MMLU-Pro validation-to-test finished"
