#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

STATE=outputs/model_benchmarks/autonomous_remaining_full_20260802/state
mkdir -p "$STATE"

models=(gemma-3-12b-it Qwen3-VL-8B-Thinking)
benchmarks=(cmmmu mmmu mmmu_pro mathvista gaokao_mm)

while :; do
  for model in "${models[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
      key="${model}__${benchmark}"
      [[ -f "$STATE/$key.completed" ]] || printf '2\n' >"$STATE/$key.attempt"
    done
  done
  BENCHCOE_GPUS="4 5 6 7" \
  BENCHCOE_MODELS="${models[*]}" \
  BENCHCOE_MAX_ATTEMPTS=4 \
  BENCHCOE_FINAL_PREFIX=back4_scheduler \
  BENCHCOE_SCHEDULER_LOG=outputs/model_benchmarks/autonomous_remaining_full_20260802/logs/back4_scheduler.log \
    bash scripts/run_autonomous_remaining_vision_tasks.sh
  failed=$(cat "$STATE/back4_scheduler_final_failed_count" 2>/dev/null || echo 1)
  [[ "$failed" -eq 0 ]] && break
  find "$STATE" -type f \( -name 'gemma-3-12b-it__*.failed*' -o -name 'Qwen3-VL-8B-Thinking__*.failed*' \) -delete
  sleep 600
done
