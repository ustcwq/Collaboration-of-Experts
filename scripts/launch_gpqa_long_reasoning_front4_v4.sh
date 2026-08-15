#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

CONFIG=configs/innovation/gpqa_long_reasoning_generation_v4b.yaml
ROOT=outputs/bench_coe/innovation/gpqa_long_reasoning/v4b_20260814
SHARD_ROOT="$ROOT/shards"
LOG_ROOT="$ROOT/logs"
VLLM_PYTHON=/home/sm5/anaconda3/envs/VLM/bin/python

mkdir -p "$SHARD_ROOT" "$LOG_ROOT"

gpu_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v wanted="$gpu" '
      {gsub(/ /, "", $0)}
      ($1 + 0) == wanted {exit !(($2 + 0) < 2048 && ($3 + 0) < 10)}
    '
}

run_shard() {
  local gpu="$1"
  local shard="$2"
  while ! gpu_idle "$gpu"; do
    printf '%s shard=%s waiting_for_gpu=%s\n' "$(date '+%F %T')" "$shard" "$gpu" >> "$LOG_ROOT/queue.log"
    sleep 20
  done
  printf '%s shard=%s starting_gpu=%s\n' "$(date '+%F %T')" "$shard" "$gpu" >> "$LOG_ROOT/queue.log"
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    BENCH_COE_VLLM_SOURCE=/home/sm5/ys/Project/vllm \
    "$VLLM_PYTHON" -m bench_coe.innovation.run_gpqa_label_free_inference \
      --config "$CONFIG" \
      --output-dir "$SHARD_ROOT/shard_$shard" \
      --gpu-id "$gpu" \
      --shard-index "$shard" \
      --shard-count 4 \
      > "$LOG_ROOT/shard_$shard.log" 2>&1
}

pids=()
for shard in 0 1 2 3; do
  run_shard "$shard" "$shard" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

PYTHONPATH=. "$VLLM_PYTHON" -m bench_coe.innovation.merge_gpqa_long_reasoning_shards \
  --config "$CONFIG" \
  --shard-root "$SHARD_ROOT" \
  --output-dir "$ROOT/merged" \
  > "$LOG_ROOT/merge.log" 2>&1
