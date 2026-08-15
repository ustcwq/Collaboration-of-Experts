#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

MODEL=Ministral-3-3B-Instruct-2512
OUTPUT=outputs/model_benchmarks/family_scale_expansion_full_20260731/text/mmlu_pro_test
LOG_ROOT=outputs/model_benchmarks/family_scale_expansion_full_20260731/logs/transformers_retry/$MODEL
mkdir -p "$LOG_ROOT"

while pgrep -f 'missing_leaderboard_family_scales_full_20260801.*DeepSeek-R1-Distill-Qwen-14B' >/dev/null 2>&1; do
  sleep 60
done

while :; do
  read -r memory utilization < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 1 |
      tr -d ' ' | tr ',' ' '
  )
  if [[ "$memory" -lt 2048 && "$utilization" -lt 10 ]]; then
    break
  fi
  sleep 60
done

tmux kill-session -t benchcoe_ministral3_mmlu_shard3_gpu4 2>/dev/null || true

launch() {
  local session=$1 gpu=$2 categories=$3 suffix=$4 log=$5
  tmux new-session -d -s "$session" \
    "cd /home/sm5/ys/FCS && while :; do env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BENCH_COE_TRANSFORMERS_BATCH_SIZE=8 /home/sm5/anaconda3/envs/Factory/bin/python -m bench_coe.evaluate_mmlu_pro_validation_models --backend transformers --attn-implementation eager --model-root models --models $MODEL --validation-file MMLU-Pro/data/test-00000-of-00001.parquet --gpu-id $gpu --output-root $OUTPUT --categories '$categories' --summary-suffix $suffix --max-new-tokens 512 --overwrite > '$LOG_ROOT/$log' 2>&1 && break; sleep 600; done"
}

launch benchcoe_ministral3_mmlu_shard3a_gpu4 4 biology .shard3a mmlu_shard3a_gpu4.log
launch benchcoe_ministral3_mmlu_shard3b_gpu1 1 law .shard3b mmlu_shard3b_gpu1.log
date '+%F %T' >outputs/bench_coe/autonomous_remaining_supervisor_20260802/ministral_gpu1_split_launched
