#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

MODEL=Ministral-3-3B-Instruct-2512
OUTPUT=outputs/model_benchmarks/family_scale_expansion_full_20260731/text/mmlu_pro_test
LOG_ROOT=outputs/model_benchmarks/family_scale_expansion_full_20260731/logs/transformers_retry/$MODEL
mkdir -p "$LOG_ROOT"

while read -r stale_session; do
  [[ -n "$stale_session" ]] && tmux kill-session -t "$stale_session" 2>/dev/null || true
done < <(tmux list-sessions -F '#S' 2>/dev/null | grep '^benchcoe_ministral3_mmlu_shard')

gpus=(0 2 3 4 5 6 7)
groups=(
  'math,history'
  'physics,computer science'
  'chemistry,philosophy'
  'law,biology'
  'engineering,business'
  'other,psychology'
  'economics,health'
)

for index in "${!gpus[@]}"; do
  gpu=${gpus[$index]}
  categories=${groups[$index]}
  session="benchcoe_ministral3_mmlu_shard${index}_gpu${gpu}"
  log="$LOG_ROOT/mmlu_shard${index}_gpu${gpu}.log"
  tmux new-session -d -s "$session" "cd /home/sm5/ys/FCS && while :; do env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BENCH_COE_TRANSFORMERS_BATCH_SIZE=4 /home/sm5/anaconda3/envs/Factory/bin/python -m bench_coe.evaluate_mmlu_pro_validation_models --backend transformers --attn-implementation eager --model-root models --models $MODEL --validation-file MMLU-Pro/data/test-00000-of-00001.parquet --gpu-id $gpu --output-root $OUTPUT --categories '$categories' --summary-suffix .shard$index --max-new-tokens 512 --overwrite > '$log' 2>&1 && break; sleep 600; done"
done

tmux has-session -t benchcoe_ministral3_mmlu_finalize 2>/dev/null && tmux kill-session -t benchcoe_ministral3_mmlu_finalize || true
tmux new-session -d -s benchcoe_ministral3_mmlu_finalize "cd /home/sm5/ys/FCS && while tmux ls 2>/dev/null | grep -q 'benchcoe_ministral3_mmlu_shard'; do sleep 60; done; python tools/finalize_mmlu_model_results.py --output-root $OUTPUT --model $MODEL --expected 12032 > '$LOG_ROOT/mmlu_finalize.log' 2>&1"
