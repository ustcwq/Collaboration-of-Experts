#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

MODEL=Ministral-3-3B-Instruct-2512
ROOT=outputs/model_benchmarks/family_scale_expansion_full_20260731
LOG_ROOT="$ROOT/logs/transformers_retry/$MODEL"
STATUS_ROOT=outputs/bench_coe/autonomous_remaining_supervisor_20260802
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS_ROOT/ministral_gate.log"
}

wait_for_sessions() {
  while tmux ls 2>/dev/null | grep -q 'benchcoe_ministral3_'; do
    sleep 60
  done
}

idle_gpu() {
  while :; do
    local row gpu memory utilization
    while IFS=, read -r gpu memory utilization; do
      gpu=${gpu// /}
      memory=${memory//[^0-9]/}
      utilization=${utilization//[^0-9]/}
      if [[ -n "$gpu" && -n "$memory" && -n "$utilization" && "$memory" -lt 2048 && "$utilization" -lt 10 ]]; then
        printf '%s\n' "$gpu"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    sleep 60
  done
}

launch_validation() {
  local gpu=$1
  tmux new-session -d -s benchcoe_ministral3_mmlu_validation_retry_gpu${gpu} \
    "cd /home/sm5/ys/FCS && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BENCH_COE_TRANSFORMERS_BATCH_SIZE=4 /home/sm5/anaconda3/envs/Factory/bin/python -m bench_coe.evaluate_mmlu_pro_validation_models --backend transformers --attn-implementation eager --model-root models --models $MODEL --validation-file MMLU-Pro/data/validation-00000-of-00001.parquet --gpu-id $gpu --output-root outputs/model_benchmarks/improve56_scale_sources_20260801/text/mmlu_validation --max-new-tokens 512 --overwrite > outputs/model_benchmarks/improve56_scale_sources_20260801/logs/${MODEL}_retry_gpu${gpu}.log 2>&1 && date '+%F %T' > $STATUS_ROOT/ministral_validation_stop_refresh_complete"
}

launch_gaokao() {
  local dataset=$1 gpu=$2
  tmux new-session -d -s benchcoe_ministral3_${dataset}_retry_gpu${gpu} \
    "cd /home/sm5/ys/FCS && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BENCH_COE_TRANSFORMERS_BATCH_SIZE=8 /home/sm5/anaconda3/envs/Factory/bin/python -m bench_coe.run_gaokao_text_smoke --backend transformers --gpu-id $gpu --attn-implementation eager --dataset $dataset --model-path models/$MODEL --model-name $MODEL --max-examples-per-task 0 --output-dir $ROOT/text/gaokao > $LOG_ROOT/${dataset}_retry_gpu${gpu}.log 2>&1"
}

while :; do
  wait_for_sessions
  if python tools/audit_ministral3_results.py --status-file "$STATUS_ROOT/ministral_status.json" \
      >>"$STATUS_ROOT/ministral_gate.log" 2>&1; then
    date '+%F %T' >"$STATUS_ROOT/ministral_complete"
    log "all Ministral base tests and validation sources are complete"
    exit 0
  fi

  missing=$(/home/sm5/anaconda3/envs/Factory/bin/python - <<'PY'
import json
from pathlib import Path
print(' '.join(json.loads(Path('outputs/bench_coe/autonomous_remaining_supervisor_20260802/ministral_status.json').read_text())['missing']))
PY
)
  log "incomplete Ministral results detected: $missing"

  if [[ " $missing " == *" mmlu_test "* ]]; then
    log "relaunching incomplete MMLU-Pro test shards"
    bash scripts/launch_ministral3_mmlu_shards.sh
    continue
  fi
  if [[ " $missing " == *" mmlu_validation "* ]]; then
    gpu=$(idle_gpu)
    log "relaunching MMLU-Pro validation source on GPU$gpu"
    launch_validation "$gpu"
    continue
  fi
  for dataset in gaokao_2010_2022 gaokao_2023_2024; do
    if [[ " $missing " == *" $dataset "* ]]; then
      gpu=$(idle_gpu)
      log "relaunching $dataset on GPU$gpu"
      launch_gaokao "$dataset" "$gpu"
      break
    fi
  done
  if [[ " $missing " == *" bbh "* || " $missing " == *" gpqa "* || " $missing " == *" mmstar_text_only "* ]]; then
    log "relaunching incomplete official text benchmarks"
    bash scripts/launch_ministral3_official_transformers.sh
    continue
  fi
done
