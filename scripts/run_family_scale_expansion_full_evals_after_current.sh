#!/usr/bin/env bash
set -euo pipefail

cd /home/sm5/ys/FCS

OUTPUT_ROOT=outputs/model_benchmarks/family_scale_expansion_full_20260731
mkdir -p "$OUTPUT_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUTPUT_ROOT/wait.log"
}

while tmux has-session -t benchcoe_scale_extension_eval_front4 2>/dev/null \
   || tmux has-session -t benchcoe_existing_scale_gap_eval 2>/dev/null; do
  log "waiting for current scale-extension evaluation sessions"
  sleep 120
done

log "starting tested-family other-scale full evaluations"
BENCHCOE_EVAL_OUTPUT_ROOT="$OUTPUT_ROOT" \
BENCHCOE_MODEL_QUEUE_FILE=configs/family_scale_expansion_eval_queue_20260731.txt \
BENCHCOE_ALLOW_EXISTING_MODELS=1 \
bash scripts/watch_downloads_and_run_scale_extension_full_evals.sh
