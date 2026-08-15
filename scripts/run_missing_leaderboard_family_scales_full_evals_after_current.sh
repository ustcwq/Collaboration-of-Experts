#!/usr/bin/env bash
set -euo pipefail

cd /home/sm5/ys/FCS

OUTPUT_ROOT=outputs/model_benchmarks/missing_leaderboard_family_scales_full_20260801
QUEUE_FILE=configs/missing_leaderboard_family_scale_eval_queue_20260801.txt
WAIT_SESSIONS=(
  benchcoe_scale_extension_eval_front4
  benchcoe_family_scale_eval
  benchcoe_front4_ready_models
)

mkdir -p "$OUTPUT_ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUTPUT_ROOT/orchestrator.log"
}

active_evaluation_session() {
  local session
  for session in "${WAIT_SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

log "missing-family single-model evaluation gate started"
while active_evaluation_session; do
  log "waiting for existing single-model evaluation sessions"
  sleep 120
done

log "starting missing-family full single-model evaluations"
BENCHCOE_EVAL_OUTPUT_ROOT="$OUTPUT_ROOT" \
BENCHCOE_MODEL_QUEUE_FILE="$QUEUE_FILE" \
BENCHCOE_ALLOW_EXISTING_MODELS=0 \
bash scripts/watch_downloads_and_run_scale_extension_full_evals.sh
log "missing-family full single-model evaluations completed"
