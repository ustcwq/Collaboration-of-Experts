#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

ROOT=outputs/bench_coe/autonomous_remaining_supervisor_20260802
mkdir -p "$ROOT"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/supervisor.log"
}

while :; do
  log "starting remaining vision tasks on every idle GPU"
  bash scripts/run_autonomous_remaining_vision_tasks.sh >>"$ROOT/vision_scheduler.log" 2>&1
  failed=$(cat outputs/model_benchmarks/autonomous_remaining_full_20260802/state/final_failed_count 2>/dev/null || echo 1)
  if [[ "$failed" -eq 0 ]]; then
    log "vision scheduler finished with every task completed"
    break
  fi
  log "$failed vision tasks remain failed; cooling down before automatic retry"
  find outputs/model_benchmarks/autonomous_remaining_full_20260802/state \
    -type f \( -name '*.failed' -o -name '*.failed_final' -o -name '*.attempt' \) -delete
  sleep 600
done

while tmux has-session -t benchcoe_pending_text_eval_gpu012 2>/dev/null; do
  log "waiting for the active pending-text queue"
  sleep 60
done

while pgrep -f 'missing_leaderboard_family_scales_full_20260801.*DeepSeek-R1-Distill-Qwen-14B' >/dev/null 2>&1; do
  log "waiting for remaining DeepSeek-14B worker processes"
  sleep 60
done

bash scripts/run_ministral3_completion_gate.sh >>"$ROOT/ministral_gate.log" 2>&1

while tmux ls 2>/dev/null | grep -q 'benchcoe_scale_'; do
  log "waiting for already-running scale-transfer experiments"
  sleep 60
done

log "materializing unified prediction views and running improve5/6"
while :; do
  BENCHCOE_FORCE_COHORTS=language_2b_4b \
    bash scripts/run_scale_transfer_improve56.sh >>"$ROOT/improve56.log" 2>&1
  summary_count=$(find outputs/bench_coe/scale_transfer_improve56_20260802 \
    -mindepth 3 -maxdepth 3 -name summary.json -type f | wc -l)
  if [[ "$summary_count" -eq 8 ]]; then
    break
  fi
  log "only $summary_count/8 improve5/6 summaries exist; retrying automatically after cooldown"
  sleep 600
done

if [[ -f tools/summarize_scale_expansion_transfer.py ]]; then
  python tools/summarize_scale_expansion_transfer.py >>"$ROOT/legacy_summary_refresh.log" 2>&1 || true
fi

date '+%F %T' >"$ROOT/completed"
log "all autonomous remaining tasks finished"
