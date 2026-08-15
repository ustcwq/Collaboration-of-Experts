#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

LOG=outputs/bench_coe/autonomous_remaining_supervisor_20260802/back4_coordinator.log
supervisor_pid=$(pgrep -fo 'bash scripts/run_autonomous_remaining_supervisor.sh' || true)
if [[ -z "$supervisor_pid" ]]; then
  printf '%s supervisor not found\n' "$(date '+%F %T')" >>"$LOG"
  exit 1
fi

kill -STOP "$supervisor_pid"
printf '%s paused supervisor pid=%s while back4 runs\n' "$(date '+%F %T')" "$supervisor_pid" >>"$LOG"

while tmux has-session -t benchcoe_autonomous_back4_vision 2>/dev/null; do
  sleep 60
done

kill -CONT "$supervisor_pid" 2>/dev/null || true
printf '%s resumed supervisor pid=%s after back4 completion\n' "$(date '+%F %T')" "$supervisor_pid" >>"$LOG"
