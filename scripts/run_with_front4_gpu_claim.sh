#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )) || [[ "$2" != "--" ]]; then
  echo "usage: $0 GPU_ID -- COMMAND [ARG ...]" >&2
  exit 2
fi

gpu=$1
shift 2
if [[ ! "$gpu" =~ ^[0-3]$ ]]; then
  echo "GPU_ID must be a physical GPU between 0 and 3" >&2
  exit 2
fi

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=${C3_PYTHON_BIN:-/home/sm5/anaconda3/envs/FactoryS/bin/python}
claim_dir=
claim_pid=

cleanup() {
  if [[ -n "$claim_pid" ]] && kill -0 "$claim_pid" 2>/dev/null; then
    kill -TERM "$claim_pid" 2>/dev/null || true
    wait "$claim_pid" 2>/dev/null || true
  fi
  if [[ -n "$claim_dir" ]]; then
    rm -f "$claim_dir/ready.json"
    rmdir "$claim_dir" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

gpu_is_idle() {
  local used
  local utilization
  IFS=, read -r used utilization < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used=${used// /}
  utilization=${utilization// /}
  [[ "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
  [[ "$used" -le 1024 && "$utilization" -le 10 ]]
}

if ! gpu_is_idle; then
  echo "GPU claim deferred: physical GPU${gpu} is no longer idle" >&2
  exit 75
fi

claim_dir=$(mktemp -d "/tmp/c3_front4_gpu${gpu}.XXXXXX")
ready_file="$claim_dir/ready.json"
CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
  "$ROOT/scripts/hold_front4_gpu_claim.py" \
  --physical-gpu "$gpu" \
  --parent-pid "$$" \
  --ready-file "$ready_file" &
claim_pid=$!

for _ in $(seq 1 30); do
  if [[ -s "$ready_file" ]]; then
    break
  fi
  if ! kill -0 "$claim_pid" 2>/dev/null; then
    wait "$claim_pid" || true
    echo "GPU claim deferred: claim process failed during initialization" >&2
    exit 75
  fi
  sleep 1
done
if [[ ! -s "$ready_file" ]]; then
  echo "GPU claim deferred: CUDA context was not ready within 30 seconds" >&2
  exit 75
fi

process_rows=
if ! process_rows=$(
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits
); then
  echo "GPU claim deferred: failed to verify compute processes on GPU${gpu}" >&2
  exit 75
fi
unexpected_pids=()
claim_seen=0
while IFS= read -r process_pid; do
  process_pid=${process_pid// /}
  [[ -n "$process_pid" ]] || continue
  if [[ "$process_pid" == "$claim_pid" ]]; then
    claim_seen=1
  else
    unexpected_pids+=("$process_pid")
  fi
done <<< "$process_rows"
if [[ "$claim_seen" -ne 1 ]]; then
  echo "GPU claim deferred: claim PID ${claim_pid} was not visible on GPU${gpu}" >&2
  exit 75
fi
if (( ${#unexpected_pids[@]} > 0 )); then
  echo "GPU claim deferred: concurrent process appeared on GPU${gpu}: ${unexpected_pids[*]}" >&2
  exit 75
fi
printf 'GPU claim acquired: %s\n' "$(tr -d '\n' < "$ready_file")" >&2

set +e
"$@"
return_code=$?
set -e
exit "$return_code"
