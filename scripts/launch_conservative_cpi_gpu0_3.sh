#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

config="${1:-configs/innovation/cpi_conservative_source_loso.yaml}"
output_root="${2:-outputs/bench_coe/innovation/cpi_conservative/source_loso_v1_20260809}"
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite Conservative-CPI run root: $output_root" >&2
  exit 2
fi
while IFS=, read -r index used; do
  index="${index// /}"
  used="${used// /}"
  if [[ "$index" =~ ^[0-3]$ ]] && (( used >= 100 )); then
    echo "Physical GPU $index is not idle (${used} MiB used)" >&2
    exit 3
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
mkdir -p "$output_root/logs"

seeds=(20260808 20260809 20260810 20260811)
pids=()
for gpu in 0 1 2 3; do
  seed="${seeds[$gpu]}"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
    python -m bench_coe.innovation.run_conservative_cpi \
      --config "$config" \
      --seed "$seed" \
      --physical-gpu "$gpu" \
      --output-dir "$output_root/seed_${seed}_gpu${gpu}" \
      >"$output_root/logs/seed_${seed}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
