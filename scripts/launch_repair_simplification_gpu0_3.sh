#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

config="${1:-configs/innovation/repair_simplification_source_loso_gpu0_3.yaml}"
output_root="${2:-outputs/bench_coe/innovation/repair_simplification/source_loso_gpu0_3_v1_20260814}"
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite scoring-simplification run root: $output_root" >&2
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
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits >"$output_root/gpu_preflight.csv"

seeds=(20260814 20260815 20260816 20260817)
pids=()
for gpu in 0 1 2 3; do
  seed="${seeds[$gpu]}"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
    python -m bench_coe.innovation.run_repair_simplification \
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
if (( status != 0 )); then
  exit "$status"
fi

python -m bench_coe.innovation.aggregate_repair_simplification \
  --config "$config" \
  --run-root "$output_root" \
  --output-dir "$output_root/aggregate"
