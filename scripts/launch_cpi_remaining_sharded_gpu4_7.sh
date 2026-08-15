#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

config="${1:-configs/innovation/cpi_remaining_source_loso_gpu4_7.yaml}"
output_root="${2:-outputs/bench_coe/innovation/cpi_remaining/source_loso_gpu4_7_v1_20260809}"
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite remaining-source run root: $output_root" >&2
  exit 2
fi
while IFS=, read -r index used; do
  index="${index// /}"
  used="${used// /}"
  if [[ "$index" =~ ^[4-7]$ ]] && (( used >= 100 )); then
    echo "Physical GPU $index is not idle (${used} MiB used)" >&2
    exit 3
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
mkdir -p "$output_root/logs"

seeds=(20260808 20260809 20260810 20260811)
shard_names=(a b c d)
shard_variants=(
  "int_none int_permutation int_random_dropout int_leave_expert_out int_leave_family_out"
  "int_missing_output int_exact_clone int_pseudo_clone int_known_swap int_full"
  "factor_legacy_dro factor_mask_mean factor_mask_dro factor_rich_mean"
  "factor_rich_dro factor_rich_mask_mean factor_rich_mask_dro"
)
pids=()
for gpu in 4 5 6 7; do
  seed="${seeds[$((gpu - 4))]}"
  for shard_index in 0 1 2 3; do
    shard="${shard_names[$shard_index]}"
    args=()
    for variant in ${shard_variants[$shard_index]}; do
      args+=(--variant "$variant")
    done
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
      python -m bench_coe.innovation.run_cpi_remaining \
        --config "$config" \
        --seed "$seed" \
        --physical-gpu "$gpu" \
        --output-dir "$output_root/shards/seed_${seed}_gpu${gpu}/shard_${shard}" \
        "${args[@]}" \
        >"$output_root/logs/seed_${seed}_gpu${gpu}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
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

for gpu in 4 5 6 7; do
  seed="${seeds[$((gpu - 4))]}"
  python -m bench_coe.innovation.merge_cpi_remaining_shards \
    --config "$config" \
    --seed "$seed" \
    --physical-gpu "$gpu" \
    --shard-root "$output_root/shards/seed_${seed}_gpu${gpu}" \
    --output-dir "$output_root/seed_${seed}_gpu${gpu}"
done
