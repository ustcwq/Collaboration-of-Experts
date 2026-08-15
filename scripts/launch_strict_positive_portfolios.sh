#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

config="${1:-configs/innovation/strict_positive_dataset_portfolios_v1.yaml}"
output_root="${2:-$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "$config")}" 
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite strict-positive portfolio run root: $output_root" >&2
  exit 2
fi

# This stage authenticates and recombines cached predictions; no GPU inference is required.
PYTHONHASHSEED=20260814 python -m bench_coe.innovation.run_strict_positive_portfolio \
  --config "$config" \
  --output-dir "$output_root"
