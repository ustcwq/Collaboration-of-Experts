#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

config="${1:-configs/innovation/domain_guarded_meta_validation_to_test_v2.yaml}"
output_root="${2:-$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "$config")}" 
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite domain-guarded run root: $output_root" >&2
  exit 2
fi

# The domain guard consumes authenticated cached predictions and label-free topology only.
PYTHONHASHSEED=20260814 python -m bench_coe.innovation.run_domain_guarded_meta \
  --config "$config" \
  --output-dir "$output_root"

