#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

config="${1:-configs/innovation/locked_musr_paper_v1.yaml}"
output_root="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "$config")"
receipt="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["test_receipt"])' "$config")"

if [[ ! -e "$receipt" ]]; then
  python -m bench_coe.innovation.test_receipt --config "$config" --output "$receipt"
fi
if [[ ! -e "$output_root/preregistration.json" ]]; then
  python -m bench_coe.innovation.prepare_locked_musr --config "$config"
fi
if [[ ! -e "$output_root/smoke/Qwen2.5-7B-Instruct_n2_gpu0/smoke_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m bench_coe.innovation.run_locked_musr_generation \
      --config "$config" --smoke-model Qwen2.5-7B-Instruct \
      --smoke-questions 2 --physical-gpu 0
fi
if [[ ! -e "$output_root/target_observables/observable_manifest.json" ]]; then
  python -m bench_coe.innovation.run_locked_musr_generation --config "$config"
fi
if [[ ! -e "$output_root/prediction_seal.json" ]]; then
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260815 CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m bench_coe.innovation.run_locked_musr_selection \
      --config "$config" --physical-gpu 0
fi
if [[ -e "$output_root/label_access_started.json" ]]; then
  echo "Locked MuSR labels have already been opened; refusing a second evaluation." >&2
  exit 5
fi
python -m bench_coe.innovation.evaluate_locked_musr --config "$config"
