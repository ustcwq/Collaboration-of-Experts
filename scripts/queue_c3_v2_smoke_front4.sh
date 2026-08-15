#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
CONFIG=configs/innovation/c3_development_v2.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v2_20260815
CERT_GENERATOR=Qwen2.5-7B-Instruct
CHECKER=General-Reasoner-7B-preview
LOG_DIR="$ROOT/$RUN_ROOT/queue_logs"
CERT_LOG="$LOG_DIR/qwen_certificate_smoke_n4.log"
CHECK_LOG="$LOG_DIR/general_target_blind_smoke_n12.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

gpu_is_idle() {
  local gpu=$1
  local used
  local utilization
  IFS=, read -r used utilization < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used=${used// /}
  utilization=${utilization// /}
  [[ "$used" -le 1024 && "$utilization" -le 10 ]]
}

wait_for_idle_gpu() {
  while true; do
    for gpu in 0 1 2 3; do
      if ! gpu_is_idle "$gpu"; then
        continue
      fi
      sleep 20
      if ! gpu_is_idle "$gpu"; then
        continue
      fi
      sleep 20
      if gpu_is_idle "$gpu"; then
        echo "$gpu"
        return 0
      fi
    done
    sleep 30
  done
}

gpu=$(wait_for_idle_gpu)
{
  date '+C3 v2 certificate smoke start %F %T'
  echo "physical_gpu=$gpu"
} >> "$CERT_LOG"
env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONHASHSEED=20260815 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificates \
    --config "$CONFIG" \
    --generator "$CERT_GENERATOR" \
    --physical-gpu "$gpu" \
    --smoke-questions 4 >> "$CERT_LOG" 2>&1
certificate_dir="$RUN_ROOT/smoke/certificates/${CERT_GENERATOR}_n4_gpu${gpu}"
certificate_path="$certificate_dir/certificates.jsonl"
certificate_manifest="$certificate_dir/certificate_manifest.json"
jq -e \
  '(.parsed_certificates / .certificates) >= 0.90 and
   .truncated_prompts == 0 and
   .labels_read == false and
   .prompt_version == "two_sided_sealed_certificate_v2" and
   .parser_version == "anchored_certificate_fields_v2"' \
  "$certificate_manifest" >/dev/null
jq -se 'map(.dataset) | unique | length >= 2' "$certificate_path" >/dev/null
date '+C3 v2 certificate smoke finish %F %T' >> "$CERT_LOG"

gpu=$(wait_for_idle_gpu)
{
  date '+C3 v2 target-blind smoke start %F %T'
  echo "physical_gpu=$gpu"
} >> "$CHECK_LOG"
env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONHASHSEED=20260815 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificate_checks \
    --config "$CONFIG" \
    --checker "$CHECKER" \
    --physical-gpu "$gpu" \
    --certificate-path "$certificate_path" \
    --smoke-certificates 12 >> "$CHECK_LOG" 2>&1
check_dir="$RUN_ROOT/smoke/checks/${CHECKER}_n12_gpu${gpu}"
check_path="$check_dir/checks.jsonl"
check_manifest="$check_dir/check_manifest.json"
jq -e \
  '(.parsed_checks / .certificates_checked) >= 0.90 and
   .truncated_prompts == 0 and
   .labels_read == false and
   .target_was_hidden == true and
   .commitments_from_stage0 == true and
   .prompt_version == "target_blind_effect_reconstruction_v2" and
   .parser_version == "target_blind_effect_fields_v2"' \
  "$check_manifest" >/dev/null
jq -se \
  '(map(.dataset) | unique | length >= 2) and
   all(.[]; .target_was_hidden == true and .commitment_source == "stage0_base_prediction")' \
  "$check_path" >/dev/null
date '+C3 v2 target-blind smoke finish %F %T' >> "$CHECK_LOG"
