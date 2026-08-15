#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
CONFIG=configs/innovation/c3_development_v1.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v1_20260815
CERT_GENERATOR=Qwen2.5-7B-Instruct
CHECKER=General-Reasoner-7B-preview
LOG_DIR="$ROOT/$RUN_ROOT/queue_logs"
CERT_LOG="$LOG_DIR/qwen_certificate_smoke_n3.log"
CHECK_LOG="$LOG_DIR/general_check_smoke_n8.log"

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

while tmux has-session -t bfj_v2_smoke_front4 2>/dev/null; do
  sleep 30
done

certificate_path=""
for candidate_path in "$RUN_ROOT"/smoke/certificates/Qwen2.5-7B-Instruct_n3_gpu*/certificates.jsonl; do
  if [[ -f "$candidate_path" ]]; then
    certificate_path=$candidate_path
    break
  fi
done

while [[ -z "$certificate_path" ]]; do
  gpu=$(wait_for_idle_gpu)
  {
    date '+C3 certificate smoke start %F %T'
    echo "physical_gpu=$gpu"
  } >> "$CERT_LOG"
  if env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificates \
        --config "$CONFIG" \
        --generator "$CERT_GENERATOR" \
        --physical-gpu "$gpu" \
        --smoke-questions 3 >> "$CERT_LOG" 2>&1; then
    certificate_path="$RUN_ROOT/smoke/certificates/${CERT_GENERATOR}_n3_gpu${gpu}/certificates.jsonl"
    date '+C3 certificate smoke finish %F %T' >> "$CERT_LOG"
    break
  fi
  date '+C3 certificate smoke attempt failed; returning to idle queue %F %T' >> "$CERT_LOG"
  sleep 60
done

certificate_manifest=$(dirname "$certificate_path")/certificate_manifest.json
jq -e \
  '(.parsed_certificates / .certificates) >= 0.90 and .truncated_prompts == 0 and .labels_read == false' \
  "$certificate_manifest" >/dev/null

while true; do
  gpu=$(wait_for_idle_gpu)
  {
    date '+C3 check smoke start %F %T'
    echo "physical_gpu=$gpu"
  } >> "$CHECK_LOG"
  if env \
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
        --smoke-certificates 8 >> "$CHECK_LOG" 2>&1; then
    check_manifest="$RUN_ROOT/smoke/checks/${CHECKER}_n8_gpu${gpu}/check_manifest.json"
    jq -e \
      '(.parsed_checks / .certificates_checked) >= 0.90 and .truncated_prompts == 0 and .labels_read == false' \
      "$check_manifest" >/dev/null
    date '+C3 check smoke finish %F %T' >> "$CHECK_LOG"
    exit 0
  fi
  date '+C3 check smoke attempt failed; returning to idle queue %F %T' >> "$CHECK_LOG"
  sleep 60
done
