#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
GPU_RUNNER=scripts/run_with_front4_gpu_claim.sh
C3_CONFIG=configs/innovation/c3_development_v8.yaml
BASELINE_CONFIG=configs/innovation/c3_pre_pair_baseline_v8.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v8_20260815
SMOKE_QUESTIONS=6
SMOKE_ROOT="$RUN_ROOT/smoke/prepair_style_n${SMOKE_QUESTIONS}"
LOG_DIR="$SMOKE_ROOT/queue_logs"

MODELS=(
  Qwen2.5-7B-Instruct
  General-Reasoner-7B-preview
  Yi-1.5-9B-Chat
  Nemotron-H-8B-Reasoning-128K
)

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
  [[ "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
  [[ "$used" -le 1024 && "$utilization" -le 10 ]]
}

wait_for_fixed_idle_gpu() {
  local gpu=$1
  local stable_polls=0
  while true; do
    if gpu_is_idle "$gpu"; then
      stable_polls=$((stable_polls + 1))
      if [[ "$stable_polls" -eq 2 ]]; then
        return 0
      fi
    else
      stable_polls=0
    fi
    sleep 5
  done
}

wait_for_v8_smoke_gate() {
  while true; do
    local manifests=(
      "$RUN_ROOT"/smoke/checks/General-Reasoner-7B-preview_n*_gpu*/check_manifest.json
    )
    for manifest in "${manifests[@]}"; do
      [[ -f "$manifest" ]] || continue
      if jq -e \
        '(.parsed_reconstructions / .reconstructions) >= 0.90 and
         .complete_isolated_trace_pair_rate >= 0.80 and
         .one_valid_one_invalid_pair_rate >= 0.20 and
         .isolated_sealed_triple_match_rate >= 0.10 and
         .truncated_model_calls == 0 and
         .labels_read == false and
         .audit_protocol == "commitment_conditioned_proof_audit_v8" and
         .proof_obligations_required == true' \
        "$manifest" >/dev/null; then
        echo "$manifest"
        return 0
      fi
    done
    sleep 30
  done
}

run_model() {
  local model=$1
  local gpu=$2
  local log="$LOG_DIR/${model}.gpu${gpu}.log"
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    {
      date '+PRePair-style smoke start %F %T'
      echo "model=$model physical_gpu=$gpu"
    } >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_pre_pair_style \
        --c3-config "$C3_CONFIG" \
        --baseline-config "$BASELINE_CONFIG" \
        --model "$model" \
        --physical-gpu "$gpu" \
        --smoke-questions "$SMOKE_QUESTIONS" >> "$log" 2>&1; then
      date '+PRePair-style smoke finish %F %T' >> "$log"
      return 0
    fi
    date '+PRePair-style startup/run failed; returning to idle queue %F %T' >> "$log"
  done
}

gate_manifest=$(wait_for_v8_smoke_gate)
{
  date '+PRePair-style successor queue released by v8 gate %F %T'
  echo "v8_gate_manifest=$gate_manifest"
} >> "$LOG_DIR/aggregate.log"

pids=()
for index in 0 1 2 3; do
  run_model "${MODELS[$index]}" "$index" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

for model in "${MODELS[@]}"; do
  manifest="$SMOKE_ROOT/models/$model/manifest.json"
  jq -e \
    '(.parsed_pointwise_calls / .pointwise_calls) >= 0.90 and
     (.parsed_pairwise_calls / .pairwise_calls) >= 0.90 and
     (.order_consistent_pairs / .order_audited_pairs) >= 0.75 and
     .truncated_prompts == 0 and
     .labels_read == false and
     .calls_per_model_per_question == 7 and
     .actual_model_calls == (.questions * 7)' \
    "$manifest" >/dev/null
done

"$PYTHON_BIN" -m bench_coe.innovation.aggregate_pre_pair_style \
  --c3-config "$C3_CONFIG" \
  --baseline-config "$BASELINE_CONFIG" \
  --smoke-questions "$SMOKE_QUESTIONS" >> "$LOG_DIR/aggregate.log" 2>&1

jq -e \
  '.status == "bounded_label_free_pre_pair_style_aggregate_smoke" and
   .labels_read == false and
   .calls_per_question == 42 and
   .new_control_calls_per_question == 28 and
   (.prediction_sha256 | type == "string" and length == 64)' \
  "$SMOKE_ROOT/aggregate/manifest.json" >/dev/null

date '+PRePair-style successor smoke complete %F %T' >> "$LOG_DIR/aggregate.log"
