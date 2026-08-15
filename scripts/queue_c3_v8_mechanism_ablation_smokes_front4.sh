#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
GPU_RUNNER=scripts/run_with_front4_gpu_claim.sh
NO_PRECOMMIT_CONFIG=configs/innovation/c3_v8_no_checker_precommit_ablation.yaml
PAIR_VISIBLE_CONFIG=configs/innovation/c3_v8_pair_visible_ablation.yaml
CANDIDATE_VISIBLE_CONFIG=configs/innovation/c3_v8_candidate_visible_commit_first_control.yaml
UNSEALED_CONFIG=configs/innovation/c3_v8_unsealed_set_aware_control.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v8_20260815
NO_PRECOMMIT_ROOT="$RUN_ROOT/mechanism_ablations/no_checker_private_precommitment"
PAIR_VISIBLE_ROOT="$RUN_ROOT/mechanism_ablations/pair_visible_with_precommitment"
CANDIDATE_VISIBLE_ROOT="$RUN_ROOT/mechanism_ablations/candidate_visible_commit_first"
UNSEALED_ROOT="$RUN_ROOT/mechanism_ablations/unsealed_set_aware"
PREPAIR_SMOKE_ROOT="$RUN_ROOT/smoke/prepair_style_n6"
CHECKER=General-Reasoner-7B-preview
LOG_DIR="$RUN_ROOT/mechanism_ablations/smoke_queue_logs"

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

wait_for_pre_pair_smoke() {
  while true; do
    local manifest="$PREPAIR_SMOKE_ROOT/aggregate/manifest.json"
    if [[ -f "$manifest" ]] && jq -e \
      '.status == "bounded_label_free_pre_pair_style_aggregate_smoke" and
       .labels_read == false and
       .calls_per_question == 42' \
      "$manifest" >/dev/null; then
      return 0
    fi
    sleep 30
  done
}

run_ablation() {
  local config=$1
  local name=$2
  local gpu=$3
  local certificate_path=$4
  local smoke_certificates=$5
  local log="$LOG_DIR/${name}.gpu${gpu}.log"
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    {
      date '+mechanism ablation smoke start %F %T'
      echo "name=$name physical_gpu=$gpu"
    } >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificate_checks \
        --config "$config" \
        --checker "$CHECKER" \
        --physical-gpu "$gpu" \
        --certificate-path "$certificate_path" \
        --smoke-certificates "$smoke_certificates" >> "$log" 2>&1; then
      date '+mechanism ablation smoke finish %F %T' >> "$log"
      return 0
    fi
    date '+mechanism ablation smoke failed; returning to idle queue %F %T' >> "$log"
  done
}

wait_for_pre_pair_smoke
certificate_manifests=(
  "$RUN_ROOT"/smoke/certificates/Qwen2.5-7B-Instruct_n6_gpu*/certificate_manifest.json
)
[[ ${#certificate_manifests[@]} -eq 1 && -f "${certificate_manifests[0]}" ]]
certificate_manifest=${certificate_manifests[0]}
certificate_path="$(dirname "$certificate_manifest")/certificates.jsonl"
smoke_certificates=$(jq -r '.nonabstaining_witnesses' "$certificate_manifest")
[[ "$smoke_certificates" -ge 3 && "$smoke_certificates" -le 6 ]]

run_ablation \
  "$NO_PRECOMMIT_CONFIG" no_checker_private_precommitment 0 \
  "$certificate_path" "$smoke_certificates" &
pid_no_precommit=$!
run_ablation \
  "$PAIR_VISIBLE_CONFIG" pair_visible_with_precommitment 1 \
  "$certificate_path" "$smoke_certificates" &
pid_pair_visible=$!
run_ablation \
  "$CANDIDATE_VISIBLE_CONFIG" candidate_visible_commit_first 2 \
  "$certificate_path" "$smoke_certificates" &
pid_candidate_visible=$!
run_ablation \
  "$UNSEALED_CONFIG" unsealed_set_aware 3 \
  "$certificate_path" "$smoke_certificates" &
pid_unsealed=$!
wait "$pid_no_precommit"
wait "$pid_pair_visible"
wait "$pid_candidate_visible"
wait "$pid_unsealed"

no_precommit_manifest="$NO_PRECOMMIT_ROOT/smoke/checks/${CHECKER}_n${smoke_certificates}_gpu0/check_manifest.json"
pair_visible_manifest="$PAIR_VISIBLE_ROOT/smoke/checks/${CHECKER}_n${smoke_certificates}_gpu1/check_manifest.json"
candidate_visible_manifest="$CANDIDATE_VISIBLE_ROOT/smoke/checks/${CHECKER}_n${smoke_certificates}_gpu2/check_manifest.json"
unsealed_manifest="$UNSEALED_ROOT/smoke/checks/${CHECKER}_n${smoke_certificates}_gpu3/check_manifest.json"

jq -e \
  '(.parsed_reconstructions / .reconstructions) >= 0.90 and
   .complete_isolated_trace_pair_rate >= 0.80 and
   .one_valid_one_invalid_pair_rate >= 0.20 and
   .isolated_sealed_triple_match_rate >= 0.10 and
   .truncated_model_calls == 0 and
   .labels_read == false and
   .audit_protocol == "isolated_trace_pointwise_v7" and
   .private_stage0_responses_read == false and
   .proof_obligations_required == false and
   .mechanism_ablation.name == "no_checker_private_precommitment"' \
  "$no_precommit_manifest" >/dev/null

jq -e \
  '(.parsed_reconstructions / .reconstructions) >= 0.90 and
   .position_invariant_pair_rate >= 0.50 and
   .sealed_triple_audit_rate >= 0.20 and
   .paired_sealed_triple_match_rate >= 0.10 and
   .truncated_model_calls == 0 and
   .labels_read == false and
   .audit_protocol == "commitment_conditioned_pair_audit_v8_ablation" and
   .private_stage0_responses_read == true and
   .proof_obligations_required == true and
   .mechanism_ablation.name == "pair_visible_with_precommitment"' \
  "$pair_visible_manifest" >/dev/null

for control_manifest in "$candidate_visible_manifest" "$unsealed_manifest"; do
  jq -e \
    '(.parsed_reconstructions / .reconstructions) >= 0.90 and
     .complete_isolated_trace_pair_rate >= 0.80 and
     .one_valid_one_invalid_pair_rate >= 0.20 and
     .isolated_sealed_triple_match_rate >= 0.10 and
     .truncated_model_calls == 0 and
     .labels_read == false and
     .target_was_hidden == false and
     .private_stage0_responses_read == true and
     .proof_obligations_required == true' \
    "$control_manifest" >/dev/null
done

jq -e \
  '.sealed_claim_was_hidden == true and
   .audit_protocol == "candidate_visible_commit_first_v8_control" and
   .mechanism_ablation.name == "candidate_visible_commit_first"' \
  "$candidate_visible_manifest" >/dev/null

jq -e \
  '.sealed_claim_was_hidden == false and
   .audit_protocol == "unsealed_set_aware_v8_control" and
   .mechanism_ablation.name == "unsealed_set_aware"' \
  "$unsealed_manifest" >/dev/null

date '+all four prompt-level mechanism/control smoke gates passed %F %T' \
  >> "$LOG_DIR/stages.log"
