#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
GPU_RUNNER=scripts/run_with_front4_gpu_claim.sh
CONFIG=configs/innovation/c3_development_v8.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v8_20260815
CERT_GENERATOR=Qwen2.5-7B-Instruct
CHECKER=General-Reasoner-7B-preview
SMOKE_WITNESSES=6
LOG_DIR="$ROOT/$RUN_ROOT/queue_logs"
CERT_LOG="$LOG_DIR/qwen_commit_then_permute_smoke_n${SMOKE_WITNESSES}.log"
CHECK_LOG="$LOG_DIR/general_commitment_proof_smoke_n${SMOKE_WITNESSES}.log"

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

wait_for_idle_gpu() {
  while true; do
    for gpu in 0 1 2 3; do
      if ! gpu_is_idle "$gpu"; then
        continue
      fi
      local stable_polls=1
      while [[ "$stable_polls" -lt 2 ]]; do
        sleep 5
        if gpu_is_idle "$gpu"; then
          stable_polls=$((stable_polls + 1))
        else
          break
        fi
      done
      if [[ "$stable_polls" -eq 2 ]]; then
        echo "$gpu"
        return 0
      fi
    done
    sleep 5
  done
}

while true; do
  gpu=$(wait_for_idle_gpu)
  {
    date '+C3 v8 commit-then-permute smoke start %F %T'
    echo "physical_gpu=$gpu"
  } >> "$CERT_LOG"
  if "$GPU_RUNNER" "$gpu" -- env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONHASHSEED=20260815 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificates \
      --config "$CONFIG" \
      --generator "$CERT_GENERATOR" \
      --physical-gpu "$gpu" \
      --smoke-questions "$SMOKE_WITNESSES" >> "$CERT_LOG" 2>&1; then
    break
  fi
  date '+C3 v8 certificate startup/run failed; returning to idle queue %F %T' >> "$CERT_LOG"
done
certificate_dir="$RUN_ROOT/smoke/certificates/${CERT_GENERATOR}_n${SMOKE_WITNESSES}_gpu${gpu}"
certificate_path="$certificate_dir/certificates.jsonl"
certificate_manifest="$certificate_dir/certificate_manifest.json"
jq -e \
  '(.parsed_witnesses / .witnesses) >= 0.90 and
   (.nonabstaining_witnesses / .witnesses) >= 0.50 and
   .truncated_model_calls == 0 and
   .all_option_effect_rate <= 0.0 and
   .required_valid_trace_counts["1"] > 0 and
   .required_valid_trace_counts["2"] > 0 and
   .labels_read == false and
   .claims_are_sealed_from_checkers == true and
   .counterfactual_pairs == true and
   .private_stage0_responses_read == true and
   .post_commit_permutation == true and
   (.base_prediction_sha256 | type == "string") and
   .prompt_version == "committed_counterfactual_permutation_v6" and
   .parser_version == "committed_counterfactual_challenge_fields_v6"' \
  "$certificate_manifest" >/dev/null
jq -se \
  '(map(.dataset) | unique | length >= 2) and
   (group_by(.witness_id) | length == 6) and
   all(group_by(.witness_id)[];
     (map(.raw_output) | unique | length == 1) and
     (map(.prompt_sha256) | unique | length == 1) and
     (map(.required_valid_trace) | unique | length == 1) and
     (map(.author_valid_trace) | unique | length == 1) and
     (map(.post_commit_permutation_applied) | unique | length == 1) and
     all(.[];
       .claim_was_sealed == true and
       .counterfactual_pair == true and
       ((.claimed_eliminated_options + .claimed_supported_options) | unique | length) <= 1))' \
  "$certificate_path" >/dev/null
date '+C3 v8 commit-then-permute smoke finish %F %T' >> "$CERT_LOG"

smoke_check_witnesses=$(jq -r '.nonabstaining_witnesses' "$certificate_manifest")
while true; do
  gpu=$(wait_for_idle_gpu)
  {
    date '+C3 v8 commitment proof smoke start %F %T'
    echo "physical_gpu=$gpu"
    echo "nonabstaining_witnesses=$smoke_check_witnesses"
  } >> "$CHECK_LOG"
  if "$GPU_RUNNER" "$gpu" -- env \
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
      --smoke-certificates "$smoke_check_witnesses" >> "$CHECK_LOG" 2>&1; then
    break
  fi
  date '+C3 v8 commitment proof startup/run failed; returning to idle queue %F %T' >> "$CHECK_LOG"
done
check_dir="$RUN_ROOT/smoke/checks/${CHECKER}_n${smoke_check_witnesses}_gpu${gpu}"
check_path="$check_dir/checks.jsonl"
check_manifest="$check_dir/check_manifest.json"
jq -e \
  '(.parsed_reconstructions / .reconstructions) >= 0.90 and
   .complete_isolated_trace_pair_rate >= 0.80 and
   .one_valid_one_invalid_pair_rate >= 0.20 and
   .isolated_sealed_triple_match_rate >= 0.10 and
   .truncated_model_calls == 0 and
   .labels_read == false and
   .target_was_hidden == true and
   .sealed_claim_was_hidden == true and
   .counterfactual_pairs == true and
   .audit_protocol == "commitment_conditioned_proof_audit_v8" and
   .isolated_trace_views == ["trace_1", "trace_2"] and
   .parity_orientations == null and
   .commitments_from_stage0 == true and
   .private_stage0_responses_read == true and
   .proof_obligations_required == true and
   (.pair_combiner_sha256 | type == "string") and
   .prompt_version == "commitment_conditioned_proof_audit_v8" and
   .parser_version == "proof_obligation_audit_fields_v8"' \
  "$check_manifest" >/dev/null
jq -se \
  '(map(.dataset) | unique | length >= 2) and
   all(.[];
     .target_was_hidden == true and
     .sealed_claim_was_hidden == true and
     .counterfactual_pair == true and
     .audit_protocol == "commitment_conditioned_proof_audit_v8" and
     .trace_under_audit == .orientation and
     .commitment_source == "stage0_base_prediction" and
     (.raw_prompt_sha256 | type == "string" and length == 64) and
     (.countertest | type == "string" and length > 0) and
     (.countertest_result | IN("SURVIVES", "BREAKS", "UNCERTAIN")) and
     (.recomputation | type == "string" and length > 0) and
     (.commitment_relation | IN("CONSISTENT", "CONFLICTS", "UNRELATED", "UNCERTAIN"))) and
   all(group_by(.witness_id)[];
     (map(.orientation) | unique | sort) == ["trace_1", "trace_2"]) and
   all(group_by(.witness_id + "::" + .orientation)[];
     (map(.raw_output) | unique | length == 1) and
     (map(.prompt_sha256) | unique | length == 1))' \
  "$check_path" >/dev/null
date '+C3 v8 commitment proof smoke finish %F %T' >> "$CHECK_LOG"
