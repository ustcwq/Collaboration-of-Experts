#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
GPU_RUNNER=scripts/run_with_front4_gpu_claim.sh
C3_CONFIG=configs/innovation/c3_development_v8.yaml
PREPAIR_CONFIG=configs/innovation/c3_pre_pair_baseline_v8.yaml
EQUAL_CONFIG=configs/innovation/c3_equal_call_baselines_v8.yaml
CFMAD_CONFIG=configs/innovation/c3_cfmad_style_v8.yaml
NO_PRECOMMIT_CONFIG=configs/innovation/c3_v8_no_checker_precommit_ablation.yaml
PAIR_VISIBLE_CONFIG=configs/innovation/c3_v8_pair_visible_ablation.yaml
CANDIDATE_VISIBLE_CONFIG=configs/innovation/c3_v8_candidate_visible_commit_first_control.yaml
UNSEALED_CONFIG=configs/innovation/c3_v8_unsealed_set_aware_control.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v8_20260815
NO_PRECOMMIT_ROOT="$RUN_ROOT/mechanism_ablations/no_checker_private_precommitment"
PAIR_VISIBLE_ROOT="$RUN_ROOT/mechanism_ablations/pair_visible_with_precommitment"
CANDIDATE_VISIBLE_ROOT="$RUN_ROOT/mechanism_ablations/candidate_visible_commit_first"
UNSEALED_ROOT="$RUN_ROOT/mechanism_ablations/unsealed_set_aware"
SMOKE_QUESTIONS=6
SMOKE_ROOT="$RUN_ROOT/smoke/prepair_style_n${SMOKE_QUESTIONS}"
CFMAD_SMOKE_ROOT="$RUN_ROOT/smoke/cfmad_style_n2"
LOG_DIR="$RUN_ROOT/full_queue_logs"
RECEIPT="$RUN_ROOT/prereg/test_receipt_pre_full_with_cfmad.json"

MODELS=(
  Qwen2.5-7B-Instruct
  General-Reasoner-7B-preview
  Yi-1.5-9B-Chat
  Nemotron-H-8B-Reasoning-128K
)

mkdir -p "$LOG_DIR" "$RUN_ROOT/prereg"
cd "$ROOT"

failure_marker="$LOG_DIR/FAILED"
trap 'status=$?; date "+full queue failed %F %T exit=$status" >> "$failure_marker"' ERR

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

wait_for_all_smoke_gates() {
  while true; do
    local aggregate="$SMOKE_ROOT/aggregate/manifest.json"
    local cfmad_aggregate="$CFMAD_SMOKE_ROOT/aggregate/manifest.json"
    if [[ -f "$aggregate" ]] && jq -e \
      '.status == "bounded_label_free_pre_pair_style_aggregate_smoke" and
       .labels_read == false and
       .calls_per_question == 42 and
       .new_control_calls_per_question == 28' \
      "$aggregate" >/dev/null && [[ -f "$cfmad_aggregate" ]] && jq -e \
      '.status == "bounded_label_free_cfmad_style_aggregate_smoke" and
       .labels_read == false and
       .calls_per_model_per_question == 10 and
       .ensemble_calls_per_question == 40 and
       .questions == 2' \
      "$cfmad_aggregate" >/dev/null; then
      local no_precommit=(
        "$NO_PRECOMMIT_ROOT"/smoke/checks/General-Reasoner-7B-preview_n*_gpu0/check_manifest.json
      )
      local pair_visible=(
        "$PAIR_VISIBLE_ROOT"/smoke/checks/General-Reasoner-7B-preview_n*_gpu1/check_manifest.json
      )
      local candidate_visible=(
        "$CANDIDATE_VISIBLE_ROOT"/smoke/checks/General-Reasoner-7B-preview_n*_gpu2/check_manifest.json
      )
      local unsealed=(
        "$UNSEALED_ROOT"/smoke/checks/General-Reasoner-7B-preview_n*_gpu3/check_manifest.json
      )
      if [[ -f "${no_precommit[0]}" && -f "${pair_visible[0]}" && \
            -f "${candidate_visible[0]}" && -f "${unsealed[0]}" ]] && jq -e \
        '.audit_protocol == "isolated_trace_pointwise_v7" and
         .one_valid_one_invalid_pair_rate >= 0.20 and
         .isolated_sealed_triple_match_rate >= 0.10 and
         .mechanism_ablation.name == "no_checker_private_precommitment"' \
        "${no_precommit[0]}" >/dev/null && jq -e \
        '.audit_protocol == "commitment_conditioned_pair_audit_v8_ablation" and
         .position_invariant_pair_rate >= 0.50 and
         .paired_sealed_triple_match_rate >= 0.10 and
         .mechanism_ablation.name == "pair_visible_with_precommitment"' \
        "${pair_visible[0]}" >/dev/null && jq -e \
        '.audit_protocol == "candidate_visible_commit_first_v8_control" and
         .target_was_hidden == false and
         .sealed_claim_was_hidden == true and
         .mechanism_ablation.name == "candidate_visible_commit_first"' \
        "${candidate_visible[0]}" >/dev/null && jq -e \
        '.audit_protocol == "unsealed_set_aware_v8_control" and
         .target_was_hidden == false and
         .sealed_claim_was_hidden == false and
         .mechanism_ablation.name == "unsealed_set_aware"' \
        "${unsealed[0]}" >/dev/null; then
        return 0
      fi
    fi
    sleep 30
  done
}

run_certificate_model() {
  local model=$1
  local gpu=$2
  local manifest="$RUN_ROOT/certificates/$model/certificate_manifest.json"
  local log="$LOG_DIR/certificate.${model}.gpu${gpu}.log"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    date '+full certificate start %F %T' >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificates \
        --config "$C3_CONFIG" \
        --generator "$model" \
        --physical-gpu "$gpu" >> "$log" 2>&1; then
      date '+full certificate finish %F %T' >> "$log"
      return 0
    fi
    date '+full certificate startup/run failed; retrying after idle %F %T' >> "$log"
  done
}

run_checker_model() {
  local model=$1
  local gpu=$2
  local config=${3:-$C3_CONFIG}
  local output_root=${4:-$RUN_ROOT}
  local tag=${5:-main}
  local manifest="$output_root/checks/$model/check_manifest.json"
  local log="$LOG_DIR/check.${tag}.${model}.gpu${gpu}.log"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    date '+full checker start %F %T' >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_c3_certificate_checks \
        --config "$config" \
        --checker "$model" \
        --physical-gpu "$gpu" >> "$log" 2>&1; then
      date '+full checker finish %F %T' >> "$log"
      return 0
    fi
    date '+full checker startup/run failed; retrying after idle %F %T' >> "$log"
  done
}

run_pre_pair_model() {
  local model=$1
  local gpu=$2
  local manifest="$RUN_ROOT/prepair_style/models/$model/manifest.json"
  local log="$LOG_DIR/prepair.${model}.gpu${gpu}.log"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    date '+full PRePair-style start %F %T' >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_pre_pair_style \
        --c3-config "$C3_CONFIG" \
        --baseline-config "$PREPAIR_CONFIG" \
        --model "$model" \
        --physical-gpu "$gpu" >> "$log" 2>&1; then
      date '+full PRePair-style finish %F %T' >> "$log"
      return 0
    fi
    date '+full PRePair-style startup/run failed; retrying after idle %F %T' >> "$log"
  done
}

run_equal_call_model() {
  local method=$1
  local model=$2
  local gpu=$3
  local manifest="$RUN_ROOT/equal_call_single_model/$method/$model/manifest.json"
  local log="$LOG_DIR/equal.${method}.${model}.gpu${gpu}.log"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    date '+full equal-call start %F %T' >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_equal_call_single_model \
        --c3-config "$C3_CONFIG" \
        --baseline-config "$EQUAL_CONFIG" \
        --model "$model" \
        --method "$method" \
        --physical-gpu "$gpu" >> "$log" 2>&1; then
      date '+full equal-call finish %F %T' >> "$log"
      return 0
    fi
    date '+full equal-call startup/run failed; retrying after idle %F %T' >> "$log"
  done
}

run_cfmad_model() {
  local model=$1
  local gpu=$2
  local manifest="$RUN_ROOT/cfmad_style/models/$model/manifest.json"
  local log="$LOG_DIR/cfmad.${model}.gpu${gpu}.log"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    date '+full CFMAD-style start %F %T' >> "$log"
    if "$GPU_RUNNER" "$gpu" -- env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED=20260815 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      "$PYTHON_BIN" -m bench_coe.innovation.run_cfmad_style \
        --c3-config "$C3_CONFIG" \
        --baseline-config "$CFMAD_CONFIG" \
        --model "$model" \
        --physical-gpu "$gpu" >> "$log" 2>&1; then
      date '+full CFMAD-style finish %F %T' >> "$log"
      return 0
    fi
    date '+full CFMAD-style startup/run failed; retrying after idle %F %T' >> "$log"
  done
}

wait_parallel_wave() {
  local pids=("$@")
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_isolated_control_wave() {
  local config=$1
  local output_root=$2
  local tag=$3
  local audit_protocol=$4
  local sealed_claim_hidden=$5
  local pids=()
  local index
  for index in 0 1 2 3; do
    run_checker_model \
      "${MODELS[$index]}" "$index" "$config" "$output_root" "$tag" &
    pids+=("$!")
  done
  wait_parallel_wave "${pids[@]}"
  local model
  for model in "${MODELS[@]}"; do
    jq -e \
      --arg protocol "$audit_protocol" \
      --argjson sealed "$sealed_claim_hidden" \
      --arg name "$tag" \
      '.status == "completed_label_free_c3_checks" and
       (.parsed_reconstructions / .reconstructions) >= 0.90 and
       .complete_isolated_trace_pair_rate >= 0.80 and
       .one_valid_one_invalid_pair_rate >= 0.20 and
       .isolated_sealed_triple_match_rate >= 0.10 and
       .truncated_model_calls == 0 and
       .labels_read == false and
       .target_was_hidden == false and
       .sealed_claim_was_hidden == $sealed and
       .audit_protocol == $protocol and
       .private_stage0_responses_read == true and
       .proof_obligations_required == true and
       .mechanism_ablation.name == $name' \
      "$output_root/checks/$model/check_manifest.json" >/dev/null
  done
  date "+all $tag control gates passed %F %T" >> "$LOG_DIR/stages.log"
}

wait_for_all_smoke_gates
date '+all bounded smoke gates passed %F %T' >> "$LOG_DIR/stages.log"

if [[ ! -f "$RECEIPT" ]]; then
  "$PYTHON_BIN" -m bench_coe.innovation.test_receipt \
    --output "$RECEIPT" \
    --config "$C3_CONFIG" \
    --config "$PREPAIR_CONFIG" \
    --config "$EQUAL_CONFIG" \
    --config "$CFMAD_CONFIG" \
    --config "$NO_PRECOMMIT_CONFIG" \
    --config "$PAIR_VISIBLE_CONFIG" \
    --config "$CANDIDATE_VISIBLE_CONFIG" \
    --config "$UNSEALED_CONFIG" >> "$LOG_DIR/test_receipt.log" 2>&1
fi
jq -e '.exit_code == 0 and .test_count >= 189' "$RECEIPT" >/dev/null

pids=()
for index in 0 1 2 3; do
  run_certificate_model "${MODELS[$index]}" "$index" &
  pids+=("$!")
done
wait_parallel_wave "${pids[@]}"
for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "completed_label_free_c3_certificates" and
     (.parsed_witnesses / .witnesses) >= 0.90 and
     (.nonabstaining_witnesses / .witnesses) >= 0.50 and
     .truncated_model_calls == 0 and
     .all_option_effect_rate == 0 and
     .labels_read == false and
     .counterfactual_pairs == true and
     .private_stage0_responses_read == true and
     .post_commit_permutation == true' \
    "$RUN_ROOT/certificates/$model/certificate_manifest.json" >/dev/null
done
date '+all full certificate gates passed %F %T' >> "$LOG_DIR/stages.log"

pids=()
for index in 0 1 2 3; do
  run_checker_model "${MODELS[$index]}" "$index" &
  pids+=("$!")
done
wait_parallel_wave "${pids[@]}"
for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "completed_label_free_c3_checks" and
     (.parsed_reconstructions / .reconstructions) >= 0.90 and
     .complete_isolated_trace_pair_rate >= 0.80 and
     .one_valid_one_invalid_pair_rate >= 0.20 and
     .isolated_sealed_triple_match_rate >= 0.10 and
     .truncated_model_calls == 0 and
     .labels_read == false and
     .target_was_hidden == true and
     .sealed_claim_was_hidden == true and
     .audit_protocol == "commitment_conditioned_proof_audit_v8" and
     .private_stage0_responses_read == true and
     .proof_obligations_required == true' \
    "$RUN_ROOT/checks/$model/check_manifest.json" >/dev/null
done
date '+all full checker gates passed %F %T' >> "$LOG_DIR/stages.log"

pids=()
for index in 0 1 2 3; do
  run_checker_model \
    "${MODELS[$index]}" "$index" "$NO_PRECOMMIT_CONFIG" \
    "$NO_PRECOMMIT_ROOT" no_precommit &
  pids+=("$!")
done
wait_parallel_wave "${pids[@]}"
for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "completed_label_free_c3_checks" and
     (.parsed_reconstructions / .reconstructions) >= 0.90 and
     .complete_isolated_trace_pair_rate >= 0.80 and
     .one_valid_one_invalid_pair_rate >= 0.20 and
     .isolated_sealed_triple_match_rate >= 0.10 and
     .truncated_model_calls == 0 and
     .labels_read == false and
     .audit_protocol == "isolated_trace_pointwise_v7" and
     .private_stage0_responses_read == false and
     .proof_obligations_required == false and
     .mechanism_ablation.name == "no_checker_private_precommitment"' \
    "$NO_PRECOMMIT_ROOT/checks/$model/check_manifest.json" >/dev/null
done
date '+all no-precommit ablation gates passed %F %T' >> "$LOG_DIR/stages.log"

pids=()
for index in 0 1 2 3; do
  run_checker_model \
    "${MODELS[$index]}" "$index" "$PAIR_VISIBLE_CONFIG" \
    "$PAIR_VISIBLE_ROOT" pair_visible &
  pids+=("$!")
done
wait_parallel_wave "${pids[@]}"
for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "completed_label_free_c3_checks" and
     (.parsed_reconstructions / .reconstructions) >= 0.90 and
     .position_invariant_pair_rate >= 0.50 and
     .sealed_triple_audit_rate >= 0.20 and
     .paired_sealed_triple_match_rate >= 0.10 and
     .truncated_model_calls == 0 and
     .labels_read == false and
     .audit_protocol == "commitment_conditioned_pair_audit_v8_ablation" and
     .private_stage0_responses_read == true and
     .proof_obligations_required == true and
     .mechanism_ablation.name == "pair_visible_with_precommitment"' \
    "$PAIR_VISIBLE_ROOT/checks/$model/check_manifest.json" >/dev/null
done
date '+all pair-visible ablation gates passed %F %T' >> "$LOG_DIR/stages.log"

run_isolated_control_wave \
  "$CANDIDATE_VISIBLE_CONFIG" "$CANDIDATE_VISIBLE_ROOT" \
  candidate_visible_commit_first candidate_visible_commit_first_v8_control true
run_isolated_control_wave \
  "$UNSEALED_CONFIG" "$UNSEALED_ROOT" \
  unsealed_set_aware unsealed_set_aware_v8_control false

pids=()
for index in 0 1 2 3; do
  run_pre_pair_model "${MODELS[$index]}" "$index" &
  pids+=("$!")
done
wait_parallel_wave "${pids[@]}"
for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "completed_label_free_pre_pair_style_model" and
     (.parsed_pointwise_calls / .pointwise_calls) >= 0.90 and
     (.parsed_pairwise_calls / .pairwise_calls) >= 0.90 and
     (.order_consistent_pairs / .order_audited_pairs) >= 0.75 and
     .truncated_prompts == 0 and
     .labels_read == false and
     .calls_per_model_per_question == 7' \
    "$RUN_ROOT/prepair_style/models/$model/manifest.json" >/dev/null
done
"$PYTHON_BIN" -m bench_coe.innovation.aggregate_pre_pair_style \
  --c3-config "$C3_CONFIG" \
  --baseline-config "$PREPAIR_CONFIG" >> "$LOG_DIR/prepair.aggregate.log" 2>&1
date '+full PRePair-style aggregate complete %F %T' >> "$LOG_DIR/stages.log"

pids=()
for index in 0 1 2 3; do
  run_cfmad_model "${MODELS[$index]}" "$index" &
  pids+=("$!")
done
wait_parallel_wave "${pids[@]}"
for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "completed_label_free_cfmad_style_model" and
     (.parsed_phase_calls.cot / .phase_calls.cot) >= 0.90 and
     (.parsed_phase_calls.abduction / .phase_calls.abduction) >= 0.90 and
     (.parsed_phase_calls.critic / .phase_calls.critic) >= 0.90 and
     (.parsed_phase_calls.defense / .phase_calls.defense) >= 0.90 and
     (.parsed_phase_calls.judge / .phase_calls.judge) >= 0.90 and
     .distinct_stance_pairs == .questions and
     .truncated_prompts == 0 and
     .labels_read == false and
     .base_predictions_read == false and
     .certificate_or_check_outputs_read == false and
     .calls_per_model_per_question == 10 and
     .actual_model_calls == (.questions * 10)' \
    "$RUN_ROOT/cfmad_style/models/$model/manifest.json" >/dev/null
done
if [[ ! -f "$RUN_ROOT/cfmad_style/aggregate/manifest.json" ]]; then
  "$PYTHON_BIN" -m bench_coe.innovation.aggregate_cfmad_style \
    --c3-config "$C3_CONFIG" \
    --baseline-config "$CFMAD_CONFIG" >> "$LOG_DIR/cfmad.aggregate.log" 2>&1
fi
jq -e \
  '.status == "completed_label_free_cfmad_style_aggregate" and
   .labels_read == false and
   .calls_per_model_per_question == 10 and
   .ensemble_calls_per_question == 40 and
   .questions == 268' \
  "$RUN_ROOT/cfmad_style/aggregate/manifest.json" >/dev/null
date '+full CFMAD-style aggregate complete %F %T' >> "$LOG_DIR/stages.log"

for method in self_consistency self_revision; do
  pids=()
  for index in 0 1 2 3; do
    run_equal_call_model "$method" "${MODELS[$index]}" "$index" &
    pids+=("$!")
  done
  wait_parallel_wave "${pids[@]}"
  for model in "${MODELS[@]}"; do
    jq -e \
      '.status == "completed_label_free_equal_call_single_model" and
       (.parsed_final_samples / .final_samples) >= 0.90 and
       .truncated_prompts == 0 and
       .labels_read == false and
       .calls_per_question == 42' \
      "$RUN_ROOT/equal_call_single_model/$method/$model/manifest.json" >/dev/null
  done
  date "+full equal-call $method complete %F %T" >> "$LOG_DIR/stages.log"
done

"$PYTHON_BIN" -m bench_coe.innovation.evaluate_c3_development \
  --config "$C3_CONFIG" \
  --equal-call-config "$EQUAL_CONFIG" \
  --prepair-config "$PREPAIR_CONFIG" \
  --cfmad-config "$CFMAD_CONFIG" \
  --mechanism-ablation-config "$NO_PRECOMMIT_CONFIG" \
  --mechanism-ablation-config "$PAIR_VISIBLE_CONFIG" \
  --mechanism-ablation-config "$CANDIDATE_VISIBLE_CONFIG" \
  --mechanism-ablation-config "$UNSEALED_CONFIG" \
  >> "$LOG_DIR/evaluation.log" 2>&1

jq -e '.status | IN("development_gate_pass", "development_gate_fail")' \
  "$RUN_ROOT/development_evaluation/development_gate.json" >/dev/null
date '+full development evaluation complete %F %T' >> "$LOG_DIR/stages.log"
