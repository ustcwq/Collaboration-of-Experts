#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
PYTHON_BIN=/home/sm5/anaconda3/envs/FactoryS/bin/python
GPU_RUNNER=scripts/run_with_front4_gpu_claim.sh
C3_CONFIG=configs/innovation/c3_development_v8.yaml
CFMAD_CONFIG=configs/innovation/c3_cfmad_style_v8.yaml
RUN_ROOT=outputs/bench_coe/innovation/cross_examined_certificates/dev_v8_20260815
SMOKE_QUESTIONS=2
SMOKE_ROOT="$RUN_ROOT/smoke/cfmad_style_n${SMOKE_QUESTIONS}"
LOG_DIR="$SMOKE_ROOT/queue_logs"
CHECKER=General-Reasoner-7B-preview

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

wait_for_prompt_control_smokes() {
  while true; do
    local no_precommit=(
      "$RUN_ROOT"/mechanism_ablations/no_checker_private_precommitment/smoke/checks/${CHECKER}_n*_gpu0/check_manifest.json
    )
    local pair_visible=(
      "$RUN_ROOT"/mechanism_ablations/pair_visible_with_precommitment/smoke/checks/${CHECKER}_n*_gpu1/check_manifest.json
    )
    local candidate_visible=(
      "$RUN_ROOT"/mechanism_ablations/candidate_visible_commit_first/smoke/checks/${CHECKER}_n*_gpu2/check_manifest.json
    )
    local unsealed=(
      "$RUN_ROOT"/mechanism_ablations/unsealed_set_aware/smoke/checks/${CHECKER}_n*_gpu3/check_manifest.json
    )
    if [[ -f "${no_precommit[0]}" && -f "${pair_visible[0]}" && \
          -f "${candidate_visible[0]}" && -f "${unsealed[0]}" ]] && jq -e \
      '(.parsed_reconstructions / .reconstructions) >= 0.90 and
       .complete_isolated_trace_pair_rate >= 0.80 and
       .one_valid_one_invalid_pair_rate >= 0.20 and
       .isolated_sealed_triple_match_rate >= 0.10 and
       .truncated_model_calls == 0 and
       .labels_read == false and
       .audit_protocol == "isolated_trace_pointwise_v7" and
       .mechanism_ablation.name == "no_checker_private_precommitment"' \
      "${no_precommit[0]}" >/dev/null && jq -e \
      '(.parsed_reconstructions / .reconstructions) >= 0.90 and
       .position_invariant_pair_rate >= 0.50 and
       .paired_sealed_triple_match_rate >= 0.10 and
       .truncated_model_calls == 0 and
       .labels_read == false and
       .audit_protocol == "commitment_conditioned_pair_audit_v8_ablation" and
       .mechanism_ablation.name == "pair_visible_with_precommitment"' \
      "${pair_visible[0]}" >/dev/null && jq -e \
      '(.parsed_reconstructions / .reconstructions) >= 0.90 and
       .complete_isolated_trace_pair_rate >= 0.80 and
       .one_valid_one_invalid_pair_rate >= 0.20 and
       .isolated_sealed_triple_match_rate >= 0.10 and
       .truncated_model_calls == 0 and
       .labels_read == false and
       .audit_protocol == "candidate_visible_commit_first_v8_control" and
       .mechanism_ablation.name == "candidate_visible_commit_first"' \
      "${candidate_visible[0]}" >/dev/null && jq -e \
      '(.parsed_reconstructions / .reconstructions) >= 0.90 and
       .complete_isolated_trace_pair_rate >= 0.80 and
       .one_valid_one_invalid_pair_rate >= 0.20 and
       .isolated_sealed_triple_match_rate >= 0.10 and
       .truncated_model_calls == 0 and
       .labels_read == false and
       .audit_protocol == "unsealed_set_aware_v8_control" and
       .mechanism_ablation.name == "unsealed_set_aware"' \
      "${unsealed[0]}" >/dev/null; then
      return 0
    fi
    sleep 30
  done
}

run_model() {
  local model=$1
  local gpu=$2
  local manifest="$SMOKE_ROOT/models/$model/manifest.json"
  local log="$LOG_DIR/${model}.gpu${gpu}.log"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  while true; do
    wait_for_fixed_idle_gpu "$gpu"
    date '+CFMAD-style smoke start %F %T' >> "$log"
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
        --physical-gpu "$gpu" \
        --smoke-questions "$SMOKE_QUESTIONS" >> "$log" 2>&1; then
      date '+CFMAD-style smoke finish %F %T' >> "$log"
      return 0
    fi
    date '+CFMAD-style smoke failed; returning to idle queue %F %T' >> "$log"
  done
}

wait_for_prompt_control_smokes

pids=()
for index in 0 1 2 3; do
  run_model "${MODELS[$index]}" "$index" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

for model in "${MODELS[@]}"; do
  jq -e \
    '.status == "bounded_label_free_cfmad_style_smoke" and
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
    "$SMOKE_ROOT/models/$model/manifest.json" >/dev/null
done

if [[ ! -f "$SMOKE_ROOT/aggregate/manifest.json" ]]; then
  "$PYTHON_BIN" -m bench_coe.innovation.aggregate_cfmad_style \
    --c3-config "$C3_CONFIG" \
    --baseline-config "$CFMAD_CONFIG" \
    --smoke-questions "$SMOKE_QUESTIONS" >> "$LOG_DIR/aggregate.log" 2>&1
fi

jq -e \
  '.status == "bounded_label_free_cfmad_style_aggregate_smoke" and
   .labels_read == false and
   .calls_per_model_per_question == 10 and
   .ensemble_calls_per_question == 40 and
   .questions == 2' \
  "$SMOKE_ROOT/aggregate/manifest.json" >/dev/null

date '+all CFMAD-style bounded smoke gates passed %F %T' >> "$LOG_DIR/stages.log"
