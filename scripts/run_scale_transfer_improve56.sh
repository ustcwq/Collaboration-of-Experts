#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS
export PYTHONPATH=.

VIEW=outputs/bench_coe/scale_transfer_views_20260802
OUTPUT=outputs/bench_coe/scale_transfer_improve56_20260802
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

export BENCH_COE_MMLU_VALIDATION_ROOT="$VIEW/text/mmlu_validation"
export BENCH_COE_MMLU_TEST_ROOT="$VIEW/text/mmlu_test"
export BENCH_COE_BBH_ROOT="$VIEW/text/bbh"
export BENCH_COE_GPQA_ROOT="$VIEW/text/gpqa"
export BENCH_COE_MMSTAR_ROOT="$VIEW/text/mmstar_text_only"
export BENCH_COE_CMMMU_VAL_ROOT="$VIEW/vision/cmmmu/val"
export BENCH_COE_MMMU_PRO_TEST_ROOT="$VIEW/vision/mmmu_pro/standard_10_options/test"
export BENCH_COE_MATHVISTA_ROOT="$VIEW/vision/mathvista/testmini"

TEXT_CASES=mmlu_val_to_mmlu_test,mmlu_val_to_bbh,mmlu_val_to_gpqa,mmlu_val_to_mmstar
VISION_CASES=mmmu_pro_to_cmmmu,mmmu_pro_to_mathvista

run_cohort() {
  local name="$1" cases="$2"
  shift 2
  local models=("$@")
  local force_cohorts=",${BENCHCOE_FORCE_COHORTS:-},"
  local force=false
  if [[ "$force_cohorts" == *",$name,"* ]]; then
    force=true
  fi
  mkdir -p "$OUTPUT/$name"
  if [[ "$force" == true || ! -f "$OUTPUT/$name/improve5/summary.json" ]]; then
    python -m bench_coe.improve5_failure_ecology_experiments \
      --cases "$cases" --include-models "${models[@]}" \
      --output-dir "$OUTPUT/$name/improve5" \
      >"$LOGS/${name}_improve5.log" 2>&1 || true
  fi
  if [[ "$force" == true || ! -f "$OUTPUT/$name/improve6/summary.json" ]]; then
    python -m bench_coe.improve6_adaptive_failure_ecology_experiments \
      --cases "$cases" --include-models "${models[@]}" \
      --output-dir "$OUTPUT/$name/improve6" \
      >"$LOGS/${name}_improve6.log" 2>&1 || true
  fi
}

python tools/materialize_scale_transfer_views.py >"$LOGS/materialize.log" 2>&1

run_cohort language_2b_4b "$TEXT_CASES" \
  Qwen3-1.7B Qwen2.5-3B-Instruct Qwen3-4B-Instruct-2507 \
  granite-3.3-2b-instruct internlm2_5-1_8b-chat gemma-2-2b-it \
  Llama-3.2-3B-Instruct DeepSeek-R1-Distill-Qwen-1.5B \
  Ministral-3-3B-Instruct-2512

run_cohort language_14b "$TEXT_CASES" \
  Qwen3-14B Qwen2.5-14B-Instruct Baichuan2-13B-Chat \
  DeepSeek-R1-Distill-Qwen-14B Mistral-Nemo-Instruct-2407

run_cohort vision_7b_9b "$VISION_CASES" \
  GLM-4.1V-9B-Thinking InternVL3_5-8B Qwen2.5-VL-7B-Instruct \
  Qwen3-VL-8B-Thinking Llama-3.1-Nemotron-Nano-VL-8B-V1

run_cohort vision_14b "$VISION_CASES" \
  Phi-4-reasoning-vision-15B InternVL3_5-14B gemma-3-12b-it

python tools/summarize_autonomous_scale_transfer.py >"$LOGS/summary.log" 2>&1 || true
date '+%F %T' >"$OUTPUT/completed"
