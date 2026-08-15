#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PYTHON=${BENCHCOE_VLM_PYTHON:-/home/sm5/anaconda3/envs/Factory/bin/python}
OUTPUT=outputs/model_benchmarks/autonomous_remaining_full_20260802
STATE="$OUTPUT/state"
LOG="$OUTPUT/logs/front4_overflow.log"
MODEL=Qwen3-VL-8B-Thinking
TASKS=(mmmu_pro mathvista gaokao_mm)
GPUS=(0 1 2 3)
mkdir -p "$STATE" "$(dirname "$LOG")" "$OUTPUT/logs/$MODEL"

declare -A pids gpu_tasks

for task in "${TASKS[@]}"; do
  [[ -f "$STATE/${MODEL}__${task}.completed" ]] || printf '4\n' >"$STATE/${MODEL}__${task}.attempt"
done

gpu_idle() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v wanted="$gpu" '{gsub(/ /,"",$0)} ($1+0)==wanted {exit !(($2+0)<2048 && ($3+0)<10)}'
}

run_task() {
  local gpu="$1" task="$2" task_log="$OUTPUT/logs/$MODEL/${task}.overflow.log"
  if [[ "$task" == gaokao_mm ]]; then
    "$PYTHON" -m bench_coe.run_gaokao_mm_babyvision_models \
      --models-dir models_v --models "$MODEL" --gpu-devices "$gpu" --parallel-workers 1 \
      --output-dir "$OUTPUT/vision/gaokao_mm" >"$task_log" 2>&1
  else
    "$PYTHON" -m bench_coe.run_multimodal_babyvision_models \
      --models-dir models_v --models "$MODEL" --benchmarks "$task" \
      --gpu-devices "$gpu" --parallel-workers 1 --output-dir "$OUTPUT/vision/$task" \
      >"$task_log" 2>&1
  fi
}

while :; do
  for gpu in "${GPUS[@]}"; do
    pid="${pids[$gpu]:-}"
    [[ -n "$pid" ]] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      task="${gpu_tasks[$gpu]}"
      if wait "$pid"; then
        date '+%F %T' >"$STATE/${MODEL}__${task}.completed"
        printf '%s GPU%s completed %s\n' "$(date '+%F %T')" "$gpu" "$task" >>"$LOG"
      else
        printf '2\n' >"$STATE/${MODEL}__${task}.attempt"
        printf '%s GPU%s failed %s; returned to shared queue\n' "$(date '+%F %T')" "$gpu" "$task" >>"$LOG"
      fi
      unset 'pids[$gpu]' 'gpu_tasks[$gpu]'
    fi
  done

  for gpu in "${GPUS[@]}"; do
    [[ -z "${pids[$gpu]:-}" ]] || continue
    gpu_idle "$gpu" || continue
    selected=""
    for task in "${TASKS[@]}"; do
      [[ -f "$STATE/${MODEL}__${task}.completed" ]] && continue
      [[ -f "$STATE/${MODEL}__${task}.overflow_running" ]] && continue
      selected="$task"
      break
    done
    [[ -n "$selected" ]] || continue
    printf '4\n' >"$STATE/${MODEL}__${selected}.attempt"
    date '+%F %T' >"$STATE/${MODEL}__${selected}.overflow_running"
    run_task "$gpu" "$selected" &
    pids[$gpu]=$!
    gpu_tasks[$gpu]="$selected"
    printf '%s GPU%s started %s\n' "$(date '+%F %T')" "$gpu" "$selected" >>"$LOG"
    sleep 3
  done

  remaining=0
  for task in "${TASKS[@]}"; do
    [[ -f "$STATE/${MODEL}__${task}.completed" ]] || remaining=$((remaining+1))
  done
  [[ "$remaining" -eq 0 && "${#pids[@]}" -eq 0 ]] && break
  sleep 20
done

date '+%F %T' >"$STATE/front4_overflow.completed"
