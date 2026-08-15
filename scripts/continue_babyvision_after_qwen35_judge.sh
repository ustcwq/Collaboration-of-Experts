#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sm5/ys/FCS
BV="$ROOT/BabyVision"
OUT="$BV/outputs/all_local_vlms_20260802"
PY=/home/sm5/anaconda3/envs/LLMs/bin/python
export PYTHONPATH="$ROOT/.codex_tmp/deepseek_transformers_438:$ROOT/.codex_tmp/DeepSeek-VL2:$ROOT/BabyVision"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while tmux has-session -t babyvision_qwen35_judge_front4 2>/dev/null; do
  sleep 30
done

run_deepseek() {
  local model=$1
  local gpu_ids=$2
  local start=$3
  local limit=$4
  local predictions=$5
  local chunk_size=$6
  "$PY" -m babyvision_eval.deepseek_official_eval \
    --data-root "$BV/data/babyvision_data" \
    --model-path "$ROOT/models_v/$model" \
    --model-name "$model" \
    --output-dir "$OUT/${model}__judge_skipped" \
    --predictions-file "$predictions" \
    --gpu-ids "$gpu_ids" \
    --prompt-mode audit_trace_json \
    --max-new-tokens 768 \
    --chunk-size "$chunk_size" \
    --start-index "$start" \
    --limit "$limit" \
    --resume
}

merge_model() {
  local model=$1
  shift
  "$PY" - "$model" "$@" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
paths = [Path(path) for path in sys.argv[2:]]
root = Path("/home/sm5/ys/FCS/BabyVision")
destination = root / "outputs/all_local_vlms_20260802" / f"{model}__judge_skipped/predictions.jsonl"
order = []
for line in (root / "data/babyvision_data/meta_data.jsonl").open(encoding="utf-8"):
    order.append(str(json.loads(line)["taskId"]))
last = {}
if destination.is_file():
    paths.insert(0, destination)
for path in paths:
    if not path.is_file():
        continue
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if row.get("model_name") == model:
            last[str(row.get("sample_id"))] = row
clean = {
    sample_id: row for sample_id, row in last.items()
    if not row.get("model_error") and row.get("model_command_returncode") == 0
}
if len(clean) != 388:
    raise SystemExit(f"{model}: expected 388 clean predictions, got {len(clean)}")
temporary = destination.with_suffix(".jsonl.tmp")
temporary.parent.mkdir(parents=True, exist_ok=True)
with temporary.open("w", encoding="utf-8") as handle:
    for sample_id in order:
        handle.write(json.dumps(clean[sample_id], ensure_ascii=False) + "\n")
temporary.replace(destination)
print(model, len(clean))
PY
}

mkdir -p "$OUT/logs_deepseek_official" "$OUT/smoke"

run_deepseek DeepSeek-VL2-Tiny 0 0 1 \
  "$OUT/smoke/DeepSeek-VL2-Tiny.predictions.jsonl" -1 \
  > "$OUT/logs_deepseek_official/DeepSeek-VL2-Tiny.smoke.log" 2>&1

tiny_dir="$OUT/DeepSeek-VL2-Tiny__judge_skipped"
mkdir -p "$tiny_dir"
pids=()
for gpu in 0 1 2 3; do
  start=$((gpu * 97))
  run_deepseek DeepSeek-VL2-Tiny "$gpu" "$start" 97 \
    "$tiny_dir/predictions.official.rank${gpu}.jsonl" -1 \
    > "$OUT/logs_deepseek_official/DeepSeek-VL2-Tiny.gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
merge_model DeepSeek-VL2-Tiny "$tiny_dir"/predictions.official.rank*.jsonl

run_deepseek DeepSeek-VL2-Small 0,1 0 1 \
  "$OUT/smoke/DeepSeek-VL2-Small.predictions.jsonl" -1 \
  > "$OUT/logs_deepseek_official/DeepSeek-VL2-Small.smoke.log" 2>&1

small_dir="$OUT/DeepSeek-VL2-Small__judge_skipped"
mkdir -p "$small_dir"
run_deepseek DeepSeek-VL2-Small 0,1 0 194 \
  "$small_dir/predictions.official.rank0.jsonl" -1 \
  > "$OUT/logs_deepseek_official/DeepSeek-VL2-Small.gpu0-1.log" 2>&1 &
pid_a=$!
run_deepseek DeepSeek-VL2-Small 2,3 194 194 \
  "$small_dir/predictions.official.rank1.jsonl" -1 \
  > "$OUT/logs_deepseek_official/DeepSeek-VL2-Small.gpu2-3.log" 2>&1 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
merge_model DeepSeek-VL2-Small "$small_dir"/predictions.official.rank*.jsonl

cd "$ROOT"
/home/sm5/anaconda3/envs/Factory/bin/python tools/run_babyvision_judge_scheduler.py \
  > "$OUT/logs_qwen35_judge_deepseek.log" 2>&1
