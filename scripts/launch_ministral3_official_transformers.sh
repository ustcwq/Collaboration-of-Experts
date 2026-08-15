#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

MODEL=Ministral-3-3B-Instruct-2512
OUTPUT=outputs/model_benchmarks/family_scale_expansion_full_20260731/text/official
SHARDS=outputs/model_benchmarks/family_scale_expansion_full_20260731/text/official_transformers_shards
WORKERS="$OUTPUT/workers/transformers_retry"
mkdir -p "$WORKERS" "$SHARDS"

complete() {
  local benchmark=$1 expected=$2
  /home/sm5/anaconda3/envs/Factory/bin/python - "$OUTPUT/$benchmark/$MODEL" "$expected" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = sum(1 for line in (root / "predictions.jsonl").open(encoding="utf-8") if line.strip())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if summary.get("status") == "completed" and int(summary.get("num_examples", -1)) == expected and rows == expected else 1)
PY
}

shard_complete() {
  local benchmark=$1 index=$2 expected=$3
  local root="$SHARDS/$benchmark/shard$index/$benchmark/$MODEL"
  /home/sm5/anaconda3/envs/Factory/bin/python - "$root" "$expected" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = sum(1 for line in (root / "predictions.jsonl").open(encoding="utf-8") if line.strip())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if summary.get("status") == "completed" and int(summary.get("num_examples", -1)) == expected and rows == expected else 1)
PY
}

while read -r session; do
  [[ -n "$session" ]] && tmux kill-session -t "$session" 2>/dev/null || true
done < <(tmux list-sessions -F '#S' 2>/dev/null | grep '^benchcoe_ministral3_official_')

/home/sm5/anaconda3/envs/Factory/bin/python - <<'PY'
import json
from pathlib import Path

model = "Ministral-3-3B-Instruct-2512"
output = Path("outputs/model_benchmarks/family_scale_expansion_full_20260731/text/official")
shards = Path("outputs/model_benchmarks/family_scale_expansion_full_20260731/text/official_transformers_shards")
workers = output / "workers/transformers_retry"
source = json.loads((output / f"workers/{model}.input.json").read_text(encoding="utf-8"))
plans = {
    "bbh": [(0, 6), (1, 6), (2, 6), (4, 6), (5, 6)],
    "gpqa": [(3, 4), (6, 4), (7, 4)],
}
for benchmark, assignments in plans.items():
    for index, (gpu, batch_size) in enumerate(assignments):
        payload = json.loads(json.dumps(source))
        payload["benchmarks"] = [benchmark]
        payload["args"].update({
            "benchmarks": benchmark,
            "backend": "transformers",
            "attn_implementation": "eager",
            "resume": False,
            "row_shard_count": len(assignments),
            "row_shard_index": index,
            "output_dir": str(shards / benchmark / f"shard{index}"),
        })
        payload["transformers_batch_size"] = batch_size
        payload["gpu"] = gpu
        (workers / f"{benchmark}.shard{index}.input.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

payload = json.loads(json.dumps(source))
payload["benchmarks"] = ["mmstar_text_only"]
payload["args"].update({
    "benchmarks": "mmstar_text_only",
    "backend": "transformers",
    "attn_implementation": "eager",
    "resume": True,
    "row_shard_count": 1,
    "row_shard_index": 0,
})
(workers / "mmstar_text_only.input.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

launch_shard() {
  local benchmark=$1 index=$2 gpu=$3 batch_size=$4
  local session="benchcoe_ministral3_official_${benchmark}_shard${index}_gpu${gpu}"
  local input="$WORKERS/${benchmark}.shard${index}.input.json"
  local result="$WORKERS/${benchmark}.shard${index}.output.json"
  local log="$WORKERS/${benchmark}.shard${index}.log"
  tmux new-session -d -s "$session" \
    "cd /home/sm5/ys/FCS && env CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BENCH_COE_TRANSFORMERS_BATCH_SIZE=$batch_size PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/sm5/anaconda3/envs/Factory/bin/python -m bench_coe.run_official_model_benchmarks --worker-input '$input' --worker-output '$result' > '$log' 2>&1"
}

need_bbh=0
need_gpqa=0
need_mmstar=0
complete bbh 6511 || need_bbh=1
complete gpqa 4768 || need_gpqa=1
complete mmstar_text_only 1500 || need_mmstar=1

if [[ "$need_bbh" -eq 1 ]]; then
  shard_complete bbh 0 1303 || launch_shard bbh 0 0 6
  shard_complete bbh 1 1302 || launch_shard bbh 1 1 6
  shard_complete bbh 2 1302 || launch_shard bbh 2 2 6
  shard_complete bbh 3 1302 || launch_shard bbh 3 4 6
  shard_complete bbh 4 1302 || launch_shard bbh 4 5 6
fi
if [[ "$need_gpqa" -eq 1 ]]; then
  shard_complete gpqa 0 1590 || launch_shard gpqa 0 3 4
  shard_complete gpqa 1 1589 || launch_shard gpqa 1 6 4
  shard_complete gpqa 2 1589 || launch_shard gpqa 2 7 4
fi

tmux new-session -d -s benchcoe_ministral3_official_finalize \
  "cd /home/sm5/ys/FCS && set -e; while tmux ls 2>/dev/null | grep -Eq 'benchcoe_ministral3_official_(bbh|gpqa)_shard'; do sleep 60; done; if [[ $need_bbh -eq 1 ]]; then env PYTHONPATH=. /home/sm5/anaconda3/envs/Factory/bin/python tools/merge_official_text_shards.py --benchmark bbh --model $MODEL --shard-root $SHARDS/bbh --shard-count 5 --output-root $OUTPUT > $WORKERS/bbh.merge.log 2>&1; fi; if [[ $need_gpqa -eq 1 ]]; then env PYTHONPATH=. /home/sm5/anaconda3/envs/Factory/bin/python tools/merge_official_text_shards.py --benchmark gpqa --model $MODEL --shard-root $SHARDS/gpqa --shard-count 3 --output-root $OUTPUT > $WORKERS/gpqa.merge.log 2>&1; fi; if [[ $need_mmstar -eq 1 ]]; then env CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BENCH_COE_TRANSFORMERS_BATCH_SIZE=16 /home/sm5/anaconda3/envs/Factory/bin/python -m bench_coe.run_official_model_benchmarks --worker-input $WORKERS/mmstar_text_only.input.json --worker-output $WORKERS/mmstar_text_only.output.json > $WORKERS/mmstar_text_only.log 2>&1; fi"
