from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BABYVISION_ROOT = ROOT / "BabyVision"
OUTPUT_ROOT = BABYVISION_ROOT / "outputs/all_local_vlms_20260802"
STATE_ROOT = OUTPUT_ROOT / "state/large_forward_shards"
LOG_ROOT = OUTPUT_ROOT / "logs_rescue"
PYTHON = "/home/sm5/anaconda3/envs/MMLU/bin/python"
EXPECTED_SAMPLES = 388

JOBS = {
    "gemma-4-26B-A4B-it": [(0, 1), (3, 4)],
    "Qwen3.6-27B": [(0, 1), (2, 7), (3, 4), (5, 6)],
}


def read_last(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        rows[str(row.get("sample_id"))] = row
    return rows


def clean_ids(path: Path) -> set[str]:
    return {
        sample_id
        for sample_id, row in read_last(path).items()
        if not row.get("model_error") and row.get("model_command_returncode") in (0, None)
    }


def all_sample_ids() -> list[str]:
    path = BABYVISION_ROOT / "data/babyvision_data/meta_data.jsonl"
    result: list[str] = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        result.append(str(row["taskId"]))
    return result


def merge_predictions(destination: Path, shards: list[Path]) -> int:
    merged = read_last(destination)
    for shard in shards:
        merged.update(read_last(shard))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in merged.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return len(clean_ids(destination))


def launch_worker(model: str, pair: tuple[int, int], ids_path: Path, predictions_path: Path) -> tuple[subprocess.Popen[Any], Any]:
    gpu_label = "-".join(str(gpu) for gpu in pair)
    log_path = LOG_ROOT / f"{model}_forward_shard_gpu{gpu_label}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        PYTHON, "-m", "babyvision_eval.eval_local",
        "--engine", "transformers",
        "--data-root", "data/babyvision_data",
        "--model-path", str(ROOT / "models_v" / model),
        "--model-name", model,
        "--output-dir", str(OUTPUT_ROOT / f"{model}__judge_skipped"),
        "--predictions-file", str(predictions_path),
        "--sample-ids-file", str(ids_path),
        "--prompt-mode", "audit_trace_json",
        "--max-input-tokens", "4096",
        "--max-new-tokens", "768",
        "--temperature", "0",
        "--dtype", "bfloat16",
        "--gpu-ids", ",".join(str(gpu) for gpu in pair),
        "--parallel-mode", "single",
        "--device-map", "auto",
        "--trust-remote-code",
        "--resume",
        "--judge-backend", "none",
    ]
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    process = subprocess.Popen(
        command,
        cwd=BABYVISION_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    sample_ids = all_sample_ids()
    workers: list[tuple[str, Path, subprocess.Popen[Any], Any]] = []
    shard_paths: dict[str, list[Path]] = {}

    for model, pairs in JOBS.items():
        directory = OUTPUT_ROOT / f"{model}__judge_skipped"
        destination = directory / "predictions.jsonl"
        remaining = [sample_id for sample_id in sample_ids if sample_id not in clean_ids(destination)]
        shards = [remaining[index::len(pairs)] for index in range(len(pairs))]
        shard_paths[model] = []
        for index, (pair, shard_ids) in enumerate(zip(pairs, shards)):
            ids_path = STATE_ROOT / f"{model}.shard{index}.ids"
            ids_path.write_text("\n".join(shard_ids) + "\n", encoding="utf-8")
            predictions_path = directory / f"predictions.forward_shard{index}.jsonl"
            shard_paths[model].append(predictions_path)
            if shard_ids:
                process, log_handle = launch_worker(model, pair, ids_path, predictions_path)
                workers.append((model, predictions_path, process, log_handle))

    failed = False
    for model, predictions_path, process, log_handle in workers:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failed = True
            print(f"[large-forward] {model} worker failed rc={return_code}: {predictions_path}")

    summary: dict[str, int] = {}
    for model, shards in shard_paths.items():
        destination = OUTPUT_ROOT / f"{model}__judge_skipped/predictions.jsonl"
        summary[model] = merge_predictions(destination, shards)
    (STATE_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed or any(count != EXPECTED_SAMPLES for count in summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
