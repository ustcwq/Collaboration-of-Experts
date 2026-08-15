from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BABYVISION_ROOT = ROOT / "BabyVision"
OUTPUT_ROOT = BABYVISION_ROOT / "outputs/all_local_vlms_20260802"
LOG_ROOT = OUTPUT_ROOT / "judge_logs"
STATE_PATH = OUTPUT_ROOT / "state/judge_status.json"
FORWARD_ONLY_FLAG = OUTPUT_ROOT / "state/forward_only.flag"
PYTHON = Path("/home/sm5/anaconda3/envs/Factory/bin/python")
JUDGE_MODEL = ROOT / "models_v/Qwen3.5-9B"
JUDGE_NAME = "Qwen3.5-9B"
EXPECTED_SAMPLES = 388
POLL_SECONDS = 5
MAX_ATTEMPTS = 2
ALLOWED_GPUS = {0, 1, 2, 3}


@dataclass
class Worker:
    target: str
    gpu: int
    attempt: int
    process: subprocess.Popen[Any]
    log_handle: Any


def jsonl_status(path: Path) -> tuple[int, int]:
    last: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return 0, 0
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        last[str(row.get("sample_id"))] = row
    clean = sum(
        not row.get("model_error") and row.get("model_command_returncode") in (0, None)
        for row in last.values()
    )
    return len(last), clean


def summary_complete(directory: Path) -> bool:
    path = directory / f"summary_judge_by_{JUDGE_NAME}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return int(payload.get("num_judged") or 0) == EXPECTED_SAMPLES


def candidates() -> list[str]:
    result: list[str] = []
    for directory in sorted(OUTPUT_ROOT.glob("*__judge_skipped")):
        total, clean = jsonl_status(directory / "predictions.jsonl")
        if total == EXPECTED_SAMPLES and clean == EXPECTED_SAMPLES and not summary_complete(directory):
            result.append(directory.name)
    return result


def gpu_snapshot() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    snapshot: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            snapshot[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return snapshot


def launch(target: str, gpu: int, attempt: int) -> Worker:
    directory = OUTPUT_ROOT / target
    log_path = LOG_ROOT / target / f"attempt{attempt}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        str(PYTHON), "-m", "babyvision_eval.eval_local",
        "--judge-only",
        "--data-root", "data/babyvision_data",
        "--output-dir", str(directory),
        "--predictions-file", str(directory / "predictions.jsonl"),
        "--judge-backend", "local_vlm",
        "--judge-local-model-path", str(JUDGE_MODEL),
        "--judge-local-model-name", JUDGE_NAME,
        "--judge-local-gpu-ids", str(gpu),
        "--judge-local-dtype", "bfloat16",
        "--judge-local-attn-implementation", "sdpa",
        "--judge-local-trust-remote-code",
        "--judge-local-max-input-tokens", "4096",
        "--judge-local-max-new-tokens", "768",
        "--judge-local-temperature", "0",
        "--resume",
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
    return Worker(target, gpu, attempt, process, log_handle)


def write_status(queue: list[str], workers: dict[int, Worker], attempts: dict[str, int], failed: set[str]) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": JUDGE_NAME,
        "queued": queue,
        "running": {
            str(gpu): {"target": worker.target, "attempt": worker.attempt, "pid": worker.process.pid}
            for gpu, worker in sorted(workers.items())
        },
        "completed": sorted(target for target in attempts if summary_complete(OUTPUT_ROOT / target)),
        "failed": sorted(failed),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if FORWARD_ONLY_FLAG.exists():
        scheduler = ROOT / "tools/run_babyvision_all_vlms_scheduler.py"
        os.execv(str(PYTHON), [str(PYTHON), str(scheduler)])
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    queue = candidates()
    attempts = {target: 0 for target in queue}
    workers: dict[int, Worker] = {}
    failed: set[str] = set()
    while queue or workers:
        for gpu, worker in list(workers.items()):
            return_code = worker.process.poll()
            if return_code is None:
                continue
            worker.log_handle.close()
            workers.pop(gpu)
            if summary_complete(OUTPUT_ROOT / worker.target):
                continue
            if attempts[worker.target] < MAX_ATTEMPTS:
                queue.append(worker.target)
            else:
                failed.add(worker.target)

        idle_gpus = [
            gpu for gpu, (memory, utilization) in sorted(gpu_snapshot().items())
            if gpu in ALLOWED_GPUS
            and gpu not in workers
            and memory < 2048
            and utilization < 10
        ]
        while queue and idle_gpus:
            target = queue.pop(0)
            if summary_complete(OUTPUT_ROOT / target):
                continue
            attempts[target] += 1
            gpu = idle_gpus.pop(0)
            workers[gpu] = launch(target, gpu, attempts[target])
        write_status(queue, workers, attempts, failed)
        time.sleep(POLL_SECONDS)
    write_status(queue, workers, attempts, failed)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
