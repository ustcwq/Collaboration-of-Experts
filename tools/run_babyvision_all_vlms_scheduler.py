from __future__ import annotations

import json
import os
import subprocess
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BABYVISION_ROOT = ROOT / "BabyVision"
MODELS_ROOT = ROOT / "models_v"
OUTPUT_ROOT = BABYVISION_ROOT / "outputs/all_local_vlms_20260802"
LOG_ROOT = OUTPUT_ROOT / "logs"
STATE_ROOT = OUTPUT_ROOT / "state"
PYTHON = Path("/home/sm5/anaconda3/envs/MMLU/bin/python")
EXPECTED_SAMPLES = 388
MAX_ATTEMPTS = 3
POLL_SECONDS = 5
FAILED_MARKER_SUFFIX = ".failed"
PERMANENTLY_SKIPPED_MODELS = {"Keye-VL-1.5-8B", "deepseek-vl2-small"}
ALLOWED_GPUS = {0, 1, 2, 3}

PRIORITY = [
    "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "InternVL3_5-8B",
    "InternVL3_5-14B",
    "gemma-3-12b-it",
    "GLM-4.1V-9B-Thinking",
    "Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "Phi-4-reasoning-vision-15B",
    "Eagle2.5-8B",
    "Keye-VL-1.5-8B",
    "LLaVA-OneVision-2-8B-Instruct",
    "MiMo-VL-7B-RL",
    "Molmo2-8B",
    "VLM-CapCurriculum-InternVL3.5-8B-Staged",
    "ZwZ-8B",
]

EXCLUDED_MODEL_TYPES = {"qwen2", "qwen3", "glm4", "sam3_video"}


@dataclass
class Worker:
    model: str
    gpu: int
    attempt: int
    process: subprocess.Popen[Any]
    log_handle: Any


def load_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def is_visual_model(path: Path) -> bool:
    config = load_config(path)
    model_type = str(config.get("model_type", ""))
    architectures = " ".join(str(item) for item in config.get("architectures", []))
    if model_type in EXCLUDED_MODEL_TYPES:
        return False
    signals = (
        "vl", "vision", "llava", "intern", "gemma3", "gemma4", "idefics",
        "paligemma", "florence", "minicpm", "molmo", "moondream", "ovis",
        "eagle", "sail", "keye", "smol", "multimodal", "conditionalgeneration",
    )
    haystack = f"{model_type} {architectures}".lower()
    return any(signal in haystack for signal in signals)


def predictions_status(path: Path) -> tuple[int, int, str | None]:
    if not path.is_file():
        return 0, 0, None
    last: dict[str, dict[str, Any]] = {}
    try:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            last[str(row.get("sample_id"))] = row
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0, 0, None
    clean = sum(
        not row.get("model_error") and row.get("model_command_returncode") in (0, None)
        for row in last.values()
    )
    model_name = next((str(row.get("model_name")) for row in last.values() if row.get("model_name")), None)
    return len(last), clean, model_name


def completed_models() -> set[str]:
    completed: set[str] = set()
    for path in (BABYVISION_ROOT / "outputs").rglob("predictions.jsonl"):
        total, clean, model_name = predictions_status(path)
        if total == EXPECTED_SAMPLES and clean == EXPECTED_SAMPLES and model_name:
            completed.add(model_name)
    return completed


def candidates() -> list[str]:
    by_real_path: dict[Path, str] = {}
    for path in sorted(MODELS_ROOT.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith(".") or path.name in {"scripts", "__pycache__"}:
            continue
        if not (path.is_dir() or path.is_symlink()) or not (path / "config.json").is_file():
            continue
        if not is_visual_model(path):
            continue
        real_path = path.resolve()
        existing = by_real_path.get(real_path)
        if existing is None or (path.name in PRIORITY and existing not in PRIORITY):
            by_real_path[real_path] = path.name
    names = list(by_real_path.values())
    priority_index = {name: index for index, name in enumerate(PRIORITY)}
    names.sort(key=lambda name: (priority_index.get(name, len(PRIORITY)), name.lower()))
    return names


def gpu_snapshot() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    snapshot: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 3:
            snapshot[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return snapshot


def externally_running_models() -> set[str]:
    result = subprocess.run(
        ["ps", "-eo", "cmd"], check=False, capture_output=True, text=True
    )
    models: set[str] = set()
    for line in result.stdout.splitlines():
        if "babyvision_eval.eval_local" not in line:
            continue
        match = re.search(r"--model-name\s+(\S+)", line)
        if match:
            models.add(match.group(1))
    return models


def output_dir(model: str) -> Path:
    return OUTPUT_ROOT / f"{model}__judge_skipped"


def is_deferred_debug_model(model: str) -> bool:
    return (STATE_ROOT / f"{model}{FAILED_MARKER_SUFFIX}").is_file()


def is_complete(model: str) -> bool:
    total, clean, _ = predictions_status(output_dir(model) / "predictions.jsonl")
    return total == EXPECTED_SAMPLES and clean == EXPECTED_SAMPLES


def generation_tokens(model: str) -> int:
    lowered = model.lower()
    return 2048 if "thinking" in lowered or "reasoning" in lowered or "reason1" in lowered else 768


def launch(model: str, gpu: int, attempt: int) -> Worker:
    destination = output_dir(model)
    destination.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / model / f"attempt{attempt}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        str(PYTHON), "-m", "babyvision_eval.eval_local",
        "--engine", "transformers",
        "--data-root", "data/babyvision_data",
        "--model-path", str(MODELS_ROOT / model),
        "--model-name", model,
        "--output-dir", str(destination),
        "--prompt-mode", "audit_trace_json",
        "--max-input-tokens", "4096",
        "--max-new-tokens", str(generation_tokens(model)),
        "--temperature", "0",
        "--gpu-ids", str(gpu),
        "--parallel-mode", "single",
        "--trust-remote-code",
        "--resume",
        "--judge-backend", "none",
    ]
    if model == "Llama-3.1-Nemotron-Nano-VL-8B-V1":
        command.extend(["--attn-implementation", "eager"])
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "BENCH_COE_CRADIO_REPO": str(ROOT / ".codex_tmp/C-RADIOv2-H-repo"),
    })
    process = subprocess.Popen(
        command,
        cwd=BABYVISION_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return Worker(model=model, gpu=gpu, attempt=attempt, process=process, log_handle=log_handle)


def write_status(queue: list[str], workers: dict[int, Worker], attempts: dict[str, int], completed: set[str], failed: set[str]) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_samples": EXPECTED_SAMPLES,
        "queued": queue,
        "running": {
            str(gpu): {"model": worker.model, "attempt": worker.attempt, "pid": worker.process.pid}
            for gpu, worker in sorted(workers.items())
        },
        "attempts": attempts,
        "completed": sorted(completed),
        "failed": sorted(failed),
    }
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    (STATE_ROOT / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    already_completed = completed_models()
    all_candidates = candidates()
    pending = [
        model for model in all_candidates
        if model not in already_completed
        and model not in PERMANENTLY_SKIPPED_MODELS
        and not is_complete(model)
    ]
    direct_queue = [model for model in pending if not is_deferred_debug_model(model)]
    deferred_queue = [model for model in pending if is_deferred_debug_model(model)]
    queue = direct_queue + deferred_queue
    deferred_models = set(deferred_queue)
    completed = set(already_completed)
    failed: set[str] = set()
    attempts: dict[str, int] = {model: 0 for model in queue}
    workers: dict[int, Worker] = {}

    (STATE_ROOT / "candidate_models.json").write_text(
        json.dumps(all_candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (STATE_ROOT / "initially_completed.json").write_text(
        json.dumps(sorted(already_completed), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    while queue or workers:
        for gpu, worker in list(workers.items()):
            return_code = worker.process.poll()
            if return_code is None:
                continue
            worker.log_handle.close()
            workers.pop(gpu)
            if is_complete(worker.model):
                completed.add(worker.model)
                (STATE_ROOT / f"{worker.model}.completed").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
            elif attempts[worker.model] < MAX_ATTEMPTS:
                queue.append(worker.model)
            else:
                failed.add(worker.model)
                (STATE_ROOT / f"{worker.model}.failed").write_text(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} return_code={return_code}\n",
                    encoding="utf-8",
                )

        snapshot = gpu_snapshot()
        idle_gpus = [
            gpu for gpu, (memory, utilization) in sorted(snapshot.items())
            if gpu in ALLOWED_GPUS
            and gpu not in workers
            and memory < 2048
            and utilization < 10
        ]
        external_models = externally_running_models()
        direct_work_remaining = any(model not in deferred_models for model in queue) or any(
            worker.model not in deferred_models for worker in workers.values()
        )
        inspected = 0
        queue_size = len(queue)
        while queue and idle_gpus and inspected < queue_size:
            model = queue.pop(0)
            inspected += 1
            if model in completed or is_complete(model):
                completed.add(model)
                continue
            if model in external_models:
                queue.append(model)
                continue
            if direct_work_remaining and model in deferred_models:
                queue.append(model)
                continue
            attempts[model] += 1
            gpu = idle_gpus.pop(0)
            workers[gpu] = launch(model, gpu, attempts[model])

        write_status(queue, workers, attempts, completed, failed)
        time.sleep(POLL_SECONDS)

    write_status(queue, workers, attempts, completed, failed)
    (STATE_ROOT / "scheduler.completed").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
