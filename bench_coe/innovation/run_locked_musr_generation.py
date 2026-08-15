from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bench_coe.run_official_model_benchmarks import (
    apply_chat_template,
    cleanup_vllm,
    import_vllm_objects,
    load_llm,
    truncate_prompt_if_needed,
)

from .artifacts import (
    environment_manifest,
    sha256_file,
    validate_test_receipt,
    write_json,
    write_jsonl,
)
from .locked_protocol import (
    FORBIDDEN_TARGET_KEYS,
    build_musr_prompt,
    extract_musr_choice,
    load_protocol,
    load_question_observables,
    model_identity_manifest,
    validate_preregistration,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen label-free MuSR expert outputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-model")
    parser.add_argument("--smoke-questions", type=int, default=2)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _model_args(config: dict[str, Any]) -> SimpleNamespace:
    generation = config["generation"]
    return SimpleNamespace(
        models_dir=Path(str(config["models_dir"])),
        backend=str(generation["backend"]),
        max_model_len=int(generation["max_model_len"]),
        attn_implementation="eager",
        gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
        trust_remote_code=bool(generation["trust_remote_code"]),
        dtype=str(generation["dtype"]),
    )


def _sampling_params(config: dict[str, Any]) -> Any:
    generation = config["generation"]
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    return SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        stop=["\nNarrative:", "\nQuestion:"],
    )


def _validate_worker_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu) or physical_gpu not in {0, 1, 2, 3}:
        raise RuntimeError(f"Worker must see exactly physical GPU {physical_gpu}; got {visible!r}")


def _generate_rows(config: dict[str, Any], llm: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sampling_params = _sampling_params(config)
    generation = config["generation"]
    max_input_tokens = (
        int(generation["max_model_len"])
        - int(generation["max_new_tokens"])
        - 8
    )
    output_rows: list[dict[str, Any]] = []
    batch_size = int(generation["batch_size"])
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts: list[str] = []
        truncation: list[tuple[bool, int | None]] = []
        for row in batch_rows:
            prompt = apply_chat_template(llm, build_musr_prompt(row))
            prompt, was_truncated, token_count = truncate_prompt_if_needed(
                llm, prompt, max_input_tokens
            )
            prompts.append(prompt)
            truncation.append((was_truncated, token_count))
        batch_started = time.perf_counter()
        generated = llm.generate(prompts, sampling_params)
        per_row_latency = (time.perf_counter() - batch_started) / max(1, len(batch_rows))
        for row, output, (was_truncated, token_count) in zip(
            batch_rows, generated, truncation, strict=True
        ):
            response = str(output.outputs[0].text)
            prediction = extract_musr_choice(response, row["option_labels"])
            result = {
                "id": row["id"],
                "task": row["task"],
                "domain": row["domain"],
                "question": row["question"],
                "input": f"{row['narrative']}\n\n{row['question']}",
                "options": row["options"],
                "prediction": prediction,
                "response": response,
                "model_error": None if prediction is not None else "unparseable_choice",
                "model_latency_seconds": per_row_latency,
                "prompt_was_truncated": bool(was_truncated),
                "prompt_token_count": token_count,
            }
            leaked = FORBIDDEN_TARGET_KEYS.intersection(result)
            if leaked:
                raise AssertionError(f"Generation emitted target label fields: {sorted(leaked)}")
            output_rows.append(result)
    return output_rows


def _generate_one_model(
    config_path: Path,
    run_root: Path,
    model_name: str,
    physical_gpu: int,
) -> dict[str, Any]:
    config = load_protocol(config_path)
    validate_protocol(config, verify_files=False)
    validate_test_receipt(Path(str(config["test_receipt"])), config_path)
    prereg = validate_preregistration(config_path, run_root)
    _validate_worker_gpu(physical_gpu)
    if model_name not in set(str(value) for value in config["experts"]):
        raise ValueError(f"Unregistered expert: {model_name}")
    rows = load_question_observables(run_root, int(config["target"]["expected_questions"]))
    final_dir = run_root / "target_observables" / model_name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    partial_dir = run_root / "generation_attempts" / f"{model_name}.{os.getpid()}"
    partial_dir.mkdir(parents=True, exist_ok=False)

    args = _model_args(config)
    model_path = args.models_dir / model_name
    llm = load_llm(args, model_name)
    started = time.time()
    try:
        generation = config["generation"]
        output_rows = _generate_rows(config, llm, rows)
        if len(output_rows) != len(rows):
            raise RuntimeError("Model generation did not cover every frozen target question")
        prediction_path = partial_dir / "observables.jsonl"
        write_jsonl(prediction_path, output_rows)
        manifest = {
            "status": "completed_label_free_generation",
            "model": model_name,
            "physical_gpu": physical_gpu,
            "questions": len(output_rows),
            "valid_predictions": sum(row["prediction"] is not None for row in output_rows),
            "truncated_prompts": sum(bool(row["prompt_was_truncated"]) for row in output_rows),
            "prediction_sha256": sha256_file(prediction_path),
            "preregistration_sha256": sha256_file(run_root / "preregistration.json"),
            "target_labels_opened": False,
            "model_identity": model_identity_manifest(model_path),
            "environment": environment_manifest(
                sys.argv,
                int(generation["seed"]),
                [config_path, run_root / "preregistration.json", run_root / "question_observables"],
            ),
            "started_unix": started,
            "finished_unix": time.time(),
        }
        write_json(partial_dir / "model_manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_dir, final_dir)
        return manifest
    finally:
        del llm
        cleanup_vllm()


def run_smoke(args: argparse.Namespace) -> None:
    config = load_protocol(args.config)
    validate_protocol(config, verify_files=False)
    validate_test_receipt(Path(str(config["test_receipt"])), args.config)
    run_root = args.run_root or Path(str(config["output_root"]))
    validate_preregistration(args.config, run_root)
    _validate_worker_gpu(args.physical_gpu)
    if args.smoke_model not in set(str(value) for value in config["experts"]):
        raise ValueError("Smoke model must belong to the frozen expert pool")
    if not 1 <= args.smoke_questions <= 8:
        raise ValueError("Bounded smoke test must use between one and eight questions")
    rows = load_question_observables(
        run_root, int(config["target"]["expected_questions"])
    )[: args.smoke_questions]
    smoke_dir = run_root / "smoke" / f"{args.smoke_model}_n{args.smoke_questions}_gpu{args.physical_gpu}"
    if smoke_dir.exists():
        raise FileExistsError(smoke_dir)
    partial_dir = run_root / "smoke_attempts" / f"{args.smoke_model}.{os.getpid()}"
    partial_dir.mkdir(parents=True, exist_ok=False)
    llm = load_llm(_model_args(config), args.smoke_model)
    try:
        output_rows = _generate_rows(config, llm, rows)
        output_path = partial_dir / "observables.jsonl"
        write_jsonl(output_path, output_rows)
        write_json(
            partial_dir / "smoke_manifest.json",
            {
                "status": "bounded_label_free_smoke_passed",
                "model": args.smoke_model,
                "physical_gpu": args.physical_gpu,
                "questions": len(output_rows),
                "prediction_sha256": sha256_file(output_path),
                "target_labels_opened": False,
            },
        )
        smoke_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_dir, smoke_dir)
    finally:
        del llm
        cleanup_vllm()
    print(f"Bounded smoke passed: {smoke_dir}")


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise ValueError("Worker paths are required")
    payload = json.loads(args.worker_input.read_text(encoding="utf-8"))
    result: dict[str, Any]
    try:
        manifest = _generate_one_model(
            Path(payload["config"]),
            Path(payload["run_root"]),
            str(payload["model"]),
            int(payload["physical_gpu"]),
        )
        result = {"status": "completed", "model": payload["model"], "manifest": manifest}
    except Exception as exc:
        result = {
            "status": "failed",
            "model": payload.get("model"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(result["traceback"], flush=True)
    write_json(args.worker_output, result)
    if result["status"] != "completed":
        raise SystemExit(1)


def _gpu_memory_used() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",", 1)]
        result[int(index)] = int(used)
    return result


def _wait_for_front_gpus(gpus: list[int]) -> None:
    while True:
        used = _gpu_memory_used()
        busy = {gpu: used.get(gpu, -1) for gpu in gpus if used.get(gpu, -1) >= 100}
        if not busy:
            return
        print(f"Waiting for registered GPUs to become idle: {busy}", flush=True)
        time.sleep(60)


def _completed_expert(run_root: Path, expert: str, expected: int) -> bool:
    model_dir = run_root / "target_observables" / expert
    manifest_path = model_dir / "model_manifest.json"
    predictions = model_dir / "observables.jsonl"
    if not manifest_path.exists() or not predictions.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "completed_label_free_generation"
        and manifest.get("model") == expert
        and int(manifest.get("questions", -1)) == expected
        and manifest.get("prediction_sha256") == sha256_file(predictions)
        and manifest.get("target_labels_opened") is False
        and manifest.get("preregistration_sha256")
        == sha256_file(run_root / "preregistration.json")
    )


def _finalize_observable_manifest(config: dict[str, Any], run_root: Path) -> Path:
    experts = tuple(sorted(str(value) for value in config["experts"]))
    hashes: dict[str, str] = {}
    model_manifests: dict[str, str] = {}
    expected_ids = {
        str(row["id"])
        for row in load_question_observables(
            run_root, int(config["target"]["expected_questions"])
        )
    }
    for expert in experts:
        model_dir = run_root / "target_observables" / expert
        predictions = model_dir / "observables.jsonl"
        manifest = model_dir / "model_manifest.json"
        observed_ids: set[str] = set()
        with predictions.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                leaked = FORBIDDEN_TARGET_KEYS.intersection(row)
                if leaked:
                    raise PermissionError(
                        f"Expert cache contains target label fields {sorted(leaked)}: {expert}"
                    )
                raw_id = str(row["id"])
                if raw_id in observed_ids:
                    raise ValueError(f"Duplicate target question for expert {expert}: {raw_id}")
                observed_ids.add(raw_id)
        if observed_ids != expected_ids:
            raise RuntimeError(f"Expert cache IDs do not match frozen questions: {expert}")
        hashes[str(predictions)] = sha256_file(predictions)
        model_manifests[str(manifest)] = sha256_file(manifest)
    payload = {
        "dataset": "musr",
        "split": "test",
        "modality": "language",
        "role": "target_observables_only",
        "expert_ids": list(experts),
        "questions": int(config["target"]["expected_questions"]),
        "output_observable_hashes": hashes,
        "model_manifest_hashes": model_manifests,
        "question_observables_sha256": sha256_file(
            run_root / "question_observables" / "questions.jsonl"
        ),
        "forbidden_fields": sorted(FORBIDDEN_TARGET_KEYS),
        "target_labels_opened": False,
    }
    path = run_root / "target_observables" / "observable_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, payload)
    return path


def run_parent(args: argparse.Namespace) -> None:
    config = load_protocol(args.config)
    validate_protocol(config, verify_files=False)
    run_root = args.run_root or Path(str(config["output_root"]))
    validate_test_receipt(Path(str(config["test_receipt"])), args.config)
    validate_preregistration(args.config, run_root)
    gpus = [int(value) for value in config["physical_gpus"]]
    expected = int(config["target"]["expected_questions"])
    experts = [str(value) for value in config["experts"]]
    pending = [expert for expert in experts if not _completed_expert(run_root, expert, expected)]
    if not pending:
        print("Every registered expert output is already complete.")
    workers_dir = run_root / "generation_workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    wave_results: list[str] = []
    for wave_start in range(0, len(pending), len(gpus)):
        wave = pending[wave_start : wave_start + len(gpus)]
        _wait_for_front_gpus(gpus[: len(wave)])
        processes: list[tuple[subprocess.Popen[Any], str, Any, Path]] = []
        for gpu, expert in zip(gpus[: len(wave)], wave, strict=True):
            attempt = 1
            while (workers_dir / f"{expert}.attempt{attempt}.input.json").exists():
                attempt += 1
            worker_input = workers_dir / f"{expert}.attempt{attempt}.input.json"
            worker_output = workers_dir / f"{expert}.attempt{attempt}.output.json"
            log_path = workers_dir / f"{expert}.attempt{attempt}.log"
            write_json(
                worker_input,
                {
                    "config": str(args.config),
                    "run_root": str(run_root),
                    "model": expert,
                    "physical_gpu": gpu,
                },
            )
            command = [
                sys.executable,
                "-m",
                "bench_coe.innovation.run_locked_musr_generation",
                "--config",
                str(args.config),
                "--worker-input",
                str(worker_input),
                "--worker-output",
                str(worker_output),
            ]
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "PYTHONHASHSEED": str(config["generation"]["seed"]),
                }
            )
            cache = workers_dir / "cache" / expert
            cache.mkdir(parents=True, exist_ok=True)
            env["VLLM_CACHE_ROOT"] = str(cache / "vllm")
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
            log_handle = log_path.open("a", encoding="utf-8")
            print(f"GPU {gpu}: generating {expert}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=str(Path.cwd()),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, expert, log_handle, worker_output))
        failed: list[str] = []
        for process, expert, log_handle, worker_output in processes:
            return_code = process.wait()
            log_handle.close()
            result = (
                json.loads(worker_output.read_text(encoding="utf-8"))
                if worker_output.exists()
                else {"status": "failed", "error": f"worker exited {return_code} without result"}
            )
            all_results[expert] = result
            if return_code or result.get("status") != "completed":
                failed.append(expert)
            print(f"{expert}: {result.get('status')}", flush=True)
        wave_number = wave_start // len(gpus) + 1
        wave_path = workers_dir / f"generation_results_wave_{wave_number}.json"
        write_json(wave_path, all_results)
        wave_results.append(str(wave_path))
        if failed:
            raise RuntimeError(f"Label-free generation failed for: {failed}")
    incomplete = [expert for expert in experts if not _completed_expert(run_root, expert, expected)]
    if incomplete:
        raise RuntimeError(f"Registered expert outputs remain incomplete: {incomplete}")
    manifest = run_root / "target_observables" / "observable_manifest.json"
    if not manifest.exists():
        manifest = _finalize_observable_manifest(config, run_root)
    final_results = workers_dir / "generation_results.json"
    if not final_results.exists():
        write_json(
            final_results,
            {"status": "all_experts_completed", "results": all_results, "wave_manifests": wave_results},
        )
    print(f"Sealed label-free expert cache: {manifest}")


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args()
    if args.worker_input is not None:
        run_worker(args)
    elif args.smoke_model is not None:
        run_smoke(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
