from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from bench_coe.run_official_model_benchmarks import (
    apply_chat_template,
    cleanup_vllm,
    import_vllm_objects,
    load_llm,
    truncate_prompt_if_needed,
)

from .artifacts import environment_manifest, sha256_file, write_json, write_jsonl
from .blind_falsification_jury import (
    FORBIDDEN_AUDIT_KEYS,
    FalsificationQuestion,
    build_falsification_prompt,
    parse_audit_output,
)


_REASON_RE = re.compile(
    r"^\s*REASON\s*:\s*(\S.*?)\s*$", re.IGNORECASE | re.MULTILINE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate label-free BFJ option audits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-questions", type=int, default=0)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("BFJ configuration must be a mapping")
    return value


def _load_questions(run_root: Path) -> list[FalsificationQuestion]:
    path = run_root / "development_observables" / "questions.jsonl"
    rows: list[FalsificationQuestion] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(value)
            if leaked:
                raise PermissionError(f"BFJ generation input contains labels: {sorted(leaked)}")
            rows.append(
                FalsificationQuestion(
                    question_id=str(value["question_id"]),
                    dataset=str(value["dataset"]),
                    environment=str(value["environment"]),
                    question=str(value["question"]),
                    options=tuple(str(item) for item in value["options"]),
                    option_labels=tuple(str(item) for item in value["option_labels"]),
                )
            )
    if len({row.question_id for row in rows}) != len(rows):
        raise ValueError("BFJ observable questions contain duplicate IDs")
    return sorted(rows, key=lambda row: row.question_id)


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
    )


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"BFJ worker must see exactly physical GPU {physical_gpu}; got {visible!r}"
        )


def _generate(
    config: dict[str, Any],
    auditor: str,
    questions: list[FalsificationQuestion],
) -> list[dict[str, Any]]:
    llm = load_llm(_model_args(config), auditor)
    try:
        generation = config["generation"]
        max_input_tokens = (
            int(generation["max_model_len"])
            - int(generation["max_new_tokens"])
            - 8
        )
        tasks: list[tuple[FalsificationQuestion, str, str, bool, int | None]] = []
        for question in questions:
            for candidate in question.option_labels:
                prompt = apply_chat_template(
                    llm, build_falsification_prompt(question, candidate)
                )
                prompt, truncated, token_count = truncate_prompt_if_needed(
                    llm, prompt, max_input_tokens
                )
                tasks.append((question, candidate, prompt, truncated, token_count))
        sampling = _sampling_params(config)
        batch_size = int(generation["batch_size"])
        result: list[dict[str, Any]] = []
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            started = time.perf_counter()
            generated = llm.generate([row[2] for row in batch], sampling)
            per_row_latency = (time.perf_counter() - started) / max(1, len(batch))
            for task, output in zip(batch, generated, strict=True):
                question, candidate, prompt, truncated, token_count = task
                raw_output = str(output.outputs[0].text)
                verdict, confidence, alternative, parse_error = parse_audit_output(
                    raw_output, question.option_labels
                )
                reason_match = _REASON_RE.search(raw_output)
                row = {
                    "question_id": question.question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "auditor_id": auditor,
                    "candidate": candidate,
                    "verdict": verdict,
                    "confidence": confidence,
                    "alternative": alternative,
                    "reason": reason_match.group(1).strip() if reason_match else None,
                    "parse_error": parse_error,
                    "raw_output": raw_output,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_was_truncated": bool(truncated),
                    "prompt_token_count": token_count,
                    "model_latency_seconds": per_row_latency,
                }
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise AssertionError(f"BFJ audit emitted labels: {sorted(leaked)}")
                result.append(row)
        return result
    finally:
        del llm
        cleanup_vllm()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    if args.auditor not in {str(value) for value in config["auditors"]}:
        raise ValueError(f"Unregistered BFJ auditor: {args.auditor}")
    _validate_gpu(args.physical_gpu)
    run_root = args.run_root or Path(str(config["output_root"]))
    questions = _load_questions(run_root)
    if args.smoke_questions:
        if not 1 <= args.smoke_questions <= 8:
            raise ValueError("BFJ smoke tests must use between one and eight questions")
        questions = questions[: args.smoke_questions]
        output_dir = (
            run_root
            / "smoke"
            / f"{args.auditor}_n{args.smoke_questions}_gpu{args.physical_gpu}"
        )
        status = "bounded_label_free_bfj_smoke"
    else:
        output_dir = run_root / "audits" / args.auditor
        status = "completed_label_free_bfj_audits"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = run_root / "audit_attempts" / f"{args.auditor}.{os.getpid()}"
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows = _generate(config, args.auditor, questions)
    expected = sum(len(question.option_labels) for question in questions)
    if len(rows) != expected:
        raise RuntimeError("BFJ audit generation did not cover every question/option")
    output_path = partial / "observations.jsonl"
    write_jsonl(output_path, rows)
    question_path = run_root / "development_observables" / "questions.jsonl"
    write_json(
        partial / "audit_manifest.json",
        {
            "status": status,
            "auditor": args.auditor,
            "physical_gpu": args.physical_gpu,
            "questions": len(questions),
            "option_audits": len(rows),
            "parsed_audits": sum(row["parse_error"] is None for row in rows),
            "truncated_prompts": sum(bool(row["prompt_was_truncated"]) for row in rows),
            "prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(build_falsification_prompt).encode("utf-8")
            ).hexdigest(),
            "parser_sha256": hashlib.sha256(
                inspect.getsource(parse_audit_output).encode("utf-8")
            ).hexdigest(),
            "observation_sha256": sha256_file(output_path),
            "question_sha256": sha256_file(question_path),
            "labels_read": False,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(config["generation"]["seed"]),
                [args.config, question_path],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed BFJ audits: {output_dir}")


if __name__ == "__main__":
    main()
