from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from collections import defaultdict
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
from .blind_falsification_jury import FORBIDDEN_AUDIT_KEYS, FalsificationQuestion
from .sealed_diagnostic_bijection import (
    CandidatePairAssignment,
    assign_candidate_pairs,
    build_diagnostic_probe_prompt,
    parse_diagnostic_probe_output,
    present_diagnostic_probe,
    presented_left_authored_outcome,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label-free sealed diagnostic bijections"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-questions", type=int, default=0)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("SDB configuration must be a mapping")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def _input_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(str(config["input_observables_root"]))
    return (
        root / "development_observables" / "questions.jsonl",
        root / "development_observables" / "base_predictions.jsonl",
        root / "development_observables" / "observable_manifest.json",
    )


def _load_questions(path: Path) -> list[FalsificationQuestion]:
    questions: list[FalsificationQuestion] = []
    for value in _read_jsonl(path):
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(value)
        if leaked:
            raise PermissionError(f"SDB question input contains labels: {sorted(leaked)}")
        questions.append(
            FalsificationQuestion(
                question_id=str(value["question_id"]),
                dataset=str(value["dataset"]),
                environment=str(value["environment"]),
                question=str(value["question"]),
                options=tuple(str(item) for item in value["options"]),
                option_labels=tuple(str(item) for item in value["option_labels"]),
            )
        )
    if len({row.question_id for row in questions}) != len(questions):
        raise ValueError("SDB observable questions contain duplicate IDs")
    return sorted(questions, key=lambda row: row.question_id)


def _load_base_observables(
    path: Path,
    questions: list[FalsificationQuestion],
    expert_order: tuple[str, ...],
) -> tuple[dict[str, dict[str, str | None]], dict[str, dict[str, str]]]:
    expected_questions = {row.question_id for row in questions}
    answers: dict[str, dict[str, str | None]] = defaultdict(dict)
    responses: dict[str, dict[str, str]] = defaultdict(dict)
    for row in _read_jsonl(path):
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(f"SDB Stage-0 input contains labels: {sorted(leaked)}")
        question_id = str(row["question_id"])
        expert_id = str(row["expert_id"])
        if question_id not in expected_questions or expert_id not in expert_order:
            continue
        if expert_id in answers[question_id]:
            raise ValueError("Duplicate SDB Stage-0 prediction")
        raw_prediction = row.get("prediction")
        answers[question_id][expert_id] = (
            None if raw_prediction is None else str(raw_prediction)
        )
        responses[question_id][expert_id] = str(row.get("response", ""))
    expected_experts = set(expert_order)
    for question in questions:
        if set(answers[question.question_id]) != expected_experts:
            raise RuntimeError(
                f"SDB lacks a complete expert pool for {question.question_id}"
            )
        if set(responses[question.question_id]) != expected_experts:
            raise RuntimeError(
                f"SDB lacks complete private responses for {question.question_id}"
            )
    return dict(answers), dict(responses)


def _assignments_by_question(
    questions: list[FalsificationQuestion],
    author_ids: tuple[str, ...],
    base_answers: dict[str, dict[str, str | None]],
    expert_order: tuple[str, ...],
) -> dict[str, dict[str, CandidatePairAssignment]]:
    return {
        question.question_id: assign_candidate_pairs(
            question,
            author_ids,
            base_answers[question.question_id],
            expert_order,
        )
        for question in questions
    }


def _stratified_smoke_questions(
    questions: list[FalsificationQuestion],
    count: int,
    seed: int,
    author: str,
) -> list[FalsificationQuestion]:
    buckets: dict[tuple[str, int], list[FalsificationQuestion]] = defaultdict(list)
    for question in questions:
        side = presented_left_authored_outcome(seed, question.question_id, author)
        buckets[(question.dataset, side)].append(question)
    selected: list[FalsificationQuestion] = []
    position = 0
    while len(selected) < count:
        added = False
        for key in sorted(buckets):
            rows = buckets[key]
            if position < len(rows):
                selected.append(rows[position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    if len(selected) != count:
        raise RuntimeError("SDB smoke selection exceeds available dataset/side strata")
    return selected


def _model_args(config: dict[str, Any]) -> SimpleNamespace:
    generation = config["probe_generation"]
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
    generation = config["probe_generation"]
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    guided_decoding = None
    if bool(generation.get("guided_regex", False)):
        from vllm.sampling_params import GuidedDecodingParams

        nonabstaining = (
            r"PROBE: [^\n]+\n"
            r"OUTCOME_1: [^\n]+\n"
            r"OUTCOME_2: [^\n]+\n"
            r"MAP_OUTCOME_1: [A-Z]\n"
            r"MAP_OUTCOME_2: [A-Z]\n"
            r"BRIDGE_1: [^\n]+\n"
            r"BRIDGE_2: [^\n]+\n"
            r"CONFIDENCE: (?:100|[0-9]{1,2})"
        )
        abstaining = (
            r"PROBE: NONE\n"
            r"OUTCOME_1: NONE\n"
            r"OUTCOME_2: NONE\n"
            r"MAP_OUTCOME_1: NONE\n"
            r"MAP_OUTCOME_2: NONE\n"
            r"BRIDGE_1: NONE\n"
            r"BRIDGE_2: NONE\n"
            r"CONFIDENCE: (?:50|[0-4]?[0-9])"
        )
        guided_decoding = GuidedDecodingParams(
            regex=rf"(?:{nonabstaining}|{abstaining})",
            disable_fallback=True,
        )
    return SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        guided_decoding=guided_decoding,
    )


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"SDB probe worker must see physical GPU {physical_gpu}; got {visible!r}"
        )


def _generate(
    config: dict[str, Any],
    author: str,
    questions: list[FalsificationQuestion],
    assignments: dict[str, dict[str, CandidatePairAssignment]],
    private_responses: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    llm = load_llm(_model_args(config), author)
    try:
        generation = config["probe_generation"]
        max_input_tokens = (
            int(generation["max_model_len"])
            - int(generation["max_new_tokens"])
            - 8
        )
        tasks: list[tuple[FalsificationQuestion, CandidatePairAssignment, str, str, bool, int]] = []
        for question in questions:
            assignment = assignments[question.question_id][author]
            raw_prompt = build_diagnostic_probe_prompt(
                question,
                private_responses[question.question_id][author],
                assignment,
            )
            prompt = apply_chat_template(llm, raw_prompt)
            prompt, truncated, token_count = truncate_prompt_if_needed(
                llm, prompt, max_input_tokens
            )
            tasks.append(
                (question, assignment, raw_prompt, prompt, truncated, token_count)
            )

        sampling = _sampling_params(config)
        batch_size = int(generation["batch_size"])
        rows: list[dict[str, Any]] = []
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            started = time.perf_counter()
            generated = llm.generate([task[3] for task in batch], sampling)
            latency = (time.perf_counter() - started) / max(1, len(batch))
            for task, output in zip(batch, generated, strict=True):
                question, assignment, raw_prompt, prompt, truncated, token_count = task
                raw_output = str(output.outputs[0].text)
                parsed = parse_diagnostic_probe_output(
                    raw_output, assignment, question.question
                )
                left_authored = presented_left_authored_outcome(
                    int(generation["seed"]), question.question_id, author
                )
                presentation = (
                    present_diagnostic_probe(parsed, left_authored)
                    if parsed.parse_error is None and not parsed.abstained
                    else None
                )
                row = {
                    "probe_id": f"{question.question_id}::{author}",
                    "question_id": question.question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "author_id": author,
                    "assignment_first": assignment.first,
                    "assignment_second": assignment.second,
                    "author_stage0_prediction": assignment.author_answer,
                    "assignment_reason": assignment.reason,
                    "probe": parsed.probe,
                    "authored_outcome_1": parsed.outcome_1,
                    "authored_outcome_2": parsed.outcome_2,
                    "sealed_map_outcome_1": parsed.map_outcome_1,
                    "sealed_map_outcome_2": parsed.map_outcome_2,
                    "sealed_bridge_1": parsed.bridge_1,
                    "sealed_bridge_2": parsed.bridge_2,
                    "confidence": parsed.confidence,
                    "parse_error": parsed.parse_error,
                    "abstained": parsed.abstained,
                    "presented_left_authored_outcome": (
                        presentation.left_authored_outcome if presentation else left_authored
                    ),
                    "presented_left_text": presentation.left_text if presentation else None,
                    "presented_right_text": presentation.right_text if presentation else None,
                    "sealed_left_candidate": (
                        presentation.left_candidate if presentation else None
                    ),
                    "sealed_right_candidate": (
                        presentation.right_candidate if presentation else None
                    ),
                    "post_commit_permutation_applied": (
                        presentation.post_commit_permutation_applied
                        if presentation
                        else left_authored == 2
                    ),
                    "mapping_was_sealed_from_checkers": True,
                    "original_task_was_sealed_from_checkers": True,
                    "raw_output": raw_output,
                    "raw_prompt_sha256": hashlib.sha256(
                        raw_prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_was_truncated": bool(truncated),
                    "prompt_token_count": token_count,
                    "model_latency_seconds": latency,
                }
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise AssertionError(f"SDB probe emitted labels: {sorted(leaked)}")
                rows.append(row)
        return rows
    finally:
        del llm
        cleanup_vllm()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    authors = tuple(str(value) for value in config["author_models"])
    experts = tuple(str(value) for value in config["experts"])
    if args.author not in authors:
        raise ValueError(f"Unregistered SDB probe author: {args.author}")
    if not set(authors).issubset(experts):
        raise ValueError("SDB authors must belong to the frozen expert pool")
    _validate_gpu(args.physical_gpu)

    run_root = args.run_root or Path(str(config["output_root"]))
    question_path, base_path, observable_manifest_path = _input_paths(config)
    observable_manifest = json.loads(
        observable_manifest_path.read_text(encoding="utf-8")
    )
    configured_hashes = config.get("input_hashes")
    if not isinstance(configured_hashes, dict):
        raise TypeError("SDB config lacks frozen input hashes")
    question_hash = sha256_file(question_path)
    base_hash = sha256_file(base_path)
    if (
        observable_manifest.get("question_sha256") != question_hash
        or configured_hashes.get("questions_sha256") != question_hash
    ):
        raise PermissionError("SDB observable question hash changed")
    if (
        observable_manifest.get("base_prediction_sha256") != base_hash
        or configured_hashes.get("base_predictions_sha256") != base_hash
    ):
        raise PermissionError("SDB observable base-prediction hash changed")
    questions = _load_questions(question_path)
    base_answers, private_responses = _load_base_observables(
        base_path, questions, experts
    )
    assignments = _assignments_by_question(
        questions, authors, base_answers, experts
    )
    if args.smoke_questions:
        if not 4 <= args.smoke_questions <= 12:
            raise ValueError("SDB probe smoke tests must use between 4 and 12 questions")
        questions = _stratified_smoke_questions(
            questions,
            args.smoke_questions,
            int(config["probe_generation"]["seed"]),
            args.author,
        )
        output_dir = (
            run_root
            / "smoke"
            / "probes"
            / f"{args.author}_n{args.smoke_questions}_gpu{args.physical_gpu}"
        )
        status = "bounded_label_free_sdb_probe_smoke"
    else:
        output_dir = run_root / "probes" / args.author
        status = "completed_label_free_sdb_probes"
    if output_dir.exists():
        raise FileExistsError(output_dir)

    partial = run_root / "probe_attempts" / f"{args.author}.{os.getpid()}"
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows = _generate(
        config, args.author, questions, assignments, private_responses
    )
    if len(rows) != len(questions):
        raise RuntimeError("SDB probe generation did not cover every selected question")
    probe_path = partial / "probes.jsonl"
    write_jsonl(probe_path, rows)
    parsed_rows = [row for row in rows if row["parse_error"] is None]
    nonabstaining = [row for row in parsed_rows if not row["abstained"]]
    generation = config["probe_generation"]
    write_json(
        partial / "probe_manifest.json",
        {
            "status": status,
            "author": args.author,
            "physical_gpu": args.physical_gpu,
            "questions": len(questions),
            "model_calls": len(rows),
            "parsed_probes": len(parsed_rows),
            "abstaining_probes": len(parsed_rows) - len(nonabstaining),
            "nonabstaining_probes": len(nonabstaining),
            "truncated_model_calls": sum(
                bool(row["prompt_was_truncated"]) for row in rows
            ),
            "mapping_bijections": sum(
                {
                    str(row["sealed_map_outcome_1"]),
                    str(row["sealed_map_outcome_2"]),
                }
                == {str(row["assignment_first"]), str(row["assignment_second"])}
                for row in nonabstaining
            ),
            "presented_left_authored_outcome_counts": {
                str(index): sum(
                    row["presented_left_authored_outcome"] == index
                    for row in nonabstaining
                )
                for index in (1, 2)
            },
            "post_commit_permutation": True,
            "mapping_was_sealed_from_checkers": True,
            "original_task_was_sealed_from_checkers": True,
            "author_prompt_reads_only_own_private_stage0_trace": True,
            "pair_assignment_reads_only_label_free_stage0_predictions": True,
            "probe_sha256": sha256_file(probe_path),
            "question_sha256": sha256_file(question_path),
            "base_prediction_sha256": sha256_file(base_path),
            "prompt_version": str(generation["prompt_version"]),
            "parser_version": str(generation["parser_version"]),
            "prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(build_diagnostic_probe_prompt).encode("utf-8")
            ).hexdigest(),
            "parser_sha256": hashlib.sha256(
                inspect.getsource(parse_diagnostic_probe_output).encode("utf-8")
            ).hexdigest(),
            "pair_assignment_sha256": hashlib.sha256(
                inspect.getsource(assign_candidate_pairs).encode("utf-8")
            ).hexdigest(),
            "presentation_sha256": hashlib.sha256(
                inspect.getsource(present_diagnostic_probe).encode("utf-8")
            ).hexdigest(),
            "labels_read": False,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(generation["seed"]),
                [args.config, question_path, base_path, observable_manifest_path],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed SDB probes: {output_dir}")


if __name__ == "__main__":
    main()
