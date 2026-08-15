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
from typing import Any, Mapping, Sequence

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
from .equal_call_single_model import (
    aggregate_equal_call_answers,
    build_independent_solution_prompt,
    build_self_revision_prompt,
    parse_equal_call_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-free equal-call single-model controls for C3"
    )
    parser.add_argument("--c3-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--method", choices=("self_consistency", "self_revision"), required=True
    )
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-questions", type=int, default=0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return value


def _load_questions(run_root: Path) -> list[FalsificationQuestion]:
    path = run_root / "development_observables" / "questions.jsonl"
    questions: list[FalsificationQuestion] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"Equal-call generation input contains labels: {sorted(leaked)}"
                )
            questions.append(
                FalsificationQuestion(
                    question_id=str(row["question_id"]),
                    dataset=str(row["dataset"]),
                    environment=str(row["environment"]),
                    question=str(row["question"]),
                    options=tuple(str(value) for value in row["options"]),
                    option_labels=tuple(str(value) for value in row["option_labels"]),
                )
            )
    if len({row.question_id for row in questions}) != len(questions):
        raise ValueError("Equal-call questions contain duplicate IDs")
    return sorted(questions, key=lambda row: row.question_id)


def _stratified_smoke_questions(
    questions: Sequence[FalsificationQuestion], count: int
) -> list[FalsificationQuestion]:
    by_dataset: dict[str, list[FalsificationQuestion]] = defaultdict(list)
    for question in questions:
        by_dataset[question.dataset].append(question)
    selected: list[FalsificationQuestion] = []
    position = 0
    while len(selected) < count:
        added = False
        for dataset in sorted(by_dataset):
            rows = by_dataset[dataset]
            if position < len(rows):
                selected.append(rows[position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    if len(selected) != count:
        raise RuntimeError("Equal-call smoke selection exceeds available questions")
    return selected


def _validate_protocol(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
) -> None:
    if int(baseline_config.get("protocol_version", -1)) != 1:
        raise ValueError("Unknown equal-call baseline protocol")
    if Path(str(baseline_config.get("c3_config", ""))).resolve() != (
        c3_config_path.resolve()
    ):
        raise PermissionError("Equal-call protocol names a different C3 configuration")
    if baseline_config.get("c3_config_sha256") != sha256_file(c3_config_path):
        raise PermissionError("Equal-call protocol is not bound to this C3 configuration")
    policy = baseline_config.get("data_policy", {})
    if (
        policy.get("generation_reads_labels") is not False
        or policy.get("model_pool_equals_prefrozen_c3_generator_checker_pool") is not True
        or policy.get("development_accuracy_used_for_model_selection") is not False
        or policy.get("certificate_or_check_outputs_used_for_model_selection") is not False
        or policy.get("target_labels_control_generation_or_aggregation") is not False
    ):
        raise PermissionError("Equal-call protocol lacks the frozen label-free boundary")
    parity_multiplier = (
        2
        if str(c3_config.get("check_generation", {}).get("prompt_version", ""))
        in {
            "blind_counterfactual_parity_v4",
            "hardened_blind_counterfactual_parity_v5",
            "blind_isolated_trace_audit_v7",
            "commitment_conditioned_proof_audit_v8",
        }
        else 1
    )
    expected_calls = len(c3_config["experts"]) + len(
        c3_config["certificate_models"]
    ) + parity_multiplier * sum(
        checker != generator
        for generator in c3_config["certificate_models"]
        for checker in c3_config["checker_models"]
    )
    frozen_pool = {str(value) for value in c3_config["certificate_models"]}.union(
        str(value) for value in c3_config["checker_models"]
    )
    if {str(value) for value in baseline_config.get("models", ())} != frozen_pool:
        raise PermissionError("Equal-call model pool differs from the prefrozen C3 pool")
    if int(baseline_config.get("calls_per_question", -1)) != expected_calls:
        raise ValueError("Equal-call budget does not match the C3 call budget")
    methods = baseline_config.get("methods", {})
    if int(methods.get("self_consistency", {}).get("samples", -1)) != expected_calls:
        raise ValueError("Self-consistency does not use the exact C3 call budget")
    revision = methods.get("self_revision", {})
    initial = int(revision.get("initial_samples", -1))
    revisions = int(revision.get("revisions_per_initial", -1))
    if initial <= 0 or revisions <= 0 or initial * (1 + revisions) != expected_calls:
        raise ValueError("Self-revision does not use the exact C3 call budget")


def _model_args(
    c3_config: Mapping[str, Any], baseline_config: Mapping[str, Any]
) -> SimpleNamespace:
    generation = baseline_config["generation"]
    return SimpleNamespace(
        models_dir=Path(str(c3_config["models_dir"])),
        backend=str(generation["backend"]),
        max_model_len=int(generation["max_model_len"]),
        attn_implementation="eager",
        gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
        trust_remote_code=bool(generation["trust_remote_code"]),
        dtype=str(generation["dtype"]),
    )


def _sampling_params(
    baseline_config: Mapping[str, Any], *, samples: int, phase: str
) -> Any:
    generation = baseline_config["generation"]
    if str(generation["backend"]) != "vllm":
        raise ValueError("Equal-call multi-sample controls currently require vLLM")
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    guided_decoding = None
    if bool(generation.get("guided_regex", False)):
        from vllm.sampling_params import GuidedDecodingParams

        guided_decoding = GuidedDecodingParams(
            regex=r"REASON: [^\n]+\nFINAL: [A-Z]",
            disable_fallback=True,
        )
    seed_offset = 0 if phase == "initial" else 1_000_003
    return SamplingParams(
        n=samples,
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(baseline_config["seed"]) + seed_offset,
        guided_decoding=guided_decoding,
    )


def _prepare_prompt(
    llm: Any, raw_prompt: str, max_input_tokens: int
) -> tuple[str, bool, int | None]:
    prompt = apply_chat_template(llm, raw_prompt)
    return truncate_prompt_if_needed(llm, prompt, max_input_tokens)


def _sample_rows(
    llm: Any,
    questions: Sequence[FalsificationQuestion],
    baseline_config: Mapping[str, Any],
    *,
    phase: str,
    samples: int,
    parent_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    generation = baseline_config["generation"]
    max_input_tokens = (
        int(generation["max_model_len"])
        - int(generation["max_new_tokens"])
        - 8
    )
    tasks: list[
        tuple[FalsificationQuestion, int | None, str, bool, int | None]
    ] = []
    if phase == "initial":
        for question in questions:
            prompt, truncated, token_count = _prepare_prompt(
                llm, build_independent_solution_prompt(question), max_input_tokens
            )
            tasks.append((question, None, prompt, truncated, token_count))
    elif phase == "revision":
        if parent_rows is None:
            raise ValueError("Revision sampling requires initial outputs")
        for question in questions:
            parents = sorted(
                parent_rows[question.question_id], key=lambda row: int(row["sample_index"])
            )
            for parent in parents:
                prompt, truncated, token_count = _prepare_prompt(
                    llm,
                    build_self_revision_prompt(question, str(parent["raw_output"])),
                    max_input_tokens,
                )
                tasks.append(
                    (
                        question,
                        int(parent["sample_index"]),
                        prompt,
                        truncated,
                        token_count,
                    )
                )
    else:
        raise ValueError(f"Unknown equal-call phase: {phase}")

    sampling = _sampling_params(
        baseline_config, samples=samples if phase == "initial" else 1, phase=phase
    )
    batch_size = int(generation["batch_size"])
    rows: list[dict[str, Any]] = []
    next_index: dict[str, int] = defaultdict(int)
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        started = time.perf_counter()
        generated = llm.generate([task[2] for task in batch], sampling)
        output_count = sum(len(output.outputs) for output in generated)
        latency = (time.perf_counter() - started) / max(1, output_count)
        for task, request_output in zip(batch, generated, strict=True):
            question, parent_index, prompt, truncated, token_count = task
            expected_outputs = samples if phase == "initial" else 1
            if len(request_output.outputs) != expected_outputs:
                raise RuntimeError("vLLM returned an unexpected equal-call sample count")
            for output in request_output.outputs:
                raw_output = str(output.text)
                answer, reason, parse_error = parse_equal_call_answer(
                    raw_output, question.option_labels
                )
                sample_index = next_index[question.question_id]
                next_index[question.question_id] += 1
                row = {
                    "question_id": question.question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "phase": phase,
                    "sample_index": sample_index,
                    "parent_sample_index": parent_index,
                    "prediction": answer,
                    "reason": reason,
                    "parse_error": parse_error,
                    "raw_output": raw_output,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_was_truncated": bool(truncated),
                    "prompt_token_count": token_count,
                    "model_latency_seconds": latency,
                }
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise AssertionError(
                        f"Equal-call sample emitted labels: {sorted(leaked)}"
                    )
                rows.append(row)
    return rows


def _aggregate_predictions(
    questions: Sequence[FalsificationQuestion],
    method: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final_phase = "initial" if method == "self_consistency" else "revision"
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == final_phase:
            by_question[str(row["question_id"])].append(row)
    result: list[dict[str, Any]] = []
    for question in questions:
        samples = sorted(
            by_question[question.question_id], key=lambda row: int(row["sample_index"])
        )
        prediction, counts = aggregate_equal_call_answers(
            [None if row.get("prediction") is None else str(row["prediction"]) for row in samples],
            question.option_labels,
        )
        result.append(
            {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "environment": question.environment,
                "prediction": prediction,
                "vote_counts": counts,
                "valid_final_samples": sum(row.get("prediction") is not None for row in samples),
                "final_samples": len(samples),
                "tie_breaking": "first_valid_sample_among_plurality_ties",
            }
        )
    return result


def _generate(
    c3_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    model: str,
    method: str,
    questions: Sequence[FalsificationQuestion],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    llm = load_llm(_model_args(c3_config, baseline_config), model)
    try:
        method_config = baseline_config["methods"][method]
        if method == "self_consistency":
            rows = _sample_rows(
                llm,
                questions,
                baseline_config,
                phase="initial",
                samples=int(method_config["samples"]),
            )
        else:
            initial = _sample_rows(
                llm,
                questions,
                baseline_config,
                phase="initial",
                samples=int(method_config["initial_samples"]),
            )
            parents: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in initial:
                parents[str(row["question_id"])].append(row)
            revisions_per_initial = int(method_config["revisions_per_initial"])
            if revisions_per_initial != 1:
                raise ValueError("Protocol v1 supports exactly one revision per initial sample")
            revision = _sample_rows(
                llm,
                questions,
                baseline_config,
                phase="revision",
                samples=1,
                parent_rows=parents,
            )
            rows = initial + revision
        return rows, _aggregate_predictions(questions, method, rows)
    finally:
        del llm
        cleanup_vllm()


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"Equal-call worker must see physical GPU {physical_gpu}; got {visible!r}"
        )


def main() -> None:
    args = parse_args()
    c3_config = _load_yaml(args.c3_config)
    baseline_config = _load_yaml(args.baseline_config)
    _validate_protocol(args.c3_config, c3_config, baseline_config)
    _validate_gpu(args.physical_gpu)
    if args.physical_gpu not in {
        int(value) for value in baseline_config.get("physical_gpus", ())
    }:
        raise PermissionError("Physical GPU is outside the equal-call protocol")
    models = tuple(str(value) for value in baseline_config["models"])
    if args.model not in models:
        raise ValueError(f"Unregistered equal-call model: {args.model}")
    if args.method not in baseline_config["methods"]:
        raise ValueError(f"Unregistered equal-call method: {args.method}")
    run_root = args.run_root or Path(str(c3_config["output_root"]))
    if Path(str(baseline_config["run_root"])) != run_root:
        raise ValueError("Equal-call run root differs from the requested C3 run root")
    questions = _load_questions(run_root)
    if args.smoke_questions:
        if not 1 <= args.smoke_questions <= 8:
            raise ValueError("Equal-call smoke tests must use between one and eight questions")
        questions = _stratified_smoke_questions(questions, args.smoke_questions)
        output_dir = (
            run_root
            / "smoke"
            / "equal_call_single_model"
            / args.method
            / f"{args.model}_n{args.smoke_questions}_gpu{args.physical_gpu}"
        )
        status = "bounded_label_free_equal_call_single_model_smoke"
    else:
        output_dir = run_root / "equal_call_single_model" / args.method / args.model
        status = "completed_label_free_equal_call_single_model"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = (
        run_root
        / "equal_call_attempts"
        / f"{args.method}.{args.model}.{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows, predictions = _generate(
        c3_config, baseline_config, args.model, args.method, questions
    )
    calls_per_question = int(baseline_config["calls_per_question"])
    if len(rows) != len(questions) * calls_per_question:
        raise RuntimeError("Equal-call generation did not consume its exact call budget")
    if len(predictions) != len(questions):
        raise RuntimeError("Equal-call aggregation lacks exact question coverage")
    sample_path = partial / "samples.jsonl"
    prediction_path = partial / "predictions.jsonl"
    write_jsonl(sample_path, rows)
    write_jsonl(prediction_path, predictions)
    question_path = run_root / "development_observables" / "questions.jsonl"
    write_json(
        partial / "manifest.json",
        {
            "status": status,
            "protocol_version": int(baseline_config["protocol_version"]),
            "model": args.model,
            "method": args.method,
            "physical_gpu": args.physical_gpu,
            "questions": len(questions),
            "calls_per_question": calls_per_question,
            "actual_model_calls": len(rows),
            "samples": len(rows),
            "parsed_samples": sum(row["parse_error"] is None for row in rows),
            "final_samples": sum(
                row["phase"] == ("initial" if args.method == "self_consistency" else "revision")
                for row in rows
            ),
            "parsed_final_samples": sum(
                row["phase"] == ("initial" if args.method == "self_consistency" else "revision")
                and row["parse_error"] is None
                for row in rows
            ),
            "truncated_prompts": sum(bool(row["prompt_was_truncated"]) for row in rows),
            "labels_read": False,
            "question_sha256": sha256_file(question_path),
            "sample_sha256": sha256_file(sample_path),
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(args.c3_config),
            "baseline_config_sha256": sha256_file(args.baseline_config),
            "independent_prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(build_independent_solution_prompt).encode("utf-8")
            ).hexdigest(),
            "revision_prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(build_self_revision_prompt).encode("utf-8")
            ).hexdigest(),
            "parser_sha256": hashlib.sha256(
                inspect.getsource(parse_equal_call_answer).encode("utf-8")
            ).hexdigest(),
            "aggregator_sha256": hashlib.sha256(
                inspect.getsource(aggregate_equal_call_answers).encode("utf-8")
            ).hexdigest(),
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(baseline_config["seed"]),
                [args.c3_config, args.baseline_config, question_path],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed equal-call single-model control: {output_dir}")


if __name__ == "__main__":
    main()
