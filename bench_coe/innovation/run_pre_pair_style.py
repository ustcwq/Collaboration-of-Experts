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
from .prepair_style import (
    PAIRWISE_ORIENTATIONS,
    aggregate_order_audited_pre_pair,
    build_pre_pair_pairwise_prompt,
    build_pre_pair_pointwise_prompt,
    candidate_vote_counts,
    parse_pre_pair_pairwise_output,
    parse_pre_pair_pointwise_output,
    rank_candidate_slate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a label-free order-audited PRePair-style C3 control"
    )
    parser.add_argument("--c3-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-questions", type=int, default=0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return value


def pre_pair_call_budget(
    c3_config: Mapping[str, Any], baseline_config: Mapping[str, Any]
) -> tuple[int, int]:
    max_challengers = int(baseline_config["candidate_selection"]["max_challengers"])
    slate_size = 1 + max_challengers
    calls_per_model = slate_size + len(PAIRWISE_ORIENTATIONS) * max_challengers
    total = len(c3_config["experts"]) + len(baseline_config["models"]) * calls_per_model
    return calls_per_model, total


def _validate_protocol(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
) -> None:
    if int(baseline_config.get("protocol_version", -1)) != 1:
        raise ValueError("Unknown PRePair-style baseline protocol")
    if Path(str(baseline_config.get("c3_config", ""))).resolve() != (
        c3_config_path.resolve()
    ):
        raise PermissionError("PRePair-style protocol names a different C3 config")
    if baseline_config.get("c3_config_sha256") != sha256_file(c3_config_path):
        raise PermissionError("PRePair-style protocol is not bound to this C3 config")
    if baseline_config.get("pointwise_prompt_version") != "isolated_candidate_analysis_v1":
        raise ValueError("Unknown PRePair-style pointwise prompt")
    if baseline_config.get("pairwise_prompt_version") != "order_audited_transfer_v1":
        raise ValueError("Unknown PRePair-style pairwise prompt")
    selection = baseline_config.get("candidate_selection", {})
    if (
        selection.get("ranking") != "plurality_then_first_frozen_expert_then_option_order"
        or int(selection.get("max_challengers", -1)) != 2
    ):
        raise PermissionError("PRePair-style candidate selection differs from the frozen protocol")
    if tuple(baseline_config.get("pairwise_orientations", ())) != PAIRWISE_ORIENTATIONS:
        raise PermissionError("PRePair-style orientation protocol differs")
    frozen_pool = {
        str(value) for value in c3_config["certificate_models"]
    }.union(str(value) for value in c3_config["checker_models"])
    models = tuple(str(value) for value in baseline_config.get("models", ()))
    if set(models) != frozen_pool or len(models) != len(frozen_pool):
        raise PermissionError("PRePair-style model pool differs from frozen C3 pool")
    policy = baseline_config.get("data_policy", {})
    if (
        policy.get("generation_reads_labels") is not False
        or policy.get("candidate_ranking_reads_labels") is not False
        or policy.get("model_pool_equals_prefrozen_c3_generator_checker_pool") is not True
        or policy.get("pointwise_candidates_are_isolated") is not True
        or policy.get("target_labels_control_generation_or_aggregation") is not False
    ):
        raise PermissionError("PRePair-style protocol lacks the frozen label firewall")
    calls_per_model, total = pre_pair_call_budget(c3_config, baseline_config)
    if int(baseline_config.get("calls_per_model_per_question", -1)) != calls_per_model:
        raise ValueError("PRePair-style per-model call budget differs")
    if int(baseline_config.get("calls_per_question", -1)) != total:
        raise ValueError("PRePair-style total call budget differs")


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
                    f"PRePair-style question input contains labels: {sorted(leaked)}"
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
        raise ValueError("PRePair-style questions contain duplicate IDs")
    return sorted(questions, key=lambda row: row.question_id)


def _load_base_answers(
    run_root: Path,
    questions: Sequence[FalsificationQuestion],
    experts: Sequence[str],
) -> dict[str, dict[str, str | None]]:
    expected_questions = {question.question_id for question in questions}
    rows: dict[str, dict[str, str | None]] = defaultdict(dict)
    path = run_root / "development_observables" / "base_predictions.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"PRePair-style base input contains labels: {sorted(leaked)}"
                )
            question_id = str(row["question_id"])
            expert = str(row["expert_id"])
            if question_id not in expected_questions or expert not in experts:
                continue
            if expert in rows[question_id]:
                raise ValueError("Duplicate PRePair-style base prediction")
            rows[question_id][expert] = (
                None if row.get("prediction") is None else str(row["prediction"])
            )
    expected_grid = {
        (question.question_id, expert) for question in questions for expert in experts
    }
    actual_grid = {
        (question_id, expert)
        for question_id, by_expert in rows.items()
        for expert in by_expert
    }
    if actual_grid != expected_grid:
        raise RuntimeError("PRePair-style base prediction grid is incomplete")
    return dict(rows)


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
        raise RuntimeError("PRePair-style smoke selection exceeds available questions")
    return selected


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


def _sampling_params(baseline_config: Mapping[str, Any], phase: str) -> Any:
    generation = baseline_config["generation"]
    if str(generation["backend"]) != "vllm":
        raise ValueError("PRePair-style control currently requires vLLM")
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    guided_decoding = None
    if bool(generation.get("guided_regex", False)):
        from vllm.sampling_params import GuidedDecodingParams

        regex = (
            r"ANALYSIS: [^\n]+"
            if phase == "pointwise"
            else r"REASON: [^\n]+\nWINNER: (?:LEFT|RIGHT|TIE)"
        )
        guided_decoding = GuidedDecodingParams(regex=regex, disable_fallback=True)
    return SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(baseline_config["seed"]) + (0 if phase == "pointwise" else 1),
        guided_decoding=guided_decoding,
    )


def _prepare_prompt(
    llm: Any, raw_prompt: str, max_input_tokens: int
) -> tuple[str, bool, int | None]:
    prompt = apply_chat_template(llm, raw_prompt)
    return truncate_prompt_if_needed(llm, prompt, max_input_tokens)


def _generate_rows(
    llm: Any,
    model: str,
    questions: Sequence[FalsificationQuestion],
    slates: Mapping[str, tuple[str, ...]],
    baseline_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation = baseline_config["generation"]
    max_input_tokens = (
        int(generation["max_model_len"])
        - int(generation["max_new_tokens"])
        - 8
    )
    pointwise_tasks: list[
        tuple[FalsificationQuestion, str, str, str, bool, int | None]
    ] = []
    for question in questions:
        for candidate in slates[question.question_id]:
            raw_prompt = build_pre_pair_pointwise_prompt(question, candidate)
            prompt, truncated, token_count = _prepare_prompt(
                llm, raw_prompt, max_input_tokens
            )
            pointwise_tasks.append(
                (question, candidate, raw_prompt, prompt, truncated, token_count)
            )
    pointwise_rows: list[dict[str, Any]] = []
    batch_size = int(generation["batch_size"])
    sampling = _sampling_params(baseline_config, "pointwise")
    for start in range(0, len(pointwise_tasks), batch_size):
        batch = pointwise_tasks[start : start + batch_size]
        started = time.perf_counter()
        generated = llm.generate([task[3] for task in batch], sampling)
        latency = (time.perf_counter() - started) / max(1, len(batch))
        for task, output in zip(batch, generated, strict=True):
            question, candidate, raw_prompt, prompt, truncated, token_count = task
            raw_output = str(output.outputs[0].text)
            analysis, parse_error = parse_pre_pair_pointwise_output(raw_output)
            pointwise_rows.append(
                {
                    "question_id": question.question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "model": model,
                    "candidate": candidate,
                    "analysis": analysis,
                    "parse_error": parse_error,
                    "raw_output": raw_output,
                    "raw_prompt_sha256": hashlib.sha256(
                        raw_prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_was_truncated": bool(truncated),
                    "prompt_token_count": token_count,
                    "model_latency_seconds": latency,
                }
            )

    pointwise_by_key = {
        (str(row["question_id"]), str(row["candidate"])): row
        for row in pointwise_rows
    }
    pairwise_tasks: list[
        tuple[
            FalsificationQuestion,
            str,
            str,
            str,
            str,
            str,
            str,
            bool,
            int | None,
        ]
    ] = []
    for question in questions:
        slate = slates[question.question_id]
        primary = slate[0]
        for challenger in slate[1:]:
            for orientation in PAIRWISE_ORIENTATIONS:
                if orientation == "primary_left":
                    left, right = primary, challenger
                else:
                    left, right = challenger, primary
                left_output = str(
                    pointwise_by_key[(question.question_id, left)]["raw_output"]
                )
                right_output = str(
                    pointwise_by_key[(question.question_id, right)]["raw_output"]
                )
                raw_prompt = build_pre_pair_pairwise_prompt(
                    question, left, right, left_output, right_output
                )
                prompt, truncated, token_count = _prepare_prompt(
                    llm, raw_prompt, max_input_tokens
                )
                pairwise_tasks.append(
                    (
                        question,
                        challenger,
                        orientation,
                        left,
                        right,
                        raw_prompt,
                        prompt,
                        truncated,
                        token_count,
                    )
                )
    pairwise_rows: list[dict[str, Any]] = []
    sampling = _sampling_params(baseline_config, "pairwise")
    for start in range(0, len(pairwise_tasks), batch_size):
        batch = pairwise_tasks[start : start + batch_size]
        started = time.perf_counter()
        generated = llm.generate([task[6] for task in batch], sampling)
        latency = (time.perf_counter() - started) / max(1, len(batch))
        for task, output in zip(batch, generated, strict=True):
            (
                question,
                challenger,
                orientation,
                left,
                right,
                raw_prompt,
                prompt,
                truncated,
                token_count,
            ) = task
            raw_output = str(output.outputs[0].text)
            winner, reason, parse_error = parse_pre_pair_pairwise_output(raw_output)
            pairwise_rows.append(
                {
                    "question_id": question.question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "model": model,
                    "challenger": challenger,
                    "orientation": orientation,
                    "left_candidate": left,
                    "right_candidate": right,
                    "winner": winner,
                    "reason": reason,
                    "parse_error": parse_error,
                    "raw_output": raw_output,
                    "raw_prompt_sha256": hashlib.sha256(
                        raw_prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_was_truncated": bool(truncated),
                    "prompt_token_count": token_count,
                    "model_latency_seconds": latency,
                }
            )
    return pointwise_rows, pairwise_rows


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"PRePair-style worker must see physical GPU {physical_gpu}; got {visible!r}"
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
        raise PermissionError("Physical GPU is outside the PRePair-style protocol")
    models = tuple(str(value) for value in baseline_config["models"])
    if args.model not in models:
        raise ValueError(f"Unregistered PRePair-style model: {args.model}")
    run_root = args.run_root or Path(str(c3_config["output_root"]))
    if Path(str(baseline_config["run_root"])).resolve() != run_root.resolve():
        raise PermissionError("PRePair-style run root differs from the requested C3 root")
    all_questions = _load_questions(run_root)
    experts = tuple(str(value) for value in c3_config["experts"])
    all_base = _load_base_answers(run_root, all_questions, experts)
    questions = all_questions
    if args.smoke_questions:
        if not 1 <= args.smoke_questions <= 8:
            raise ValueError("PRePair-style smoke must use between one and eight questions")
        questions = _stratified_smoke_questions(all_questions, args.smoke_questions)
        output_dir = (
            run_root
            / "smoke"
            / f"prepair_style_n{args.smoke_questions}"
            / "models"
            / args.model
        )
        status = "bounded_label_free_pre_pair_style_smoke"
    else:
        output_dir = run_root / "prepair_style" / "models" / args.model
        status = "completed_label_free_pre_pair_style_model"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    max_challengers = int(baseline_config["candidate_selection"]["max_challengers"])
    slates = {
        question.question_id: rank_candidate_slate(
            question,
            all_base[question.question_id],
            experts,
            max_challengers,
        )
        for question in questions
    }
    if any(len(slate) != 1 + max_challengers for slate in slates.values()):
        raise RuntimeError("PRePair-style question lacks the frozen candidate slate size")
    partial = (
        run_root
        / "prepair_style_attempts"
        / f"{args.model}.{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    llm = load_llm(_model_args(c3_config, baseline_config), args.model)
    try:
        pointwise_rows, pairwise_rows = _generate_rows(
            llm, args.model, questions, slates, baseline_config
        )
    finally:
        del llm
        cleanup_vllm()
    calls_per_model, _ = pre_pair_call_budget(c3_config, baseline_config)
    if len(pointwise_rows) + len(pairwise_rows) != len(questions) * calls_per_model:
        raise RuntimeError("PRePair-style generation did not consume its exact model budget")
    pairwise_by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        pairwise_by_question[str(row["question_id"])].append(row)
    predictions = []
    for question in questions:
        slate = slates[question.question_id]
        top2 = aggregate_order_audited_pre_pair(
            question,
            slate,
            (args.model,),
            pairwise_by_question[question.question_id],
            challenger_limit=1,
        )
        top3 = aggregate_order_audited_pre_pair(
            question,
            slate,
            (args.model,),
            pairwise_by_question[question.question_id],
        )
        predictions.append(
            {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "environment": question.environment,
                "model": args.model,
                "candidate_slate": list(slate),
                "candidate_vote_counts": candidate_vote_counts(
                    question, all_base[question.question_id]
                ),
                "top2": top2,
                "budget_matched_top3": top3,
            }
        )
    pointwise_path = partial / "pointwise.jsonl"
    pairwise_path = partial / "pairwise.jsonl"
    prediction_path = partial / "predictions.jsonl"
    write_jsonl(pointwise_path, pointwise_rows)
    write_jsonl(pairwise_path, pairwise_rows)
    write_jsonl(prediction_path, predictions)
    question_path = run_root / "development_observables" / "questions.jsonl"
    base_path = run_root / "development_observables" / "base_predictions.jsonl"
    source_hashes = {
        "candidate_ranker_sha256": hashlib.sha256(
            inspect.getsource(rank_candidate_slate).encode("utf-8")
        ).hexdigest(),
        "pointwise_prompt_builder_sha256": hashlib.sha256(
            inspect.getsource(build_pre_pair_pointwise_prompt).encode("utf-8")
        ).hexdigest(),
        "pointwise_parser_sha256": hashlib.sha256(
            inspect.getsource(parse_pre_pair_pointwise_output).encode("utf-8")
        ).hexdigest(),
        "pairwise_prompt_builder_sha256": hashlib.sha256(
            inspect.getsource(build_pre_pair_pairwise_prompt).encode("utf-8")
        ).hexdigest(),
        "pairwise_parser_sha256": hashlib.sha256(
            inspect.getsource(parse_pre_pair_pairwise_output).encode("utf-8")
        ).hexdigest(),
        "aggregator_sha256": hashlib.sha256(
            inspect.getsource(aggregate_order_audited_pre_pair).encode("utf-8")
        ).hexdigest(),
    }
    write_json(
        partial / "manifest.json",
        {
            "status": status,
            "protocol_version": int(baseline_config["protocol_version"]),
            "model": args.model,
            "physical_gpu": args.physical_gpu,
            "questions": len(questions),
            "selected_question_ids": [row.question_id for row in questions],
            "candidate_slate_size": 1 + max_challengers,
            "calls_per_model_per_question": calls_per_model,
            "actual_model_calls": len(pointwise_rows) + len(pairwise_rows),
            "pointwise_calls": len(pointwise_rows),
            "parsed_pointwise_calls": sum(
                row["parse_error"] is None for row in pointwise_rows
            ),
            "pairwise_calls": len(pairwise_rows),
            "parsed_pairwise_calls": sum(
                row["parse_error"] is None for row in pairwise_rows
            ),
            "order_audited_pairs": len(questions) * max_challengers,
            "order_consistent_pairs": sum(
                model_outcome != "ABSTAIN"
                for row in predictions
                for challenge in row["budget_matched_top3"]["per_challenger"].values()
                for model_outcome in challenge["model_outcomes"].values()
            ),
            "truncated_prompts": sum(
                bool(row["prompt_was_truncated"])
                for row in pointwise_rows + pairwise_rows
            ),
            "labels_read": False,
            "question_sha256": sha256_file(question_path),
            "base_prediction_sha256": sha256_file(base_path),
            "pointwise_sha256": sha256_file(pointwise_path),
            "pairwise_sha256": sha256_file(pairwise_path),
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(args.c3_config),
            "baseline_config_sha256": sha256_file(args.baseline_config),
            **source_hashes,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(baseline_config["seed"]),
                [args.c3_config, args.baseline_config, question_path, base_path],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed label-free PRePair-style model control: {output_dir}")


if __name__ == "__main__":
    main()
