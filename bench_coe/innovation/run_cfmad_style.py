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
from typing import Any, Callable, Mapping, Sequence

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
from .cfmad_style import (
    build_cfmad_abduction_prompt,
    build_cfmad_cot_prompt,
    build_cfmad_critic_prompt,
    build_cfmad_defense_prompt,
    build_cfmad_judge_prompt,
    parse_cfmad_abduction_output,
    parse_cfmad_critic_output,
    parse_cfmad_defense_output,
    parse_cfmad_final_output,
    select_primary_candidate,
    select_seeded_counterfactual_candidate,
)


PHASES = ("cot", "abduction", "critic", "defense", "judge")
PHASE_SEED_OFFSETS = {
    "cot": 0,
    "abduction": 100_003,
    "critic": 200_003,
    "defense": 300_007,
    "judge": 400_009,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a label-free CFMAD-style staged prior-art control"
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
                    f"CFMAD-style generation input contains labels: {sorted(leaked)}"
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
        raise ValueError("CFMAD-style questions contain duplicate IDs")
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
        raise RuntimeError("CFMAD-style smoke selection exceeds available questions")
    return selected


def cfmad_calls_per_model_per_question(config: Mapping[str, Any]) -> int:
    protocol = config["method"]
    cot_samples = int(protocol["cot_samples"])
    preset_stances = int(protocol["preset_stances"])
    debate_rounds = int(protocol["debate_rounds"])
    calls = cot_samples + preset_stances * (1 + 2 * debate_rounds) + 1
    if calls != int(config["calls_per_model_per_question"]):
        raise ValueError("CFMAD-style declared call budget differs from its stages")
    return calls


def _validate_protocol(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
) -> None:
    if int(baseline_config.get("protocol_version", -1)) != 1:
        raise ValueError("Unknown CFMAD-style protocol")
    if Path(str(baseline_config.get("c3_config", ""))).resolve() != (
        c3_config_path.resolve()
    ):
        raise PermissionError("CFMAD-style protocol names a different C3 configuration")
    if baseline_config.get("c3_config_sha256") != sha256_file(c3_config_path):
        raise PermissionError("CFMAD-style protocol is not bound to this C3 configuration")
    frozen_pool = tuple(
        dict.fromkeys(
            [str(value) for value in c3_config["certificate_models"]]
            + [str(value) for value in c3_config["checker_models"]]
        )
    )
    models = tuple(str(value) for value in baseline_config.get("models", ()))
    if models != frozen_pool:
        raise PermissionError("CFMAD-style model pool/order differs from frozen C3 pool")
    if tuple(int(value) for value in baseline_config.get("physical_gpus", ())) != (
        0,
        1,
        2,
        3,
    ):
        raise PermissionError("CFMAD-style physical GPU boundary differs")
    method = baseline_config.get("method", {})
    if (
        int(method.get("cot_samples", -1)) != 3
        or int(method.get("preset_stances", -1)) != 2
        or int(method.get("debate_rounds", -1)) != 1
        or method.get("second_stance_selection")
        != "seeded_uniform_from_remaining_options"
        or method.get("same_backbone_for_all_roles") is not True
    ):
        raise PermissionError("CFMAD-style staged protocol differs from the preregistration")
    calls = cfmad_calls_per_model_per_question(baseline_config)
    if int(baseline_config.get("ensemble_calls_per_question", -1)) != calls * len(models):
        raise ValueError("CFMAD-style ensemble call budget differs")
    policy = baseline_config.get("data_policy", {})
    required_false = (
        "generation_reads_labels",
        "base_predictions_read",
        "certificate_or_check_outputs_read",
        "development_accuracy_used_for_selection",
        "target_labels_control_generation_or_aggregation",
    )
    if any(policy.get(key) is not False for key in required_false) or (
        policy.get("prediction_artifacts_frozen_before_evaluation") is not True
        or policy.get("counterfactual_choice_is_seeded_and_label_free") is not True
        or policy.get("model_pool_equals_prefrozen_c3_generator_checker_pool") is not True
    ):
        raise PermissionError("CFMAD-style protocol lacks the frozen label-free boundary")
    generation = baseline_config.get("generation", {})
    if (
        generation.get("backend") != "vllm"
        or float(generation.get("temperature", -1.0)) != 0.0
        or not bool(generation.get("guided_regex", False))
    ):
        raise PermissionError("CFMAD-style deterministic generation contract differs")


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
    baseline_config: Mapping[str, Any], phase: str, *, samples: int = 1
) -> Any:
    if phase not in PHASES:
        raise ValueError(f"Unknown CFMAD-style phase: {phase}")
    generation = baseline_config["generation"]
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    from vllm.sampling_params import GuidedDecodingParams

    regex = {
        "cot": r"REASON: [^\n]+\nFINAL: [A-Z]",
        "abduction": r"STANCE: [A-Z]\nARGUMENT: [^\n]+",
        "critic": r"CRITIQUE: [^\n]+",
        "defense": r"DEFENSE: [^\n]+",
        "judge": r"REASON: [^\n]+\nFINAL: [A-Z]",
    }[phase]
    return SamplingParams(
        n=samples,
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(baseline_config["seed"]) + PHASE_SEED_OFFSETS[phase],
        guided_decoding=GuidedDecodingParams(regex=regex, disable_fallback=True),
    )


def _prepare_prompt(
    llm: Any,
    raw_prompt: str,
    baseline_config: Mapping[str, Any],
) -> tuple[str, bool, int | None]:
    generation = baseline_config["generation"]
    max_input_tokens = (
        int(generation["max_model_len"])
        - int(generation["max_new_tokens"])
        - 8
    )
    prompt = apply_chat_template(llm, raw_prompt)
    return truncate_prompt_if_needed(llm, prompt, max_input_tokens)


def _prompt_metadata(
    raw_prompt: str,
    prompt: str,
    truncated: bool,
    token_count: int | None,
    latency: float,
) -> dict[str, Any]:
    return {
        "raw_prompt_sha256": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_was_truncated": bool(truncated),
        "prompt_token_count": token_count,
        "model_latency_seconds": latency,
    }


def _assert_label_free(row: Mapping[str, Any]) -> None:
    leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
    if leaked:
        raise AssertionError(f"CFMAD-style artifact emitted labels: {sorted(leaked)}")


def _generate_cot_rows(
    llm: Any,
    model: str,
    questions: Sequence[FalsificationQuestion],
    baseline_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[tuple[FalsificationQuestion, str, str, bool, int | None]] = []
    for question in questions:
        raw_prompt = build_cfmad_cot_prompt(question)
        prompt, truncated, token_count = _prepare_prompt(
            llm, raw_prompt, baseline_config
        )
        tasks.append((question, raw_prompt, prompt, truncated, token_count))
    samples = int(baseline_config["method"]["cot_samples"])
    sampling = _sampling_params(baseline_config, "cot", samples=samples)
    batch_size = int(baseline_config["generation"]["batch_size"])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        started = time.perf_counter()
        generated = llm.generate([task[2] for task in batch], sampling)
        output_count = sum(len(output.outputs) for output in generated)
        latency = (time.perf_counter() - started) / max(1, output_count)
        for task, request_output in zip(batch, generated, strict=True):
            question, raw_prompt, prompt, truncated, token_count = task
            if len(request_output.outputs) != samples:
                raise RuntimeError("vLLM returned an unexpected CFMAD-style CoT count")
            for sample_index, output in enumerate(request_output.outputs):
                raw_output = str(output.text)
                prediction, reason, parse_error = parse_cfmad_final_output(
                    raw_output, question.option_labels
                )
                row = {
                    "question_id": question.question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "model": model,
                    "phase": "cot",
                    "sample_index": sample_index,
                    "prediction": prediction,
                    "reason": reason,
                    "parse_error": parse_error,
                    "raw_output": raw_output,
                    **_prompt_metadata(
                        raw_prompt, prompt, truncated, token_count, latency
                    ),
                }
                _assert_label_free(row)
                rows.append(row)
    return rows


def _candidate_assignments(
    questions: Sequence[FalsificationQuestion],
    cot_rows: Sequence[Mapping[str, Any]],
    baseline_config: Mapping[str, Any],
    model: str,
) -> dict[str, dict[str, Any]]:
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cot_rows:
        by_question[str(row["question_id"])].append(row)
    assignments: dict[str, dict[str, Any]] = {}
    for question in questions:
        rows = sorted(
            by_question[question.question_id], key=lambda value: int(value["sample_index"])
        )
        primary, counts, tie_breaking = select_primary_candidate(
            [
                None if row.get("prediction") is None else str(row["prediction"])
                for row in rows
            ],
            question.option_labels,
        )
        counterfactual, index, digest = select_seeded_counterfactual_candidate(
            question.option_labels,
            primary,
            seed=int(baseline_config["seed"]),
            question_id=question.question_id,
            model_id=model,
        )
        assignments[question.question_id] = {
            "cot_predictions": [row.get("prediction") for row in rows],
            "cot_vote_counts": counts,
            "primary_candidate": primary,
            "counterfactual_candidate": counterfactual,
            "primary_tie_breaking": tie_breaking,
            "counterfactual_index_within_remaining": index,
            "counterfactual_selection_sha256": digest,
        }
    return assignments


def _phase_parser(
    phase: str, raw_output: str, candidate: str
) -> tuple[str | None, str | None]:
    if phase == "abduction":
        return parse_cfmad_abduction_output(raw_output, candidate)
    if phase == "critic":
        return parse_cfmad_critic_output(raw_output)
    if phase == "defense":
        return parse_cfmad_defense_output(raw_output)
    raise ValueError(f"Unknown CFMAD-style debate phase: {phase}")


def _generate_debate_phase(
    llm: Any,
    model: str,
    questions: Sequence[FalsificationQuestion],
    assignments: Mapping[str, Mapping[str, Any]],
    baseline_config: Mapping[str, Any],
    *,
    phase: str,
    previous: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    builders: dict[str, Callable[..., str]] = {
        "abduction": build_cfmad_abduction_prompt,
        "critic": build_cfmad_critic_prompt,
        "defense": build_cfmad_defense_prompt,
    }
    builder = builders[phase]
    tasks: list[
        tuple[FalsificationQuestion, int, str, str, str, bool, int | None]
    ] = []
    for question in questions:
        assignment = assignments[question.question_id]
        candidates = (
            str(assignment["primary_candidate"]),
            str(assignment["counterfactual_candidate"]),
        )
        for stance_index, candidate in enumerate(candidates):
            if phase == "abduction":
                raw_prompt = builder(question, candidate)
            elif phase == "critic":
                raw_prompt = builder(
                    question,
                    candidate,
                    str(previous[(question.question_id, stance_index, "abduction")]["raw_output"]),
                )
            else:
                raw_prompt = builder(
                    question,
                    candidate,
                    str(previous[(question.question_id, stance_index, "abduction")]["raw_output"]),
                    str(previous[(question.question_id, stance_index, "critic")]["raw_output"]),
                )
            prompt, truncated, token_count = _prepare_prompt(
                llm, raw_prompt, baseline_config
            )
            tasks.append(
                (
                    question,
                    stance_index,
                    candidate,
                    raw_prompt,
                    prompt,
                    truncated,
                    token_count,
                )
            )
    sampling = _sampling_params(baseline_config, phase)
    batch_size = int(baseline_config["generation"]["batch_size"])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        started = time.perf_counter()
        generated = llm.generate([task[4] for task in batch], sampling)
        latency = (time.perf_counter() - started) / max(1, len(batch))
        for task, request_output in zip(batch, generated, strict=True):
            (
                question,
                stance_index,
                candidate,
                raw_prompt,
                prompt,
                truncated,
                token_count,
            ) = task
            if len(request_output.outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected CFMAD-style phase count")
            raw_output = str(request_output.outputs[0].text)
            content, parse_error = _phase_parser(phase, raw_output, candidate)
            row = {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "environment": question.environment,
                "model": model,
                "phase": phase,
                "stance_index": stance_index,
                "candidate": candidate,
                "content": content,
                "parse_error": parse_error,
                "raw_output": raw_output,
                **_prompt_metadata(
                    raw_prompt, prompt, truncated, token_count, latency
                ),
            }
            _assert_label_free(row)
            rows.append(row)
    return rows


def _generate_judge_rows(
    llm: Any,
    model: str,
    questions: Sequence[FalsificationQuestion],
    assignments: Mapping[str, Mapping[str, Any]],
    baseline_config: Mapping[str, Any],
    previous: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tasks: list[
        tuple[FalsificationQuestion, list[dict[str, str]], str, str, bool, int | None]
    ] = []
    for question in questions:
        assignment = assignments[question.question_id]
        candidates = (
            str(assignment["primary_candidate"]),
            str(assignment["counterfactual_candidate"]),
        )
        trajectories = [
            {
                "candidate": candidate,
                "abduction": str(
                    previous[(question.question_id, stance_index, "abduction")][
                        "raw_output"
                    ]
                ),
                "critic": str(
                    previous[(question.question_id, stance_index, "critic")]["raw_output"]
                ),
                "defense": str(
                    previous[(question.question_id, stance_index, "defense")]["raw_output"]
                ),
            }
            for stance_index, candidate in enumerate(candidates)
        ]
        raw_prompt = build_cfmad_judge_prompt(question, trajectories)
        prompt, truncated, token_count = _prepare_prompt(
            llm, raw_prompt, baseline_config
        )
        tasks.append(
            (question, trajectories, raw_prompt, prompt, truncated, token_count)
        )
    sampling = _sampling_params(baseline_config, "judge")
    batch_size = int(baseline_config["generation"]["batch_size"])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        started = time.perf_counter()
        generated = llm.generate([task[3] for task in batch], sampling)
        latency = (time.perf_counter() - started) / max(1, len(batch))
        for task, request_output in zip(batch, generated, strict=True):
            question, trajectories, raw_prompt, prompt, truncated, token_count = task
            if len(request_output.outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected CFMAD-style judge count")
            raw_output = str(request_output.outputs[0].text)
            prediction, reason, parse_error = parse_cfmad_final_output(
                raw_output, question.option_labels
            )
            assignment = assignments[question.question_id]
            row = {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "environment": question.environment,
                "model": model,
                "phase": "judge",
                "prediction": prediction,
                "reason": reason,
                "parse_error": parse_error,
                "raw_output": raw_output,
                "trajectories": trajectories,
                **dict(assignment),
                "tie_breaking": assignment["primary_tie_breaking"],
                **_prompt_metadata(
                    raw_prompt, prompt, truncated, token_count, latency
                ),
            }
            _assert_label_free(row)
            rows.append(row)
    return rows


def _generate(
    c3_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    model: str,
    questions: Sequence[FalsificationQuestion],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    llm = load_llm(_model_args(c3_config, baseline_config), model)
    try:
        cot_rows = _generate_cot_rows(
            llm, model, questions, baseline_config
        )
        assignments = _candidate_assignments(
            questions, cot_rows, baseline_config, model
        )
        previous: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        debate_rows: list[dict[str, Any]] = []
        for phase in ("abduction", "critic", "defense"):
            phase_rows = _generate_debate_phase(
                llm,
                model,
                questions,
                assignments,
                baseline_config,
                phase=phase,
                previous=previous,
            )
            for row in phase_rows:
                previous[
                    (
                        str(row["question_id"]),
                        int(row["stance_index"]),
                        phase,
                    )
                ] = row
            debate_rows.extend(phase_rows)
        judge_rows = _generate_judge_rows(
            llm,
            model,
            questions,
            assignments,
            baseline_config,
            previous,
        )
        return cot_rows, debate_rows, judge_rows
    finally:
        del llm
        cleanup_vllm()


def _source_hashes() -> dict[str, str]:
    functions = {
        "cot_prompt_builder_sha256": build_cfmad_cot_prompt,
        "abduction_prompt_builder_sha256": build_cfmad_abduction_prompt,
        "critic_prompt_builder_sha256": build_cfmad_critic_prompt,
        "defense_prompt_builder_sha256": build_cfmad_defense_prompt,
        "judge_prompt_builder_sha256": build_cfmad_judge_prompt,
        "final_parser_sha256": parse_cfmad_final_output,
        "abduction_parser_sha256": parse_cfmad_abduction_output,
        "critic_parser_sha256": parse_cfmad_critic_output,
        "defense_parser_sha256": parse_cfmad_defense_output,
        "primary_selector_sha256": select_primary_candidate,
        "counterfactual_selector_sha256": select_seeded_counterfactual_candidate,
    }
    return {
        key: hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
        for key, function in functions.items()
    }


def _model_root(
    run_root: Path, model: str, smoke_questions: int
) -> Path:
    if smoke_questions:
        return (
            run_root
            / "smoke"
            / f"cfmad_style_n{smoke_questions}"
            / "models"
            / model
        )
    return run_root / "cfmad_style" / "models" / model


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"CFMAD-style worker must see physical GPU {physical_gpu}; got {visible!r}"
        )


def main() -> None:
    args = parse_args()
    c3_config = _load_yaml(args.c3_config)
    baseline_config = _load_yaml(args.baseline_config)
    _validate_protocol(args.c3_config, c3_config, baseline_config)
    _validate_gpu(args.physical_gpu)
    if args.physical_gpu not in {
        int(value) for value in baseline_config["physical_gpus"]
    }:
        raise PermissionError("Physical GPU is outside the CFMAD-style protocol")
    models = tuple(str(value) for value in baseline_config["models"])
    if args.model not in models:
        raise ValueError(f"Unregistered CFMAD-style model: {args.model}")
    run_root = args.run_root or Path(str(c3_config["output_root"]))
    if Path(str(baseline_config["run_root"])).resolve() != run_root.resolve():
        raise ValueError("CFMAD-style run root differs from the requested C3 run root")
    questions = _load_questions(run_root)
    if args.smoke_questions:
        if not 1 <= args.smoke_questions <= 8:
            raise ValueError("CFMAD-style smoke tests must use one to eight questions")
        questions = _stratified_smoke_questions(questions, args.smoke_questions)
    output_dir = _model_root(run_root, args.model, args.smoke_questions)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = (
        run_root
        / "cfmad_style_attempts"
        / f"{args.model}.{args.smoke_questions}.{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cot_rows, debate_rows, prediction_rows = _generate(
        c3_config, baseline_config, args.model, questions
    )
    cot_path = partial / "cot.jsonl"
    debate_path = partial / "debates.jsonl"
    prediction_path = partial / "predictions.jsonl"
    write_jsonl(cot_path, cot_rows)
    write_jsonl(debate_path, debate_rows)
    write_jsonl(prediction_path, prediction_rows)
    calls_per_question = cfmad_calls_per_model_per_question(baseline_config)
    actual_calls = len(cot_rows) + len(debate_rows) + len(prediction_rows)
    if actual_calls != len(questions) * calls_per_question:
        raise RuntimeError("CFMAD-style generation did not consume its exact call budget")
    if len(prediction_rows) != len(questions):
        raise RuntimeError("CFMAD-style judge coverage is incomplete")
    question_path = run_root / "development_observables" / "questions.jsonl"
    status = (
        "bounded_label_free_cfmad_style_smoke"
        if args.smoke_questions
        else "completed_label_free_cfmad_style_model"
    )
    phase_rows = cot_rows + debate_rows + prediction_rows
    phase_counts = {
        phase: sum(row["phase"] == phase for row in phase_rows) for phase in PHASES
    }
    parsed_phase_counts = {
        phase: sum(
            row["phase"] == phase and row["parse_error"] is None
            for row in phase_rows
        )
        for phase in PHASES
    }
    write_json(
        partial / "manifest.json",
        {
            "status": status,
            "protocol_version": int(baseline_config["protocol_version"]),
            "model": args.model,
            "physical_gpu": args.physical_gpu,
            "questions": len(questions),
            "selected_question_ids": [question.question_id for question in questions],
            "calls_per_model_per_question": calls_per_question,
            "actual_model_calls": actual_calls,
            "phase_calls": phase_counts,
            "parsed_phase_calls": parsed_phase_counts,
            "distinct_stance_pairs": sum(
                row["primary_candidate"] != row["counterfactual_candidate"]
                for row in prediction_rows
            ),
            "truncated_prompts": sum(
                bool(row["prompt_was_truncated"]) for row in phase_rows
            ),
            "labels_read": False,
            "base_predictions_read": False,
            "certificate_or_check_outputs_read": False,
            "question_sha256": sha256_file(question_path),
            "cot_sha256": sha256_file(cot_path),
            "debate_sha256": sha256_file(debate_path),
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(args.c3_config),
            "baseline_config_sha256": sha256_file(args.baseline_config),
            **_source_hashes(),
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
    print(f"Completed label-free CFMAD-style model: {output_dir}")


if __name__ == "__main__":
    main()
