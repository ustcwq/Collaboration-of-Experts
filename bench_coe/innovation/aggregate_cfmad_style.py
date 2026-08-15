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
from typing import Any, Mapping, Sequence

from .artifacts import environment_manifest, sha256_file, write_json, write_jsonl
from .blind_falsification_jury import FORBIDDEN_AUDIT_KEYS, FalsificationQuestion
from .cfmad_style import (
    CFMAD_STYLE_METHOD,
    aggregate_cfmad_model_predictions,
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
from .run_cfmad_style import (
    PHASES,
    _load_questions,
    _load_yaml,
    _model_root,
    _source_hashes,
    _stratified_smoke_questions,
    _validate_protocol,
    cfmad_calls_per_model_per_question,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and aggregate label-free CFMAD-style artifacts"
    )
    parser.add_argument("--c3-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-questions", type=int, default=0)
    return parser.parse_args()


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


def _recorded_input_hash(environment: Mapping[str, Any], path: Path) -> str | None:
    values = environment.get("input_hashes", {})
    if not isinstance(values, dict):
        return None
    direct = values.get(str(path))
    if isinstance(direct, str):
        return direct
    resolved = path.resolve()
    for raw_path, value in values.items():
        try:
            if Path(str(raw_path)).resolve() == resolved and isinstance(value, str):
                return value
        except OSError:
            continue
    return None


def cfmad_model_method(model: str) -> str:
    return f"cfmad_style::{model}"


def _aggregate_root(run_root: Path, smoke_questions: int) -> Path:
    if smoke_questions:
        return run_root / "smoke" / f"cfmad_style_n{smoke_questions}" / "aggregate"
    return run_root / "cfmad_style" / "aggregate"


def _row_metadata_matches(
    row: Mapping[str, Any], question: FalsificationQuestion, model: str
) -> bool:
    return (
        row.get("dataset") == question.dataset
        and row.get("environment") == question.environment
        and row.get("model") == model
    )


def _raw_prompt_digest(raw_prompt: str) -> str:
    return hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()


def _replay_assignments(
    questions: Sequence[FalsificationQuestion],
    cot_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_config: Mapping[str, Any],
    model: str,
) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    for question in questions:
        rows = sorted(
            cot_by_question[question.question_id],
            key=lambda value: int(value["sample_index"]),
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


def authenticate_cfmad_models(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config_path: Path,
    baseline_config: Mapping[str, Any],
    run_root: Path,
    *,
    smoke_questions: int = 0,
) -> tuple[
    tuple[FalsificationQuestion, ...],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    _validate_protocol(c3_config_path, c3_config, baseline_config)
    if Path(str(baseline_config["run_root"])).resolve() != run_root.resolve():
        raise PermissionError("CFMAD-style sidecar points to a different run root")
    all_questions = _load_questions(run_root)
    questions = (
        _stratified_smoke_questions(all_questions, smoke_questions)
        if smoke_questions
        else all_questions
    )
    question_by_id = {question.question_id: question for question in questions}
    models = tuple(str(value) for value in baseline_config["models"])
    calls_per_model = cfmad_calls_per_model_per_question(baseline_config)
    cot_samples = int(baseline_config["method"]["cot_samples"])
    preset_stances = int(baseline_config["method"]["preset_stances"])
    debate_rounds = int(baseline_config["method"]["debate_rounds"])
    if preset_stances != 2 or debate_rounds != 1:
        raise PermissionError("Unsupported CFMAD-style replay shape")
    expected_status = (
        "bounded_label_free_cfmad_style_smoke"
        if smoke_questions
        else "completed_label_free_cfmad_style_model"
    )
    question_path = run_root / "development_observables" / "questions.jsonl"
    source_hashes = _source_hashes()
    all_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    model_quality: list[dict[str, Any]] = []
    model_artifact_hashes: dict[str, dict[str, str]] = {}

    for model in models:
        directory = _model_root(run_root, model, smoke_questions)
        cot_path = directory / "cot.jsonl"
        debate_path = directory / "debates.jsonl"
        prediction_path = directory / "predictions.jsonl"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hashes = {
            "question_sha256": sha256_file(question_path),
            "cot_sha256": sha256_file(cot_path),
            "debate_sha256": sha256_file(debate_path),
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(c3_config_path),
            "baseline_config_sha256": sha256_file(baseline_config_path),
            **source_hashes,
        }
        if (
            manifest.get("status") != expected_status
            or manifest.get("model") != model
            or manifest.get("labels_read") is not False
            or manifest.get("base_predictions_read") is not False
            or manifest.get("certificate_or_check_outputs_read") is not False
            or int(manifest.get("protocol_version", -1))
            != int(baseline_config["protocol_version"])
            or int(manifest.get("questions", -1)) != len(questions)
            or list(manifest.get("selected_question_ids", ()))
            != [question.question_id for question in questions]
            or int(manifest.get("calls_per_model_per_question", -1))
            != calls_per_model
            or int(manifest.get("actual_model_calls", -1))
            != len(questions) * calls_per_model
        ):
            raise PermissionError(f"CFMAD-style model manifest differs: {model}")
        for key, digest in expected_hashes.items():
            if manifest.get(key) != digest:
                raise PermissionError(f"CFMAD-style {key} differs: {model}")
        environment = manifest.get("environment", {})
        for path in (c3_config_path, baseline_config_path, question_path):
            if _recorded_input_hash(environment, path) != sha256_file(path):
                raise PermissionError(f"CFMAD-style provenance differs: {model}")

        cot_rows = _read_jsonl(cot_path)
        expected_cot = {
            (question.question_id, sample_index)
            for question in questions
            for sample_index in range(cot_samples)
        }
        actual_cot: set[tuple[str, int]] = set()
        cot_by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        parsed_by_phase = {phase: 0 for phase in PHASES}
        truncated = 0
        for row in cot_rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"CFMAD-style CoT row contains labels: {sorted(leaked)}"
                )
            question_id = str(row["question_id"])
            question = question_by_id.get(question_id)
            key = (question_id, int(row["sample_index"]))
            if question is None or key in actual_cot:
                raise ValueError("Unknown or duplicate CFMAD-style CoT row")
            if row.get("phase") != "cot" or not _row_metadata_matches(
                row, question, model
            ):
                raise PermissionError("CFMAD-style CoT metadata differs")
            reparsed = parse_cfmad_final_output(
                str(row["raw_output"]), question.option_labels
            )
            stored = (
                None if row.get("prediction") is None else str(row["prediction"]),
                None if row.get("reason") is None else str(row["reason"]),
                None if row.get("parse_error") is None else str(row["parse_error"]),
            )
            if reparsed != stored:
                raise PermissionError("CFMAD-style CoT parser replay differs")
            raw_prompt = build_cfmad_cot_prompt(question)
            if row.get("raw_prompt_sha256") != _raw_prompt_digest(raw_prompt):
                raise PermissionError("CFMAD-style CoT prompt replay differs")
            actual_cot.add(key)
            cot_by_question[question_id].append(row)
            parsed_by_phase["cot"] += int(reparsed[2] is None)
            truncated += int(bool(row["prompt_was_truncated"]))
        if actual_cot != expected_cot or len(cot_rows) != len(expected_cot):
            raise RuntimeError(f"CFMAD-style CoT grid is incomplete: {model}")
        assignments = _replay_assignments(
            questions, cot_by_question, baseline_config, model
        )

        debate_rows = _read_jsonl(debate_path)
        expected_debate = {
            (question.question_id, stance_index, phase)
            for question in questions
            for stance_index in range(preset_stances)
            for phase in ("abduction", "critic", "defense")
        }
        actual_debate: set[tuple[str, int, str]] = set()
        debate_by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        for row in debate_rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"CFMAD-style debate row contains labels: {sorted(leaked)}"
                )
            question_id = str(row["question_id"])
            stance_index = int(row["stance_index"])
            phase = str(row["phase"])
            question = question_by_id.get(question_id)
            key = (question_id, stance_index, phase)
            if question is None or key in actual_debate or key not in expected_debate:
                raise ValueError("Unknown or duplicate CFMAD-style debate row")
            candidates = (
                str(assignments[question_id]["primary_candidate"]),
                str(assignments[question_id]["counterfactual_candidate"]),
            )
            candidate = candidates[stance_index]
            if row.get("candidate") != candidate or not _row_metadata_matches(
                row, question, model
            ):
                raise PermissionError("CFMAD-style debate metadata differs")
            raw_output = str(row["raw_output"])
            if phase == "abduction":
                reparsed = parse_cfmad_abduction_output(raw_output, candidate)
            elif phase == "critic":
                reparsed = parse_cfmad_critic_output(raw_output)
            else:
                reparsed = parse_cfmad_defense_output(raw_output)
            stored = (
                None if row.get("content") is None else str(row["content"]),
                None if row.get("parse_error") is None else str(row["parse_error"]),
            )
            if reparsed != stored:
                raise PermissionError("CFMAD-style debate parser replay differs")
            actual_debate.add(key)
            debate_by_key[key] = row
            parsed_by_phase[phase] += int(reparsed[1] is None)
            truncated += int(bool(row["prompt_was_truncated"]))
        if actual_debate != expected_debate or len(debate_rows) != len(expected_debate):
            raise RuntimeError(f"CFMAD-style debate grid is incomplete: {model}")
        for question in questions:
            assignment = assignments[question.question_id]
            candidates = (
                str(assignment["primary_candidate"]),
                str(assignment["counterfactual_candidate"]),
            )
            for stance_index, candidate in enumerate(candidates):
                abduction = debate_by_key[
                    (question.question_id, stance_index, "abduction")
                ]
                critic = debate_by_key[(question.question_id, stance_index, "critic")]
                defense = debate_by_key[(question.question_id, stance_index, "defense")]
                raw_prompts = {
                    "abduction": build_cfmad_abduction_prompt(question, candidate),
                    "critic": build_cfmad_critic_prompt(
                        question, candidate, str(abduction["raw_output"])
                    ),
                    "defense": build_cfmad_defense_prompt(
                        question,
                        candidate,
                        str(abduction["raw_output"]),
                        str(critic["raw_output"]),
                    ),
                }
                for phase, row in (
                    ("abduction", abduction),
                    ("critic", critic),
                    ("defense", defense),
                ):
                    if row.get("raw_prompt_sha256") != _raw_prompt_digest(
                        raw_prompts[phase]
                    ):
                        raise PermissionError(
                            f"CFMAD-style {phase} prompt replay differs"
                        )

        prediction_rows = _read_jsonl(prediction_path)
        predictions_by_question: dict[str, dict[str, Any]] = {}
        for row in prediction_rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"CFMAD-style prediction row contains labels: {sorted(leaked)}"
                )
            question_id = str(row["question_id"])
            question = question_by_id.get(question_id)
            if question is None or question_id in predictions_by_question:
                raise ValueError("Unknown or duplicate CFMAD-style prediction row")
            if row.get("phase") != "judge" or not _row_metadata_matches(
                row, question, model
            ):
                raise PermissionError("CFMAD-style judge metadata differs")
            assignment = assignments[question_id]
            candidates = (
                str(assignment["primary_candidate"]),
                str(assignment["counterfactual_candidate"]),
            )
            trajectories = [
                {
                    "candidate": candidate,
                    "abduction": str(
                        debate_by_key[(question_id, stance_index, "abduction")][
                            "raw_output"
                        ]
                    ),
                    "critic": str(
                        debate_by_key[(question_id, stance_index, "critic")]["raw_output"]
                    ),
                    "defense": str(
                        debate_by_key[(question_id, stance_index, "defense")]["raw_output"]
                    ),
                }
                for stance_index, candidate in enumerate(candidates)
            ]
            reparsed = parse_cfmad_final_output(
                str(row["raw_output"]), question.option_labels
            )
            stored = (
                None if row.get("prediction") is None else str(row["prediction"]),
                None if row.get("reason") is None else str(row["reason"]),
                None if row.get("parse_error") is None else str(row["parse_error"]),
            )
            expected_assignment = {**assignment, "tie_breaking": assignment["primary_tie_breaking"]}
            if (
                reparsed != stored
                or row.get("trajectories") != trajectories
                or any(row.get(key) != value for key, value in expected_assignment.items())
            ):
                raise PermissionError("CFMAD-style judge replay differs")
            raw_prompt = build_cfmad_judge_prompt(question, trajectories)
            if row.get("raw_prompt_sha256") != _raw_prompt_digest(raw_prompt):
                raise PermissionError("CFMAD-style judge prompt replay differs")
            predictions_by_question[question_id] = row
            parsed_by_phase["judge"] += int(reparsed[2] is None)
            truncated += int(bool(row["prompt_was_truncated"]))
        if set(predictions_by_question) != set(question_by_id):
            raise RuntimeError(f"CFMAD-style judge coverage is incomplete: {model}")

        phase_calls = {
            "cot": len(cot_rows),
            "abduction": len(questions) * preset_stances,
            "critic": len(questions) * preset_stances * debate_rounds,
            "defense": len(questions) * preset_stances * debate_rounds,
            "judge": len(prediction_rows),
        }
        distinct_pairs = sum(
            row["primary_candidate"] != row["counterfactual_candidate"]
            for row in prediction_rows
        )
        if (
            manifest.get("phase_calls") != phase_calls
            or manifest.get("parsed_phase_calls") != parsed_by_phase
            or int(manifest.get("distinct_stance_pairs", -1)) != distinct_pairs
            or int(manifest.get("truncated_prompts", -1)) != truncated
        ):
            raise RuntimeError(f"CFMAD-style manifest counts differ: {model}")
        acceptance = baseline_config["acceptance"]
        phase_rates = {
            phase: parsed_by_phase[phase] / max(1, phase_calls[phase])
            for phase in PHASES
        }
        if (
            phase_rates["cot"] < float(acceptance["minimum_cot_parse_rate"])
            or min(
                phase_rates["abduction"],
                phase_rates["critic"],
                phase_rates["defense"],
            )
            < float(acceptance["minimum_debate_parse_rate"])
            or phase_rates["judge"] < float(acceptance["minimum_judge_parse_rate"])
            or truncated > int(acceptance["maximum_prompt_truncations"])
            or distinct_pairs != len(questions)
        ):
            raise RuntimeError(f"CFMAD-style generation quality gate failed: {model}")
        all_predictions[model] = predictions_by_question
        model_quality.append(
            {
                "model": model,
                "questions": len(questions),
                "calls_per_question": calls_per_model,
                "phase_calls": phase_calls,
                "phase_parse_rates": phase_rates,
                "truncated_prompts": truncated,
                "distinct_stance_pair_rate": distinct_pairs / max(1, len(questions)),
            }
        )
        model_artifact_hashes[model] = {
            "cot_sha256": expected_hashes["cot_sha256"],
            "debate_sha256": expected_hashes["debate_sha256"],
            "prediction_sha256": expected_hashes["prediction_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        }
    quality = {
        "models": model_quality,
        "model_artifact_hashes": model_artifact_hashes,
        "calls_per_model_per_question": calls_per_model,
        "ensemble_calls_per_question": calls_per_model * len(models),
        "require_primary_strictly_beats_all_cfmad_style_methods": bool(
            baseline_config["acceptance"].get(
                "require_primary_strictly_beats_all_cfmad_style_methods", False
            )
        ),
    }
    return tuple(questions), all_predictions, quality


def expected_aggregate_rows(
    questions: Sequence[FalsificationQuestion],
    model_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    models = tuple(str(value) for value in baseline_config["models"])
    rows: list[dict[str, Any]] = []
    for question in questions:
        per_model = {
            model: model_rows[model][question.question_id].get("prediction")
            for model in models
        }
        ensemble, counts, tie_breaking = aggregate_cfmad_model_predictions(
            per_model, models, question.option_labels
        )
        methods = {
            cfmad_model_method(model): per_model[model] for model in models
        }
        methods[CFMAD_STYLE_METHOD] = ensemble
        rows.append(
            {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "environment": question.environment,
                "models": list(models),
                "model_predictions": per_model,
                "predictions": methods,
                "ensemble_vote_counts": counts,
                "ensemble_tie_breaking": tie_breaking,
            }
        )
    return rows


def authenticate_completed_cfmad(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config_path: Path,
    run_root: Path,
) -> tuple[dict[str, dict[str, str | None]], dict[str, float], dict[str, Any]]:
    baseline_config = _load_yaml(baseline_config_path)
    questions, model_rows, quality = authenticate_cfmad_models(
        c3_config_path,
        c3_config,
        baseline_config_path,
        baseline_config,
        run_root,
    )
    directory = _aggregate_root(run_root, 0)
    prediction_path = directory / "predictions.jsonl"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = expected_aggregate_rows(questions, model_rows, baseline_config)
    actual_rows = _read_jsonl(prediction_path)
    if actual_rows != expected_rows:
        raise PermissionError("CFMAD-style aggregate predictions differ from replay")
    aggregator_hash = hashlib.sha256(
        inspect.getsource(aggregate_cfmad_model_predictions).encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("status") != "completed_label_free_cfmad_style_aggregate"
        or manifest.get("labels_read") is not False
        or int(manifest.get("questions", -1)) != len(questions)
        or manifest.get("prediction_sha256") != sha256_file(prediction_path)
        or manifest.get("c3_config_sha256") != sha256_file(c3_config_path)
        or manifest.get("baseline_config_sha256") != sha256_file(baseline_config_path)
        or manifest.get("model_artifact_hashes")
        != quality["model_artifact_hashes"]
        or manifest.get("aggregator_sha256") != aggregator_hash
    ):
        raise PermissionError("CFMAD-style aggregate manifest differs")
    method_names = [cfmad_model_method(model) for model in baseline_config["models"]]
    method_names.append(CFMAD_STYLE_METHOD)
    predictions = {
        method: {
            str(row["question_id"]): (
                None
                if row["predictions"][method] is None
                else str(row["predictions"][method])
            )
            for row in actual_rows
        }
        for method in method_names
    }
    calls_per_model = float(quality["calls_per_model_per_question"])
    budgets = {
        cfmad_model_method(model): calls_per_model
        for model in baseline_config["models"]
    }
    budgets[CFMAD_STYLE_METHOD] = float(quality["ensemble_calls_per_question"])
    return predictions, budgets, quality


def main() -> None:
    args = parse_args()
    c3_config = _load_yaml(args.c3_config)
    baseline_config = _load_yaml(args.baseline_config)
    run_root = args.run_root or Path(str(c3_config["output_root"]))
    questions, model_rows, quality = authenticate_cfmad_models(
        args.c3_config,
        c3_config,
        args.baseline_config,
        baseline_config,
        run_root,
        smoke_questions=args.smoke_questions,
    )
    output_dir = _aggregate_root(run_root, args.smoke_questions)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started = time.time()
    rows = expected_aggregate_rows(questions, model_rows, baseline_config)
    partial = (
        run_root
        / "cfmad_style_attempts"
        / f"aggregate.{args.smoke_questions}.{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    prediction_path = partial / "predictions.jsonl"
    write_jsonl(prediction_path, rows)
    status = (
        "bounded_label_free_cfmad_style_aggregate_smoke"
        if args.smoke_questions
        else "completed_label_free_cfmad_style_aggregate"
    )
    aggregator_hash = hashlib.sha256(
        inspect.getsource(aggregate_cfmad_model_predictions).encode("utf-8")
    ).hexdigest()
    write_json(
        partial / "manifest.json",
        {
            "status": status,
            "protocol_version": int(baseline_config["protocol_version"]),
            "questions": len(questions),
            "methods": [
                *(cfmad_model_method(str(model)) for model in baseline_config["models"]),
                CFMAD_STYLE_METHOD,
            ],
            "calls_per_model_per_question": quality["calls_per_model_per_question"],
            "ensemble_calls_per_question": quality["ensemble_calls_per_question"],
            "model_artifact_hashes": quality["model_artifact_hashes"],
            "labels_read": False,
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(args.c3_config),
            "baseline_config_sha256": sha256_file(args.baseline_config),
            "aggregator_sha256": aggregator_hash,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(baseline_config["seed"]),
                [
                    args.c3_config,
                    args.baseline_config,
                    run_root / "development_observables" / "questions.jsonl",
                    *(
                        _model_root(run_root, str(model), args.smoke_questions)
                        for model in baseline_config["models"]
                    ),
                ],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed label-free CFMAD-style aggregation: {output_dir}")


if __name__ == "__main__":
    main()
