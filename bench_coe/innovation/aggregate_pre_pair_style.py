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
from .run_pre_pair_style import (
    _load_base_answers,
    _load_questions,
    _load_yaml,
    _stratified_smoke_questions,
    _validate_protocol,
    pre_pair_call_budget,
)


TOP2_METHOD = "prepair_style_order_audited_top2"
BUDGET_MATCHED_METHOD = "prepair_style_budget_matched_top3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and aggregate label-free PRePair-style model artifacts"
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


def _source_hashes() -> dict[str, str]:
    return {
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


def _model_root(run_root: Path, model: str, smoke_questions: int) -> Path:
    if smoke_questions:
        return (
            run_root
            / "smoke"
            / f"prepair_style_n{smoke_questions}"
            / "models"
            / model
        )
    return run_root / "prepair_style" / "models" / model


def _aggregate_root(run_root: Path, smoke_questions: int) -> Path:
    if smoke_questions:
        return run_root / "smoke" / f"prepair_style_n{smoke_questions}" / "aggregate"
    return run_root / "prepair_style" / "aggregate"


def authenticate_pre_pair_models(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config_path: Path,
    baseline_config: Mapping[str, Any],
    run_root: Path,
    *,
    smoke_questions: int = 0,
) -> tuple[
    tuple[FalsificationQuestion, ...],
    list[dict[str, Any]],
    dict[str, Any],
]:
    _validate_protocol(c3_config_path, c3_config, baseline_config)
    if Path(str(baseline_config["run_root"])).resolve() != run_root.resolve():
        raise PermissionError("PRePair-style sidecar points to a different run root")
    all_questions = _load_questions(run_root)
    questions = (
        _stratified_smoke_questions(all_questions, smoke_questions)
        if smoke_questions
        else all_questions
    )
    question_by_id = {question.question_id: question for question in questions}
    experts = tuple(str(value) for value in c3_config["experts"])
    all_base = _load_base_answers(run_root, all_questions, experts)
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
    models = tuple(str(value) for value in baseline_config["models"])
    calls_per_model, total_call_budget = pre_pair_call_budget(
        c3_config, baseline_config
    )
    expected_status = (
        "bounded_label_free_pre_pair_style_smoke"
        if smoke_questions
        else "completed_label_free_pre_pair_style_model"
    )
    question_path = run_root / "development_observables" / "questions.jsonl"
    base_path = run_root / "development_observables" / "base_predictions.jsonl"
    source_hashes = _source_hashes()
    all_pairwise_rows: list[dict[str, Any]] = []
    model_quality: list[dict[str, Any]] = []
    model_artifact_hashes: dict[str, dict[str, str]] = {}
    for model in models:
        directory = _model_root(run_root, model, smoke_questions)
        pointwise_path = directory / "pointwise.jsonl"
        pairwise_path = directory / "pairwise.jsonl"
        prediction_path = directory / "predictions.jsonl"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hashes = {
            "question_sha256": sha256_file(question_path),
            "base_prediction_sha256": sha256_file(base_path),
            "pointwise_sha256": sha256_file(pointwise_path),
            "pairwise_sha256": sha256_file(pairwise_path),
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(c3_config_path),
            "baseline_config_sha256": sha256_file(baseline_config_path),
            **source_hashes,
        }
        if (
            manifest.get("status") != expected_status
            or manifest.get("model") != model
            or manifest.get("labels_read") is not False
            or int(manifest.get("protocol_version", -1))
            != int(baseline_config["protocol_version"])
            or int(manifest.get("questions", -1)) != len(questions)
            or list(manifest.get("selected_question_ids", ()))
            != [question.question_id for question in questions]
            or int(manifest.get("candidate_slate_size", -1))
            != 1 + max_challengers
            or int(manifest.get("calls_per_model_per_question", -1))
            != calls_per_model
            or int(manifest.get("actual_model_calls", -1))
            != len(questions) * calls_per_model
        ):
            raise PermissionError(f"PRePair-style model manifest differs: {model}")
        for key, digest in expected_hashes.items():
            if manifest.get(key) != digest:
                raise PermissionError(f"PRePair-style {key} differs: {model}")
        environment = manifest.get("environment", {})
        if (
            _recorded_input_hash(environment, c3_config_path)
            != sha256_file(c3_config_path)
            or _recorded_input_hash(environment, baseline_config_path)
            != sha256_file(baseline_config_path)
            or _recorded_input_hash(environment, question_path)
            != sha256_file(question_path)
            or _recorded_input_hash(environment, base_path) != sha256_file(base_path)
        ):
            raise PermissionError(f"PRePair-style provenance differs: {model}")

        pointwise_rows = _read_jsonl(pointwise_path)
        expected_pointwise = {
            (question.question_id, candidate)
            for question in questions
            for candidate in slates[question.question_id]
        }
        actual_pointwise: set[tuple[str, str]] = set()
        pointwise_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        parsed_pointwise = 0
        truncated = 0
        for row in pointwise_rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"PRePair-style pointwise row contains labels: {sorted(leaked)}"
                )
            question_id = str(row["question_id"])
            candidate = str(row["candidate"])
            question = question_by_id.get(question_id)
            key = (question_id, candidate)
            if question is None or key in actual_pointwise:
                raise ValueError("Unknown or duplicate PRePair-style pointwise row")
            if (
                row.get("model") != model
                or row.get("dataset") != question.dataset
                or row.get("environment") != question.environment
            ):
                raise PermissionError("PRePair-style pointwise metadata differs")
            analysis, parse_error = parse_pre_pair_pointwise_output(
                str(row["raw_output"])
            )
            if row.get("analysis") != analysis or row.get("parse_error") != parse_error:
                raise PermissionError("PRePair-style pointwise parser replay differs")
            raw_prompt = build_pre_pair_pointwise_prompt(question, candidate)
            if row.get("raw_prompt_sha256") != hashlib.sha256(
                raw_prompt.encode("utf-8")
            ).hexdigest():
                raise PermissionError("PRePair-style pointwise prompt replay differs")
            actual_pointwise.add(key)
            pointwise_by_key[key] = row
            parsed_pointwise += int(parse_error is None)
            truncated += int(bool(row["prompt_was_truncated"]))
        if actual_pointwise != expected_pointwise or len(pointwise_rows) != len(
            expected_pointwise
        ):
            raise RuntimeError(f"PRePair-style pointwise grid is incomplete: {model}")

        pairwise_rows = _read_jsonl(pairwise_path)
        expected_pairwise = {
            (question.question_id, challenger, orientation)
            for question in questions
            for challenger in slates[question.question_id][1:]
            for orientation in PAIRWISE_ORIENTATIONS
        }
        actual_pairwise: set[tuple[str, str, str]] = set()
        parsed_pairwise = 0
        for row in pairwise_rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"PRePair-style pairwise row contains labels: {sorted(leaked)}"
                )
            question_id = str(row["question_id"])
            challenger = str(row["challenger"])
            orientation = str(row["orientation"])
            question = question_by_id.get(question_id)
            key = (question_id, challenger, orientation)
            if question is None or key in actual_pairwise:
                raise ValueError("Unknown or duplicate PRePair-style pairwise row")
            primary = slates[question_id][0]
            expected_left, expected_right = (
                (primary, challenger)
                if orientation == "primary_left"
                else (challenger, primary)
            )
            if (
                row.get("model") != model
                or row.get("dataset") != question.dataset
                or row.get("environment") != question.environment
                or row.get("left_candidate") != expected_left
                or row.get("right_candidate") != expected_right
            ):
                raise PermissionError("PRePair-style pairwise metadata differs")
            winner, reason, parse_error = parse_pre_pair_pairwise_output(
                str(row["raw_output"])
            )
            if (
                row.get("winner") != winner
                or row.get("reason") != reason
                or row.get("parse_error") != parse_error
            ):
                raise PermissionError("PRePair-style pairwise parser replay differs")
            raw_prompt = build_pre_pair_pairwise_prompt(
                question,
                expected_left,
                expected_right,
                str(pointwise_by_key[(question_id, expected_left)]["raw_output"]),
                str(pointwise_by_key[(question_id, expected_right)]["raw_output"]),
            )
            if row.get("raw_prompt_sha256") != hashlib.sha256(
                raw_prompt.encode("utf-8")
            ).hexdigest():
                raise PermissionError("PRePair-style pairwise prompt replay differs")
            actual_pairwise.add(key)
            parsed_pairwise += int(parse_error is None)
            truncated += int(bool(row["prompt_was_truncated"]))
            all_pairwise_rows.append(row)
        if actual_pairwise != expected_pairwise or len(pairwise_rows) != len(
            expected_pairwise
        ):
            raise RuntimeError(f"PRePair-style pairwise grid is incomplete: {model}")

        by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pairwise_rows:
            by_question[str(row["question_id"])].append(row)
        prediction_rows = _read_jsonl(prediction_path)
        if len(prediction_rows) != len(questions):
            raise RuntimeError("PRePair-style per-model predictions are incomplete")
        seen_predictions: set[str] = set()
        order_consistent_pairs = 0
        for row in prediction_rows:
            question_id = str(row["question_id"])
            question = question_by_id.get(question_id)
            if question is None or question_id in seen_predictions:
                raise ValueError("Unknown or duplicate PRePair-style prediction")
            slate = slates[question_id]
            expected_top2 = aggregate_order_audited_pre_pair(
                question,
                slate,
                (model,),
                by_question[question_id],
                challenger_limit=1,
            )
            expected_top3 = aggregate_order_audited_pre_pair(
                question, slate, (model,), by_question[question_id]
            )
            if (
                row.get("candidate_slate") != list(slate)
                or row.get("candidate_vote_counts")
                != candidate_vote_counts(question, all_base[question_id])
                or row.get("top2") != expected_top2
                or row.get("budget_matched_top3") != expected_top3
            ):
                raise PermissionError("PRePair-style per-model aggregation replay differs")
            order_consistent_pairs += sum(
                model_outcome != "ABSTAIN"
                for challenge in expected_top3["per_challenger"].values()
                for model_outcome in challenge["model_outcomes"].values()
            )
            seen_predictions.add(question_id)
        if seen_predictions != set(question_by_id):
            raise RuntimeError("PRePair-style prediction coverage differs")
        replayed_counts = {
            "pointwise_calls": len(pointwise_rows),
            "parsed_pointwise_calls": parsed_pointwise,
            "pairwise_calls": len(pairwise_rows),
            "parsed_pairwise_calls": parsed_pairwise,
            "order_audited_pairs": len(questions) * max_challengers,
            "order_consistent_pairs": order_consistent_pairs,
            "truncated_prompts": truncated,
        }
        if any(int(manifest.get(key, -1)) != value for key, value in replayed_counts.items()):
            raise RuntimeError(f"PRePair-style manifest counts differ: {model}")
        model_quality.append(
            {
                "model": model,
                **replayed_counts,
                "pointwise_parse_rate": parsed_pointwise / max(1, len(pointwise_rows)),
                "pairwise_parse_rate": parsed_pairwise / max(1, len(pairwise_rows)),
                "order_consistent_pair_rate": order_consistent_pairs
                / max(1, len(questions) * max_challengers),
            }
        )
        model_artifact_hashes[model] = {
            "pointwise_sha256": expected_hashes["pointwise_sha256"],
            "pairwise_sha256": expected_hashes["pairwise_sha256"],
            "prediction_sha256": expected_hashes["prediction_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        }

    acceptance = baseline_config["acceptance"]
    if any(
        float(row["pointwise_parse_rate"])
        < float(acceptance["minimum_pointwise_parse_rate"])
        or float(row["pairwise_parse_rate"])
        < float(acceptance["minimum_pairwise_parse_rate"])
        or float(row["order_consistent_pair_rate"])
        < float(acceptance["minimum_order_consistent_pair_rate"])
        or int(row["truncated_prompts"])
        > int(acceptance["maximum_prompt_truncations"])
        for row in model_quality
    ):
        raise RuntimeError("PRePair-style generation quality is below the frozen gate")
    quality = {
        "models": model_quality,
        "model_artifact_hashes": model_artifact_hashes,
        "calls_per_model_per_question": calls_per_model,
        "nominal_total_calls_per_question": total_call_budget,
        "new_control_calls_per_question": len(models) * calls_per_model,
        "require_primary_strictly_beats_both_methods": bool(
            acceptance["require_primary_strictly_beats_both_methods"]
        ),
    }
    return tuple(questions), all_pairwise_rows, quality


def expected_aggregate_rows(
    questions: Sequence[FalsificationQuestion],
    pairwise_rows: Sequence[Mapping[str, Any]],
    c3_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    run_root: Path,
) -> list[dict[str, Any]]:
    experts = tuple(str(value) for value in c3_config["experts"])
    all_questions = _load_questions(run_root)
    all_base = _load_base_answers(run_root, all_questions, experts)
    max_challengers = int(baseline_config["candidate_selection"]["max_challengers"])
    models = tuple(str(value) for value in baseline_config["models"])
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        by_question[str(row["question_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for question in questions:
        slate = rank_candidate_slate(
            question,
            all_base[question.question_id],
            experts,
            max_challengers,
        )
        top2 = aggregate_order_audited_pre_pair(
            question,
            slate,
            models,
            by_question[question.question_id],
            challenger_limit=1,
        )
        top3 = aggregate_order_audited_pre_pair(
            question, slate, models, by_question[question.question_id]
        )
        rows.append(
            {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "environment": question.environment,
                "candidate_slate": list(slate),
                "candidate_vote_counts": candidate_vote_counts(
                    question, all_base[question.question_id]
                ),
                "predictions": {
                    TOP2_METHOD: top2["prediction"],
                    BUDGET_MATCHED_METHOD: top3["prediction"],
                },
                "top2": top2,
                "budget_matched_top3": top3,
            }
        )
    return rows


def authenticate_completed_pre_pair(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config_path: Path,
    run_root: Path,
) -> tuple[dict[str, dict[str, str | None]], dict[str, float], dict[str, Any]]:
    baseline_config = _load_yaml(baseline_config_path)
    questions, pairwise_rows, quality = authenticate_pre_pair_models(
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
    expected_rows = expected_aggregate_rows(
        questions, pairwise_rows, c3_config, baseline_config, run_root
    )
    actual_rows = _read_jsonl(prediction_path)
    if actual_rows != expected_rows:
        raise PermissionError("PRePair-style aggregate predictions differ from replay")
    if (
        manifest.get("status") != "completed_label_free_pre_pair_style_aggregate"
        or manifest.get("labels_read") is not False
        or int(manifest.get("questions", -1)) != len(questions)
        or manifest.get("prediction_sha256") != sha256_file(prediction_path)
        or manifest.get("c3_config_sha256") != sha256_file(c3_config_path)
        or manifest.get("baseline_config_sha256") != sha256_file(baseline_config_path)
        or manifest.get("model_artifact_hashes")
        != quality["model_artifact_hashes"]
        or manifest.get("aggregator_sha256") != _source_hashes()["aggregator_sha256"]
    ):
        raise PermissionError("PRePair-style aggregate manifest differs")
    predictions = {
        method: {
            str(row["question_id"]): str(row["predictions"][method])
            for row in actual_rows
        }
        for method in (TOP2_METHOD, BUDGET_MATCHED_METHOD)
    }
    models = tuple(str(value) for value in baseline_config["models"])
    max_challengers = int(baseline_config["candidate_selection"]["max_challengers"])
    top2_calls = len(c3_config["experts"]) + len(models) * (
        2 + len(PAIRWISE_ORIENTATIONS)
    )
    _, full_calls = pre_pair_call_budget(c3_config, baseline_config)
    budgets = {TOP2_METHOD: float(top2_calls), BUDGET_MATCHED_METHOD: float(full_calls)}
    return predictions, budgets, quality


def main() -> None:
    args = parse_args()
    c3_config = _load_yaml(args.c3_config)
    baseline_config = _load_yaml(args.baseline_config)
    run_root = args.run_root or Path(str(c3_config["output_root"]))
    questions, pairwise_rows, quality = authenticate_pre_pair_models(
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
    rows = expected_aggregate_rows(
        questions, pairwise_rows, c3_config, baseline_config, run_root
    )
    partial = (
        run_root
        / "prepair_style_attempts"
        / f"aggregate.{args.smoke_questions}.{os.getpid()}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    prediction_path = partial / "predictions.jsonl"
    write_jsonl(prediction_path, rows)
    status = (
        "bounded_label_free_pre_pair_style_aggregate_smoke"
        if args.smoke_questions
        else "completed_label_free_pre_pair_style_aggregate"
    )
    write_json(
        partial / "manifest.json",
        {
            "status": status,
            "protocol_version": int(baseline_config["protocol_version"]),
            "questions": len(questions),
            "methods": [TOP2_METHOD, BUDGET_MATCHED_METHOD],
            "calls_per_question": int(baseline_config["calls_per_question"]),
            "new_control_calls_per_question": quality[
                "new_control_calls_per_question"
            ],
            "model_artifact_hashes": quality["model_artifact_hashes"],
            "labels_read": False,
            "prediction_sha256": sha256_file(prediction_path),
            "c3_config_sha256": sha256_file(args.c3_config),
            "baseline_config_sha256": sha256_file(args.baseline_config),
            "aggregator_sha256": _source_hashes()["aggregator_sha256"],
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(baseline_config["seed"]),
                [
                    args.c3_config,
                    args.baseline_config,
                    run_root / "development_observables" / "questions.jsonl",
                    run_root / "development_observables" / "base_predictions.jsonl",
                    *(
                        _model_root(run_root, model, args.smoke_questions)
                        for model in baseline_config["models"]
                    ),
                ],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed label-free PRePair-style aggregation: {output_dir}")


if __name__ == "__main__":
    main()
