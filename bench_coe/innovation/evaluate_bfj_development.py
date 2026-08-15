from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import environment_manifest, sha256_file, write_csv, write_json, write_jsonl
from .blind_falsification_jury import (
    FORBIDDEN_AUDIT_KEYS,
    AuditObservation,
    BFJDecision,
    BFJVariant,
    BasePrediction,
    BlindFalsificationJury,
    FalsificationQuestion,
    build_falsification_prompt,
    candidate_label_key,
    parse_audit_output,
    uncalibrated_falsification_vote,
)
from .evaluation import exact_mcnemar
from .locked_protocol import stratified_paired_bootstrap_delta
from .schema import SourceTrainingLabels


@dataclass(frozen=True)
class BFJDevelopmentData:
    questions: tuple[FalsificationQuestion, ...]
    base_predictions: tuple[BasePrediction, ...]
    audits: tuple[AuditObservation, ...]
    answers: Mapping[str, str]
    dataset_by_question: Mapping[str, str]
    environment_by_question: Mapping[str, str]
    audit_quality: Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BFJ with nested leave-one-environment-out development folds"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("BFJ configuration must be a mapping")
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


def _source_labels(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    answers: Mapping[str, str],
    environment_by_question: Mapping[str, str],
) -> SourceTrainingLabels:
    correctness: dict[tuple[str, str], bool] = {}
    for question in questions:
        answer = answers[question.question_id]
        for candidate in question.option_labels:
            correctness[(question.question_id, candidate_label_key(candidate))] = (
                candidate == answer
            )
    for row in base_predictions:
        correctness[(row.question_id, row.expert_id)] = (
            row.answer == answers[row.question_id]
        )
    return SourceTrainingLabels._from_source_adapter(
        "bfj_development_union",
        "nested_leave_one_environment_out",
        correctness,
        environment_by_question,
    )


def _expected_audit_keys(
    questions: Sequence[FalsificationQuestion], auditor: str
) -> set[tuple[str, str, str]]:
    return {
        (question.question_id, auditor, candidate)
        for question in questions
        for candidate in question.option_labels
    }


def load_and_authenticate_development_data(
    config_path: Path, config: Mapping[str, Any], run_root: Path
) -> BFJDevelopmentData:
    question_path = run_root / "development_observables" / "questions.jsonl"
    base_path = run_root / "development_observables" / "base_predictions.jsonl"
    observable_manifest_path = (
        run_root / "development_observables" / "observable_manifest.json"
    )
    label_path = run_root / "development_labels" / "labels.jsonl"
    label_manifest_path = run_root / "development_labels" / "label_manifest.json"
    observable_manifest = json.loads(observable_manifest_path.read_text(encoding="utf-8"))
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    if observable_manifest.get("question_sha256") != sha256_file(question_path):
        raise PermissionError("BFJ question observables changed after preparation")
    if observable_manifest.get("base_prediction_sha256") != sha256_file(base_path):
        raise PermissionError("BFJ base predictions changed after preparation")
    if label_manifest.get("label_sha256") != sha256_file(label_path):
        raise PermissionError("BFJ development labels changed after preparation")
    if observable_manifest.get("generation_reads_labels") is not False:
        raise PermissionError("BFJ observable manifest does not enforce label-free generation")

    question_rows = _read_jsonl(question_path)
    base_rows = _read_jsonl(base_path)
    label_rows = _read_jsonl(label_path)
    for row in question_rows + base_rows:
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(f"BFJ observable contains labels: {sorted(leaked)}")
    questions = tuple(
        FalsificationQuestion(
            question_id=str(row["question_id"]),
            dataset=str(row["dataset"]),
            environment=str(row["environment"]),
            question=str(row["question"]),
            options=tuple(str(value) for value in row["options"]),
            option_labels=tuple(str(value) for value in row["option_labels"]),
        )
        for row in question_rows
    )
    question_by_id = {row.question_id: row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("BFJ evaluation questions contain duplicate IDs")
    expected_count = sum(int(row["expected_questions"]) for row in config["datasets"])
    if len(questions) != expected_count:
        raise RuntimeError("BFJ evaluation question count differs from the protocol")

    experts = tuple(str(value) for value in config["experts"])
    base_predictions = tuple(
        BasePrediction(
            question_id=str(row["question_id"]),
            expert_id=str(row["expert_id"]),
            answer=None if row.get("prediction") is None else str(row["prediction"]),
        )
        for row in base_rows
    )
    expected_base_keys = {
        (question.question_id, expert) for question in questions for expert in experts
    }
    actual_base_keys = {(row.question_id, row.expert_id) for row in base_predictions}
    if actual_base_keys != expected_base_keys or len(base_predictions) != len(expected_base_keys):
        raise RuntimeError("BFJ base predictions do not form a complete question/expert grid")

    answers = {str(row["question_id"]): str(row["answer"]) for row in label_rows}
    if set(answers) != set(question_by_id) or len(answers) != len(label_rows):
        raise RuntimeError("BFJ development labels are not aligned one-to-one with questions")
    for question_id, answer in answers.items():
        if answer not in question_by_id[question_id].option_labels:
            raise ValueError(f"BFJ development answer is outside the option set: {question_id}")

    prompt_hash = hashlib.sha256(
        inspect.getsource(build_falsification_prompt).encode("utf-8")
    ).hexdigest()
    parser_hash = hashlib.sha256(
        inspect.getsource(parse_audit_output).encode("utf-8")
    ).hexdigest()
    all_audits: list[AuditObservation] = []
    quality_rows: list[dict[str, Any]] = []
    expected_option_audits = sum(len(question.option_labels) for question in questions)
    for auditor in (str(value) for value in config["auditors"]):
        audit_dir = run_root / "audits" / auditor
        observation_path = audit_dir / "observations.jsonl"
        manifest_path = audit_dir / "audit_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_label_free_bfj_audits":
            raise PermissionError(f"BFJ auditor has no completed manifest: {auditor}")
        if manifest.get("auditor") != auditor or manifest.get("labels_read") is not False:
            raise PermissionError(f"BFJ auditor identity or label boundary is invalid: {auditor}")
        if int(manifest.get("questions", -1)) != len(questions):
            raise RuntimeError(f"BFJ auditor question count is incomplete: {auditor}")
        if int(manifest.get("option_audits", -1)) != expected_option_audits:
            raise RuntimeError(f"BFJ auditor option coverage is incomplete: {auditor}")
        if manifest.get("question_sha256") != sha256_file(question_path):
            raise PermissionError(f"BFJ auditor used different questions: {auditor}")
        if manifest.get("observation_sha256") != sha256_file(observation_path):
            raise PermissionError(f"BFJ auditor observations changed: {auditor}")
        if manifest.get("prompt_builder_sha256") != prompt_hash:
            raise PermissionError(f"BFJ prompt implementation drifted: {auditor}")
        if manifest.get("parser_sha256") != parser_hash:
            raise PermissionError(f"BFJ parser implementation drifted: {auditor}")
        recorded_config_hash = (
            manifest.get("environment", {})
            .get("input_hashes", {})
            .get(str(config_path))
        )
        if recorded_config_hash != sha256_file(config_path):
            raise PermissionError(f"BFJ auditor used a different protocol config: {auditor}")

        rows = _read_jsonl(observation_path)
        actual_keys: set[tuple[str, str, str]] = set()
        parsed = 0
        truncated = 0
        latency = 0.0
        by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(f"BFJ audit contains labels: {sorted(leaked)}")
            question_id = str(row["question_id"])
            candidate = str(row["candidate"])
            identity = (question_id, str(row["auditor_id"]), candidate)
            if identity in actual_keys:
                raise ValueError(f"Duplicate BFJ audit observation: {identity}")
            actual_keys.add(identity)
            question = question_by_id.get(question_id)
            if question is None or candidate not in question.option_labels:
                raise ValueError(f"BFJ audit has an unknown question/candidate: {identity}")
            reparsed = parse_audit_output(str(row["raw_output"]), question.option_labels)
            stored = (
                str(row["verdict"]),
                int(row["confidence"]),
                None if row.get("alternative") is None else str(row["alternative"]),
                None if row.get("parse_error") is None else str(row["parse_error"]),
            )
            if reparsed != stored:
                raise PermissionError(f"BFJ stored parse differs from frozen parser: {identity}")
            observation = AuditObservation(
                question_id=question_id,
                auditor_id=auditor,
                candidate=candidate,
                verdict=stored[0],
                confidence=stored[1],
                alternative=stored[2],
                parse_error=stored[3],
            )
            all_audits.append(observation)
            parsed += int(observation.parse_error is None)
            truncated += int(bool(row["prompt_was_truncated"]))
            latency += float(row["model_latency_seconds"])
            by_dataset[question.dataset]["total"] += 1
            by_dataset[question.dataset]["parsed"] += int(
                observation.parse_error is None
            )
        expected_keys = _expected_audit_keys(questions, auditor)
        if actual_keys != expected_keys or len(rows) != len(expected_keys):
            raise RuntimeError(f"BFJ auditor does not cover every option exactly once: {auditor}")
        if parsed != int(manifest["parsed_audits"]):
            raise RuntimeError(f"BFJ parsed count differs from its manifest: {auditor}")
        if truncated != int(manifest["truncated_prompts"]):
            raise RuntimeError(f"BFJ truncation count differs from its manifest: {auditor}")
        quality_rows.append(
            {
                "auditor": auditor,
                "audits": len(rows),
                "parsed": parsed,
                "parse_rate": parsed / max(1, len(rows)),
                "truncated": truncated,
                "latency_seconds": latency,
                "by_dataset": {
                    dataset: {
                        "audits": int(counts["total"]),
                        "parsed": int(counts["parsed"]),
                        "parse_rate": counts["parsed"] / max(1, counts["total"]),
                    }
                    for dataset, counts in sorted(by_dataset.items())
                },
            }
        )
    if len(all_audits) != expected_option_audits * len(config["auditors"]):
        raise RuntimeError("BFJ combined audit grid is incomplete")
    return BFJDevelopmentData(
        questions=questions,
        base_predictions=base_predictions,
        audits=tuple(all_audits),
        answers=answers,
        dataset_by_question={row.question_id: row.dataset for row in questions},
        environment_by_question={row.question_id: row.environment for row in questions},
        audit_quality={"auditors": quality_rows},
    )


def leave_one_environment_out(
    question_ids: Iterable[str], environment_by_question: Mapping[str, str]
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    ids = set(question_ids)
    by_environment: dict[str, list[str]] = defaultdict(list)
    for question_id in ids:
        by_environment[str(environment_by_question[question_id])].append(question_id)
    if len(by_environment) < 3:
        raise ValueError("Nested BFJ evaluation requires at least three environments")
    return [
        (
            environment,
            tuple(sorted(ids.difference(heldout_ids))),
            tuple(sorted(heldout_ids)),
        )
        for environment, heldout_ids in sorted(by_environment.items())
    ]


def variants_from_config(config: Mapping[str, Any]) -> tuple[BFJVariant, ...]:
    grid = config["variant_grid"]
    names = (
        "prior_strength",
        "evidence_strength",
        "smoothing",
        "use_confidence_bins",
        "calibrate_self_bias",
        "open_option_set",
        "intervention_margin",
    )
    values = [tuple(grid[name]) for name in names]
    variants: list[BFJVariant] = []
    for index, combination in enumerate(itertools.product(*values)):
        parameters = dict(zip(names, combination, strict=True))
        variants.append(
            BFJVariant(
                name=f"bfj_grid_{index:03d}",
                prior_strength=float(parameters["prior_strength"]),
                evidence_strength=float(parameters["evidence_strength"]),
                smoothing=float(parameters["smoothing"]),
                use_confidence_bins=bool(parameters["use_confidence_bins"]),
                calibrate_self_bias=bool(parameters["calibrate_self_bias"]),
                open_option_set=bool(parameters["open_option_set"]),
                intervention_margin=float(parameters["intervention_margin"]),
                max_abs_log_likelihood_ratio=float(
                    grid["max_abs_log_likelihood_ratio"]
                ),
                confidence_threshold=int(grid["confidence_threshold"]),
            )
        )
    if len({variant.name for variant in variants}) != len(variants):
        raise AssertionError("BFJ variant names are not unique")
    return tuple(variants)


def _subset_rows(rows: Sequence[Any], question_ids: Iterable[str]) -> list[Any]:
    keep = set(question_ids)
    return [row for row in rows if row.question_id in keep]


def _structure_key(variant: BFJVariant) -> tuple[bool, bool, int]:
    return (
        variant.use_confidence_bins,
        variant.calibrate_self_bias,
        variant.confidence_threshold,
    )


def _answers_from_decisions(decisions: Sequence[BFJDecision]) -> dict[str, str]:
    return {row.question_id: row.answer for row in decisions}


def select_variant_nested(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    audits: Sequence[AuditObservation],
    labels: SourceTrainingLabels,
    answers: Mapping[str, str],
    variants: Sequence[BFJVariant],
) -> tuple[BFJVariant, list[dict[str, Any]]]:
    question_ids = tuple(sorted(question.question_id for question in questions))
    folds = leave_one_environment_out(question_ids, labels.environment_by_question)
    variant_index = {variant.name: index for index, variant in enumerate(variants)}
    correct_by_environment: dict[str, dict[str, int]] = {
        variant.name: {} for variant in variants
    }
    count_by_environment: dict[str, int] = {}
    variants_by_structure: dict[tuple[bool, bool, int], list[BFJVariant]] = defaultdict(list)
    for variant in variants:
        variants_by_structure[_structure_key(variant)].append(variant)

    question_by_id = {row.question_id: row for row in questions}
    for inner_environment, train_ids, validation_ids in folds:
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        validation_questions = [question_by_id[question_id] for question_id in validation_ids]
        train_base = _subset_rows(base_predictions, train_ids)
        validation_base = _subset_rows(base_predictions, validation_ids)
        train_audits = _subset_rows(audits, train_ids)
        validation_audits = _subset_rows(audits, validation_ids)
        train_labels = labels.subset(train_ids)
        count_by_environment[inner_environment] = len(validation_ids)
        for grouped_variants in variants_by_structure.values():
            fitted = BlindFalsificationJury(grouped_variants[0]).fit(
                train_questions, train_base, train_audits, train_labels
            )
            for variant in grouped_variants:
                predictor = copy.copy(fitted)
                predictor.variant = variant
                predicted = _answers_from_decisions(
                    predictor.predict(
                        validation_questions, validation_base, validation_audits
                    )
                )
                correct_by_environment[variant.name][inner_environment] = sum(
                    predicted[question_id] == answers[question_id]
                    for question_id in validation_ids
                )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        environment_accuracies = [
            correct_by_environment[variant.name][environment]
            / count_by_environment[environment]
            for environment in sorted(count_by_environment)
        ]
        total_correct = sum(correct_by_environment[variant.name].values())
        total_count = sum(count_by_environment.values())
        rows.append(
            {
                **asdict(variant),
                "inner_environment_count": len(environment_accuracies),
                "macro_environment_accuracy": float(np.mean(environment_accuracies)),
                "micro_accuracy": total_correct / max(1, total_count),
                "correct": total_correct,
                "samples": total_count,
                "variant_order": variant_index[variant.name],
                "selected": False,
            }
        )
    selected_row = sorted(
        rows,
        key=lambda row: (
            -float(row["macro_environment_accuracy"]),
            -float(row["micro_accuracy"]),
            int(row["variant_order"]),
        ),
    )[0]
    selected_row["selected"] = True
    selected = variants[variant_index[str(selected_row["name"])]]
    return selected, rows


def _expert_accuracies(
    experts: Sequence[str], question_ids: Iterable[str], labels: SourceTrainingLabels
) -> dict[str, float]:
    ids = tuple(question_ids)
    return {
        expert: sum(bool(labels.get(question_id, expert)) for question_id in ids)
        / max(1, len(ids))
        for expert in experts
    }


def _tie_break(candidates: Iterable[str], reference: str) -> str:
    values = sorted(set(candidates))
    if not values:
        raise ValueError("Cannot resolve an empty BFJ candidate set")
    return reference if reference in values else values[0]


def _majority_answer(
    question: FalsificationQuestion,
    base_by_expert: Mapping[str, str | None],
    reference: str,
) -> str:
    counts = Counter(
        answer for answer in base_by_expert.values() if answer in question.option_labels
    )
    if not counts:
        return reference if reference in question.option_labels else question.option_labels[0]
    maximum = max(counts.values())
    return _tie_break((answer for answer, count in counts.items() if count == maximum), reference)


def _weighted_answer(
    question: FalsificationQuestion,
    base_by_expert: Mapping[str, str | None],
    expert_accuracy: Mapping[str, float],
    reference: str,
) -> str:
    scores = {label: 0.0 for label in question.option_labels}
    for expert, answer in base_by_expert.items():
        if answer in scores:
            scores[str(answer)] += expert_accuracy[expert]
    maximum = max(scores.values())
    return _tie_break(
        (answer for answer, value in scores.items() if abs(value - maximum) <= 1e-12),
        reference,
    )


def _fit_predict(
    variant: BFJVariant,
    train_questions: Sequence[FalsificationQuestion],
    train_base: Sequence[BasePrediction],
    train_audits: Sequence[AuditObservation],
    train_labels: SourceTrainingLabels,
    target_questions: Sequence[FalsificationQuestion],
    target_base: Sequence[BasePrediction],
    target_audits: Sequence[AuditObservation],
) -> tuple[BlindFalsificationJury, list[BFJDecision]]:
    model = BlindFalsificationJury(variant).fit(
        train_questions, train_base, train_audits, train_labels
    )
    return model, model.predict(target_questions, target_base, target_audits)


def _ablation_variants(selected: BFJVariant) -> dict[str, BFJVariant]:
    return {
        "bfj_no_self_bias_calibration": replace(
            selected, name="bfj_no_self_bias_calibration", calibrate_self_bias=False
        ),
        "bfj_closed_candidate_set": replace(
            selected, name="bfj_closed_candidate_set", open_option_set=False
        ),
        "bfj_no_confidence_bins": replace(
            selected, name="bfj_no_confidence_bins", use_confidence_bins=False
        ),
        "bfj_no_audit_evidence": replace(
            selected, name="bfj_no_audit_evidence", evidence_strength=0.0
        ),
        "bfj_no_base_prior": replace(
            selected, name="bfj_no_base_prior", prior_strength=0.0
        ),
    }


def generate_nested_oof_predictions(
    config: Mapping[str, Any], data: BFJDevelopmentData
) -> tuple[
    dict[str, dict[str, str | None]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    experts = tuple(str(value) for value in config["experts"])
    auditors = tuple(str(value) for value in config["auditors"])
    variants = variants_from_config(config)
    labels = _source_labels(
        data.questions,
        data.base_predictions,
        data.answers,
        data.environment_by_question,
    )
    question_by_id = {row.question_id: row for row in data.questions}
    base_by_question: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in data.base_predictions:
        base_by_question[row.question_id][row.expert_id] = row.answer
    audits_by_question: dict[str, list[AuditObservation]] = defaultdict(list)
    for row in data.audits:
        audits_by_question[row.question_id].append(row)

    predictions: dict[str, dict[str, str | None]] = {
        f"single::{expert}": {
            question_id: base_by_question[question_id][expert]
            for question_id in sorted(question_by_id)
        }
        for expert in experts
    }
    core_methods = (
        "best_single_nested_oof",
        "majority_vote",
        "source_weighted_vote",
        "uncalibrated_falsification_vote",
        "bfj_primary_nested",
        "bfj_no_self_bias_calibration",
        "bfj_closed_candidate_set",
        "bfj_no_confidence_bins",
        "bfj_no_audit_evidence",
        "bfj_no_base_prior",
    )
    predictions.update({method: {} for method in core_methods})
    predictions.update({f"bfj_single_auditor::{auditor}": {} for auditor in auditors})
    nested_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    primary_diagnostics: dict[str, dict[str, Any]] = {}
    outer_folds = leave_one_environment_out(
        question_by_id, data.environment_by_question
    )
    for outer_index, (outer_environment, train_ids, heldout_ids) in enumerate(outer_folds):
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        heldout_questions = [question_by_id[question_id] for question_id in heldout_ids]
        train_base = _subset_rows(data.base_predictions, train_ids)
        heldout_base = _subset_rows(data.base_predictions, heldout_ids)
        train_audits = _subset_rows(data.audits, train_ids)
        heldout_audits = _subset_rows(data.audits, heldout_ids)
        train_labels = labels.subset(train_ids)
        selected, search_rows = select_variant_nested(
            train_questions,
            train_base,
            train_audits,
            train_labels,
            data.answers,
            variants,
        )
        for row in search_rows:
            nested_rows.append({"outer_environment": outer_environment, **row})

        expert_accuracy = _expert_accuracies(experts, train_ids, train_labels)
        reference_expert = sorted(
            experts, key=lambda expert: (-expert_accuracy[expert], expert)
        )[0]
        primary_model, primary_decisions = _fit_predict(
            selected,
            train_questions,
            train_base,
            train_audits,
            train_labels,
            heldout_questions,
            heldout_base,
            heldout_audits,
        )
        if primary_model.reference_expert_ != reference_expert:
            raise AssertionError("BFJ and baseline source-fitted reference experts differ")
        primary_by_id = {row.question_id: row for row in primary_decisions}
        for question_id in heldout_ids:
            question = question_by_id[question_id]
            base = base_by_question[question_id]
            reference_answer = base.get(reference_expert)
            if reference_answer not in question.option_labels:
                reference_answer = question.option_labels[0]
            predictions["best_single_nested_oof"][question_id] = reference_answer
            predictions["majority_vote"][question_id] = _majority_answer(
                question, base, str(reference_answer)
            )
            predictions["source_weighted_vote"][question_id] = _weighted_answer(
                question, base, expert_accuracy, str(reference_answer)
            )
            predictions["uncalibrated_falsification_vote"][question_id] = (
                uncalibrated_falsification_vote(
                    question,
                    audits_by_question[question_id],
                    str(reference_answer),
                )
            )
            decision = primary_by_id[question_id]
            predictions["bfj_primary_nested"][question_id] = decision.answer
            primary_diagnostics[question_id] = {
                **dict(decision.diagnostics),
                "outer_environment": outer_environment,
                "outer_reference_expert": reference_expert,
                "selected_variant": asdict(selected),
                "candidate_scores": dict(decision.candidate_scores),
                "candidate_probabilities": dict(decision.candidate_probabilities),
                "fallback_reason": decision.fallback_reason,
                "open_set_rescue": decision.open_set_rescue,
            }

        for method, variant in _ablation_variants(selected).items():
            _, decisions = _fit_predict(
                variant,
                train_questions,
                train_base,
                train_audits,
                train_labels,
                heldout_questions,
                heldout_base,
                heldout_audits,
            )
            predictions[method].update(_answers_from_decisions(decisions))
        for auditor in auditors:
            train_single = [row for row in train_audits if row.auditor_id == auditor]
            heldout_single = [row for row in heldout_audits if row.auditor_id == auditor]
            auditor_variant = replace(selected, name=f"bfj_single_auditor::{auditor}")
            _, decisions = _fit_predict(
                auditor_variant,
                train_questions,
                train_base,
                train_single,
                train_labels,
                heldout_questions,
                heldout_base,
                heldout_single,
            )
            predictions[f"bfj_single_auditor::{auditor}"].update(
                _answers_from_decisions(decisions)
            )
        outer_rows.append(
            {
                "outer_fold": outer_index,
                "outer_environment": outer_environment,
                "train_questions": len(train_ids),
                "heldout_questions": len(heldout_ids),
                "reference_expert": reference_expert,
                "reference_train_accuracy": expert_accuracy[reference_expert],
                "selected_variant": asdict(selected),
            }
        )

    expected_ids = set(question_by_id)
    for method, values in predictions.items():
        if set(values) != expected_ids:
            raise RuntimeError(f"Nested BFJ method has incomplete OOF predictions: {method}")
    individual_accuracy = {
        expert: sum(
            predictions[f"single::{expert}"][question_id] == data.answers[question_id]
            for question_id in expected_ids
        )
        / len(expected_ids)
        for expert in experts
    }
    descriptive_best = sorted(
        experts, key=lambda expert: (-individual_accuracy[expert], expert)
    )[0]
    predictions["full_development_best_single_descriptive"] = dict(
        predictions[f"single::{descriptive_best}"]
    )
    for row in outer_rows:
        row["full_development_best_single_descriptive"] = descriptive_best
        row["full_development_best_single_accuracy"] = individual_accuracy[descriptive_best]
    return predictions, nested_rows, outer_rows, primary_diagnostics


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _correctness(
    predictions: Mapping[str, str | None], answers: Mapping[str, str]
) -> dict[str, bool]:
    if set(predictions) != set(answers):
        raise ValueError("BFJ metric inputs are not aligned")
    return {
        question_id: predictions[question_id] == answer
        for question_id, answer in answers.items()
    }


def _method_summary(
    method: str,
    predictions: Mapping[str, str | None],
    data: BFJDevelopmentData,
    nominal_calls: float,
) -> dict[str, Any]:
    correct = _correctness(predictions, data.answers)
    successes = sum(correct.values())
    by_dataset: dict[str, list[bool]] = defaultdict(list)
    by_environment: dict[str, list[bool]] = defaultdict(list)
    for question_id, value in correct.items():
        by_dataset[data.dataset_by_question[question_id]].append(value)
        by_environment[data.environment_by_question[question_id]].append(value)
    ci = _wilson_interval(successes, len(correct))
    return {
        "method": method,
        "samples": len(correct),
        "correct": successes,
        "accuracy": successes / max(1, len(correct)),
        "wilson_ci95": list(ci),
        "macro_dataset_accuracy": float(
            np.mean([np.mean(values) for values in by_dataset.values()])
        ),
        "macro_environment_accuracy": float(
            np.mean([np.mean(values) for values in by_environment.values()])
        ),
        "per_dataset_accuracy": {
            key: float(np.mean(values)) for key, values in sorted(by_dataset.items())
        },
        "per_environment_accuracy": {
            key: float(np.mean(values)) for key, values in sorted(by_environment.items())
        },
        "nominal_model_calls_per_question": nominal_calls,
    }


def _comparison(
    candidate_name: str,
    reference_name: str,
    predictions: Mapping[str, Mapping[str, str | None]],
    data: BFJDevelopmentData,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    candidate = _correctness(predictions[candidate_name], data.answers)
    reference = _correctness(predictions[reference_name], data.answers)
    ids = sorted(candidate)
    candidate_array = np.asarray([candidate[question_id] for question_id in ids], dtype=int)
    reference_array = np.asarray([reference[question_id] for question_id in ids], dtype=int)
    rescue, harm, p_value = exact_mcnemar(candidate_array, reference_array)
    ci = stratified_paired_bootstrap_delta(
        candidate,
        reference,
        data.environment_by_question,
        seed=seed,
        samples=samples,
    )
    per_dataset_delta: dict[str, float] = {}
    for dataset in sorted(set(data.dataset_by_question.values())):
        dataset_ids = [
            question_id
            for question_id in ids
            if data.dataset_by_question[question_id] == dataset
        ]
        per_dataset_delta[dataset] = float(
            np.mean(
                [
                    float(candidate[question_id]) - float(reference[question_id])
                    for question_id in dataset_ids
                ]
            )
        )
    return {
        "comparison": f"{candidate_name}_vs_{reference_name}",
        "candidate": candidate_name,
        "reference": reference_name,
        "samples": len(ids),
        "candidate_accuracy": float(candidate_array.mean()),
        "reference_accuracy": float(reference_array.mean()),
        "delta": float((candidate_array - reference_array).mean()),
        "rescue_count": rescue,
        "harm_count": harm,
        "exact_mcnemar_p": p_value,
        "stratified_paired_bootstrap_delta_ci95": list(ci),
        "per_dataset_delta": per_dataset_delta,
    }


def _call_budget(method: str, config: Mapping[str, Any], data: BFJDevelopmentData) -> float:
    expert_count = len(config["experts"])
    mean_options = float(np.mean([len(question.option_labels) for question in data.questions]))
    if method.startswith("single::") or method in {
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
    }:
        return 1.0
    if method in {"majority_vote", "source_weighted_vote"}:
        return float(expert_count)
    auditor_calls = mean_options * (
        1 if method.startswith("bfj_single_auditor::") else len(config["auditors"])
    )
    return float(expert_count) + auditor_calls


def evaluate_predictions(
    config: Mapping[str, Any],
    data: BFJDevelopmentData,
    predictions: Mapping[str, Mapping[str, str | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method_rows = [
        _method_summary(method, values, data, _call_budget(method, config, data))
        for method, values in sorted(predictions.items())
    ]
    primary = "bfj_primary_nested"
    references = [
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
        "majority_vote",
        "source_weighted_vote",
        "uncalibrated_falsification_vote",
        "bfj_no_self_bias_calibration",
        "bfj_closed_candidate_set",
        "bfj_no_confidence_bins",
        "bfj_no_audit_evidence",
        "bfj_no_base_prior",
    ]
    references.extend(
        method for method in predictions if method.startswith("bfj_single_auditor::")
    )
    comparison_config = config["outer_validation"]
    comparisons = [
        _comparison(
            primary,
            reference,
            predictions,
            data,
            seed=int(comparison_config["bootstrap_seed"]) + index,
            samples=int(comparison_config["bootstrap_samples"]),
        )
        for index, reference in enumerate(references)
    ]
    by_reference = {row["reference"]: row for row in comparisons}
    primary_comparison = by_reference["best_single_nested_oof"]
    descriptive_comparison = by_reference["full_development_best_single_descriptive"]
    acceptance = config["acceptance"]
    minimum_delta = float(acceptance["minimum_oof_delta_vs_deployable_best_single_pp"]) / 100.0
    checks = {
        "delta_vs_deployable_best_single": float(primary_comparison["delta"])
        >= minimum_delta,
        "nonnegative_on_every_dataset": all(
            float(value)
            >= float(acceptance["minimum_per_dataset_delta_pp"]) / 100.0
            for value in primary_comparison["per_dataset_delta"].values()
        ),
        "paired_ci_low_above_zero": (
            float(primary_comparison["stratified_paired_bootstrap_delta_ci95"][0])
            > 0.0
            if bool(acceptance["require_paired_bootstrap_ci_low_gt_zero"])
            else True
        ),
        "mcnemar_significant": (
            float(primary_comparison["exact_mcnemar_p"])
            < float(acceptance["require_two_sided_mcnemar_p_lt"])
        ),
        "all_required_ablations_present": all(
            reference in by_reference
            for reference in (
                "bfj_no_self_bias_calibration",
                "bfj_closed_candidate_set",
                "bfj_no_confidence_bins",
                "bfj_no_audit_evidence",
                "bfj_no_base_prior",
            )
        ),
    }
    decision = {
        "status": "development_gate_pass" if all(checks.values()) else "development_gate_fail",
        "checks": checks,
        "required_delta": minimum_delta,
        "primary_vs_deployable_best_single": primary_comparison,
        "primary_vs_full_development_best_single_descriptive": descriptive_comparison,
        "beats_every_fixed_single_model": all(
            next(row for row in method_rows if row["method"] == primary)["accuracy"]
            > row["accuracy"]
            for row in method_rows
            if str(row["method"]).startswith("single::")
        ),
        "blind_test_authorized": bool(all(checks.values()))
        and bool(acceptance["blind_test_authorized_only_after_all_development_gates_pass"]),
        "claim_boundary": (
            "Development-only nested OOF evidence; no confirmatory or global novelty claim."
        ),
    }
    return method_rows, comparisons, decision


def _plot_results(
    output_dir: Path,
    method_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchcoe_bfj_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method = {str(row["method"]): row for row in method_rows}
    shortlist = [
        "bfj_primary_nested",
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
        "majority_vote",
        "source_weighted_vote",
        "uncalibrated_falsification_vote",
        "bfj_no_self_bias_calibration",
        "bfj_closed_candidate_set",
        "bfj_no_audit_evidence",
    ]
    shortlist = [method for method in shortlist if method in by_method]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.05, 1.0]})
    positions = np.arange(len(shortlist))
    accuracies = [100.0 * float(by_method[method]["accuracy"]) for method in shortlist]
    colors = ["#147D64" if method == "bfj_primary_nested" else "#66717E" for method in shortlist]
    axes[0].barh(positions, accuracies, color=colors)
    axes[0].set_yticks(positions, shortlist)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Nested OOF accuracy (%)")
    axes[0].set_title("BFJ development methods")
    axes[0].grid(axis="x", alpha=0.2)
    for position, value in zip(positions, accuracies, strict=True):
        axes[0].text(value + 0.15, position, f"{value:.2f}", va="center", fontsize=8)

    shown = [
        row
        for row in comparisons
        if row["reference"]
        in {
            "best_single_nested_oof",
            "full_development_best_single_descriptive",
            "majority_vote",
            "source_weighted_vote",
            "uncalibrated_falsification_vote",
            "bfj_no_self_bias_calibration",
            "bfj_closed_candidate_set",
            "bfj_no_audit_evidence",
        }
    ]
    positions = np.arange(len(shown))
    deltas = np.asarray([100.0 * float(row["delta"]) for row in shown])
    lows = np.asarray(
        [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][0]) for row in shown]
    )
    highs = np.asarray(
        [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][1]) for row in shown]
    )
    axes[1].errorbar(
        deltas,
        positions,
        xerr=np.vstack((deltas - lows, highs - deltas)),
        fmt="o",
        color="#263238",
        ecolor="#147D64",
        capsize=4,
    )
    axes[1].axvline(0.0, color="#B3261E", linewidth=1.2)
    axes[1].set_yticks(positions, [str(row["reference"]) for row in shown])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("BFJ paired delta (percentage points)")
    axes[1].set_title("Environment-stratified paired comparisons")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Blind Falsification Jury: nested development evaluation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "bfj_development_results.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "bfj_development_results.pdf", bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    config: Mapping[str, Any],
    method_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    priority = {
        "bfj_primary_nested": 0,
        "best_single_nested_oof": 1,
        "full_development_best_single_descriptive": 2,
        "majority_vote": 3,
        "source_weighted_vote": 4,
        "uncalibrated_falsification_vote": 5,
    }
    shown = sorted(
        method_rows,
        key=lambda row: (priority.get(str(row["method"]), 20), str(row["method"])),
    )
    lines = [
        "# Blind Falsification Jury development result",
        "",
        f"Decision: **{decision['status']}**",
        "",
        "This report is development-only. Every reported BFJ prediction is from an outer "
        "leave-one-environment-out fold; its variant was selected by a second leave-one-environment-out "
        "loop inside the outer training fold. Held-out labels never entered fitting or selection.",
        "",
        "## Accuracy",
        "",
        "| Method | Calls/question | Micro | MMLU-Pro | GPQA | Environment macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in shown:
        per_dataset = row["per_dataset_accuracy"]
        lines.append(
            f"| {row['method']} | {float(row['nominal_model_calls_per_question']):.2f} | "
            f"{100*float(row['accuracy']):.2f}% | "
            f"{100*float(per_dataset.get('mmlu_pro', 0.0)):.2f}% | "
            f"{100*float(per_dataset.get('gpqa', 0.0)):.2f}% | "
            f"{100*float(row['macro_environment_accuracy']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Reference | Delta | Stratified 95% CI | Rescue/Harm | McNemar p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        ci = row["stratified_paired_bootstrap_delta_ci95"]
        lines.append(
            f"| {row['reference']} | {100*float(row['delta']):+.2f} pp | "
            f"[{100*float(ci[0]):+.2f}, {100*float(ci[1]):+.2f}] pp | "
            f"{row['rescue_count']}/{row['harm_count']} | "
            f"{float(row['exact_mcnemar_p']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["checks"].items()],
            "",
            f"Blind test authorized: **{decision['blind_test_authorized']}**.",
            "",
            "The full-development best single is descriptive because its identity was selected with "
            "all development labels. The nested OOF best-single baseline is the deployable primary "
            "reference. Pretraining contamination is not claimed to be excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    run_root = args.run_root or Path(str(config["output_root"]))
    output_dir = args.output_dir or run_root / "development_evaluation"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started = time.time()
    data = load_and_authenticate_development_data(args.config, config, run_root)
    predictions, nested_rows, outer_rows, primary_diagnostics = (
        generate_nested_oof_predictions(config, data)
    )
    method_rows, comparisons, decision = evaluate_predictions(config, data, predictions)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_rows = [
        {
            "question_id": question_id,
            "dataset": data.dataset_by_question[question_id],
            "environment": data.environment_by_question[question_id],
            "development_answer": data.answers[question_id],
            "predictions": {
                method: values[question_id] for method, values in sorted(predictions.items())
            },
            "correctness": {
                method: values[question_id] == data.answers[question_id]
                for method, values in sorted(predictions.items())
            },
            "bfj_primary_diagnostics": primary_diagnostics[question_id],
        }
        for question_id in sorted(data.answers)
    ]
    write_jsonl(output_dir / "per_query_predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "nested_variant_search.jsonl", nested_rows)
    write_jsonl(output_dir / "outer_fold_selections.jsonl", outer_rows)
    write_json(output_dir / "method_summaries.json", method_rows)
    write_csv(
        output_dir / "method_summaries.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in method_rows
        ],
    )
    write_json(output_dir / "paired_comparisons.json", comparisons)
    write_csv(
        output_dir / "paired_comparisons.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in comparisons
        ],
    )
    write_json(output_dir / "audit_quality.json", data.audit_quality)
    write_json(output_dir / "development_gate.json", decision)
    _plot_results(output_dir, method_rows, comparisons)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(
        _markdown_report(config, method_rows, comparisons, decision), encoding="utf-8"
    )
    artifact_paths = [
        output_dir / "per_query_predictions.jsonl",
        output_dir / "nested_variant_search.jsonl",
        output_dir / "outer_fold_selections.jsonl",
        output_dir / "method_summaries.json",
        output_dir / "paired_comparisons.json",
        output_dir / "audit_quality.json",
        output_dir / "development_gate.json",
        report_path,
        output_dir / "bfj_development_results.png",
        output_dir / "bfj_development_results.pdf",
    ]
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "status": decision["status"],
            "protocol": "outer LOEO with inner LOEO variant selection",
            "development_only": True,
            "questions": len(data.questions),
            "methods": len(predictions),
            "started_unix": started,
            "finished_unix": time.time(),
            "artifact_hashes": {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in artifact_paths
            },
            "environment": environment_manifest(
                sys.argv,
                int(config["seed"]),
                [
                    args.config,
                    run_root / "development_observables",
                    run_root / "development_labels",
                    run_root / "audits",
                ],
            ),
        },
    )
    print(f"Completed nested BFJ development evaluation: {output_dir}")


if __name__ == "__main__":
    main()
