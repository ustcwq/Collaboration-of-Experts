from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import (
    environment_manifest,
    innovation_code_manifest,
    manifest_sha256,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from .blind_falsification_jury import (
    FORBIDDEN_AUDIT_KEYS,
    BasePrediction,
    FalsificationQuestion,
)
from .evaluate_bfj_development import (
    _comparison,
    _expert_accuracies,
    _majority_answer,
    _method_summary,
    _source_labels,
    _subset_rows,
    _weighted_answer,
    leave_one_environment_out,
)
from .schema import SourceTrainingLabels
from .sealed_diagnostic_bijection import (
    CandidatePairAssignment,
    PresentedDiagnosticProbe,
    assign_candidate_pairs,
    build_blind_probe_check_prompt,
    build_diagnostic_probe_prompt,
    parse_blind_probe_check_output,
    parse_diagnostic_probe_output,
    present_diagnostic_probe,
    presented_left_authored_outcome,
    reveal_probe_candidate,
)
from .sealed_diagnostic_court import (
    DiagnosticProbe,
    DiagnosticProbeCheck,
    SDBDecision,
    SDBVariant,
    SealedDiagnosticBijectionCourt,
)


@dataclass(frozen=True)
class SDBDevelopmentData:
    questions: tuple[FalsificationQuestion, ...]
    base_predictions: tuple[BasePrediction, ...]
    probes: tuple[DiagnosticProbe, ...]
    checks: tuple[DiagnosticProbeCheck, ...]
    answers: Mapping[str, str]
    dataset_by_question: Mapping[str, str]
    environment_by_question: Mapping[str, str]
    base_response_by_key: Mapping[tuple[str, str], str]
    generation_quality: Mapping[str, Any]
    input_artifact_hashes: Mapping[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and evaluate SDB with nested development OOF"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
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


def _recorded_input_hash(environment: Mapping[str, Any], path: Path) -> str | None:
    inputs = environment.get("input_hashes", {})
    if not isinstance(inputs, dict):
        return None
    direct = inputs.get(str(path))
    if isinstance(direct, str):
        return direct
    resolved = path.resolve()
    for raw_path, value in inputs.items():
        try:
            if Path(str(raw_path)).resolve() == resolved and isinstance(value, str):
                return value
        except OSError:
            continue
    return None


def _assert_current_code(manifest: Mapping[str, Any], identity: str) -> None:
    current = manifest_sha256(innovation_code_manifest())
    recorded = manifest.get("environment", {}).get(
        "innovation_code_manifest_sha256"
    )
    if recorded != current:
        raise PermissionError(f"SDB innovation code drifted after {identity}")


def _observable_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    root = Path(str(config["input_observables_root"]))
    return (
        root / "development_observables" / "questions.jsonl",
        root / "development_observables" / "base_predictions.jsonl",
        root / "development_observables" / "observable_manifest.json",
        root / "development_labels" / "labels.jsonl",
        root / "development_labels" / "label_manifest.json",
    )


def _load_observables_and_labels(
    config: Mapping[str, Any],
) -> tuple[
    tuple[FalsificationQuestion, ...],
    tuple[BasePrediction, ...],
    dict[str, str],
    dict[tuple[str, str], str],
    dict[str, str],
]:
    question_path, base_path, observable_manifest_path, label_path, label_manifest_path = (
        _observable_paths(config)
    )
    observable_manifest = json.loads(
        observable_manifest_path.read_text(encoding="utf-8")
    )
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    question_hash = sha256_file(question_path)
    base_hash = sha256_file(base_path)
    label_hash = sha256_file(label_path)
    configured = config.get("input_hashes", {})
    if (
        observable_manifest.get("question_sha256") != question_hash
        or configured.get("questions_sha256") != question_hash
    ):
        raise PermissionError("SDB questions changed after protocol freeze")
    if (
        observable_manifest.get("base_prediction_sha256") != base_hash
        or configured.get("base_predictions_sha256") != base_hash
    ):
        raise PermissionError("SDB base predictions changed after protocol freeze")
    if label_manifest.get("label_sha256") != label_hash:
        raise PermissionError("SDB development labels changed after preparation")
    if observable_manifest.get("generation_reads_labels") is not False:
        raise PermissionError("SDB observable manifest lacks a label-free boundary")

    question_rows = _read_jsonl(question_path)
    base_rows = _read_jsonl(base_path)
    for row in question_rows + base_rows:
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(f"SDB observable contains labels: {sorted(leaked)}")
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
        raise ValueError("SDB development questions contain duplicate IDs")
    experts = tuple(str(value) for value in config["experts"])
    base_predictions = tuple(
        BasePrediction(
            question_id=str(row["question_id"]),
            expert_id=str(row["expert_id"]),
            answer=None if row.get("prediction") is None else str(row["prediction"]),
        )
        for row in base_rows
    )
    expected_base = {
        (question.question_id, expert) for question in questions for expert in experts
    }
    actual_base = {(row.question_id, row.expert_id) for row in base_predictions}
    if actual_base != expected_base or len(base_predictions) != len(expected_base):
        raise RuntimeError("SDB base predictions are not a complete question/expert grid")
    base_responses = {
        (str(row["question_id"]), str(row["expert_id"])): str(
            row.get("response", "")
        )
        for row in base_rows
    }
    label_rows = _read_jsonl(label_path)
    answers = {str(row["question_id"]): str(row["answer"]) for row in label_rows}
    if set(answers) != set(question_by_id) or len(answers) != len(label_rows):
        raise RuntimeError("SDB development answers are not aligned one-to-one")
    for question_id, answer in answers.items():
        if answer not in question_by_id[question_id].option_labels:
            raise ValueError("SDB development answer is outside its query option set")
    hashes = {
        str(question_path): question_hash,
        str(base_path): base_hash,
        str(observable_manifest_path): sha256_file(observable_manifest_path),
        str(label_path): label_hash,
        str(label_manifest_path): sha256_file(label_manifest_path),
    }
    return questions, base_predictions, answers, base_responses, hashes


def _base_answers_by_question(
    base_predictions: Sequence[BasePrediction],
) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in base_predictions:
        result[row.question_id][row.expert_id] = row.answer
    return dict(result)


def _assignment_for_row(
    assignments: Mapping[str, Mapping[str, CandidatePairAssignment]],
    row: Mapping[str, Any],
) -> CandidatePairAssignment:
    question_id = str(row["question_id"])
    author = str(row["author_id"])
    assignment = assignments.get(question_id, {}).get(author)
    if assignment is None:
        raise ValueError("SDB probe has no deterministic candidate assignment")
    return assignment


def _authenticate_probes(
    config_path: Path,
    config: Mapping[str, Any],
    run_root: Path,
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    base_responses: Mapping[tuple[str, str], str],
) -> tuple[
    tuple[DiagnosticProbe, ...],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    question_by_id = {row.question_id: row for row in questions}
    experts = tuple(str(value) for value in config["experts"])
    authors = tuple(str(value) for value in config["author_models"])
    assignments = {
        question.question_id: assign_candidate_pairs(
            question,
            authors,
            _base_answers_by_question(base_predictions)[question.question_id],
            experts,
        )
        for question in questions
    }
    prompt_hash = hashlib.sha256(
        inspect.getsource(build_diagnostic_probe_prompt).encode("utf-8")
    ).hexdigest()
    parser_hash = hashlib.sha256(
        inspect.getsource(parse_diagnostic_probe_output).encode("utf-8")
    ).hexdigest()
    assignment_hash = hashlib.sha256(
        inspect.getsource(assign_candidate_pairs).encode("utf-8")
    ).hexdigest()
    presentation_hash = hashlib.sha256(
        inspect.getsource(present_diagnostic_probe).encode("utf-8")
    ).hexdigest()
    probes: list[DiagnosticProbe] = []
    row_by_id: dict[str, dict[str, Any]] = {}
    quality: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    generation = config["probe_generation"]
    gates = config["smoke_acceptance"]
    for author in authors:
        directory = run_root / "probes" / author
        path = directory / "probes.jsonl"
        manifest_path = directory / "probe_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "completed_label_free_sdb_probes"
            or manifest.get("author") != author
            or manifest.get("labels_read") is not False
        ):
            raise PermissionError(f"Incomplete or label-crossed SDB author: {author}")
        _assert_current_code(manifest, f"probe author {author}")
        if manifest.get("probe_sha256") != sha256_file(path):
            raise PermissionError(f"SDB probe file changed: {author}")
        if _recorded_input_hash(manifest.get("environment", {}), config_path) != sha256_file(
            config_path
        ):
            raise PermissionError(f"SDB author used a different config: {author}")
        expected_manifest = {
            "prompt_version": generation["prompt_version"],
            "parser_version": generation["parser_version"],
            "prompt_builder_sha256": prompt_hash,
            "parser_sha256": parser_hash,
            "pair_assignment_sha256": assignment_hash,
            "presentation_sha256": presentation_hash,
            "mapping_was_sealed_from_checkers": True,
            "original_task_was_sealed_from_checkers": True,
            "post_commit_permutation": True,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise PermissionError(f"SDB probe protocol drifted: {author}")
        rows = _read_jsonl(path)
        if len(rows) != len(questions):
            raise RuntimeError(f"SDB author lacks complete question coverage: {author}")
        parsed = 0
        nonabstaining = 0
        bijections = 0
        truncated = 0
        seen_questions: set[str] = set()
        by_dataset: Counter[str] = Counter()
        for row in rows:
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(f"SDB probe contains labels: {sorted(leaked)}")
            question = question_by_id.get(str(row["question_id"]))
            if question is None or str(row["author_id"]) != author:
                raise ValueError("SDB probe identity is not registered")
            if question.question_id in seen_questions:
                raise ValueError("Duplicate SDB author/question probe")
            seen_questions.add(question.question_id)
            if row.get("dataset") != question.dataset or row.get("environment") != question.environment:
                raise PermissionError("SDB probe question metadata drifted")
            assignment = _assignment_for_row(assignments, row)
            if (
                row.get("assignment_first") != assignment.first
                or row.get("assignment_second") != assignment.second
                or row.get("author_stage0_prediction") != assignment.author_answer
                or row.get("assignment_reason") != assignment.reason
            ):
                raise PermissionError("SDB candidate assignment does not replay")
            raw_prompt = build_diagnostic_probe_prompt(
                question,
                base_responses[(question.question_id, author)],
                assignment,
            )
            if row.get("raw_prompt_sha256") != hashlib.sha256(
                raw_prompt.encode("utf-8")
            ).hexdigest():
                raise PermissionError("SDB author pre-chat prompt does not replay")
            parsed_output = parse_diagnostic_probe_output(
                str(row["raw_output"]), assignment, question.question
            )
            left_index = presented_left_authored_outcome(
                int(generation["seed"]), question.question_id, author
            )
            presentation = (
                present_diagnostic_probe(parsed_output, left_index)
                if parsed_output.parse_error is None and not parsed_output.abstained
                else None
            )
            replay = {
                "probe": parsed_output.probe,
                "authored_outcome_1": parsed_output.outcome_1,
                "authored_outcome_2": parsed_output.outcome_2,
                "sealed_map_outcome_1": parsed_output.map_outcome_1,
                "sealed_map_outcome_2": parsed_output.map_outcome_2,
                "sealed_bridge_1": parsed_output.bridge_1,
                "sealed_bridge_2": parsed_output.bridge_2,
                "confidence": parsed_output.confidence,
                "parse_error": parsed_output.parse_error,
                "abstained": parsed_output.abstained,
                "presented_left_authored_outcome": left_index,
                "presented_left_text": presentation.left_text if presentation else None,
                "presented_right_text": presentation.right_text if presentation else None,
                "sealed_left_candidate": presentation.left_candidate if presentation else None,
                "sealed_right_candidate": presentation.right_candidate if presentation else None,
                "post_commit_permutation_applied": left_index == 2,
            }
            if any(row.get(key) != value for key, value in replay.items()):
                raise PermissionError("SDB parsed probe or sealed presentation does not replay")
            probe_id = f"{question.question_id}::{author}"
            if row.get("probe_id") != probe_id or probe_id in row_by_id:
                raise ValueError("SDB probe ID is not unique and canonical")
            row_by_id[probe_id] = row
            probe = DiagnosticProbe(
                probe_id=probe_id,
                question_id=question.question_id,
                author_id=author,
                first_candidate=assignment.first,
                second_candidate=assignment.second,
                author_stage0_prediction=assignment.author_answer,
                confidence=parsed_output.confidence,
                parse_error=parsed_output.parse_error,
                abstained=parsed_output.abstained,
                left_candidate=presentation.left_candidate if presentation else None,
                right_candidate=presentation.right_candidate if presentation else None,
            )
            probes.append(probe)
            parsed += int(probe.parse_error is None)
            nonabstaining += int(probe.parse_error is None and not probe.abstained)
            bijections += int(
                probe.parse_error is None
                and not probe.abstained
                and {probe.left_candidate, probe.right_candidate}
                == {probe.first_candidate, probe.second_candidate}
            )
            truncated += int(bool(row["prompt_was_truncated"]))
            by_dataset[question.dataset] += 1
        replayed_counts = {
            "questions": len(rows),
            "model_calls": len(rows),
            "parsed_probes": parsed,
            "nonabstaining_probes": nonabstaining,
            "mapping_bijections": bijections,
            "truncated_model_calls": truncated,
        }
        if any(manifest.get(key) != value for key, value in replayed_counts.items()):
            raise PermissionError(f"SDB probe manifest counts drifted: {author}")
        parse_rate = parsed / max(1, len(rows))
        nonabstaining_rate = nonabstaining / max(1, len(rows))
        bijection_rate = bijections / max(1, nonabstaining)
        if (
            parse_rate < float(gates["minimum_probe_parse_rate"])
            or nonabstaining_rate < float(gates["minimum_nonabstaining_probe_rate"])
            or bijection_rate < float(gates["minimum_mapping_bijection_rate"])
            or truncated > int(gates["maximum_prompt_truncations"])
        ):
            raise RuntimeError(f"SDB full probe quality gate failed: {author}")
        quality.append(
            {
                "author": author,
                "questions": len(rows),
                "parsed": parsed,
                "parse_rate": parse_rate,
                "nonabstaining": nonabstaining,
                "nonabstaining_rate": nonabstaining_rate,
                "bijection_rate": bijection_rate,
                "truncated": truncated,
                "by_dataset": dict(sorted(by_dataset.items())),
            }
        )
        artifact_hashes[str(path)] = sha256_file(path)
        artifact_hashes[str(manifest_path)] = sha256_file(manifest_path)
    expected_ids = {
        f"{question.question_id}::{author}"
        for question in questions
        for author in authors
    }
    if set(row_by_id) != expected_ids:
        raise RuntimeError("SDB combined probe grid is incomplete")
    return tuple(probes), row_by_id, quality, artifact_hashes


def _presentation_from_probe_row(row: Mapping[str, Any]) -> PresentedDiagnosticProbe:
    return PresentedDiagnosticProbe(
        probe=str(row["probe"]),
        left_text=str(row["presented_left_text"]),
        right_text=str(row["presented_right_text"]),
        left_candidate=str(row["sealed_left_candidate"]),
        right_candidate=str(row["sealed_right_candidate"]),
        left_authored_outcome=int(row["presented_left_authored_outcome"]),
        post_commit_permutation_applied=bool(row["post_commit_permutation_applied"]),
    )


def _authenticate_checks(
    config_path: Path,
    config: Mapping[str, Any],
    run_root: Path,
    probes: Sequence[DiagnosticProbe],
    probe_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[DiagnosticProbeCheck, ...], list[dict[str, Any]], dict[str, str]]:
    checkers = tuple(str(value) for value in config["checker_models"])
    probe_by_id = {row.probe_id: row for row in probes}
    prompt_hash = hashlib.sha256(
        inspect.getsource(build_blind_probe_check_prompt).encode("utf-8")
    ).hexdigest()
    parser_hash = hashlib.sha256(
        inspect.getsource(parse_blind_probe_check_output).encode("utf-8")
    ).hexdigest()
    reveal_hash = hashlib.sha256(
        inspect.getsource(reveal_probe_candidate).encode("utf-8")
    ).hexdigest()
    checks: list[DiagnosticProbeCheck] = []
    quality: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    seen_check_ids: set[str] = set()
    gates = config["smoke_acceptance"]
    generation = config["check_generation"]
    for checker in checkers:
        directory = run_root / "checks" / checker
        raw_path = directory / "raw_checks.jsonl"
        path = directory / "checks.jsonl"
        manifest_path = directory / "check_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "completed_label_free_sdb_checks"
            or manifest.get("checker") != checker
            or manifest.get("labels_read") is not False
        ):
            raise PermissionError(f"Incomplete or label-crossed SDB checker: {checker}")
        _assert_current_code(manifest, f"checker {checker}")
        raw_hash = sha256_file(raw_path)
        if manifest.get("raw_checks_sha256") != raw_hash:
            raise PermissionError(f"SDB raw checks changed: {checker}")
        if manifest.get("check_sha256") != sha256_file(path):
            raise PermissionError(f"SDB revealed checks changed: {checker}")
        if _recorded_input_hash(manifest.get("environment", {}), config_path) != sha256_file(
            config_path
        ):
            raise PermissionError(f"SDB checker used a different config: {checker}")
        protocol_values = {
            "prompt_version": generation["prompt_version"],
            "parser_version": generation["parser_version"],
            "prompt_builder_sha256": prompt_hash,
            "parser_sha256": parser_hash,
            "reveal_sha256": reveal_hash,
            "original_task_was_hidden": True,
            "candidate_pair_was_hidden": True,
            "outcome_mapping_was_hidden": True,
            "mapping_revealed_after_raw_outputs_frozen": True,
        }
        if any(manifest.get(key) != value for key, value in protocol_values.items()):
            raise PermissionError(f"SDB check protocol drifted: {checker}")
        raw_rows = _read_jsonl(raw_path)
        rows = _read_jsonl(path)
        raw_by_id = {str(row["check_id"]): row for row in raw_rows}
        final_by_id = {str(row["check_id"]): row for row in rows}
        if len(raw_by_id) != len(raw_rows) or set(raw_by_id) != set(final_by_id):
            raise RuntimeError("SDB raw and revealed check identities differ")
        expected_probe_ids = {
            probe.probe_id
            for probe in probes
            if probe.author_id != checker
            and probe.parse_error is None
            and not probe.abstained
        }
        if {str(row["probe_id"]) for row in rows} != expected_probe_ids:
            raise RuntimeError(f"SDB checker coverage is incomplete: {checker}")
        parsed_count = 0
        decided_count = 0
        truncated = 0
        side_counts: Counter[str] = Counter()
        for check_id, raw in raw_by_id.items():
            final = final_by_id[check_id]
            if any(final.get(key) != value for key, value in raw.items()):
                raise PermissionError("SDB revealed check mutated its frozen raw fields")
            if final.get("frozen_raw_checks_sha256") != raw_hash:
                raise PermissionError("SDB revealed check is not bound to raw artifact")
            if final.get("mapping_revealed_after_raw_outputs_frozen") is not True:
                raise PermissionError("SDB mapping reveal preceded raw-output freeze")
            probe_id = str(raw["probe_id"])
            probe = probe_by_id.get(probe_id)
            probe_row = probe_rows.get(probe_id)
            if probe is None or probe_row is None:
                raise ValueError("SDB check references an unknown probe")
            if probe.author_id == checker:
                raise PermissionError("SDB checker inspected its own probe")
            raw_prompt = build_blind_probe_check_prompt(
                str(probe_row["probe"]),
                str(probe_row["presented_left_text"]),
                str(probe_row["presented_right_text"]),
            )
            if raw.get("raw_prompt_sha256") != hashlib.sha256(
                raw_prompt.encode("utf-8")
            ).hexdigest():
                raise PermissionError("SDB blind checker pre-chat prompt does not replay")
            parsed = parse_blind_probe_check_output(str(raw["raw_output"]))
            replayed = {
                "outcome_side": parsed.outcome_side,
                "derivation": parsed.derivation,
                "confidence": parsed.confidence,
                "parse_error": parsed.parse_error,
                "uncertain": parsed.uncertain,
            }
            if any(raw.get(key) != value for key, value in replayed.items()):
                raise PermissionError("SDB raw checker parse does not replay")
            selected, rejected = reveal_probe_candidate(
                parsed, _presentation_from_probe_row(probe_row)
            )
            if (
                final.get("selected_candidate") != selected
                or final.get("rejected_candidate") != rejected
            ):
                raise PermissionError("SDB delayed candidate reveal does not replay")
            check = DiagnosticProbeCheck(
                check_id=check_id,
                probe_id=probe_id,
                question_id=probe.question_id,
                author_id=probe.author_id,
                checker_id=checker,
                outcome_side=parsed.outcome_side,
                selected_candidate=selected,
                rejected_candidate=rejected,
                confidence=parsed.confidence,
                parse_error=parsed.parse_error,
                uncertain=parsed.uncertain,
            )
            if check_id in seen_check_ids:
                raise ValueError("Duplicate SDB check across checker artifacts")
            seen_check_ids.add(check_id)
            checks.append(check)
            parsed_count += int(check.parse_error is None)
            decided_count += int(check.parse_error is None and not check.uncertain)
            truncated += int(bool(raw["prompt_was_truncated"]))
            if check.parse_error is None and not check.uncertain:
                side_counts[str(check.outcome_side)] += 1
        replayed_counts = {
            "input_probes": len(rows),
            "model_calls": len(rows),
            "parsed_checks": parsed_count,
            "decided_checks": decided_count,
            "truncated_model_calls": truncated,
            "presented_side_counts": {
                side: side_counts[side] for side in ("LEFT", "RIGHT")
            },
        }
        if any(manifest.get(key) != value for key, value in replayed_counts.items()):
            raise PermissionError(f"SDB checker manifest counts drifted: {checker}")
        parse_rate = parsed_count / max(1, len(rows))
        decided_rate = decided_count / max(1, len(rows))
        largest_side = max(side_counts.values(), default=0) / max(1, decided_count)
        if (
            parse_rate < float(gates["minimum_check_parse_rate"])
            or decided_rate < float(gates["minimum_decided_check_rate"])
            or largest_side
            > float(gates["maximum_single_presented_side_selection_rate"])
            or truncated > int(gates["maximum_prompt_truncations"])
        ):
            raise RuntimeError(f"SDB full checker quality gate failed: {checker}")
        quality.append(
            {
                "checker": checker,
                "checks": len(rows),
                "parsed": parsed_count,
                "parse_rate": parse_rate,
                "decided": decided_count,
                "decided_rate": decided_rate,
                "largest_presented_side_selection_rate": largest_side,
                "truncated": truncated,
            }
        )
        artifact_hashes[str(raw_path)] = raw_hash
        artifact_hashes[str(path)] = sha256_file(path)
        artifact_hashes[str(manifest_path)] = sha256_file(manifest_path)
    return tuple(checks), quality, artifact_hashes


def load_and_authenticate_sdb_data(
    config_path: Path, config: Mapping[str, Any], run_root: Path
) -> SDBDevelopmentData:
    smoke_gate_path = run_root / "smoke" / "smoke_gate.json"
    smoke_gate = json.loads(smoke_gate_path.read_text(encoding="utf-8"))
    if (
        smoke_gate.get("status") != "passed_label_free_sdb_smoke_gates"
        or smoke_gate.get("labels_read_for_generation_or_decision") is not False
        or smoke_gate.get("policy", {}).get("full_run_authorized") is not True
    ):
        raise PermissionError("SDB full evaluation lacks a passed label-free smoke gate")
    if smoke_gate.get("sha256", {}).get(str(config_path)) != sha256_file(config_path):
        raise PermissionError("SDB smoke gate authenticates a different config")
    questions, base_predictions, answers, base_responses, input_hashes = (
        _load_observables_and_labels(config)
    )
    probes, probe_rows, probe_quality, probe_hashes = _authenticate_probes(
        config_path,
        config,
        run_root,
        questions,
        base_predictions,
        base_responses,
    )
    checks, check_quality, check_hashes = _authenticate_checks(
        config_path, config, run_root, probes, probe_rows
    )
    input_hashes.update(probe_hashes)
    input_hashes.update(check_hashes)
    input_hashes[str(smoke_gate_path)] = sha256_file(smoke_gate_path)
    return SDBDevelopmentData(
        questions=questions,
        base_predictions=base_predictions,
        probes=probes,
        checks=checks,
        answers=answers,
        dataset_by_question={row.question_id: row.dataset for row in questions},
        environment_by_question={row.question_id: row.environment for row in questions},
        base_response_by_key=base_responses,
        generation_quality={
            "probe_authors": probe_quality,
            "probe_checkers": check_quality,
            "smoke_gate_sha256": sha256_file(smoke_gate_path),
        },
        input_artifact_hashes=input_hashes,
    )


def variants_from_config(config: Mapping[str, Any]) -> tuple[SDBVariant, ...]:
    grid = config["variant_grid"]
    variants: list[SDBVariant] = []
    index = 0
    for regularization_c in grid["regularization_c"]:
        for minimum_confidence in grid["minimum_probe_confidence"]:
            for margin in grid["intervention_margin"]:
                variants.append(
                    SDBVariant(
                        name=f"sdb_grid_{index:03d}",
                        regularization_c=float(regularization_c),
                        intervention_margin=float(margin),
                        minimum_probe_confidence=int(minimum_confidence),
                        use_author_identity=bool(grid["use_author_identity"]),
                        use_checker_identity=bool(grid["use_checker_identity"]),
                        use_author_checker_interaction=bool(
                            grid["use_author_checker_interaction"]
                        ),
                        use_checker_stage0_relation=bool(
                            grid["use_checker_stage0_relation"]
                        ),
                        open_option_set=bool(grid["open_option_set"]),
                    )
                )
                index += 1
    return tuple(variants)


def _answers_from_decisions(decisions: Sequence[SDBDecision]) -> dict[str, str]:
    return {row.question_id: row.answer for row in decisions}


def _variant_structure(variant: SDBVariant) -> tuple[Any, ...]:
    return (
        variant.regularization_c,
        variant.minimum_probe_confidence,
        variant.use_author_identity,
        variant.use_checker_identity,
        variant.use_author_checker_interaction,
        variant.use_checker_stage0_relation,
        variant.open_option_set,
    )


def select_sdb_variant_nested(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    probes: Sequence[DiagnosticProbe],
    checks: Sequence[DiagnosticProbeCheck],
    labels: SourceTrainingLabels,
    answers: Mapping[str, str],
    variants: Sequence[SDBVariant],
    seed: int,
) -> tuple[SDBVariant, list[dict[str, Any]]]:
    question_ids = tuple(sorted(row.question_id for row in questions))
    if not set(question_ids).issubset(answers):
        raise PermissionError("Nested SDB answer scope is incomplete")
    folds = leave_one_environment_out(question_ids, labels.environment_by_question)
    question_by_id = {row.question_id: row for row in questions}
    variant_index = {variant.name: index for index, variant in enumerate(variants)}
    by_structure: dict[tuple[Any, ...], list[SDBVariant]] = defaultdict(list)
    for variant in variants:
        by_structure[_variant_structure(variant)].append(variant)
    correct_by_environment = {
        variant.name: {} for variant in variants
    }
    count_by_environment: dict[str, int] = {}
    for fold_index, (environment, train_ids, validation_ids) in enumerate(folds):
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        validation_questions = [
            question_by_id[question_id] for question_id in validation_ids
        ]
        train_base = _subset_rows(base_predictions, train_ids)
        validation_base = _subset_rows(base_predictions, validation_ids)
        train_probes = _subset_rows(probes, train_ids)
        validation_probes = _subset_rows(probes, validation_ids)
        train_checks = _subset_rows(checks, train_ids)
        validation_checks = _subset_rows(checks, validation_ids)
        train_labels = labels.subset(train_ids)
        count_by_environment[environment] = len(validation_ids)
        for grouped in by_structure.values():
            fitted = SealedDiagnosticBijectionCourt(
                grouped[0], seed=seed + fold_index
            ).fit(
                train_questions,
                train_base,
                train_probes,
                train_checks,
                train_labels,
            )
            for variant in grouped:
                predicted = _answers_from_decisions(
                    fitted.with_variant(variant).predict(
                        validation_questions,
                        validation_base,
                        validation_probes,
                        validation_checks,
                    )
                )
                correct_by_environment[variant.name][environment] = sum(
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
    return variants[variant_index[str(selected_row["name"])]], rows


def _mapped_vote(
    question: FalsificationQuestion,
    checks: Sequence[DiagnosticProbeCheck],
    reference: str,
) -> str:
    counts = Counter(
        row.selected_candidate
        for row in checks
        if row.parse_error is None
        and not row.uncertain
        and row.selected_candidate in question.option_labels
    )
    if not counts:
        return reference
    maximum = max(counts.values())
    winners = [candidate for candidate, count in counts.items() if count == maximum]
    return reference if reference in winners else min(
        winners, key=question.option_labels.index
    )


def _ablation_variants(selected: SDBVariant) -> dict[str, SDBVariant]:
    return {
        "sdb_no_author_identity": replace(
            selected, name="sdb_no_author_identity", use_author_identity=False
        ),
        "sdb_no_checker_identity": replace(
            selected, name="sdb_no_checker_identity", use_checker_identity=False
        ),
        "sdb_no_author_checker_interaction": replace(
            selected,
            name="sdb_no_author_checker_interaction",
            use_author_checker_interaction=False,
        ),
        "sdb_no_checker_stage0_relation": replace(
            selected,
            name="sdb_no_checker_stage0_relation",
            use_checker_stage0_relation=False,
        ),
    }


def generate_nested_sdb_predictions(
    config: Mapping[str, Any], data: SDBDevelopmentData
) -> tuple[
    dict[str, dict[str, str | None]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    experts = tuple(str(value) for value in config["experts"])
    authors = tuple(str(value) for value in config["author_models"])
    checkers = tuple(str(value) for value in config["checker_models"])
    variants = variants_from_config(config)
    labels = _source_labels(
        data.questions,
        data.base_predictions,
        data.answers,
        data.environment_by_question,
    )
    question_by_id = {row.question_id: row for row in data.questions}
    base_by_question = _base_answers_by_question(data.base_predictions)
    checks_by_question: dict[str, list[DiagnosticProbeCheck]] = defaultdict(list)
    for row in data.checks:
        checks_by_question[row.question_id].append(row)
    predictions: dict[str, dict[str, str | None]] = {
        f"single::{expert}": {
            question_id: base_by_question[question_id][expert]
            for question_id in sorted(question_by_id)
        }
        for expert in experts
    }
    methods = (
        "best_single_nested_oof",
        "majority_vote",
        "source_weighted_vote",
        "mapped_probe_vote",
        "sdb_primary_nested",
        "sdb_no_author_identity",
        "sdb_no_checker_identity",
        "sdb_no_author_checker_interaction",
        "sdb_no_checker_stage0_relation",
        "sdb_no_diagnostic_evidence",
        "sdb_no_conservative_intervention_gate",
    )
    predictions.update({method: {} for method in methods})
    predictions.update({f"sdb_single_author::{author}": {} for author in authors})
    predictions.update({f"sdb_single_checker::{checker}": {} for checker in checkers})
    nested_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    primary_diagnostics: dict[str, dict[str, Any]] = {}
    outer_folds = leave_one_environment_out(
        question_by_id, data.environment_by_question
    )
    seed = int(config["seed"])
    for outer_index, (outer_environment, train_ids, heldout_ids) in enumerate(outer_folds):
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        heldout_questions = [question_by_id[question_id] for question_id in heldout_ids]
        train_base = _subset_rows(data.base_predictions, train_ids)
        heldout_base = _subset_rows(data.base_predictions, heldout_ids)
        train_probes = _subset_rows(data.probes, train_ids)
        heldout_probes = _subset_rows(data.probes, heldout_ids)
        train_checks = _subset_rows(data.checks, train_ids)
        heldout_checks = _subset_rows(data.checks, heldout_ids)
        train_labels = labels.subset(train_ids)
        selected, search_rows = select_sdb_variant_nested(
            train_questions,
            train_base,
            train_probes,
            train_checks,
            train_labels,
            data.answers,
            variants,
            seed + outer_index * 1000,
        )
        nested_rows.extend(
            {"outer_environment": outer_environment, **row} for row in search_rows
        )
        expert_accuracy = _expert_accuracies(experts, train_ids, train_labels)
        reference_expert = sorted(
            experts, key=lambda expert: (-expert_accuracy[expert], expert)
        )[0]
        model = SealedDiagnosticBijectionCourt(
            selected, seed=seed + outer_index
        ).fit(
            train_questions,
            train_base,
            train_probes,
            train_checks,
            train_labels,
        )
        if model.reference_expert_ != reference_expert:
            raise AssertionError("SDB and source best-single references differ")
        primary = {
            row.question_id: row
            for row in model.predict(
                heldout_questions,
                heldout_base,
                heldout_probes,
                heldout_checks,
            )
        }
        no_gate_variant = replace(
            selected,
            name="sdb_no_conservative_intervention_gate",
            intervention_margin=0.0,
        )
        no_gate = _answers_from_decisions(
            model.with_variant(no_gate_variant).predict(
                heldout_questions,
                heldout_base,
                heldout_probes,
                heldout_checks,
            )
        )
        predictions["sdb_no_conservative_intervention_gate"].update(no_gate)
        no_evidence_model = SealedDiagnosticBijectionCourt(
            replace(selected, name="sdb_no_diagnostic_evidence"),
            seed=seed + outer_index,
        ).fit(train_questions, train_base, (), (), train_labels)
        predictions["sdb_no_diagnostic_evidence"].update(
            _answers_from_decisions(
                no_evidence_model.predict(heldout_questions, heldout_base, (), ())
            )
        )
        for method, variant in _ablation_variants(selected).items():
            ablation_model = SealedDiagnosticBijectionCourt(
                variant, seed=seed + outer_index
            ).fit(
                train_questions,
                train_base,
                train_probes,
                train_checks,
                train_labels,
            )
            predictions[method].update(
                _answers_from_decisions(
                    ablation_model.predict(
                        heldout_questions,
                        heldout_base,
                        heldout_probes,
                        heldout_checks,
                    )
                )
            )
        for author in authors:
            train_author_probes = [row for row in train_probes if row.author_id == author]
            heldout_author_probes = [
                row for row in heldout_probes if row.author_id == author
            ]
            probe_ids = {
                row.probe_id for row in train_author_probes + heldout_author_probes
            }
            train_author_checks = [
                row for row in train_checks if row.probe_id in probe_ids
            ]
            heldout_author_checks = [
                row for row in heldout_checks if row.probe_id in probe_ids
            ]
            author_model = SealedDiagnosticBijectionCourt(
                replace(selected, name=f"sdb_single_author::{author}"),
                seed=seed + outer_index,
            ).fit(
                train_questions,
                train_base,
                train_author_probes,
                train_author_checks,
                train_labels,
            )
            predictions[f"sdb_single_author::{author}"].update(
                _answers_from_decisions(
                    author_model.predict(
                        heldout_questions,
                        heldout_base,
                        heldout_author_probes,
                        heldout_author_checks,
                    )
                )
            )
        for checker in checkers:
            train_checker_checks = [
                row for row in train_checks if row.checker_id == checker
            ]
            heldout_checker_checks = [
                row for row in heldout_checks if row.checker_id == checker
            ]
            checker_model = SealedDiagnosticBijectionCourt(
                replace(selected, name=f"sdb_single_checker::{checker}"),
                seed=seed + outer_index,
            ).fit(
                train_questions,
                train_base,
                train_probes,
                train_checker_checks,
                train_labels,
            )
            predictions[f"sdb_single_checker::{checker}"].update(
                _answers_from_decisions(
                    checker_model.predict(
                        heldout_questions,
                        heldout_base,
                        heldout_probes,
                        heldout_checker_checks,
                    )
                )
            )
        for question_id in heldout_ids:
            question = question_by_id[question_id]
            base = base_by_question[question_id]
            reference = base.get(reference_expert)
            if reference not in question.option_labels:
                reference = question.option_labels[0]
            reference = str(reference)
            predictions["best_single_nested_oof"][question_id] = reference
            predictions["majority_vote"][question_id] = _majority_answer(
                question, base, reference
            )
            predictions["source_weighted_vote"][question_id] = _weighted_answer(
                question, base, expert_accuracy, reference
            )
            predictions["mapped_probe_vote"][question_id] = _mapped_vote(
                question, checks_by_question[question_id], reference
            )
            decision = primary[question_id]
            predictions["sdb_primary_nested"][question_id] = decision.answer
            primary_diagnostics[question_id] = {
                **dict(decision.diagnostics),
                "outer_environment": outer_environment,
                "outer_reference_expert": reference_expert,
                "outer_expert_accuracy": expert_accuracy,
                "selected_variant": asdict(selected),
                "selected_expert_id": decision.selected_expert_id,
                "candidate_logits": dict(decision.candidate_logits),
                "candidate_probabilities": dict(decision.candidate_probabilities),
                "fallback_reason": decision.fallback_reason,
                "open_set_rescue": decision.open_set_rescue,
                "valid_mask": {
                    expert: base.get(expert) in question.option_labels
                    for expert in experts
                },
                "missing_mask": {
                    expert: base.get(expert) not in question.option_labels
                    for expert in experts
                },
            }
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
            raise RuntimeError(f"Nested SDB method has incomplete predictions: {method}")
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
        row["full_development_best_single_accuracy"] = individual_accuracy[
            descriptive_best
        ]
    return predictions, nested_rows, outer_rows, primary_diagnostics


def _call_budget(
    method: str, config: Mapping[str, Any], data: SDBDevelopmentData
) -> float:
    expert_count = len(config["experts"])
    authors = len(config["author_models"])
    checkers_per_probe = len(config["checker_models"]) - 1
    if method.startswith("single::") or method in {
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
    }:
        return 1.0
    if method in {"majority_vote", "source_weighted_vote", "sdb_no_diagnostic_evidence"}:
        return float(expert_count)
    if method.startswith("sdb_single_author::"):
        return float(expert_count + 1 + checkers_per_probe)
    if method.startswith("sdb_single_checker::"):
        return float(expert_count + authors + authors - 1)
    return float(expert_count + authors + authors * checkers_per_probe)


def evaluate_sdb_predictions(
    config: Mapping[str, Any],
    data: SDBDevelopmentData,
    predictions: Mapping[str, Mapping[str, str | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method_rows = [
        _method_summary(method, values, data, _call_budget(method, config, data))
        for method, values in sorted(predictions.items())
    ]
    primary = "sdb_primary_nested"
    references = [
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
        "majority_vote",
        "source_weighted_vote",
        "mapped_probe_vote",
        "sdb_no_author_identity",
        "sdb_no_checker_identity",
        "sdb_no_author_checker_interaction",
        "sdb_no_checker_stage0_relation",
        "sdb_no_diagnostic_evidence",
        "sdb_no_conservative_intervention_gate",
    ]
    references.extend(
        method
        for method in predictions
        if method.startswith("sdb_single_author::")
        or method.startswith("sdb_single_checker::")
    )
    validation = config["outer_validation"]
    comparisons = [
        _comparison(
            primary,
            reference,
            predictions,
            data,
            seed=int(validation["bootstrap_seed"]) + index,
            samples=int(validation["bootstrap_samples"]),
        )
        for index, reference in enumerate(references)
    ]
    by_reference = {str(row["reference"]): row for row in comparisons}
    deployable = by_reference["best_single_nested_oof"]
    descriptive = by_reference["full_development_best_single_descriptive"]
    acceptance = config["acceptance"]
    primary_accuracy = next(
        float(row["accuracy"]) for row in method_rows if row["method"] == primary
    )
    fixed_single = [
        float(row["accuracy"])
        for row in method_rows
        if str(row["method"]).startswith("single::")
    ]
    required_ablations = {
        "sdb_no_author_identity",
        "sdb_no_checker_identity",
        "sdb_no_author_checker_interaction",
        "sdb_no_checker_stage0_relation",
        "sdb_no_diagnostic_evidence",
        "sdb_no_conservative_intervention_gate",
    }
    absent_generated_controls = {
        "cove_march_style_isolated_probe_control",
        "mapping_visible_equal_call_control",
        "direct_two_candidate_equal_call_control",
        "equal_call_single_model_baselines",
    }
    checks = {
        "delta_vs_deployable_best_single": float(deployable["delta"])
        >= float(acceptance["minimum_oof_delta_vs_deployable_best_single_pp"]) / 100.0,
        "delta_vs_full_development_best_single": float(descriptive["delta"])
        >= float(acceptance["minimum_oof_delta_vs_full_development_best_single_pp"])
        / 100.0,
        "nonnegative_on_every_dataset": all(
            float(value)
            >= float(acceptance["minimum_per_dataset_delta_pp"]) / 100.0
            for value in deployable["per_dataset_delta"].values()
        ),
        "paired_ci_low_above_zero": (
            float(deployable["stratified_paired_bootstrap_delta_ci95"][0]) > 0.0
            if bool(acceptance["require_paired_bootstrap_ci_low_gt_zero"])
            else True
        ),
        "mcnemar_significant": float(deployable["exact_mcnemar_p"])
        < float(acceptance["require_two_sided_mcnemar_p_lt"]),
        "beats_every_fixed_single_model": all(
            primary_accuracy > accuracy for accuracy in fixed_single
        ),
        "all_required_feature_ablations_present": required_ablations.issubset(
            predictions
        ),
        "cove_march_style_isolated_probe_control_present": False,
        "mapping_visible_equal_call_control_present": False,
        "direct_two_candidate_equal_call_control_present": False,
        "equal_call_single_model_baselines_present": False,
    }
    decision = {
        "status": "development_gate_pass" if all(checks.values()) else "development_gate_fail",
        "checks": checks,
        "primary_vs_deployable_best_single": deployable,
        "primary_vs_full_development_best_single_descriptive": descriptive,
        "blind_test_authorized": bool(all(checks.values()))
        and bool(acceptance["blind_test_authorized_only_after_all_development_gates_pass"]),
        "pending_required_generated_controls": sorted(absent_generated_controls),
        "claim_boundary": (
            "Development-only nested OOF evidence; controls remain mandatory and no "
            "confirmatory or global novelty claim is authorized."
        ),
    }
    return method_rows, comparisons, decision


def _plot_results(
    output_dir: Path,
    method_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchcoe_sdb_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method = {str(row["method"]): row for row in method_rows}
    shortlist = [
        "sdb_primary_nested",
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
        "majority_vote",
        "source_weighted_vote",
        "mapped_probe_vote",
        "sdb_no_diagnostic_evidence",
    ]
    shortlist = [method for method in shortlist if method in by_method]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    positions = np.arange(len(shortlist))
    accuracies = [100.0 * float(by_method[method]["accuracy"]) for method in shortlist]
    colors = ["#147D64" if method == "sdb_primary_nested" else "#66717E" for method in shortlist]
    axes[0].barh(positions, accuracies, color=colors)
    axes[0].set_yticks(positions, shortlist)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Nested OOF accuracy (%)")
    axes[0].set_title("SDB development methods")
    shown = [
        row
        for row in comparisons
        if row["reference"]
        in {"best_single_nested_oof", "majority_vote", "mapped_probe_vote"}
    ]
    positions = np.arange(len(shown))
    delta = [100.0 * float(row["delta"]) for row in shown]
    low = [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][0]) for row in shown]
    high = [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][1]) for row in shown]
    errors = np.asarray([[value - lo for value, lo in zip(delta, low, strict=True)], [hi - value for value, hi in zip(delta, high, strict=True)]])
    axes[1].errorbar(delta, positions, xerr=errors, fmt="o", color="#147D64", capsize=4)
    axes[1].axvline(0.0, color="#333333", linewidth=1)
    axes[1].set_yticks(positions, [str(row["reference"]) for row in shown])
    axes[1].set_xlabel("SDB accuracy delta (percentage points)")
    axes[1].set_title("Environment-stratified paired bootstrap")
    fig.tight_layout()
    fig.savefig(output_dir / "sdb_development_results.png", dpi=180)
    plt.close(fig)


def _markdown_report(
    method_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    lines = [
        "# SDB v9 development-only nested OOF report",
        "",
        "| Method | Accuracy | MMLU-Pro validation | GPQA Diamond | Calls/question |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(method_rows, key=lambda item: -float(item["accuracy"])):
        per_dataset = row["per_dataset_accuracy"]
        lines.append(
            f"| {row['method']} | {100 * float(row['accuracy']):.4f}% | "
            f"{100 * float(per_dataset.get('mmlu_pro', 0.0)):.4f}% | "
            f"{100 * float(per_dataset.get('gpqa', 0.0)):.4f}% | "
            f"{float(row['nominal_model_calls_per_question']):.1f} |"
        )
    lines.extend(["", "## Paired comparisons", "", "| Reference | Delta | 95% CI | McNemar p |", "| --- | ---: | ---: | ---: |"])
    for row in comparisons:
        ci = row["stratified_paired_bootstrap_delta_ci95"]
        lines.append(
            f"| {row['reference']} | {100 * float(row['delta']):+.4f} pp | "
            f"[{100 * float(ci[0]):+.4f}, {100 * float(ci[1]):+.4f}] | "
            f"{float(row['exact_mcnemar_p']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Status: `{decision['status']}`.",
            f"Blind test authorized: `{str(decision['blind_test_authorized']).lower()}`.",
            "",
            "This is source-development evidence only. Missing generated controls keep the "
            "publication mechanism claim and every blind test unauthorized.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    run_root = args.run_root or Path(str(config["output_root"]))
    output_dir = args.output_dir or run_root / "development_evaluation"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    data = load_and_authenticate_sdb_data(args.config, config, run_root)
    predictions, nested_rows, outer_rows, diagnostics = generate_nested_sdb_predictions(
        config, data
    )
    prediction_rows: list[dict[str, Any]] = []
    question_by_id = {row.question_id: row for row in data.questions}
    for method, values in sorted(predictions.items()):
        for question_id, prediction in sorted(values.items()):
            diagnostic = diagnostics.get(question_id, {}) if method == "sdb_primary_nested" else {}
            question = question_by_id[question_id]
            prediction_rows.append(
                {
                    "method": method,
                    "question_id": question_id,
                    "dataset": question.dataset,
                    "environment": question.environment,
                    "prediction": prediction,
                    "selected_answer_cluster": prediction,
                    "selected_expert_id": diagnostic.get("selected_expert_id"),
                    "cluster_scores": diagnostic.get("candidate_probabilities", {}),
                    "expert_scores": diagnostic.get("outer_expert_accuracy", {}),
                    "fallback_reason": diagnostic.get("fallback_reason"),
                    "observable_features": diagnostic,
                    "valid_mask": diagnostic.get("valid_mask", {}),
                    "missing_mask": diagnostic.get("missing_mask", {}),
                    "tie_breaking": diagnostic.get(
                        "tie_breaking", "method_specific_deterministic"
                    ),
                }
            )
    prediction_path = output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    prediction_hash = sha256_file(prediction_path)
    write_json(
        output_dir / "prediction_manifest.json",
        {
            "status": "frozen_sdb_development_oof_predictions",
            "prediction_sha256": prediction_hash,
            "methods": len(predictions),
            "questions": len(data.questions),
            "rows": len(prediction_rows),
            "labels_used_only_for_nested_source_fitting": True,
            "target_labels_read": False,
            "config_sha256": sha256_file(args.config),
            "input_artifact_hashes": dict(data.input_artifact_hashes),
            "environment": environment_manifest(
                sys.argv,
                int(config["seed"]),
                [args.config, *[Path(path) for path in data.input_artifact_hashes]],
            ),
        },
    )

    method_rows, comparisons, decision = evaluate_sdb_predictions(
        config, data, predictions
    )
    write_json(output_dir / "method_results.json", method_rows)
    write_csv(output_dir / "method_results.csv", list(method_rows))
    write_json(output_dir / "paired_comparisons.json", comparisons)
    write_json(output_dir / "development_gate.json", decision)
    write_json(output_dir / "nested_selection.json", nested_rows)
    write_json(output_dir / "outer_folds.json", outer_rows)
    write_json(output_dir / "generation_quality.json", data.generation_quality)
    (output_dir / "REPORT.md").write_text(
        _markdown_report(method_rows, comparisons, decision), encoding="utf-8"
    )
    _plot_results(output_dir, method_rows, comparisons)
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "status": "completed_sdb_development_only_evaluation",
            "prediction_sha256": prediction_hash,
            "development_labels_read_after_prediction_artifact_frozen": True,
            "blind_or_target_labels_read": False,
            "config_sha256": sha256_file(args.config),
            "started_unix": started,
            "finished_unix": time.time(),
            "blind_test_authorized": decision["blind_test_authorized"],
        },
    )
    print(f"Completed SDB development evaluation: {output_dir}")


if __name__ == "__main__":
    main()
