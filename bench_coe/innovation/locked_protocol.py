from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .artifacts import (
    innovation_code_manifest,
    manifest_sha256,
    sha256_file,
    write_json,
    write_jsonl,
)


FORBIDDEN_TARGET_KEYS = frozenset(
    {"answer", "answer_index", "answer_choice", "gold", "target", "correct", "correctness", "is_correct", "score"}
)
SAFE_MUSR_COLUMNS = ("narrative", "question", "choices")
REQUIRED_MUSR_COLUMNS = SAFE_MUSR_COLUMNS + ("answer_index", "answer_choice")
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FINAL_CHOICE_RE = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)\s*\(?\s*([A-Z])\s*\)?",
    re.IGNORECASE,
)
PAREN_CHOICE_RE = re.compile(r"\(([A-Z])\)", re.IGNORECASE)


def load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol must be a mapping: {path}")
    return payload


def protocol_sha256(path: Path) -> str:
    return sha256_file(path)


def canonical_question_id(raw_id: str) -> str:
    return f"musr::test::{raw_id}"


def parse_choices(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        text = str(value or "").strip()
        parsed: Any = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("MuSR choices must encode a list")
        parsed = list(parsed)
    choices = [str(item).strip() for item in parsed]
    if not 2 <= len(choices) <= len(CHOICE_LABELS) or any(not item for item in choices):
        raise ValueError(f"Invalid MuSR option list with {len(choices)} entries")
    return choices


def _validate_raw_header(fieldnames: Iterable[str] | None, path: Path) -> None:
    fields = tuple(fieldnames or ())
    missing = set(REQUIRED_MUSR_COLUMNS).difference(fields)
    if missing:
        raise ValueError(f"MuSR file is missing required columns {sorted(missing)}: {path}")


def read_musr_observable_rows(raw_files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read only question-side fields; label columns are deliberately never accessed."""

    rows: list[dict[str, Any]] = []
    for spec in raw_files:
        task = str(spec["task"])
        path = Path(str(spec["path"]))
        count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_raw_header(reader.fieldnames, path)
            for index, raw in enumerate(reader):
                choices = parse_choices(raw["choices"])
                row = {
                    "id": f"{task}:{index:04d}",
                    "task": task,
                    "domain": task,
                    "narrative": str(raw["narrative"] or "").strip(),
                    "question": str(raw["question"] or "").strip(),
                    "options": choices,
                    "option_labels": list(CHOICE_LABELS[: len(choices)]),
                }
                leaked = FORBIDDEN_TARGET_KEYS.intersection(row)
                if leaked:
                    raise AssertionError(f"Observable materializer emitted labels: {sorted(leaked)}")
                if not row["narrative"] or not row["question"]:
                    raise ValueError(f"Empty MuSR question field at {task}:{index}")
                rows.append(row)
                count += 1
        expected = int(spec["expected_questions"])
        if count != expected:
            raise RuntimeError(f"MuSR task {task} has {count} rows; protocol requires {expected}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("MuSR observable IDs are not unique")
    return rows


def validate_protocol(config: Mapping[str, Any], *, verify_files: bool = True) -> None:
    policy = config.get("target_use_policy", {})
    if policy.get("target_answers_visible") != "evaluator_only_after_prediction_seal":
        raise ValueError("Protocol must isolate target answers behind the prediction seal")
    required_forbidden = {
        "fitting",
        "hyperparameter_selection",
        "method_selection",
        "prompt_selection",
        "parser_selection",
        "expert_pool_selection",
        "seed_selection",
        "subset_selection",
        "stopping_rule",
    }
    if not required_forbidden.issubset(set(policy.get("forbidden_target_uses", []))):
        raise ValueError("Protocol omits a required target-label prohibition")
    experts = tuple(str(value) for value in config.get("experts", []))
    if not experts or len(experts) != len(set(experts)):
        raise ValueError("Frozen expert pool must be non-empty and unique")
    methods = config.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise ValueError("No methods are preregistered")
    primary = config.get("hypotheses", {}).get("primary", {})
    candidate = str(primary.get("candidate", ""))
    reference = str(primary.get("reference", ""))
    if candidate not in methods or reference not in methods:
        raise ValueError("Primary candidate/reference must be preregistered methods")
    if methods[candidate].get("role") != "primary_method":
        raise ValueError("Primary candidate is not tagged as the primary method")
    if int(methods[candidate]["nominal_model_calls"]) != int(methods[reference]["nominal_model_calls"]):
        raise ValueError("Primary comparison must use equal nominal model-call budgets")
    pool_size = len(experts)
    for name, method in methods.items():
        calls = int(method["nominal_model_calls"])
        if name == "source_best_single":
            if calls != 1:
                raise ValueError("Source-best efficiency reference must use one model call")
        elif calls != pool_size:
            raise ValueError(f"Method {name} does not use the frozen full-pool budget")
    target = config.get("target", {})
    raw_files = target.get("raw_files", [])
    expected_total = sum(int(item["expected_questions"]) for item in raw_files)
    if expected_total != int(target.get("expected_questions", -1)):
        raise ValueError("Per-task and total target counts disagree")
    gpus = [int(value) for value in config.get("physical_gpus", [])]
    if gpus != [0, 1, 2, 3]:
        raise ValueError("Locked experiment is restricted to physical GPUs 0-3")
    if verify_files:
        for spec in raw_files:
            path = Path(str(spec["path"]))
            if sha256_file(path) != str(spec["sha256"]):
                raise PermissionError(f"Frozen raw target hash mismatch: {path}")
        family_map = Path(str(config["family_map"]))
        source_registry = Path(str(config["source_registry"]))
        if sha256_file(family_map) != str(config["family_map_sha256"]):
            raise PermissionError("Frozen expert family map hash mismatch")
        if sha256_file(source_registry) != str(config["source_registry_sha256"]):
            raise PermissionError("Frozen source registry hash mismatch")
        models_dir = Path(str(config["models_dir"]))
        missing = [expert for expert in experts if not (models_dir / expert).is_dir()]
        if missing:
            raise FileNotFoundError(f"Frozen local models are missing: {missing}")


def _prior_musr_artifacts(output_root: Path) -> list[str]:
    outputs = Path("outputs")
    if not outputs.exists():
        return []
    found: list[str] = []
    for path in outputs.rglob("*"):
        if output_root in (path, *path.parents):
            continue
        if "receipts" in path.parts or "audits" in path.parts:
            continue
        lowered = path.name.lower()
        if "musr" in lowered:
            found.append(str(path))
    return sorted(found)


def prepare_locked_run(config_path: Path) -> Path:
    config = load_protocol(config_path)
    validate_protocol(config)
    output_root = Path(str(config["output_root"]))
    if output_root.exists():
        raise FileExistsError(output_root)
    prior_artifacts = _prior_musr_artifacts(output_root)
    if prior_artifacts:
        raise RuntimeError(f"MuSR is not experimenter-untouched; prior artifacts exist: {prior_artifacts[:5]}")

    rows = read_musr_observable_rows(config["target"]["raw_files"])
    if len(rows) != int(config["target"]["expected_questions"]):
        raise RuntimeError("Materialized MuSR question count differs from protocol")
    observable_dir = output_root / "question_observables"
    observable_dir.mkdir(parents=True, exist_ok=False)
    questions_path = observable_dir / "questions.jsonl"
    write_jsonl(questions_path, rows)
    task_counts = Counter(str(row["task"]) for row in rows)
    observable_manifest = {
        "dataset": "musr",
        "split": "test",
        "role": "target_questions_only",
        "questions": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "row_keys": sorted(rows[0]),
        "forbidden_keys": sorted(FORBIDDEN_TARGET_KEYS),
        "questions_sha256": sha256_file(questions_path),
        "raw_file_hashes": {
            str(item["path"]): sha256_file(Path(str(item["path"])))
            for item in config["target"]["raw_files"]
        },
        "label_values_accessed": False,
    }
    manifest_path = observable_dir / "manifest.json"
    write_json(manifest_path, observable_manifest)
    shutil.copyfile(config_path, output_root / "frozen_protocol.yaml")
    preregistration = {
        "experiment_id": config["experiment_id"],
        "status": "frozen_before_model_predictions",
        "frozen_unix": time.time(),
        "protocol_path": str(config_path),
        "protocol_sha256": protocol_sha256(config_path),
        "frozen_protocol_sha256": sha256_file(output_root / "frozen_protocol.yaml"),
        "question_manifest_sha256": sha256_file(manifest_path),
        "question_observables_sha256": sha256_file(questions_path),
        "innovation_code_manifest_sha256": manifest_sha256(innovation_code_manifest()),
        "prior_musr_output_artifacts": prior_artifacts,
        "target_labels_opened": False,
        "methods": config["methods"],
        "hypotheses": config["hypotheses"],
        "claim_boundary": config["claim_boundary"],
    }
    prereg_path = output_root / "preregistration.json"
    write_json(prereg_path, preregistration)
    return prereg_path


def validate_preregistration(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_protocol(config_path)
    # Prediction processes authenticate the frozen, sanitized boundary and must
    # not reopen the raw target CSVs, whose bytes also contain answer columns.
    validate_protocol(config, verify_files=False)
    prereg_path = output_root / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("protocol_sha256") != protocol_sha256(config_path):
        raise PermissionError("Current protocol differs from preregistration")
    if prereg.get("frozen_protocol_sha256") != sha256_file(output_root / "frozen_protocol.yaml"):
        raise PermissionError("Frozen protocol copy was modified")
    current_code = manifest_sha256(innovation_code_manifest())
    if prereg.get("innovation_code_manifest_sha256") != current_code:
        raise PermissionError("Innovation code changed after preregistration")
    questions_path = output_root / "question_observables" / "questions.jsonl"
    if prereg.get("question_observables_sha256") != sha256_file(questions_path):
        raise PermissionError("Question observables changed after preregistration")
    return prereg


def load_question_observables(output_root: Path, expected_questions: int) -> list[dict[str, Any]]:
    manifest_path = output_root / "question_observables" / "manifest.json"
    questions_path = output_root / "question_observables" / "questions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("questions_sha256") != sha256_file(questions_path):
        raise PermissionError("Question-observable hash mismatch")
    rows: list[dict[str, Any]] = []
    with questions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                leaked = FORBIDDEN_TARGET_KEYS.intersection(row)
                if leaked:
                    raise PermissionError(f"Question observable contains label keys: {sorted(leaked)}")
                rows.append(row)
    if len(rows) != expected_questions:
        raise RuntimeError("Question-observable count mismatch")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Question-observable IDs are not unique")
    return rows


def build_musr_prompt(row: Mapping[str, Any]) -> str:
    options = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(row["option_labels"], row["options"], strict=True)
    )
    return (
        "Solve the following multiple-choice reasoning problem. Use the narrative and all "
        "constraints, reason step by step, and finish with exactly `Final answer: (X)`, where "
        "X is one listed option letter.\n\n"
        f"Narrative:\n{row['narrative']}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Choices:\n{options}\n\n"
        "Reasoning:"
    )


def extract_musr_choice(text: str, option_labels: Iterable[str]) -> str | None:
    allowed = {str(value).upper() for value in option_labels}
    explicit = [match.group(1).upper() for match in FINAL_CHOICE_RE.finditer(str(text or ""))]
    explicit = [value for value in explicit if value in allowed]
    if explicit:
        return explicit[-1]
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines:
        final_line = lines[-1].strip("` *_.,:;")
        match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", final_line, flags=re.IGNORECASE)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    parenthesized = [match.group(1).upper() for match in PAREN_CHOICE_RE.finditer(str(text or ""))]
    parenthesized = [value for value in parenthesized if value in allowed]
    return parenthesized[-1] if parenthesized else None


def model_identity_manifest(model_path: Path) -> dict[str, Any]:
    hash_names = {
        ".msc",
        ".mv",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
    content_hashes = {
        path.name: sha256_file(path)
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.name in hash_names
    }
    weight_files = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.suffix in {".safetensors", ".bin"}
    ]
    return {
        "resolved_path": str(model_path.resolve()),
        "identity_file_hashes": content_hashes,
        "identity_manifest_sha256": manifest_sha256(content_hashes),
        "weight_file_inventory": weight_files,
        "weight_bytes": sum(int(item["bytes"]) for item in weight_files),
        "weight_content_hash_note": "large weight content hashes not recomputed; local revision markers and exact inventory are bound",
    }


def create_label_access_marker(output_root: Path, prediction_seal_sha256: str) -> Path:
    marker = output_root / "label_access_started.json"
    if marker.exists():
        raise FileExistsError("Target labels have already been opened for this experiment")
    payload = {
        "status": "target_label_access_irreversibly_started",
        "started_unix": time.time(),
        "prediction_seal_sha256": prediction_seal_sha256,
        "rerun_allowed": False,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(marker, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return marker


def stratified_paired_bootstrap_delta(
    candidate: Mapping[str, bool],
    reference: Mapping[str, bool],
    stratum_by_question: Mapping[str, str],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    ids = sorted(candidate)
    if set(ids) != set(reference) or set(ids) != set(stratum_by_question):
        raise ValueError("Stratified paired bootstrap inputs are not aligned")
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for question_id in ids:
        by_stratum[str(stratum_by_question[question_id])].append(question_id)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for draw in range(samples):
        total = 0.0
        count = 0
        for stratum_ids in by_stratum.values():
            indices = rng.integers(0, len(stratum_ids), size=len(stratum_ids))
            for index in indices:
                question_id = stratum_ids[int(index)]
                total += float(candidate[question_id]) - float(reference[question_id])
                count += 1
        deltas[draw] = total / max(1, count)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def audit_gpqa_units(cache_root: Path, raw_root: Path) -> dict[str, Any]:
    config_files = {
        "diamond": raw_root / "gpqa_diamond.csv",
        "main": raw_root / "gpqa_main.csv",
        "extended": raw_root / "gpqa_extended.csv",
    }
    ids_by_config: dict[str, set[str]] = {}
    rows_by_config: dict[str, int] = {}
    for name, path in config_files.items():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Record ID" not in tuple(reader.fieldnames or ()):
                raise ValueError(f"GPQA file has no Record ID column: {path}")
            ids = [str(row["Record ID"]).strip() for row in reader]
        ids_by_config[name] = set(ids)
        rows_by_config[name] = len(ids)
    cached_rows = 0
    cached_record_ids: set[str] = set()
    epochs: Counter[int] = Counter()
    expert_dirs = sorted(path for path in cache_root.iterdir() if path.is_dir())
    representative: Path | None = None
    for expert_dir in expert_dirs:
        candidate = expert_dir / "predictions.jsonl"
        if candidate.exists():
            representative = candidate
            break
    if representative is not None:
        with representative.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                cached_rows += 1
                cached_record_ids.add(str(row["record_id"]))
                epochs[int(row.get("epoch", 0))] += 1
    union_ids = set().union(*ids_by_config.values())
    return {
        "official_primary_recommendation": "diamond_only_one_prediction_per_Record_ID",
        "statistical_unit": "Record ID, not shuffled-choice epoch and not overlapping config row",
        "raw_rows_by_config": rows_by_config,
        "unique_record_ids_by_config": {key: len(value) for key, value in ids_by_config.items()},
        "unique_record_ids_union": len(union_ids),
        "pairwise_overlap": {
            f"{left}_and_{right}": len(ids_by_config[left].intersection(ids_by_config[right]))
            for left, right in (("diamond", "main"), ("diamond", "extended"), ("main", "extended"))
        },
        "cached_representative_file": str(representative) if representative else None,
        "cached_rows": cached_rows,
        "cached_unique_record_ids": len(cached_record_ids),
        "cached_rows_by_epoch": dict(sorted(epochs.items())),
        "independence_warning": "choice permutations are repeated measures; overlapping configs must not be pooled as independent questions",
    }
