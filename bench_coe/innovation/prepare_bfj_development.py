from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .artifacts import environment_manifest, sha256_file, write_json, write_jsonl
from .blind_falsification_jury import FORBIDDEN_AUDIT_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize label-separated BFJ development questions and base outputs"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
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


def _normalise_prediction(value: object, labels: Iterable[str]) -> str | None:
    allowed = {str(label) for label in labels}
    if value is None:
        return None
    candidate = str(value).strip().upper()
    return candidate if candidate in allowed else None


def _dataset_config(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    for value in config["datasets"]:
        if value.get("name") == name:
            return dict(value)
    raise KeyError(f"Missing BFJ dataset configuration: {name}")


def _mmlu_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    dataset = _dataset_config(config, "mmlu_pro_validation")
    root = Path(str(dataset["cache_path"]))
    experts = tuple(str(value) for value in config["experts"])
    by_expert: dict[str, dict[int, dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for expert in experts:
        rows: dict[int, dict[str, Any]] = {}
        paths = sorted((root / expert / "CoT" / "validation").glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No MMLU-Pro validation files for {expert}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"MMLU-Pro cache is not a list: {path}")
            input_hashes[str(path)] = sha256_file(path)
            for row in payload:
                raw_id = int(row["question_id"])
                if raw_id in rows:
                    raise ValueError(f"Duplicate MMLU-Pro question for {expert}: {raw_id}")
                rows[raw_id] = dict(row)
        by_expert[expert] = rows
    reference = "Qwen2.5-7B-Instruct"
    if reference not in by_expert:
        reference = experts[0]
    raw_ids = tuple(sorted(by_expert[reference]))
    expected = int(dataset["expected_questions"])
    if len(raw_ids) != expected:
        raise RuntimeError(f"Expected {expected} MMLU-Pro rows, found {len(raw_ids)}")
    for expert, rows in by_expert.items():
        if tuple(sorted(rows)) != raw_ids:
            raise RuntimeError(f"MMLU-Pro expert IDs do not align: {expert}")

    questions: list[dict[str, Any]] = []
    base: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for raw_id in raw_ids:
        row = by_expert[reference][raw_id]
        options = [str(value) for value in row["options"]]
        option_labels = list(string.ascii_uppercase[: len(options)])
        answer = str(row["answer"]).strip().upper()
        if answer not in option_labels:
            raise ValueError(f"Invalid MMLU-Pro answer for question {raw_id}")
        question_id = f"mmlu_pro:validation:{raw_id}"
        category = str(row["category"])
        environment = f"mmlu_pro::{category}"
        questions.append(
            {
                "question_id": question_id,
                "dataset": "mmlu_pro",
                "split": "validation",
                "environment": environment,
                "question": str(row["question"]),
                "options": options,
                "option_labels": option_labels,
                "raw_question_id": str(raw_id),
            }
        )
        labels.append(
            {
                "question_id": question_id,
                "dataset": "mmlu_pro",
                "environment": environment,
                "answer": answer,
            }
        )
        for expert in experts:
            expert_row = by_expert[expert][raw_id]
            if (
                str(expert_row["question"]) != str(row["question"])
                or list(expert_row["options"]) != list(row["options"])
                or str(expert_row["answer"]).strip().upper() != answer
            ):
                raise RuntimeError(f"MMLU-Pro cache mismatch for {expert}/{raw_id}")
            base.append(
                {
                    "question_id": question_id,
                    "dataset": "mmlu_pro",
                    "expert_id": expert,
                    "prediction": _normalise_prediction(
                        expert_row.get("pred"), option_labels
                    ),
                    "response": str(expert_row.get("model_outputs", "")),
                    "model_error": None,
                }
            )
    return questions, base, labels, input_hashes


def _gpqa_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    dataset = _dataset_config(config, "gpqa_diamond_epoch0")
    label_root = Path(str(dataset["label_cache_path"]))
    observable_root = Path(str(dataset["observable_cache_path"]))
    reference_model = str(dataset["reference_model"])
    raw_path = label_root / reference_model / "predictions.jsonl"
    selected = [
        row
        for row in _read_jsonl(raw_path)
        if str(row["config"]) == str(dataset["config"])
        and int(row["epoch"]) == int(dataset["epoch"])
    ]
    expected = int(dataset["expected_questions"])
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} GPQA rows, found {len(selected)}")
    by_raw_id = {str(row["id"]): row for row in selected}
    if len(by_raw_id) != expected:
        raise ValueError("GPQA development selection contains duplicate IDs")
    record_ids = [str(row["record_id"]) for row in selected]
    if len(set(record_ids)) != expected:
        raise ValueError("GPQA statistical units are not unique Record IDs")

    experts = tuple(str(value) for value in config["experts"])
    observable_by_expert: dict[str, dict[str, dict[str, Any]]] = {}
    input_hashes = {str(raw_path): sha256_file(raw_path)}
    for expert in experts:
        path = observable_root / expert / "observables.jsonl"
        rows = {str(row["id"]): row for row in _read_jsonl(path)}
        missing = set(by_raw_id).difference(rows)
        if missing:
            raise RuntimeError(f"GPQA observables are incomplete for {expert}")
        observable_by_expert[expert] = rows
        input_hashes[str(path)] = sha256_file(path)

    questions: list[dict[str, Any]] = []
    base: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for raw_id, row in sorted(by_raw_id.items(), key=lambda item: str(item[1]["record_id"])):
        options = [str(value) for value in row["options"]]
        option_labels = list(string.ascii_uppercase[: len(options)])
        answer = str(row["answer"]).strip().upper()
        if answer not in option_labels:
            raise ValueError(f"Invalid GPQA answer for {raw_id}")
        record_id = str(row["record_id"])
        question_id = f"gpqa:diamond:{record_id}"
        domain = str(row["domain"])
        environment = f"gpqa::{domain.lower()}"
        questions.append(
            {
                "question_id": question_id,
                "dataset": "gpqa",
                "split": "diamond_epoch0",
                "environment": environment,
                "question": str(row["question"]),
                "options": options,
                "option_labels": option_labels,
                "raw_question_id": raw_id,
                "record_id": record_id,
            }
        )
        labels.append(
            {
                "question_id": question_id,
                "dataset": "gpqa",
                "environment": environment,
                "answer": answer,
            }
        )
        for expert in experts:
            expert_row = observable_by_expert[expert][raw_id]
            base.append(
                {
                    "question_id": question_id,
                    "dataset": "gpqa",
                    "expert_id": expert,
                    "prediction": _normalise_prediction(
                        expert_row.get("prediction"), option_labels
                    ),
                    "response": str(expert_row.get("response", "")),
                    "model_error": expert_row.get("model_error"),
                }
            )
    return questions, base, labels, input_hashes


def prepare(config_path: Path, run_root: Path | None = None) -> Path:
    config = _load_config(config_path)
    root = run_root or Path(str(config["output_root"]))
    if root.exists():
        raise FileExistsError(root)
    mmlu_questions, mmlu_base, mmlu_labels, mmlu_hashes = _mmlu_rows(config)
    gpqa_questions, gpqa_base, gpqa_labels, gpqa_hashes = _gpqa_rows(config)
    questions = sorted(mmlu_questions + gpqa_questions, key=lambda row: row["question_id"])
    base = sorted(
        mmlu_base + gpqa_base,
        key=lambda row: (row["question_id"], row["expert_id"]),
    )
    labels = sorted(mmlu_labels + gpqa_labels, key=lambda row: row["question_id"])
    expected_questions = sum(int(row["expected_questions"]) for row in config["datasets"])
    if len(questions) != expected_questions or len(labels) != expected_questions:
        raise RuntimeError("BFJ development question count does not match the protocol")
    expected_base = expected_questions * len(config["experts"])
    if len(base) != expected_base:
        raise RuntimeError("BFJ base-output count does not match the expert pool")
    for row in questions + base:
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(f"BFJ observable row contains label fields: {sorted(leaked)}")

    observable_dir = root / "development_observables"
    label_dir = root / "development_labels"
    question_path = observable_dir / "questions.jsonl"
    base_path = observable_dir / "base_predictions.jsonl"
    label_path = label_dir / "labels.jsonl"
    write_jsonl(question_path, questions)
    write_jsonl(base_path, base)
    write_jsonl(label_path, labels)
    input_hashes = {**mmlu_hashes, **gpqa_hashes}
    write_json(
        observable_dir / "observable_manifest.json",
        {
            "status": "development_observables_label_separated",
            "questions": len(questions),
            "base_predictions": len(base),
            "experts": list(config["experts"]),
            "question_sha256": sha256_file(question_path),
            "base_prediction_sha256": sha256_file(base_path),
            "input_hashes": input_hashes,
            "forbidden_fields": sorted(FORBIDDEN_AUDIT_KEYS),
            "generation_reads_labels": False,
            "environment": environment_manifest(
                ["prepare_bfj_development", "--config", str(config_path)],
                int(config["seed"]),
                [config_path],
            ),
        },
    )
    write_json(
        label_dir / "label_manifest.json",
        {
            "status": "development_labels_not_confirmatory_test",
            "questions": len(labels),
            "label_sha256": sha256_file(label_path),
            "allowed_use": "nested_source_development_only",
            "forbidden_use": "claiming_blind_confirmation",
        },
    )
    return root


def main() -> None:
    args = parse_args()
    root = prepare(args.config, args.run_root)
    print(f"Prepared BFJ development artifacts: {root}")


if __name__ == "__main__":
    main()
