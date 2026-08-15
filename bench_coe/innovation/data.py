from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .schema import (
    CanonicalPredictionRecord,
    EvaluationLabels,
    ExpertPool,
    ObservableQueryBatch,
    SourceTrainingLabels,
)


UNCERTAINTY_TERMS = (
    "not sure",
    "uncertain",
    "maybe",
    "might",
    "cannot determine",
    "can't determine",
    "not enough information",
    "unknown",
    "guess",
    "ambiguous",
)
OPTION_RE = re.compile(r"(?:answer|final answer|correct answer)\s*(?:is|:)?\s*\(?([A-Z])\)?", re.I)
OBSERVABLE_METADATA_KEYS = (
    "question",
    "input",
    "options",
    "base_question_id",
    "epoch",
    "record_id",
    "config",
    "category",
    "subcategory",
    "subject",
    "domain",
    "subdomain",
    "task",
    "difficulty",
    "type",
    "question_type",
    "answer_type",
    "grade",
    "context",
    "skills",
    "img_type",
)
FORBIDDEN_OBSERVABLE_ROW_KEYS = frozenset(
    {"answer", "gold", "target", "correct", "correctness", "is_correct", "score"}
)
_ADAPTER_CAPABILITY = object()


def load_family_map(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    families = payload.get("families", {}) if isinstance(payload, dict) else {}
    if not isinstance(families, dict):
        raise ValueError(f"Invalid family mapping in {path}")
    return {str(expert): str(family) for expert, family in families.items()}


def normalize_answer(value: Any, output: str = "") -> str | None:
    if value is not None and str(value).strip():
        text = str(value).strip()
    else:
        match = OPTION_RE.search(output or "")
        lines = str(output or "").strip().splitlines()
        text = match.group(1) if match else (lines[-1] if lines else "")
    text = re.sub(r"\s+", " ", str(text or "").strip().lower())
    text = text.strip(" .,:;()[]{}\"'")
    return text[:48] if text else None


def lexical_uncertainty(output: str) -> float:
    lowered = output.lower()
    return math.log1p(sum(lowered.count(term) for term in UNCERTAINTY_TERMS))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction cache must contain a row list: {path}")
    return payload


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("id", row.get("question_id", row.get("pid")))
    if value is None:
        raise KeyError(f"Missing question ID in row keys={sorted(row)}")
    return str(value)


def _subject(row: Mapping[str, Any]) -> str:
    for key in ("subject", "domain", "category", "task", "subcategory"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "UNKNOWN"


def _correctness(row: Mapping[str, Any]) -> bool | None:
    if row.get("is_correct") is not None:
        return bool(row["is_correct"])
    if row.get("score") is not None:
        return float(row["score"]) > 0.0
    prediction = row.get("pred", row.get("prediction"))
    gold = row.get("answer", row.get("target"))
    if prediction is None or gold is None:
        return None
    return str(prediction).strip() == str(gold).strip()


def _validate_registry_entry(
    registry_path: Path,
    registry_sha256: str,
    cache_path: Path,
    dataset: str,
    split: str,
    modality: str,
    allowed_roles: set[str],
) -> dict[str, Any]:
    if _sha256(registry_path) != registry_sha256:
        raise PermissionError(f"Dataset registry hash mismatch: {registry_path}")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", []) if isinstance(payload, dict) else []
    expected_path = cache_path.resolve()
    matches = [
        entry
        for entry in datasets
        if isinstance(entry, dict)
        and str(entry.get("dataset")) == dataset
        and str(entry.get("split")) == split
        and str(entry.get("modality")) == modality
        and Path(str(entry.get("cache_path", ""))).resolve() == expected_path
        and str(entry.get("source_or_target")) in allowed_roles
    ]
    if len(matches) != 1:
        raise PermissionError(
            f"Cache is not registered for roles {sorted(allowed_roles)}: "
            f"dataset={dataset}, split={split}, modality={modality}, path={cache_path}"
        )
    return matches[0]


class CacheAdapter:
    """Materialize observables and labels through intentionally separate APIs."""

    def __init__(
        self,
        cache_path: Path,
        dataset: str,
        split: str,
        modality: str,
        family_by_expert: Mapping[str, str],
        expert_ids: Iterable[str] | None = None,
        *,
        _validated_role: str,
        _capability: object,
    ) -> None:
        if _capability is not _ADAPTER_CAPABILITY or _validated_role not in {"source", "target", "projection"}:
            raise PermissionError("CacheAdapter must be opened through a validating class method")
        self.cache_path = cache_path
        self.dataset = dataset
        self.split = split
        self.modality = modality
        self.data_role = _validated_role
        discovered = tuple(sorted(path.name for path in cache_path.iterdir() if path.is_dir()))
        self.expert_ids = tuple(sorted(expert_ids)) if expert_ids is not None else discovered
        missing_families = set(self.expert_ids).difference(family_by_expert)
        if missing_families:
            raise ValueError(f"Unreviewed expert families: {sorted(missing_families)}")
        self.family_by_expert = {expert: family_by_expert[expert] for expert in self.expert_ids}
        self._rows_cache: dict[str, dict[str, dict[str, Any]]] | None = None

    @classmethod
    def from_source_registry(
        cls,
        cache_path: Path,
        dataset: str,
        split: str,
        modality: str,
        family_by_expert: Mapping[str, str],
        expert_ids: Iterable[str] | None,
        registry_path: Path,
        registry_sha256: str,
    ) -> CacheAdapter:
        _validate_registry_entry(
            registry_path,
            registry_sha256,
            cache_path,
            dataset,
            split,
            modality,
            {"source", "both"},
        )
        return cls(
            cache_path,
            dataset,
            split,
            modality,
            family_by_expert,
            expert_ids,
            _validated_role="source",
            _capability=_ADAPTER_CAPABILITY,
        )

    @classmethod
    def from_target_observables(
        cls,
        cache_path: Path,
        dataset: str,
        split: str,
        modality: str,
        family_by_expert: Mapping[str, str],
        expert_ids: Iterable[str],
        manifest_sha256: str,
    ) -> CacheAdapter:
        manifest_path = cache_path / "observable_manifest.json"
        if not manifest_path.exists() or _sha256(manifest_path) != manifest_sha256:
            raise PermissionError(f"Target observable manifest hash mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experts = tuple(sorted(expert_ids))
        expected = {
            "dataset": dataset,
            "split": split,
            "modality": modality,
            "role": "target_observables_only",
            "expert_ids": list(experts),
        }
        for key, value in expected.items():
            observed = sorted(manifest.get(key, [])) if key == "expert_ids" else manifest.get(key)
            if observed != value:
                raise PermissionError(f"Target observable manifest field mismatch: {key}")
        recorded_hashes = manifest.get("output_observable_hashes", {})
        if not isinstance(recorded_hashes, dict):
            raise PermissionError("Target observable manifest has no output hashes")
        by_expert = {Path(path).parent.name: str(value) for path, value in recorded_hashes.items()}
        for expert in experts:
            path = cache_path / expert / "observables.jsonl"
            if by_expert.get(expert) != _sha256(path):
                raise PermissionError(f"Target observable hash mismatch: {path}")
        return cls(
            cache_path,
            dataset,
            split,
            modality,
            family_by_expert,
            experts,
            _validated_role="target",
            _capability=_ADAPTER_CAPABILITY,
        )

    def _prediction_paths(self, expert_id: str) -> tuple[Path, ...]:
        model_dir = self.cache_path / expert_id
        names = ("predictions.json", "predictions.jsonl") if self.data_role in {"source", "projection"} else ("observables.jsonl",)
        for name in names:
            candidate = model_dir / name
            if candidate.exists():
                return (candidate,)
        if self.data_role == "source":
            split_dir = model_dir / "CoT" / self.split
            category_files = tuple(sorted(split_dir.glob("*.json"))) if split_dir.is_dir() else ()
            if category_files:
                return category_files
        return ()

    def _prediction_path(self, expert_id: str) -> Path | None:
        paths = self._prediction_paths(expert_id)
        return paths[0] if len(paths) == 1 else None

    def _load_rows_by_expert(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._rows_cache is not None:
            return self._rows_cache
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for expert_id in self.expert_ids:
            paths = self._prediction_paths(expert_id)
            if not paths:
                result[expert_id] = {}
                continue
            rows_by_id: dict[str, dict[str, Any]] = {}
            for path in paths:
                for row in _read_rows(path):
                    if self.data_role == "target":
                        forbidden = FORBIDDEN_OBSERVABLE_ROW_KEYS.intersection(row)
                        if forbidden:
                            raise ValueError(f"Target observable cache contains label fields {sorted(forbidden)}: {path}")
                    raw_id = _row_id(row)
                    if raw_id in rows_by_id:
                        raise ValueError(f"Duplicate question ID {raw_id!r} in {path}")
                    rows_by_id[raw_id] = row
            result[expert_id] = rows_by_id
        self._rows_cache = result
        return result

    def _raw_ids(self, limit: int | None) -> tuple[str, ...]:
        rows = self._load_rows_by_expert()
        raw_ids = sorted(set().union(*(set(expert_rows) for expert_rows in rows.values())))
        return tuple(raw_ids[:limit] if limit is not None else raw_ids)

    def _canonical_id(self, raw_id: str) -> str:
        return f"{self.dataset}::{self.split}::{raw_id}"

    def load_observables(self, limit: int | None = None) -> ObservableQueryBatch:
        rows_by_expert = self._load_rows_by_expert()
        records: list[CanonicalPredictionRecord] = []
        for raw_id in self._raw_ids(limit):
            pending: list[CanonicalPredictionRecord] = []
            for expert_id in self.expert_ids:
                row = rows_by_expert[expert_id].get(raw_id)
                if row is None:
                    pending.append(
                        CanonicalPredictionRecord(
                            dataset=self.dataset,
                            split=self.split,
                            question_id=self._canonical_id(raw_id),
                            raw_question_id=raw_id,
                            subject="UNKNOWN",
                            modality=self.modality,
                            expert_id=expert_id,
                            expert_family=self.family_by_expert[expert_id],
                            raw_answer=None,
                            raw_output="",
                            normalized_answer=None,
                            per_query_cluster_id=None,
                            uncertainty=0.0,
                            valid_output=False,
                            missing_reason="cache_row_missing",
                        )
                    )
                    continue
                prediction_present = "pred" in row or "prediction" in row
                raw_answer = row.get("pred", row.get("prediction"))
                output = str(row.get("model_outputs", row.get("response", "")) or "")
                # A present-but-null cached prediction stays invalid. Recovering a
                # different answer from response text would disagree with the
                # cached benchmark judge.
                normalized = normalize_answer(raw_answer, "" if prediction_present else output)
                error = row.get("model_error")
                valid = normalized is not None and error in (None, "")
                metadata = {key: row[key] for key in OBSERVABLE_METADATA_KEYS if row.get(key) not in (None, "")}
                pending.append(
                    CanonicalPredictionRecord(
                        dataset=self.dataset,
                        split=self.split,
                        question_id=self._canonical_id(raw_id),
                        raw_question_id=raw_id,
                        subject=_subject(row),
                        modality=self.modality,
                        expert_id=expert_id,
                        expert_family=self.family_by_expert[expert_id],
                        raw_answer=None if raw_answer is None else str(raw_answer),
                        raw_output=output,
                        normalized_answer=normalized if valid else None,
                        per_query_cluster_id=None,
                        uncertainty=lexical_uncertainty(output),
                        valid_output=valid,
                        missing_reason=None if valid else (str(error) if error else "empty_or_unparseable_output"),
                        inference_cost=float(row["model_latency_seconds"]) if row.get("model_latency_seconds") is not None else None,
                        observable_metadata=metadata,
                    )
                )
            answer_to_cluster = {
                answer: cluster_id
                for cluster_id, answer in enumerate(sorted({r.normalized_answer for r in pending if r.valid_output and r.normalized_answer is not None}))
            }
            records.extend(
                replace(record, per_query_cluster_id=answer_to_cluster.get(record.normalized_answer))
                if record.valid_output
                else record
                for record in pending
            )
        return ObservableQueryBatch(
            dataset=self.dataset,
            split=self.split,
            modality=self.modality,
            pool=ExpertPool(self.expert_ids, self.family_by_expert),
            records=tuple(records),
        )

    def _load_correctness(self, limit: int | None) -> tuple[dict[tuple[str, str], bool], dict[str, str]]:
        rows_by_expert = self._load_rows_by_expert()
        correctness: dict[tuple[str, str], bool] = {}
        environments: dict[str, str] = {}
        for raw_id in self._raw_ids(limit):
            canonical_id = self._canonical_id(raw_id)
            for expert_id in self.expert_ids:
                row = rows_by_expert[expert_id].get(raw_id)
                if row is None:
                    continue
                value = _correctness(row)
                if value is not None:
                    correctness[(canonical_id, expert_id)] = value
                environments.setdefault(canonical_id, _subject(row))
        return correctness, environments

    def load_source_labels(self, limit: int | None = None) -> SourceTrainingLabels:
        if self.data_role != "source":
            raise PermissionError("A target-role cache adapter cannot export source training labels")
        correctness, environments = self._load_correctness(limit)
        return SourceTrainingLabels._from_source_adapter(self.dataset, self.split, correctness, environments)

    def load_evaluation_labels(self, limit: int | None = None) -> EvaluationLabels:
        raise PermissionError("Evaluation labels must be loaded by EvaluationLabelAdapter in a separate evaluator process")


class EvaluationLabelAdapter:
    """Label-only raw-cache reader intended for the isolated evaluation CLI."""

    def __init__(
        self,
        cache_path: Path,
        dataset: str,
        split: str,
        expert_ids: Iterable[str],
        *,
        _capability: object,
    ) -> None:
        if _capability is not _ADAPTER_CAPABILITY:
            raise PermissionError("EvaluationLabelAdapter must be opened through the dataset registry")
        self.cache_path = cache_path
        self.dataset = dataset
        self.split = split
        self.expert_ids = tuple(sorted(expert_ids))

    @classmethod
    def from_registry(
        cls,
        cache_path: Path,
        dataset: str,
        split: str,
        modality: str,
        expert_ids: Iterable[str],
        registry_path: Path,
        registry_sha256: str,
    ) -> EvaluationLabelAdapter:
        _validate_registry_entry(
            registry_path,
            registry_sha256,
            cache_path,
            dataset,
            split,
            modality,
            {"source", "target", "both"},
        )
        return cls(cache_path, dataset, split, expert_ids, _capability=_ADAPTER_CAPABILITY)

    def load(self, limit: int | None = None) -> EvaluationLabels:
        correctness: dict[tuple[str, str], bool] = {}
        all_ids: set[str] = set()
        rows_by_expert: dict[str, dict[str, dict[str, Any]]] = {}
        for expert in self.expert_ids:
            model_dir = self.cache_path / expert
            path = next((candidate for candidate in (model_dir / "predictions.json", model_dir / "predictions.jsonl") if candidate.exists()), None)
            rows = {_row_id(row): row for row in _read_rows(path)} if path is not None else {}
            rows_by_expert[expert] = rows
            all_ids.update(rows)
        raw_ids = sorted(all_ids)
        if limit is not None:
            raw_ids = raw_ids[:limit]
        for raw_id in raw_ids:
            question_id = f"{self.dataset}::{self.split}::{raw_id}"
            for expert in self.expert_ids:
                row = rows_by_expert[expert].get(raw_id)
                if row is None:
                    continue
                value = _correctness(row)
                if value is not None:
                    correctness[(question_id, expert)] = value
        return EvaluationLabels(self.dataset, self.split, correctness)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_label_free_observables(
    raw_cache_path: Path,
    output_path: Path,
    dataset: str,
    split: str,
    modality: str,
    family_by_expert: Mapping[str, str],
    expert_ids: Iterable[str],
    registry_path: Path,
    registry_sha256: str,
) -> dict[str, Any]:
    """Trusted preprocessing boundary; output files contain no label-like keys."""

    if output_path.exists():
        raise FileExistsError(output_path)
    experts = tuple(sorted(expert_ids))
    _validate_registry_entry(
        registry_path,
        registry_sha256,
        raw_cache_path,
        dataset,
        split,
        modality,
        {"target", "both"},
    )
    raw = CacheAdapter(
        raw_cache_path,
        dataset,
        split,
        modality,
        family_by_expert,
        experts,
        _validated_role="projection",
        _capability=_ADAPTER_CAPABILITY,
    )
    batch = raw.load_observables()
    input_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for expert in experts:
        for source_path in raw._prediction_paths(expert):
            input_hashes[str(source_path)] = _sha256(source_path)
        model_dir = output_path / expert
        model_dir.mkdir(parents=True, exist_ok=False)
        target_path = model_dir / "observables.jsonl"
        with target_path.open("w", encoding="utf-8") as handle:
            for record in batch.records:
                if record.expert_id != expert:
                    continue
                row = {
                    "id": record.raw_question_id,
                    "subject": record.subject,
                    "prediction": record.raw_answer,
                    "response": record.raw_output,
                    "model_error": record.missing_reason if not record.valid_output else None,
                    "model_latency_seconds": record.inference_cost,
                    **dict(record.observable_metadata),
                }
                forbidden = FORBIDDEN_OBSERVABLE_ROW_KEYS.intersection(row)
                if forbidden:
                    raise AssertionError(f"Exporter emitted forbidden fields: {sorted(forbidden)}")
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_hashes[str(target_path)] = _sha256(target_path)
    manifest = {
        "dataset": dataset,
        "split": split,
        "modality": modality,
        "role": "target_observables_only",
        "expert_ids": list(experts),
        "questions": len(batch.question_ids),
        "input_raw_cache_hashes": input_hashes,
        "output_observable_hashes": output_hashes,
        "forbidden_fields": sorted(FORBIDDEN_OBSERVABLE_ROW_KEYS),
    }
    manifest_path = output_path / "observable_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def assert_disjoint(source: ObservableQueryBatch, target: ObservableQueryBatch) -> None:
    source_ids = set(source.question_ids)
    overlap = source_ids.intersection(target.question_ids)
    if overlap:
        raise ValueError(f"Source/target canonical question IDs overlap: {sorted(overlap)[:5]}")
    if source.dataset == target.dataset:
        source_raw = {record.raw_question_id for record in source.records}
        target_raw = {record.raw_question_id for record in target.records}
        raw_overlap = source_raw.intersection(target_raw)
        if raw_overlap:
            raise ValueError(f"Source/target raw question IDs overlap within dataset: {sorted(raw_overlap)[:5]}")
