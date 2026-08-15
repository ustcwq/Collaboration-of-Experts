from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np

from .schema import CanonicalPredictionRecord, ObservableQueryBatch


def records_by_question(batch: ObservableQueryBatch) -> dict[str, tuple[CanonicalPredictionRecord, ...]]:
    grouped: dict[str, list[CanonicalPredictionRecord]] = {question_id: [] for question_id in batch.question_ids}
    for record in batch.records:
        grouped[record.question_id].append(record)
    return {key: tuple(sorted(value, key=lambda item: item.expert_id)) for key, value in grouped.items()}


def topology_features(records: Iterable[CanonicalPredictionRecord]) -> dict[str, float]:
    rows = tuple(records)
    valid = tuple(record for record in rows if record.valid_output and record.per_query_cluster_id is not None)
    counts = Counter(record.per_query_cluster_id for record in valid)
    total = len(rows)
    valid_count = len(valid)
    proportions = sorted((count / max(1, valid_count) for count in counts.values()), reverse=True)
    entropy = -sum(value * math.log(value + 1e-12) for value in proportions)
    if valid_count > 1:
        entropy /= math.log(valid_count)
    top1 = proportions[0] if proportions else 0.0
    top2 = proportions[1] if len(proportions) > 1 else 0.0
    top_cluster = proportions.index(top1) if proportions else -1
    families_by_cluster: dict[int, set[str]] = {}
    for record in valid:
        families_by_cluster.setdefault(int(record.per_query_cluster_id), set()).add(record.expert_family)
    family_counts = sorted((len(value) for value in families_by_cluster.values()), reverse=True)
    uncertainties = np.asarray([record.uncertainty for record in valid], dtype=float)
    return {
        "valid_experts": float(valid_count),
        "valid_fraction": valid_count / max(1, total),
        "missing_fraction": 1.0 - valid_count / max(1, total),
        "answer_clusters": float(len(counts)),
        "cluster_fraction": len(counts) / max(1, valid_count),
        "partition_entropy": float(entropy),
        "top1_share": float(top1),
        "top2_share": float(top2),
        "cluster_margin": float(top1 - top2),
        "top_cluster_family_breadth": float(family_counts[0] if family_counts else 0),
        "mean_uncertainty": float(uncertainties.mean()) if len(uncertainties) else 0.0,
        "std_uncertainty": float(uncertainties.std()) if len(uncertainties) else 0.0,
        "max_uncertainty": float(uncertainties.max()) if len(uncertainties) else 0.0,
    }


TOPOLOGY_VECTOR_KEYS = (
    "valid_fraction",
    "missing_fraction",
    "cluster_fraction",
    "partition_entropy",
    "top1_share",
    "top2_share",
    "cluster_margin",
    "top_cluster_family_breadth",
    "mean_uncertainty",
    "std_uncertainty",
    "max_uncertainty",
)


def topology_matrix(batch: ObservableQueryBatch) -> tuple[tuple[str, ...], np.ndarray]:
    grouped = records_by_question(batch)
    ids = tuple(sorted(grouped))
    matrix = np.asarray(
        [[topology_features(grouped[question_id])[key] for key in TOPOLOGY_VECTOR_KEYS] for question_id in ids],
        dtype=float,
    )
    return ids, np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def expert_observable_features(
    records: Iterable[CanonicalPredictionRecord],
    source_accuracy: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    rows = tuple(records)
    topo = topology_features(rows)
    valid = [record for record in rows if record.valid_output and record.per_query_cluster_id is not None]
    counts = Counter(record.per_query_cluster_id for record in valid)
    families: dict[int, set[str]] = {}
    for record in valid:
        families.setdefault(int(record.per_query_cluster_id), set()).add(record.expert_family)
    result: dict[str, list[float]] = {}
    for record in rows:
        cid = record.per_query_cluster_id
        result[record.expert_id] = [
            float((source_accuracy or {}).get(record.expert_id, 0.0)),
            counts.get(cid, 0) / max(1, len(valid)) if cid is not None else 0.0,
            topo["partition_entropy"],
            topo["cluster_margin"],
            float(len(families.get(int(cid), set()))) if cid is not None else 0.0,
            1.0 if record.valid_output else 0.0,
            float(record.uncertainty),
            float(record.inference_cost or 0.0),
        ]
    return result


def observable_to_legacy(batch: ObservableQueryBatch) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    full: dict[str, dict[str, dict[str, Any]]] = {expert: {} for expert in batch.pool.expert_ids}
    grouped = records_by_question(batch)
    rows: list[dict[str, Any]] = []
    for question_id in sorted(grouped):
        question_records = grouped[question_id]
        representative = question_records[0]
        row_metadata = dict(representative.observable_metadata)
        row_metadata.update({"id": question_id, "benchmark": batch.dataset, "source_dataset": batch.dataset})
        rows.append(row_metadata)
        for record in question_records:
            full[record.expert_id][question_id] = {
                **dict(record.observable_metadata),
                "id": question_id,
                "benchmark": batch.dataset,
                "source_dataset": batch.dataset,
                "pred": record.raw_answer,
                "prediction": record.raw_answer,
                "response": record.raw_output,
                "model_outputs": record.raw_output,
                "model_error": record.missing_reason,
            }
    return full, rows

