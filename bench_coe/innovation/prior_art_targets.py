from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .artifacts import sha256_file
from .schema import ExpertPool, ObservableQueryBatch


def project_observable_pool(
    batch: ObservableQueryBatch,
    expert_ids: Iterable[str],
) -> ObservableQueryBatch:
    requested = tuple(sorted(str(value) for value in expert_ids))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Projected target expert pool must be non-empty and unique")
    unknown = set(requested).difference(batch.pool.expert_ids)
    if unknown:
        raise ValueError(f"Projected target pool contains unknown experts: {sorted(unknown)}")
    keep = set(requested)
    return ObservableQueryBatch(
        dataset=batch.dataset,
        split=batch.split,
        modality=batch.modality,
        pool=ExpertPool(
            requested,
            {expert: batch.pool.family_by_expert[expert] for expert in requested},
        ),
        records=tuple(record for record in batch.records if record.expert_id in keep),
    )


def authenticate_prediction_package(
    seed_dir: Path,
    target_manifest: dict[str, Any],
) -> None:
    hashes = target_manifest.get("prediction_hashes_before_evaluation")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Target prediction package has no authenticated predictions")
    for method, expected_hash in hashes.items():
        relative = target_manifest.get("prediction_paths", {}).get(method)
        if not isinstance(relative, str):
            raise RuntimeError(f"Target prediction path is missing for method {method}")
        path = seed_dir / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Target prediction hash mismatch: {path}")


def target_environment_by_question(batch: ObservableQueryBatch) -> dict[str, str]:
    return {
        question_id: batch.for_question(question_id)[0].subject
        for question_id in batch.question_ids
    }
