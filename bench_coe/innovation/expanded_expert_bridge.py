from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluation import exact_mcnemar, paired_bootstrap_delta, selection_correctness
from .schema import EvaluationLabels, Selection


def row_id(row: Mapping[str, Any]) -> str:
    value = row.get("id", row.get("question_id", row.get("pid")))
    if value is None:
        raise KeyError(f"Missing question ID in row keys={sorted(row)}")
    return str(value)


def filter_rows_by_id_prefix(
    rows: Iterable[Mapping[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    """Project a mixed raw cache onto one deterministic, ID-defined source split."""

    selected = [dict(row) for row in rows if row_id(row).startswith(prefix)]
    ids = [row_id(row) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Projected source cache contains duplicate question IDs")
    return sorted(selected, key=row_id)


def annotate_bridge_predictions(
    selections: Sequence[Selection],
    *,
    source_dataset: str,
    source_split: str,
    source_accuracy: Mapping[str, float],
) -> list[Selection]:
    result: list[Selection] = []
    for selection in selections:
        features = dict(selection.observable_features)
        features.update(
            {
                "expanded_expert_bridge": True,
                "source_dataset": source_dataset,
                "source_split": source_split,
                "source_accuracy_by_expert": dict(sorted(source_accuracy.items())),
                "uses_target_labels": False,
            }
        )
        result.append(replace(selection, observable_features=features))
    return result


def cross_pool_paired_comparison(
    candidate: Sequence[Selection],
    candidate_labels: EvaluationLabels,
    reference: Sequence[Selection],
    reference_labels: EvaluationLabels,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compare aligned selections whose selected experts come from different pools."""

    candidate_map = selection_correctness(list(candidate), candidate_labels)
    reference_map = selection_correctness(list(reference), reference_labels)
    if set(candidate_map) != set(reference_map):
        raise ValueError("Cross-pool candidate/reference question IDs are not aligned")
    ids = sorted(candidate_map)
    candidate_values = np.asarray([candidate_map[qid] for qid in ids], dtype=np.int8)
    reference_values = np.asarray([reference_map[qid] for qid in ids], dtype=np.int8)
    rescue, harm, p_value = exact_mcnemar(candidate_values, reference_values)
    ci_low, ci_high = paired_bootstrap_delta(
        candidate_values,
        reference_values,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    return (
        {
            "samples": len(ids),
            "accuracy": float(candidate_values.mean()) if len(ids) else 0.0,
            "fcrg_full_accuracy": float(reference_values.mean()) if len(ids) else 0.0,
            "delta_vs_fcrg_full": (
                float((candidate_values - reference_values).mean()) if len(ids) else 0.0
            ),
            "rescue_count": rescue,
            "harm_count": harm,
            "exact_mcnemar_p": p_value,
            "paired_bootstrap_delta_ci95": [ci_low, ci_high],
        },
        candidate_values,
        reference_values,
    )
