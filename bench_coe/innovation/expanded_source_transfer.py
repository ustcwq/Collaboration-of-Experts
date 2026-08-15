from __future__ import annotations

from dataclasses import replace
from typing import Any

from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import SourceBestSelector, source_accuracy


def leave_one_environment_out_source_best(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
) -> tuple[list[Selection], dict[str, Any]]:
    """Select an expert for each held-out source environment without same-env labels."""

    environments = sorted(set(labels.environment_by_question.values()))
    if len(environments) < 2:
        raise ValueError("Leave-one-environment-out selection requires at least two environments")
    predictions: list[Selection] = []
    fold_audit: dict[str, Any] = {}
    covered: set[str] = set()
    for environment in environments:
        heldout_ids = sorted(
            question_id
            for question_id, value in labels.environment_by_question.items()
            if value == environment
        )
        train_ids = sorted(set(batch.question_ids).difference(heldout_ids))
        if not heldout_ids or not train_ids:
            raise ValueError(f"Invalid source fold for environment {environment!r}")
        train_batch = batch.subset(train_ids)
        train_labels = labels.subset(train_ids)
        accuracies = source_accuracy(train_batch, train_labels)
        fold_rows = SourceBestSelector().fit(train_batch, train_labels).predict(
            batch.subset(heldout_ids)
        )
        for row in fold_rows:
            features = dict(row.observable_features)
            features.update(
                {
                    "expanded_source_loso": True,
                    "heldout_environment": environment,
                    "training_environment_count": len(environments) - 1,
                    "training_question_count": len(train_ids),
                    "heldout_environment_labels_used": False,
                    "source_accuracy_by_expert_excluding_heldout": dict(
                        sorted(accuracies.items())
                    ),
                }
            )
            predictions.append(replace(row, observable_features=features))
        covered.update(heldout_ids)
        fold_audit[environment] = {
            "heldout_questions": len(heldout_ids),
            "training_questions": len(train_ids),
            "source_accuracy_by_expert": dict(sorted(accuracies.items())),
        }

    if covered != set(batch.question_ids) or len(predictions) != len(batch.question_ids):
        raise RuntimeError("Leave-one-environment-out folds did not cover every source row once")
    return sorted(predictions, key=lambda row: row.question_id), {
        "environments": len(environments),
        "questions": len(predictions),
        "folds": fold_audit,
    }
