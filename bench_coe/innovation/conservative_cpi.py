from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import Selection, SourceTrainingLabels


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    macro_delta: float
    micro_delta: float
    worst_environment_delta: float
    nonnegative_environment_fraction: float
    switch_count: int
    feasible: bool
    utility: float


def _aligned(
    candidate: Sequence[Selection],
    baseline: Sequence[Selection],
) -> tuple[list[str], dict[str, Selection], dict[str, Selection]]:
    candidate_by_id = {item.question_id: item for item in candidate}
    baseline_by_id = {item.question_id: item for item in baseline}
    if len(candidate_by_id) != len(candidate) or len(baseline_by_id) != len(baseline):
        raise ValueError("Conservative CPI predictions contain duplicate question IDs")
    if set(candidate_by_id) != set(baseline_by_id):
        raise ValueError("Conservative CPI candidate and baseline IDs are not aligned")
    return sorted(candidate_by_id), candidate_by_id, baseline_by_id


def proposal_margin(candidate: Selection, baseline: Selection) -> float:
    if candidate.question_id != baseline.question_id:
        raise ValueError("Cannot compare selections from different questions")
    if candidate.selected_cluster_id is None or baseline.selected_cluster_id is None:
        return -1.0
    if candidate.selected_cluster_id == baseline.selected_cluster_id:
        return 1.0
    candidate_score = candidate.cluster_scores.get(str(candidate.selected_cluster_id))
    baseline_score = candidate.cluster_scores.get(str(baseline.selected_cluster_id))
    if candidate_score is None or baseline_score is None:
        return -1.0
    return float(candidate_score) - float(baseline_score)


def apply_conservative_gate(
    candidate: Sequence[Selection],
    baseline: Sequence[Selection],
    threshold: float,
    method: str = "conservative_cpi",
) -> list[Selection]:
    ids, candidate_by_id, baseline_by_id = _aligned(candidate, baseline)
    result: list[Selection] = []
    for question_id in ids:
        proposal = candidate_by_id[question_id]
        reference = baseline_by_id[question_id]
        margin = proposal_margin(proposal, reference)
        same_cluster = proposal.selected_cluster_id == reference.selected_cluster_id
        accept = same_cluster or margin >= threshold
        features: dict[str, Any] = {
            **dict(proposal.observable_features),
            "method": method,
            "proposal_margin": margin,
            "conservative_threshold": threshold,
            "proposal_cluster_id": proposal.selected_cluster_id,
            "source_best_cluster_id": reference.selected_cluster_id,
            "proposal_expert_id": proposal.selected_expert_id,
            "source_best_expert_id": reference.selected_expert_id,
            "proposal_accepted": bool(accept and not same_cluster),
        }
        if accept:
            result.append(replace(proposal, fallback_reason=None, observable_features=features))
        else:
            result.append(
                Selection(
                    question_id=question_id,
                    selected_cluster_id=reference.selected_cluster_id,
                    selected_expert_id=reference.selected_expert_id,
                    normalized_answer=reference.normalized_answer,
                    cluster_scores=dict(proposal.cluster_scores),
                    expert_scores=dict(reference.expert_scores),
                    fallback_reason="conservative_margin_below_threshold",
                    observable_features=features,
                    tie_breaking=reference.tie_breaking,
                )
            )
    return result


def calibrate_threshold(
    candidate: Sequence[Selection],
    baseline: Sequence[Selection],
    labels: SourceTrainingLabels,
    environment_by_question: Mapping[str, str],
    thresholds: Sequence[float],
    min_worst_delta: float,
    min_micro_delta: float,
    worst_weight: float,
) -> tuple[float, list[ThresholdCalibration]]:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Conservative CPI thresholds require source training labels")
    ids, _, baseline_by_id = _aligned(candidate, baseline)
    if set(ids).difference(environment_by_question):
        raise ValueError("Conservative CPI calibration is missing source environments")
    grid = sorted(set(float(value) for value in thresholds))
    if not grid or grid[-1] <= 1.0:
        raise ValueError("Threshold grid must include a no-switch fallback above 1.0")
    baseline_correct = {
        question_id: float(bool(labels.get(question_id, baseline_by_id[question_id].selected_expert_id or "")))
        for question_id in ids
    }
    diagnostics: list[ThresholdCalibration] = []
    for threshold in grid:
        gated = apply_conservative_gate(candidate, baseline, threshold)
        deltas_by_environment: dict[str, list[float]] = {}
        deltas: list[float] = []
        switches = 0
        for selection in gated:
            question_id = selection.question_id
            value = float(bool(labels.get(question_id, selection.selected_expert_id or ""))) - baseline_correct[question_id]
            deltas.append(value)
            environment = str(environment_by_question[question_id])
            deltas_by_environment.setdefault(environment, []).append(value)
            switches += int(selection.selected_cluster_id != baseline_by_id[question_id].selected_cluster_id)
        environment_deltas = [float(np.mean(values)) for values in deltas_by_environment.values()]
        macro_delta = float(np.mean(environment_deltas)) if environment_deltas else 0.0
        micro_delta = float(np.mean(deltas)) if deltas else 0.0
        worst_delta = min(environment_deltas, default=0.0)
        nonnegative = float(np.mean([value >= 0.0 for value in environment_deltas])) if environment_deltas else 1.0
        feasible = worst_delta >= min_worst_delta and micro_delta >= min_micro_delta
        utility = macro_delta + worst_weight * worst_delta if feasible else float("-inf")
        diagnostics.append(
            ThresholdCalibration(
                threshold=threshold,
                macro_delta=macro_delta,
                micro_delta=micro_delta,
                worst_environment_delta=worst_delta,
                nonnegative_environment_fraction=nonnegative,
                switch_count=switches,
                feasible=feasible,
                utility=utility,
            )
        )
    selected = max(
        diagnostics,
        key=lambda row: (
            row.feasible,
            row.utility,
            row.macro_delta,
            row.micro_delta,
            row.worst_environment_delta,
            row.threshold,
        ),
    )
    if not selected.feasible:
        raise AssertionError("No feasible conservative fallback threshold was found")
    return selected.threshold, diagnostics


def grouped_environment_folds(
    labels: SourceTrainingLabels,
    groups: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    if groups < 2:
        raise ValueError("Conservative CPI requires at least two inner groups")
    environments = sorted(set(labels.environment_by_question.values()))
    if len(environments) < groups:
        raise ValueError("Not enough source environments for grouped inner OOF")
    all_ids = set(labels.environment_by_question)
    result: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for group in range(groups):
        heldout_environments = tuple(environment for index, environment in enumerate(environments) if index % groups == group)
        heldout = set(heldout_environments)
        train_ids = tuple(sorted(qid for qid in all_ids if labels.environment_by_question[qid] not in heldout))
        test_ids = tuple(sorted(qid for qid in all_ids if labels.environment_by_question[qid] in heldout))
        result.append((heldout_environments, train_ids, test_ids))
    if set().union(*(set(test_ids) for _, _, test_ids in result)) != all_ids:
        raise AssertionError("Inner grouped OOF does not cover every calibration question")
    return result
