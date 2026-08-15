from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .features import records_by_question
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


@dataclass(frozen=True)
class ConditionedConsensusVariant:
    name: str
    prior_strength: float
    reliability_power: float
    uncertainty_temperature: float
    family_balance: bool
    validity_power: float


def fit_conditioned_expert_profiles(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    subject_groups: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Conditioned expert profiles require SourceTrainingLabels")
    if batch.dataset != labels.dataset or batch.split != labels.split:
        raise ValueError("Conditioned source observables and labels are not aligned")
    grouped = records_by_question(batch)
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"questions": 0, "valid": 0, "correct": 0})
    )
    global_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"questions": 0, "valid": 0, "correct": 0}
    )
    missing_subjects: set[str] = set()
    for question_id, rows in grouped.items():
        subject = rows[0].subject
        group = subject_groups.get(subject)
        if group is None:
            missing_subjects.add(subject)
            continue
        for row in rows:
            value = bool(labels.get(question_id, row.expert_id))
            for target in (counts[group][row.expert_id], global_counts[row.expert_id]):
                target["questions"] += 1
                target["valid"] += int(row.valid_output)
                target["correct"] += int(row.valid_output and value)
    if missing_subjects:
        raise ValueError(f"Source subjects lack configured groups: {sorted(missing_subjects)}")
    if set(global_counts) != set(batch.pool.expert_ids):
        raise ValueError("Conditioned profiles do not cover the source expert pool")
    return {
        "groups": {
            group: {expert: dict(values) for expert, values in sorted(by_expert.items())}
            for group, by_expert in sorted(counts.items())
        },
        "global": {expert: dict(values) for expert, values in sorted(global_counts.items())},
        "source_subject_groups": dict(sorted(subject_groups.items())),
    }


def _expert_weight(
    expert: str,
    group: str,
    profiles: Mapping[str, Any],
    variant: ConditionedConsensusVariant,
) -> tuple[float, float, float]:
    global_values = profiles["global"][expert]
    global_valid = float(global_values["valid"])
    global_accuracy = float(global_values["correct"]) / max(global_valid, 1.0)
    group_values = profiles["groups"].get(group, {}).get(
        expert, {"questions": 0, "valid": 0, "correct": 0}
    )
    group_valid = float(group_values["valid"])
    reliability = (
        float(group_values["correct"]) + variant.prior_strength * global_accuracy
    ) / max(group_valid + variant.prior_strength, 1e-12)
    global_validity = float(global_values["valid"]) / max(float(global_values["questions"]), 1.0)
    group_validity = (
        float(group_values["valid"]) + variant.prior_strength * global_validity
    ) / max(float(group_values["questions"]) + variant.prior_strength, 1e-12)
    weight = max(reliability, 1e-6) ** variant.reliability_power
    weight *= max(group_validity, 1e-6) ** variant.validity_power
    return float(weight), float(reliability), float(group_validity)


def conditioned_expert_consensus(
    target: ObservableQueryBatch,
    profiles: Mapping[str, Any],
    target_subject_groups: Mapping[str, str],
    variant: ConditionedConsensusVariant,
    *,
    reference: Sequence[Selection],
) -> list[Selection]:
    reference_by_id = {row.question_id: row for row in reference}
    if set(reference_by_id) != set(target.question_ids):
        raise ValueError("Conditioned consensus/reference IDs are not aligned")
    grouped = records_by_question(target)
    missing_subjects = {
        rows[0].subject
        for rows in grouped.values()
        if rows[0].subject not in target_subject_groups
    }
    if missing_subjects:
        raise ValueError(f"Target subjects lack configured groups: {sorted(missing_subjects)}")
    result: list[Selection] = []
    for question_id in target.question_ids:
        rows = grouped[question_id]
        subject = rows[0].subject
        group = target_subject_groups[subject]
        reference_row = reference_by_id[question_id]
        valid_rows = [row for row in rows if row.valid_output and row.normalized_answer is not None]
        family_counts = Counter(row.expert_family for row in valid_rows)
        answer_scores: dict[str, float] = defaultdict(float)
        expert_scores: dict[str, float] = {}
        reliability: dict[str, float] = {}
        validity: dict[str, float] = {}
        by_answer: dict[str, list[Any]] = defaultdict(list)
        for row in valid_rows:
            weight, expert_reliability, expert_validity = _expert_weight(
                row.expert_id, group, profiles, variant
            )
            if variant.family_balance:
                weight /= max(1, family_counts[row.expert_family])
            if variant.uncertainty_temperature > 0.0:
                weight *= float(
                    np.exp(
                        -max(float(row.uncertainty), 0.0)
                        / variant.uncertainty_temperature
                    )
                )
            answer = str(row.normalized_answer)
            answer_scores[answer] += weight
            expert_scores[row.expert_id] = weight
            reliability[row.expert_id] = expert_reliability
            validity[row.expert_id] = expert_validity
            by_answer[answer].append(row)
        if not answer_scores:
            chosen_answer = reference_row.normalized_answer
            chosen_expert = reference_row.selected_expert_id
            chosen_cluster = reference_row.selected_cluster_id
            winner_share = 0.0
            winner_advantage = 0.0
            fallback_reason = "no_valid_expert_output"
        else:
            ranked = sorted(
                answer_scores,
                key=lambda answer: (
                    -answer_scores[answer],
                    answer != reference_row.normalized_answer,
                    answer,
                ),
            )
            chosen_answer = ranked[0]
            total = sum(answer_scores.values())
            winner_share = answer_scores[chosen_answer] / max(total, 1e-12)
            reference_score = answer_scores.get(str(reference_row.normalized_answer), 0.0)
            winner_advantage = (answer_scores[chosen_answer] - reference_score) / max(total, 1e-12)
            selected_record = sorted(
                by_answer[chosen_answer],
                key=lambda row: (
                    -expert_scores[row.expert_id],
                    row.uncertainty,
                    row.expert_id,
                ),
            )[0]
            chosen_expert = selected_record.expert_id
            chosen_cluster = selected_record.per_query_cluster_id
            fallback_reason = None
        cluster_scores: dict[str, float] = {}
        for answer, score in answer_scores.items():
            cluster = sorted(
                by_answer[answer],
                key=lambda row: (row.per_query_cluster_id, row.expert_id),
            )[0].per_query_cluster_id
            if cluster is not None:
                cluster_scores[str(cluster)] = float(score)
        valid_mask = {row.expert_id: bool(row.valid_output) for row in rows}
        result.append(
            Selection(
                question_id=question_id,
                selected_cluster_id=chosen_cluster,
                selected_expert_id=chosen_expert,
                normalized_answer=chosen_answer,
                cluster_scores=cluster_scores,
                expert_scores=expert_scores,
                fallback_reason=fallback_reason,
                observable_features={
                    "method": variant.name,
                    "conditioned_expert_consensus": True,
                    "source_conditioning_group": group,
                    "source_prior_strength": variant.prior_strength,
                    "source_reliability_power": variant.reliability_power,
                    "uncertainty_temperature": variant.uncertainty_temperature,
                    "family_balance": variant.family_balance,
                    "source_validity_power": variant.validity_power,
                    "consensus_winner_share": winner_share,
                    "consensus_winner_advantage": winner_advantage,
                    "valid_fraction": sum(valid_mask.values()) / max(1, len(valid_mask)),
                    "valid_mask": valid_mask,
                    "missing_mask": {expert: not value for expert, value in valid_mask.items()},
                    "source_expert_reliability": reliability,
                    "source_expert_validity": validity,
                    "conditioned_consensus_uses_target_labels": False,
                },
                tie_breaking=(
                    "conditioned_cluster_score_then_reference_answer_then_answer;"
                    "expert_weight_then_uncertainty_then_expert_id"
                ),
            )
        )
    return result
