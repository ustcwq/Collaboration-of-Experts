from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .conditioned_expert_consensus import _expert_weight
from .features import records_by_question
from .schema import ObservableQueryBatch, Selection


@dataclass(frozen=True)
class AdaptiveConsensusVariant:
    name: str
    prior_strength: float
    reliability_power: float
    uncertainty_temperature: float
    validity_power: float
    support_power: float
    aggregation: str
    family_balance: bool
    em_mix: float
    em_iterations: int


def _source_rates(
    target: ObservableQueryBatch,
    profiles: Mapping[str, Any],
    target_subject_groups: Mapping[str, str],
    variant: AdaptiveConsensusVariant,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    groups = set(target_subject_groups.values())
    reliability: dict[str, dict[str, float]] = {}
    validity: dict[str, dict[str, float]] = {}
    for group in sorted(groups):
        reliability[group] = {}
        validity[group] = {}
        for expert in target.pool.expert_ids:
            _, expert_reliability, expert_validity = _expert_weight(
                expert, group, profiles, variant
            )
            reliability[group][expert] = expert_reliability
            validity[group][expert] = expert_validity
    return reliability, validity


def _expert_vote_weight(
    reliability: float,
    validity: float,
    uncertainty: float,
    variant: AdaptiveConsensusVariant,
) -> float:
    weight = max(reliability, 1e-6) ** variant.reliability_power
    weight *= max(validity, 1e-6) ** variant.validity_power
    if variant.uncertainty_temperature > 0.0:
        weight *= float(
            np.exp(
                -max(float(uncertainty), 0.0)
                / variant.uncertainty_temperature
            )
        )
    return float(weight)


def _answer_scores(
    rows: Sequence[Any],
    group: str,
    reliability: Mapping[str, Mapping[str, float]],
    validity: Mapping[str, Mapping[str, float]],
    variant: AdaptiveConsensusVariant,
    *,
    exclude_expert: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    valid_rows = [
        row
        for row in rows
        if row.expert_id != exclude_expert
        and row.valid_output
        and row.normalized_answer is not None
    ]
    if not valid_rows:
        return {}, {}
    family_counts = Counter(row.expert_family for row in valid_rows)
    by_answer: dict[str, list[Any]] = defaultdict(list)
    expert_weights: dict[str, float] = {}
    for row in valid_rows:
        weight = _expert_vote_weight(
            reliability[group][row.expert_id],
            validity[group][row.expert_id],
            row.uncertainty,
            variant,
        )
        if variant.family_balance:
            weight /= max(1, family_counts[row.expert_family])
        expert_weights[row.expert_id] = weight
        by_answer[str(row.normalized_answer)].append(row)

    answer_scores: dict[str, float] = {}
    answer_count = len(by_answer)
    valid_count = len(valid_rows)
    if variant.aggregation == "sum":
        for answer, supporters in by_answer.items():
            score = sum(expert_weights[row.expert_id] for row in supporters)
            support = len(supporters) / max(1, valid_count)
            answer_scores[answer] = float(score * max(support, 1e-12) ** variant.support_power)
    elif variant.aggregation == "noisy_or":
        for answer, supporters in by_answer.items():
            miss_probability = 1.0
            for row in supporters:
                miss_probability *= 1.0 - min(max(expert_weights[row.expert_id], 0.0), 1.0)
            support = len(supporters) / max(1, valid_count)
            answer_scores[answer] = float(
                (1.0 - miss_probability) * max(support, 1e-12) ** variant.support_power
            )
    elif variant.aggregation == "dawid_skene":
        chance = 1.0 / max(answer_count, 1)
        log_scores: dict[str, float] = {}
        for answer, supporters in by_answer.items():
            support = len(supporters) / max(1, valid_count)
            value = variant.support_power * float(np.log(max(support, 1e-12)))
            for row in valid_rows:
                source_rate = reliability[group][row.expert_id]
                logit = float(
                    np.log(max(source_rate, 1e-6) / max(1.0 - source_rate, 1e-6))
                )
                calibrated = 1.0 / (1.0 + float(np.exp(-variant.reliability_power * logit)))
                uncertainty_factor = 1.0
                if variant.uncertainty_temperature > 0.0:
                    uncertainty_factor = float(
                        np.exp(
                            -max(float(row.uncertainty), 0.0)
                            / variant.uncertainty_temperature
                        )
                    )
                calibrated = chance + (calibrated - chance) * uncertainty_factor
                calibrated *= max(validity[group][row.expert_id], 1e-6) ** variant.validity_power
                calibrated = min(max(calibrated, 1e-6), 1.0 - 1e-6)
                family_scale = (
                    1.0 / max(1, family_counts[row.expert_family])
                    if variant.family_balance
                    else 1.0
                )
                if str(row.normalized_answer) == answer:
                    likelihood = calibrated
                else:
                    likelihood = (1.0 - calibrated) / max(answer_count - 1, 1)
                value += family_scale * float(np.log(max(likelihood, 1e-12)))
            log_scores[answer] = value
        maximum = max(log_scores.values())
        answer_scores = {
            answer: float(np.exp(np.clip(value - maximum, -60.0, 0.0)))
            for answer, value in log_scores.items()
        }
    else:
        raise ValueError(f"Unknown adaptive aggregation: {variant.aggregation}")
    return answer_scores, expert_weights


def adapt_target_reliability(
    target: ObservableQueryBatch,
    profiles: Mapping[str, Any],
    target_subject_groups: Mapping[str, str],
    variant: AdaptiveConsensusVariant,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    grouped = records_by_question(target)
    missing = {
        rows[0].subject
        for rows in grouped.values()
        if rows[0].subject not in target_subject_groups
    }
    if missing:
        raise ValueError(f"Target subjects lack configured groups: {sorted(missing)}")
    source_reliability, validity = _source_rates(
        target, profiles, target_subject_groups, variant
    )
    reliability = {
        group: dict(values) for group, values in source_reliability.items()
    }
    if variant.em_mix <= 0.0 or variant.em_iterations <= 0:
        return reliability, validity
    if not 0.0 <= variant.em_mix <= 1.0:
        raise ValueError("Adaptive EM mix must be in [0, 1]")

    for _ in range(variant.em_iterations):
        totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for question_id in target.question_ids:
            rows = grouped[question_id]
            group = target_subject_groups[rows[0].subject]
            for row in rows:
                if not row.valid_output or row.normalized_answer is None:
                    continue
                scores, _ = _answer_scores(
                    rows,
                    group,
                    reliability,
                    validity,
                    variant,
                    exclude_expert=row.expert_id,
                )
                scale = sum(scores.values())
                if scale <= 0.0:
                    continue
                totals[group][row.expert_id] += scores.get(
                    str(row.normalized_answer), 0.0
                ) / scale
                counts[group][row.expert_id] += 1
        for group in reliability:
            for expert in reliability[group]:
                count = counts[group].get(expert, 0)
                if count:
                    target_rate = totals[group][expert] / count
                    reliability[group][expert] = float(
                        (1.0 - variant.em_mix) * source_reliability[group][expert]
                        + variant.em_mix * target_rate
                    )
    return reliability, validity


def adaptive_expert_consensus(
    target: ObservableQueryBatch,
    profiles: Mapping[str, Any],
    target_subject_groups: Mapping[str, str],
    variant: AdaptiveConsensusVariant,
    *,
    reference: Sequence[Selection],
) -> list[Selection]:
    reference_by_id = {row.question_id: row for row in reference}
    if set(reference_by_id) != set(target.question_ids):
        raise ValueError("Adaptive consensus/reference IDs are not aligned")
    grouped = records_by_question(target)
    reliability, validity = adapt_target_reliability(
        target, profiles, target_subject_groups, variant
    )
    result: list[Selection] = []
    for question_id in target.question_ids:
        rows = grouped[question_id]
        subject = rows[0].subject
        group = target_subject_groups[subject]
        reference_row = reference_by_id[question_id]
        answer_scores, expert_scores = _answer_scores(
            rows, group, reliability, validity, variant
        )
        by_answer: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            if row.valid_output and row.normalized_answer is not None:
                by_answer[str(row.normalized_answer)].append(row)
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
            reference_score = answer_scores.get(
                str(reference_row.normalized_answer), 0.0
            )
            winner_advantage = (
                answer_scores[chosen_answer] - reference_score
            ) / max(total, 1e-12)
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
            cluster = min(
                row.per_query_cluster_id
                for row in by_answer[answer]
                if row.per_query_cluster_id is not None
            )
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
                    "adaptive_expert_consensus": True,
                    "source_conditioning_group": group,
                    "adaptive_aggregation": variant.aggregation,
                    "source_prior_strength": variant.prior_strength,
                    "source_reliability_power": variant.reliability_power,
                    "uncertainty_temperature": variant.uncertainty_temperature,
                    "source_validity_power": variant.validity_power,
                    "support_power": variant.support_power,
                    "family_balance": variant.family_balance,
                    "target_unlabeled_em_mix": variant.em_mix,
                    "target_unlabeled_em_iterations": variant.em_iterations,
                    "consensus_winner_share": winner_share,
                    "consensus_winner_advantage": winner_advantage,
                    "valid_mask": valid_mask,
                    "missing_mask": {
                        expert: not value for expert, value in valid_mask.items()
                    },
                    "adapted_expert_reliability": reliability[group],
                    "source_expert_validity": validity[group],
                    "adaptive_consensus_uses_target_labels": False,
                },
                tie_breaking=(
                    "adaptive_cluster_score_then_reference_answer_then_answer;"
                    "expert_weight_then_uncertainty_then_expert_id"
                ),
            )
        )
    return result
