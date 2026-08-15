from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .features import records_by_question
from .schema import CanonicalPredictionRecord, ObservableQueryBatch, Selection


@dataclass(frozen=True)
class PermutationConsensusVariant:
    name: str
    vote_mode: str
    source_power: float
    consistency_power: float
    family_balance: bool
    minimum_share: float
    minimum_advantage: float

    def __post_init__(self) -> None:
        if self.vote_mode not in {"raw", "expert_normalized", "expert_majority"}:
            raise ValueError(f"Unknown permutation vote mode: {self.vote_mode}")


def normalize_option(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _metadata(record: CanonicalPredictionRecord) -> tuple[str, int, tuple[str, ...]]:
    metadata = record.observable_metadata
    if "base_question_id" not in metadata or "epoch" not in metadata or "options" not in metadata:
        raise ValueError("GPQA permutation consensus requires base_question_id, epoch, and options")
    options = metadata["options"]
    if not isinstance(options, (list, tuple)) or len(options) < 2:
        raise ValueError("GPQA options metadata must be a non-empty option sequence")
    return str(metadata["base_question_id"]), int(metadata["epoch"]), tuple(
        normalize_option(value) for value in options
    )


def _semantic_answer(record: CanonicalPredictionRecord) -> str | None:
    if not record.valid_output or record.normalized_answer is None:
        return None
    answer = str(record.normalized_answer).strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "Z":
        return None
    _, _, options = _metadata(record)
    index = ord(answer) - ord("A")
    return options[index] if index < len(options) else None


def _reference_semantic(
    reference: Selection,
    rows: Sequence[CanonicalPredictionRecord],
) -> str | None:
    if reference.normalized_answer is None:
        return None
    answer = str(reference.normalized_answer).strip().upper()
    if len(answer) != 1 or not "A" <= answer <= "Z":
        return None
    _, _, options = _metadata(rows[0])
    index = ord(answer) - ord("A")
    return options[index] if index < len(options) else None


def gpqa_permutation_consensus(
    target: ObservableQueryBatch,
    source_accuracy: Mapping[str, float],
    reference: Sequence[Selection],
    variant: PermutationConsensusVariant,
) -> list[Selection]:
    if set(source_accuracy) != set(target.pool.expert_ids):
        raise ValueError("Source accuracy does not cover the GPQA target expert pool")
    reference_by_id = {row.question_id: row for row in reference}
    if set(reference_by_id) != set(target.question_ids):
        raise ValueError("GPQA permutation reference IDs are not aligned")

    grouped = records_by_question(target)
    by_base: dict[str, list[str]] = defaultdict(list)
    epochs_by_base: dict[str, set[int]] = defaultdict(set)
    for question_id, rows in grouped.items():
        base_id, epoch, _ = _metadata(rows[0])
        by_base[base_id].append(question_id)
        epochs_by_base[base_id].add(epoch)

    family_counts = Counter(target.pool.family_by_expert.values())
    result: list[Selection] = []
    for base_id, question_ids in sorted(by_base.items()):
        option_sets = {
            frozenset(_metadata(grouped[question_id][0])[2]) for question_id in question_ids
        }
        if len(option_sets) != 1:
            raise ValueError(f"GPQA shuffled option sets differ for base question {base_id}")

        votes_by_expert: dict[str, list[str]] = defaultdict(list)
        for question_id in question_ids:
            for record in grouped[question_id]:
                semantic = _semantic_answer(record)
                if semantic is not None:
                    votes_by_expert[record.expert_id].append(semantic)

        consistency: dict[str, float] = {}
        expert_weight: dict[str, float] = {}
        semantic_scores: dict[str, float] = defaultdict(float)
        for expert in target.pool.expert_ids:
            votes = votes_by_expert.get(expert, [])
            counts = Counter(votes)
            consistency[expert] = max(counts.values(), default=0) / max(1, len(votes))
            weight = max(float(source_accuracy[expert]), 1e-6) ** variant.source_power
            weight *= max(consistency[expert], 1e-6) ** variant.consistency_power
            if variant.family_balance:
                weight /= max(1, family_counts[target.pool.family_by_expert[expert]])
            expert_weight[expert] = weight
            if not votes:
                continue
            if variant.vote_mode == "expert_majority":
                winner = sorted(counts, key=lambda value: (-counts[value], value))[0]
                semantic_scores[winner] += weight
            else:
                divisor = len(votes) if variant.vote_mode == "expert_normalized" else 1
                for semantic, count in counts.items():
                    semantic_scores[semantic] += weight * count / divisor

        for question_id in sorted(question_ids):
            rows = grouped[question_id]
            reference_row = reference_by_id[question_id]
            reference_semantic = _reference_semantic(reference_row, rows)
            ranked = sorted(
                semantic_scores,
                key=lambda value: (
                    -semantic_scores[value],
                    value != reference_semantic,
                    value,
                ),
            )
            winner = ranked[0] if ranked else reference_semantic
            total = sum(semantic_scores.values())
            winner_score = semantic_scores.get(winner or "", 0.0)
            reference_score = semantic_scores.get(reference_semantic or "", 0.0)
            share = winner_score / max(total, 1e-12)
            advantage = (winner_score - reference_score) / max(total, 1e-12)
            supporters = [record for record in rows if _semantic_answer(record) == winner]
            switch = bool(
                winner is not None
                and winner != reference_semantic
                and supporters
                and share + 1e-12 >= variant.minimum_share
                and advantage + 1e-12 >= variant.minimum_advantage
            )
            common_features = {
                "method": variant.name,
                "gpqa_permutation_consensus": True,
                "base_question_id": base_id,
                "epochs_observed": sorted(epochs_by_base[base_id]),
                "semantic_winner_share": share,
                "semantic_winner_advantage_over_reference": advantage,
                "semantic_supporting_experts_current_epoch": len(supporters),
                "vote_mode": variant.vote_mode,
                "source_power": variant.source_power,
                "consistency_power": variant.consistency_power,
                "family_balance": variant.family_balance,
                "minimum_share": variant.minimum_share,
                "minimum_advantage": variant.minimum_advantage,
                "uses_target_labels": False,
            }
            if not switch:
                features = dict(reference_row.observable_features)
                features.update(common_features)
                features["permutation_consensus_switched"] = False
                result.append(replace(reference_row, observable_features=features))
                continue

            selected = sorted(
                supporters,
                key=lambda record: (
                    -expert_weight[record.expert_id],
                    -consistency[record.expert_id],
                    record.uncertainty,
                    record.expert_id,
                ),
            )[0]
            cluster_scores: dict[str, float] = {}
            for record in rows:
                semantic = _semantic_answer(record)
                if semantic is not None and record.per_query_cluster_id is not None:
                    cluster_scores[str(record.per_query_cluster_id)] = semantic_scores.get(
                        semantic, 0.0
                    )
            common_features["permutation_consensus_switched"] = True
            common_features["selected_expert_consistency"] = consistency[selected.expert_id]
            result.append(
                Selection(
                    question_id=question_id,
                    selected_cluster_id=selected.per_query_cluster_id,
                    selected_expert_id=selected.expert_id,
                    normalized_answer=selected.normalized_answer,
                    cluster_scores=cluster_scores,
                    expert_scores={
                        expert: float(value) for expert, value in sorted(expert_weight.items())
                    },
                    fallback_reason=None,
                    observable_features=common_features,
                    tie_breaking=(
                        "semantic_score_then_reference_semantic_then_option_text;"
                        "source_weight_then_consistency_then_uncertainty_then_expert_id"
                    ),
                )
            )
    return sorted(result, key=lambda row: row.question_id)
