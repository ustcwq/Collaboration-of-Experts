from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .schema import Selection


@dataclass(frozen=True)
class ConsensusVariant:
    name: str
    subset: str
    global_weighting: str
    confidence_power: float = 0.0
    fallback_share: float = 0.0
    minimum_advantage: float = 0.0


def _method_family(method: str) -> str:
    if method.startswith("fcrg_"):
        return "fcrg"
    if method.startswith("cascade_fcrg_"):
        return "cascade"
    if method.startswith(("knop_", "knora_", "ola_", "lca_", "mcb_")):
        return "dynamic_selection"
    if method.startswith("smoothie_"):
        return "smoothie"
    if method.startswith("more_"):
        return "more"
    if method.startswith(("learned_", "meta_des_")):
        return "learned"
    if method.startswith(("global_", "local_", "agreement_")):
        return "global_local"
    if method.startswith(("majority_", "uncertainty_", "fast_")):
        return "simple"
    return method.split("_", 1)[0]


def _selected_confidence(selection: Selection) -> float:
    values = np.asarray(list(selection.cluster_scores.values()), dtype=float)
    if len(values) <= 1 or not np.all(np.isfinite(values)):
        return 1.0
    selected = float(selection.cluster_scores.get(str(selection.selected_cluster_id), np.max(values)))
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low + 1e-12:
        return 1.0
    return float(np.clip((selected - low) / (high - low), 0.0, 1.0))


def _aligned_rows(
    rows_by_method: Mapping[str, Sequence[Selection]],
) -> tuple[tuple[str, ...], dict[str, dict[str, Selection]]]:
    if not rows_by_method:
        raise ValueError("Consensus requires at least one method")
    indexed = {
        method: {row.question_id: row for row in rows}
        for method, rows in sorted(rows_by_method.items())
    }
    first_ids = set(next(iter(indexed.values())))
    if not first_ids:
        raise ValueError("Consensus methods contain no predictions")
    for method, rows in indexed.items():
        if set(rows) != first_ids:
            raise ValueError(f"Consensus method IDs are not aligned: {method}")
    return tuple(sorted(first_ids)), indexed


def method_subsets(methods: Sequence[str]) -> dict[str, tuple[str, ...]]:
    ordered = tuple(sorted(methods))
    core = tuple(
        method
        for method in ordered
        if not method.startswith(("fcrg_", "cascade_fcrg_", "knop_k"))
    )
    nonstochastic = tuple(
        method
        for method in ordered
        if not any(token in method for token in ("random", "relabel", "learned_mlp"))
    )
    core_fcrg = tuple(
        method
        for method in ordered
        if method in core or method == "fcrg_full"
    )
    result = {
        "all": ordered,
        "core": core,
        "nonstochastic": nonstochastic,
        "core_fcrg": core_fcrg,
    }
    for name, subset in result.items():
        if not subset:
            raise ValueError(f"Consensus subset is empty: {name}")
    return result


def _answer_signatures(
    methods: Sequence[str],
    question_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Selection]],
) -> dict[str, tuple[str | None, ...]]:
    return {
        method: tuple(indexed[method][question_id].normalized_answer for question_id in question_ids)
        for method in methods
    }


def _agreement_reliability(
    methods: Sequence[str],
    question_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Selection]],
) -> dict[str, float]:
    signatures = _answer_signatures(methods, question_ids, indexed)
    majority: list[str | None] = []
    for index in range(len(question_ids)):
        counts = Counter(
            signature[index]
            for signature in signatures.values()
            if signature[index] is not None
        )
        majority.append(sorted(counts, key=lambda answer: (-counts[answer], str(answer)))[0] if counts else None)
    reliability: dict[str, float] = {}
    for method, signature in signatures.items():
        comparable = [
            int(answer is not None and answer == reference)
            for answer, reference in zip(signature, majority, strict=True)
            if reference is not None
        ]
        reliability[method] = float(np.mean(comparable)) if comparable else 0.0
    return reliability


def global_method_weights(
    mode: str,
    methods: Sequence[str],
    question_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Selection]],
) -> dict[str, float]:
    methods = tuple(methods)
    weights = {method: 1.0 for method in methods}
    if mode == "equal":
        return weights
    if mode in {"signature", "signature_agreement"}:
        signatures = _answer_signatures(methods, question_ids, indexed)
        counts = Counter(signatures.values())
        weights = {method: 1.0 / counts[signatures[method]] for method in methods}
    elif mode == "family":
        counts = Counter(_method_family(method) for method in methods)
        weights = {method: 1.0 / counts[_method_family(method)] for method in methods}
    elif mode != "agreement":
        raise ValueError(f"Unknown consensus weighting: {mode}")
    if mode in {"agreement", "signature_agreement"}:
        reliability = _agreement_reliability(methods, question_ids, indexed)
        for method in methods:
            weights[method] *= max(reliability[method], 1e-3) ** 2
    scale = sum(weights.values())
    return {method: value * len(methods) / max(scale, 1e-12) for method, value in weights.items()}


def consensus_selections(
    rows_by_method: Mapping[str, Sequence[Selection]],
    variant: ConsensusVariant,
    *,
    reference_method: str = "fcrg_full",
    external_method_weights: Mapping[str, float] | None = None,
) -> list[Selection]:
    question_ids, indexed = _aligned_rows(rows_by_method)
    subsets = method_subsets(tuple(indexed))
    if variant.subset not in subsets:
        raise ValueError(f"Unknown consensus subset: {variant.subset}")
    methods = subsets[variant.subset]
    if reference_method not in indexed:
        raise ValueError(f"Missing consensus reference method: {reference_method}")
    weights = global_method_weights(
        variant.global_weighting,
        methods,
        question_ids,
        indexed,
    )
    if external_method_weights is not None:
        missing = set(methods).difference(external_method_weights)
        if missing:
            raise ValueError(f"External consensus weights are incomplete: {sorted(missing)}")
        weights = {
            method: weights[method] * max(float(external_method_weights[method]), 0.0)
            for method in methods
        }
        if sum(weights.values()) <= 0.0:
            raise ValueError("External consensus weights contain no positive mass")
    selections: list[Selection] = []
    for question_id in question_ids:
        answer_scores: dict[str, float] = defaultdict(float)
        voters: dict[str, list[tuple[str, Selection, float]]] = defaultdict(list)
        total = 0.0
        for method in methods:
            row = indexed[method][question_id]
            if row.normalized_answer is None or row.selected_expert_id is None:
                continue
            confidence = _selected_confidence(row) ** variant.confidence_power
            contribution = weights[method] * confidence
            answer = str(row.normalized_answer)
            answer_scores[answer] += contribution
            voters[answer].append((method, row, contribution))
            total += contribution
        reference = indexed[reference_method][question_id]
        if not answer_scores:
            chosen = reference
            winner_share = 0.0
            winner_advantage = 0.0
            fallback_reason = "no_valid_method_vote"
        else:
            ranked = sorted(
                answer_scores,
                key=lambda answer: (
                    -answer_scores[answer],
                    answer != reference.normalized_answer,
                    answer,
                ),
            )
            winner = ranked[0]
            winner_share = answer_scores[winner] / max(total, 1e-12)
            reference_score = answer_scores.get(str(reference.normalized_answer), 0.0)
            winner_advantage = (answer_scores[winner] - reference_score) / max(total, 1e-12)
            if (
                winner != reference.normalized_answer
                and winner_advantage + 1e-12 < variant.minimum_advantage
            ):
                chosen = reference
                fallback_reason = "consensus_advantage_below_source_frozen_threshold"
            elif winner_share + 1e-12 < variant.fallback_share:
                chosen = reference
                fallback_reason = "consensus_share_below_source_frozen_threshold"
            else:
                chosen = sorted(
                    voters[winner],
                    key=lambda item: (-item[2], item[0], item[1].selected_expert_id or ""),
                )[0][1]
                fallback_reason = None
        cluster_scores: dict[str, float] = {}
        for answer, score in sorted(answer_scores.items()):
            representative = sorted(
                voters[answer],
                key=lambda item: (-item[2], item[0], item[1].selected_expert_id or ""),
            )[0][1]
            if representative.selected_cluster_id is not None:
                cluster_scores[str(representative.selected_cluster_id)] = float(score)
        features = dict(chosen.observable_features)
        features.update(
            {
                "method": variant.name,
                "method_consensus": True,
                "consensus_subset": variant.subset,
                "consensus_global_weighting": variant.global_weighting,
                "consensus_confidence_power": variant.confidence_power,
                "consensus_fallback_share": variant.fallback_share,
                "consensus_minimum_advantage": variant.minimum_advantage,
                "consensus_winner_share": winner_share,
                "consensus_winner_advantage": winner_advantage,
                "consensus_method_count": len(methods),
                "consensus_external_method_weights": external_method_weights is not None,
                "consensus_uses_target_labels": False,
            }
        )
        selections.append(
            replace(
                chosen,
                cluster_scores=cluster_scores,
                fallback_reason=fallback_reason or chosen.fallback_reason,
                observable_features=features,
                tie_breaking=(
                    "weighted_query_local_answer_vote;reference_answer_on_tie;"
                    "contribution_then_method_then_expert;" + chosen.tie_breaking
                ),
            )
        )
    return selections


def apply_consensus_gate(
    ungated: Sequence[Selection],
    reference: Sequence[Selection],
    *,
    name: str,
    fallback_share: float,
    minimum_advantage: float,
) -> list[Selection]:
    reference_by_id = {row.question_id: row for row in reference}
    if {row.question_id for row in ungated} != set(reference_by_id):
        raise ValueError("Consensus gate/reference IDs are not aligned")
    result: list[Selection] = []
    for row in ungated:
        base = reference_by_id[row.question_id]
        share = float(row.observable_features["consensus_winner_share"])
        advantage = float(row.observable_features["consensus_winner_advantage"])
        switched = row.normalized_answer != base.normalized_answer
        reason: str | None = None
        if switched and advantage + 1e-12 < minimum_advantage:
            chosen = base
            reason = "consensus_advantage_below_frozen_threshold"
        elif switched and share + 1e-12 < fallback_share:
            chosen = base
            reason = "consensus_share_below_frozen_threshold"
        else:
            chosen = row
        features = dict(chosen.observable_features)
        features.update(
            {
                "method": name,
                "guarded_method_consensus": True,
                "consensus_gate_fallback_share": fallback_share,
                "consensus_gate_minimum_advantage": minimum_advantage,
                "consensus_ungated_winner_share": share,
                "consensus_ungated_winner_advantage": advantage,
                "consensus_gate_switched_from_reference": chosen.normalized_answer != base.normalized_answer,
                "consensus_gate_uses_target_labels": False,
            }
        )
        result.append(
            replace(
                chosen,
                fallback_reason=reason or chosen.fallback_reason,
                observable_features=features,
                tie_breaking="frozen_consensus_gate;" + chosen.tie_breaking,
            )
        )
    return result


def default_consensus_variants() -> tuple[ConsensusVariant, ...]:
    variants: list[ConsensusVariant] = []
    for subset in ("all", "core", "nonstochastic", "core_fcrg"):
        for weighting in ("equal", "signature", "family", "agreement", "signature_agreement"):
            for confidence_power in (0.0, 1.0):
                for fallback_share in (0.0, 0.4):
                    suffix = str(confidence_power).replace(".", "p")
                    fallback = str(fallback_share).replace(".", "p")
                    variants.append(
                        ConsensusVariant(
                            name=(
                                f"mcons__{subset}__{weighting}__c{suffix}__f{fallback}"
                            ),
                            subset=subset,
                            global_weighting=weighting,
                            confidence_power=confidence_power,
                            fallback_share=fallback_share,
                        )
                    )
    return tuple(variants)
