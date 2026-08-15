from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
from scipy.stats import binomtest

from .features import records_by_question
from .schema import EvaluationLabels, ObservableQueryBatch, Selection


def selection_correctness(selections: list[Selection], labels: EvaluationLabels) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for selection in selections:
        if selection.selected_expert_id is None:
            result[selection.question_id] = False
        else:
            result[selection.question_id] = bool(labels.get(selection.question_id, selection.selected_expert_id))
    return result


def paired_bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    seed: int = 20260808,
    samples: int = 1000,
) -> tuple[float, float]:
    if len(candidate) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(samples, len(candidate)))
    deltas = (candidate[indices] - baseline[indices]).mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def exact_mcnemar(candidate: np.ndarray, baseline: np.ndarray) -> tuple[int, int, float]:
    rescue = int(np.sum((candidate == 1) & (baseline == 0)))
    harm = int(np.sum((candidate == 0) & (baseline == 1)))
    discordant = rescue + harm
    p_value = float(binomtest(min(rescue, harm), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return rescue, harm, p_value


def paired_selection_comparison(
    name: str,
    candidate_selections: list[Selection],
    reference_selections: list[Selection],
    labels: EvaluationLabels,
    seed: int = 20260808,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    candidate_map = selection_correctness(candidate_selections, labels)
    reference_map = selection_correctness(reference_selections, labels)
    if set(candidate_map) != set(reference_map):
        raise ValueError(f"Paired comparison {name} has unaligned question IDs")
    ids = sorted(candidate_map)
    candidate = np.asarray([candidate_map[qid] for qid in ids], dtype=int)
    reference = np.asarray([reference_map[qid] for qid in ids], dtype=int)
    rescue, harm, p_value = exact_mcnemar(candidate, reference)
    ci_low, ci_high = paired_bootstrap_delta(candidate, reference, seed=seed, samples=bootstrap_samples)
    return {
        "comparison": name,
        "samples": len(ids),
        "candidate_accuracy": float(candidate.mean()) if len(candidate) else 0.0,
        "reference_accuracy": float(reference.mean()) if len(reference) else 0.0,
        "delta": float((candidate - reference).mean()) if len(candidate) else 0.0,
        "rescue_count": rescue,
        "harm_count": harm,
        "exact_mcnemar_p": p_value,
        "paired_bootstrap_delta_ci95": [ci_low, ci_high],
    }


def hierarchical_paired_bootstrap(
    candidate_by_seed: np.ndarray,
    reference_by_seed: np.ndarray,
    seed: int,
    samples: int = 10000,
) -> tuple[float, float]:
    """Crossed seed-by-query bootstrap with one shared query resample per draw."""
    if candidate_by_seed.shape != reference_by_seed.shape or candidate_by_seed.ndim != 2:
        raise ValueError("Hierarchical bootstrap requires aligned [seed, query] arrays")
    seed_count, query_count = candidate_by_seed.shape
    if seed_count == 0 or query_count == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        sampled_queries = rng.integers(0, query_count, size=query_count)
        delta_matrix = candidate_by_seed - reference_by_seed
        deltas[index] = float(delta_matrix[np.ix_(sampled_seeds, sampled_queries)].mean())
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    probs = np.asarray(list(counts.values()), dtype=float) / len(values)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def evaluate(
    method: str,
    selections: list[Selection],
    baseline: list[Selection],
    batch: ObservableQueryBatch,
    labels: EvaluationLabels,
    bootstrap_samples: int = 1000,
    seed: int = 20260808,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = tuple(sorted(batch.question_ids))
    candidate_map = {selection.question_id: selection for selection in selections}
    baseline_map = {selection.question_id: selection for selection in baseline}
    if set(candidate_map) != set(ids) or set(baseline_map) != set(ids):
        raise ValueError("Prediction/evaluation question IDs are not aligned")
    candidate_correct = selection_correctness(selections, labels)
    baseline_correct = selection_correctness(baseline, labels)
    candidate = np.asarray([candidate_correct[question_id] for question_id in ids], dtype=int)
    reference = np.asarray([baseline_correct[question_id] for question_id in ids], dtype=int)
    rescue, harm, p_value = exact_mcnemar(candidate, reference)
    ci_low, ci_high = paired_bootstrap_delta(candidate, reference, seed=seed, samples=bootstrap_samples)
    oracle = np.asarray(
        [any(bool(labels.get(question_id, expert)) for expert in batch.pool.expert_ids) for question_id in ids],
        dtype=int,
    )
    unchanged_correct = int(np.sum((candidate == 1) & (reference == 1)))
    unchanged_wrong = int(np.sum((candidate == 0) & (reference == 0)))
    switches = [candidate_map[qid].selected_expert_id != baseline_map[qid].selected_expert_id for qid in ids]
    selected_experts = [candidate_map[qid].selected_expert_id or "<none>" for qid in ids]
    selected_clusters = [str(candidate_map[qid].selected_cluster_id) for qid in ids]
    grouped = records_by_question(batch)
    subject_correct: dict[str, list[int]] = defaultdict(list)
    family_count: Counter[str] = Counter()
    missing_correct: dict[str, list[int]] = defaultdict(list)
    per_query: list[dict[str, Any]] = []
    for index, question_id in enumerate(ids):
        selection = candidate_map[question_id]
        rows = grouped[question_id]
        subject = rows[0].subject if rows else "UNKNOWN"
        subject_correct[subject].append(int(candidate[index]))
        family = batch.pool.family_by_expert.get(selection.selected_expert_id or "", "<none>")
        family_count[family] += 1
        has_missing = any(not record.valid_output for record in rows)
        missing_correct["has_missing" if has_missing else "complete"].append(int(candidate[index]))
        per_query.append(
            {
                "question_id": question_id,
                "method": method,
                "selected_answer_cluster": selection.selected_cluster_id,
                "selected_expert": selection.selected_expert_id,
                "normalized_answer": selection.normalized_answer,
                "correct": bool(candidate[index]),
                "baseline_correct": bool(reference[index]),
                "rescued": bool(candidate[index] and not reference[index]),
                "harmed": bool(reference[index] and not candidate[index]),
                "cluster_scores": dict(selection.cluster_scores),
                "expert_scores": dict(selection.expert_scores),
                "fallback_reason": selection.fallback_reason,
                "observable_features": dict(selection.observable_features),
                "tie_breaking": selection.tie_breaking,
            }
        )
    accuracy = float(candidate.mean())
    baseline_accuracy = float(reference.mean())
    oracle_accuracy = float(oracle.mean())
    oracle_denominator = oracle_accuracy - baseline_accuracy
    summary: dict[str, Any] = {
        "method": method,
        "samples": len(ids),
        "accuracy": accuracy,
        "source_best_accuracy": baseline_accuracy,
        "delta_vs_source_best_single": accuracy - baseline_accuracy,
        "rescue_count": rescue,
        "rescue_rate": rescue / max(1, len(ids)),
        "harm_count": harm,
        "harm_rate": harm / max(1, len(ids)),
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "switch_count": int(sum(switches)),
        "switch_rate": float(np.mean(switches)),
        "switch_precision": rescue / max(1, rescue + harm),
        "net_uplift": (rescue - harm) / max(1, len(ids)),
        "oracle_accuracy": oracle_accuracy,
        "oracle_gap_closed": (accuracy - baseline_accuracy) / oracle_denominator if abs(oracle_denominator) > 1e-12 else None,
        "paired_bootstrap_delta_ci95": [ci_low, ci_high],
        "exact_mcnemar_p": p_value,
        "per_subject_accuracy": {key: float(np.mean(value)) for key, value in sorted(subject_correct.items())},
        "per_family_selection_rate": {key: value / len(ids) for key, value in sorted(family_count.items())},
        "expert_selection_entropy": entropy(selected_experts),
        "answer_cluster_selection_entropy": entropy(selected_clusters),
        "missing_output_impact": {key: float(np.mean(value)) for key, value in sorted(missing_correct.items()) if value},
    }
    return summary, per_query


def holm_adjust(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    still_rejecting = True
    for index, (name, p_value) in enumerate(ordered):
        adjusted_p = min(1.0, max(running, (total - index) * p_value))
        running = adjusted_p
        threshold = alpha / (total - index)
        rejected = still_rejecting and p_value <= threshold
        still_rejecting = still_rejecting and rejected
        adjusted[name] = {"raw_p": p_value, "holm_adjusted_p": adjusted_p, "reject": rejected}
    return adjusted
