from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .conservative_cpi import ThresholdCalibration, _aligned
from .cpi import (
    INTERVENTIONS,
    FingerprintTable,
    InvariantClusterScorer,
    PoolExample,
    _masked_max,
    _masked_mean,
    _stable_rng,
    apply_intervention,
    apply_known_swap,
    canonicalize_exact_clones,
    collate_pool_examples,
    make_pool_example,
    selections_from_scores,
)
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


@dataclass(frozen=True)
class CategoricalScores:
    question_id: str
    cluster_logits: Mapping[int, float]
    none_logit: float


class CategoricalClusterScorer(InvariantClusterScorer):
    def __init__(self, input_dim: int, hidden_dim: int = 48, linear: bool = False) -> None:
        super().__init__(input_dim, hidden_dim=hidden_dim, linear=linear)
        none_input_dim = 2 * self.hidden_dim + 8
        if linear:
            self.none_head: nn.Module = nn.Linear(none_input_dim, 1)
        else:
            self.none_head = nn.Sequential(
                nn.Linear(none_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, batch) -> torch.Tensor:
        encoded = self.phi(batch.expert_features)
        global_mean = _masked_mean(encoded, batch.expert_mask, dim=1)
        global_max = _masked_max(encoded, batch.expert_mask, dim=1)
        cluster_count = batch.cluster_mask.shape[1]
        cluster_indices = torch.arange(cluster_count, device=encoded.device)[None, :, None]
        member_mask = (batch.cluster_assignment[:, None, :] == cluster_indices) & batch.expert_mask[:, None, :]
        expanded = encoded[:, None, :, :].expand(-1, cluster_count, -1, -1)
        cluster_mean = _masked_mean(expanded, member_mask, dim=2)
        cluster_max = _masked_max(expanded, member_mask, dim=2)
        repeated_global_mean = global_mean[:, None, :].expand(-1, cluster_count, -1)
        repeated_global_max = global_max[:, None, :].expand(-1, cluster_count, -1)
        cluster_summary = torch.cat(
            [cluster_mean, cluster_max, repeated_global_mean, repeated_global_max, batch.cluster_extra],
            dim=-1,
        )
        cluster_logits = self.rho(cluster_summary).squeeze(-1).masked_fill(~batch.cluster_mask, -1e9)
        extra_mean = _masked_mean(batch.cluster_extra, batch.cluster_mask, dim=1)
        extra_max = _masked_max(batch.cluster_extra, batch.cluster_mask, dim=1)
        none_summary = torch.cat([global_mean, global_max, extra_mean, extra_max], dim=-1)
        none_logit = self.none_head(none_summary)
        return torch.cat([cluster_logits, none_logit], dim=1)


def categorical_target(example: PoolExample) -> int | None:
    clusters = sorted({int(value) for value in example.cluster_ids if value >= 0})
    missing = set(clusters).difference(example.cluster_labels)
    if missing:
        raise ValueError(f"Categorical CPI is missing labels for clusters: {sorted(missing)}")
    correct = [cluster for cluster in clusters if float(example.cluster_labels[cluster]) >= 0.5]
    if len(correct) > 1:
        raise ValueError("Categorical CPI requires at most one correct answer cluster")
    return correct[0] if correct else None


def train_categorical_scorer(
    examples: Sequence[PoolExample],
    input_dim: int,
    device: torch.device | str,
    seed: int,
    variant: str,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    hidden_dim: int = 48,
    linear: bool = False,
    replacement_examples: Mapping[str, PoolExample] | None = None,
    swap_mapping: Mapping[str, str] | None = None,
) -> tuple[CategoricalClusterScorer, list[float]]:
    if variant not in {"none", "full", *INTERVENTIONS}:
        raise ValueError(f"Unknown categorical CPI training variant: {variant}")
    for example in examples:
        categorical_target(example)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = CategoricalClusterScorer(input_dim, hidden_dim=hidden_dim, linear=linear).to(device)
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buffer)
    model.initialization_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()  # type: ignore[attr-defined]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for epoch in range(epochs):
        augmented: list[PoolExample] = []
        for example in examples:
            augmented.append(example)
            if variant == "none":
                augmented.append(example)
            else:
                intervention = (
                    INTERVENTIONS[
                        (
                            epoch
                            + int.from_bytes(hashlib.sha256(example.question_id.encode()).digest()[:2], "little")
                        )
                        % len(INTERVENTIONS)
                    ]
                    if variant == "full"
                    else variant
                )
                rng = _stable_rng(seed, epoch, example.question_id, intervention)
                if intervention == "known_swap":
                    if replacement_examples is None or swap_mapping is None:
                        raise ValueError("known_swap training requires real replacement examples and a frozen mapping")
                    augmented.append(apply_known_swap(example, replacement_examples[example.question_id], swap_mapping, rng))
                else:
                    augmented.append(apply_intervention(example, intervention, rng))
        order = _stable_rng(seed, epoch, "categorical-order").permutation(len(augmented))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch_size):
            selected = [augmented[int(index)] for index in order[start : start + batch_size]]
            selected = [example for example in selected if np.any(example.valid)]
            if not selected:
                continue
            for example in selected:
                categorical_target(example)
            batch = collate_pool_examples(selected, device)
            logits = model(batch)
            correct = (batch.targets >= 0.5) & batch.cluster_mask
            correct_count = correct.sum(dim=1)
            if bool(torch.any(correct_count > 1)):
                raise ValueError("Categorical CPI batch contains multiple correct clusters")
            target = torch.where(
                correct_count == 1,
                correct.to(torch.int64).argmax(dim=1),
                torch.full_like(correct_count, logits.shape[1] - 1),
            )
            loss = functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return model, history


@torch.no_grad()
def score_categorical_logits(
    model: CategoricalClusterScorer,
    examples: Sequence[PoolExample],
    device: torch.device | str,
    batch_size: int = 256,
) -> list[CategoricalScores]:
    model.eval()
    result: list[CategoricalScores] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate_pool_examples(chunk, device)
        logits = model(batch).cpu().numpy()
        none_column = logits.shape[1] - 1
        for row, (example, cluster_values) in enumerate(zip(chunk, batch.cluster_values)):
            result.append(
                CategoricalScores(
                    question_id=example.question_id,
                    cluster_logits={
                        cluster: float(logits[row, index]) for index, cluster in enumerate(cluster_values)
                    },
                    none_logit=float(logits[row, none_column]),
                )
            )
    return result


def scaled_probabilities(scores: CategoricalScores, temperature: float) -> tuple[dict[int, float], float]:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Categorical CPI temperature must be positive and finite")
    clusters = sorted(scores.cluster_logits)
    values = np.asarray([scores.cluster_logits[cluster] for cluster in clusters] + [scores.none_logit], dtype=float)
    values = values / temperature
    values -= np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return {cluster: float(probabilities[index]) for index, cluster in enumerate(clusters)}, float(probabilities[-1])


def categorical_selections(
    batch: ObservableQueryBatch,
    examples: Sequence[PoolExample],
    outputs: Sequence[CategoricalScores],
    temperature: float,
    method: str,
) -> list[Selection]:
    if len(examples) != len(outputs) or any(
        example.question_id != output.question_id for example, output in zip(examples, outputs)
    ):
        raise ValueError("Categorical CPI examples and scores are not aligned")
    score_rows: list[dict[int, float]] = []
    none_by_question: dict[str, float] = {}
    for output in outputs:
        cluster_scores, none_probability = scaled_probabilities(output, temperature)
        score_rows.append(cluster_scores)
        none_by_question[output.question_id] = none_probability
    selections = selections_from_scores(batch, examples, score_rows, method)
    return [
        replace(
            selection,
            observable_features={
                **dict(selection.observable_features),
                "temperature": temperature,
                "none_correct_probability": none_by_question[selection.question_id],
            },
        )
        for selection in selections
    ]


def none_fallback_selections(
    candidate: Sequence[Selection],
    baseline: Sequence[Selection],
    method: str = "cpi_ce_none_fallback",
) -> list[Selection]:
    ids, candidate_by_id, baseline_by_id = _aligned(candidate, baseline)
    result: list[Selection] = []
    for question_id in ids:
        proposal = candidate_by_id[question_id]
        reference = baseline_by_id[question_id]
        proposal_probability = float(proposal.cluster_scores.get(str(proposal.selected_cluster_id), 0.0))
        none_probability = float(proposal.observable_features.get("none_correct_probability", 0.0))
        none_wins = none_probability >= proposal_probability
        features = {
            **dict(proposal.observable_features),
            "method": method,
            "none_wins": none_wins,
            "source_best_cluster_id": reference.selected_cluster_id,
            "source_best_expert_id": reference.selected_expert_id,
        }
        if not none_wins:
            result.append(replace(proposal, observable_features=features))
        else:
            result.append(
                Selection(
                    question_id=question_id,
                    selected_cluster_id=reference.selected_cluster_id,
                    selected_expert_id=reference.selected_expert_id,
                    normalized_answer=reference.normalized_answer,
                    cluster_scores=dict(proposal.cluster_scores),
                    expert_scores=dict(reference.expert_scores),
                    fallback_reason="none_correct_wins",
                    observable_features=features,
                    tie_breaking=reference.tie_breaking,
                )
            )
    return result


def none_aware_margin(candidate: Selection, baseline: Selection) -> float:
    if candidate.question_id != baseline.question_id:
        raise ValueError("Cannot compare selections from different questions")
    if candidate.selected_cluster_id is None or baseline.selected_cluster_id is None:
        return -1.0
    proposal = candidate.cluster_scores.get(str(candidate.selected_cluster_id))
    source_best = candidate.cluster_scores.get(str(baseline.selected_cluster_id))
    if proposal is None or source_best is None:
        return -1.0
    none_probability = float(candidate.observable_features.get("none_correct_probability", 0.0))
    return float(proposal) - max(float(source_best), none_probability)


def apply_categorical_gate(
    candidate: Sequence[Selection],
    baseline: Sequence[Selection],
    threshold: float,
    method: str = "cpi_ce_calibrated",
) -> list[Selection]:
    ids, candidate_by_id, baseline_by_id = _aligned(candidate, baseline)
    result: list[Selection] = []
    for question_id in ids:
        proposal = candidate_by_id[question_id]
        reference = baseline_by_id[question_id]
        same_cluster = proposal.selected_cluster_id == reference.selected_cluster_id
        margin = none_aware_margin(proposal, reference)
        accept = not same_cluster and margin >= threshold
        features: dict[str, Any] = {
            **dict(proposal.observable_features),
            "method": method,
            "none_aware_margin": margin,
            "calibrated_threshold": threshold,
            "proposal_cluster_id": proposal.selected_cluster_id,
            "proposal_expert_id": proposal.selected_expert_id,
            "source_best_cluster_id": reference.selected_cluster_id,
            "source_best_expert_id": reference.selected_expert_id,
            "proposal_accepted": accept,
        }
        if accept:
            result.append(replace(proposal, fallback_reason=None, observable_features=features))
        else:
            reason = "same_cluster_source_best" if same_cluster else "none_or_margin_fallback"
            result.append(
                Selection(
                    question_id=question_id,
                    selected_cluster_id=reference.selected_cluster_id,
                    selected_expert_id=reference.selected_expert_id,
                    normalized_answer=reference.normalized_answer,
                    cluster_scores=dict(proposal.cluster_scores),
                    expert_scores=dict(reference.expert_scores),
                    fallback_reason=reason,
                    observable_features=features,
                    tie_breaking=reference.tie_breaking,
                )
            )
    return result


def temperature_nll(
    outputs: Sequence[CategoricalScores],
    examples: Sequence[PoolExample],
    labels: SourceTrainingLabels,
    temperature: float,
) -> float:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("CPI-CE temperature calibration requires source training labels")
    examples_by_id = {example.question_id: example for example in examples}
    if set(examples_by_id) != {output.question_id for output in outputs}:
        raise ValueError("CPI-CE calibration examples and scores are not aligned")
    losses: list[float] = []
    for output in outputs:
        target = categorical_target(examples_by_id[output.question_id])
        cluster_probabilities, none_probability = scaled_probabilities(output, temperature)
        probability = none_probability if target is None else cluster_probabilities[target]
        losses.append(-float(np.log(max(probability, 1e-12))))
    return float(np.mean(losses)) if losses else 0.0


def calibrate_none_aware_threshold(
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
        raise TypeError("CPI-CE thresholds require source training labels")
    ids, _, baseline_by_id = _aligned(candidate, baseline)
    if set(ids).difference(environment_by_question):
        raise ValueError("CPI-CE calibration is missing source environments")
    grid = sorted(set(float(value) for value in thresholds))
    if not grid or grid[-1] <= 1.0:
        raise ValueError("CPI-CE threshold grid must include a no-switch fallback above 1.0")
    baseline_correct = {
        question_id: float(bool(labels.get(question_id, baseline_by_id[question_id].selected_expert_id or "")))
        for question_id in ids
    }
    diagnostics: list[ThresholdCalibration] = []
    for threshold in grid:
        gated = apply_categorical_gate(candidate, baseline, threshold)
        environment_deltas: dict[str, list[float]] = {}
        deltas: list[float] = []
        switches = 0
        for selection in gated:
            question_id = selection.question_id
            delta = (
                float(bool(labels.get(question_id, selection.selected_expert_id or "")))
                - baseline_correct[question_id]
            )
            deltas.append(delta)
            environment_deltas.setdefault(str(environment_by_question[question_id]), []).append(delta)
            switches += int(selection.selected_cluster_id != baseline_by_id[question_id].selected_cluster_id)
        by_environment = [float(np.mean(values)) for values in environment_deltas.values()]
        macro = float(np.mean(by_environment)) if by_environment else 0.0
        micro = float(np.mean(deltas)) if deltas else 0.0
        worst = min(by_environment, default=0.0)
        nonnegative = float(np.mean([value >= 0.0 for value in by_environment])) if by_environment else 1.0
        feasible = worst >= min_worst_delta and micro >= min_micro_delta
        utility = macro + worst_weight * worst if feasible else float("-inf")
        diagnostics.append(
            ThresholdCalibration(
                threshold=threshold,
                macro_delta=macro,
                micro_delta=micro,
                worst_environment_delta=worst,
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
        raise AssertionError("No feasible CPI-CE fallback threshold was found")
    return selected.threshold, diagnostics


def calibrate_temperature_and_threshold(
    batch: ObservableQueryBatch,
    outputs: Sequence[CategoricalScores],
    examples: Sequence[PoolExample],
    baseline: Sequence[Selection],
    labels: SourceTrainingLabels,
    environment_by_question: Mapping[str, str],
    temperatures: Sequence[float],
    thresholds: Sequence[float],
    min_worst_delta: float,
    min_micro_delta: float,
    worst_weight: float,
) -> tuple[float, float, list[dict[str, float | bool]], list[ThresholdCalibration]]:
    temperature_rows = [
        {
            "temperature": float(temperature),
            "nll": temperature_nll(outputs, examples, labels, float(temperature)),
            "selected": False,
        }
        for temperature in sorted(set(float(value) for value in temperatures))
    ]
    if not temperature_rows:
        raise ValueError("CPI-CE temperature grid is empty")
    selected_temperature = min(
        temperature_rows,
        key=lambda row: (float(row["nll"]), abs(float(np.log(float(row["temperature"])))), float(row["temperature"])),
    )
    selected_temperature["selected"] = True
    temperature = float(selected_temperature["temperature"])
    candidate = categorical_selections(batch, examples, outputs, temperature, "inner_oof_cpi_ce")
    threshold, diagnostics = calibrate_none_aware_threshold(
        candidate,
        baseline,
        labels,
        environment_by_question,
        thresholds,
        min_worst_delta,
        min_micro_delta,
        worst_weight,
    )
    return temperature, threshold, temperature_rows, diagnostics


def predict_categorical(
    model: CategoricalClusterScorer,
    batch: ObservableQueryBatch,
    fingerprints: FingerprintTable,
    device: torch.device | str,
    temperature: float,
    method: str,
) -> tuple[list[Selection], list[CategoricalScores], list[PoolExample]]:
    examples = [make_pool_example(batch, question_id, fingerprints) for question_id in batch.question_ids]
    outputs = score_categorical_logits(model, examples, device)
    return categorical_selections(batch, examples, outputs, temperature, method), outputs, examples


def max_categorical_probability_difference(
    first: Sequence[CategoricalScores],
    second: Sequence[CategoricalScores],
    temperature: float = 1.0,
) -> float:
    if len(first) != len(second):
        raise ValueError("Categorical invariance predictions have different sample counts")
    differences: list[float] = []
    for left, right in zip(first, second):
        left_clusters, left_none = scaled_probabilities(left, temperature)
        right_clusters, right_none = scaled_probabilities(right, temperature)
        if left.question_id != right.question_id or set(left_clusters) != set(right_clusters):
            raise ValueError("Categorical invariance predictions are not aligned")
        differences.append(abs(left_none - right_none))
        differences.extend(abs(left_clusters[key] - right_clusters[key]) for key in left_clusters)
    return max(differences, default=0.0)
