from __future__ import annotations

import hashlib
import io
import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .cpi import (
    INTERVENTIONS,
    CollatedPoolBatch,
    FingerprintTable,
    PoolExample,
    _masked_max,
    _masked_mean,
    _stable_rng,
    apply_intervention,
    apply_known_swap,
    canonicalize_exact_clones,
    collate_pool_examples,
)
from .cpi_ce import CategoricalScores, categorical_selections, categorical_target
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


@dataclass(frozen=True)
class RemainingVariant:
    name: str
    training_variant: str
    fingerprint_mode: str
    cluster_features: str
    objective: str


INTERVENTION_VARIANTS = tuple(
    RemainingVariant(f"int_{name}", name, "legacy", "legacy", "mean")
    for name in ("none", *INTERVENTIONS, "full")
)

FACTORIAL_VARIANTS = (
    RemainingVariant("factor_legacy_dro", "full", "legacy", "legacy", "dro"),
    RemainingVariant("factor_mask_mean", "full", "mask", "legacy", "mean"),
    RemainingVariant("factor_mask_dro", "full", "mask", "legacy", "dro"),
    RemainingVariant("factor_rich_mean", "full", "legacy", "rich", "mean"),
    RemainingVariant("factor_rich_dro", "full", "legacy", "rich", "dro"),
    RemainingVariant("factor_rich_mask_mean", "full", "mask", "rich", "mean"),
    RemainingVariant("factor_rich_mask_dro", "full", "mask", "rich", "dro"),
)

FITTED_VARIANTS = (*INTERVENTION_VARIANTS, *FACTORIAL_VARIANTS)
ALL_VARIANT_NAMES = tuple(variant.name for variant in FITTED_VARIANTS) + ("factor_legacy_mean",)
METHODS = tuple(f"{variant}__{suffix}" for variant in ALL_VARIANT_NAMES for suffix in ("raw", "none_fallback"))
PRIMARY_METHOD = "factor_rich_mask_dro__none_fallback"
RICH_CLUSTER_FEATURE_NAMES = (
    "member_share_valid",
    "family_breadth_fraction",
    "uncertainty_mean",
    "uncertainty_max",
    "uncertainty_std",
    "source_accuracy_mean",
    "source_accuracy_max",
    "source_accuracy_min",
    "source_accuracy_std",
    "family_entropy",
    "fingerprint_dispersion",
    "member_share_total_pool",
    "valid_fraction",
    "missing_fraction",
    "singleton",
    "plurality_margin",
)


def fit_masked_source_fingerprints(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    rank: int = 4,
    extra_batch: ObservableQueryBatch | None = None,
    extra_labels: SourceTrainingLabels | None = None,
) -> FingerprintTable:
    """Fit fingerprints without treating invalid outputs as observed errors."""
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Mask-aware fingerprints require SourceTrainingLabels")
    if (labels.dataset, labels.split) != (batch.dataset, batch.split):
        raise ValueError("Source label provenance does not match the observable batch")
    question_ids = batch.question_ids
    expert_ids = batch.pool.expert_ids
    records = {(row.question_id, row.expert_id): row for row in batch.records}
    observed = np.asarray(
        [
            [
                bool(records[(question_id, expert)].valid_output)
                and labels.get(question_id, expert) is not None
                for question_id in question_ids
            ]
            for expert in expert_ids
        ],
        dtype=bool,
    )
    correctness = np.asarray(
        [
            [float(bool(labels.get(question_id, expert))) for question_id in question_ids]
            for expert in expert_ids
        ],
        dtype=np.float64,
    )
    observed_count = observed.sum(axis=0)
    item_mean = np.divide(
        (correctness * observed).sum(axis=0),
        observed_count,
        out=np.full(len(question_ids), 0.5, dtype=np.float64),
        where=observed_count > 0,
    )
    centered = np.where(observed, correctness - item_mean[None, :], 0.0)
    use_rank = min(rank, centered.shape[0], centered.shape[1])
    if use_rank:
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        embedding = u[:, :use_rank] * singular[:use_rank] / max(1.0, np.sqrt(centered.shape[1]))
        for column in range(use_rank):
            anchor = int(np.argmax(np.abs(embedding[:, column])))
            if embedding[anchor, column] < 0:
                embedding[:, column] *= -1.0
    else:
        embedding = np.zeros((len(expert_ids), 0), dtype=np.float64)
        vt = np.zeros((0, centered.shape[1]), dtype=np.float64)
    if use_rank < rank:
        embedding = np.pad(embedding, ((0, 0), (0, rank - use_rank)))

    all_experts = list(expert_ids)
    correctness_by_expert = {expert: correctness[index] for index, expert in enumerate(expert_ids)}
    observed_by_expert = {expert: observed[index] for index, expert in enumerate(expert_ids)}
    embedding_by_expert = {expert: embedding[index] for index, expert in enumerate(expert_ids)}
    family_by_expert = dict(batch.pool.family_by_expert)
    batches = [batch]
    if extra_batch is not None:
        if extra_labels is None or (extra_labels.dataset, extra_labels.split) != (batch.dataset, batch.split):
            raise ValueError("Real swap experts require aligned source labels")
        if extra_batch.question_ids != question_ids:
            raise ValueError("Real swap expert rows do not align with source questions")
        batches.append(extra_batch)
        extra_records = {(row.question_id, row.expert_id): row for row in extra_batch.records}
        for expert in extra_batch.pool.expert_ids:
            row_observed = np.asarray(
                [
                    bool(extra_records[(question_id, expert)].valid_output)
                    and extra_labels.get(question_id, expert) is not None
                    for question_id in question_ids
                ],
                dtype=bool,
            )
            row_correct = np.asarray(
                [float(bool(extra_labels.get(question_id, expert))) for question_id in question_ids],
                dtype=np.float64,
            )
            projected = np.where(row_observed, row_correct - item_mean, 0.0) @ vt[:use_rank].T
            projected /= max(1.0, np.sqrt(len(question_ids)))
            if use_rank < rank:
                projected = np.pad(projected, (0, rank - use_rank))
            correctness_by_expert[expert] = row_correct
            observed_by_expert[expert] = row_observed
            embedding_by_expert[expert] = projected
            if expert not in all_experts:
                all_experts.append(expert)
            family_by_expert[expert] = extra_batch.pool.family_by_expert[expert]

    family_names = tuple(sorted(set(family_by_expert.values())))
    feature_names = (
        "source_observed_accuracy",
        "source_valid_rate",
        *(f"masked_correctness_svd_{index}" for index in range(rank)),
        *(f"family::{family}" for family in family_names),
        "family::<unknown>",
        "log_cost",
        "cost_missing",
    )
    grouped_costs: dict[str, list[float]] = {expert: [] for expert in all_experts}
    valid_by_expert: dict[str, list[float]] = {expert: [] for expert in all_experts}
    for current_batch in batches:
        for record in current_batch.records:
            valid_by_expert[record.expert_id].append(float(record.valid_output))
            if record.inference_cost is not None and np.isfinite(record.inference_cost):
                grouped_costs[record.expert_id].append(float(record.inference_cost))
    all_costs = [cost for costs in grouped_costs.values() for cost in costs]
    fallback_cost = float(np.median(all_costs)) if all_costs else 1.0
    values: dict[str, tuple[float, ...]] = {}
    for expert in all_experts:
        family = family_by_expert[expert]
        family_vector = [float(family == candidate) for candidate in family_names] + [0.0]
        mask = observed_by_expert[expert]
        correct = correctness_by_expert[expert]
        accuracy = (float(correct[mask].sum()) + 1.0) / (float(mask.sum()) + 2.0)
        validity = valid_by_expert[expert]
        valid_rate = (sum(validity) + 1.0) / (len(validity) + 2.0)
        costs = grouped_costs[expert]
        cost = float(np.median(costs)) if costs else fallback_cost
        values[expert] = tuple(
            float(value)
            for value in (
                accuracy,
                valid_rate,
                *embedding_by_expert[expert].tolist(),
                *family_vector,
                np.log1p(max(0.0, cost)),
                float(not costs),
            )
        )
    return FingerprintTable(tuple(all_experts), family_names, values, tuple(feature_names))


def collate_rich_pool_examples(
    examples: Sequence[PoolExample],
    device: torch.device | str,
) -> CollatedPoolBatch:
    canonical = [canonicalize_exact_clones(example) for example in examples]
    if not canonical:
        raise ValueError("Cannot collate an empty remaining-source batch")
    feature_dim = canonical[0].fingerprints.shape[1] + 2
    max_experts = max(len(example.expert_ids) for example in canonical)
    cluster_values = [tuple(sorted({int(value) for value in example.cluster_ids if value >= 0})) for example in canonical]
    max_clusters = max(1, max(len(values) for values in cluster_values))
    features = np.zeros((len(canonical), max_experts, feature_dim), dtype=np.float32)
    expert_mask = np.zeros((len(canonical), max_experts), dtype=bool)
    assignment = np.full((len(canonical), max_experts), -1, dtype=np.int64)
    cluster_mask = np.zeros((len(canonical), max_clusters), dtype=bool)
    extras = np.zeros((len(canonical), max_clusters, len(RICH_CLUSTER_FEATURE_NAMES)), dtype=np.float32)
    targets = np.zeros((len(canonical), max_clusters), dtype=np.float32)
    for row, (example, values) in enumerate(zip(canonical, cluster_values)):
        token_features = np.concatenate(
            [
                example.fingerprints.astype(np.float32),
                example.valid.astype(np.float32)[:, None],
                example.uncertainties.astype(np.float32)[:, None],
            ],
            axis=1,
        )
        features[row, : len(example.expert_ids)] = token_features
        expert_mask[row, : len(example.expert_ids)] = True
        position = {cluster: index for index, cluster in enumerate(values)}
        for expert_index, cluster in enumerate(example.cluster_ids):
            if int(cluster) >= 0:
                assignment[row, expert_index] = position[int(cluster)]
        cluster_mask[row, : len(values)] = True
        valid_count = max(1, int(example.valid.sum()))
        total_count = max(1, len(example.expert_ids))
        pool_families = {family for family, valid in zip(example.family_ids, example.valid) if valid}
        family_denominator = max(1, len(pool_families))
        shares = {cluster: int(np.sum((example.cluster_ids == cluster) & example.valid)) / valid_count for cluster in values}
        for column, cluster in enumerate(values):
            members = np.flatnonzero((example.cluster_ids == cluster) & example.valid)
            member_families = [example.family_ids[index] for index in members]
            family_counts = np.asarray(list(Counter(member_families).values()), dtype=float)
            family_probabilities = family_counts / max(1.0, family_counts.sum())
            family_entropy = -float(np.sum(family_probabilities * np.log(family_probabilities + 1e-12)))
            if family_denominator > 1:
                family_entropy /= math.log(family_denominator)
            uncertainty = example.uncertainties[members]
            fingerprints = example.fingerprints[members]
            source_accuracy = fingerprints[:, 0]
            other_share = max((share for key, share in shares.items() if key != cluster), default=0.0)
            extras[row, column] = (
                len(members) / valid_count,
                len(set(member_families)) / family_denominator,
                float(uncertainty.mean()),
                float(uncertainty.max()),
                float(uncertainty.std()),
                float(source_accuracy.mean()),
                float(source_accuracy.max()),
                float(source_accuracy.min()),
                float(source_accuracy.std()),
                family_entropy,
                float(np.std(fingerprints, axis=0).mean()),
                len(members) / total_count,
                float(example.valid.mean()),
                1.0 - float(example.valid.mean()),
                float(len(members) == 1),
                shares[cluster] - other_share,
            )
            targets[row, column] = float(example.cluster_labels.get(cluster, 0.0))
    return CollatedPoolBatch(
        expert_features=torch.as_tensor(features, device=device),
        expert_mask=torch.as_tensor(expert_mask, device=device),
        cluster_assignment=torch.as_tensor(assignment, device=device),
        cluster_mask=torch.as_tensor(cluster_mask, device=device),
        cluster_extra=torch.as_tensor(extras, device=device),
        targets=torch.as_tensor(targets, device=device),
        cluster_values=cluster_values,
    )


class RemainingCategoricalScorer(nn.Module):
    def __init__(self, input_dim: int, extra_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_dim
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(4 * hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: CollatedPoolBatch) -> torch.Tensor:
        encoded = self.phi(batch.expert_features)
        global_mean = _masked_mean(encoded, batch.expert_mask, dim=1)
        global_max = _masked_max(encoded, batch.expert_mask, dim=1)
        cluster_count = batch.cluster_mask.shape[1]
        cluster_indices = torch.arange(cluster_count, device=encoded.device)[None, :, None]
        member_mask = (batch.cluster_assignment[:, None, :] == cluster_indices) & batch.expert_mask[:, None, :]
        expanded = encoded[:, None, :, :].expand(-1, cluster_count, -1, -1)
        cluster_mean = _masked_mean(expanded, member_mask, dim=2)
        cluster_max = _masked_max(expanded, member_mask, dim=2)
        summary = torch.cat(
            [
                cluster_mean,
                cluster_max,
                global_mean[:, None, :].expand(-1, cluster_count, -1),
                global_max[:, None, :].expand(-1, cluster_count, -1),
                batch.cluster_extra,
            ],
            dim=-1,
        )
        cluster_logits = self.rho(summary).squeeze(-1).masked_fill(~batch.cluster_mask, -1e9)
        none_summary = torch.cat(
            [
                global_mean,
                global_max,
                _masked_mean(batch.cluster_extra, batch.cluster_mask, dim=1),
                _masked_max(batch.cluster_extra, batch.cluster_mask, dim=1),
            ],
            dim=-1,
        )
        return torch.cat([cluster_logits, self.none_head(none_summary)], dim=1)


def smooth_subject_dro_loss(
    losses: torch.Tensor,
    environments: Sequence[str],
    alpha: float,
    tau: float,
) -> torch.Tensor:
    if losses.ndim != 1 or len(losses) != len(environments):
        raise ValueError("Subject-DRO losses and environments are not aligned")
    if not 0.0 <= alpha <= 1.0 or tau <= 0.0:
        raise ValueError("Subject-DRO requires alpha in [0,1] and positive tau")
    groups = sorted(set(environments))
    group_losses = torch.stack(
        [losses[torch.as_tensor([value == group for value in environments], device=losses.device)].mean() for group in groups]
    )
    smooth_worst = tau * (torch.logsumexp(group_losses / tau, dim=0) - math.log(len(groups)))
    return (1.0 - alpha) * losses.mean() + alpha * smooth_worst


def _collate(examples: Sequence[PoolExample], device: torch.device | str, mode: str) -> CollatedPoolBatch:
    if mode == "legacy":
        return collate_pool_examples(examples, device)
    if mode == "rich":
        return collate_rich_pool_examples(examples, device)
    raise ValueError(f"Unknown cluster feature mode: {mode}")


def train_remaining_scorer(
    examples: Sequence[PoolExample],
    environments: Mapping[str, str],
    input_dim: int,
    device: torch.device | str,
    seed: int,
    variant: RemainingVariant,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    dro_alpha: float,
    dro_tau: float,
    replacement_examples: Mapping[str, PoolExample] | None = None,
    swap_mapping: Mapping[str, str] | None = None,
) -> tuple[RemainingCategoricalScorer, list[float]]:
    if variant.training_variant not in {"none", "full", *INTERVENTIONS}:
        raise ValueError(f"Unknown remaining-source intervention: {variant.training_variant}")
    if variant.objective not in {"mean", "dro"}:
        raise ValueError(f"Unknown remaining-source objective: {variant.objective}")
    if set(example.question_id for example in examples).difference(environments):
        raise ValueError("Remaining-source training examples lack environments")
    for example in examples:
        categorical_target(example)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    extra_dim = 4 if variant.cluster_features == "legacy" else len(RICH_CLUSTER_FEATURE_NAMES)
    model = RemainingCategoricalScorer(input_dim, extra_dim, hidden_dim=hidden_dim).to(device)
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buffer)
    model.initialization_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()  # type: ignore[attr-defined]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for epoch in range(epochs):
        augmented: list[tuple[PoolExample, str]] = []
        for example in examples:
            environment = str(environments[example.question_id])
            augmented.append((example, environment))
            if variant.training_variant == "none":
                augmented.append((example, environment))
                continue
            intervention = variant.training_variant
            if intervention == "full":
                offset = int.from_bytes(hashlib.sha256(example.question_id.encode()).digest()[:2], "little")
                intervention = INTERVENTIONS[(epoch + offset) % len(INTERVENTIONS)]
            rng = _stable_rng(seed, epoch, example.question_id, intervention)
            if intervention == "known_swap":
                if replacement_examples is None or swap_mapping is None:
                    raise ValueError("Known-swap training requires real replacement examples")
                changed = apply_known_swap(example, replacement_examples[example.question_id], swap_mapping, rng)
            else:
                changed = apply_intervention(example, intervention, rng)
            augmented.append((changed, environment))
        order = _stable_rng(seed, epoch, "remaining-order").permutation(len(augmented))
        epoch_losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch_size):
            rows = [augmented[int(index)] for index in order[start : start + batch_size]]
            rows = [(example, environment) for example, environment in rows if np.any(example.valid)]
            if not rows:
                continue
            selected = [row[0] for row in rows]
            batch = _collate(selected, device, variant.cluster_features)
            logits = model(batch)
            correct = (batch.targets >= 0.5) & batch.cluster_mask
            correct_count = correct.sum(dim=1)
            if bool(torch.any(correct_count > 1)):
                raise ValueError("Remaining-source batch contains multiple correct clusters")
            target = torch.where(
                correct_count == 1,
                correct.to(torch.int64).argmax(dim=1),
                torch.full_like(correct_count, logits.shape[1] - 1),
            )
            row_loss = functional.cross_entropy(logits, target, reduction="none")
            loss = (
                row_loss.mean()
                if variant.objective == "mean"
                else smooth_subject_dro_loss(row_loss, [row[1] for row in rows], dro_alpha, dro_tau)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    return model, history


@torch.no_grad()
def score_remaining_logits(
    model: RemainingCategoricalScorer,
    examples: Sequence[PoolExample],
    device: torch.device | str,
    cluster_features: str,
    batch_size: int = 256,
) -> list[CategoricalScores]:
    model.eval()
    result: list[CategoricalScores] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = _collate(chunk, device, cluster_features)
        logits = model(batch).cpu().numpy()
        for row, (example, clusters) in enumerate(zip(chunk, batch.cluster_values)):
            result.append(
                CategoricalScores(
                    question_id=example.question_id,
                    cluster_logits={cluster: float(logits[row, index]) for index, cluster in enumerate(clusters)},
                    none_logit=float(logits[row, -1]),
                )
            )
    return result


def predict_remaining(
    model: RemainingCategoricalScorer,
    batch: ObservableQueryBatch,
    examples: Sequence[PoolExample],
    device: torch.device | str,
    cluster_features: str,
    method: str,
) -> tuple[list[Selection], list[CategoricalScores]]:
    outputs = score_remaining_logits(model, examples, device, cluster_features)
    selections = categorical_selections(batch, examples, outputs, 1.0, method)
    return selections, outputs


def max_remaining_invariance_difference(
    model: RemainingCategoricalScorer,
    examples: Sequence[PoolExample],
    device: torch.device | str,
    cluster_features: str,
    intervention: str,
    seed: int,
) -> float:
    original = score_remaining_logits(model, examples, device, cluster_features)
    changed = [
        apply_intervention(example, intervention, _stable_rng(seed, example.question_id, intervention))
        for example in examples
    ]
    altered = score_remaining_logits(model, changed, device, cluster_features)
    differences: list[float] = []
    for left, right in zip(original, altered):
        if set(left.cluster_logits) != set(right.cluster_logits):
            raise ValueError("Invariant comparison has unaligned answer clusters")
        differences.extend(abs(left.cluster_logits[key] - right.cluster_logits[key]) for key in left.cluster_logits)
        differences.append(abs(left.none_logit - right.none_logit))
    return max(differences, default=0.0)
