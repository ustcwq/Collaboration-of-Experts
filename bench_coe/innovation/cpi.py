from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .features import records_by_question
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


INTERVENTIONS = (
    "permutation",
    "random_dropout",
    "leave_expert_out",
    "leave_family_out",
    "missing_output",
    "exact_clone",
    "pseudo_clone",
    "known_swap",
)


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "::".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


@dataclass(frozen=True)
class FingerprintTable:
    expert_ids: tuple[str, ...]
    family_names: tuple[str, ...]
    values: Mapping[str, tuple[float, ...]]
    feature_names: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class PoolExample:
    question_id: str
    expert_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    fingerprints: np.ndarray
    cluster_ids: np.ndarray
    uncertainties: np.ndarray
    valid: np.ndarray
    cluster_labels: Mapping[int, float]
    normalized_answers: Mapping[int, str]

    def __post_init__(self) -> None:
        length = len(self.expert_ids)
        if not (
            len(self.family_ids)
            == len(self.fingerprints)
            == len(self.cluster_ids)
            == len(self.uncertainties)
            == len(self.valid)
            == length
        ):
            raise ValueError("PoolExample fields have inconsistent lengths")
        if any(valid and cluster < 0 for valid, cluster in zip(self.valid, self.cluster_ids)):
            raise ValueError("Every valid token must belong to a cluster")


def fit_source_fingerprints(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    rank: int = 4,
    extra_batch: ObservableQueryBatch | None = None,
    extra_labels: SourceTrainingLabels | None = None,
) -> FingerprintTable:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("CPI fingerprints may be fitted only with SourceTrainingLabels")
    if (labels.dataset, labels.split) != (batch.dataset, batch.split):
        raise ValueError("Source label provenance does not match the observable batch")
    expert_ids = batch.pool.expert_ids
    question_ids = batch.question_ids
    correctness = np.asarray(
        [
            [float(bool(labels.get(question_id, expert))) for question_id in question_ids]
            for expert in expert_ids
        ],
        dtype=np.float64,
    )
    centered = correctness - correctness.mean(axis=0, keepdims=True)
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
    embedding_by_expert = {expert: embedding[index] for index, expert in enumerate(expert_ids)}
    family_by_expert = dict(batch.pool.family_by_expert)
    batches = [batch]
    if extra_batch is not None:
        if extra_labels is None or (extra_labels.dataset, extra_labels.split) != (batch.dataset, batch.split):
            raise ValueError("Real swap experts require aligned source labels")
        if extra_batch.question_ids != question_ids:
            raise ValueError("Real swap expert rows do not align with the source questions")
        batches.append(extra_batch)
        base_item_mean = correctness.mean(axis=0)
        for expert in extra_batch.pool.expert_ids:
            row = np.asarray([float(bool(extra_labels.get(qid, expert))) for qid in question_ids], dtype=np.float64)
            correctness_by_expert[expert] = row
            projected = (row - base_item_mean) @ vt[:use_rank].T / max(1.0, np.sqrt(len(question_ids)))
            if use_rank < rank:
                projected = np.pad(projected, (0, rank - use_rank))
            embedding_by_expert[expert] = projected
            if expert not in all_experts:
                all_experts.append(expert)
            family_by_expert[expert] = extra_batch.pool.family_by_expert[expert]

    family_names = tuple(sorted(set(family_by_expert.values())))
    feature_names = (
        "source_global_accuracy",
        "source_valid_rate",
        *(f"correctness_svd_{index}" for index in range(rank)),
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
    observed_costs = [value for values in grouped_costs.values() for value in values]
    fallback_cost = float(np.median(observed_costs)) if observed_costs else 1.0

    values: dict[str, tuple[float, ...]] = {}
    for expert in all_experts:
        family = family_by_expert[expert]
        family_vector = [float(family == candidate) for candidate in family_names] + [0.0]
        costs = grouped_costs[expert]
        cost = float(np.median(costs)) if costs else fallback_cost
        expert_correctness = correctness_by_expert[expert]
        global_accuracy = (expert_correctness.sum() + 1.0) / (len(expert_correctness) + 2.0)
        observed_valid = valid_by_expert[expert]
        valid_rate = (sum(observed_valid) + 1.0) / (len(observed_valid) + 2.0)
        vector = [
            global_accuracy,
            valid_rate,
            *embedding_by_expert[expert].tolist(),
            *family_vector,
            np.log1p(max(0.0, cost)),
            float(not costs),
        ]
        values[expert] = tuple(float(value) for value in vector)
    return FingerprintTable(tuple(all_experts), family_names, values, tuple(feature_names))


def make_pool_example(
    batch: ObservableQueryBatch,
    question_id: str,
    fingerprints: FingerprintTable,
    labels: SourceTrainingLabels | None = None,
) -> PoolExample:
    if labels is not None and (not isinstance(labels, SourceTrainingLabels) or labels.role != "source"):
        raise TypeError("CPI training examples require SourceTrainingLabels")
    if labels is not None and (labels.dataset, labels.split) != (batch.dataset, batch.split):
        raise ValueError("CPI training label provenance does not match the observable batch")
    records = batch.for_question(question_id)
    expert_ids = tuple(record.expert_id for record in records)
    missing = set(expert_ids).difference(fingerprints.values)
    if missing:
        raise ValueError(f"Missing source fingerprints for experts: {sorted(missing)}")
    cluster_ids = np.asarray(
        [int(record.per_query_cluster_id) if record.valid_output and record.per_query_cluster_id is not None else -1 for record in records],
        dtype=np.int64,
    )
    cluster_labels: dict[int, float] = {}
    normalized_answers: dict[int, str] = {}
    for cluster in batch.clusters(question_id):
        normalized_answers[cluster.cluster_id] = cluster.normalized_answer
        if labels is not None:
            member_labels = [labels.get(question_id, expert) for expert in cluster.expert_ids]
            observed = [float(value) for value in member_labels if value is not None]
            cluster_labels[cluster.cluster_id] = float(np.mean(observed) >= 0.5) if observed else 0.0
    return PoolExample(
        question_id=question_id,
        expert_ids=expert_ids,
        family_ids=tuple(record.expert_family for record in records),
        fingerprints=np.asarray([fingerprints.values[expert] for expert in expert_ids], dtype=np.float32),
        cluster_ids=cluster_ids,
        uncertainties=np.asarray([np.log1p(max(0.0, record.uncertainty)) for record in records], dtype=np.float32),
        valid=np.asarray([record.valid_output for record in records], dtype=bool),
        cluster_labels=cluster_labels,
        normalized_answers=normalized_answers,
    )


def _take(example: PoolExample, indices: Sequence[int]) -> PoolExample:
    selected = np.asarray(indices, dtype=np.int64)
    return replace(
        example,
        expert_ids=tuple(example.expert_ids[index] for index in selected),
        family_ids=tuple(example.family_ids[index] for index in selected),
        fingerprints=example.fingerprints[selected].copy(),
        cluster_ids=example.cluster_ids[selected].copy(),
        uncertainties=example.uncertainties[selected].copy(),
        valid=example.valid[selected].copy(),
    )


def remove_expert(example: PoolExample, expert_id: str) -> PoolExample:
    return _take(example, [index for index, value in enumerate(example.expert_ids) if value != expert_id])


def remove_family(example: PoolExample, family_id: str) -> PoolExample:
    return _take(example, [index for index, value in enumerate(example.family_ids) if value != family_id])


def subset_pool_size(example: PoolExample, size: int, rng: np.random.Generator) -> PoolExample:
    if size <= 0:
        raise ValueError("Pool size must be positive")
    if size >= len(example.expert_ids):
        return example
    valid_indices = np.flatnonzero(example.valid)
    chosen = rng.choice(len(example.expert_ids), size=size, replace=False)
    if not np.any(example.valid[chosen]) and len(valid_indices):
        chosen[0] = int(rng.choice(valid_indices))
    return _take(example, sorted(set(int(value) for value in chosen)))


def relabel_clusters(example: PoolExample, mapping: Mapping[int, int]) -> PoolExample:
    clusters = np.asarray([mapping.get(int(value), int(value)) if value >= 0 else -1 for value in example.cluster_ids], dtype=np.int64)
    labels = {mapping.get(int(key), int(key)): value for key, value in example.cluster_labels.items()}
    answers = {mapping.get(int(key), int(key)): value for key, value in example.normalized_answers.items()}
    return replace(example, cluster_ids=clusters, cluster_labels=labels, normalized_answers=answers)


def apply_intervention(example: PoolExample, name: str, rng: np.random.Generator) -> PoolExample:
    if name == "none":
        return example
    if name not in INTERVENTIONS:
        raise ValueError(f"Unknown CPI intervention: {name}")
    count = len(example.expert_ids)
    valid_indices = np.flatnonzero(example.valid)
    if name == "permutation":
        return _take(example, rng.permutation(count).tolist())
    if name in {"random_dropout", "leave_expert_out"}:
        if len(valid_indices) <= 1:
            return example
        if name == "random_dropout":
            keep = rng.random(count) >= 0.2
            if not np.any(keep & example.valid):
                keep[int(rng.choice(valid_indices))] = True
            indices = np.flatnonzero(keep).tolist()
        else:
            removed = int(rng.choice(valid_indices))
            indices = [index for index in range(count) if index != removed]
        return _take(example, indices)
    if name == "leave_family_out":
        families = sorted({example.family_ids[index] for index in valid_indices})
        if len(families) <= 1:
            return example
        removed_family = str(rng.choice(families))
        indices = [index for index in range(count) if example.family_ids[index] != removed_family]
        return _take(example, indices)
    if name == "missing_output":
        if not len(valid_indices):
            return example
        index = int(rng.choice(valid_indices))
        valid = example.valid.copy()
        clusters = example.cluster_ids.copy()
        valid[index] = False
        clusters[index] = -1
        return replace(example, valid=valid, cluster_ids=clusters)
    if name in {"exact_clone", "pseudo_clone"}:
        if not len(valid_indices):
            return example
        index = int(rng.choice(valid_indices))
        fingerprints = np.concatenate([example.fingerprints, example.fingerprints[index : index + 1].copy()], axis=0)
        uncertainty = np.concatenate([example.uncertainties, example.uncertainties[index : index + 1]])
        if name == "pseudo_clone":
            fingerprints[-1, 0] = np.clip(fingerprints[-1, 0] + 0.025, 0.0, 1.0)
            uncertainty[-1] += 0.01
        return replace(
            example,
            expert_ids=(*example.expert_ids, f"{name}::{example.expert_ids[index]}"),
            family_ids=(*example.family_ids, example.family_ids[index]),
            fingerprints=fingerprints,
            cluster_ids=np.concatenate([example.cluster_ids, example.cluster_ids[index : index + 1]]),
            uncertainties=uncertainty,
            valid=np.concatenate([example.valid, example.valid[index : index + 1]]),
        )
    if name == "known_swap":
        raise ValueError("known_swap requires apply_known_swap with a real configured replacement pool")
    raise AssertionError(name)


def apply_known_swap(
    example: PoolExample,
    replacements: PoolExample,
    swap_mapping: Mapping[str, str],
    rng: np.random.Generator,
) -> PoolExample:
    available = [
        (removed, donor)
        for removed, donor in sorted(swap_mapping.items())
        if removed in example.expert_ids and donor in replacements.expert_ids
    ]
    if not available:
        raise ValueError("No configured real expert swap is available for this pool")
    removed, donor = available[int(rng.integers(0, len(available)))]
    removed_index = example.expert_ids.index(removed)
    donor_index = replacements.expert_ids.index(donor)
    kept = _take(example, [index for index in range(len(example.expert_ids)) if index != removed_index])
    donor_valid = bool(replacements.valid[donor_index])
    donor_cluster = int(replacements.cluster_ids[donor_index])
    answers = dict(kept.normalized_answers)
    labels = dict(kept.cluster_labels)
    aligned_cluster = -1
    if donor_valid and donor_cluster >= 0:
        donor_answer = replacements.normalized_answers[donor_cluster]
        aligned_cluster = next((cluster for cluster, answer in answers.items() if answer == donor_answer), max(answers, default=-1) + 1)
        answers[aligned_cluster] = donor_answer
        if aligned_cluster not in labels:
            labels[aligned_cluster] = float(replacements.cluster_labels.get(donor_cluster, 0.0))
    return replace(
        kept,
        expert_ids=(*kept.expert_ids, donor),
        family_ids=(*kept.family_ids, replacements.family_ids[donor_index]),
        fingerprints=np.concatenate([kept.fingerprints, replacements.fingerprints[donor_index : donor_index + 1]], axis=0),
        cluster_ids=np.concatenate([kept.cluster_ids, np.asarray([aligned_cluster], dtype=np.int64)]),
        uncertainties=np.concatenate([kept.uncertainties, replacements.uncertainties[donor_index : donor_index + 1]]),
        valid=np.concatenate([kept.valid, np.asarray([donor_valid], dtype=bool)]),
        cluster_labels=labels,
        normalized_answers=answers,
    )


def canonicalize_exact_clones(example: PoolExample) -> PoolExample:
    indices: list[int] = []
    seen: set[tuple] = set()
    for index in range(len(example.expert_ids)):
        key = (
            example.family_ids[index],
            int(example.cluster_ids[index]),
            bool(example.valid[index]),
            round(float(example.uncertainties[index]), 8),
            tuple(np.round(example.fingerprints[index], 8).tolist()),
        )
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return _take(example, indices)


def _token_features(example: PoolExample) -> np.ndarray:
    return np.concatenate(
        [
            example.fingerprints.astype(np.float32),
            example.valid.astype(np.float32)[:, None],
            example.uncertainties.astype(np.float32)[:, None],
        ],
        axis=1,
    )


@dataclass
class CollatedPoolBatch:
    expert_features: torch.Tensor
    expert_mask: torch.Tensor
    cluster_assignment: torch.Tensor
    cluster_mask: torch.Tensor
    cluster_extra: torch.Tensor
    targets: torch.Tensor
    cluster_values: list[tuple[int, ...]]


def collate_pool_examples(examples: Sequence[PoolExample], device: torch.device | str) -> CollatedPoolBatch:
    canonical = [canonicalize_exact_clones(example) for example in examples]
    if not canonical:
        raise ValueError("Cannot collate an empty CPI batch")
    feature_dim = _token_features(canonical[0]).shape[1]
    max_experts = max(len(example.expert_ids) for example in canonical)
    cluster_values = [tuple(sorted({int(value) for value in example.cluster_ids if value >= 0})) for example in canonical]
    max_clusters = max(1, max(len(values) for values in cluster_values))
    batch_size = len(canonical)
    features = np.zeros((batch_size, max_experts, feature_dim), dtype=np.float32)
    expert_mask = np.zeros((batch_size, max_experts), dtype=bool)
    assignment = np.full((batch_size, max_experts), -1, dtype=np.int64)
    cluster_mask = np.zeros((batch_size, max_clusters), dtype=bool)
    extras = np.zeros((batch_size, max_clusters, 4), dtype=np.float32)
    targets = np.zeros((batch_size, max_clusters), dtype=np.float32)
    for row, (example, values) in enumerate(zip(canonical, cluster_values)):
        token_features = _token_features(example)
        features[row, : len(example.expert_ids)] = token_features
        expert_mask[row, : len(example.expert_ids)] = True
        position = {cluster: index for index, cluster in enumerate(values)}
        for expert_index, cluster in enumerate(example.cluster_ids):
            if int(cluster) >= 0:
                assignment[row, expert_index] = position[int(cluster)]
        cluster_mask[row, : len(values)] = True
        valid_count = max(1, int(np.sum(example.valid)))
        pool_families = max(1, len({family for family, valid in zip(example.family_ids, example.valid) if valid}))
        for column, cluster in enumerate(values):
            members = np.flatnonzero((example.cluster_ids == cluster) & example.valid)
            member_families = {example.family_ids[index] for index in members}
            uncertainty = example.uncertainties[members]
            extras[row, column] = (
                len(members) / valid_count,
                len(member_families) / pool_families,
                float(np.mean(uncertainty)) if len(uncertainty) else 0.0,
                float(np.max(uncertainty)) if len(uncertainty) else 0.0,
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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    maximum = masked.max(dim=dim).values
    has_member = mask.any(dim=dim).unsqueeze(-1)
    return torch.where(has_member, maximum, torch.zeros_like(maximum))


class InvariantClusterScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, linear: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim if linear else hidden_dim
        self.linear = linear
        if linear:
            self.phi: nn.Module = nn.Identity()
            self.rho: nn.Module = nn.Linear(4 * input_dim + 4, 1)
        else:
            self.phi = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.rho = nn.Sequential(
                nn.Linear(4 * hidden_dim + 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, batch: CollatedPoolBatch) -> torch.Tensor:
        encoded = self.phi(batch.expert_features)
        global_mean = _masked_mean(encoded, batch.expert_mask, dim=1)
        global_max = _masked_max(encoded, batch.expert_mask, dim=1)
        cluster_count = batch.cluster_mask.shape[1]
        cluster_indices = torch.arange(cluster_count, device=encoded.device)[None, :, None]
        member_mask = (
            (batch.cluster_assignment[:, None, :] == cluster_indices)
            & batch.expert_mask[:, None, :]
        )
        expanded = encoded[:, None, :, :].expand(-1, cluster_count, -1, -1)
        cluster_mean = _masked_mean(expanded, member_mask, dim=2)
        cluster_max = _masked_max(expanded, member_mask, dim=2)
        repeated_global_mean = global_mean[:, None, :].expand(-1, cluster_count, -1)
        repeated_global_max = global_max[:, None, :].expand(-1, cluster_count, -1)
        summary = torch.cat(
            [cluster_mean, cluster_max, repeated_global_mean, repeated_global_max, batch.cluster_extra],
            dim=-1,
        )
        logits = self.rho(summary).squeeze(-1)
        return logits.masked_fill(~batch.cluster_mask, -1e9)


def train_cluster_scorer(
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
) -> tuple[InvariantClusterScorer, list[float]]:
    if variant not in {"none", "full", *INTERVENTIONS}:
        raise ValueError(f"Unknown training variant: {variant}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = InvariantClusterScorer(input_dim, hidden_dim=hidden_dim, linear=linear).to(device)
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buffer)
    model.initialization_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()  # type: ignore[attr-defined]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    history: list[float] = []
    for epoch in range(epochs):
        augmented: list[PoolExample] = []
        for example in examples:
            augmented.append(example)
            if variant == "none":
                augmented.append(example)
            else:
                intervention = (
                    INTERVENTIONS[(epoch + int.from_bytes(hashlib.sha256(example.question_id.encode()).digest()[:2], "little")) % len(INTERVENTIONS)]
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
        order = _stable_rng(seed, epoch, "order").permutation(len(augmented))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch_size):
            selected = [augmented[int(index)] for index in order[start : start + batch_size]]
            selected = [example for example in selected if np.any(example.valid)]
            if not selected:
                continue
            batch = collate_pool_examples(selected, device)
            logits = model(batch)
            raw_loss = loss_function(logits, batch.targets)
            loss = (raw_loss * batch.cluster_mask).sum() / batch.cluster_mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return model, history


@torch.no_grad()
def score_examples(
    model: InvariantClusterScorer,
    examples: Sequence[PoolExample],
    device: torch.device | str,
    batch_size: int = 256,
) -> list[dict[int, float]]:
    model.eval()
    result: list[dict[int, float]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate_pool_examples(chunk, device)
        probabilities = torch.softmax(model(batch), dim=-1).cpu().numpy()
        for row, cluster_values in enumerate(batch.cluster_values):
            result.append({cluster: float(probabilities[row, index]) for index, cluster in enumerate(cluster_values)})
    return result


def selections_from_scores(
    batch: ObservableQueryBatch,
    examples: Sequence[PoolExample],
    scores: Sequence[Mapping[int, float]],
    method: str,
) -> list[Selection]:
    result: list[Selection] = []
    for example, cluster_scores in zip(examples, scores):
        valid_scores = {int(cluster): float(score) for cluster, score in cluster_scores.items()}
        if not valid_scores:
            result.append(Selection(example.question_id, None, None, None, {}, {}, "no_valid_output", {"method": method}))
            continue
        selected_cluster = sorted(valid_scores, key=lambda key: (-valid_scores[key], key))[0]
        candidates = sorted(
            expert
            for expert, valid, cluster in zip(example.expert_ids, example.valid, example.cluster_ids)
            if valid and int(cluster) == selected_cluster
        )
        original_candidates = [expert for expert in candidates if expert in batch.pool.expert_ids]
        representative = (original_candidates or candidates)[0] if candidates else None
        if representative is not None and representative not in {
            expert for expert, valid in zip(example.expert_ids, example.valid) if valid
        }:
            raise AssertionError("CPI selected an expert outside the available intervention pool")
        result.append(
            Selection(
                question_id=example.question_id,
                selected_cluster_id=selected_cluster,
                selected_expert_id=representative,
                normalized_answer=example.normalized_answers.get(selected_cluster),
                cluster_scores={str(key): value for key, value in sorted(valid_scores.items())},
                expert_scores={},
                fallback_reason=None if representative else "intervention_cluster_has_no_original_representative",
                observable_features={
                    "method": method,
                    "canonical_pool_size": len(canonicalize_exact_clones(example).expert_ids),
                    "available_expert_ids": list(example.expert_ids),
                    "valid_mask": [bool(value) for value in example.valid],
                    "family_ids": list(example.family_ids),
                },
            )
        )
    return result


def predict_selections(
    model: InvariantClusterScorer,
    batch: ObservableQueryBatch,
    fingerprints: FingerprintTable,
    device: torch.device | str,
    method: str,
    intervention: str = "none",
    seed: int = 0,
    replacement_examples: Mapping[str, PoolExample] | None = None,
    swap_mapping: Mapping[str, str] | None = None,
) -> tuple[list[Selection], list[dict[int, float]], list[PoolExample]]:
    examples: list[PoolExample] = []
    for question_id in batch.question_ids:
        example = make_pool_example(batch, question_id, fingerprints)
        if intervention != "none":
            rng = _stable_rng(seed, question_id, intervention)
            if intervention == "known_swap":
                if replacement_examples is None or swap_mapping is None:
                    raise ValueError("known_swap prediction requires real replacement examples and a frozen mapping")
                example = apply_known_swap(example, replacement_examples[question_id], swap_mapping, rng)
            else:
                example = apply_intervention(example, intervention, rng)
        examples.append(example)
    scores = score_examples(model, examples, device)
    return selections_from_scores(batch, examples, scores, method), scores, examples


def max_probability_difference(
    first: Sequence[Mapping[int, float]],
    second: Sequence[Mapping[int, float]],
) -> float:
    if len(first) != len(second):
        raise ValueError("Invariant predictions have different sample counts")
    differences: list[float] = []
    for left, right in zip(first, second):
        if set(left) != set(right):
            raise ValueError("Invariant predictions have different aligned cluster sets")
        for cluster in left:
            differences.append(abs(float(left[cluster]) - float(right[cluster])))
    return max(differences, default=0.0)


def clone_invariance_loss(
    model: InvariantClusterScorer,
    examples: Sequence[PoolExample],
    device: torch.device | str,
    seed: int,
) -> float:
    original = score_examples(model, examples, device)
    clones = [apply_intervention(example, "exact_clone", _stable_rng(seed, example.question_id, "clone")) for example in examples]
    cloned = score_examples(model, clones, device)
    return max_probability_difference(original, cloned)


def subject_folds(labels: SourceTrainingLabels) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    environments = sorted(set(labels.environment_by_question.values()))
    all_ids = set(labels.environment_by_question)
    return [
        (
            environment,
            tuple(sorted(question_id for question_id in all_ids if labels.environment_by_question[question_id] != environment)),
            tuple(sorted(question_id for question_id in all_ids if labels.environment_by_question[question_id] == environment)),
        )
        for environment in environments
    ]
