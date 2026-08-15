from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .features import observable_to_legacy, records_by_question, topology_features
from .schema import ExpertPool, ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import correctness_matrix


ABLATION_METHODS = (
    "m0_full",
    "m1_no_local",
    "m2_no_local_no_global",
    "m3_h1_support",
    "m3_cluster_h1_support",
    "m4_h1",
    "m4_cluster_mean_h1",
    "m4_h1_no_self",
    "m5_h1_h2",
    "m5_h1_h2_randomized",
    "m5_h1_h2_symmetric",
    "m5_h1_h2_no_self",
    "m5_h1_h2_centrality",
    "m6_support",
    "m7_local",
    "m8_global",
)

GRAPH_MODE_BY_METHOD = {
    "m4_h1_no_self": "no_self",
    "m5_h1_h2_randomized": "randomized",
    "m5_h1_h2_symmetric": "symmetric",
    "m5_h1_h2_no_self": "no_self",
    "m5_h1_h2_centrality": "column_centrality",
}

POOL_SHIFT_METHODS = (
    "m0_full",
    "m1_no_local",
    "m2_no_local_no_global",
    "m3_h1_support",
    "m3_cluster_h1_support",
    "m4_h1",
    "m4_cluster_mean_h1",
    "m5_h1_h2",
    "m6_support",
    "m7_local",
    "m8_global",
)


@dataclass(frozen=True)
class RepairComponents:
    question_ids: tuple[str, ...]
    expert_ids: tuple[str, ...]
    local: np.ndarray
    support: np.ndarray
    uncertainty: np.ndarray
    global_accuracy: np.ndarray
    repair_graph: np.ndarray
    failure_weights: np.ndarray
    hop1: np.ndarray
    hop2: np.ndarray
    valid_mask: np.ndarray
    cluster_ids: np.ndarray
    graph_mode: str = "raw"

    def __post_init__(self) -> None:
        rows = len(self.question_ids)
        cols = len(self.expert_ids)
        for name in ("local", "support", "uncertainty", "failure_weights", "hop1", "hop2"):
            value = getattr(self, name)
            if value.shape != (rows, cols):
                raise ValueError(f"{name} has shape {value.shape}, expected {(rows, cols)}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.global_accuracy.shape != (cols,) or self.repair_graph.shape != (cols, cols):
            raise ValueError("Global accuracy or repair graph has an invalid shape")
        if self.valid_mask.shape != (rows, cols) or self.cluster_ids.shape != (rows, cols):
            raise ValueError("Validity or answer-cluster arrays have an invalid shape")


def graph_variant(graph: np.ndarray, mode: str, seed: int) -> np.ndarray:
    graph = np.asarray(graph, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Repair graph must be square")
    if mode == "raw":
        result = graph.copy()
    elif mode == "no_self":
        result = graph.copy()
        np.fill_diagonal(result, 0.0)
    elif mode == "symmetric":
        result = 0.5 * (graph + graph.T)
    elif mode == "column_centrality":
        result = np.broadcast_to(graph.mean(axis=0), graph.shape).copy()
    elif mode == "randomized":
        rng = np.random.default_rng(seed)
        result = graph.copy()
        for source in range(len(graph)):
            destinations = np.asarray([index for index in range(len(graph)) if index != source], dtype=int)
            result[source, destinations] = rng.permutation(graph[source, destinations])
    else:
        raise ValueError(f"Unknown repair-graph mode: {mode}")
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _observable_state(batch: ObservableQueryBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = records_by_question(batch)
    rows = len(batch.question_ids)
    cols = len(batch.pool.expert_ids)
    support = np.zeros((rows, cols), dtype=float)
    uncertainty = np.zeros((rows, cols), dtype=float)
    valid = np.zeros((rows, cols), dtype=bool)
    cluster_ids = np.full((rows, cols), -1, dtype=int)
    expert_index = {expert: index for index, expert in enumerate(batch.pool.expert_ids)}
    for row_index, question_id in enumerate(batch.question_ids):
        records = grouped[question_id]
        cluster_counts: dict[int, int] = {}
        for record in records:
            if record.valid_output and record.per_query_cluster_id is not None:
                cluster_id = int(record.per_query_cluster_id)
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        for record in records:
            col = expert_index[record.expert_id]
            uncertainty[row_index, col] = min(1.0, max(0.0, float(record.uncertainty) / 4.0))
            if record.valid_output and record.per_query_cluster_id is not None:
                cluster_id = int(record.per_query_cluster_id)
                valid[row_index, col] = True
                cluster_ids[row_index, col] = cluster_id
                # Missing outputs are not answer clusters and remain in the pool-size denominator.
                support[row_index, col] = cluster_counts[cluster_id] / max(1, cols)
    return support, uncertainty, valid, cluster_ids


def repair_hops(
    support: np.ndarray,
    uncertainty: np.ndarray,
    repair_graph: np.ndarray,
    device: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    failure = np.clip(1.0 - support + 0.40 * uncertainty, 0.0, 1.6)
    denominator = failure.sum(axis=1, keepdims=True)
    weights = np.divide(
        failure,
        denominator,
        out=np.full_like(failure, 1.0 / max(1, failure.shape[1])),
        where=denominator > 1e-12,
    )
    if device is None:
        hop1 = weights @ repair_graph
        hop1_normalized = hop1 / np.maximum(hop1.sum(axis=1, keepdims=True), 1e-12)
        hop2 = hop1_normalized @ repair_graph
    else:
        import torch

        weight_tensor = torch.as_tensor(weights, dtype=torch.float64, device=device)
        graph_tensor = torch.as_tensor(repair_graph, dtype=torch.float64, device=device)
        hop1_tensor = weight_tensor @ graph_tensor
        normalized = hop1_tensor / torch.clamp(hop1_tensor.sum(dim=1, keepdim=True), min=1e-12)
        hop2_tensor = normalized @ graph_tensor
        hop1 = hop1_tensor.detach().cpu().numpy()
        hop2 = hop2_tensor.detach().cpu().numpy()
    return weights, np.nan_to_num(hop1), np.nan_to_num(hop2)


def fit_repair_components(
    train_batch: ObservableQueryBatch,
    train_labels: SourceTrainingLabels,
    target_batch: ObservableQueryBatch,
    *,
    neighbors: int = 32,
    graph_mode: str = "raw",
    seed: int = 0,
    device: Any | None = None,
) -> RepairComponents:
    if not isinstance(train_labels, SourceTrainingLabels) or train_labels.role != "source":
        raise TypeError("Repair scoring may be fitted only with SourceTrainingLabels")
    if (train_labels.dataset, train_labels.split) != (train_batch.dataset, train_batch.split):
        raise ValueError("Source label provenance does not match the training observables")
    if train_batch.pool.expert_ids != target_batch.pool.expert_ids:
        raise ValueError("Repair scoring requires the fitted expert pool")

    from bench_coe.improve5_failure_ecology_experiments import correction_graph, local_output_success

    source_full, _ = observable_to_legacy(train_batch)
    target_full, _ = observable_to_legacy(target_batch)
    source_y = correctness_matrix(train_batch, train_labels)
    local, _, _, _ = local_output_success(
        source_full,
        target_full,
        source_y,
        list(train_batch.pool.expert_ids),
        list(train_batch.question_ids),
        list(target_batch.question_ids),
        neighbors,
    )
    support, uncertainty, valid, cluster_ids = _observable_state(target_batch)
    raw_graph = np.asarray(correction_graph(source_y), dtype=float)
    repair_graph = graph_variant(raw_graph, graph_mode, seed)
    weights, hop1, hop2 = repair_hops(support, uncertainty, repair_graph, device=device)
    return RepairComponents(
        question_ids=target_batch.question_ids,
        expert_ids=target_batch.pool.expert_ids,
        local=np.nan_to_num(np.asarray(local, dtype=float)),
        support=support,
        uncertainty=uncertainty,
        global_accuracy=np.nan_to_num(source_y.mean(axis=0)),
        repair_graph=repair_graph,
        failure_weights=weights,
        hop1=hop1,
        hop2=hop2,
        valid_mask=valid,
        cluster_ids=cluster_ids,
        graph_mode=graph_mode,
    )


def with_graph_mode(
    components: RepairComponents,
    raw_graph: np.ndarray,
    mode: str,
    *,
    seed: int,
    device: Any | None = None,
) -> RepairComponents:
    repair_graph = graph_variant(raw_graph, mode, seed)
    weights, hop1, hop2 = repair_hops(
        components.support,
        components.uncertainty,
        repair_graph,
        device=device,
    )
    return replace(
        components,
        repair_graph=repair_graph,
        failure_weights=weights,
        hop1=hop1,
        hop2=hop2,
        graph_mode=mode,
    )


def expert_score_matrix(
    method: str,
    components: RepairComponents,
    *,
    beta: float = 0.5,
    alpha: float = 0.5,
) -> np.ndarray:
    local = components.local
    hop1 = components.hop1
    hop2 = components.hop2
    support = components.support
    global_accuracy = components.global_accuracy[None, :]
    if method == "m0_full":
        score = 0.30 * local + 0.25 * hop1 + 0.18 * hop2 + 0.16 * support + 0.11 * global_accuracy
    elif method == "m1_no_local":
        score = 0.25 * hop1 + 0.18 * hop2 + 0.16 * support + 0.11 * global_accuracy
    elif method == "m2_no_local_no_global":
        score = 0.25 * hop1 + 0.18 * hop2 + 0.16 * support
    elif method in {"m3_h1_support", "m3_cluster_h1_support"}:
        score = beta * hop1 + (1.0 - beta) * support
    elif method in {"m4_h1", "m4_cluster_mean_h1", "m4_h1_no_self"}:
        score = hop1.copy()
    elif method.startswith("m5_h1_h2"):
        score = alpha * hop1 + (1.0 - alpha) * hop2
    elif method == "m6_support":
        score = support.copy()
    elif method == "m7_local":
        score = local.copy()
    elif method == "m8_global":
        score = np.broadcast_to(global_accuracy, local.shape).copy()
    else:
        raise ValueError(f"Unknown repair-score ablation: {method}")
    return np.nan_to_num(score, nan=-1e12, posinf=1e12, neginf=-1e12)


def _best_with_ties(
    candidates: list[int],
    primary: np.ndarray,
    support: np.ndarray,
    global_accuracy: np.ndarray,
    stable_ids: list[str],
    tolerance: float,
) -> int:
    best = candidates[0]
    for candidate in candidates[1:]:
        if primary[candidate] > primary[best] + tolerance:
            best = candidate
            continue
        if abs(primary[candidate] - primary[best]) > tolerance:
            continue
        if support[candidate] > support[best] + tolerance:
            best = candidate
            continue
        if abs(support[candidate] - support[best]) > tolerance:
            continue
        if global_accuracy[candidate] > global_accuracy[best] + tolerance:
            best = candidate
            continue
        if (
            abs(global_accuracy[candidate] - global_accuracy[best]) <= tolerance
            and stable_ids[candidate] < stable_ids[best]
        ):
            best = candidate
    return best


def selections_from_components(
    batch: ObservableQueryBatch,
    components: RepairComponents,
    method: str,
    *,
    beta: float = 0.5,
    alpha: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> list[Selection]:
    if batch.question_ids != components.question_ids or batch.pool.expert_ids != components.expert_ids:
        raise ValueError("Component rows are not aligned with the observable batch")
    expert_scores = expert_score_matrix(method, components, beta=beta, alpha=alpha)
    grouped = records_by_question(batch)
    result: list[Selection] = []
    cluster_level = method in {"m3_cluster_h1_support", "m4_cluster_mean_h1"}
    for row_index, question_id in enumerate(batch.question_ids):
        records = grouped[question_id]
        record_by_expert = {record.expert_id: record for record in records}
        valid_indices = [index for index, value in enumerate(components.valid_mask[row_index]) if value]
        features: dict[str, Any] = dict(topology_features(records))
        features.update({"beta": float(beta), "alpha": float(alpha), "graph_mode": components.graph_mode})
        if not valid_indices:
            result.append(
                Selection(
                    question_id,
                    None,
                    None,
                    None,
                    {},
                    {expert: float(expert_scores[row_index, index]) for index, expert in enumerate(components.expert_ids)},
                    "no_valid_output",
                    features,
                    "no_valid_output",
                )
            )
            continue

        by_cluster: dict[int, list[int]] = {}
        for index in valid_indices:
            by_cluster.setdefault(int(components.cluster_ids[row_index, index]), []).append(index)
        cluster_scores: dict[int, float] = {}
        if cluster_level:
            for cluster_id, members in by_cluster.items():
                if method == "m3_cluster_h1_support":
                    mean_hop1 = float(np.mean(components.hop1[row_index, members]))
                    cluster_support = len(members) / max(1, len(components.expert_ids))
                    cluster_scores[cluster_id] = beta * mean_hop1 + (1.0 - beta) * cluster_support
                else:
                    cluster_scores[cluster_id] = float(np.mean(components.hop1[row_index, members]))
            cluster_candidates = sorted(cluster_scores)
            cluster_primary = np.asarray([cluster_scores[key] for key in cluster_candidates], dtype=float)
            cluster_support = np.asarray(
                [len(by_cluster[key]) / max(1, len(components.expert_ids)) for key in cluster_candidates], dtype=float
            )
            cluster_global = np.asarray(
                [float(np.mean(components.global_accuracy[by_cluster[key]])) for key in cluster_candidates], dtype=float
            )
            selected_position = _best_with_ties(
                list(range(len(cluster_candidates))),
                cluster_primary,
                cluster_support,
                cluster_global,
                [str(value) for value in cluster_candidates],
                tie_tolerance,
            )
            selected_cluster = cluster_candidates[selected_position]
            selected_index = _best_with_ties(
                by_cluster[selected_cluster],
                components.hop1[row_index],
                components.support[row_index],
                components.global_accuracy,
                list(components.expert_ids),
                tie_tolerance,
            )
            row_expert_scores = {
                expert: float(cluster_scores[int(components.cluster_ids[row_index, index])])
                if components.valid_mask[row_index, index]
                else float(expert_scores[row_index, index])
                for index, expert in enumerate(components.expert_ids)
            }
            tie_rule = "cluster_score_then_support_then_mean_global_then_cluster_id; expert_h1_then_global_then_id"
        else:
            selected_index = _best_with_ties(
                valid_indices,
                expert_scores[row_index],
                components.support[row_index],
                components.global_accuracy,
                list(components.expert_ids),
                tie_tolerance,
            )
            selected_cluster = int(components.cluster_ids[row_index, selected_index])
            cluster_scores = {
                cluster_id: float(max(expert_scores[row_index, member] for member in members))
                for cluster_id, members in by_cluster.items()
            }
            row_expert_scores = {
                expert: float(expert_scores[row_index, index]) for index, expert in enumerate(components.expert_ids)
            }
            tie_rule = "expert_score_then_answer_support_then_source_global_then_expert_id"

        selected_expert = components.expert_ids[selected_index]
        selected_record = record_by_expert[selected_expert]
        features.update(
            {
                "selected_local": float(components.local[row_index, selected_index]),
                "selected_hop1": float(components.hop1[row_index, selected_index]),
                "selected_hop2": float(components.hop2[row_index, selected_index]),
                "selected_support": float(components.support[row_index, selected_index]),
                "selected_global": float(components.global_accuracy[selected_index]),
                "selected_failure_weight": float(components.failure_weights[row_index, selected_index]),
            }
        )
        result.append(
            Selection(
                question_id=question_id,
                selected_cluster_id=selected_cluster,
                selected_expert_id=selected_expert,
                normalized_answer=selected_record.normalized_answer,
                cluster_scores={str(key): float(value) for key, value in sorted(cluster_scores.items())},
                expert_scores=dict(sorted(row_expert_scores.items())),
                fallback_reason=None,
                observable_features=features,
                tie_breaking=tie_rule,
            )
        )
    return result


def source_accuracy_of_selections(
    selections: list[Selection],
    labels: SourceTrainingLabels | Mapping[tuple[str, str], bool],
) -> float:
    correctness = labels.correctness if isinstance(labels, SourceTrainingLabels) else labels
    values = [bool(correctness.get((item.question_id, item.selected_expert_id or ""), False)) for item in selections]
    return float(np.mean(values)) if values else 0.0


def subset_expert_pool(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    expert_ids: tuple[str, ...],
) -> tuple[ObservableQueryBatch, SourceTrainingLabels]:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Pool projection requires SourceTrainingLabels")
    requested = tuple(sorted(expert_ids))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Projected expert pool must be non-empty and unique")
    unknown = set(requested).difference(batch.pool.expert_ids)
    if unknown:
        raise ValueError(f"Projected expert pool contains unknown experts: {sorted(unknown)}")
    keep = set(requested)
    projected_batch = ObservableQueryBatch(
        dataset=batch.dataset,
        split=batch.split,
        modality=batch.modality,
        pool=ExpertPool(requested, {expert: batch.pool.family_by_expert[expert] for expert in requested}),
        records=tuple(record for record in batch.records if record.expert_id in keep),
    )
    projected_labels = SourceTrainingLabels._from_source_adapter(
        labels.dataset,
        labels.split,
        {key: value for key, value in labels.correctness.items() if key[1] in keep},
        labels.environment_by_question,
    )
    return projected_batch, projected_labels
