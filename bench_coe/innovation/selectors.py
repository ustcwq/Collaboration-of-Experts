from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Iterable, Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import observable_to_legacy, records_by_question, topology_features, topology_matrix
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


def source_accuracy(batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> dict[str, float]:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Selectors may be fitted only with SourceTrainingLabels")
    if (labels.dataset, labels.split) != (batch.dataset, batch.split):
        raise ValueError("Source label provenance does not match the observable batch")
    result: dict[str, float] = {}
    for expert_id in batch.pool.expert_ids:
        values = [labels.get(question_id, expert_id) for question_id in batch.question_ids]
        observed = [float(value) for value in values if value is not None]
        result[expert_id] = float(np.mean(observed)) if observed else 0.0
    return result


def correctness_matrix(batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> np.ndarray:
    return np.asarray(
        [
            [float(bool(labels.get(question_id, expert_id))) for expert_id in batch.pool.expert_ids]
            for question_id in batch.question_ids
        ],
        dtype=float,
    )


def _selection_from_expert_scores(
    question_id: str,
    records: Iterable,
    expert_scores: Mapping[str, float],
    cluster_reducer: str = "max",
    fallback_reason: str | None = None,
) -> Selection:
    rows = tuple(records)
    valid = [record for record in rows if record.valid_output and record.per_query_cluster_id is not None]
    features = topology_features(rows)
    features["valid_mask"] = {record.expert_id: bool(record.valid_output) for record in rows}
    features["missing_mask"] = {record.expert_id: not bool(record.valid_output) for record in rows}
    if not valid:
        return Selection(question_id, None, None, None, {}, dict(expert_scores), "no_valid_output", features)
    by_cluster: dict[int, list[float]] = defaultdict(list)
    by_cluster_records: dict[int, list] = defaultdict(list)
    for record in valid:
        by_cluster[int(record.per_query_cluster_id)].append(float(expert_scores.get(record.expert_id, 0.0)))
        by_cluster_records[int(record.per_query_cluster_id)].append(record)
    if cluster_reducer == "sum":
        cluster_scores = {cluster_id: float(sum(values)) for cluster_id, values in by_cluster.items()}
    elif cluster_reducer == "mean":
        cluster_scores = {cluster_id: float(np.mean(values)) for cluster_id, values in by_cluster.items()}
    else:
        cluster_scores = {cluster_id: float(max(values)) for cluster_id, values in by_cluster.items()}
    selected_cluster = sorted(cluster_scores, key=lambda key: (-cluster_scores[key], key))[0]
    candidates = by_cluster_records[selected_cluster]
    selected_record = sorted(candidates, key=lambda record: (-float(expert_scores.get(record.expert_id, 0.0)), record.expert_id))[0]
    return Selection(
        question_id=question_id,
        selected_cluster_id=selected_cluster,
        selected_expert_id=selected_record.expert_id,
        normalized_answer=selected_record.normalized_answer,
        cluster_scores={str(key): value for key, value in sorted(cluster_scores.items())},
        expert_scores={key: float(value) for key, value in sorted(expert_scores.items())},
        fallback_reason=fallback_reason,
        observable_features=features,
    )


class Selector(ABC):
    name = "selector"

    @abstractmethod
    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "Selector":
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        raise NotImplementedError


class SourceBestSelector(Selector):
    name = "source_best_single"

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "SourceBestSelector":
        self.accuracy_ = source_accuracy(batch, labels)
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        grouped = records_by_question(batch)
        return [
            _selection_from_expert_scores(question_id, grouped[question_id], self.accuracy_)
            for question_id in sorted(grouped)
        ]


class MajorityVoteSelector(Selector):
    name = "majority_vote"

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "MajorityVoteSelector":
        self.accuracy_ = source_accuracy(batch, labels)
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        result: list[Selection] = []
        for question_id, rows in sorted(records_by_question(batch).items()):
            scores = {record.expert_id: 1.0 for record in rows}
            result.append(_selection_from_expert_scores(question_id, rows, scores, cluster_reducer="sum"))
        return result


class SourceWeightedVoteSelector(Selector):
    name = "source_accuracy_weighted_vote"

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "SourceWeightedVoteSelector":
        self.accuracy_ = source_accuracy(batch, labels)
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        return [
            _selection_from_expert_scores(question_id, rows, self.accuracy_, cluster_reducer="sum")
            for question_id, rows in sorted(records_by_question(batch).items())
        ]


class FamilyBalancedVoteSelector(Selector):
    name = "family_balanced_vote"

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "FamilyBalancedVoteSelector":
        self.accuracy_ = source_accuracy(batch, labels)
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        result: list[Selection] = []
        for question_id, rows in sorted(records_by_question(batch).items()):
            family_counts = Counter(record.expert_family for record in rows if record.valid_output)
            scores = {
                record.expert_id: 1.0 / family_counts[record.expert_family] if record.valid_output else 0.0
                for record in rows
            }
            result.append(_selection_from_expert_scores(question_id, rows, scores, cluster_reducer="sum"))
        return result


class OutputProfileKNNSelector(Selector):
    name = "output_profile_knn"

    def __init__(self, neighbors: int = 32, global_weight: float = 0.0) -> None:
        self.neighbors = neighbors
        self.global_weight = global_weight

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "OutputProfileKNNSelector":
        ids, features = topology_matrix(batch)
        self.source_ids_ = ids
        self.scaler_ = StandardScaler().fit(features)
        scaled = self.scaler_.transform(features)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.neighbors, len(ids)), metric="euclidean").fit(scaled)
        self.y_ = np.asarray(
            [[float(bool(labels.get(question_id, expert))) for expert in batch.pool.expert_ids] for question_id in ids],
            dtype=float,
        )
        self.experts_ = batch.pool.expert_ids
        self.accuracy_ = source_accuracy(batch, labels)
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        if batch.pool.expert_ids != self.experts_:
            raise ValueError("KNN baseline requires the fitted expert pool")
        ids, features = topology_matrix(batch)
        distances, indices = self.nn_.kneighbors(self.scaler_.transform(features))
        result: list[Selection] = []
        grouped = records_by_question(batch)
        global_vector = np.asarray([self.accuracy_[expert] for expert in self.experts_])
        for row_index, question_id in enumerate(ids):
            weights = 1.0 / np.maximum(distances[row_index], 1e-8)
            weights /= weights.sum()
            local = np.sum(self.y_[indices[row_index]] * weights[:, None], axis=0)
            scores = (1.0 - self.global_weight) * local + self.global_weight * global_vector
            result.append(
                _selection_from_expert_scores(
                    question_id,
                    grouped[question_id],
                    {expert: float(scores[index]) for index, expert in enumerate(self.experts_)},
                )
            )
        return result


class GlobalLocalSelector(OutputProfileKNNSelector):
    name = "global_local_competence"

    def __init__(self, neighbors: int = 32) -> None:
        super().__init__(neighbors=neighbors, global_weight=0.30)


class LegacyDARESelector(Selector):
    name = "dare_reliability"

    def __init__(self, neighbors: int = 32) -> None:
        self.neighbors = neighbors

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "LegacyDARESelector":
        from bench_coe.improve5_failure_ecology_experiments import split_robust_reliability

        self.source_batch_ = batch
        self.source_full_, self.source_rows_ = observable_to_legacy(batch)
        self.source_y_ = correctness_matrix(batch, labels)
        self.experts_ = list(batch.pool.expert_ids)
        self.reliability_ = split_robust_reliability(
            self.source_rows_, list(batch.question_ids), self.source_y_, self.experts_
        )
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        from bench_coe.improve5_failure_ecology_experiments import local_output_success

        if tuple(self.experts_) != batch.pool.expert_ids:
            raise ValueError("Legacy DARE requires the fitted expert pool")
        target_full, _ = observable_to_legacy(batch)
        local, _, target_group, target_uncertainty = local_output_success(
            self.source_full_,
            target_full,
            self.source_y_,
            self.experts_,
            list(self.source_batch_.question_ids),
            list(batch.question_ids),
            self.neighbors,
        )
        behavior_stability = target_group * (1.0 - 0.35 * target_uncertainty)
        scores = 0.44 * local + 0.28 * self.reliability_[None, :] + 0.24 * behavior_stability + 0.04 * self.source_y_.mean(axis=0)[None, :]
        grouped = records_by_question(batch)
        return [
            _selection_from_expert_scores(
                question_id,
                grouped[question_id],
                {expert: float(scores[row_index, col]) for col, expert in enumerate(self.experts_)},
            )
            for row_index, question_id in enumerate(batch.question_ids)
        ]


class LegacyRepairChainSelector(Selector):
    name = "repair_chain"

    def __init__(self, neighbors: int = 32) -> None:
        self.neighbors = neighbors

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "LegacyRepairChainSelector":
        from bench_coe.improve5_failure_ecology_experiments import correction_graph

        self.source_batch_ = batch
        self.source_full_, _ = observable_to_legacy(batch)
        self.source_y_ = correctness_matrix(batch, labels)
        self.experts_ = list(batch.pool.expert_ids)
        self.repair_ = correction_graph(self.source_y_)
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        from bench_coe.improve5_failure_ecology_experiments import local_output_success

        if tuple(self.experts_) != batch.pool.expert_ids:
            raise ValueError("Legacy RepairChain requires the fitted expert pool")
        target_full, _ = observable_to_legacy(batch)
        local, _, target_group, target_uncertainty = local_output_success(
            self.source_full_,
            target_full,
            self.source_y_,
            self.experts_,
            list(self.source_batch_.question_ids),
            list(batch.question_ids),
            self.neighbors,
        )
        global_accuracy = self.source_y_.mean(axis=0)
        failure = np.clip(1.0 - target_group + 0.40 * target_uncertainty, 0.0, 1.6)
        denom = failure.sum(axis=1, keepdims=True)
        weights = np.divide(failure, denom, out=np.full_like(failure, 1.0 / failure.shape[1]), where=denom > 1e-12)
        hop1 = weights @ self.repair_
        hop1_norm = hop1 / np.maximum(hop1.sum(axis=1, keepdims=True), 1e-12)
        hop2 = hop1_norm @ self.repair_
        scores = 0.30 * local + 0.25 * hop1 + 0.18 * hop2 + 0.16 * target_group + 0.11 * global_accuracy[None, :]
        grouped = records_by_question(batch)
        return [
            _selection_from_expert_scores(
                question_id,
                grouped[question_id],
                {expert: float(scores[row_index, col]) for col, expert in enumerate(self.experts_)},
            )
            for row_index, question_id in enumerate(batch.question_ids)
        ]


class RandomSelector(Selector):
    def __init__(self, seed: int, clusters: bool) -> None:
        self.seed = seed
        self.clusters = clusters
        self.name = "random_answer_cluster" if clusters else "random_expert"

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "RandomSelector":
        return self

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        result: list[Selection] = []
        for question_id, rows in sorted(records_by_question(batch).items()):
            digest = hashlib.sha256(f"{self.seed}:{question_id}".encode()).digest()
            value = int.from_bytes(digest[:8], "big")
            valid = [record for record in rows if record.valid_output]
            if not valid:
                result.append(_selection_from_expert_scores(question_id, rows, {}))
                continue
            if self.clusters:
                cluster_ids = sorted({int(record.per_query_cluster_id) for record in valid if record.per_query_cluster_id is not None})
                chosen = cluster_ids[value % len(cluster_ids)]
                scores = {record.expert_id: float(record.per_query_cluster_id == chosen) for record in rows}
            else:
                chosen_expert = sorted(record.expert_id for record in valid)[value % len(valid)]
                scores = {record.expert_id: float(record.expert_id == chosen_expert) for record in rows}
            result.append(_selection_from_expert_scores(question_id, rows, scores))
        return result


def baseline_selectors(seed: int = 20260808, neighbors: int = 32) -> list[Selector]:
    return [
        SourceBestSelector(),
        MajorityVoteSelector(),
        SourceWeightedVoteSelector(),
        FamilyBalancedVoteSelector(),
        OutputProfileKNNSelector(neighbors=neighbors),
        GlobalLocalSelector(neighbors=neighbors),
        LegacyDARESelector(neighbors=neighbors),
        LegacyRepairChainSelector(neighbors=neighbors),
        RandomSelector(seed, clusters=False),
        RandomSelector(seed, clusters=True),
    ]
