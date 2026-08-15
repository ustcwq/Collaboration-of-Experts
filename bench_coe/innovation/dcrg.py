from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import expert_observable_features, records_by_question, topology_matrix
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import Selector, _selection_from_expert_scores, correctness_matrix, source_accuracy


@dataclass(frozen=True)
class RescueEdge:
    source_expert: str
    target_expert: str
    raw_support: int
    raw_conditional_correctness: float
    expected_baseline: float
    residual_mean: float
    standard_error: float
    pooled_lcb: float
    eligible_environments: int
    sign_consistency: float
    environment_lcb_q10: float
    stable: bool
    stable_weight: float
    environment_signs: dict[str, int]


@dataclass
class _BinaryFailureModel:
    scaler: StandardScaler | None
    classifier: LogisticRegression | None
    constant: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None or self.scaler is None:
            return np.full(len(features), self.constant, dtype=float)
        return self.classifier.predict_proba(self.scaler.transform(features))[:, 1]


def validate_oof_splits(splits: list[tuple[np.ndarray, np.ndarray]], size: int) -> None:
    heldout: list[int] = []
    for train, test in splits:
        if set(train).intersection(test):
            raise ValueError("OOF train/test overlap")
        heldout.extend(int(value) for value in test)
    if sorted(heldout) != list(range(size)):
        raise ValueError("OOF folds do not cover every row exactly once")


def make_oof_splits(environments: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = min(max(2, folds), len(environments))
    counts = {value: int(np.sum(environments == value)) for value in set(environments.tolist())}
    folds = min(folds, max(2, min(counts.values())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [(train, test) for train, test in splitter.split(np.zeros(len(environments)), environments)]
    validate_oof_splits(splits, len(environments))
    return splits


def cross_fitted_expected_correctness(
    correctness: np.ndarray,
    environments: np.ndarray,
    observable_features: np.ndarray,
    folds: int,
    seed: int,
    adjust_difficulty: bool = True,
    adjust_environment: bool = True,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    rows, experts = correctness.shape
    if observable_features.shape[0] != rows:
        raise ValueError("Observable nuisance features do not align with correctness rows")
    splits = make_oof_splits(environments, folds, seed)
    if adjust_environment:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        environment_features = encoder.fit_transform(environments.reshape(-1, 1))
    else:
        environment_features = np.empty((rows, 0), dtype=float)
    expected = np.zeros((experts, experts, rows), dtype=float)
    for source_index in range(experts):
        for target_index in range(experts):
            difficulty = observable_features if adjust_difficulty else np.zeros_like(observable_features)
            features = np.column_stack([difficulty, environment_features])
            labels = correctness[:, target_index].astype(int)
            predictions = np.zeros(rows, dtype=float)
            for train, test in splits:
                prevalence = (labels[train].sum() + 1.0) / (len(train) + 2.0)
                if len(np.unique(labels[train])) < 2:
                    predictions[test] = prevalence
                    continue
                scaler = StandardScaler().fit(features[train])
                classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                classifier.fit(scaler.transform(features[train]), labels[train])
                predictions[test] = classifier.predict_proba(scaler.transform(features[test]))[:, 1]
            expected[source_index, target_index] = np.clip(predictions, 1e-4, 1.0 - 1e-4)
    return expected, splits


def _standard_error(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def estimate_rescue_graphs(
    correctness: np.ndarray,
    expected: np.ndarray,
    environments: np.ndarray,
    expert_ids: tuple[str, ...],
    min_support: int = 8,
    min_environments: int = 3,
    sign_threshold: float = 0.8,
    shrinkage: float = 16.0,
    lcb_z: float = 1.0,
    self_loops: bool = False,
) -> tuple[dict[str, np.ndarray], list[RescueEdge]]:
    experts = correctness.shape[1]
    global_accuracy = correctness.mean(axis=0)
    graphs = {name: np.zeros((experts, experts), dtype=float) for name in ("raw_c", "c_minus_g", "residual", "stable")}
    edges: list[RescueEdge] = []
    environment_names = sorted(set(environments.tolist()))
    for source_index in range(experts):
        for target_index in range(experts):
            if source_index == target_index and not self_loops:
                continue
            failed = correctness[:, source_index] < 0.5
            residuals = correctness[:, target_index] - expected[source_index, target_index]
            values = residuals[failed]
            support = int(failed.sum())
            raw_c = float(correctness[failed, target_index].mean()) if support else 0.0
            baseline = float(expected[source_index, target_index][failed].mean()) if support else float(global_accuracy[target_index])
            residual_mean = float(values.mean()) if support else 0.0
            standard_error = _standard_error(values)
            pooled_lcb = residual_mean - lcb_z * standard_error
            environment_lcbs: list[float] = []
            environment_signs: dict[str, int] = {}
            for environment in environment_names:
                mask = failed & (environments == environment)
                count = int(mask.sum())
                if count < min_support:
                    continue
                env_values = residuals[mask]
                env_mean = float(env_values.mean())
                weight = count / (count + shrinkage)
                shrunk_mean = weight * env_mean + (1.0 - weight) * residual_mean
                env_lcb = shrunk_mean - lcb_z * _standard_error(env_values)
                environment_lcbs.append(env_lcb)
                environment_signs[str(environment)] = 1 if env_mean >= 0.0 else -1
            eligible = len(environment_lcbs)
            sign_consistency = (
                sum(value >= 0 for value in environment_signs.values()) / eligible if eligible else 0.0
            )
            q10 = float(np.quantile(environment_lcbs, 0.1)) if environment_lcbs else 0.0
            stable = (
                support >= min_support
                and eligible >= min_environments
                and pooled_lcb > 0.0
                and sign_consistency >= sign_threshold
                and q10 > 0.0
            )
            stable_weight = q10 if stable else 0.0
            if source_index != target_index or self_loops:
                graphs["raw_c"][source_index, target_index] = max(raw_c, 0.0)
                graphs["c_minus_g"][source_index, target_index] = max(raw_c - global_accuracy[target_index], 0.0)
                graphs["residual"][source_index, target_index] = max(residual_mean, 0.0)
                graphs["stable"][source_index, target_index] = stable_weight
            edges.append(
                RescueEdge(
                    source_expert=expert_ids[source_index],
                    target_expert=expert_ids[target_index],
                    raw_support=support,
                    raw_conditional_correctness=raw_c,
                    expected_baseline=baseline,
                    residual_mean=residual_mean,
                    standard_error=standard_error,
                    pooled_lcb=pooled_lcb,
                    eligible_environments=eligible,
                    sign_consistency=sign_consistency,
                    environment_lcb_q10=q10,
                    stable=stable,
                    stable_weight=stable_weight,
                    environment_signs=environment_signs,
                )
            )
    if not self_loops:
        for graph in graphs.values():
            np.fill_diagonal(graph, 0.0)
    return graphs, edges


def failure_feature_tensor(batch: ObservableQueryBatch, accuracy: dict[str, float]) -> np.ndarray:
    grouped = records_by_question(batch)
    rows: list[list[list[float]]] = []
    for question_id in batch.question_ids:
        records = grouped[question_id]
        features = expert_observable_features(records, accuracy)
        rows.append([features[expert][:7] for expert in batch.pool.expert_ids])
    return np.asarray(rows, dtype=float)


def fit_failure_models(
    features: np.ndarray,
    correctness: np.ndarray,
    seed: int,
) -> list[_BinaryFailureModel]:
    models: list[_BinaryFailureModel] = []
    for expert_index in range(correctness.shape[1]):
        labels = (1.0 - correctness[:, expert_index]).astype(int)
        constant = float((labels.sum() + 1.0) / (len(labels) + 2.0))
        if len(np.unique(labels)) < 2:
            models.append(_BinaryFailureModel(None, None, constant))
            continue
        scaler = StandardScaler().fit(features[:, expert_index, :])
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        classifier.fit(scaler.transform(features[:, expert_index, :]), labels)
        models.append(_BinaryFailureModel(scaler, classifier, constant))
    return models


def randomize_outgoing_edges(graph: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    randomized = np.zeros_like(graph)
    for source_index, row in enumerate(graph):
        weights = row[row > 0].copy()
        if not len(weights):
            continue
        candidates = np.asarray([index for index in range(len(row)) if index != source_index], dtype=int)
        destinations = rng.choice(candidates, size=len(weights), replace=False)
        rng.shuffle(weights)
        randomized[source_index, destinations] = weights
    return randomized


class DCRGSelector(Selector):
    name = "dcrg_stable"

    def __init__(
        self,
        seed: int = 20260808,
        folds: int = 5,
        min_support: int = 8,
        min_environments: int = 3,
        graph_mode: str = "stable",
        self_loops: bool = False,
        two_hop: bool = False,
        randomized_graph: bool = False,
        adjust_difficulty: bool = True,
        adjust_environment: bool = True,
    ) -> None:
        self.seed = seed
        self.folds = folds
        self.min_support = min_support
        self.min_environments = min_environments
        self.graph_mode = graph_mode
        self.self_loops = self_loops
        self.two_hop = two_hop
        self.randomized_graph = randomized_graph
        self.adjust_difficulty = adjust_difficulty
        self.adjust_environment = adjust_environment

    def fit(self, batch: ObservableQueryBatch, labels: SourceTrainingLabels) -> "DCRGSelector":
        if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
            raise TypeError("DCRG may be fitted only with SourceTrainingLabels")
        self.experts_ = batch.pool.expert_ids
        self.accuracy_ = source_accuracy(batch, labels)
        self.correctness_ = correctness_matrix(batch, labels)
        environments = np.asarray([labels.environment_by_question[qid] for qid in batch.question_ids], dtype=str)
        nuisance_ids, nuisance_features = topology_matrix(batch)
        if nuisance_ids != batch.question_ids:
            raise ValueError("Observable nuisance feature order is not aligned")
        self.expected_, self.oof_splits_ = cross_fitted_expected_correctness(
            self.correctness_,
            environments,
            nuisance_features,
            self.folds,
            self.seed,
            adjust_difficulty=self.adjust_difficulty,
            adjust_environment=self.adjust_environment,
        )
        self.graphs_, self.edges_ = estimate_rescue_graphs(
            self.correctness_,
            self.expected_,
            environments,
            self.experts_,
            min_support=self.min_support,
            min_environments=self.min_environments,
            self_loops=self.self_loops,
        )
        source_features = failure_feature_tensor(batch, self.accuracy_)
        self.failure_models_ = fit_failure_models(source_features, self.correctness_, self.seed)
        return self

    def edge_rows(self) -> list[dict]:
        return [asdict(edge) for edge in self.edges_]

    def predict_failure(self, batch: ObservableQueryBatch) -> np.ndarray:
        if batch.pool.expert_ids != self.experts_:
            raise ValueError("DCRG currently requires a fitted/masked common expert pool")
        features = failure_feature_tensor(batch, self.accuracy_)
        columns = [model.predict(features[:, index, :]) for index, model in enumerate(self.failure_models_)]
        return np.clip(np.column_stack(columns), 0.0, 1.0)

    def predict_with_mode(self, batch: ObservableQueryBatch, graph_mode: str) -> list[Selection]:
        if graph_mode not in self.graphs_:
            raise ValueError(f"Unknown graph mode: {graph_mode}")
        failure = self.predict_failure(batch)
        graph = self.graphs_[graph_mode]
        if self.randomized_graph:
            graph = randomize_outgoing_edges(graph, self.seed)
        rescue = failure @ graph
        if self.two_hop:
            rescue = rescue + rescue @ graph
        expert_scores = np.clip(1.0 - failure + rescue, 0.0, 1.0)
        if not np.isfinite(expert_scores).all():
            raise FloatingPointError("DCRG produced non-finite expert scores")
        grouped = records_by_question(batch)
        return [
            _selection_from_expert_scores(
                question_id,
                grouped[question_id],
                {expert: float(expert_scores[row_index, col]) for col, expert in enumerate(self.experts_)},
            )
            for row_index, question_id in enumerate(batch.question_ids)
        ]

    def predict(self, batch: ObservableQueryBatch) -> list[Selection]:
        return self.predict_with_mode(batch, self.graph_mode)
