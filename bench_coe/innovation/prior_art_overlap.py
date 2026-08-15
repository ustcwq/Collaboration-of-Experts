from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import records_by_question, topology_features
from .repair_simplification import RepairComponents
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import correctness_matrix, source_accuracy


CLASSIC_BASELINE_METHODS = (
    "global_best_posthoc",
    "majority_answer_support",
    "ola_metadata",
    "lca_support_class",
    "mcb_dcs_structured",
    "knop_output_profile",
    "knora_u_output_profile",
    "knora_e_output_profile",
    "meta_des_logistic",
)

RESPONSE_SELECTION_METHODS = (
    "more_style_structured",
    "more_style_minilm",
    "smoothie_global_spectral",
    "smoothie_local_spectral",
    "smoothie_global_minilm",
    "smoothie_local_minilm",
    "uncertainty_only",
    "agreement_x_global",
    "local_knn_only",
    "global_local_rank",
    "learned_logistic_selector",
    "learned_mlp_selector",
    "oprs_robust_output_profile",
)

EMBEDDING_RESPONSE_METHODS = (
    "more_style_minilm",
    "smoothie_global_minilm",
    "smoothie_local_minilm",
)

FCRG_METHODS = (
    "fcrg_full",
    "fcrg_g_only",
    "fcrg_a_only",
    "fcrg_l_only",
    "fcrg_column_mean_only",
    "fcrg_h1_only",
    "fcrg_h2_only",
    "fcrg_h1_h2",
    "fcrg_no_failure_conditioning",
    "fcrg_no_a_no_u",
    "fcrg_no_l",
    "fcrg_no_g",
    "fcrg_no_self",
    "fcrg_row_normalized",
    "fcrg_column_normalized",
    "fcrg_row_softmax",
    "fcrg_symmetric",
    "fcrg_random_edges",
    "fcrg_degree_relabel",
    "fcrg_depth_1",
    "fcrg_depth_2",
    "fcrg_depth_3",
    "fcrg_depth_4",
    "fcrg_depth_5",
)


@dataclass(frozen=True)
class ObservableMatrices:
    question_ids: tuple[str, ...]
    expert_ids: tuple[str, ...]
    profile: np.ndarray
    metadata: np.ndarray
    support: np.ndarray
    uncertainty: np.ndarray
    valid: np.ndarray
    output_length: np.ndarray
    family_breadth: np.ndarray


@dataclass(frozen=True)
class BaselineBundle:
    selections: Mapping[str, list[Selection]]
    diagnostics: Mapping[str, Any]


def _validate_source(
    train_batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    target_batch: ObservableQueryBatch,
) -> None:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Prior-art selectors may be fitted only with SourceTrainingLabels")
    if (labels.dataset, labels.split) != (train_batch.dataset, train_batch.split):
        raise ValueError("Source label provenance does not match the training observables")
    if train_batch.pool.expert_ids != target_batch.pool.expert_ids:
        raise ValueError("Prior-art selectors require the fitted expert pool")


def _scaled_uncertainty(value: float) -> float:
    return min(1.0, max(0.0, float(value) / 4.0))


def _query_state(batch: ObservableQueryBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = records_by_question(batch)
    rows = len(batch.question_ids)
    cols = len(batch.pool.expert_ids)
    support = np.zeros((rows, cols), dtype=float)
    uncertainty = np.zeros((rows, cols), dtype=float)
    valid = np.zeros((rows, cols), dtype=bool)
    output_length = np.zeros((rows, cols), dtype=float)
    family_breadth = np.zeros((rows, cols), dtype=float)
    expert_index = {expert: index for index, expert in enumerate(batch.pool.expert_ids)}
    for row_index, question_id in enumerate(batch.question_ids):
        records = grouped[question_id]
        counts = Counter(
            int(record.per_query_cluster_id)
            for record in records
            if record.valid_output and record.per_query_cluster_id is not None
        )
        families: dict[int, set[str]] = defaultdict(set)
        for record in records:
            if record.valid_output and record.per_query_cluster_id is not None:
                families[int(record.per_query_cluster_id)].add(record.expert_family)
        for record in records:
            col = expert_index[record.expert_id]
            uncertainty[row_index, col] = _scaled_uncertainty(record.uncertainty)
            output_length[row_index, col] = math.log1p(len(record.raw_output)) / 10.0
            if record.valid_output and record.per_query_cluster_id is not None:
                cluster_id = int(record.per_query_cluster_id)
                valid[row_index, col] = True
                support[row_index, col] = counts[cluster_id] / max(1, cols)
                family_breadth[row_index, col] = len(families[cluster_id]) / max(1, cols)
    return support, uncertainty, valid, output_length, family_breadth


def output_profile_matrix(batch: ObservableQueryBatch) -> np.ndarray:
    """Query-local output profile; raw answer identities never cross queries."""
    grouped = records_by_question(batch)
    experts = batch.pool.expert_ids
    rows: list[list[float]] = []
    for question_id in batch.question_ids:
        records = {record.expert_id: record for record in grouped[question_id]}
        topo = topology_features(records.values())
        row = [
            topo["valid_fraction"],
            topo["cluster_fraction"],
            topo["partition_entropy"],
            topo["top1_share"],
            topo["top2_share"],
            topo["cluster_margin"],
            topo["mean_uncertainty"] / 4.0,
            topo["std_uncertainty"] / 4.0,
        ]
        for expert in experts:
            record = records[expert]
            row.extend(
                [
                    1.0 if record.valid_output else 0.0,
                    _scaled_uncertainty(record.uncertainty),
                ]
            )
        for left in range(len(experts)):
            left_record = records[experts[left]]
            for right in range(left + 1, len(experts)):
                right_record = records[experts[right]]
                both_valid = left_record.valid_output and right_record.valid_output
                same = both_valid and left_record.per_query_cluster_id == right_record.per_query_cluster_id
                row.extend([1.0 if both_valid else 0.0, 1.0 if same else 0.0])
        rows.append(row)
    return np.nan_to_num(np.asarray(rows, dtype=float))


def _metadata_dicts(batch: ObservableQueryBatch) -> list[dict[str, float]]:
    grouped = records_by_question(batch)
    result: list[dict[str, float]] = []
    for question_id in batch.question_ids:
        records = grouped[question_id]
        representative = records[0]
        values: dict[str, float] = {}
        for key, value in sorted(representative.observable_metadata.items()):
            if isinstance(value, (list, tuple, set)):
                for member in value:
                    values[f"{key}={member}"] = 1.0
            else:
                values[f"{key}={value}"] = 1.0
        if not values:
            values["metadata=<missing>"] = 1.0
        result.append(values)
    return result


def observable_matrices(
    train_batch: ObservableQueryBatch,
    target_batch: ObservableQueryBatch,
) -> tuple[ObservableMatrices, ObservableMatrices]:
    vectorizer = DictVectorizer(sparse=False, sort=True)
    train_metadata = vectorizer.fit_transform(_metadata_dicts(train_batch))
    target_metadata = vectorizer.transform(_metadata_dicts(target_batch))

    def make(batch: ObservableQueryBatch, metadata: np.ndarray) -> ObservableMatrices:
        support, uncertainty, valid, output_length, family_breadth = _query_state(batch)
        return ObservableMatrices(
            question_ids=batch.question_ids,
            expert_ids=batch.pool.expert_ids,
            profile=output_profile_matrix(batch),
            metadata=np.nan_to_num(np.asarray(metadata, dtype=float)),
            support=support,
            uncertainty=uncertainty,
            valid=valid,
            output_length=output_length,
            family_breadth=family_breadth,
        )

    return make(train_batch, train_metadata), make(target_batch, target_metadata)


def nearest_rows(
    train: np.ndarray,
    target: np.ndarray,
    neighbors: int,
    *,
    exclude_aligned_self: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.ndim != 2 or target.ndim != 2 or train.shape[1] != target.shape[1]:
        raise ValueError("Nearest-neighbor matrices are not aligned")
    if len(train) == 0:
        raise ValueError("Nearest-neighbor source is empty")
    train_norm = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    target_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(target_norm @ train_norm.T, -1.0, 1.0)
    distance = 1.0 - similarity
    if exclude_aligned_self:
        if len(train) != len(target):
            raise ValueError("Aligned self-exclusion requires equal row counts")
        np.fill_diagonal(distance, np.inf)
    k = min(max(1, int(neighbors)), len(train) - int(exclude_aligned_self))
    if k <= 0:
        raise ValueError("At least two source rows are required for self-excluded neighbors")
    indices = np.argsort(distance, axis=1, kind="mergesort")[:, :k]
    selected_distance = np.take_along_axis(distance, indices, axis=1)
    weights = 1.0 / np.maximum(selected_distance, 1e-6)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    selected_similarity = np.take_along_axis(similarity, indices, axis=1)
    return indices, weights, selected_similarity


def local_competence(y: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("nk,nkm->nm", weights, y[indices])


def _rank_scores(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    width = values.shape[1]
    for row_index, row in enumerate(values):
        order = np.argsort(-row, kind="mergesort")
        ranks = np.empty(width, dtype=float)
        ranks[order] = np.arange(width, dtype=float)
        result[row_index] = 1.0 - ranks / max(1, width - 1)
    return result


def _cluster_selection(
    batch: ObservableQueryBatch,
    score_matrix: np.ndarray,
    method: str,
    *,
    reducer: str = "max",
    source_global: np.ndarray | None = None,
    extra_features: Mapping[str, Any] | None = None,
) -> list[Selection]:
    if score_matrix.shape != (len(batch.question_ids), len(batch.pool.expert_ids)):
        raise ValueError(f"Score matrix for {method} has invalid shape {score_matrix.shape}")
    grouped = records_by_question(batch)
    experts = batch.pool.expert_ids
    global_scores = np.zeros(len(experts), dtype=float) if source_global is None else source_global
    result: list[Selection] = []
    for row_index, question_id in enumerate(batch.question_ids):
        records = grouped[question_id]
        by_expert = {record.expert_id: record for record in records}
        valid_indices = [
            col
            for col, expert in enumerate(experts)
            if by_expert[expert].valid_output and by_expert[expert].per_query_cluster_id is not None
        ]
        features: dict[str, Any] = dict(topology_features(records))
        features.update(
            {
                "method": method,
                "valid_mask": {expert: bool(by_expert[expert].valid_output) for expert in experts},
                "missing_mask": {expert: not bool(by_expert[expert].valid_output) for expert in experts},
            }
        )
        if extra_features:
            features.update(extra_features)
        expert_scores = {expert: float(score_matrix[row_index, col]) for col, expert in enumerate(experts)}
        if not valid_indices:
            result.append(
                Selection(
                    question_id,
                    None,
                    None,
                    None,
                    {},
                    expert_scores,
                    "no_valid_output",
                    features,
                    "no_valid_output",
                )
            )
            continue
        by_cluster: dict[int, list[int]] = defaultdict(list)
        for col in valid_indices:
            cluster_id = int(by_expert[experts[col]].per_query_cluster_id)
            by_cluster[cluster_id].append(col)
        cluster_scores: dict[int, float] = {}
        for cluster_id, members in by_cluster.items():
            values = score_matrix[row_index, members]
            if reducer == "sum":
                cluster_scores[cluster_id] = float(values.sum())
            elif reducer == "mean":
                cluster_scores[cluster_id] = float(values.mean())
            else:
                cluster_scores[cluster_id] = float(values.max())
        selected_cluster = sorted(cluster_scores, key=lambda key: (-cluster_scores[key], key))[0]
        selected_col = sorted(
            by_cluster[selected_cluster],
            key=lambda col: (-score_matrix[row_index, col], -global_scores[col], experts[col]),
        )[0]
        selected_expert = experts[selected_col]
        result.append(
            Selection(
                question_id=question_id,
                selected_cluster_id=selected_cluster,
                selected_expert_id=selected_expert,
                normalized_answer=by_expert[selected_expert].normalized_answer,
                cluster_scores={str(key): float(value) for key, value in sorted(cluster_scores.items())},
                expert_scores=expert_scores,
                fallback_reason=None,
                observable_features=features,
                tie_breaking=(
                    f"cluster_{reducer}_score_then_cluster_id; "
                    "expert_score_then_source_global_then_expert_id"
                ),
            )
        )
    return result


def fixed_global_best_selections(
    batch: ObservableQueryBatch,
    global_accuracy: np.ndarray,
    *,
    method: str = "fast_global_best_single_call",
) -> list[Selection]:
    experts = batch.pool.expert_ids
    chosen_col = sorted(range(len(experts)), key=lambda col: (-global_accuracy[col], experts[col]))[0]
    chosen_expert = experts[chosen_col]
    grouped = records_by_question(batch)
    result: list[Selection] = []
    for question_id in batch.question_ids:
        records = grouped[question_id]
        by_expert = {record.expert_id: record for record in records}
        record = by_expert[chosen_expert]
        valid = bool(record.valid_output and record.per_query_cluster_id is not None)
        features = dict(topology_features(records))
        features.update(
            {
                "method": method,
                "single_called_expert": chosen_expert,
                "valid_mask": {expert: bool(by_expert[expert].valid_output) for expert in experts},
                "missing_mask": {expert: not bool(by_expert[expert].valid_output) for expert in experts},
            }
        )
        result.append(
            Selection(
                question_id=question_id,
                selected_cluster_id=int(record.per_query_cluster_id) if valid else None,
                selected_expert_id=chosen_expert,
                normalized_answer=record.normalized_answer if valid else None,
                cluster_scores={str(record.per_query_cluster_id): float(global_accuracy[chosen_col])} if valid else {},
                expert_scores={expert: float(global_accuracy[col]) for col, expert in enumerate(experts)},
                fallback_reason=None if valid else "single_called_expert_missing",
                observable_features=features,
                tie_breaking="source_global_then_expert_id_fixed_before_query",
            )
        )
    return result


def _agreement_quality(batch: ObservableQueryBatch) -> np.ndarray:
    grouped = records_by_question(batch)
    experts = batch.pool.expert_ids
    numerator = np.zeros((len(experts), len(experts)), dtype=float)
    denominator = np.zeros_like(numerator)
    for question_id in batch.question_ids:
        by_expert = {record.expert_id: record for record in grouped[question_id]}
        for left, left_expert in enumerate(experts):
            for right in range(left + 1, len(experts)):
                right_expert = experts[right]
                a = by_expert[left_expert]
                b = by_expert[right_expert]
                if not (a.valid_output and b.valid_output):
                    continue
                denominator[left, right] += 1.0
                denominator[right, left] += 1.0
                if a.per_query_cluster_id == b.per_query_cluster_id:
                    numerator[left, right] += 1.0
                    numerator[right, left] += 1.0
    agreement = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    if len(experts) == 1:
        return np.ones(1, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(agreement)
    quality = np.abs(eigenvectors[:, int(np.argmax(eigenvalues))])
    if quality.sum() <= 1e-12:
        quality = np.ones(len(experts), dtype=float)
    return quality / quality.sum()


def _spectral_quality(agreement: np.ndarray) -> np.ndarray:
    if len(agreement) == 1:
        return np.ones(1, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(np.nan_to_num(agreement))
    quality = np.abs(eigenvectors[:, int(np.argmax(eigenvalues))])
    if quality.sum() <= 1e-12:
        quality = np.ones(len(agreement), dtype=float)
    return quality / quality.sum()


def response_embedding_observables(
    batch: ObservableQueryBatch,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = len(batch.question_ids)
    cols = len(batch.pool.expert_ids)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (rows, cols):
        raise ValueError("Response embeddings do not align with the observable batch")
    _, _, valid, _, _ = _query_state(batch)
    grouped = records_by_question(batch)
    expert_features = np.zeros((rows, cols, 4), dtype=float)
    profiles: list[np.ndarray] = []
    source_agreement = np.zeros((cols, cols), dtype=float)
    source_counts = np.zeros((cols, cols), dtype=float)
    for row_index, question_id in enumerate(batch.question_ids):
        similarity = np.clip(embeddings[row_index] @ embeddings[row_index].T, -1.0, 1.0)
        pair_valid = valid[row_index, :, None] & valid[row_index, None, :]
        similarity = np.where(pair_valid, similarity, 0.0)
        source_agreement += similarity * pair_valid
        source_counts += pair_valid
        by_expert = {record.expert_id: record for record in grouped[question_id]}
        profile_values: list[float] = []
        for left in range(cols):
            for right in range(left + 1, cols):
                profile_values.extend(
                    [float(pair_valid[left, right]), float(similarity[left, right])]
                )
        profiles.append(np.asarray(profile_values, dtype=float))
        for col, expert in enumerate(batch.pool.expert_ids):
            peers = [index for index in range(cols) if index != col and pair_valid[col, index]]
            if not peers:
                continue
            values = similarity[col, peers]
            record = by_expert[expert]
            same_cluster = [
                index
                for index in peers
                if by_expert[batch.pool.expert_ids[index]].per_query_cluster_id
                == record.per_query_cluster_id
            ]
            expert_features[row_index, col] = np.asarray(
                [
                    float(values.mean()),
                    float(values.max()),
                    float(values.std()),
                    float(similarity[col, same_cluster].mean()) if same_cluster else 0.0,
                ]
            )
    agreement = np.divide(
        source_agreement,
        source_counts,
        out=np.zeros_like(source_agreement),
        where=source_counts > 0,
    )
    np.fill_diagonal(agreement, 0.0)
    profile = np.nan_to_num(np.stack(profiles))
    return np.nan_to_num(expert_features), profile, agreement


def _fit_binary_model(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[Any | None, float]:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None, float(y[0]) if len(y) else 0.0
    if kind == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="tanh",
                solver="lbfgs",
                alpha=0.01,
                max_iter=500,
                random_state=seed,
            ),
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    model.fit(np.nan_to_num(x), y)
    return model, float(y.mean())


def _positive_probability(model: Any | None, constant: float, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), constant, dtype=float)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    positive = classes.index(1)
    return np.asarray(model.predict_proba(np.nan_to_num(x))[:, positive], dtype=float)


def _expert_rows(
    matrices: ObservableMatrices,
    global_rows: np.ndarray,
    local_metadata: np.ndarray,
    local_profile: np.ndarray,
    *,
    include_metadata: bool,
    include_profile: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    expert_count = len(matrices.expert_ids)
    for row_index in range(len(matrices.question_ids)):
        for col in range(expert_count):
            base = np.asarray(
                [
                    global_rows[row_index, col],
                    local_metadata[row_index, col],
                    local_profile[row_index, col],
                    matrices.support[row_index, col],
                    1.0 - matrices.uncertainty[row_index, col],
                    float(matrices.valid[row_index, col]),
                    matrices.output_length[row_index, col],
                    matrices.family_breadth[row_index, col],
                ],
                dtype=float,
            )
            one_hot = np.zeros(expert_count, dtype=float)
            one_hot[col] = 1.0
            parts = [base, one_hot]
            if include_metadata:
                parts.append(matrices.metadata[row_index])
            if include_profile:
                parts.append(matrices.profile[row_index])
            rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=float)


def _scores_from_flat(values: np.ndarray, row_count: int, expert_count: int) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(row_count, expert_count)


def fit_predict_prior_art_baselines(
    train_batch: ObservableQueryBatch,
    train_labels: SourceTrainingLabels,
    target_batch: ObservableQueryBatch,
    *,
    neighbors: int,
    seed: int,
    mcb_behavior_threshold: float = 0.70,
    mcb_min_neighbors: int = 5,
    include_mlp: bool = True,
    response_encoder: Any | None = None,
) -> BaselineBundle:
    _validate_source(train_batch, train_labels, target_batch)
    train, target = observable_matrices(train_batch, target_batch)
    y = correctness_matrix(train_batch, train_labels)
    global_vector = np.asarray([source_accuracy(train_batch, train_labels)[expert] for expert in train.expert_ids])
    global_target = np.broadcast_to(global_vector, (len(target.question_ids), len(target.expert_ids))).copy()
    global_loo = np.divide(
        y.sum(axis=0, keepdims=True) - y,
        max(1, len(y) - 1),
    )

    meta_target_idx, meta_target_w, _ = nearest_rows(train.metadata, target.metadata, neighbors)
    profile_target_idx, profile_target_w, profile_target_similarity = nearest_rows(
        train.profile, target.profile, neighbors
    )
    meta_train_idx, meta_train_w, _ = nearest_rows(
        train.metadata, train.metadata, neighbors, exclude_aligned_self=True
    )
    profile_train_idx, profile_train_w, _ = nearest_rows(
        train.profile, train.profile, neighbors, exclude_aligned_self=True
    )
    ola_target = local_competence(y, meta_target_idx, meta_target_w)
    ola_train = local_competence(y, meta_train_idx, meta_train_w)
    knop_target = local_competence(y, profile_target_idx, profile_target_w)
    knop_train = local_competence(y, profile_train_idx, profile_train_w)

    selections: dict[str, list[Selection]] = {}
    selections["global_best_posthoc"] = _cluster_selection(
        target_batch, global_target, "global_best_posthoc", source_global=global_vector
    )
    selections["fast_global_best_single_call"] = fixed_global_best_selections(target_batch, global_vector)
    selections["majority_answer_support"] = _cluster_selection(
        target_batch,
        np.ones_like(global_target),
        "majority_answer_support",
        reducer="sum",
        source_global=global_vector,
    )
    selections["ola_metadata"] = _cluster_selection(
        target_batch, ola_target, "ola_metadata", source_global=global_vector
    )

    lca_scores = np.zeros_like(ola_target)
    train_support_count = np.rint(train.support * len(train.expert_ids)).astype(int)
    target_support_count = np.rint(target.support * len(target.expert_ids)).astype(int)
    for row_index in range(len(target.question_ids)):
        candidates = meta_target_idx[row_index]
        candidate_weights = meta_target_w[row_index]
        for col in range(len(target.expert_ids)):
            matching = train_support_count[candidates, col] == target_support_count[row_index, col]
            matching &= train.valid[candidates, col] == target.valid[row_index, col]
            if not matching.any():
                matching = np.ones(len(candidates), dtype=bool)
            weights = candidate_weights[matching]
            weights = weights / max(weights.sum(), 1e-12)
            lca_scores[row_index, col] = float(np.dot(weights, y[candidates[matching], col]))
    selections["lca_support_class"] = _cluster_selection(
        target_batch, lca_scores, "lca_support_class", source_global=global_vector
    )

    mcb_scores = np.zeros_like(ola_target)
    wide_k = min(len(train.question_ids), max(neighbors * 2, neighbors))
    mcb_idx, mcb_w, _ = nearest_rows(train.metadata, target.metadata, wide_k)
    train_profile_norm = train.profile / np.maximum(np.linalg.norm(train.profile, axis=1, keepdims=True), 1e-12)
    target_profile_norm = target.profile / np.maximum(np.linalg.norm(target.profile, axis=1, keepdims=True), 1e-12)
    for row_index in range(len(target.question_ids)):
        candidates = mcb_idx[row_index]
        behavior_similarity = train_profile_norm[candidates] @ target_profile_norm[row_index]
        keep = behavior_similarity >= mcb_behavior_threshold
        minimum = min(mcb_min_neighbors, len(candidates))
        if int(keep.sum()) < minimum:
            order = np.argsort(-behavior_similarity, kind="mergesort")
            keep = np.zeros(len(candidates), dtype=bool)
            keep[order[:minimum]] = True
        selected = candidates[keep]
        weights = mcb_w[row_index][keep] * np.maximum(behavior_similarity[keep] + 1.0, 1e-6)
        weights /= max(weights.sum(), 1e-12)
        mcb_scores[row_index] = weights @ y[selected]
    selections["mcb_dcs_structured"] = _cluster_selection(
        target_batch, mcb_scores, "mcb_dcs_structured", source_global=global_vector
    )
    selections["knop_output_profile"] = _cluster_selection(
        target_batch, knop_target, "knop_output_profile", source_global=global_vector
    )
    selections["local_knn_only"] = _cluster_selection(
        target_batch, knop_target, "local_knn_only", source_global=global_vector
    )

    knora_u = np.asarray(
        [(y[index].sum(axis=0) > 0.0).astype(float) for index in profile_target_idx],
        dtype=float,
    )
    selections["knora_u_output_profile"] = _cluster_selection(
        target_batch,
        knora_u,
        "knora_u_output_profile",
        reducer="sum",
        source_global=global_vector,
    )
    knora_e = np.zeros_like(knop_target)
    knora_e_fallbacks = 0
    for row_index, neighbor_indices in enumerate(profile_target_idx):
        found = False
        for size in range(len(neighbor_indices), 0, -1):
            perfect = np.all(y[neighbor_indices[:size]] > 0.5, axis=0)
            if perfect.any():
                knora_e[row_index] = perfect.astype(float)
                found = True
                break
        if not found:
            knora_e[row_index] = global_vector
            knora_e_fallbacks += 1
    selections["knora_e_output_profile"] = _cluster_selection(
        target_batch,
        knora_e,
        "knora_e_output_profile",
        reducer="sum",
        source_global=global_vector,
    )

    meta_train_x = _expert_rows(
        train,
        global_loo,
        ola_train,
        knop_train,
        include_metadata=False,
        include_profile=False,
    )
    meta_target_x = _expert_rows(
        target,
        global_target,
        ola_target,
        knop_target,
        include_metadata=False,
        include_profile=False,
    )
    labels_flat = y.reshape(-1).astype(int)
    meta_model, meta_constant = _fit_binary_model("logistic", meta_train_x, labels_flat, seed)
    meta_scores = _scores_from_flat(
        _positive_probability(meta_model, meta_constant, meta_target_x), len(target.question_ids), len(target.expert_ids)
    )
    selections["meta_des_logistic"] = _cluster_selection(
        target_batch, meta_scores, "meta_des_logistic", source_global=global_vector
    )

    more_train_x = _expert_rows(
        train,
        global_loo,
        np.zeros_like(ola_train),
        np.zeros_like(knop_train),
        include_metadata=False,
        include_profile=False,
    )
    more_target_x = _expert_rows(
        target,
        global_target,
        np.zeros_like(ola_target),
        np.zeros_like(knop_target),
        include_metadata=False,
        include_profile=False,
    )
    more_model, more_constant = _fit_binary_model("logistic", more_train_x, labels_flat, seed)
    more_scores = _scores_from_flat(
        _positive_probability(more_model, more_constant, more_target_x),
        len(target.question_ids),
        len(target.expert_ids),
    )
    selections["more_style_structured"] = _cluster_selection(
        target_batch, more_scores, "more_style_structured", source_global=global_vector
    )

    spectral_global = _agreement_quality(train_batch)
    spectral_matrix = np.broadcast_to(spectral_global, global_target.shape).copy()
    selections["smoothie_global_spectral"] = _cluster_selection(
        target_batch,
        spectral_matrix,
        "smoothie_global_spectral",
        source_global=global_vector,
    )
    if len(train.expert_ids) > 1:
        source_peer_agreement = np.clip(
            (train.support * len(train.expert_ids) - train.valid.astype(float)) / (len(train.expert_ids) - 1),
            0.0,
            1.0,
        )
    else:
        source_peer_agreement = train.valid.astype(float)
    smoothie_local = local_competence(source_peer_agreement, profile_target_idx, profile_target_w)
    selections["smoothie_local_spectral"] = _cluster_selection(
        target_batch, smoothie_local, "smoothie_local_spectral", source_global=global_vector
    )

    embedding_diagnostics: dict[str, Any] | None = None
    if response_encoder is not None:
        train_embeddings = response_encoder.encode_batch(train_batch)
        target_embeddings = response_encoder.encode_batch(target_batch)
        train_embedding_features, train_embedding_profile, train_semantic_agreement = (
            response_embedding_observables(train_batch, train_embeddings)
        )
        target_embedding_features, target_embedding_profile, _ = response_embedding_observables(
            target_batch, target_embeddings
        )
        more_minilm_train_x = np.concatenate(
            [more_train_x, train_embedding_features.reshape(-1, train_embedding_features.shape[2])],
            axis=1,
        )
        more_minilm_target_x = np.concatenate(
            [more_target_x, target_embedding_features.reshape(-1, target_embedding_features.shape[2])],
            axis=1,
        )
        more_minilm_model, more_minilm_constant = _fit_binary_model(
            "logistic", more_minilm_train_x, labels_flat, seed
        )
        more_minilm_scores = _scores_from_flat(
            _positive_probability(more_minilm_model, more_minilm_constant, more_minilm_target_x),
            len(target.question_ids),
            len(target.expert_ids),
        )
        selections["more_style_minilm"] = _cluster_selection(
            target_batch, more_minilm_scores, "more_style_minilm", source_global=global_vector
        )

        semantic_global = _spectral_quality(train_semantic_agreement)
        selections["smoothie_global_minilm"] = _cluster_selection(
            target_batch,
            np.broadcast_to(semantic_global, global_target.shape).copy(),
            "smoothie_global_minilm",
            source_global=global_vector,
        )
        semantic_train_profile = np.concatenate([train.profile, train_embedding_profile], axis=1)
        semantic_target_profile = np.concatenate([target.profile, target_embedding_profile], axis=1)
        semantic_indices, semantic_weights, _ = nearest_rows(
            semantic_train_profile, semantic_target_profile, neighbors
        )
        smoothie_local_minilm = local_competence(
            train_embedding_features[:, :, 0], semantic_indices, semantic_weights
        )
        selections["smoothie_local_minilm"] = _cluster_selection(
            target_batch,
            smoothie_local_minilm,
            "smoothie_local_minilm",
            source_global=global_vector,
        )
        embedding_diagnostics = {
            **response_encoder.diagnostics(),
            "expert_feature_order": [
                "mean_peer_cosine",
                "max_peer_cosine",
                "std_peer_cosine",
                "same_answer_cluster_peer_cosine",
            ],
            "semantic_profile_dimensions": int(semantic_train_profile.shape[1]),
            "smoothie_global_quality": {
                expert: float(semantic_global[col]) for col, expert in enumerate(train.expert_ids)
            },
        }

    uncertainty_scores = 1.0 - target.uncertainty
    selections["uncertainty_only"] = _cluster_selection(
        target_batch, uncertainty_scores, "uncertainty_only", source_global=global_vector
    )
    selections["agreement_x_global"] = _cluster_selection(
        target_batch,
        target.support * global_target,
        "agreement_x_global",
        source_global=global_vector,
    )
    rank_scores = 0.5 * _rank_scores(knop_target) + 0.5 * _rank_scores(global_target)
    selections["global_local_rank"] = _cluster_selection(
        target_batch, rank_scores, "global_local_rank", source_global=global_vector
    )

    learned_train_x = _expert_rows(
        train,
        global_loo,
        ola_train,
        knop_train,
        include_metadata=True,
        include_profile=True,
    )
    learned_target_x = _expert_rows(
        target,
        global_target,
        ola_target,
        knop_target,
        include_metadata=True,
        include_profile=True,
    )
    learned_logistic, learned_constant = _fit_binary_model("logistic", learned_train_x, labels_flat, seed)
    logistic_scores = _scores_from_flat(
        _positive_probability(learned_logistic, learned_constant, learned_target_x),
        len(target.question_ids),
        len(target.expert_ids),
    )
    selections["learned_logistic_selector"] = _cluster_selection(
        target_batch, logistic_scores, "learned_logistic_selector", source_global=global_vector
    )
    if include_mlp:
        learned_mlp, mlp_constant = _fit_binary_model("mlp", learned_train_x, labels_flat, seed)
        mlp_scores = _scores_from_flat(
            _positive_probability(learned_mlp, mlp_constant, learned_target_x),
            len(target.question_ids),
            len(target.expert_ids),
        )
        selections["learned_mlp_selector"] = _cluster_selection(
            target_batch, mlp_scores, "learned_mlp_selector", source_global=global_vector
        )

    diagnostics = {
        "adaptation_scope": {
            "ola_metadata": "sanitized categorical query metadata; no raw question embedding available",
            "lca_support_class": "query-local answer-support count substitutes for cross-query class identity",
            "mcb_dcs_structured": "metadata neighborhood filtered by output-profile cosine similarity",
            "knop_output_profile": "pairwise expert agreement, validity, uncertainty, and topology profile",
            "meta_des_logistic": "fixed logistic competence meta-classifier; not an official META-DES model",
            "more_style_structured": "agreement/confidence/output-statistics adaptation; no answer embedding",
            "more_style_minilm": "source-fitted answer selector with local MiniLM response-similarity features",
            "smoothie_global_spectral": "label-free spectral pairwise-agreement adaptation",
            "smoothie_local_spectral": "label-free local peer-agreement adaptation",
            "smoothie_global_minilm": "label-free spectral quality from MiniLM response similarity",
            "smoothie_local_minilm": "label-free output-neighborhood quality from MiniLM response similarity",
        },
        "neighbors": int(neighbors),
        "mcb_behavior_threshold": float(mcb_behavior_threshold),
        "mcb_min_neighbors": int(mcb_min_neighbors),
        "knora_e_fallback_count": int(knora_e_fallbacks),
        "spectral_global_quality": {
            expert: float(spectral_global[col]) for col, expert in enumerate(train.expert_ids)
        },
        "lexical_uncertainty_nonzero_fraction": float(np.mean(target.uncertainty > 0.0)),
        "profile_dimensions": int(train.profile.shape[1]),
        "metadata_dimensions": int(train.metadata.shape[1]),
        "response_embedding": embedding_diagnostics,
    }
    return BaselineBundle(selections=selections, diagnostics=diagnostics)


def knop_sensitivity_selections(
    train_batch: ObservableQueryBatch,
    train_labels: SourceTrainingLabels,
    target_batch: ObservableQueryBatch,
    neighbor_values: Sequence[int],
) -> dict[str, list[Selection]]:
    _validate_source(train_batch, train_labels, target_batch)
    train, target = observable_matrices(train_batch, target_batch)
    y = correctness_matrix(train_batch, train_labels)
    global_vector = np.asarray([source_accuracy(train_batch, train_labels)[expert] for expert in train.expert_ids])
    result: dict[str, list[Selection]] = {}
    for neighbors in sorted({int(value) for value in neighbor_values}):
        indices, weights, _ = nearest_rows(train.profile, target.profile, neighbors)
        scores = local_competence(y, indices, weights)
        method = f"knop_k{neighbors}"
        result[method] = _cluster_selection(
            target_batch,
            scores,
            method,
            source_global=global_vector,
            extra_features={"knn_k": neighbors},
        )
    return result


def graph_variant(graph: np.ndarray, mode: str, seed: int) -> np.ndarray:
    graph = np.nan_to_num(np.asarray(graph, dtype=float))
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("FCRG graph must be square")
    if mode == "raw":
        result = graph.copy()
    elif mode == "no_self":
        result = graph.copy()
        np.fill_diagonal(result, 0.0)
    elif mode == "row_normalized":
        result = np.divide(graph, graph.sum(axis=1, keepdims=True), out=np.zeros_like(graph), where=graph.sum(axis=1, keepdims=True) > 1e-12)
    elif mode == "column_normalized":
        result = np.divide(graph, graph.sum(axis=0, keepdims=True), out=np.zeros_like(graph), where=graph.sum(axis=0, keepdims=True) > 1e-12)
    elif mode == "row_softmax":
        shifted = graph - graph.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        result = exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)
    elif mode == "symmetric":
        result = 0.5 * (graph + graph.T)
    elif mode == "random_edges":
        rng = np.random.default_rng(seed)
        result = graph.copy()
        for source in range(len(graph)):
            destinations = np.asarray([index for index in range(len(graph)) if index != source], dtype=int)
            result[source, destinations] = rng.permutation(graph[source, destinations])
    elif mode == "degree_relabel":
        permutation = np.random.default_rng(seed).permutation(len(graph))
        result = graph[np.ix_(permutation, permutation)]
    elif mode == "column_mean":
        result = np.broadcast_to(graph.mean(axis=0), graph.shape).copy()
    else:
        raise ValueError(f"Unknown FCRG graph mode: {mode}")
    return np.nan_to_num(result)


def repair_hop_sequence(
    support: np.ndarray,
    uncertainty: np.ndarray,
    graph: np.ndarray,
    *,
    max_hops: int,
    failure_mode: str = "query",
    device: Any | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if max_hops < 1:
        raise ValueError("FCRG propagation requires at least one hop")
    if failure_mode == "uniform":
        weights = np.full_like(support, 1.0 / max(1, support.shape[1]))
    elif failure_mode == "query":
        failure = np.clip(1.0 - support + 0.40 * uncertainty, 0.0, 1.6)
        denominator = failure.sum(axis=1, keepdims=True)
        weights = np.divide(
            failure,
            denominator,
            out=np.full_like(failure, 1.0 / max(1, failure.shape[1])),
            where=denominator > 1e-12,
        )
    else:
        raise ValueError(f"Unknown failure mode: {failure_mode}")
    if device is None:
        current = weights
        hops: list[np.ndarray] = []
        for _ in range(max_hops):
            propagated = current @ graph
            hops.append(np.nan_to_num(propagated))
            current = propagated / np.maximum(propagated.sum(axis=1, keepdims=True), 1e-12)
        return weights, tuple(hops)

    import torch

    current_tensor = torch.as_tensor(weights, dtype=torch.float64, device=device)
    graph_tensor = torch.as_tensor(graph, dtype=torch.float64, device=device)
    tensor_hops = []
    for _ in range(max_hops):
        propagated_tensor = current_tensor @ graph_tensor
        tensor_hops.append(propagated_tensor)
        current_tensor = propagated_tensor / torch.clamp(
            propagated_tensor.sum(dim=1, keepdim=True), min=1e-12
        )
    return weights, tuple(np.nan_to_num(value.detach().cpu().numpy()) for value in tensor_hops)


def depth_weights(depth: int, total: float = 0.43, decay: float = 0.72) -> np.ndarray:
    if depth < 1:
        raise ValueError("Depth must be positive")
    raw = np.asarray([decay**index for index in range(depth)], dtype=float)
    return total * raw / raw.sum()


def fcrg_score_matrix(
    method: str,
    components: RepairComponents,
    *,
    graph_mode: str = "raw",
    seed: int = 0,
    device: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    graph = graph_variant(components.repair_graph, graph_mode, seed)
    failure_mode = "uniform" if method in {"fcrg_no_failure_conditioning", "fcrg_no_a_no_u"} else "query"
    depth = int(method.rsplit("_", 1)[1]) if method.startswith("fcrg_depth_") else 5
    weights, hops = repair_hop_sequence(
        components.support,
        components.uncertainty,
        graph,
        max_hops=max(2, depth),
        failure_mode=failure_mode,
        device=device,
    )
    local = components.local
    support = components.support
    global_scores = np.broadcast_to(components.global_accuracy, local.shape)
    hop1, hop2 = hops[0], hops[1]
    if method == "fcrg_g_only":
        score = global_scores
    elif method == "fcrg_a_only":
        score = support
    elif method == "fcrg_l_only":
        score = local
    elif method == "fcrg_column_mean_only":
        score = np.broadcast_to(components.repair_graph.mean(axis=0), local.shape).copy()
    elif method == "fcrg_h1_only":
        score = hop1
    elif method == "fcrg_h2_only":
        score = hop2
    elif method == "fcrg_h1_h2":
        score = 0.25 * hop1 + 0.18 * hop2
    elif method == "fcrg_no_a_no_u":
        score = 0.30 * local + 0.25 * hop1 + 0.18 * hop2 + 0.11 * global_scores
    elif method == "fcrg_no_l":
        score = 0.25 * hop1 + 0.18 * hop2 + 0.16 * support + 0.11 * global_scores
    elif method == "fcrg_no_g":
        score = 0.30 * local + 0.25 * hop1 + 0.18 * hop2 + 0.16 * support
    elif method.startswith("fcrg_depth_"):
        hop_weights = depth_weights(depth)
        graph_score = sum(hop_weights[index] * hops[index] for index in range(depth))
        score = 0.30 * local + graph_score + 0.16 * support + 0.11 * global_scores
    else:
        score = 0.30 * local + 0.25 * hop1 + 0.18 * hop2 + 0.16 * support + 0.11 * global_scores
    return np.nan_to_num(score), {
        "graph_mode": graph_mode,
        "failure_mode": failure_mode,
        "depth": depth if method.startswith("fcrg_depth_") else 2,
        "row_sum_min": float(graph.sum(axis=1).min()),
        "row_sum_max": float(graph.sum(axis=1).max()),
        "column_sum_min": float(graph.sum(axis=0).min()),
        "column_sum_max": float(graph.sum(axis=0).max()),
        "diagonal_mean": float(np.diag(graph).mean()),
        "failure_weight_row_sum_error": float(np.abs(weights.sum(axis=1) - 1.0).max()),
    }


def fcrg_ablation_selections(
    batch: ObservableQueryBatch,
    components: RepairComponents,
    *,
    seed: int,
    device: Any | None,
) -> tuple[dict[str, list[Selection]], dict[str, Any]]:
    graph_by_method = {
        "fcrg_no_self": "no_self",
        "fcrg_row_normalized": "row_normalized",
        "fcrg_column_normalized": "column_normalized",
        "fcrg_row_softmax": "row_softmax",
        "fcrg_symmetric": "symmetric",
        "fcrg_random_edges": "random_edges",
        "fcrg_degree_relabel": "degree_relabel",
    }
    selections: dict[str, list[Selection]] = {}
    diagnostics: dict[str, Any] = {}
    for method in FCRG_METHODS:
        graph_mode = graph_by_method.get(method, "raw")
        scores, method_diagnostics = fcrg_score_matrix(
            method,
            components,
            graph_mode=graph_mode,
            seed=seed,
            device=device,
        )
        selections[method] = _cluster_selection(
            batch,
            scores,
            method,
            source_global=components.global_accuracy,
            extra_features={"graph_mode": graph_mode, "propagation_depth": method_diagnostics["depth"]},
        )
        diagnostics[method] = method_diagnostics
    return selections, diagnostics


def learned_fcrg_selections(
    batch: ObservableQueryBatch,
    components: RepairComponents,
    model: Any | None,
    constant: float,
) -> list[Selection]:
    feature_cube = np.stack(
        [
            components.local,
            components.hop1,
            components.hop2,
            components.support,
            np.broadcast_to(components.global_accuracy, components.local.shape),
        ],
        axis=2,
    )
    scores = _scores_from_flat(
        _positive_probability(model, constant, feature_cube.reshape(-1, feature_cube.shape[2])),
        len(components.question_ids),
        len(components.expert_ids),
    )
    return _cluster_selection(
        batch,
        scores,
        "fcrg_learned_weights",
        source_global=components.global_accuracy,
        extra_features={"weight_learning": "inner-environment OOF logistic competence model"},
    )


def fit_fcrg_weight_model(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[Any | None, float, dict[str, Any]]:
    model, constant = _fit_binary_model("logistic", features, labels, seed)
    diagnostics: dict[str, Any] = {"constant": constant, "feature_order": ["L", "H1", "H2", "A", "G"]}
    if model is not None:
        logistic = model[-1]
        diagnostics["standardized_coefficients"] = [float(value) for value in logistic.coef_[0]]
        diagnostics["intercept"] = float(logistic.intercept_[0])
    return model, constant, diagnostics


def fcrg_feature_rows(components: RepairComponents) -> np.ndarray:
    return np.stack(
        [
            components.local,
            components.hop1,
            components.hop2,
            components.support,
            np.broadcast_to(components.global_accuracy, components.local.shape),
        ],
        axis=2,
    ).reshape(-1, 5)


def cascade_selections(
    batch: ObservableQueryBatch,
    fast: Sequence[Selection],
    full: Sequence[Selection],
    threshold: float,
) -> tuple[list[Selection], dict[str, Any]]:
    fast_by_id = {selection.question_id: selection for selection in fast}
    full_by_id = {selection.question_id: selection for selection in full}
    grouped = records_by_question(batch)
    result: list[Selection] = []
    triggered = 0
    for question_id in batch.question_ids:
        fast_selection = fast_by_id[question_id]
        by_expert = {record.expert_id: record for record in grouped[question_id]}
        fast_record = by_expert[fast_selection.selected_expert_id or ""]
        trigger = not fast_record.valid_output or _scaled_uncertainty(fast_record.uncertainty) > threshold
        chosen = full_by_id[question_id] if trigger else fast_selection
        triggered += int(trigger)
        features = dict(chosen.observable_features)
        features.update(
            {
                "cascade_threshold": float(threshold),
                "cascade_triggered": bool(trigger),
                "fast_expert_uncertainty": _scaled_uncertainty(fast_record.uncertainty),
            }
        )
        result.append(
            Selection(
                question_id=chosen.question_id,
                selected_cluster_id=chosen.selected_cluster_id,
                selected_expert_id=chosen.selected_expert_id,
                normalized_answer=chosen.normalized_answer,
                cluster_scores=chosen.cluster_scores,
                expert_scores=chosen.expert_scores,
                fallback_reason="cascade_full_repair" if trigger else "cascade_fast_path",
                observable_features=features,
                tie_breaking=chosen.tie_breaking,
            )
        )
    fraction = triggered / max(1, len(batch.question_ids))
    return result, {
        "threshold": float(threshold),
        "trigger_count": triggered,
        "trigger_fraction": fraction,
        "mean_nominal_calls": 1.0 + fraction * (len(batch.pool.expert_ids) - 1),
    }
