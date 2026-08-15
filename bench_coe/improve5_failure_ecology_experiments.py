from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.improve2_capability_routing_experiments import (
    CASES,
    CaseSpec,
    augmented_text,
    bool_matrix,
    build_vectorizer,
    choices_from_scores,
    complete_models,
    evaluate_choices,
    first_complete_rows,
    infer_ids,
    load_full_predictions,
    matrix_array,
    nearest_source_indices,
    render_report,
)
from bench_coe.improve4_failure_modeling_experiments import (
    answer_partition_features,
    local_scores_from_features,
    output_bundle,
    source_dataset_splits,
    subset_full,
)
from bench_coe.materialize_innovation_strategies import fmt_pct, write_text
from bench_coe.offline_router_innovation_experiments import (
    best_model_for_ids,
    instance_oracle_accuracy,
    row_id,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean implementation of improve5.md failure ecology ideas. "
            "Routing uses source correctness and unlabeled target inputs/outputs; target correctness is final scoring only."
        )
    )
    parser.add_argument("--cases", default="portfolio_to_bbh,portfolio_to_gpqa,portfolio_to_mmstar")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/improve5_failure_ecology"))
    parser.add_argument("--include-models", nargs="*", default=[])
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--state-clusters", type=int, default=16)
    parser.add_argument("--sparse-sizes", default="4,6,8")
    parser.add_argument("--conservative-margin", type=float, default=0.06)
    parser.add_argument("--skip-lobo", action="store_true")
    parser.add_argument("--include-contradiction", action="store_true")
    parser.add_argument("--methods", default="", help="Optional comma-separated improve5 methods to run.")
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def select_cases(value: str) -> list[CaseSpec]:
    by_id = {case.case_id: case for case in CASES}
    if value == "all":
        return list(CASES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"Unknown case ids: {missing}; valid ids: {sorted(by_id)}")
    return [by_id[item] for item in selected]


def normalize_dense(features: Any) -> Any:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return StandardScaler().fit_transform(arr)


def answer_group_matrix(answers: list[list[str]]) -> Any:
    import numpy as np

    rows: list[list[float]] = []
    for row_answers in answers:
        counts = Counter(row_answers)
        total = max(1, len(row_answers))
        rows.append([counts[answer] / total for answer in row_answers])
    return np.asarray(rows, dtype=float)


def uncertainty_matrix(stats: list[list[dict[str, float]]]) -> Any:
    import numpy as np

    return np.asarray(
        [[min(1.0, values.get("uncertainty", 0.0) / 4.0) for values in row_stats] for row_stats in stats],
        dtype=float,
    )


def output_state_features(
    full: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    ids: list[str],
) -> tuple[Any, list[list[str]], Any, Any]:
    answers, _, stats = output_bundle(full, models, ids)
    features = answer_partition_features(answers, stats)
    group_share = answer_group_matrix(answers)
    uncertainty = uncertainty_matrix(stats)
    return features, answers, group_share, uncertainty


def source_smoothed_state_scores(source_y: Any, labels: Any, clusters: int, global_acc: Any, alpha: float = 8.0) -> Any:
    import numpy as np

    scores = np.zeros((clusters, source_y.shape[1]), dtype=float)
    for cid in range(clusters):
        mask = labels == cid
        count = int(mask.sum())
        if count:
            scores[cid] = (source_y[mask].sum(axis=0) + alpha * global_acc) / (count + alpha)
        else:
            scores[cid] = global_acc
    return scores


def fate_failure_ecology_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    clusters: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    x_source_raw, _, source_group, _ = output_state_features(source_full, models, source_ids)
    x_target_raw, _, target_group, _ = output_state_features(target_full, models, target_ids)
    combined = normalize_dense(np.vstack([x_source_raw, x_target_raw]))
    x_source = combined[: len(source_ids)]
    x_target = combined[len(source_ids) :]
    k = min(max(2, clusters), max(2, len(source_ids) // 8), len(source_ids) + len(target_ids))
    clusterer = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, n_init=10)
    source_labels = clusterer.fit_predict(x_source)
    target_labels = clusterer.predict(x_target)
    global_acc = source_y.mean(axis=0)
    state_scores = source_smoothed_state_scores(source_y, source_labels, k, global_acc)
    scores = 0.74 * state_scores[target_labels] + 0.18 * target_group + 0.08 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "FATE failure ecology state -> expert success",
        "state_clusters": k,
        "source_cluster_histogram": dict(Counter(int(item) for item in source_labels).most_common()),
        "target_cluster_histogram": dict(Counter(int(item) for item in target_labels).most_common()),
        "target_inputs": "unlabeled expert answer/output ecology",
    }


def correction_graph(source_y: Any) -> Any:
    import numpy as np

    global_acc = source_y.mean(axis=0)
    graph = np.zeros((source_y.shape[1], source_y.shape[1]), dtype=float)
    for src in range(source_y.shape[1]):
        failed = source_y[:, src] < 0.5
        if failed.any():
            graph[src] = (source_y[failed].sum(axis=0) + 4.0 * global_acc) / (float(failed.sum()) + 4.0)
        else:
            graph[src] = global_acc
    return graph


def local_output_success(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[Any, list[list[str]], Any, Any]:
    x_source, _, _, _ = output_state_features(source_full, models, source_ids)
    x_target, target_answers, target_group, target_uncertainty = output_state_features(target_full, models, target_ids)
    return local_scores_from_features(x_source, x_target, source_y, k), target_answers, target_group, target_uncertainty


def ecr_correction_graph_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    local, _, target_group, target_uncertainty = local_output_success(
        source_full, target_full, source_y, models, source_ids, target_ids, k
    )
    global_acc = source_y.mean(axis=0)
    repair = correction_graph(source_y)
    failure_signal = np.clip(1.0 - target_group + 0.35 * target_uncertainty, 0.0, 1.5)
    denom = failure_signal.sum(axis=1, keepdims=True)
    weights = np.divide(
        failure_signal,
        denom,
        out=np.full_like(failure_signal, 1.0 / failure_signal.shape[1]),
        where=denom > 1e-12,
    )
    repair_scores = weights @ repair
    scores = 0.36 * local + 0.34 * repair_scores + 0.18 * target_group + 0.16 * global_acc[None, :] - 0.04 * failure_signal
    return choices_from_scores(target_ids, models, scores), {
        "idea": "ECR expert correction relationship graph",
        "k": min(k, len(source_ids)),
        "score": "local output success + source repair graph + target failure signal",
    }


def split_robust_reliability(
    source_rows: list[dict[str, Any]],
    source_ids: list[str],
    source_y: Any,
    models: list[str],
) -> Any:
    import numpy as np

    splits = source_dataset_splits(source_rows, source_ids)
    global_acc = source_y.mean(axis=0)
    if len(splits) < 2:
        return global_acc
    index = {rid: idx for idx, rid in enumerate(source_ids)}
    per_split = []
    for ids in splits.values():
        rows = [index[rid] for rid in ids if rid in index]
        if rows:
            per_split.append(source_y[rows].mean(axis=0))
    if len(per_split) < 2:
        return global_acc
    arr = np.vstack(per_split)
    reliability = arr.mean(axis=0) - 0.55 * arr.std(axis=0)
    reliability = np.maximum(reliability, 0.0)
    return reliability


def dare_reliability_choices(
    source_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    local, _, target_group, target_uncertainty = local_output_success(
        source_full, target_full, source_y, models, source_ids, target_ids, k
    )
    reliability = split_robust_reliability(source_rows, source_ids, source_y, models)
    behavior_stability = target_group * (1.0 - 0.35 * target_uncertainty)
    scores = 0.44 * local + 0.28 * reliability[None, :] + 0.24 * behavior_stability + 0.04 * source_y.mean(axis=0)[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "DARE domain-agnostic reliability from behavior stability",
        "k": min(k, len(source_ids)),
        "reliability": "source split mean - 0.55 std, modulated by target answer stability",
    }


def ecc_score_matrix(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[Any, list[list[str]], Any]:
    import numpy as np

    local, target_answers, target_group, _ = local_output_success(
        source_full, target_full, source_y, models, source_ids, target_ids, k
    )
    global_acc = source_y.mean(axis=0)
    repair = correction_graph(source_y)
    scores = np.zeros((len(target_ids), len(models)), dtype=float)
    for ridx, answers in enumerate(target_answers):
        groups: dict[str, list[int]] = defaultdict(list)
        for midx, answer in enumerate(answers):
            groups[answer].append(midx)
        for member_idx in groups.values():
            size_bonus = len(member_idx) / len(models)
            local_group = float(local[ridx, member_idx].mean())
            prior_group = float(global_acc[member_idx].mean())
            repair_group = float(repair[:, member_idx].mean())
            group_score = 0.56 * local_group + 0.22 * prior_group + 0.16 * size_bonus + 0.06 * repair_group
            for midx in member_idx:
                scores[ridx, midx] = group_score + 0.012 * local[ridx, midx] + 0.006 * global_acc[midx]
    scores += 0.01 * target_group
    return scores, target_answers, target_group


def ecc_code_decoder_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    scores, _, _ = ecc_score_matrix(source_full, target_full, source_y, models, source_ids, target_ids, k)
    return choices_from_scores(target_ids, models, scores), {
        "idea": "ECC answer-code decoder score baseline inside improve5",
        "k": min(k, len(source_ids)),
    }


def conservative_ecc_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
    margin: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    scores, _, _ = ecc_score_matrix(source_full, target_full, source_y, models, source_ids, target_ids, k)
    global_acc = source_y.mean(axis=0)
    baseline_idx = int(np.argmax(global_acc))
    best_idx = np.argmax(scores, axis=1)
    choices: dict[str, str] = {}
    switches = 0
    for ridx, rid in enumerate(target_ids):
        candidate = int(best_idx[ridx])
        if scores[ridx, candidate] - scores[ridx, baseline_idx] >= margin:
            choices[rid] = models[candidate]
            switches += int(candidate != baseline_idx)
        else:
            choices[rid] = models[baseline_idx]
    return choices, {
        "idea": "oracle-gap conservative ECC routing",
        "baseline": models[baseline_idx],
        "margin": margin,
        "switch_rate": switches / max(1, len(target_ids)),
    }


def greedy_oracle_subset(source_y: Any, models: list[str], size: int) -> list[int]:
    import numpy as np

    chosen: list[int] = []
    covered = np.zeros(source_y.shape[0], dtype=bool)
    global_acc = source_y.mean(axis=0)
    for _ in range(min(size, len(models))):
        best_idx = None
        best_key = None
        for midx in range(len(models)):
            if midx in chosen:
                continue
            new_covered = covered | (source_y[:, midx] > 0.5)
            gain = float(new_covered.mean() - covered.mean())
            key = (gain, float(global_acc[midx]))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = midx
        if best_idx is None:
            break
        chosen.append(best_idx)
        covered = covered | (source_y[:, best_idx] > 0.5)
    return chosen


def sparse_ecc_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
    subset_size: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    subset_idx = greedy_oracle_subset(source_y, models, subset_size)
    subset_models = [models[midx] for midx in subset_idx]
    subset_source_full = subset_full(source_full, subset_models, source_ids)
    subset_target_full = subset_full(target_full, subset_models, target_ids)
    subset_y = source_y[:, subset_idx]
    scores, _, _ = ecc_score_matrix(
        subset_source_full,
        subset_target_full,
        subset_y,
        subset_models,
        source_ids,
        target_ids,
        k,
    )
    return choices_from_scores(target_ids, subset_models, scores), {
        "idea": "Sparse ECC routing with source greedy oracle coverage subset",
        "subset_size": len(subset_models),
        "subset_models": subset_models,
        "source_subset_oracle": float((subset_y.max(axis=1) > 0.5).mean()) if len(subset_models) else 0.0,
    }


def teacher_free_distilled_router_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    teacher_scores, _, _ = ecc_score_matrix(source_full, source_full, source_y, models, source_ids, source_ids, k)
    labels = np.argmax(teacher_scores, axis=1)
    unique = sorted(set(int(item) for item in labels))
    if len(unique) == 1:
        model = models[unique[0]]
        return {rid: model for rid in target_ids}, {
            "idea": "teacher-free distilled output router",
            "status": "single_teacher_class",
            "teacher_model": model,
        }
    _, x_source, x_target = build_vectorizer(source_rows, target_rows)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=1)
    clf.fit(x_source, labels)
    pred = clf.predict(x_target)
    return {rid: models[int(pred[idx])] for idx, rid in enumerate(target_ids)}, {
        "idea": "teacher-free question-only distillation of ECC output router",
        "teacher_label_histogram": {models[idx]: int(count) for idx, count in Counter(labels).most_common()},
        "uses_target_outputs": False,
    }


def benchmark_transfer_graph_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.preprocessing import normalize

    splits = source_dataset_splits(source_rows, source_ids)
    global_acc = source_y.mean(axis=0)
    if len(splits) < 2:
        return choices_from_scores(target_ids, models, np.tile(global_acc, (len(target_ids), 1))), {
            "idea": "cross-benchmark expert transfer graph",
            "status": "fallback_source_global_best",
        }
    _, x_source, x_target = build_vectorizer(source_rows, target_rows)
    x_source = normalize(x_source, norm="l2", copy=False)
    x_target = normalize(x_target, norm="l2", copy=False)
    index = {rid: idx for idx, rid in enumerate(source_ids)}
    split_names = sorted(splits)
    centroids = []
    split_acc = []
    for name in split_names:
        rows = [index[rid] for rid in splits[name] if rid in index]
        if not rows:
            continue
        centroid = x_source[rows].mean(axis=0)
        centroids.append(np.asarray(centroid).ravel())
        split_acc.append(source_y[rows].mean(axis=0))
    if not centroids:
        return choices_from_scores(target_ids, models, np.tile(global_acc, (len(target_ids), 1))), {
            "idea": "cross-benchmark expert transfer graph",
            "status": "fallback_source_global_best",
        }
    centroid_matrix = normalize(np.vstack(centroids), norm="l2")
    sims = np.maximum(0.0, x_target @ centroid_matrix.T)
    sims = np.asarray(sims)
    denom = sims.sum(axis=1, keepdims=True)
    weights = np.divide(sims, denom, out=np.full_like(sims, 1.0 / sims.shape[1]), where=denom > 1e-12)
    scores = weights @ np.vstack(split_acc)
    scores = 0.84 * scores + 0.16 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "cross-benchmark expert transfer graph",
        "source_splits": {name: len(ids) for name, ids in splits.items()},
        "uses_target_outputs": False,
    }


def semantic_contradiction_features(outputs: list[list[str]], answers: list[list[str]]) -> Any:
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer

    if not outputs:
        return np.zeros((0, 0), dtype=float)
    rows = len(outputs)
    models = len(outputs[0])
    docs = [text for row in outputs for text in row]
    vectorizer = HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=2**14,
        alternate_sign=False,
        norm="l2",
    )
    mat = vectorizer.transform(docs)
    features: list[list[float]] = []
    for ridx in range(rows):
        block = mat[ridx * models : (ridx + 1) * models]
        sim = (block @ block.T).toarray()
        row_features: list[float] = []
        for midx in range(models):
            same = [j for j in range(models) if j != midx and answers[ridx][j] == answers[ridx][midx]]
            diff = [j for j in range(models) if j != midx and answers[ridx][j] != answers[ridx][midx]]
            same_sim = float(sim[midx, same].mean()) if same else 0.0
            diff_sim = float(sim[midx, diff].mean()) if diff else 0.0
            row_features.extend([same_sim, diff_sim, max(0.0, diff_sim - same_sim)])
        features.append(row_features)
    return np.asarray(features, dtype=float)


def contradiction_routing_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    source_answers, source_outputs, source_stats = output_bundle(source_full, models, source_ids)
    target_answers, target_outputs, target_stats = output_bundle(target_full, models, target_ids)
    source_base = answer_partition_features(source_answers, source_stats)
    target_base = answer_partition_features(target_answers, target_stats)
    source_sem = semantic_contradiction_features(source_outputs, source_answers)
    target_sem = semantic_contradiction_features(target_outputs, target_answers)
    x_source = np.hstack([source_base, source_sem])
    x_target = np.hstack([target_base, target_sem])
    local = local_scores_from_features(x_source, x_target, source_y, k)
    global_acc = source_y.mean(axis=0)
    target_group = answer_group_matrix(target_answers)
    scores = 0.72 * local + 0.17 * target_group + 0.11 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "output semantic contradiction routing",
        "k": min(k, len(source_ids)),
        "features": "answer ecology + hashed output similarity/contradiction",
    }


MethodBuilder = Callable[
    [
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, dict[str, Any]]],
        dict[str, dict[str, dict[str, Any]]],
        Any,
        list[str],
        list[str],
        list[str],
        argparse.Namespace,
    ],
    tuple[dict[str, str], dict[str, Any]],
]


def build_methods(args: argparse.Namespace) -> dict[str, MethodBuilder]:
    methods: dict[str, MethodBuilder] = {
        "ecc_code_decoder": lambda sr, tr, sf, tf, sy, models, sids, tids, a: ecc_code_decoder_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "fate_failure_ecology": lambda sr, tr, sf, tf, sy, models, sids, tids, a: fate_failure_ecology_choices(
            sf, tf, sy, models, sids, tids, a.state_clusters, a.seed
        ),
        "ecr_correction_graph": lambda sr, tr, sf, tf, sy, models, sids, tids, a: ecr_correction_graph_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "dare_reliability": lambda sr, tr, sf, tf, sy, models, sids, tids, a: dare_reliability_choices(
            sr, sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "conservative_ecc": lambda sr, tr, sf, tf, sy, models, sids, tids, a: conservative_ecc_choices(
            sf, tf, sy, models, sids, tids, a.knn_k, a.conservative_margin
        ),
        "teacher_free_distilled_ecc": lambda sr, tr, sf, tf, sy, models, sids, tids, a: teacher_free_distilled_router_choices(
            sr, tr, sf, sy, models, sids, tids, a.knn_k, a.seed
        ),
        "benchmark_transfer_graph": lambda sr, tr, sf, tf, sy, models, sids, tids, a: benchmark_transfer_graph_choices(
            sr, tr, sy, models, sids, tids
        ),
    }
    for size in [int(item) for item in args.sparse_sizes.split(",") if item.strip()]:
        methods[f"sparse_ecc_k{size}"] = (
            lambda sr, tr, sf, tf, sy, models, sids, tids, a, subset_size=size: sparse_ecc_choices(
                sf, tf, sy, models, sids, tids, a.knn_k, subset_size
            )
        )
    if args.include_contradiction:
        methods["semantic_contradiction"] = lambda sr, tr, sf, tf, sy, models, sids, tids, a: contradiction_routing_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        )
    if args.methods.strip():
        requested = {item.strip() for item in args.methods.split(",") if item.strip()}
        missing = sorted(requested.difference(methods))
        if missing:
            raise ValueError(f"Unknown requested methods: {missing}; valid methods: {sorted(methods)}")
        methods = {name: builder for name, builder in methods.items() if name in requested}
    return methods


def meta_lobo_choice(
    method_names: list[str],
    method_builders: dict[str, MethodBuilder],
    source_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    source_matrix: dict[str, dict[str, bool]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_rows: list[dict[str, Any]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    target_ids: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    splits = source_dataset_splits(source_rows, source_ids)
    if len(splits) < 3:
        best_model, _ = best_model_for_ids(source_matrix, source_ids)
        return {rid: best_model for rid in target_ids}, {
            "idea": "leave-one-benchmark-out meta routing",
            "status": "fallback_source_global_best",
            "reason": "fewer than three source benchmark splits",
        }

    rows_by_id = {row_id(row): row for row in source_rows}
    scores: dict[str, list[float]] = {name: [] for name in method_names}
    scores["source_global_best"] = []
    for heldout_name, heldout_ids in splits.items():
        train_ids = [rid for name, ids in splits.items() if name != heldout_name for rid in ids]
        if not train_ids or not heldout_ids:
            continue
        train_rows = [rows_by_id[rid] for rid in train_ids]
        heldout_rows = [rows_by_id[rid] for rid in heldout_ids]
        train_full = subset_full(source_full, models, train_ids)
        heldout_full = subset_full(source_full, models, heldout_ids)
        train_matrix = {model: {rid: source_matrix[model][rid] for rid in train_ids} for model in models}
        heldout_matrix = {model: {rid: source_matrix[model][rid] for rid in heldout_ids} for model in models}
        train_y = np.asarray([[1.0 if train_matrix[model][rid] else 0.0 for model in models] for rid in train_ids], dtype=float)
        best_train_model, _ = best_model_for_ids(train_matrix, train_ids)
        baseline_choices = {rid: best_train_model for rid in heldout_ids}
        scores["source_global_best"].append(
            sum(1 for rid in heldout_ids if heldout_matrix[baseline_choices[rid]].get(rid, False)) / len(heldout_ids)
        )
        for name in method_names:
            choices, _ = method_builders[name](train_rows, heldout_rows, train_full, heldout_full, train_y, models, train_ids, heldout_ids, args)
            acc = sum(1 for rid in heldout_ids if heldout_matrix[choices[rid]].get(rid, False)) / len(heldout_ids)
            scores[name].append(acc)

    mean_scores = {name: (sum(vals) / len(vals) if vals else -1.0) for name, vals in scores.items()}
    selected = max(mean_scores, key=lambda name: mean_scores[name])
    if selected == "source_global_best":
        best_model, _ = best_model_for_ids(source_matrix, source_ids)
        choices = {rid: best_model for rid in target_ids}
        meta = {"selected_method": selected}
    else:
        choices, meta = method_builders[selected](source_rows, target_rows, source_full, target_full, source_y, models, source_ids, target_ids, args)
        meta = dict(meta)
        meta["selected_method"] = selected
    meta.update(
        {
            "idea": "leave-one-benchmark-out meta routing",
            "source_splits": {name: len(ids) for name, ids in splits.items()},
            "meta_scores": {name: round(score, 6) for name, score in sorted(mean_scores.items())},
        }
    )
    return choices, meta


def render_improve5_report(path: Path, case: CaseSpec, rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], target_matrix: dict[str, dict[str, bool]], choice_maps: dict[str, dict[str, str]]) -> None:
    render_report(path, case, rows, target_rows, target_matrix, choice_maps)
    text = path.read_text(encoding="utf-8")
    text = text.replace("Improve2 clean capability routing", "Improve5 clean failure-ecology routing")
    text = text.replace(
        "All improve2 methods above are calibrated without target labels.",
        "All improve5 methods use source correctness plus unlabeled target inputs/outputs only; target correctness is final scoring only.",
    )
    path.write_text(text, encoding="utf-8")


def run_case(case: CaseSpec, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_full_raw = load_full_predictions(case.source_kind, case.source_root)
    target_full_raw = load_full_predictions(case.target_kind, case.target_root)
    source_ids = infer_ids(source_full_raw)
    target_ids = infer_ids(target_full_raw)
    source_bool_raw = bool_matrix(source_full_raw)
    target_bool_raw = bool_matrix(target_full_raw)
    source_complete = complete_models(source_bool_raw, source_ids)
    target_complete = complete_models(target_bool_raw, target_ids)
    included = set(args.include_models or [])
    excluded = set(args.exclude_models or [])
    models = sorted(set(source_complete).intersection(target_complete).difference(excluded))
    if included:
        models = [model for model in models if model in included]
    if not models:
        raise RuntimeError(f"No common models for {case.case_id}")

    source_full = subset_full(source_full_raw, models, source_ids)
    target_full = subset_full(target_full_raw, models, target_ids)
    source_matrix = {model: source_complete[model] for model in models}
    target_matrix = {model: target_complete[model] for model in models}
    source_rows = first_complete_rows(source_full, source_ids)
    target_rows = first_complete_rows(target_full, target_ids)
    source_y = matrix_array(source_matrix, models, source_ids)
    best_source_model, best_source_acc = best_model_for_ids(source_matrix, source_ids)
    best_target_model, best_target_acc = best_model_for_ids(target_matrix, target_ids)
    oracle_acc = instance_oracle_accuracy(target_matrix, target_ids)

    method_builders = build_methods(args)
    choice_maps: dict[str, dict[str, str]] = {"source_global_best": {rid: best_source_model for rid in target_ids}}
    metadata: dict[str, dict[str, Any]] = {"source_global_best": {"source_accuracy": best_source_acc}}
    for name, builder in method_builders.items():
        choices, meta = builder(source_rows, target_rows, source_full, target_full, source_y, models, source_ids, target_ids, args)
        choice_maps[name] = choices
        metadata[name] = meta
    if not args.skip_lobo:
        choices, meta = meta_lobo_choice(
            list(method_builders),
            method_builders,
            source_rows,
            source_full,
            source_matrix,
            source_y,
            models,
            source_ids,
            target_rows,
            target_full,
            target_ids,
            args,
        )
        choice_maps["lobo_meta_router"] = choices
        metadata["lobo_meta_router"] = meta

    rows = []
    for method, choices in choice_maps.items():
        row = evaluate_choices(
            method,
            choices,
            target_rows,
            target_matrix,
            target_ids,
            best_target_model,
            best_target_acc,
            oracle_acc,
            metadata.get(method, {}),
        )
        row.update(
            {
                "case_id": case.case_id,
                "source_root": str(case.source_root),
                "target_root": str(case.target_root),
                "source_samples": len(source_ids),
                "source_global_best": best_source_model,
                "source_global_best_accuracy": best_source_acc,
            }
        )
        rows.append(row)

    case_dir = args.output_dir / case.case_id
    write_csv(case_dir / "improve5_results.csv", rows)
    write_json(case_dir / "improve5_results.json", rows)
    write_json(case_dir / "choices_by_method.json", choice_maps)
    write_json(
        case_dir / "manifest.json",
        {
            "case_id": case.case_id,
            "title": case.title,
            "source_kind": case.source_kind,
            "source_root": str(case.source_root),
            "target_kind": case.target_kind,
            "target_root": str(case.target_root),
            "source_samples": len(source_ids),
            "target_samples": len(target_ids),
            "models": models,
            "excluded_models": sorted(excluded),
            "note": "Routing uses source correctness and unlabeled target inputs/outputs only. Target correctness is used only for final reporting.",
        },
    )
    render_improve5_report(
        case_dir / f"Bench_Harness_Result_improve5_{case.case_id}.txt",
        case,
        rows,
        target_rows,
        target_matrix,
        choice_maps,
    )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for case in select_cases(args.cases):
        all_rows.extend(run_case(case, args))
    write_csv(args.output_dir / "summary.csv", all_rows)
    write_json(args.output_dir / "summary.json", all_rows)
    lines = [
        "# Improve5 Clean Failure Ecology Results",
        "",
        "Routing uses source correctness plus unlabeled target inputs/outputs. Target correctness is final scoring only.",
        "",
        "| Case | Method | Target Acc | Best Single | Gain | Transfer Ratio | Models |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(all_rows, key=lambda item: (item["case_id"], -float(item["target_accuracy"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_id']}`",
                    f"`{row['method']}`",
                    fmt_pct(row["target_accuracy"]),
                    f"{fmt_pct(row['best_single_target'])} ({row['best_single_model_target']})",
                    f"{float(row['gain_vs_best_single_target']) * 100:+.2f}%",
                    f"{float(row['transfer_ratio']) * 100:+.2f}%",
                    str(row["models_used"]),
                ]
            )
            + " |"
        )
    write_text(args.output_dir / "summary.md", lines)
    print(json.dumps(all_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
