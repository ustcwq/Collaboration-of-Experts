from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
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
    render_report,
)
from bench_coe.improve4_failure_modeling_experiments import (
    answer_partition_features,
    local_scores_from_features,
    output_bundle,
    source_dataset_splits,
    subset_full,
)
from bench_coe.improve5_failure_ecology_experiments import (
    answer_group_matrix,
    benchmark_transfer_graph_choices,
    contradiction_routing_choices,
    correction_graph,
    dare_reliability_choices,
    ecc_code_decoder_choices,
    ecc_score_matrix,
    ecr_correction_graph_choices,
    fate_failure_ecology_choices,
    semantic_contradiction_features,
)
from bench_coe.materialize_innovation_strategies import fmt_pct, write_text
from bench_coe.offline_router_innovation_experiments import (
    best_model_for_ids,
    instance_oracle_accuracy,
    row_id,
    write_csv,
    write_json,
)


REASONING_MARKERS = (
    "because",
    "therefore",
    "however",
    "thus",
    "so",
    "maybe",
    "likely",
    "not sure",
    "uncertain",
    "cannot determine",
    "contradiction",
    "final answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean implementation of improve6.md adaptive failure-ecology ideas. "
            "Routing uses source correctness plus unlabeled target inputs/outputs only."
        )
    )
    parser.add_argument("--cases", default="portfolio_to_bbh,portfolio_to_gpqa,portfolio_to_mmstar")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/improve6_adaptive_failure_ecology"))
    parser.add_argument("--include-models", nargs="*", default=[])
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--state-clusters", type=int, default=16)
    parser.add_argument("--sparse-sizes", default="4,6,8")
    parser.add_argument("--oasis-threshold", type=float, default=0.5)
    parser.add_argument("--include-semantic", action="store_true")
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--methods", default="", help="Optional comma-separated improve6 methods to run.")
    parser.add_argument("--meta-only", action="store_true", help="Run only source_global_best plus Meta-FATE.")
    parser.add_argument(
        "--meta-methods",
        default="fate_failure_ecology,ecr_correction_graph,dare_reliability,ecc_code_decoder,repair_chain,benchmark_transfer_graph",
        help="Comma-separated strategy set used by Meta-FATE.",
    )
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


def marker_features(outputs: list[list[str]]) -> Any:
    import numpy as np

    rows: list[list[float]] = []
    for row_outputs in outputs:
        row: list[float] = []
        lengths = []
        marker_totals = []
        for output in row_outputs:
            text = str(output or "").lower()
            tokens = re.findall(r"\w+", text)
            token_count = max(1, len(tokens))
            lengths.append(math.log1p(token_count))
            marker_count = sum(text.count(marker) for marker in REASONING_MARKERS)
            marker_totals.append(marker_count / token_count)
            row.extend(
                [
                    math.log1p(token_count) / 8.0,
                    min(1.0, marker_count / 8.0),
                    min(1.0, text.count("not") / 8.0),
                    min(1.0, len(re.findall(r"\d+", text)) / 12.0),
                    1.0 if any(ch in output for ch in "=<>") else 0.0,
                ]
            )
        if lengths:
            row.extend(
                [
                    float(np.mean(lengths)) / 8.0,
                    float(np.std(lengths)) / 4.0,
                    float(np.max(lengths) - np.min(lengths)) / 6.0,
                    float(np.mean(marker_totals)) if marker_totals else 0.0,
                    float(np.std(marker_totals)) if marker_totals else 0.0,
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=float)


def multiview_features(
    full: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    ids: list[str],
    include_semantic: bool,
) -> tuple[Any, list[list[str]], Any]:
    import numpy as np

    answers, outputs, stats = output_bundle(full, models, ids)
    base = answer_partition_features(answers, stats)
    markers = marker_features(outputs)
    parts = [base, markers]
    if include_semantic:
        parts.append(semantic_contradiction_features(outputs, answers))
    features = np.nan_to_num(np.hstack(parts), nan=0.0, posinf=0.0, neginf=0.0)
    return features, answers, answer_group_matrix(answers)


def scale_features(x_source: Any, x_target: Any) -> tuple[Any, Any]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    return scaler.fit_transform(x_source), scaler.transform(x_target)


def neuro_ecc_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    include_semantic: bool,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    x_source, _, _ = multiview_features(source_full, models, source_ids, include_semantic)
    x_target, _, target_group = multiview_features(target_full, models, target_ids, include_semantic)
    x_source, x_target = scale_features(x_source, x_target)
    reg = MLPRegressor(
        hidden_layer_sizes=(96, 32),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=180,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reg.fit(x_source, source_y)
    pred = np.clip(reg.predict(x_target), 0.0, 1.0)
    global_acc = source_y.mean(axis=0)
    scores = 0.72 * pred + 0.18 * target_group + 0.10 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "Neuro-ECC learned nonlinear expert error code",
        "include_semantic": include_semantic,
        "hidden_layers": [96, 32],
    }


def repair_chain_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    from bench_coe.improve5_failure_ecology_experiments import local_output_success

    local, _, target_group, target_uncertainty = local_output_success(
        source_full, target_full, source_y, models, source_ids, target_ids, k
    )
    repair = correction_graph(source_y)
    global_acc = source_y.mean(axis=0)
    failure_signal = np.clip(1.0 - target_group + 0.40 * target_uncertainty, 0.0, 1.6)
    denom = failure_signal.sum(axis=1, keepdims=True)
    weights = np.divide(
        failure_signal,
        denom,
        out=np.full_like(failure_signal, 1.0 / failure_signal.shape[1]),
        where=denom > 1e-12,
    )
    hop1 = weights @ repair
    hop1_norm = hop1 / np.maximum(hop1.sum(axis=1, keepdims=True), 1e-12)
    hop2 = hop1_norm @ repair
    scores = 0.30 * local + 0.25 * hop1 + 0.18 * hop2 + 0.16 * target_group + 0.11 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "RepairChain-CoE adaptive two-hop expert repair graph",
        "k": min(k, len(source_ids)),
    }


def oasis_conservative_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
    threshold: float,
    include_semantic: bool,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x_source, _, _ = multiview_features(source_full, models, source_ids, include_semantic)
    x_target, _, _ = multiview_features(target_full, models, target_ids, include_semantic)
    x_source, x_target = scale_features(x_source, x_target)
    global_acc = source_y.mean(axis=0)
    baseline_idx = int(np.argmax(global_acc))
    teacher_scores, _, _ = ecc_score_matrix(source_full, source_full, source_y, models, source_ids, source_ids, k)
    teacher_idx = np.argmax(teacher_scores, axis=1)
    row_index = np.arange(len(source_ids))
    utility = source_y[row_index, teacher_idx] - source_y[:, baseline_idx]
    positive = utility > 0.0
    if len(set(positive.astype(int))) < 2:
        repair_prob = np.full(len(target_ids), float(positive.mean()))
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(x_source, positive.astype(int))
        repair_prob = clf.predict_proba(x_target)[:, 1]
    target_scores, _, _ = ecc_score_matrix(source_full, target_full, source_y, models, source_ids, target_ids, k)
    candidate_idx = np.argmax(target_scores, axis=1)
    choices: dict[str, str] = {}
    switches = 0
    for ridx, rid in enumerate(target_ids):
        cand = int(candidate_idx[ridx])
        if repair_prob[ridx] >= threshold and target_scores[ridx, cand] >= target_scores[ridx, baseline_idx]:
            choices[rid] = models[cand]
            switches += int(cand != baseline_idx)
        else:
            choices[rid] = models[baseline_idx]
    return choices, {
        "idea": "OASIS oracle-gap conservative structural repair selection",
        "baseline": models[baseline_idx],
        "threshold": threshold,
        "switch_rate": switches / max(1, len(target_ids)),
        "include_semantic": include_semantic,
    }


def multiview_signature_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
    include_semantic: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    x_source, _, _ = multiview_features(source_full, models, source_ids, include_semantic)
    x_target, _, target_group = multiview_features(target_full, models, target_ids, include_semantic)
    local = local_scores_from_features(x_source, x_target, source_y, k)
    global_acc = source_y.mean(axis=0)
    scores = 0.66 * local + 0.22 * target_group + 0.12 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "Multi-view failure signature local routing",
        "k": min(k, len(source_ids)),
        "include_semantic": include_semantic,
    }


def multiview_fate_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    clusters: int,
    include_semantic: bool,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    x_source, _, _ = multiview_features(source_full, models, source_ids, include_semantic)
    x_target, _, target_group = multiview_features(target_full, models, target_ids, include_semantic)
    x_source, x_target = scale_features(x_source, x_target)
    k = min(max(2, clusters), max(2, len(source_ids) // 8), len(source_ids))
    clusterer = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, n_init=10)
    source_labels = clusterer.fit_predict(x_source)
    target_labels = clusterer.predict(x_target)
    global_acc = source_y.mean(axis=0)
    state_scores = np.zeros((k, len(models)), dtype=float)
    for cid in range(k):
        mask = source_labels == cid
        count = int(mask.sum())
        state_scores[cid] = (source_y[mask].sum(axis=0) + 8.0 * global_acc) / (count + 8.0) if count else global_acc
    scores = 0.72 * state_scores[target_labels] + 0.18 * target_group + 0.10 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "FATE++ multi-view failure ecology state model",
        "state_clusters": k,
        "include_semantic": include_semantic,
        "target_cluster_histogram": dict(Counter(int(item) for item in target_labels).most_common()),
    }


def correction_subset(source_y: Any, models: list[str], size: int) -> list[int]:
    import numpy as np

    chosen: list[int] = []
    global_acc = source_y.mean(axis=0)
    for _ in range(min(size, len(models))):
        best_idx = None
        best_score = -1e9
        for midx in range(len(models)):
            if midx in chosen:
                continue
            subset = chosen + [midx]
            coverage = float((source_y[:, subset].max(axis=1) > 0.5).mean())
            correction_terms = []
            diversity_terms = []
            for i in subset:
                others = [j for j in subset if j != i]
                if others:
                    failed = source_y[:, i] < 0.5
                    if failed.any():
                        correction_terms.append(float((source_y[failed][:, others].max(axis=1) > 0.5).mean()))
                    for j in others:
                        diversity_terms.append(float((source_y[:, i] != source_y[:, j]).mean()))
            correction = sum(correction_terms) / len(correction_terms) if correction_terms else 0.0
            diversity = sum(diversity_terms) / len(diversity_terms) if diversity_terms else 0.0
            score = coverage + 0.18 * correction + 0.04 * diversity + 0.02 * float(global_acc[subset].mean())
            if score > best_score:
                best_score = score
                best_idx = midx
        if best_idx is None:
            break
        chosen.append(best_idx)
    return chosen


def sparse_eccpp_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
    subset_size: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    subset_idx = correction_subset(source_y, models, subset_size)
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
        "idea": "Sparse-ECC++ correction-aware expert subset design",
        "subset_size": len(subset_models),
        "subset_models": subset_models,
        "source_subset_oracle": float((subset_y.max(axis=1) > 0.5).mean()) if len(subset_models) else 0.0,
    }


def fate_state_distilled_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    clusters: int,
    include_semantic: bool,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import LogisticRegression

    x_state, _, _ = multiview_features(source_full, models, source_ids, include_semantic)
    x_state, _ = scale_features(x_state, x_state)
    k = min(max(2, clusters), max(2, len(source_ids) // 8), len(source_ids))
    clusterer = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, n_init=10)
    labels = clusterer.fit_predict(x_state)
    global_acc = source_y.mean(axis=0)
    state_scores = np.zeros((k, len(models)), dtype=float)
    for cid in range(k):
        mask = labels == cid
        count = int(mask.sum())
        state_scores[cid] = (source_y[mask].sum(axis=0) + 8.0 * global_acc) / (count + 8.0) if count else global_acc
    _, x_source_text, x_target_text = build_vectorizer(source_rows, target_rows)
    if len(set(labels)) < 2:
        scores = np.tile(state_scores[int(labels[0])], (len(target_ids), 1))
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(x_source_text, labels)
        probs = clf.predict_proba(x_target_text)
        full_probs = np.zeros((len(target_ids), k), dtype=float)
        for col, label in enumerate(clf.classes_):
            full_probs[:, int(label)] = probs[:, col]
        scores = full_probs @ state_scores
    scores = 0.94 * scores + 0.06 * global_acc[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "idea": "FATE Distillation++ question-only failure-state distillation",
        "state_clusters": k,
        "include_semantic": include_semantic,
        "uses_target_outputs": False,
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
    if args.meta_only:
        return {}
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
        "repair_chain": lambda sr, tr, sf, tf, sy, models, sids, tids, a: repair_chain_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "neuro_ecc": lambda sr, tr, sf, tf, sy, models, sids, tids, a: neuro_ecc_choices(
            sf, tf, sy, models, sids, tids, False, a.seed
        ),
        "multiview_neuro_ecc": lambda sr, tr, sf, tf, sy, models, sids, tids, a: neuro_ecc_choices(
            sf, tf, sy, models, sids, tids, a.include_semantic, a.seed
        ),
        "multiview_signature": lambda sr, tr, sf, tf, sy, models, sids, tids, a: multiview_signature_choices(
            sf, tf, sy, models, sids, tids, a.knn_k, a.include_semantic
        ),
        "multiview_fate": lambda sr, tr, sf, tf, sy, models, sids, tids, a: multiview_fate_choices(
            sf, tf, sy, models, sids, tids, a.state_clusters, a.include_semantic, a.seed
        ),
        "oasis_conservative": lambda sr, tr, sf, tf, sy, models, sids, tids, a: oasis_conservative_choices(
            sf, tf, sy, models, sids, tids, a.knn_k, a.oasis_threshold, a.include_semantic, a.seed
        ),
        "fate_state_distilled": lambda sr, tr, sf, tf, sy, models, sids, tids, a: fate_state_distilled_choices(
            sr, tr, sf, sy, models, sids, tids, a.state_clusters, a.include_semantic, a.seed
        ),
        "benchmark_transfer_graph": lambda sr, tr, sf, tf, sy, models, sids, tids, a: benchmark_transfer_graph_choices(
            sr, tr, sy, models, sids, tids
        ),
    }
    for size in [int(item) for item in args.sparse_sizes.split(",") if item.strip()]:
        methods[f"sparse_eccpp_k{size}"] = (
            lambda sr, tr, sf, tf, sy, models, sids, tids, a, subset_size=size: sparse_eccpp_choices(
                sf, tf, sy, models, sids, tids, a.knn_k, subset_size
            )
        )
    if args.include_semantic:
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


def meta_base_methods(args: argparse.Namespace) -> dict[str, MethodBuilder]:
    methods: dict[str, MethodBuilder] = {
        "fate_failure_ecology": lambda sr, tr, sf, tf, sy, models, sids, tids, a: fate_failure_ecology_choices(
            sf, tf, sy, models, sids, tids, a.state_clusters, a.seed
        ),
        "ecr_correction_graph": lambda sr, tr, sf, tf, sy, models, sids, tids, a: ecr_correction_graph_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "dare_reliability": lambda sr, tr, sf, tf, sy, models, sids, tids, a: dare_reliability_choices(
            sr, sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "ecc_code_decoder": lambda sr, tr, sf, tf, sy, models, sids, tids, a: ecc_code_decoder_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "repair_chain": lambda sr, tr, sf, tf, sy, models, sids, tids, a: repair_chain_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "benchmark_transfer_graph": lambda sr, tr, sf, tf, sy, models, sids, tids, a: benchmark_transfer_graph_choices(
            sr, tr, sy, models, sids, tids
        ),
    }
    if args.include_semantic:
        methods["semantic_contradiction"] = lambda sr, tr, sf, tf, sy, models, sids, tids, a: contradiction_routing_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        )
    requested = {item.strip() for item in args.meta_methods.split(",") if item.strip()}
    if requested:
        missing = sorted(requested.difference(methods))
        if missing:
            raise ValueError(f"Unknown requested meta methods: {missing}; valid methods: {sorted(methods)}")
        methods = {name: builder for name, builder in methods.items() if name in requested}
    return methods


def meta_fate_sample_selector(
    method_builders: dict[str, MethodBuilder],
    target_choice_maps: dict[str, dict[str, str]],
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_matrix: dict[str, dict[str, bool]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    splits = source_dataset_splits(source_rows, source_ids)
    if len(splits) < 3:
        best_model, _ = best_model_for_ids(source_matrix, source_ids)
        return {rid: best_model for rid in target_ids}, {
            "idea": "Meta-FATE sample strategy selector",
            "status": "fallback_source_global_best",
        }
    rows_by_id = {row_id(row): row for row in source_rows}
    source_index = {rid: idx for idx, rid in enumerate(source_ids)}
    method_names = list(method_builders)
    train_x_parts = []
    train_y_parts = []
    split_scores: dict[str, dict[str, float]] = {}
    for heldout_name, heldout_ids in splits.items():
        train_ids = [rid for name, ids in splits.items() if name != heldout_name for rid in ids]
        if not train_ids or not heldout_ids:
            continue
        train_rows = [rows_by_id[rid] for rid in train_ids]
        heldout_rows = [rows_by_id[rid] for rid in heldout_ids]
        train_full = subset_full(source_full, models, train_ids)
        heldout_full = subset_full(source_full, models, heldout_ids)
        train_y = source_y[[source_index[rid] for rid in train_ids]]
        heldout_matrix = {model: {rid: source_matrix[model][rid] for rid in heldout_ids} for model in models}
        x_heldout, _, _ = multiview_features(heldout_full, models, heldout_ids, args.include_semantic)
        label = np.zeros((len(heldout_ids), len(method_names)), dtype=float)
        split_scores[heldout_name] = {}
        for midx, name in enumerate(method_names):
            choices, _ = method_builders[name](train_rows, heldout_rows, train_full, heldout_full, train_y, models, train_ids, heldout_ids, args)
            correct = np.asarray([1.0 if heldout_matrix[choices[rid]].get(rid, False) else 0.0 for rid in heldout_ids], dtype=float)
            label[:, midx] = correct
            split_scores[heldout_name][name] = float(correct.mean())
        train_x_parts.append(x_heldout)
        train_y_parts.append(label)
    if not train_x_parts:
        best_model, _ = best_model_for_ids(source_matrix, source_ids)
        return {rid: best_model for rid in target_ids}, {
            "idea": "Meta-FATE sample strategy selector",
            "status": "fallback_source_global_best",
            "reason": "no_lobo_training_rows",
        }
    x_train = np.vstack(train_x_parts)
    y_train = np.vstack(train_y_parts)
    x_target, _, _ = multiview_features(target_full, models, target_ids, args.include_semantic)
    reg = ExtraTreesRegressor(n_estimators=96, max_depth=12, min_samples_leaf=8, random_state=args.seed, n_jobs=1)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_target)
    selected = np.argmax(pred, axis=1)
    choices: dict[str, str] = {}
    strategy_hist: Counter[str] = Counter()
    for ridx, rid in enumerate(target_ids):
        strategy = method_names[int(selected[ridx])]
        strategy_hist[strategy] += 1
        choices[rid] = target_choice_maps[strategy][rid]
    return choices, {
        "idea": "Meta-FATE sample-level strategy selector trained by source LOBO",
        "strategies": method_names,
        "strategy_histogram": dict(strategy_hist.most_common()),
        "source_lobo_scores": split_scores,
    }


def render_improve6_report(path: Path, case: CaseSpec, rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], target_matrix: dict[str, dict[str, bool]], choice_maps: dict[str, dict[str, str]]) -> None:
    render_report(path, case, rows, target_rows, target_matrix, choice_maps)
    text = path.read_text(encoding="utf-8")
    text = text.replace("Improve2 clean capability routing", "Improve6 clean adaptive failure-ecology routing")
    text = text.replace(
        "All improve2 methods above are calibrated without target labels.",
        "All improve6 methods use source correctness plus unlabeled target inputs/outputs only; target correctness is final scoring only.",
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

    if not args.skip_meta and (args.meta_only or not args.methods.strip()):
        meta_builders = meta_base_methods(args)
        for name, builder in meta_builders.items():
            if name not in choice_maps:
                choices, meta = builder(source_rows, target_rows, source_full, target_full, source_y, models, source_ids, target_ids, args)
                choice_maps[name] = choices
                metadata[name] = meta
        choices, meta = meta_fate_sample_selector(
            meta_builders,
            choice_maps,
            source_rows,
            target_rows,
            source_full,
            target_full,
            source_matrix,
            source_y,
            models,
            source_ids,
            target_ids,
            args,
        )
        choice_maps["meta_fate_sample_selector"] = choices
        metadata["meta_fate_sample_selector"] = meta

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
    write_csv(case_dir / "improve6_results.csv", rows)
    write_json(case_dir / "improve6_results.json", rows)
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
    render_improve6_report(
        case_dir / f"Bench_Harness_Result_improve6_{case.case_id}.txt",
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
        "# Improve6 Clean Adaptive Failure Ecology Results",
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
