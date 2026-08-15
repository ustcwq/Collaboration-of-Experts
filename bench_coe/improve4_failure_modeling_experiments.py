from __future__ import annotations

import argparse
import json
import math
import re
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
from bench_coe.materialize_innovation_strategies import fmt_pct, write_text
from bench_coe.offline_router_innovation_experiments import (
    best_model_for_ids,
    instance_oracle_accuracy,
    row_id,
    write_csv,
    write_json,
)


UNCERTAINTY_TERMS = (
    "not sure",
    "uncertain",
    "maybe",
    "might",
    "cannot determine",
    "can't determine",
    "not enough information",
    "unknown",
    "guess",
    "ambiguous",
)

OPTION_RE = re.compile(r"(?:answer|final answer|correct answer)\s*(?:is|:)?\s*\(?([A-Z])\)?", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean implementation of improve4.md failure-modeling ideas. "
            "Routing uses source correctness plus unlabeled target expert outputs; target labels are final scoring only."
        )
    )
    parser.add_argument("--cases", default="portfolio_to_bbh,portfolio_to_gpqa,portfolio_to_mmstar")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/improve4_failure_modeling"))
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--failure-clusters", type=int, default=12)
    parser.add_argument("--skip-lobo", action="store_true", help="Skip leave-one-benchmark-out meta routing.")
    parser.add_argument("--include-ot", action="store_true", help="Also run the slower text-distribution OT baseline.")
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


def normalize_answer(value: Any, output: str) -> str:
    if value is not None and str(value).strip():
        text = str(value).strip()
    else:
        match = OPTION_RE.search(output or "")
        lines = str(output or "").strip().splitlines()
        text = match.group(1) if match else (lines[-1] if lines else "")
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;()[]{}\"'")
    if not text:
        return "<empty>"
    if len(text) > 48:
        text = text[:48]
    return text


def output_bundle(full: dict[str, dict[str, dict[str, Any]]], models: list[str], ids: list[str]) -> tuple[list[list[str]], list[list[str]], list[list[dict[str, float]]]]:
    answers: list[list[str]] = []
    outputs: list[list[str]] = []
    stats: list[list[dict[str, float]]] = []
    for rid in ids:
        row_answers: list[str] = []
        row_outputs: list[str] = []
        row_stats: list[dict[str, float]] = []
        for model in models:
            row = full[model][rid]
            output = str(row.get("model_outputs", row.get("response", "")) or "")
            pred = row.get("pred", row.get("prediction"))
            answer = normalize_answer(pred, output)
            tokens = re.findall(r"\w+", output.lower())
            uncertainty = sum(output.lower().count(term) for term in UNCERTAINTY_TERMS)
            row_answers.append(answer)
            row_outputs.append(output)
            row_stats.append(
                {
                    "answer_empty": 1.0 if answer == "<empty>" else 0.0,
                    "log_output_len": math.log1p(len(output)),
                    "log_token_len": math.log1p(len(tokens)),
                    "uncertainty": math.log1p(uncertainty),
                    "truncated": 1.0 if row.get("prompt_was_truncated") else 0.0,
                    "log_prompt_tokens": math.log1p(float(row.get("prompt_token_count") or 0.0)),
                }
            )
        answers.append(row_answers)
        outputs.append(row_outputs)
        stats.append(row_stats)
    return answers, outputs, stats


def answer_partition_features(answers: list[list[str]], stats: list[list[dict[str, float]]]) -> Any:
    import numpy as np

    features: list[list[float]] = []
    for row_answers, row_stats in zip(answers, stats):
        counts = Counter(row_answers)
        total = max(1, len(row_answers))
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(total + 1e-12)
        majority = max(probs) if probs else 0.0
        row_feature: list[float] = [entropy, majority, len(counts) / total]
        for midx, answer in enumerate(row_answers):
            row_feature.append(counts[answer] / total)
            row_feature.append(1.0 if counts[answer] == max(counts.values()) else 0.0)
            row_feature.extend(
                [
                    row_stats[midx]["answer_empty"],
                    row_stats[midx]["log_output_len"] / 8.0,
                    row_stats[midx]["log_token_len"] / 8.0,
                    row_stats[midx]["uncertainty"] / 4.0,
                    row_stats[midx]["truncated"],
                    row_stats[midx]["log_prompt_tokens"] / 10.0,
                ]
            )
        for i in range(total):
            for j in range(i + 1, total):
                row_feature.append(1.0 if row_answers[i] == row_answers[j] else 0.0)
        features.append(row_feature)
    return np.asarray(features, dtype=float)


def text_features(source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> tuple[Any, Any]:
    _, x_source, x_target = build_vectorizer(source_rows, target_rows)
    return x_source, x_target


def hstack_features(*parts: Any) -> Any:
    import scipy.sparse as sp

    sparse_parts = [part if sp.issparse(part) else sp.csr_matrix(part) for part in parts]
    return sp.hstack(sparse_parts, format="csr")


def local_scores_from_features(x_source: Any, x_target: Any, source_y: Any, k: int) -> Any:
    import numpy as np
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize

    x_source = x_source if sp.issparse(x_source) else sp.csr_matrix(x_source)
    x_target = x_target if sp.issparse(x_target) else sp.csr_matrix(x_target)
    x_source = normalize(x_source, norm="l2", copy=False)
    x_target = normalize(x_target, norm="l2", copy=False)
    idx, weights = nearest_source_indices(x_source, x_target, k)
    return np.einsum("ij,ijk->ik", weights, source_y[idx])


def source_failure_clusters(source_y: Any, n_clusters: int, seed: int) -> tuple[Any, Any]:
    import numpy as np
    from sklearn.cluster import KMeans

    k = min(max(2, n_clusters), max(2, min(len(source_y), len(set(map(tuple, source_y.astype(int)))))))
    clusterer = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = clusterer.fit_predict(source_y)
    cluster_scores = np.zeros((k, source_y.shape[1]), dtype=float)
    for cid in range(k):
        indices = labels == cid
        if indices.any():
            cluster_scores[cid] = source_y[indices].mean(axis=0)
        else:
            cluster_scores[cid] = source_y.mean(axis=0)
    return labels, cluster_scores


def fame_manifold_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
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
    from sklearn.linear_model import LogisticRegression

    source_answers, _, source_stats = output_bundle(source_full, models, source_ids)
    target_answers, _, target_stats = output_bundle(target_full, models, target_ids)
    source_out = answer_partition_features(source_answers, source_stats)
    target_out = answer_partition_features(target_answers, target_stats)
    x_source = source_out
    x_target = target_out

    labels, cluster_scores = source_failure_clusters(source_y, clusters, seed)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(x_source, labels)
    probs = clf.predict_proba(x_target)
    scores = probs @ cluster_scores
    return choices_from_scores(target_ids, models, scores), {
        "idea": "FAME failure manifold alignment",
        "failure_clusters": int(cluster_scores.shape[0]),
        "cluster_histogram": dict(Counter(int(item) for item in labels).most_common()),
        "target_features": "unlabeled expert answer partition/output stats",
    }


def fame_output_knn_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    source_answers, _, source_stats = output_bundle(source_full, models, source_ids)
    target_answers, _, target_stats = output_bundle(target_full, models, target_ids)
    x_source = answer_partition_features(source_answers, source_stats)
    x_target = answer_partition_features(target_answers, target_stats)
    scores = local_scores_from_features(x_source, x_target, source_y, k)
    return choices_from_scores(target_ids, models, scores), {
        "idea": "FAME output-neighborhood repair",
        "k": min(k, len(source_ids)),
        "target_features": "unlabeled expert answer partition/output stats",
    }


def ecc_code_decoder_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    source_answers, _, source_stats = output_bundle(source_full, models, source_ids)
    target_answers, _, target_stats = output_bundle(target_full, models, target_ids)
    x_source = answer_partition_features(source_answers, source_stats)
    x_target = answer_partition_features(target_answers, target_stats)
    local = local_scores_from_features(x_source, x_target, source_y, k)
    global_acc = source_y.mean(axis=0)
    complement = np.zeros((len(models), len(models)), dtype=float)
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            base_wrong = source_y[:, i] < 0.5
            if base_wrong.any():
                complement[i, j] = source_y[base_wrong, j].mean()
    choices: dict[str, str] = {}
    for ridx, rid in enumerate(target_ids):
        groups: dict[str, list[int]] = defaultdict(list)
        for midx, answer in enumerate(target_answers[ridx]):
            groups[answer].append(midx)
        group_scores: list[tuple[float, list[int]]] = []
        for member_idx in groups.values():
            size_bonus = len(member_idx) / len(models)
            local_group = float(local[ridx, member_idx].mean())
            prior_group = float(global_acc[member_idx].mean())
            repair_group = float(complement[:, member_idx].mean())
            group_scores.append((0.56 * local_group + 0.22 * prior_group + 0.16 * size_bonus + 0.06 * repair_group, member_idx))
        _, chosen_members = max(group_scores, key=lambda item: item[0])
        best_idx = max(chosen_members, key=lambda midx: (local[ridx, midx], global_acc[midx]))
        choices[rid] = models[int(best_idx)]
    return choices, {
        "idea": "ECC-CoE answer-code decoder",
        "k": min(k, len(source_ids)),
        "score": "local_code_success + source_prior + answer_group_size + complementarity",
    }


def error_awareness_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import Ridge

    source_answers, _, source_stats = output_bundle(source_full, models, source_ids)
    target_answers, _, target_stats = output_bundle(target_full, models, target_ids)
    source_out = answer_partition_features(source_answers, source_stats)
    target_out = answer_partition_features(target_answers, target_stats)
    x_source = source_out
    x_target = target_out
    reg = Ridge(alpha=1.0)
    reg.fit(x_source, source_y)
    scores = np.clip(reg.predict(x_target), 0.0, 1.0)
    return choices_from_scores(target_ids, models, scores), {
        "idea": "zero-shot counterfactual expert error awareness from source outputs",
        "model": "ridge_correctness_predictor(unlabeled expert outputs)",
    }


def optimal_transport_prior_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    models: list[str],
    target_ids: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    vectorizer, x_source, x_target = build_vectorizer(source_rows, target_rows)
    k = min(24, max(2, len(source_rows) // 100), max(2, len(target_rows) // 20))
    clusterer = MiniBatchKMeans(n_clusters=k, random_state=20260716, batch_size=1024, n_init=10)
    clusterer.fit(x_source)
    source_labels = clusterer.predict(x_source)
    target_labels = clusterer.predict(x_target)
    global_acc = source_y.mean(axis=0)
    cluster_acc = np.zeros((k, len(models)), dtype=float)
    for cid in range(k):
        mask = source_labels == cid
        cluster_acc[cid] = source_y[mask].mean(axis=0) if mask.any() else global_acc
    target_hist = Counter(int(item) for item in target_labels)
    source_hist = Counter(int(item) for item in source_labels)
    scores = np.zeros((len(target_ids), len(models)), dtype=float)
    for ridx, cid in enumerate(target_labels):
        scarcity = math.sqrt((target_hist[int(cid)] + 1.0) / (source_hist[int(cid)] + 1.0))
        scores[ridx] = 0.72 * cluster_acc[int(cid)] + 0.28 * global_acc * scarcity
    return choices_from_scores(target_ids, models, scores), {
        "idea": "optimal-transport-style source target cluster prior",
        "clusters": k,
        "note": "uses unlabeled target feature distribution only",
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


def build_methods() -> dict[str, MethodBuilder]:
    return {
        "fame_manifold_alignment": lambda sr, tr, sf, tf, sy, models, sids, tids, args: fame_manifold_choices(
            sr, tr, sf, tf, sy, models, sids, tids, args.failure_clusters, args.seed
        ),
        "fame_output_knn_repair": lambda sr, tr, sf, tf, sy, models, sids, tids, args: fame_output_knn_choices(
            sf, tf, sy, models, sids, tids, args.knn_k
        ),
        "ecc_code_decoder": lambda sr, tr, sf, tf, sy, models, sids, tids, args: ecc_code_decoder_choices(
            sf, tf, sy, models, sids, tids, args.knn_k
        ),
        "error_awareness_output_ridge": lambda sr, tr, sf, tf, sy, models, sids, tids, args: error_awareness_choices(
            sr, tr, sf, tf, sy, models, sids, tids
        ),
        "ot_cluster_prior": lambda sr, tr, sf, tf, sy, models, sids, tids, args: optimal_transport_prior_choices(
            sr, tr, sy, models, tids
        ),
    }


def subset_full(full: dict[str, dict[str, dict[str, Any]]], models: list[str], ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    return {model: {rid: full[model][rid] for rid in ids} for model in models}


def source_dataset_splits(source_rows: list[dict[str, Any]], source_ids: list[str]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = defaultdict(list)
    row_by_id = {row_id(row): row for row in source_rows}
    for rid in source_ids:
        row = row_by_id[rid]
        dataset = str(row.get("source_dataset") or row.get("benchmark") or "source")
        splits[dataset].append(rid)
    return dict(splits)


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
        scores.setdefault("source_global_best", []).append(
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


def render_improve4_report(path: Path, case: CaseSpec, rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], target_matrix: dict[str, dict[str, bool]], choice_maps: dict[str, dict[str, str]]) -> None:
    render_report(path, case, rows, target_rows, target_matrix, choice_maps)
    text = path.read_text(encoding="utf-8")
    text = text.replace("Improve2 clean capability routing", "Improve4 clean failure-modeling routing")
    text = text.replace("All improve2 methods above are calibrated without target labels.", "All improve4 methods above use only source correctness and unlabeled target expert outputs for routing.")
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
    excluded = set(args.exclude_models or [])
    models = sorted(set(source_complete).intersection(target_complete).difference(excluded))
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

    method_builders = build_methods()
    if not args.include_ot:
        method_builders.pop("ot_cluster_prior", None)
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
    write_csv(case_dir / "improve4_results.csv", rows)
    write_json(case_dir / "improve4_results.json", rows)
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
            "note": "Routing uses source correctness and unlabeled target expert outputs only. Target correctness is used only for final reporting.",
        },
    )
    render_improve4_report(
        case_dir / f"Bench_Harness_Result_improve4_{case.case_id}.txt",
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
        "# Improve4 Clean Failure Modeling Results",
        "",
        "Routing uses source correctness plus unlabeled target expert outputs. Target correctness is final scoring only.",
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
