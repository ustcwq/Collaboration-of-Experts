from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.offline_router_innovation_experiments import (
    CASES,
    CaseSpec,
    augmented_text_for_row,
    accuracy_for_choices,
    best_model_for_ids,
    best_model_with_wilson,
    choices_from_group_mapping,
    craft_current_repair_lcb_choices,
    current_choice_map,
    deterministic_probe_split,
    dynamic_subject_tfidf_choices,
    failure_boundary_group_choices,
    group_mapping_from_probe,
    group_value,
    parse_float_tag,
    parse_harm_weight_tag,
    is_correct_row,
    keep_complete_models,
    parse_or_confidence_fallback_choices,
    read_json,
    read_jsonl,
    risk_controlled_group_mapping,
    robust_minimax_model,
    row_id,
    text_for_row,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the best offline innovation strategies into heldout Bench-Harness style outputs."
    )
    parser.add_argument(
        "--strategy-summary",
        type=Path,
        default=Path("outputs/bench_coe/innovation_experiments/summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/innovation_experiments/materialized"),
    )
    parser.add_argument("--cases", default="all", help="Comma-separated case ids, or all.")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--risk-margin", type=float, default=0.0)
    parser.add_argument("--wilson-z", type=float, default=1.0)
    return parser.parse_args()


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_strategy_summary(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["case_id"]: row for row in rows}


def select_cases(value: str) -> list[CaseSpec]:
    by_id = {case.case_id: case for case in CASES}
    if value == "all":
        return list(CASES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"Unknown case ids: {missing}; valid ids: {sorted(by_id)}")
    return [by_id[item] for item in selected]


def load_full_jsonl_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        path = model_dir / "predictions.jsonl"
        if path.exists():
            matrix[model_dir.name] = {row_id(row): row for row in read_jsonl(path)}
    return matrix


def load_full_json_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        path = model_dir / "predictions.json"
        if path.exists():
            matrix[model_dir.name] = {row_id(row): row for row in read_json(path)}
    return matrix


def load_full_mmlu_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        result_dir = model_dir / "CoT" / "all"
        if not result_dir.is_dir():
            continue
        rows_by_id: dict[str, dict[str, Any]] = {}
        for path in sorted(result_dir.glob("*.json")):
            payload = read_json(path)
            if isinstance(payload, list):
                rows_by_id.update({row_id(row): row for row in payload})
        if rows_by_id:
            matrix[model_dir.name] = rows_by_id
    return matrix


def load_full_predictions(case: CaseSpec) -> dict[str, dict[str, dict[str, Any]]]:
    if case.single_kind == "jsonl":
        return load_full_jsonl_predictions(case.single_path)
    if case.single_kind == "json":
        return load_full_json_predictions(case.single_path)
    if case.single_kind == "mmlu_pro":
        return load_full_mmlu_predictions(case.single_path)
    raise ValueError(case.single_kind)


def bool_matrix(full_matrix: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, bool]]:
    return {
        model: {rid: is_correct_row(row) for rid, row in rows.items()}
        for model, rows in full_matrix.items()
    }


def route_rows(case: CaseSpec) -> list[dict[str, Any]]:
    path = case.route_dir / "test_predictions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list.")
    return rows


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def table_row(name_width: int, col_width: int, name: str, values: list[str]) -> str:
    return "| " + " | ".join([name.ljust(name_width)] + [value.ljust(col_width) for value in values]) + " |"


def values_for_group(value: Any) -> list[str]:
    if value is None or value == "":
        return ["Unknown"]
    if isinstance(value, list):
        return [str(item) for item in value] or ["Unknown"]
    return [str(value)]


def summarize_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> dict[str, Any]:
    total_correct = 0
    total_wrong = 0
    groups: dict[str, dict[str, dict[str, float]]] = {
        key: defaultdict(lambda: {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0})
        for key in group_keys
    }
    routed_model_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}
    )
    for row in rows:
        is_correct = bool(row["improved_is_correct"])
        total_correct += 1 if is_correct else 0
        total_wrong += 0 if is_correct else 1
        for key in group_keys:
            for value in values_for_group(row.get(key)):
                stats = groups[key][value]
                stats["correct" if is_correct else "wrong"] += 1
        model_stats = routed_model_stats[str(row["improved_routed_model"])]
        model_stats["correct" if is_correct else "wrong"] += 1
    for stats_by_key in list(groups.values()) + [routed_model_stats]:
        for stats in stats_by_key.values():
            denom = stats["correct"] + stats["wrong"]
            stats["accuracy"] = stats["correct"] / denom if denom else 0.0
    total = total_correct + total_wrong
    summary: dict[str, Any] = {
        "total": {
            "correct": float(total_correct),
            "wrong": float(total_wrong),
            "accuracy": total_correct / total if total else 0.0,
        },
        "examples": total,
        "routed_model": dict(sorted(routed_model_stats.items())),
    }
    for key, stats_by_key in groups.items():
        summary[key] = dict(sorted(stats_by_key.items()))
    return summary


def summarize_boolean_choices(
    rows: list[dict[str, Any]],
    ids: list[str],
    model_for_id: dict[str, str],
    matrix: dict[str, dict[str, bool]],
    group_key: str,
) -> dict[str, Any]:
    row_by_id = {row_id(row): row for row in rows}
    correct = 0
    by_group: dict[str, dict[str, float]] = defaultdict(lambda: {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0})
    for rid in ids:
        model = model_for_id[rid]
        ok = bool(matrix[model].get(rid, False))
        correct += 1 if ok else 0
        group = group_value(row_by_id[rid], group_key)
        by_group[group]["correct" if ok else "wrong"] += 1
    for stats in by_group.values():
        denom = stats["correct"] + stats["wrong"]
        stats["accuracy"] = stats["correct"] / denom if denom else 0.0
    return {
        "accuracy": correct / len(ids) if ids else 0.0,
        "by_group": dict(sorted(by_group.items())),
    }


def single_model_summaries(
    rows: list[dict[str, Any]],
    ids: list[str],
    matrix: dict[str, dict[str, bool]],
    group_key: str,
) -> list[dict[str, Any]]:
    summaries = []
    for model in sorted(matrix):
        choices = {rid: model for rid in ids}
        summary = summarize_boolean_choices(rows, ids, choices, matrix, group_key)
        summaries.append(
            {
                "model": model,
                "accuracy": summary["accuracy"],
                "by_group": summary["by_group"],
            }
        )
    summaries.sort(key=lambda row: (-float(row["accuracy"]), str(row["model"])))
    return summaries


def multilabel_tfidf_choices(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover
        return {rid: global_model for rid in eval_ids}, {"fallback": f"sklearn unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform([text_for_row(row_by_id[rid]) for rid in probe_ids])
    x_eval = vectorizer.transform([text_for_row(row_by_id[rid]) for rid in eval_ids])
    prob_columns: list[list[float]] = []
    fitted = 0
    for model in models:
        y = [1 if matrix[model].get(rid, False) else 0 for rid in probe_ids]
        prior = sum(y) / len(y) if y else 0.0
        if len(set(y)) < 2:
            prob_columns.append([prior] * len(eval_ids))
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(x_train, y)
        probs = clf.predict_proba(x_eval)[:, 1].tolist()
        prob_columns.append([0.75 * p + 0.25 * prior for p in probs])
        fitted += 1

    choices: dict[str, str] = {}
    for idx, rid in enumerate(eval_ids):
        best_idx = max(range(len(models)), key=lambda midx: prob_columns[midx][idx])
        choices[rid] = models[best_idx] if models else global_model
    return choices, {
        "models": len(models),
        "features": len(vectorizer.vocabulary_),
        "fitted_binary_classifiers": fitted,
    }


def multilabel_tfidf_metadata_choices(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover
        return {rid: global_model for rid in eval_ids}, {"fallback": f"sklearn unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    )
    x_eval = vectorizer.transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    )
    prob_columns: list[list[float]] = []
    fitted = 0
    for model in models:
        y = [1 if matrix[model].get(rid, False) else 0 for rid in probe_ids]
        prior = sum(y) / len(y) if y else 0.0
        if len(set(y)) < 2:
            prob_columns.append([prior] * len(eval_ids))
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(x_train, y)
        probs = clf.predict_proba(x_eval)[:, 1].tolist()
        prob_columns.append([0.75 * p + 0.25 * prior for p in probs])
        fitted += 1
    choices: dict[str, str] = {}
    for idx, rid in enumerate(eval_ids):
        best_idx = max(range(len(models)), key=lambda midx: prob_columns[midx][idx])
        choices[rid] = models[best_idx] if models else global_model
    return choices, {
        "models": len(models),
        "features": len(vectorizer.vocabulary_),
        "fitted_binary_classifiers": fitted,
        "metadata_keys": ",".join(metadata_keys),
    }


def knn_tfidf_choices(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbors: int,
    smoothing: float = 3.0,
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return {rid: global_model for rid in eval_ids}, {"fallback": f"sklearn/numpy unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    )
    x_eval = vectorizer.transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    )
    k = min(neighbors, len(probe_ids))
    y_by_model = {
        model: np.array([1.0 if matrix[model].get(rid, False) else 0.0 for rid in probe_ids])
        for model in models
    }
    priors = {model: float(y.mean()) for model, y in y_by_model.items()}
    choices: dict[str, str] = {}
    for idx, rid in enumerate(eval_ids):
        sims = (x_eval[idx] @ x_train.T).toarray().ravel()
        if k < len(sims):
            top_idx = np.argpartition(sims, -k)[-k:]
        else:
            top_idx = np.arange(len(sims))
        weights = sims[top_idx] + 1e-6
        denom = float(weights.sum() + smoothing)
        best_model = global_model
        best_score = -1.0
        for model in models:
            y = y_by_model[model][top_idx]
            score = float((weights * y).sum() + smoothing * priors[model]) / denom
            if score > best_score:
                best_model = model
                best_score = score
        choices[rid] = best_model
    return choices, {
        "neighbors": k,
        "smoothing": smoothing,
        "models": len(models),
        "features": len(vectorizer.vocabulary_),
        "metadata_keys": ",".join(metadata_keys),
    }


def complementarity_knn_tfidf_choices(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbors: int,
    harm_weight: float,
    smoothing: float = 5.0,
    min_switch_gain: float = 0.0,
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return {rid: global_model for rid in eval_ids}, {"fallback": f"sklearn/numpy unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    )
    x_eval = vectorizer.transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    )
    k = min(neighbors, len(probe_ids))
    global_correct = np.array(
        [1.0 if matrix[global_model].get(rid, False) else 0.0 for rid in probe_ids]
    )
    deltas_by_model: dict[str, Any] = {}
    priors: dict[str, float] = {}
    positive_prior_models = 0
    for model in models:
        model_correct = np.array([1.0 if matrix[model].get(rid, False) else 0.0 for rid in probe_ids])
        fixes_global_error = (model_correct == 1.0) & (global_correct == 0.0)
        breaks_global_success = (model_correct == 0.0) & (global_correct == 1.0)
        delta = fixes_global_error.astype(float) - harm_weight * breaks_global_success.astype(float)
        deltas_by_model[model] = delta
        priors[model] = float(delta.mean()) if len(delta) else 0.0
        positive_prior_models += 1 if priors[model] > 0 else 0

    choices: dict[str, str] = {}
    switches = 0
    for idx, rid in enumerate(eval_ids):
        sims = (x_eval[idx] @ x_train.T).toarray().ravel()
        if k < len(sims):
            top_idx = np.argpartition(sims, -k)[-k:]
        else:
            top_idx = np.arange(len(sims))
        weights = sims[top_idx] + 1e-6
        denom = float(weights.sum() + smoothing)
        best_model = global_model
        best_score = min_switch_gain
        for model in models:
            if model == global_model:
                continue
            delta = deltas_by_model[model][top_idx]
            score = float((weights * delta).sum() + smoothing * priors[model]) / denom
            if score > best_score:
                best_model = model
                best_score = score
        choices[rid] = best_model
        switches += 1 if best_model != global_model else 0

    return choices, {
        "neighbors": k,
        "smoothing": smoothing,
        "harm_weight": harm_weight,
        "min_switch_gain": min_switch_gain,
        "models": len(models),
        "positive_prior_models": positive_prior_models,
        "switch_rate": switches / len(eval_ids) if eval_ids else 0.0,
        "features": len(vectorizer.vocabulary_),
        "metadata_keys": ",".join(metadata_keys),
    }


def paired_lcb_complementarity_knn_tfidf_choices(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbors: int,
    harm_weight: float,
    z_value: float,
    smoothing: float = 5.0,
    min_switch_lcb: float = 0.0,
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return {rid: global_model for rid in eval_ids}, {"fallback": f"sklearn/numpy unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    model_index = {model: idx for idx, model in enumerate(models)}
    global_idx = model_index.get(global_model, 0)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    )
    x_eval = vectorizer.transform(
        [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    )
    k = min(neighbors, len(probe_ids))
    correctness = np.array(
        [[1.0 if matrix[model].get(rid, False) else 0.0 for rid in probe_ids] for model in models]
    )
    global_correct = correctness[global_idx]
    fixes_global_error = (correctness == 1.0) & (global_correct == 0.0)
    breaks_global_success = (correctness == 0.0) & (global_correct == 1.0)
    deltas = fixes_global_error.astype(float) - harm_weight * breaks_global_success.astype(float)
    priors = deltas.mean(axis=1) if deltas.shape[1] else np.zeros(len(models))
    second_priors = (deltas * deltas).mean(axis=1) if deltas.shape[1] else np.zeros(len(models))

    choices: dict[str, str] = {}
    switches = 0
    mean_lcbs: list[float] = []
    for idx, rid in enumerate(eval_ids):
        sims = (x_eval[idx] @ x_train.T).toarray().ravel()
        if k < len(sims):
            top_idx = np.argpartition(sims, -k)[-k:]
        else:
            top_idx = np.arange(len(sims))
        weights = sims[top_idx] + 1e-6
        sum_w = float(weights.sum())
        denom = sum_w + smoothing
        local_deltas = deltas[:, top_idx]
        weighted_sum = local_deltas @ weights
        weighted_sq = (local_deltas * local_deltas) @ weights
        mean = (weighted_sum + smoothing * priors) / denom
        second = (weighted_sq + smoothing * second_priors) / denom
        variance = np.maximum(second - mean * mean, 0.0)
        neff = (denom * denom) / float((weights * weights).sum() + smoothing)
        lcb = mean - z_value * np.sqrt(variance / max(neff, 1.0))
        lcb[global_idx] = min_switch_lcb
        best_idx = int(np.argmax(lcb))
        best_lcb = float(lcb[best_idx])
        if best_lcb > min_switch_lcb:
            model = models[best_idx]
        else:
            model = global_model
        choices[rid] = model
        switches += 1 if model != global_model else 0
        mean_lcbs.append(best_lcb)

    return choices, {
        "neighbors": k,
        "smoothing": smoothing,
        "harm_weight": harm_weight,
        "lcb_z": z_value,
        "min_switch_lcb": min_switch_lcb,
        "models": len(models),
        "positive_prior_models": int((priors > 0).sum()),
        "switch_rate": switches / len(eval_ids) if eval_ids else 0.0,
        "mean_best_lcb": sum(mean_lcbs) / len(mean_lcbs) if mean_lcbs else 0.0,
        "features": len(vectorizer.vocabulary_),
        "metadata_keys": ",".join(metadata_keys),
    }


def choices_for_strategy(
    method: str,
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    wilson_z: float,
    margin: float,
    metadata_keys: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    full_ids = [row_id(row) for row in rows]
    best_full_model, best_full_acc = best_model_for_ids(matrix, full_ids)
    global_model, global_probe_acc, global_probe_lb = best_model_with_wilson(
        matrix, probe_ids, wilson_z
    )

    if method == "probe_current_router":
        current = current_choice_map(rows, matrix, best_full_model)
        return {rid: current[rid] for rid in eval_ids}, {
            "global_model_from_probe": global_model,
            "global_probe_accuracy": global_probe_acc,
            "best_full_model": best_full_model,
            "best_full_accuracy": best_full_acc,
        }
    if method == "probe_global_best":
        return {rid: global_model for rid in eval_ids}, {
            "global_model_from_probe": global_model,
            "global_probe_accuracy": global_probe_acc,
            "global_probe_wilson_lower": global_probe_lb,
            "best_full_model": best_full_model,
            "best_full_accuracy": best_full_acc,
        }
    if method == "probe_risk_parse_conf_fallback":
        choices, meta = parse_or_confidence_fallback_choices(
            rows, probe_ids, eval_ids, matrix, global_model, best_full_model
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta
    if method == "probe_multilabel_tfidf":
        choices, meta = multilabel_tfidf_choices(rows, probe_ids, eval_ids, matrix, global_model)
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta
    if method == "probe_multilabel_tfidf_metadata":
        choices, meta = multilabel_tfidf_metadata_choices(
            rows, probe_ids, eval_ids, matrix, global_model, metadata_keys
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta
    if method.startswith("probe_knn_tfidf_k"):
        neighbors = int(method[len("probe_knn_tfidf_k") :])
        choices, meta = knn_tfidf_choices(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            global_model,
            metadata_keys,
            neighbors=neighbors,
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta
    if method.startswith("probe_complementarity_knn_tfidf_k"):
        suffix = method[len("probe_complementarity_knn_tfidf_k") :]
        k_text, harm_text = suffix.split("_h", 1)
        choices, meta = complementarity_knn_tfidf_choices(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            global_model,
            metadata_keys,
            neighbors=int(k_text),
            harm_weight=parse_harm_weight_tag(harm_text),
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta
    if method.startswith("probe_paired_lcb_complementarity_knn_tfidf_k"):
        suffix = method[len("probe_paired_lcb_complementarity_knn_tfidf_k") :]
        k_text, rest = suffix.split("_h", 1)
        harm_text, z_text = rest.split("_z", 1)
        choices, meta = paired_lcb_complementarity_knn_tfidf_choices(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            global_model,
            metadata_keys,
            neighbors=int(k_text),
            harm_weight=parse_harm_weight_tag(harm_text),
            z_value=parse_float_tag(z_text),
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta

    if method.startswith("probe_craft_current_repair_lcb_k"):
        suffix = method[len("probe_craft_current_repair_lcb_k") :]
        k_text, rest = suffix.split("_h", 1)
        harm_text, z_text = rest.split("_z", 1)
        choices, meta = craft_current_repair_lcb_choices(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            best_full_model,
            metadata_keys,
            neighbors=int(k_text),
            harm_weight=parse_harm_weight_tag(harm_text),
            z_value=parse_float_tag(z_text),
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta

    if method.startswith("probe_dynamic_subject_tfidf_c"):
        clusters = int(method[len("probe_dynamic_subject_tfidf_c") :])
        choices, meta = dynamic_subject_tfidf_choices(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            global_model,
            clusters=clusters,
            seed=seed,
            z=wilson_z,
        )
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return choices, meta

    if method.startswith("probe_robust_minimax_") and method.endswith("_global"):
        key = method[len("probe_robust_minimax_") : -len("_global")]
        robust_model, meta = robust_minimax_model(rows, probe_ids, matrix, key, wilson_z)
        meta.update({"global_model_from_probe": global_model, "best_full_model": best_full_model})
        return {rid: robust_model for rid in eval_ids}, meta

    if method.startswith("probe_failure_boundary_") and method.endswith("_guard"):
        key = method[len("probe_failure_boundary_") : -len("_guard")]
        robust_model, robust_meta = robust_minimax_model(rows, probe_ids, matrix, key, wilson_z)
        choices, meta = failure_boundary_group_choices(
            rows=rows,
            probe_ids=probe_ids,
            eval_ids=eval_ids,
            matrix=matrix,
            key=key,
            guard_model=robust_model,
            fallback_model=best_full_model,
            z=wilson_z,
            margin=margin,
            harm_weight=1.25,
        )
        meta.update(
            {
                "global_model_from_probe": global_model,
                "best_full_model": best_full_model,
                "robust_minimax": robust_meta,
            }
        )
        return choices, meta

    if method.startswith("probe_risk_controlled_") and method.endswith("_mapping"):
        key = method[len("probe_risk_controlled_") : -len("_mapping")]
        mapping = risk_controlled_group_mapping(
            rows, probe_ids, matrix, key, global_model, wilson_z, margin
        )
        return choices_from_group_mapping(rows, eval_ids, key, mapping), {
            "mapping_key": key,
            "global_model_from_probe": global_model,
            "best_full_model": best_full_model,
        }

    if method.startswith("probe_") and method.endswith("_mapping"):
        key = method[len("probe_") : -len("_mapping")]
        mapping = group_mapping_from_probe(rows, probe_ids, matrix, key, global_model, wilson_z)
        return choices_from_group_mapping(rows, eval_ids, key, mapping), {
            "mapping_key": key,
            "global_model_from_probe": global_model,
            "best_full_model": best_full_model,
        }

    raise ValueError(f"Unsupported strategy method: {method}")


def materialized_rows(
    rows: list[dict[str, Any]],
    eval_ids: list[str],
    choices: dict[str, str],
    full_predictions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    row_by_id = {row_id(row): row for row in rows}
    out_rows: list[dict[str, Any]] = []
    for rid in eval_ids:
        source = dict(row_by_id[rid])
        model = choices[rid]
        cached = full_predictions[model][rid]
        item = dict(source)
        item["original_routed_model"] = source.get("routed_model")
        item["original_is_correct"] = is_correct_row(source)
        item["improved_routed_model"] = model
        item["improved_is_correct"] = is_correct_row(cached)
        item["improved_prediction"] = cached.get("pred", cached.get("prediction"))
        item["improved_response"] = cached.get("model_outputs", cached.get("response", ""))
        item["improved_source_id"] = rid
        out_rows.append(item)
    return out_rows


def render_bench_harness(
    path: Path,
    case: CaseSpec,
    strategy: dict[str, Any],
    rows: list[dict[str, Any]],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    improved_rows: list[dict[str, Any]],
    improved_summary: dict[str, Any],
    primary_group_key: str,
    current_eval_summary: dict[str, Any],
) -> None:
    single_rows = single_model_summaries(rows, eval_ids, matrix, primary_group_key)
    best_single = single_rows[0]
    group_columns = sorted(improved_summary.get(primary_group_key, {})) + ["Average"]
    name_width = 40
    col_width = 18
    counts = []
    for group in group_columns[:-1]:
        stats = improved_summary[primary_group_key][group]
        counts.append(str(int(stats["correct"] + stats["wrong"])))
    counts.append(str(int(improved_summary["examples"])))
    lines = [
        "=" * 100,
        f"Bench-Harness: Innovation Bench-CoE -> {case.case_id}",
        "=" * 100,
        f"| Benchmark: {case.benchmark}",
        f"| Strategy: {strategy['method']}",
        f"| Budget per {strategy['split_key']}: {strategy['budget_per_group']}",
        f"| Probe samples: {strategy['probe_samples']}",
        f"| Eval heldout samples: {strategy['eval_samples']}",
        f"| Single model source: {case.single_path}",
        "",
        table_row(name_width, col_width, "Model / Metric", group_columns),
        table_row(name_width, col_width, "-" * 28, ["-" * 12 for _ in group_columns]),
        table_row(name_width, col_width, "Qs (Count)", counts),
        table_row(name_width, col_width, "-" * 28, ["-" * 12 for _ in group_columns]),
    ]
    for item in single_rows:
        prefix = "* " if item["model"] == best_single["model"] else "  "
        values = [
            fmt_pct(item["by_group"].get(group, {}).get("accuracy"))
            if group in item["by_group"]
            else "N/A"
            for group in group_columns[:-1]
        ]
        values.append(fmt_pct(float(item["accuracy"])))
        lines.append(table_row(name_width, col_width, prefix + item["model"], values))
    lines.append(table_row(name_width, col_width, "-" * 28, ["-" * 12 for _ in group_columns]))

    current_values = [
        fmt_pct(current_eval_summary["by_group"].get(group, {}).get("accuracy"))
        if group in current_eval_summary["by_group"]
        else "N/A"
        for group in group_columns[:-1]
    ]
    current_values.append(fmt_pct(float(current_eval_summary["accuracy"])))
    lines.append(table_row(name_width, col_width, "Original Bench-CoE (heldout)", current_values))

    improved_values = [
        fmt_pct(improved_summary[primary_group_key][group]["accuracy"])
        for group in group_columns[:-1]
    ]
    improved_acc = float(improved_summary["total"]["accuracy"])
    improved_values.append(fmt_pct(improved_acc))
    lines.append(table_row(name_width, col_width, "Innovation Bench-CoE (heldout)", improved_values))
    lines.append(
        table_row(
            name_width,
            col_width,
            "Gain (vs Best Single)",
            [""] * (len(group_columns) - 1)
            + [f"{(improved_acc - float(best_single['accuracy'])) * 100:+.2f}%"],
        )
    )
    lines.append("")
    lines.append("Improved routed models:")
    for model, count in Counter(row["improved_routed_model"] for row in improved_rows).most_common():
        lines.append(f"- {model}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {path.parent / 'test_predictions.json'}")
    write_text(path, lines)


def materialize_one(
    case: CaseSpec,
    strategy_row: dict[str, Any],
    output_root: Path,
    seed: int,
    wilson_z: float,
    margin: float,
) -> dict[str, Any]:
    rows = route_rows(case)
    ids = [row_id(row) for row in rows]
    full_predictions = load_full_predictions(case)
    matrix_raw = bool_matrix(full_predictions)
    matrix, missing_counts = keep_complete_models(matrix_raw, ids)
    full_predictions = {model: full_predictions[model] for model in matrix}
    if not matrix:
        raise RuntimeError(f"No complete model cache for {case.case_id}")

    budget = int(float(strategy_row["best_probe_budget"]))
    split_key = str(strategy_row["best_probe_split_key"] or case.primary_group_key)
    method = str(strategy_row["best_probe_method"])
    case_seed = seed + (zlib.crc32(case.case_id.encode("utf-8")) % 100000)
    probe_ids, eval_ids = deterministic_probe_split(rows, split_key, budget, case_seed)
    choices, meta = choices_for_strategy(
        method,
        rows,
        probe_ids,
        eval_ids,
        matrix,
        wilson_z,
        margin,
        case.group_keys,
        case_seed,
    )
    improved = materialized_rows(rows, eval_ids, choices, full_predictions)
    summary = summarize_rows(improved, case.group_keys)
    summary["case_id"] = case.case_id
    summary["benchmark"] = case.benchmark
    summary["strategy"] = {
        "method": method,
        "budget_per_group": budget,
        "split_key": split_key,
        "probe_samples": len(probe_ids),
        "eval_samples": len(eval_ids),
        "seed": case_seed,
        **meta,
    }
    current_choices = current_choice_map(rows, matrix, best_model_for_ids(matrix, ids)[0])
    current_eval_summary = summarize_boolean_choices(
        rows, eval_ids, current_choices, matrix, case.primary_group_key
    )
    summary["original_bench_coe_heldout"] = current_eval_summary
    best_eval_model, best_eval_acc = best_model_for_ids(matrix, eval_ids)
    summary["best_single_heldout"] = {
        "model": best_eval_model,
        "accuracy": best_eval_acc,
    }
    summary["gain_vs_best_single_heldout"] = float(summary["total"]["accuracy"]) - best_eval_acc

    out_dir = output_root / f"{case.case_id}_{method}_budget{budget}"
    write_json(out_dir / "probe_ids.json", probe_ids)
    write_json(out_dir / "eval_ids.json", eval_ids)
    write_json(out_dir / "strategy.json", summary["strategy"])
    write_json(out_dir / "test_predictions.json", improved)
    write_json(out_dir / "test_summary.json", summary)
    render_bench_harness(
        out_dir / f"Bench_Harness_Result_{case.case_id}_{method}.txt",
        case,
        summary["strategy"],
        rows,
        eval_ids,
        matrix,
        improved,
        summary,
        case.primary_group_key,
        current_eval_summary,
    )
    return {
        "case_id": case.case_id,
        "benchmark": case.benchmark,
        "method": method,
        "budget_per_group": budget,
        "probe_samples": len(probe_ids),
        "eval_samples": len(eval_ids),
        "accuracy": summary["total"]["accuracy"],
        "original_bench_coe_heldout": current_eval_summary["accuracy"],
        "best_single_heldout": best_eval_acc,
        "best_single_model_heldout": best_eval_model,
        "gain_vs_best_single_heldout": summary["gain_vs_best_single_heldout"],
        "output_dir": str(out_dir),
    }


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Materialized Innovation Bench-CoE Strategies",
        "",
        "| Case | Method | Heldout Acc | Original Heldout | Best Single Heldout | Gain | Output |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_id']}`",
                    f"`{row['method']}`",
                    fmt_pct(float(row["accuracy"])),
                    fmt_pct(float(row["original_bench_coe_heldout"])),
                    f"{fmt_pct(float(row['best_single_heldout']))} ({row['best_single_model_heldout']})",
                    f"{float(row['gain_vs_best_single_heldout']) * 100:+.2f}%",
                    f"`{row['output_dir']}`",
                ]
            )
            + " |"
        )
    write_text(path, lines)


def main() -> None:
    args = parse_args()
    strategies = read_strategy_summary(args.strategy_summary)
    cases = select_cases(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.case_id not in strategies:
            raise KeyError(f"Missing strategy summary row for {case.case_id}")
        print(f"[materialize] {case.case_id}: {strategies[case.case_id]['best_probe_method']}")
        rows.append(
            materialize_one(
                case=case,
                strategy_row=strategies[case.case_id],
                output_root=args.output_dir,
                seed=args.seed,
                wilson_z=args.wilson_z,
                margin=args.risk_margin,
            )
        )
    write_csv(args.output_dir / "materialized_summary.csv", rows)
    write_json(args.output_dir / "materialized_summary.json", rows)
    write_summary_md(args.output_dir / "materialized_summary.md", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
