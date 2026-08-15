from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_SINGLE_ROOT = Path("outputs/model_benchmarks/official_code_local_models")
MM_SINGLE_ROOT = Path("outputs/multimodal_babyvision_models")
MMLU_RESULTS_ROOT = Path("MMLU-Pro/results")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    benchmark: str
    route_dir: Path
    single_kind: str
    single_path: Path
    group_keys: tuple[str, ...]
    primary_group_key: str


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "gaokao_bert_mmlu_pro",
        "mmlu_pro",
        Path("outputs/bench_coe/mmlu_pro_subject_bert_bench_coe_gaokao10epoch_front4"),
        "mmlu_pro",
        MMLU_RESULTS_ROOT,
        ("category", "src", "routed_subject"),
        "category",
    ),
    CaseSpec(
        "gaokao_bert_bbh",
        "bbh",
        Path("outputs/bench_coe/bbh_subject_bert_bench_coe_gaokao10epoch_front4"),
        "jsonl",
        TEXT_SINGLE_ROOT / "bbh",
        ("task", "routed_subject"),
        "task",
    ),
    CaseSpec(
        "gaokao_bert_gpqa",
        "gpqa",
        Path("outputs/bench_coe/gpqa_diamond_subject_bert_bench_coe_gaokao10epoch_front4"),
        "jsonl",
        TEXT_SINGLE_ROOT / "gpqa",
        ("domain", "subdomain", "routed_subject"),
        "domain",
    ),
    CaseSpec(
        "gaokao_bert_mmstar",
        "mmstar_text_only",
        Path("outputs/bench_coe/mmstar_text_only_subject_bert_bench_coe_gaokao10epoch_front4"),
        "jsonl",
        TEXT_SINGLE_ROOT / "mmstar_text_only",
        ("category", "l2_category", "routed_subject"),
        "category",
    ),
    CaseSpec(
        "qwen3vl_mmlu_pro",
        "mmlu_pro",
        Path("outputs/bench_coe/mmlu_pro_qwen3vl_gaokao_mm_router_front4"),
        "mmlu_pro",
        MMLU_RESULTS_ROOT,
        ("category", "src", "routed_subject"),
        "category",
    ),
    CaseSpec(
        "qwen3vl_bbh",
        "bbh",
        Path("outputs/bench_coe/bbh_qwen3vl_gaokao_mm_router_front4"),
        "jsonl",
        TEXT_SINGLE_ROOT / "bbh",
        ("task", "routed_subject"),
        "task",
    ),
    CaseSpec(
        "qwen3vl_gpqa",
        "gpqa",
        Path("outputs/bench_coe/gpqa_diamond_qwen3vl_gaokao_mm_router_front4"),
        "jsonl",
        TEXT_SINGLE_ROOT / "gpqa",
        ("domain", "subdomain", "routed_subject"),
        "domain",
    ),
    CaseSpec(
        "qwen3vl_mmstar",
        "mmstar_text_only",
        Path("outputs/bench_coe/mmstar_text_only_qwen3vl_gaokao_mm_router_front4"),
        "jsonl",
        TEXT_SINGLE_ROOT / "mmstar_text_only",
        ("category", "l2_category", "routed_subject"),
        "category",
    ),
    CaseSpec(
        "qwen3vl_cmmmu",
        "cmmmu",
        Path("outputs/bench_coe/cmmmu_qwen3vl_gaokao_mm_router_front4"),
        "json",
        MM_SINGLE_ROOT / "cmmmu" / "val",
        ("category", "subcategory", "difficulty", "type", "routed_subject"),
        "category",
    ),
    CaseSpec(
        "qwen3vl_mathvista",
        "mathvista",
        Path("outputs/bench_coe/mathvista_qwen3vl_gaokao_mm_router_front4"),
        "json",
        MM_SINGLE_ROOT / "mathvista" / "testmini",
        ("category", "task", "context", "grade", "question_type", "answer_type", "skills", "routed_subject"),
        "task",
    ),
    CaseSpec(
        "qwen3vl_mmmu_pro",
        "mmmu_pro",
        Path("outputs/bench_coe/mmmu_pro_standard_10_options_qwen3vl_gaokao_mm_router_front4"),
        "json",
        MM_SINGLE_ROOT / "mmmu_pro" / "standard_10_options" / "test",
        ("domain", "subject", "difficulty", "img_type", "routed_subject"),
        "domain",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline Bench-CoE routability, risk-control, probe-mapping, and lightweight multi-label experiments."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/innovation_experiments"))
    parser.add_argument("--cases", default="all", help="Comma-separated case ids, or all.")
    parser.add_argument("--budgets", default="5,10,20", help="Probe samples per group.")
    parser.add_argument(
        "--probe-split-keys",
        default="primary",
        help="Probe split keys: primary, all, or a comma-separated list of row metadata keys.",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--risk-margin", type=float, default=0.0)
    parser.add_argument("--wilson-z", type=float, default=1.0)
    parser.add_argument("--skip-multilabel", action="store_true")
    parser.add_argument(
        "--only-complementarity",
        action="store_true",
        help="Evaluate only current/global baselines plus the complementarity-weighted probe router.",
    )
    parser.add_argument(
        "--only-paired-lcb",
        action="store_true",
        help="Evaluate only current/global baselines plus the paired local LCB complementarity router.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_id(row: dict[str, Any]) -> str:
    value = row.get("id", row.get("question_id"))
    if value is None:
        raise KeyError(f"Missing row id in keys={sorted(row)}")
    return str(value)


def is_correct_row(row: dict[str, Any]) -> bool:
    if row.get("is_correct") is not None:
        return bool(row["is_correct"])
    pred = row.get("pred", row.get("prediction"))
    gold = row.get("answer", row.get("target"))
    return pred is not None and str(pred).strip() == str(gold).strip()


def route_row_correct(row: dict[str, Any]) -> bool:
    return is_correct_row(row)


def text_for_row(row: dict[str, Any]) -> str:
    if row.get("route_text"):
        return str(row["route_text"])
    if row.get("prompt"):
        return str(row["prompt"])
    if row.get("input"):
        return str(row["input"])
    if row.get("question"):
        pieces = [str(row["question"])]
        options = row.get("options")
        if isinstance(options, list):
            pieces.extend(str(opt) for opt in options)
        return "\n".join(pieces)
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def augmented_text_for_row(row: dict[str, Any], metadata_keys: tuple[str, ...]) -> str:
    meta_parts: list[str] = []
    for key in metadata_keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            token = str(item).replace(" ", "_").replace("\n", "_")
            meta_parts.append(f"meta_{key}={token}")
    base = text_for_row(row)
    if not meta_parts:
        return base
    return " ".join(meta_parts) + "\n" + base


def group_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        return "Unknown"
    if isinstance(value, list):
        if not value:
            return "Unknown"
        return "|".join(str(item) for item in value)
    return str(value)


def load_route_rows(case: CaseSpec) -> list[dict[str, Any]]:
    path = case.route_dir / "test_predictions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a row list.")
    return rows


def load_jsonl_single_predictions(root: Path) -> dict[str, dict[str, bool]]:
    matrix: dict[str, dict[str, bool]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        path = model_dir / "predictions.jsonl"
        summary = model_dir / "summary.json"
        if not path.exists() or not summary.exists():
            continue
        try:
            summary_payload = read_json(summary)
        except Exception:
            summary_payload = {}
        if summary_payload.get("status") not in (None, "completed"):
            continue
        model = model_dir.name
        matrix[model] = {row_id(row): is_correct_row(row) for row in read_jsonl(path)}
    return matrix


def load_json_single_predictions(root: Path) -> dict[str, dict[str, bool]]:
    matrix: dict[str, dict[str, bool]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        path = model_dir / "predictions.json"
        summary = model_dir / "summary.json"
        if not path.exists() or not summary.exists():
            continue
        try:
            summary_payload = read_json(summary)
        except Exception:
            summary_payload = {}
        if summary_payload.get("status") not in (None, "completed"):
            continue
        model = model_dir.name
        matrix[model] = {row_id(row): is_correct_row(row) for row in read_json(path)}
    return matrix


def load_mmlu_single_predictions(root: Path) -> dict[str, dict[str, bool]]:
    matrix: dict[str, dict[str, bool]] = {}
    for model_dir in sorted(root.iterdir()):
        result_dir = model_dir / "CoT" / "all"
        if not result_dir.is_dir():
            continue
        model = model_dir.name
        model_rows: dict[str, bool] = {}
        for path in sorted(result_dir.glob("*.json")):
            try:
                rows = read_json(path)
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                model_rows[row_id(row)] = is_correct_row(row)
        if model_rows:
            matrix[model] = model_rows
    return matrix


def load_single_matrix(case: CaseSpec) -> dict[str, dict[str, bool]]:
    if case.single_kind == "jsonl":
        return load_jsonl_single_predictions(case.single_path)
    if case.single_kind == "json":
        return load_json_single_predictions(case.single_path)
    if case.single_kind == "mmlu_pro":
        return load_mmlu_single_predictions(case.single_path)
    raise ValueError(case.single_kind)


def keep_complete_models(
    matrix: dict[str, dict[str, bool]],
    ids: list[str],
    min_coverage: float = 0.999,
) -> tuple[dict[str, dict[str, bool]], dict[str, int]]:
    needed = set(ids)
    kept: dict[str, dict[str, bool]] = {}
    missing_counts: dict[str, int] = {}
    for model, values in matrix.items():
        missing = len(needed.difference(values))
        missing_counts[model] = missing
        if len(ids) == 0 or missing / len(ids) <= 1.0 - min_coverage:
            kept[model] = values
    return kept, missing_counts


def accuracy_for_model(values: dict[str, bool], ids: list[str]) -> float:
    if not ids:
        return 0.0
    return sum(1 for rid in ids if values.get(rid, False)) / len(ids)


def accuracy_for_choices(
    ids: list[str],
    model_for_id: dict[str, str],
    matrix: dict[str, dict[str, bool]],
) -> float:
    if not ids:
        return 0.0
    correct = 0
    for rid in ids:
        model = model_for_id[rid]
        if matrix[model].get(rid, False):
            correct += 1
    return correct / len(ids)


def best_model_for_ids(matrix: dict[str, dict[str, bool]], ids: list[str]) -> tuple[str, float]:
    best_model = ""
    best_acc = -1.0
    for model, values in sorted(matrix.items()):
        acc = accuracy_for_model(values, ids)
        if acc > best_acc:
            best_model = model
            best_acc = acc
    return best_model, best_acc


def current_choice_map(rows: list[dict[str, Any]], matrix: dict[str, dict[str, bool]], best_fallback: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for row in rows:
        model = str(row.get("routed_model") or "")
        if model not in matrix:
            model = best_fallback
        choices[row_id(row)] = model
    return choices


def group_oracle(
    rows: list[dict[str, Any]],
    ids: list[str],
    matrix: dict[str, dict[str, bool]],
    key: str,
) -> dict[str, Any]:
    by_group: dict[str, list[str]] = defaultdict(list)
    row_by_id = {row_id(row): row for row in rows}
    for rid in ids:
        by_group[group_value(row_by_id[rid], key)].append(rid)
    group_to_model: dict[str, str] = {}
    choices: dict[str, str] = {}
    for group, group_ids in sorted(by_group.items()):
        model, _ = best_model_for_ids(matrix, group_ids)
        group_to_model[group] = model
        for rid in group_ids:
            choices[rid] = model
    return {
        "accuracy": accuracy_for_choices(ids, choices, matrix),
        "num_groups": len(by_group),
        "group_to_model": group_to_model,
    }


def instance_oracle_accuracy(matrix: dict[str, dict[str, bool]], ids: list[str]) -> float:
    if not ids:
        return 0.0
    return sum(1 for rid in ids if any(values.get(rid, False) for values in matrix.values())) / len(ids)


def wilson_lower(correct: int, total: int, z: float) -> float:
    if total <= 0:
        return 0.0
    phat = correct / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (centre - radius) / denom


def deterministic_probe_split(
    rows: list[dict[str, Any]],
    key: str,
    budget_per_group: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_group[group_value(row, key)].append(row_id(row))
    probe: list[str] = []
    eval_ids: list[str] = []
    for group, group_ids in sorted(by_group.items()):
        shuffled = list(group_ids)
        rng.shuffle(shuffled)
        take = min(budget_per_group, max(0, len(shuffled) - 1))
        probe.extend(shuffled[:take])
        eval_ids.extend(shuffled[take:])
    return sorted(probe), sorted(eval_ids)


def best_model_with_wilson(matrix: dict[str, dict[str, bool]], ids: list[str], z: float) -> tuple[str, float, float]:
    best_model = ""
    best_lb = -1.0
    best_acc = -1.0
    for model, values in sorted(matrix.items()):
        correct = sum(1 for rid in ids if values.get(rid, False))
        acc = correct / len(ids) if ids else 0.0
        lb = wilson_lower(correct, len(ids), z)
        if lb > best_lb or (lb == best_lb and acc > best_acc):
            best_model = model
            best_lb = lb
            best_acc = acc
    return best_model, best_acc, best_lb


def group_mapping_from_probe(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    key: str,
    global_model: str,
    z: float,
) -> dict[str, str]:
    row_by_id = {row_id(row): row for row in rows}
    by_group: dict[str, list[str]] = defaultdict(list)
    for rid in probe_ids:
        by_group[group_value(row_by_id[rid], key)].append(rid)
    mapping: dict[str, str] = {}
    for group, ids in by_group.items():
        if ids:
            model, _, _ = best_model_with_wilson(matrix, ids, z)
            mapping[group] = model
    return defaultdict(lambda: global_model, mapping)


def choices_from_group_mapping(
    rows: list[dict[str, Any]],
    ids: list[str],
    key: str,
    mapping: dict[str, str],
) -> dict[str, str]:
    row_by_id = {row_id(row): row for row in rows}
    return {rid: mapping[group_value(row_by_id[rid], key)] for rid in ids}


def risk_controlled_group_mapping(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    key: str,
    global_model: str,
    z: float,
    margin: float,
) -> dict[str, str]:
    row_by_id = {row_id(row): row for row in rows}
    by_group: dict[str, list[str]] = defaultdict(list)
    for rid in probe_ids:
        by_group[group_value(row_by_id[rid], key)].append(rid)
    global_correct = sum(1 for rid in probe_ids if matrix[global_model].get(rid, False))
    global_lb = wilson_lower(global_correct, len(probe_ids), z)
    mapping: dict[str, str] = {}
    for group, ids in by_group.items():
        model, _, lb = best_model_with_wilson(matrix, ids, z)
        mapping[group] = model if lb > global_lb + margin else global_model
    return defaultdict(lambda: global_model, mapping)


def parse_or_confidence_fallback_choices(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    best_fallback: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    thresholds = [None, 0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
    row_by_id = {row_id(row): row for row in rows}

    def build(ids: list[str], threshold: float | None) -> dict[str, str]:
        choices: dict[str, str] = {}
        for rid in ids:
            row = row_by_id[rid]
            model = str(row.get("routed_model") or "")
            if model not in matrix:
                model = best_fallback
            fallback = False
            if row.get("route_parse_ok") is False:
                fallback = True
            confidence = row.get("route_confidence")
            if threshold is not None and confidence is not None:
                try:
                    fallback = fallback or float(confidence) < threshold
                except Exception:
                    pass
            choices[rid] = global_model if fallback else model
        return choices

    best_threshold = None
    best_probe_acc = -1.0
    for threshold in thresholds:
        choices = build(probe_ids, threshold)
        acc = accuracy_for_choices(probe_ids, choices, matrix) if probe_ids else 0.0
        if acc > best_probe_acc:
            best_probe_acc = acc
            best_threshold = threshold
    return build(eval_ids, best_threshold), {
        "threshold": best_threshold,
        "probe_accuracy": best_probe_acc,
    }


def run_multilabel_tfidf(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
) -> tuple[float | None, dict[str, Any]]:
    if len(probe_ids) < 30 or len(eval_ids) < 10:
        return None, {"skipped": "not enough probe/eval rows"}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover
        return None, {"skipped": f"sklearn unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    train_texts = [text_for_row(row_by_id[rid]) for rid in probe_ids]
    eval_texts = [text_for_row(row_by_id[rid]) for rid in eval_ids]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)

    priors = []
    prob_columns = []
    fitted = 0
    for model in models:
        y = [1 if matrix[model].get(rid, False) else 0 for rid in probe_ids]
        prior = sum(y) / len(y)
        priors.append(prior)
        if len(set(y)) < 2:
            prob_columns.append([prior] * len(eval_ids))
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(x_train, y)
        probs = clf.predict_proba(x_eval)[:, 1].tolist()
        # Blend with the probe prior to reduce overfitting under tiny budgets.
        prob_columns.append([0.75 * p + 0.25 * prior for p in probs])
        fitted += 1

    choices: dict[str, str] = {}
    for idx, rid in enumerate(eval_ids):
        best_idx = max(range(len(models)), key=lambda midx: prob_columns[midx][idx])
        choices[rid] = models[best_idx] if models else global_model
    acc = accuracy_for_choices(eval_ids, choices, matrix)
    return acc, {
        "fitted_binary_classifiers": fitted,
        "models": len(models),
        "features": len(vectorizer.vocabulary_),
    }


def run_multilabel_tfidf_metadata(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
) -> tuple[float | None, dict[str, Any]]:
    if len(probe_ids) < 30 or len(eval_ids) < 10:
        return None, {"skipped": "not enough probe/eval rows"}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover
        return None, {"skipped": f"sklearn unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    train_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    eval_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)

    prob_columns = []
    fitted = 0
    for model in models:
        y = [1 if matrix[model].get(rid, False) else 0 for rid in probe_ids]
        prior = sum(y) / len(y)
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
    acc = accuracy_for_choices(eval_ids, choices, matrix)
    return acc, {
        "fitted_binary_classifiers": fitted,
        "models": len(models),
        "features": len(vectorizer.vocabulary_),
        "metadata_keys": ",".join(metadata_keys),
    }


def run_knn_tfidf(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbors: int,
    smoothing: float = 3.0,
) -> tuple[float | None, dict[str, Any]]:
    if len(probe_ids) < 5 or len(eval_ids) < 10:
        return None, {"skipped": "not enough probe/eval rows"}
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return None, {"skipped": f"sklearn/numpy unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    train_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    eval_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
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
    acc = accuracy_for_choices(eval_ids, choices, matrix)
    return acc, {
        "neighbors": k,
        "smoothing": smoothing,
        "models": len(models),
        "features": len(vectorizer.vocabulary_),
        "metadata_keys": ",".join(metadata_keys),
    }


def harm_weight_tag(value: float) -> str:
    return str(value).replace(".", "p")


def parse_harm_weight_tag(value: str) -> float:
    return float(value.replace("p", "."))


def float_tag(value: float) -> str:
    return str(value).replace(".", "p")


def parse_float_tag(value: str) -> float:
    return float(value.replace("p", "."))


def run_complementarity_knn_tfidf(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbors: int,
    harm_weight: float = 1.0,
    smoothing: float = 5.0,
    min_switch_gain: float = 0.0,
) -> tuple[float | None, dict[str, Any]]:
    """Route only when a local expert shows positive net complementarity vs global_model."""
    if len(probe_ids) < 5 or len(eval_ids) < 10:
        return None, {"skipped": "not enough probe/eval rows"}
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return None, {"skipped": f"sklearn/numpy unavailable: {exc}"}

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    train_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    eval_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
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

    acc = accuracy_for_choices(eval_ids, choices, matrix)
    return acc, {
        "neighbors": k,
        "smoothing": smoothing,
        "harm_weight": harm_weight,
        "min_switch_gain": min_switch_gain,
        "models": len(models),
        "positive_prior_models": positive_prior_models,
        "switch_rate": switches / len(eval_ids) if eval_ids else 0.0,
        "features": len(vectorizer.vocabulary_),
        "metadata_keys": ",".join(metadata_keys),
        "global_model": global_model,
    }


def run_complementarity_knn_tfidf_grid(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbor_values: tuple[int, ...],
    harm_weights: tuple[float, ...],
    smoothing: float = 5.0,
    min_switch_gain: float = 0.0,
) -> list[dict[str, Any]]:
    if len(probe_ids) < 5 or len(eval_ids) < 10:
        return [
            {
                "method": "probe_complementarity_knn_tfidf_grid",
                "accuracy": "",
                "meta_skipped": "not enough probe/eval rows",
            }
        ]
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return [
            {
                "method": "probe_complementarity_knn_tfidf_grid",
                "accuracy": "",
                "meta_skipped": f"sklearn/numpy unavailable: {exc}",
            }
        ]

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    model_index = {model: idx for idx, model in enumerate(models)}
    global_idx = model_index.get(global_model, 0)
    train_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    eval_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
    sims_matrix = (x_eval @ x_train.T).toarray()
    max_neighbors = min(max(neighbor_values), len(probe_ids))
    top_indices: list[Any] = []
    top_weights: list[Any] = []
    for sims in sims_matrix:
        if max_neighbors < len(sims):
            idx = np.argpartition(sims, -max_neighbors)[-max_neighbors:]
        else:
            idx = np.arange(len(sims))
        order = np.argsort(sims[idx])[::-1]
        idx = idx[order]
        top_indices.append(idx)
        top_weights.append(sims[idx] + 1e-6)
    top_indices_arr = np.vstack(top_indices)
    top_weights_arr = np.vstack(top_weights)

    correctness = np.array(
        [[1.0 if matrix[model].get(rid, False) else 0.0 for rid in probe_ids] for model in models]
    )
    eval_correctness = np.array(
        [[1.0 if matrix[model].get(rid, False) else 0.0 for rid in eval_ids] for model in models]
    )
    eval_rows: list[dict[str, Any]] = []
    global_correct = correctness[global_idx]
    for harm_weight in harm_weights:
        fixes_global_error = (correctness == 1.0) & (global_correct == 0.0)
        breaks_global_success = (correctness == 0.0) & (global_correct == 1.0)
        deltas = fixes_global_error.astype(float) - harm_weight * breaks_global_success.astype(float)
        priors = deltas.mean(axis=1) if deltas.shape[1] else np.zeros(len(models))
        positive_prior_models = int((priors > 0).sum())
        for neighbors in neighbor_values:
            if len(probe_ids) < max(5, neighbors):
                continue
            k = min(neighbors, len(probe_ids))
            choices: dict[str, str] = {}
            switches = 0
            for rid, idx, weights in zip(eval_ids, top_indices, top_weights):
                local_idx = idx[:k]
                local_weights = weights[:k]
                denom = float(local_weights.sum() + smoothing)
                scores = (deltas[:, local_idx] @ local_weights + smoothing * priors) / denom
                scores[global_idx] = min_switch_gain
                best_idx = int(np.argmax(scores))
                if scores[best_idx] > min_switch_gain:
                    model = models[best_idx]
                else:
                    model = global_model
                choices[rid] = model
                switches += 1 if model != global_model else 0
            eval_rows.append(
                {
                    "method": (
                        f"probe_complementarity_knn_tfidf_k{neighbors}"
                        f"_h{harm_weight_tag(harm_weight)}"
                    ),
                    "accuracy": accuracy_for_choices(eval_ids, choices, matrix),
                    "meta_neighbors": k,
                    "meta_smoothing": smoothing,
                    "meta_harm_weight": harm_weight,
                    "meta_min_switch_gain": min_switch_gain,
                    "meta_models": len(models),
                    "meta_positive_prior_models": positive_prior_models,
                    "meta_switch_rate": switches / len(eval_ids) if eval_ids else 0.0,
                    "meta_features": len(vectorizer.vocabulary_),
                    "meta_metadata_keys": ",".join(metadata_keys),
                    "meta_global_model": global_model,
                }
            )
    return eval_rows


def run_paired_lcb_complementarity_knn_tfidf_grid(
    rows: list[dict[str, Any]],
    probe_ids: list[str],
    eval_ids: list[str],
    matrix: dict[str, dict[str, bool]],
    global_model: str,
    metadata_keys: tuple[str, ...],
    neighbor_values: tuple[int, ...],
    harm_weights: tuple[float, ...],
    z_values: tuple[float, ...],
    smoothing: float = 5.0,
    min_switch_lcb: float = 0.0,
) -> list[dict[str, Any]]:
    if len(probe_ids) < 5 or len(eval_ids) < 10:
        return [
            {
                "method": "probe_paired_lcb_complementarity_knn_tfidf_grid",
                "accuracy": "",
                "meta_skipped": "not enough probe/eval rows",
            }
        ]
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover
        return [
            {
                "method": "probe_paired_lcb_complementarity_knn_tfidf_grid",
                "accuracy": "",
                "meta_skipped": f"sklearn/numpy unavailable: {exc}",
            }
        ]

    row_by_id = {row_id(row): row for row in rows}
    models = sorted(matrix)
    model_index = {model: idx for idx, model in enumerate(models)}
    global_idx = model_index.get(global_model, 0)
    train_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in probe_ids]
    eval_texts = [augmented_text_for_row(row_by_id[rid], metadata_keys) for rid in eval_ids]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
    sims_matrix = (x_eval @ x_train.T).toarray()
    max_neighbors = min(max(neighbor_values), len(probe_ids))
    top_indices: list[Any] = []
    top_weights: list[Any] = []
    for sims in sims_matrix:
        if max_neighbors < len(sims):
            idx = np.argpartition(sims, -max_neighbors)[-max_neighbors:]
        else:
            idx = np.arange(len(sims))
        order = np.argsort(sims[idx])[::-1]
        idx = idx[order]
        top_indices.append(idx)
        top_weights.append(sims[idx] + 1e-6)
    top_indices_arr = np.vstack(top_indices)
    top_weights_arr = np.vstack(top_weights)

    correctness = np.array(
        [[1.0 if matrix[model].get(rid, False) else 0.0 for rid in probe_ids] for model in models]
    )
    eval_correctness = np.array(
        [[1.0 if matrix[model].get(rid, False) else 0.0 for rid in eval_ids] for model in models]
    )
    eval_rows: list[dict[str, Any]] = []
    global_correct = correctness[global_idx]
    for harm_weight in harm_weights:
        fixes_global_error = (correctness == 1.0) & (global_correct == 0.0)
        breaks_global_success = (correctness == 0.0) & (global_correct == 1.0)
        deltas = fixes_global_error.astype(float) - harm_weight * breaks_global_success.astype(float)
        priors = deltas.mean(axis=1) if deltas.shape[1] else np.zeros(len(models))
        second_priors = (deltas * deltas).mean(axis=1) if deltas.shape[1] else np.zeros(len(models))
        positive_prior_models = int((priors > 0).sum())
        for neighbors in neighbor_values:
            if len(probe_ids) < max(5, neighbors):
                continue
            k = min(neighbors, len(probe_ids))
            local_idx = top_indices_arr[:, :k]
            local_weights = top_weights_arr[:, :k]
            sum_w = local_weights.sum(axis=1)
            denom = sum_w + smoothing
            neff = (denom * denom) / ((local_weights * local_weights).sum(axis=1) + smoothing)
            mean_matrix = np.empty((len(models), len(eval_ids)), dtype=float)
            variance_matrix = np.empty((len(models), len(eval_ids)), dtype=float)
            for model_idx in range(len(models)):
                local_deltas = deltas[model_idx, local_idx]
                weighted_sum = (local_deltas * local_weights).sum(axis=1)
                weighted_sq = ((local_deltas * local_deltas) * local_weights).sum(axis=1)
                mean = (weighted_sum + smoothing * priors[model_idx]) / denom
                second = (weighted_sq + smoothing * second_priors[model_idx]) / denom
                mean_matrix[model_idx] = mean
                variance_matrix[model_idx] = np.maximum(second - mean * mean, 0.0)
            for z_value in z_values:
                lcb_matrix = mean_matrix - z_value * np.sqrt(
                    variance_matrix / np.maximum(neff, 1.0)[None, :]
                )
                lcb_matrix[global_idx, :] = min_switch_lcb
                best_indices = np.argmax(lcb_matrix, axis=0)
                best_lcbs = lcb_matrix[best_indices, np.arange(len(eval_ids))]
                choice_indices = np.where(best_lcbs > min_switch_lcb, best_indices, global_idx)
                switches = int(np.sum(choice_indices != global_idx))
                acc = float(eval_correctness[choice_indices, np.arange(len(eval_ids))].mean())
                eval_rows.append(
                    {
                        "method": (
                            f"probe_paired_lcb_complementarity_knn_tfidf_k{neighbors}"
                            f"_h{harm_weight_tag(harm_weight)}_z{float_tag(z_value)}"
                        ),
                        "accuracy": acc,
                        "meta_neighbors": k,
                        "meta_smoothing": smoothing,
                        "meta_harm_weight": harm_weight,
                        "meta_lcb_z": z_value,
                        "meta_min_switch_lcb": min_switch_lcb,
                        "meta_models": len(models),
                        "meta_positive_prior_models": positive_prior_models,
                        "meta_switch_rate": switches / len(eval_ids) if eval_ids else 0.0,
                        "meta_mean_best_lcb": float(best_lcbs.mean()) if len(best_lcbs) else 0.0,
                        "meta_features": len(vectorizer.vocabulary_),
                        "meta_metadata_keys": ",".join(metadata_keys),
                        "meta_global_model": global_model,
                    }
                )
    return eval_rows


def evaluate_probe_budget(
    rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
    budget: int,
    split_key: str,
    group_keys: tuple[str, ...],
    seed: int,
    wilson_z: float,
    margin: float,
    run_multilabel: bool,
    only_complementarity: bool,
    only_paired_lcb: bool,
) -> list[dict[str, Any]]:
    probe_ids, eval_ids = deterministic_probe_split(rows, split_key, budget, seed)
    if not probe_ids or not eval_ids:
        return []
    full_ids = [row_id(row) for row in rows]
    best_full_model, _ = best_model_for_ids(matrix, full_ids)
    global_model, global_probe_acc, _ = best_model_with_wilson(matrix, probe_ids, wilson_z)
    current_choices = current_choice_map(rows, matrix, best_full_model)
    current_eval_acc = accuracy_for_choices(eval_ids, current_choices, matrix)
    rows_out: list[dict[str, Any]] = [
        {
            "method": "probe_current_router",
            "budget_per_group": budget,
            "split_key": split_key,
            "probe_samples": len(probe_ids),
            "eval_samples": len(eval_ids),
            "accuracy": current_eval_acc,
            "selected_model": "",
        },
        {
            "method": "probe_global_best",
            "budget_per_group": budget,
            "split_key": split_key,
            "probe_samples": len(probe_ids),
            "eval_samples": len(eval_ids),
            "accuracy": accuracy_for_model(matrix[global_model], eval_ids),
            "selected_model": global_model,
            "probe_accuracy": global_probe_acc,
        },
    ]

    def append_complementarity_rows() -> None:
        for row in run_complementarity_knn_tfidf_grid(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            global_model,
            group_keys,
            neighbor_values=(3, 5, 10, 20, 50, 100),
            harm_weights=(1.0, 1.25, 1.5),
        ):
            rows_out.append(
                {
                    "budget_per_group": budget,
                    "split_key": split_key,
                    "probe_samples": len(probe_ids),
                    "eval_samples": len(eval_ids),
                    "selected_model": "",
                    **row,
                }
            )

    def append_paired_lcb_rows() -> None:
        for row in run_paired_lcb_complementarity_knn_tfidf_grid(
            rows,
            probe_ids,
            eval_ids,
            matrix,
            global_model,
            group_keys,
            neighbor_values=(3, 5, 10, 20, 50, 100),
            harm_weights=(1.0, 1.25, 1.5),
            z_values=(0.25, 0.5),
        ):
            rows_out.append(
                {
                    "budget_per_group": budget,
                    "split_key": split_key,
                    "probe_samples": len(probe_ids),
                    "eval_samples": len(eval_ids),
                    "selected_model": "",
                    **row,
                }
            )

    if only_paired_lcb:
        if run_multilabel:
            append_paired_lcb_rows()
        return rows_out

    if only_complementarity:
        if run_multilabel:
            append_complementarity_rows()
        return rows_out

    fallback_choices, fallback_meta = parse_or_confidence_fallback_choices(
        rows, probe_ids, eval_ids, matrix, global_model, best_full_model
    )
    rows_out.append(
        {
            "method": "probe_risk_parse_conf_fallback",
            "budget_per_group": budget,
            "split_key": split_key,
            "probe_samples": len(probe_ids),
            "eval_samples": len(eval_ids),
            "accuracy": accuracy_for_choices(eval_ids, fallback_choices, matrix),
            "selected_model": global_model,
            **{f"meta_{k}": v for k, v in fallback_meta.items()},
        }
    )

    for key in group_keys:
        mapping = group_mapping_from_probe(rows, probe_ids, matrix, key, global_model, wilson_z)
        choices = choices_from_group_mapping(rows, eval_ids, key, mapping)
        rows_out.append(
            {
                "method": f"probe_{key}_mapping",
                "budget_per_group": budget,
                "split_key": split_key,
                "probe_samples": len(probe_ids),
                "eval_samples": len(eval_ids),
                "accuracy": accuracy_for_choices(eval_ids, choices, matrix),
                "selected_model": "",
                "num_groups": len(set(group_value(row, key) for row in rows)),
            }
        )

        rc_mapping = risk_controlled_group_mapping(
            rows, probe_ids, matrix, key, global_model, wilson_z, margin
        )
        rc_choices = choices_from_group_mapping(rows, eval_ids, key, rc_mapping)
        rows_out.append(
            {
                "method": f"probe_risk_controlled_{key}_mapping",
                "budget_per_group": budget,
                "split_key": split_key,
                "probe_samples": len(probe_ids),
                "eval_samples": len(eval_ids),
                "accuracy": accuracy_for_choices(eval_ids, rc_choices, matrix),
                "selected_model": "",
                "num_groups": len(set(group_value(row, key) for row in rows)),
            }
        )

    if run_multilabel:
        multilabel_acc, meta = run_multilabel_tfidf(rows, probe_ids, eval_ids, matrix, global_model)
        rows_out.append(
            {
                "method": "probe_multilabel_tfidf",
                "budget_per_group": budget,
                "split_key": split_key,
                "probe_samples": len(probe_ids),
                "eval_samples": len(eval_ids),
                "accuracy": multilabel_acc if multilabel_acc is not None else "",
                "selected_model": "",
                **{f"meta_{k}": v for k, v in meta.items()},
            }
        )
        multilabel_meta_acc, meta = run_multilabel_tfidf_metadata(
            rows, probe_ids, eval_ids, matrix, global_model, group_keys
        )
        rows_out.append(
            {
                "method": "probe_multilabel_tfidf_metadata",
                "budget_per_group": budget,
                "split_key": split_key,
                "probe_samples": len(probe_ids),
                "eval_samples": len(eval_ids),
                "accuracy": multilabel_meta_acc if multilabel_meta_acc is not None else "",
                "selected_model": "",
                **{f"meta_{k}": v for k, v in meta.items()},
            }
        )
        for neighbors in (3, 5, 10, 20, 50, 100):
            if len(probe_ids) < max(5, neighbors):
                continue
            knn_acc, meta = run_knn_tfidf(
                rows,
                probe_ids,
                eval_ids,
                matrix,
                global_model,
                group_keys,
                neighbors=neighbors,
            )
            rows_out.append(
                {
                    "method": f"probe_knn_tfidf_k{neighbors}",
                    "budget_per_group": budget,
                    "split_key": split_key,
                    "probe_samples": len(probe_ids),
                    "eval_samples": len(eval_ids),
                    "accuracy": knn_acc if knn_acc is not None else "",
                    "selected_model": "",
                    **{f"meta_{k}": v for k, v in meta.items()},
                }
            )
        append_complementarity_rows()
        append_paired_lcb_rows()
    return rows_out


def resolve_probe_split_keys(value: str, case: CaseSpec) -> list[str]:
    if value == "primary":
        return [case.primary_group_key]
    if value == "all":
        keys = [case.primary_group_key]
        keys.extend(key for key in case.group_keys if key not in keys)
        return keys
    return [item.strip() for item in value.split(",") if item.strip()]


def route_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    parse_present = any("route_parse_ok" in row for row in rows)
    confidence_present = any("route_confidence" in row for row in rows)
    return {
        "route_parse_ok_rate": (
            sum(1 for row in rows if row.get("route_parse_ok") is True) / n if n and parse_present else None
        ),
        "route_parse_fail_count": (
            sum(1 for row in rows if row.get("route_parse_ok") is False) if parse_present else None
        ),
        "route_confidence_mean": (
            sum(float(row.get("route_confidence", 0.0)) for row in rows if row.get("route_confidence") is not None)
            / max(1, sum(1 for row in rows if row.get("route_confidence") is not None))
            if confidence_present
            else None
        ),
        "routed_model_counts": dict(Counter(str(row.get("routed_model")) for row in rows).most_common()),
        "routed_subject_counts": dict(Counter(str(row.get("routed_subject")) for row in rows).most_common()),
    }


def run_case(
    case: CaseSpec,
    output_dir: Path,
    budgets: list[int],
    seed: int,
    wilson_z: float,
    margin: float,
    run_multilabel: bool,
    probe_split_keys: list[str],
    only_complementarity: bool,
    only_paired_lcb: bool,
) -> dict[str, Any]:
    rows = load_route_rows(case)
    ids = [row_id(row) for row in rows]
    raw_matrix = load_single_matrix(case)
    matrix, missing_counts = keep_complete_models(raw_matrix, ids)
    if not matrix:
        raise RuntimeError(f"No complete single-model predictions found for {case.case_id}")

    best_model, best_acc = best_model_for_ids(matrix, ids)
    current_choices = current_choice_map(rows, matrix, best_model)
    reported_current_acc = sum(1 for row in rows if route_row_correct(row)) / len(rows) if rows else 0.0
    current_acc = accuracy_for_choices(ids, current_choices, matrix)
    full_rows: list[dict[str, Any]] = [
        {
            "method": "reported_current_router",
            "accuracy": reported_current_acc,
            "samples": len(ids),
            "selected_model": "",
            "scope": "reported_bench_coe_file",
        },
        {
            "method": "current_router_recomposed",
            "accuracy": current_acc,
            "samples": len(ids),
            "selected_model": "",
            "scope": "single_cache_recomposition",
        },
        {
            "method": "best_single_full",
            "accuracy": best_acc,
            "samples": len(ids),
            "selected_model": best_model,
            "scope": "full_oracle_diagnostic",
        },
        {
            "method": "instance_oracle_full",
            "accuracy": instance_oracle_accuracy(matrix, ids),
            "samples": len(ids),
            "selected_model": "",
            "scope": "full_oracle_diagnostic",
        },
    ]
    full_group_oracles: dict[str, Any] = {}
    for key in case.group_keys:
        oracle = group_oracle(rows, ids, matrix, key)
        full_group_oracles[key] = oracle
        full_rows.append(
            {
                "method": f"oracle_{key}_mapping_full",
                "accuracy": oracle["accuracy"],
                "samples": len(ids),
                "selected_model": "",
                "scope": "full_oracle_diagnostic",
                "num_groups": oracle["num_groups"],
            }
        )

    probe_rows: list[dict[str, Any]] = []
    case_seed = seed + (zlib.crc32(case.case_id.encode("utf-8")) % 100000)
    for split_key in probe_split_keys:
        for budget in budgets:
            probe_rows.extend(
                evaluate_probe_budget(
                    rows=rows,
                    matrix=matrix,
                    budget=budget,
                    split_key=split_key,
                    group_keys=case.group_keys,
                    seed=case_seed,
                    wilson_z=wilson_z,
                    margin=margin,
                    run_multilabel=run_multilabel,
                    only_complementarity=only_complementarity,
                    only_paired_lcb=only_paired_lcb,
                )
            )

    case_out = output_dir / case.case_id
    case_payload = dict(case.__dict__)
    case_payload["route_dir"] = str(case.route_dir)
    case_payload["single_path"] = str(case.single_path)
    write_json(
        case_out / "summary.json",
        {
            "case": case_payload,
            "num_rows": len(rows),
            "num_models_raw": len(raw_matrix),
            "num_models_used": len(matrix),
            "models_used": sorted(matrix),
            "missing_counts": missing_counts,
            "route_diagnostics": route_diagnostics(rows),
            "full_results": full_rows,
            "probe_results": probe_rows,
            "full_group_oracles": full_group_oracles,
        },
    )
    write_csv(case_out / "full_results.csv", full_rows)
    write_csv(case_out / "probe_results.csv", probe_rows)
    write_case_markdown(case_out / "README.md", case, full_rows, probe_rows, rows, matrix)

    best_probe = max(
        (row for row in probe_rows if isinstance(row.get("accuracy"), float)),
        key=lambda row: float(row["accuracy"]),
        default={},
    )
    instance_row = next(row for row in full_rows if row["method"] == "instance_oracle_full")
    return {
        "case_id": case.case_id,
        "benchmark": case.benchmark,
        "samples": len(rows),
        "models_used": len(matrix),
        "reported_current_router": reported_current_acc,
        "current_router_recomposed": current_acc,
        "best_single_full": best_acc,
        "best_single_model": best_model,
        "instance_oracle_full": instance_row["accuracy"],
        "best_group_oracle_full": max(
            row["accuracy"] for row in full_rows if row["method"].startswith("oracle_")
        ),
        "best_group_oracle_method": max(
            (row for row in full_rows if row["method"].startswith("oracle_")),
            key=lambda row: float(row["accuracy"]),
        )["method"],
        "best_probe_method": best_probe.get("method", ""),
        "best_probe_accuracy": best_probe.get("accuracy", ""),
        "best_probe_budget": best_probe.get("budget_per_group", ""),
        "best_probe_split_key": best_probe.get("split_key", ""),
        "route_parse_ok_rate": route_diagnostics(rows).get("route_parse_ok_rate"),
    }


def fmt_pct(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def write_case_markdown(
    path: Path,
    case: CaseSpec,
    full_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
) -> None:
    lines = [
        f"# {case.case_id}",
        "",
        f"- Benchmark: `{case.benchmark}`",
        f"- Samples: {len(rows)}",
        f"- Models used: {len(matrix)}",
        "",
        "## Full Diagnostic",
        "",
        "| Method | Accuracy | Scope | Selected model |",
        "| --- | ---: | --- | --- |",
    ]
    for row in sorted(full_rows, key=lambda item: -float(item["accuracy"])):
        lines.append(
            f"| `{row['method']}` | {fmt_pct(row['accuracy'])} | {row.get('scope', '')} | {row.get('selected_model', '')} |"
        )
    lines.extend(
        [
            "",
            "## Best Probe-Heldout Results",
            "",
            "| Budget | Method | Accuracy | Eval samples | Selected model |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        if isinstance(row.get("accuracy"), float):
            by_budget[int(row["budget_per_group"])].append(row)
    for budget, budget_rows in sorted(by_budget.items()):
        for row in sorted(budget_rows, key=lambda item: -float(item["accuracy"]))[:6]:
            lines.append(
                f"| {budget} | `{row['method']}` | {fmt_pct(row['accuracy'])} | {row.get('eval_samples', '')} | {row.get('selected_model', '')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Offline Bench-CoE Innovation Experiments",
        "",
        "| Case | Reported current | Recomputed current | Best single | Best group oracle | Instance oracle | Best probe-heldout | Best probe method |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_id']}`",
                    fmt_pct(row["reported_current_router"]),
                    fmt_pct(row["current_router_recomposed"]),
                    f"{fmt_pct(row['best_single_full'])} ({row['best_single_model']})",
                    f"{fmt_pct(row['best_group_oracle_full'])} ({row['best_group_oracle_method']})",
                    fmt_pct(row["instance_oracle_full"]),
                    fmt_pct(row["best_probe_accuracy"]),
                    f"`{row['best_probe_method']}`",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_cases(value: str) -> list[CaseSpec]:
    if value == "all":
        return list(CASES)
    wanted = [item.strip() for item in value.split(",") if item.strip()]
    by_id = {case.case_id: case for case in CASES}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise ValueError(f"Unknown case ids: {missing}; valid ids: {sorted(by_id)}")
    return [by_id[item] for item in wanted]


def main() -> None:
    args = parse_args()
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_cases(args.cases)
    summary_rows: list[dict[str, Any]] = []
    for case in selected:
        probe_split_keys = resolve_probe_split_keys(args.probe_split_keys, case)
        print(f"[case] {case.case_id}")
        summary_rows.append(
            run_case(
                case=case,
                output_dir=args.output_dir,
                budgets=budgets,
                seed=args.seed,
                wilson_z=args.wilson_z,
                margin=args.risk_margin,
                run_multilabel=not args.skip_multilabel,
                probe_split_keys=probe_split_keys,
                only_complementarity=args.only_complementarity,
                only_paired_lcb=args.only_paired_lcb,
            )
        )
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_json(args.output_dir / "summary.json", summary_rows)
    write_summary_markdown(args.output_dir / "summary.md", summary_rows)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
