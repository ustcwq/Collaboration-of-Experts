from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.materialize_innovation_strategies import (
    fmt_pct,
    load_full_mmlu_predictions,
    summarize_boolean_choices,
    table_row,
    write_text,
)
from bench_coe.gaokao_utils import (
    TASK_TO_SUBJECT as GAOKAO_TASK_TO_SUBJECT,
    load_gaokao2010_2022_full_predictions,
    score_example_slots,
)
from bench_coe.offline_router_innovation_experiments import (
    accuracy_for_choices,
    best_model_for_ids,
    group_value,
    instance_oracle_accuracy,
    is_correct_row,
    row_id,
    text_for_row,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    source_kind: str
    source_root: Path
    target_kind: str
    target_root: Path
    group_key: str
    title: str


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "cmmmu_dev_to_val",
        "json",
        env_path("BENCH_COE_CMMMU_DEV_ROOT", "outputs/multimodal_babyvision_models/cmmmu/dev"),
        "json",
        env_path("BENCH_COE_CMMMU_VAL_ROOT", "outputs/multimodal_babyvision_models/cmmmu/val"),
        "category",
        "CMMMU dev -> CMMMU val",
    ),
    CaseSpec(
        "mmmu_pro_to_cmmmu",
        "json",
        env_path("BENCH_COE_MMMU_PRO_TEST_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/test"),
        "json",
        env_path("BENCH_COE_CMMMU_VAL_ROOT", "outputs/multimodal_babyvision_models/cmmmu/val"),
        "category",
        "MMMU-Pro standard 10-options test -> CMMMU val",
    ),
    CaseSpec(
        "mmmu_pro_to_mathvista",
        "json",
        env_path("BENCH_COE_MMMU_PRO_TEST_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/test"),
        "json",
        env_path("BENCH_COE_MATHVISTA_ROOT", "outputs/multimodal_babyvision_models/mathvista/testmini"),
        "task",
        "MMMU-Pro standard 10-options test -> MathVista testmini",
    ),
    CaseSpec(
        "mmmu_pro_to_gaokao_mm",
        "json",
        env_path("BENCH_COE_MMMU_PRO_TEST_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/test"),
        "gaokao_mm",
        env_path("BENCH_COE_GAOKAO_MM_ROOT", "outputs/model_benchmarks/autonomous_remaining_full_20260802/vision/gaokao_mm"),
        "subject",
        "MMMU-Pro standard 10-options test -> GAOKAO-MM",
    ),
    CaseSpec(
        "gaokao_mm_to_cmmmu",
        "gaokao_mm",
        env_path("BENCH_COE_GAOKAO_MM_ROOT", "outputs/model_benchmarks/autonomous_remaining_full_20260802/vision/gaokao_mm"),
        "json",
        env_path("BENCH_COE_CMMMU_VAL_ROOT", "outputs/multimodal_babyvision_models/cmmmu/val"),
        "category",
        "GAOKAO-MM -> CMMMU val",
    ),
    CaseSpec(
        "gaokao_mm_to_mathvista",
        "gaokao_mm",
        env_path("BENCH_COE_GAOKAO_MM_ROOT", "outputs/model_benchmarks/autonomous_remaining_full_20260802/vision/gaokao_mm"),
        "json",
        env_path("BENCH_COE_MATHVISTA_ROOT", "outputs/multimodal_babyvision_models/mathvista/testmini"),
        "task",
        "GAOKAO-MM -> MathVista testmini",
    ),
    CaseSpec(
        "mmmu_pro_val_to_cmmmu",
        "json",
        env_path("BENCH_COE_MMMU_PRO_VALIDATION_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id"),
        "json",
        env_path("BENCH_COE_CMMMU_VAL_ROOT", "outputs/multimodal_babyvision_models/cmmmu/val"),
        "category",
        "MMMU-Pro validation-id subset -> CMMMU val",
    ),
    CaseSpec(
        "mmmu_pro_val_to_mathvista",
        "json",
        env_path("BENCH_COE_MMMU_PRO_VALIDATION_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id"),
        "json",
        env_path("BENCH_COE_MATHVISTA_ROOT", "outputs/multimodal_babyvision_models/mathvista/testmini"),
        "task",
        "MMMU-Pro validation-id subset -> MathVista testmini",
    ),
    CaseSpec(
        "mmmu_pro_val_to_mmmu_pro_test",
        "json",
        env_path("BENCH_COE_MMMU_PRO_VALIDATION_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id"),
        "json",
        env_path("BENCH_COE_MMMU_PRO_TEST_ID_ROOT", "outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/test_id"),
        "domain",
        "MMMU-Pro validation-id subset -> MMMU-Pro test-id subset",
    ),
    CaseSpec(
        "mmlu_val_to_mmlu_test",
        "mmlu_validation",
        env_path("BENCH_COE_MMLU_VALIDATION_ROOT", "outputs/bench_coe/mmlu_pro_validation_single_models"),
        "mmlu_test",
        env_path("BENCH_COE_MMLU_TEST_ROOT", "MMLU-Pro/results"),
        "category",
        "MMLU-Pro validation -> MMLU-Pro test",
    ),
    CaseSpec(
        "mmlu_val_to_gaokao2010_2022",
        "mmlu_validation",
        env_path("BENCH_COE_MMLU_VALIDATION_ROOT", "outputs/bench_coe/mmlu_pro_validation_single_models"),
        "gaokao2010_2022",
        env_path("BENCH_COE_GAOKAO_2010_2022_ROOT", "GAOKAO-Bench-2010-2022/Data"),
        "subject",
        "MMLU-Pro validation -> GAOKAO-Bench-2010-2022 objective questions",
    ),
    CaseSpec(
        "gaokao2010_2022_to_mmlu_test",
        "gaokao2010_2022",
        env_path("BENCH_COE_GAOKAO_2010_2022_ROOT", "GAOKAO-Bench-2010-2022/Data"),
        "mmlu_test",
        env_path("BENCH_COE_MMLU_TEST_ROOT", "MMLU-Pro/results"),
        "category",
        "GAOKAO-Bench-2010-2022 objective questions -> MMLU-Pro test",
    ),
    CaseSpec(
        "gaokao2010_2022_to_bbh",
        "gaokao2010_2022",
        env_path("BENCH_COE_GAOKAO_2010_2022_ROOT", "GAOKAO-Bench-2010-2022/Data"),
        "jsonl",
        env_path("BENCH_COE_BBH_ROOT", "outputs/model_benchmarks/official_code_local_models/bbh"),
        "task",
        "GAOKAO-Bench-2010-2022 objective questions -> BBH",
    ),
    CaseSpec(
        "gaokao2010_2022_to_gpqa",
        "gaokao2010_2022",
        env_path("BENCH_COE_GAOKAO_2010_2022_ROOT", "GAOKAO-Bench-2010-2022/Data"),
        "jsonl",
        env_path("BENCH_COE_GPQA_ROOT", "outputs/model_benchmarks/official_code_local_models/gpqa"),
        "domain",
        "GAOKAO-Bench-2010-2022 objective questions -> GPQA diamond",
    ),
    CaseSpec(
        "gaokao2010_2022_to_mmstar",
        "gaokao2010_2022",
        env_path("BENCH_COE_GAOKAO_2010_2022_ROOT", "GAOKAO-Bench-2010-2022/Data"),
        "jsonl",
        env_path("BENCH_COE_MMSTAR_ROOT", "outputs/model_benchmarks/official_code_local_models/mmstar_text_only"),
        "category",
        "GAOKAO-Bench-2010-2022 objective questions -> MMStar text-only",
    ),
    CaseSpec(
        "mmlu_val_to_bbh",
        "mmlu_validation",
        env_path("BENCH_COE_MMLU_VALIDATION_ROOT", "outputs/bench_coe/mmlu_pro_validation_single_models"),
        "jsonl",
        env_path("BENCH_COE_BBH_ROOT", "outputs/model_benchmarks/official_code_local_models/bbh"),
        "task",
        "MMLU-Pro validation -> BBH",
    ),
    CaseSpec(
        "mmlu_val_to_gpqa",
        "mmlu_validation",
        env_path("BENCH_COE_MMLU_VALIDATION_ROOT", "outputs/bench_coe/mmlu_pro_validation_single_models"),
        "jsonl",
        env_path("BENCH_COE_GPQA_ROOT", "outputs/model_benchmarks/official_code_local_models/gpqa"),
        "domain",
        "MMLU-Pro validation -> GPQA diamond",
    ),
    CaseSpec(
        "mmlu_val_to_mmstar",
        "mmlu_validation",
        env_path("BENCH_COE_MMLU_VALIDATION_ROOT", "outputs/bench_coe/mmlu_pro_validation_single_models"),
        "jsonl",
        env_path("BENCH_COE_MMSTAR_ROOT", "outputs/model_benchmarks/official_code_local_models/mmstar_text_only"),
        "category",
        "MMLU-Pro validation -> MMStar text-only",
    ),
    CaseSpec(
        "portfolio_to_bbh",
        "portfolio_text_except_bbh",
        Path("."),
        "jsonl",
        Path("outputs/model_benchmarks/official_code_local_models/bbh"),
        "task",
        "Text capability portfolio excluding BBH -> BBH",
    ),
    CaseSpec(
        "portfolio_to_gpqa",
        "portfolio_text_except_gpqa",
        Path("."),
        "jsonl",
        Path("outputs/model_benchmarks/official_code_local_models/gpqa"),
        "domain",
        "Text capability portfolio excluding GPQA -> GPQA diamond",
    ),
    CaseSpec(
        "portfolio_to_mmstar",
        "portfolio_text_except_mmstar",
        Path("."),
        "jsonl",
        Path("outputs/model_benchmarks/official_code_local_models/mmstar_text_only"),
        "category",
        "Text capability portfolio excluding MMStar -> MMStar text-only",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean implementation of improve2.md capability-aware routing ideas. "
            "Only source labels are used for calibration; target labels are used once for final scoring."
        )
    )
    parser.add_argument("--cases", default="all", help="Comma-separated case ids, or all.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/improve2_capability_routing"))
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--expert-clusters", type=int, default=3)
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list.")
    return payload


def load_json_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
            except Exception:
                summary = {}
            if summary.get("status") not in (None, "completed"):
                continue
        pred_path = model_dir / "predictions.json"
        if not pred_path.exists():
            pred_path = model_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue
        matrix[model_dir.name] = {row_id(row): row for row in load_rows(pred_path)}
    return matrix


def load_jsonl_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
            except Exception:
                summary = {}
            if summary.get("status") not in (None, "completed"):
                continue
        pred_path = model_dir / "predictions.jsonl"
        if pred_path.exists():
            matrix[model_dir.name] = {row_id(row): row for row in read_jsonl(pred_path)}
    return matrix


def load_mmlu_validation_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        result_dir = model_dir / "CoT" / "validation"
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


def load_gaokao_jsonl_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    full: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        pred_path = model_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue
        rows_by_id: dict[str, dict[str, Any]] = {}
        for example in read_jsonl(pred_path):
            keyword = str(example.get("keyword", example.get("task", "")))
            index = example.get("index")
            for scored in score_example_slots(keyword, example):
                rid = f"{keyword}:{index}:{scored['answer_idx']}"
                rows_by_id[rid] = {
                    **example,
                    "id": rid,
                    "benchmark": "gaokao_bench_2010_2022",
                    "subject": GAOKAO_TASK_TO_SUBJECT.get(keyword, keyword),
                    "task": keyword,
                    "answer": scored["expected"],
                    "target": scored["expected"],
                    "pred": scored["predicted"],
                    "prediction": scored["predicted"],
                    "response": example.get("model_output", ""),
                    "model_outputs": example.get("model_output", ""),
                    "score": scored["score"],
                    "is_correct": scored["is_correct"],
                    "has_partial_credit": scored["has_partial_credit"],
                }
        if rows_by_id:
            full[model_dir.name] = rows_by_id
    return full


def normalize_gaokao_mm_answer(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip().upper() for item in values if str(item).strip()]


def load_gaokao_mm_predictions(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    full: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        rows_by_id: dict[str, dict[str, Any]] = {}
        for path in sorted(model_dir.glob("*_2010-2023_*.json")):
            payload = read_json(path)
            keyword = str(payload.get("keyword", ""))
            subject = keyword.removeprefix("2010-2023_").removesuffix("_MCQs").replace("_", " ")
            for item in payload.get("example", []):
                rid = f"{keyword}:{item.get('index')}"
                answer = normalize_gaokao_mm_answer(item.get("standard_answer", []))
                prediction = normalize_gaokao_mm_answer(item.get("model_answer", []))
                rows_by_id[rid] = {
                    "id": rid,
                    "benchmark": "gaokao_mm",
                    "subject": subject,
                    "task": keyword,
                    "question_type": payload.get("question_type"),
                    "question": item.get("question", ""),
                    "answer": answer,
                    "target": answer,
                    "pred": prediction,
                    "prediction": prediction,
                    "response": item.get("model_output", ""),
                    "model_outputs": item.get("model_output", ""),
                    "score": float(bool(answer) and prediction == answer),
                    "is_correct": bool(answer) and prediction == answer,
                    "year": item.get("year"),
                    "category": item.get("category"),
                    "combined_image": item.get("combined_image"),
                }
        if rows_by_id:
            full[model_dir.name] = rows_by_id
    return full


def load_full_predictions(kind: str, root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if kind == "json":
        return load_json_predictions(root)
    if kind == "jsonl":
        return load_jsonl_predictions(root)
    if kind == "mmlu_validation":
        return load_mmlu_validation_predictions(root)
    if kind == "mmlu_test":
        return load_full_mmlu_predictions(root)
    if kind == "gaokao2010_2022":
        if root.is_dir() and any((path / "predictions.jsonl").exists() for path in root.iterdir() if path.is_dir()):
            return load_gaokao_jsonl_predictions(root)
        return load_gaokao2010_2022_full_predictions(root)
    if kind == "gaokao_mm":
        return load_gaokao_mm_predictions(root)
    if kind.startswith("portfolio_text_except_"):
        return load_text_portfolio_predictions(kind.removeprefix("portfolio_text_except_"))
    raise ValueError(kind)


def prefixed_full_matrix(
    dataset: str,
    full: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model, rows_by_id in full.items():
        out[model] = {}
        for rid, row in rows_by_id.items():
            item = dict(row)
            item["id"] = f"{dataset}:{rid}"
            item["source_dataset"] = dataset
            out[model][item["id"]] = item
    return out


def merge_full_matrices(parts: list[tuple[str, dict[str, dict[str, dict[str, Any]]]]]) -> dict[str, dict[str, dict[str, Any]]]:
    common_models = sorted(set.intersection(*(set(full) for _, full in parts if full)))
    merged: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in common_models}
    for dataset, full in parts:
        prefixed = prefixed_full_matrix(dataset, full)
        for model in common_models:
            merged[model].update(prefixed.get(model, {}))
    return merged


def load_text_portfolio_predictions(excluded: str) -> dict[str, dict[str, dict[str, Any]]]:
    roots = {
        "mmlu_val": ("mmlu_validation", Path("outputs/bench_coe/mmlu_pro_validation_single_models")),
        "bbh": ("jsonl", Path("outputs/model_benchmarks/official_code_local_models/bbh")),
        "gpqa": ("jsonl", Path("outputs/model_benchmarks/official_code_local_models/gpqa")),
        "mmstar": ("jsonl", Path("outputs/model_benchmarks/official_code_local_models/mmstar_text_only")),
    }
    parts: list[tuple[str, dict[str, dict[str, dict[str, Any]]]]] = []
    for dataset, (kind, root) in roots.items():
        if dataset == excluded:
            continue
        parts.append((dataset, load_full_predictions(kind, root)))
    return merge_full_matrices(parts)


def bool_matrix(full: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, bool]]:
    return {
        model: {rid: is_correct_row(row) for rid, row in rows.items()}
        for model, rows in full.items()
    }


def complete_models(matrix: dict[str, dict[str, bool]], ids: list[str]) -> dict[str, dict[str, bool]]:
    return {model: values for model, values in matrix.items() if all(rid in values for rid in ids)}


def infer_ids(full: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    for rows_by_id in full.values():
        return sorted(rows_by_id)
    return []


def first_complete_rows(full: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> list[dict[str, Any]]:
    for rows_by_id in full.values():
        if all(rid in rows_by_id for rid in ids):
            return [rows_by_id[rid] for rid in ids]
    raise RuntimeError("No complete row source found.")


def augmented_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "category",
        "src",
        "task",
        "domain",
        "subdomain",
        "subcategory",
        "type",
        "difficulty",
        "l2_category",
        "bench",
        "question_type",
        "answer_type",
    ):
        value = row.get(key)
        if value is None or value == "":
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            token = str(item).replace(" ", "_").replace("\n", "_")
            parts.append(f"meta_{key}={token}")
    parts.append(text_for_row(row))
    return "\n".join(parts)


def matrix_array(matrix: dict[str, dict[str, bool]], models: list[str], ids: list[str]) -> Any:
    import numpy as np

    return np.asarray([[1.0 if matrix[model].get(rid, False) else 0.0 for model in models] for rid in ids], dtype=float)


def build_vectorizer(source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]] | None = None) -> tuple[Any, Any, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    source_texts = [augmented_text(row) for row in source_rows]
    target_texts = [augmented_text(row) for row in target_rows or []]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(source_texts + target_texts)
    return vectorizer, vectorizer.transform(source_texts), vectorizer.transform(target_texts) if target_rows else None


def nearest_source_indices(x_source: Any, x_target: Any, k: int) -> tuple[Any, Any]:
    import numpy as np

    sims = x_target @ x_source.T
    sims = sims.toarray()
    kk = min(k, sims.shape[1])
    if kk <= 0:
        return np.empty((sims.shape[0], 0), dtype=int), np.empty((sims.shape[0], 0), dtype=float)
    if kk < sims.shape[1]:
        idx = np.argpartition(sims, -kk, axis=1)[:, -kk:]
    else:
        idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
    row_idx = np.arange(sims.shape[0])[:, None]
    weights = sims[row_idx, idx]
    order = np.argsort(-weights, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    weights = np.take_along_axis(weights, order, axis=1)
    weights = np.maximum(weights, 0.0)
    denom = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, denom, out=np.full_like(weights, 1.0 / kk), where=denom > 1e-12)
    return idx, weights


def choices_from_scores(ids: list[str], models: list[str], scores: Any) -> dict[str, str]:
    import numpy as np

    best = np.argmax(scores, axis=1)
    return {rid: models[int(best[idx])] for idx, rid in enumerate(ids)}


def cape_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    models: list[str],
    components: int,
    seed: int,
    transductive: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.linear_model import Ridge

    target_ids = [row_id(row) for row in target_rows]
    k = min(max(2, components), max(2, min(source_y.shape) - 1))
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=1000)
    source_cap = nmf.fit_transform(source_y + 1e-4)
    expert_cap = nmf.components_.T
    _, x_source, x_target = build_vectorizer(source_rows, target_rows if transductive else None)
    if x_target is None:
        vectorizer, x_source, _ = build_vectorizer(source_rows, None)
        x_target = vectorizer.transform([augmented_text(row) for row in target_rows])
    reg = Ridge(alpha=1.0)
    reg.fit(x_source, source_cap)
    pred_cap = np.maximum(reg.predict(x_target), 0.0)
    scores = pred_cap @ expert_cap.T
    return choices_from_scores(target_ids, models, scores), {
        "components": k,
        "transductive_inputs": transductive,
        "explained_source_density": float(source_y.mean()),
    }


def relative_advantage_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    models: list[str],
    transductive: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.linear_model import Ridge

    target_ids = [row_id(row) for row in target_rows]
    row_median = np.median(source_y, axis=1, keepdims=True)
    advantage = source_y - row_median
    _, x_source, x_target = build_vectorizer(source_rows, target_rows if transductive else None)
    if x_target is None:
        vectorizer, x_source, _ = build_vectorizer(source_rows, None)
        x_target = vectorizer.transform([augmented_text(row) for row in target_rows])
    reg = Ridge(alpha=1.0)
    reg.fit(x_source, advantage)
    scores = reg.predict(x_target)
    return choices_from_scores(target_ids, models, scores), {"transductive_inputs": transductive}


def source_group_transferability(
    source_rows: list[dict[str, Any]],
    source_ids: list[str],
    source_matrix: dict[str, dict[str, bool]],
    models: list[str],
) -> dict[str, float]:
    by_group: dict[str, list[str]] = defaultdict(list)
    row_by_id = {row_id(row): row for row in source_rows}
    for rid in source_ids:
        group = group_value(row_by_id[rid], "category")
        by_group[group].append(rid)
    transfer: dict[str, float] = {}
    for model in models:
        accs = []
        for ids in by_group.values():
            if ids:
                accs.append(sum(1 for rid in ids if source_matrix[model].get(rid, False)) / len(ids))
        if not accs:
            transfer[model] = 0.0
            continue
        mean = sum(accs) / len(accs)
        var = sum((acc - mean) ** 2 for acc in accs) / len(accs)
        transfer[model] = mean - math.sqrt(var)
    return transfer


def self_fingerprint_knn_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    source_matrix: dict[str, dict[str, bool]],
    models: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    target_ids = [row_id(row) for row in target_rows]
    source_ids = [row_id(row) for row in source_rows]
    _, x_source, x_target = build_vectorizer(source_rows, target_rows)
    idx, weights = nearest_source_indices(x_source, x_target, k)
    local_success = np.einsum("ij,ijk->ik", weights, source_y[idx])
    transfer = source_group_transferability(source_rows, source_ids, source_matrix, models)
    transfer_vec = np.asarray([transfer[model] for model in models], dtype=float)
    source_global = source_y.mean(axis=0)
    fragility = 1.0 - source_global
    scores = 0.82 * local_success + 0.13 * transfer_vec[None, :] - 0.05 * fragility[None, :]
    return choices_from_scores(target_ids, models, scores), {
        "k": min(k, len(source_rows)),
        "fingerprint": "local_success+cross_group_transfer-fragility",
    }


def negative_memory_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    models: list[str],
    k: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    target_ids = [row_id(row) for row in target_rows]
    _, x_source, x_target = build_vectorizer(source_rows, target_rows)
    idx, weights = nearest_source_indices(x_source, x_target, k)
    local_failure = np.einsum("ij,ijk->ik", weights, 1.0 - source_y[idx])
    global_success = source_y.mean(axis=0)
    scores = global_success[None, :] - 0.75 * local_failure
    return choices_from_scores(target_ids, models, scores), {"k": min(k, len(source_rows))}


def disagreement_topology_choices(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_y: Any,
    models: list[str],
    k: int,
    clusters: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    target_ids = [row_id(row) for row in target_rows]
    model_count = len(models)
    n_clusters = min(max(2, clusters), model_count)
    agreement = np.zeros((model_count, model_count), dtype=float)
    for i in range(model_count):
        for j in range(model_count):
            agreement[i, j] = float((source_y[:, i] == source_y[:, j]).mean())
    distance = 1.0 - agreement
    try:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    labels = clusterer.fit_predict(distance)
    _, x_source, x_target = build_vectorizer(source_rows, target_rows)
    idx, weights = nearest_source_indices(x_source, x_target, k)
    local_success = np.einsum("ij,ijk->ik", weights, source_y[idx])
    global_success = source_y.mean(axis=0)
    choices: dict[str, str] = {}
    for row_idx, rid in enumerate(target_ids):
        cluster_scores = []
        for cluster_id in range(n_clusters):
            member_idx = [midx for midx, label in enumerate(labels) if int(label) == cluster_id]
            if not member_idx:
                cluster_scores.append(-1.0)
                continue
            cluster_scores.append(float(local_success[row_idx, member_idx].mean()))
        chosen_cluster = int(np.argmax(cluster_scores))
        member_idx = [midx for midx, label in enumerate(labels) if int(label) == chosen_cluster]
        best_idx = max(member_idx, key=lambda midx: (global_success[midx], local_success[row_idx, midx]))
        choices[rid] = models[int(best_idx)]
    return choices, {
        "k": min(k, len(source_rows)),
        "clusters": n_clusters,
        "cluster_sizes": dict(Counter(int(label) for label in labels)),
    }


def evaluate_choices(
    method: str,
    choices: dict[str, str],
    target_rows: list[dict[str, Any]],
    target_matrix: dict[str, dict[str, bool]],
    target_ids: list[str],
    best_single_model: str,
    best_single_acc: float,
    oracle_acc: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    acc = accuracy_for_choices(target_ids, choices, target_matrix)
    oracle_gain = oracle_acc - best_single_acc
    gain = acc - best_single_acc
    transfer_ratio = gain / oracle_gain if oracle_gain > 1e-12 else 0.0
    return {
        "method": method,
        "target_accuracy": acc,
        "target_samples": len(target_ids),
        "best_single_target": best_single_acc,
        "best_single_model_target": best_single_model,
        "gain_vs_best_single_target": gain,
        "instance_oracle_target": oracle_acc,
        "oracle_gain": oracle_gain,
        "transfer_ratio": transfer_ratio,
        "models_used": len(target_matrix),
        "routed_models": json.dumps(dict(Counter(choices.values()).most_common()), ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


def render_report(
    path: Path,
    case: CaseSpec,
    rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    target_matrix: dict[str, dict[str, bool]],
    choice_maps: dict[str, dict[str, str]],
) -> None:
    target_ids = [row_id(row) for row in target_rows]
    best_model, best_acc = best_model_for_ids(target_matrix, target_ids)
    oracle_acc = instance_oracle_accuracy(target_matrix, target_ids)
    columns = sorted({group_value(row, case.group_key) for row in target_rows}) + ["Average"]
    name_width = 42
    col_width = 14
    lines = [
        "=" * 110,
        f"Improve2 clean capability routing: {case.title}",
        "=" * 110,
        "| Calibration: source split only; target labels are used only for final scoring.",
        f"| Best single on target: {best_model} ({fmt_pct(best_acc)})",
        f"| Instance oracle on target: {fmt_pct(oracle_acc)}",
        "",
        table_row(name_width, col_width, "Method", columns),
        table_row(name_width, col_width, "-" * 20, ["-" * 10 for _ in columns]),
    ]
    for row in sorted(rows, key=lambda item: -float(item["target_accuracy"])):
        method = str(row["method"])
        choices = choice_maps[method]
        summary = summarize_boolean_choices(target_rows, target_ids, choices, target_matrix, case.group_key)
        values = [
            fmt_pct(summary["by_group"].get(group, {}).get("accuracy")) if group in summary["by_group"] else "N/A"
            for group in columns[:-1]
        ]
        values.append(fmt_pct(summary["accuracy"]))
        lines.append(table_row(name_width, col_width, method, values))
    lines.append("")
    lines.append("All improve2 methods above are calibrated without target labels.")
    write_text(path, lines)


def run_case(case: CaseSpec, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_full = load_full_predictions(case.source_kind, case.source_root)
    target_full = load_full_predictions(case.target_kind, case.target_root)
    source_ids = infer_ids(source_full)
    target_ids = infer_ids(target_full)
    if not source_ids:
        raise RuntimeError(f"No source rows for {case.case_id}: {case.source_root}")
    if not target_ids:
        raise RuntimeError(f"No target rows for {case.case_id}: {case.target_root}")
    source_bool_raw = bool_matrix(source_full)
    target_bool_raw = bool_matrix(target_full)
    source_complete = complete_models(source_bool_raw, source_ids)
    target_complete = complete_models(target_bool_raw, target_ids)
    excluded = set(args.exclude_models or [])
    models = sorted(set(source_complete).intersection(target_complete).difference(excluded))
    if not models:
        raise RuntimeError(f"No common complete models for {case.case_id}")

    source_matrix = {model: source_complete[model] for model in models}
    target_matrix = {model: target_complete[model] for model in models}
    source_rows = first_complete_rows(source_full, source_ids)
    target_rows = first_complete_rows(target_full, target_ids)
    source_y = matrix_array(source_matrix, models, source_ids)
    best_source_model, best_source_acc = best_model_for_ids(source_matrix, source_ids)
    best_target_model, best_target_acc = best_model_for_ids(target_matrix, target_ids)
    oracle_acc = instance_oracle_accuracy(target_matrix, target_ids)

    choice_maps: dict[str, dict[str, str]] = {}
    metadata_by_method: dict[str, dict[str, Any]] = {}
    choice_maps["source_global_best"] = {rid: best_source_model for rid in target_ids}
    metadata_by_method["source_global_best"] = {"source_accuracy": best_source_acc}

    choices, meta = cape_choices(
        source_rows,
        target_rows,
        source_y,
        models,
        args.components,
        args.seed,
        transductive=False,
    )
    choice_maps["cape_capability_nmf"] = choices
    metadata_by_method["cape_capability_nmf"] = meta

    choices, meta = cape_choices(
        source_rows,
        target_rows,
        source_y,
        models,
        args.components,
        args.seed,
        transductive=True,
    )
    choice_maps["shadow_cape_input_aware_nmf"] = choices
    metadata_by_method["shadow_cape_input_aware_nmf"] = meta

    choices, meta = self_fingerprint_knn_choices(
        source_rows,
        target_rows,
        source_y,
        source_matrix,
        models,
        args.knn_k,
    )
    choice_maps["self_expert_fingerprint_knn"] = choices
    metadata_by_method["self_expert_fingerprint_knn"] = meta

    choices, meta = negative_memory_choices(source_rows, target_rows, source_y, models, args.knn_k)
    choice_maps["negative_expert_memory"] = choices
    metadata_by_method["negative_expert_memory"] = meta

    choices, meta = disagreement_topology_choices(
        source_rows,
        target_rows,
        source_y,
        models,
        args.knn_k,
        args.expert_clusters,
    )
    choice_maps["disagreement_topology"] = choices
    metadata_by_method["disagreement_topology"] = meta

    choices, meta = relative_advantage_choices(source_rows, target_rows, source_y, models, transductive=False)
    choice_maps["relative_advantage_ridge"] = choices
    metadata_by_method["relative_advantage_ridge"] = meta

    choices, meta = relative_advantage_choices(source_rows, target_rows, source_y, models, transductive=True)
    choice_maps["shadow_relative_advantage"] = choices
    metadata_by_method["shadow_relative_advantage"] = meta

    rows: list[dict[str, Any]] = []
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
            metadata_by_method.get(method, {}),
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
    write_csv(case_dir / "improve2_results.csv", rows)
    write_json(case_dir / "improve2_results.json", rows)
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
            "note": "Only source labels/correctness are used for routing calibration. Target correctness is used only for final reporting.",
        },
    )
    render_report(
        case_dir / f"Bench_Harness_Result_improve2_{case.case_id}.txt",
        case,
        rows,
        target_rows,
        target_matrix,
        choice_maps,
    )
    return rows


def select_cases(value: str) -> list[CaseSpec]:
    by_id = {case.case_id: case for case in CASES}
    if value == "all":
        return list(CASES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"Unknown case ids: {missing}; valid ids: {sorted(by_id)}")
    return [by_id[item] for item in selected]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for case in select_cases(args.cases):
        all_rows.extend(run_case(case, args))
    write_csv(args.output_dir / "summary.csv", all_rows)
    write_json(args.output_dir / "summary.json", all_rows)
    lines = [
        "# Improve2 Clean Capability Routing Results",
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
