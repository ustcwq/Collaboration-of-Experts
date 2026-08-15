from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.improve2_capability_routing_experiments import (
    CASES as TEXT_CASES,
    bool_matrix,
    complete_models,
    evaluate_choices,
    first_complete_rows,
    group_value,
    infer_ids,
    load_full_predictions,
)
from bench_coe.improve4_failure_modeling_experiments import output_bundle, subset_full
from bench_coe.materialize_innovation_strategies import fmt_pct, summarize_boolean_choices, table_row, write_text
from bench_coe.multimodal_gaokao_mm_ood_innovation import TARGETS, TASK_TO_SUBJECT, gaokao_item_score
from bench_coe.offline_router_innovation_experiments import (
    accuracy_for_choices,
    best_model_for_ids,
    instance_oracle_accuracy,
    row_id,
    write_csv,
    write_json,
)
from bench_coe.shared_eval_utils import infer_option_count, paired_bootstrap_delta


@dataclass(frozen=True)
class OrbitCase:
    case_id: str
    source_kind: str
    source_root: Path
    target_kind: str
    target_root: Path
    group_key: str
    title: str


TEXT_CASE_BY_ID = {case.case_id: case for case in TEXT_CASES}
ORBIT_CASES: tuple[OrbitCase, ...] = (
    OrbitCase(
        "mmlu_val_to_mmlu_test",
        TEXT_CASE_BY_ID["mmlu_val_to_mmlu_test"].source_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_mmlu_test"].source_root,
        TEXT_CASE_BY_ID["mmlu_val_to_mmlu_test"].target_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_mmlu_test"].target_root,
        "category",
        "MMLU-Pro validation -> MMLU-Pro test",
    ),
    OrbitCase(
        "mmlu_val_to_gaokao2010_2022",
        TEXT_CASE_BY_ID["mmlu_val_to_gaokao2010_2022"].source_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_gaokao2010_2022"].source_root,
        TEXT_CASE_BY_ID["mmlu_val_to_gaokao2010_2022"].target_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_gaokao2010_2022"].target_root,
        "subject",
        "MMLU-Pro validation -> GAOKAO-Bench-2010-2022 objective questions",
    ),
    OrbitCase(
        "mmlu_val_to_bbh",
        TEXT_CASE_BY_ID["mmlu_val_to_bbh"].source_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_bbh"].source_root,
        TEXT_CASE_BY_ID["mmlu_val_to_bbh"].target_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_bbh"].target_root,
        "task",
        "MMLU-Pro validation -> BBH",
    ),
    OrbitCase(
        "mmlu_val_to_gpqa",
        TEXT_CASE_BY_ID["mmlu_val_to_gpqa"].source_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_gpqa"].source_root,
        TEXT_CASE_BY_ID["mmlu_val_to_gpqa"].target_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_gpqa"].target_root,
        "domain",
        "MMLU-Pro validation -> GPQA diamond",
    ),
    OrbitCase(
        "mmlu_val_to_mmstar",
        TEXT_CASE_BY_ID["mmlu_val_to_mmstar"].source_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_mmstar"].source_root,
        TEXT_CASE_BY_ID["mmlu_val_to_mmstar"].target_kind,
        TEXT_CASE_BY_ID["mmlu_val_to_mmstar"].target_root,
        "category",
        "MMLU-Pro validation -> MMStar text-only",
    ),
    OrbitCase(
        "portfolio_to_bbh",
        TEXT_CASE_BY_ID["portfolio_to_bbh"].source_kind,
        TEXT_CASE_BY_ID["portfolio_to_bbh"].source_root,
        TEXT_CASE_BY_ID["portfolio_to_bbh"].target_kind,
        TEXT_CASE_BY_ID["portfolio_to_bbh"].target_root,
        "task",
        "Text capability portfolio excluding BBH -> BBH",
    ),
    OrbitCase(
        "portfolio_to_gpqa",
        TEXT_CASE_BY_ID["portfolio_to_gpqa"].source_kind,
        TEXT_CASE_BY_ID["portfolio_to_gpqa"].source_root,
        TEXT_CASE_BY_ID["portfolio_to_gpqa"].target_kind,
        TEXT_CASE_BY_ID["portfolio_to_gpqa"].target_root,
        "domain",
        "Text capability portfolio excluding GPQA -> GPQA diamond",
    ),
    OrbitCase(
        "portfolio_to_mmstar",
        TEXT_CASE_BY_ID["portfolio_to_mmstar"].source_kind,
        TEXT_CASE_BY_ID["portfolio_to_mmstar"].source_root,
        TEXT_CASE_BY_ID["portfolio_to_mmstar"].target_kind,
        TEXT_CASE_BY_ID["portfolio_to_mmstar"].target_root,
        "category",
        "Text capability portfolio excluding MMStar -> MMStar text-only",
    ),
    OrbitCase(
        "gaokao_mm_to_cmmmu",
        "gaokao_mm",
        Path("outputs/gaokao_mm_babyvision_models"),
        "json",
        TARGETS["cmmmu"].single_root,
        TARGETS["cmmmu"].primary_group,
        "GAOKAO-MM -> CMMMU val",
    ),
    OrbitCase(
        "gaokao_mm_to_mathvista",
        "gaokao_mm",
        Path("outputs/gaokao_mm_babyvision_models"),
        "json",
        TARGETS["mathvista"].single_root,
        TARGETS["mathvista"].primary_group,
        "GAOKAO-MM -> MathVista testmini",
    ),
    OrbitCase(
        "gaokao_mm_to_mmmu_pro",
        "gaokao_mm",
        Path("outputs/gaokao_mm_babyvision_models"),
        "json",
        TARGETS["mmmu_pro"].single_root,
        TARGETS["mmmu_pro"].primary_group,
        "GAOKAO-MM -> MMMU-Pro standard 10-options test",
    ),
    OrbitCase(
        "mmmu_pro_to_cmmmu",
        "json",
        TARGETS["mmmu_pro"].single_root,
        "json",
        TARGETS["cmmmu"].single_root,
        TARGETS["cmmmu"].primary_group,
        "MMMU-Pro standard 10-options test -> CMMMU val",
    ),
    OrbitCase(
        "mmmu_pro_to_mathvista",
        "json",
        TARGETS["mmmu_pro"].single_root,
        "json",
        TARGETS["mathvista"].single_root,
        TARGETS["mathvista"].primary_group,
        "MMMU-Pro standard 10-options test -> MathVista testmini",
    ),
    OrbitCase(
        "mmmu_pro_val_to_cmmmu",
        "json",
        Path("outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id"),
        "json",
        TARGETS["cmmmu"].single_root,
        TARGETS["cmmmu"].primary_group,
        "MMMU-Pro validation-id subset -> CMMMU val",
    ),
    OrbitCase(
        "mmmu_pro_val_to_mathvista",
        "json",
        Path("outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id"),
        "json",
        TARGETS["mathvista"].single_root,
        TARGETS["mathvista"].primary_group,
        "MMMU-Pro validation-id subset -> MathVista testmini",
    ),
    OrbitCase(
        "mmmu_pro_val_to_mmmu_pro_test",
        "json",
        Path("outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id"),
        "json",
        Path("outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/test_id"),
        TARGETS["mmmu_pro"].primary_group,
        "MMMU-Pro validation-id subset -> MMMU-Pro test-id subset",
    ),
)


DEFAULT_TEXT_EXCLUDES = ("Qwen3.5-9B", "DeepSeek-R1-0528-Qwen3-8B", "Qwen3-8B")
DEFAULT_VL_EXCLUDES = ("Qwen3.5-9B", "DeepSeek-R1-0528-Qwen3-8B", "Qwen3-8B", "Qwen3-VL-4B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ORBIT-CoE cache-only validation: PARD audit, LEAF lineage-aware base selection, "
            "BRES bias-residual evidence, and QUID unroutability masking. "
            "Routing uses source labels plus target unlabeled expert outputs only."
        )
    )
    parser.add_argument("--cases", default="all", help="Comma-separated case ids, or all.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/orbit_coe_cache_all_datasets"))
    parser.add_argument("--exclude-models", nargs="*", default=None)
    parser.add_argument("--text-exclude-models", nargs="*", default=list(DEFAULT_TEXT_EXCLUDES))
    parser.add_argument("--vl-exclude-models", nargs="*", default=list(DEFAULT_VL_EXCLUDES))
    parser.add_argument("--leaf-iters", type=int, default=8)
    parser.add_argument("--leaf-source-weight", type=float, default=0.35)
    parser.add_argument("--lineage-cap", type=float, default=1.25)
    parser.add_argument("--quid-leaf-margin", type=float, default=0.18)
    parser.add_argument("--quid-bres-margin", type=float, default=0.35)
    parser.add_argument("--quid-max-coverage-null", type=float, default=0.92)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_cases(value: str) -> list[OrbitCase]:
    by_id = {case.case_id: case for case in ORBIT_CASES}
    if value == "all":
        return list(ORBIT_CASES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"Unknown ORBIT case ids: {missing}; valid ids: {sorted(by_id)}")
    return [by_id[item] for item in selected]


def normalize_answer(value: Any) -> str:
    if value is None:
        return "<empty>"
    if isinstance(value, list):
        value = "".join(str(item) for item in value)
    text = str(value).strip()
    if not text:
        return "<empty>"
    text = re.sub(r"\s+", " ", text)
    return text.strip().upper()


def load_gaokao_mm_source_full(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    full: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
            except Exception:
                summary = {}
            if summary.get("status") != "completed":
                continue
        rows_by_id: dict[str, dict[str, Any]] = {}
        for path in sorted(model_dir.glob(f"{model_dir.name}_2010-2023_*.json")):
            payload = read_json(path)
            keyword = str(payload.get("keyword", ""))
            subject = TASK_TO_SUBJECT.get(keyword, keyword)
            question_type = str(payload.get("question_type", ""))
            for item in payload.get("example", []):
                rid = f"{keyword}:{item.get('index')}"
                score = gaokao_item_score(keyword, item)
                if score is None:
                    continue
                model_answer = normalize_answer(item.get("model_answer", []))
                gold = normalize_answer(item.get("standard_answer", []))
                rows_by_id[rid] = {
                    "id": rid,
                    "benchmark": "gaokao_mm",
                    "subject": subject,
                    "keyword": keyword,
                    "question_type": question_type,
                    "question": item.get("question", ""),
                    "answer": gold,
                    "pred": model_answer,
                    "prediction": model_answer,
                    "response": item.get("model_output", ""),
                    "model_outputs": item.get("model_output", ""),
                    "score": float(score),
                    "is_correct": bool(score >= 1.0),
                    "has_partial_credit": bool(0.0 < score < 1.0),
                    "model_error": item.get("model_error"),
                    "model_input_truncated": item.get("model_input_truncated"),
                    "picture": item.get("picture", []),
                    "resolved_picture": item.get("resolved_picture", []),
                    "combined_image": item.get("combined_image"),
                }
        if rows_by_id:
            full[model_dir.name] = rows_by_id
    return full


def load_orbit_full(kind: str, root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if kind == "gaokao_mm":
        return load_gaokao_mm_source_full(root)
    return load_full_predictions(kind, root)


def is_vl_orbit_case(case: OrbitCase) -> bool:
    roots = f"{case.source_root} {case.target_root}".lower()
    return case.source_kind == "gaokao_mm" or "multimodal_babyvision_models" in roots


def score_matrix(full: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model, rows_by_id in full.items():
        out[model] = {}
        for rid, row in rows_by_id.items():
            if row.get("score") is not None:
                out[model][rid] = float(row["score"])
            elif row.get("is_correct") is not None:
                out[model][rid] = 1.0 if bool(row["is_correct"]) else 0.0
            else:
                pred = row.get("pred", row.get("prediction"))
                gold = row.get("answer", row.get("target"))
                out[model][rid] = 1.0 if pred is not None and str(pred).strip() == str(gold).strip() else 0.0
    return out


def complete_score_models(matrix: dict[str, dict[str, float]], ids: list[str]) -> dict[str, dict[str, float]]:
    return {model: values for model, values in matrix.items() if all(rid in values for rid in ids)}


def matrix_array_float(matrix: dict[str, dict[str, float]], models: list[str], ids: list[str]) -> Any:
    import numpy as np

    return np.asarray([[float(matrix[model].get(rid, 0.0)) for model in models] for rid in ids], dtype=float)


def lineage_key(model: str) -> str:
    name = model.lower()
    if "qwen" in name and ("vl" in name or "vision" in name):
        return "qwen-vl"
    if "qwen" in name:
        return "qwen-text"
    if "deepseek" in name:
        return "deepseek"
    if "internvl" in name:
        return "internvl"
    if "internlm" in name:
        return "internlm"
    if "minicpm" in name:
        return "minicpm-v"
    if "smolvlm" in name:
        return "smolvlm"
    if "lfm" in name:
        return "lfm-vl"
    if "gemma" in name:
        return "gemma"
    if "glm" in name:
        return "glm"
    if "kimi" in name:
        return "kimi-vl"
    if "llama" in name:
        return "llama"
    if "yi" in name:
        return "yi"
    if "granite" in name:
        return "granite"
    if "ministral" in name:
        return "ministral"
    if "mammoth" in name:
        return "mammoth"
    if "nemotron" in name:
        return "nemotron"
    if "baichuan" in name:
        return "baichuan"
    if "aya" in name:
        return "aya"
    if "general-reasoner" in name:
        return "general-reasoner"
    return re.split(r"[-_/]", name)[0] or name


def build_lineages(models: list[str]) -> dict[str, list[str]]:
    lineages: dict[str, list[str]] = defaultdict(list)
    for model in models:
        lineages[lineage_key(model)].append(model)
    return dict(sorted(lineages.items()))


def model_answer_matrix(full: dict[str, dict[str, dict[str, Any]]], models: list[str], ids: list[str]) -> list[list[str]]:
    answers, _, _ = output_bundle(full, models, ids)
    return [[normalize_answer(answer) for answer in row] for row in answers]


def leaf_estimate(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np

    lineages = build_lineages(models)
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    source_acc = np.asarray(source_y.mean(axis=0), dtype=float)
    source_var = np.asarray(source_y.var(axis=0), dtype=float)
    target_answers = model_answer_matrix(target_full, models, target_ids)

    reliability = np.clip(source_acc.copy(), 1e-4, 1.0 - 1e-4)
    posteriors: list[dict[str, float]] = []
    for _ in range(max(1, args.leaf_iters)):
        posteriors = []
        vote_support_sum = np.zeros(len(models), dtype=float)
        vote_support_count = np.zeros(len(models), dtype=float)
        for row_answers in target_answers:
            candidates = sorted({ans for ans in row_answers if ans and ans != "<EMPTY>" and ans != "<empty>"})
            if not candidates:
                posteriors.append({})
                continue
            scores = {candidate: 0.0 for candidate in candidates}
            for lineage_models in lineages.values():
                lineage_votes = Counter(row_answers[model_to_idx[model]] for model in lineage_models)
                lineage_total = max(1, sum(lineage_votes.values()))
                lineage_rel = float(np.mean([reliability[model_to_idx[model]] for model in lineage_models]))
                lineage_weight = max(0.01, math.log((lineage_rel + 1e-4) / (1.0 - lineage_rel + 1e-4)))
                lineage_weight = max(-1.5, min(args.lineage_cap, lineage_weight))
                for candidate in candidates:
                    share = lineage_votes.get(candidate, 0) / lineage_total
                    if share > 0.0:
                        scores[candidate] += lineage_weight * share
            max_score = max(scores.values())
            exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
            denom = sum(exp_scores.values()) or 1.0
            posterior = {key: value / denom for key, value in exp_scores.items()}
            posteriors.append(posterior)
            for midx, ans in enumerate(row_answers):
                vote_support_sum[midx] += posterior.get(ans, 0.0)
                vote_support_count[midx] += 1.0
        pseudo_rel = np.divide(vote_support_sum, np.maximum(vote_support_count, 1.0))
        reliability = np.clip(
            args.leaf_source_weight * source_acc + (1.0 - args.leaf_source_weight) * pseudo_rel,
            1e-4,
            1.0 - 1e-4,
        )

    n = max(1, len(target_ids))
    # Conservative lower bound: target pseudo-reliability with source variance and binomial width.
    lcb = reliability - np.sqrt(np.maximum(reliability * (1.0 - reliability), source_var) / n)
    base_idx = int(np.argmax(lcb))
    posterior_choices: dict[str, str] = {}
    answer_choices: dict[str, str] = {}
    margins: dict[str, float] = {}
    for rid, row_answers, posterior in zip(target_ids, target_answers, posteriors):
        if not posterior:
            posterior_choices[rid] = models[base_idx]
            answer_choices[rid] = row_answers[base_idx]
            margins[rid] = 0.0
            continue
        ranked = sorted(posterior.items(), key=lambda item: (-item[1], item[0]))
        best_answer = ranked[0][0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margins[rid] = float(ranked[0][1] - second)
        candidate_indices = [idx for idx, ans in enumerate(row_answers) if ans == best_answer]
        if not candidate_indices:
            chosen_idx = base_idx
        else:
            chosen_idx = max(candidate_indices, key=lambda idx: (reliability[idx], source_acc[idx], -idx))
        posterior_choices[rid] = models[int(chosen_idx)]
        answer_choices[rid] = best_answer
    return {
        "models": models,
        "lineages": lineages,
        "source_accuracy": {model: float(source_acc[idx]) for idx, model in enumerate(models)},
        "target_reliability": {model: float(reliability[idx]) for idx, model in enumerate(models)},
        "target_lcb": {model: float(lcb[idx]) for idx, model in enumerate(models)},
        "base_model": models[base_idx],
        "base_lcb": float(lcb[base_idx]),
        "posterior_choices": posterior_choices,
        "answer_choices": answer_choices,
        "margins": margins,
        "posteriors": posteriors,
        "target_answers": target_answers,
    }


def answer_bias_models(
    source_answers: list[list[str]],
    source_y: Any,
    models: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for midx, model in enumerate(models):
        correct_total = float(source_y[:, midx].sum())
        total = float(source_y.shape[0])
        global_acc = (correct_total + 2.0) / (total + 4.0)
        vote_count: Counter[str] = Counter()
        vote_correct: Counter[str] = Counter()
        wrong_vote_count: Counter[str] = Counter()
        for row_idx, row_answers in enumerate(source_answers):
            ans = row_answers[midx]
            vote_count[ans] += 1
            if float(source_y[row_idx, midx]) >= 0.5:
                vote_correct[ans] += 1
            else:
                wrong_vote_count[ans] += 1
        evidence: dict[str, float] = {}
        wrong_total = max(1, sum(wrong_vote_count.values()))
        for ans, count in vote_count.items():
            precision = (vote_correct[ans] + 2.0 * global_acc) / (count + 2.0)
            wrong_bias = (wrong_vote_count[ans] + 1.0) / (wrong_total + len(vote_count) + 1.0)
            precision = min(0.995, max(0.005, precision))
            evidence[ans] = math.log(precision / (1.0 - precision)) - 0.35 * math.log(wrong_bias + 1e-6)
        out[model] = {
            "global_acc": global_acc,
            "evidence": evidence,
            "default_evidence": math.log(global_acc / (1.0 - global_acc)),
            "wrong_bias": dict(wrong_vote_count),
        }
    return out


def bres_choices(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    leaf: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    source_answers = model_answer_matrix(source_full, models, source_ids)
    target_answers = leaf["target_answers"]
    bias = answer_bias_models(source_answers, source_y, models)
    lineages = leaf["lineages"]
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    target_rel = leaf["target_reliability"]

    choices: dict[str, str] = {}
    answer_choices: dict[str, str] = {}
    margins: dict[str, float] = {}
    score_debug: dict[str, dict[str, float]] = {}
    for rid, row_answers in zip(target_ids, target_answers):
        candidates = sorted({ans for ans in row_answers if ans and ans != "<EMPTY>" and ans != "<empty>"})
        if not candidates:
            choices[rid] = leaf["base_model"]
            answer_choices[rid] = row_answers[model_to_idx[leaf["base_model"]]]
            margins[rid] = 0.0
            continue
        scores = {candidate: 0.0 for candidate in candidates}
        for lineage_name, lineage_models in lineages.items():
            lineage_scores = {candidate: 0.0 for candidate in candidates}
            for model in lineage_models:
                midx = model_to_idx[model]
                ans = row_answers[midx]
                if ans not in scores:
                    continue
                evidence = bias[model]["evidence"].get(ans, bias[model]["default_evidence"])
                # Target pseudo-reliability gates source residual evidence without using target labels.
                reliability_gate = max(0.25, float(target_rel[model]))
                lineage_scores[ans] += evidence * reliability_gate
            for candidate, value in lineage_scores.items():
                if value > 0:
                    scores[candidate] += min(args.lineage_cap, value)
                else:
                    scores[candidate] += max(-args.lineage_cap, value)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_answer = ranked[0][0]
        second = ranked[1][1] if len(ranked) > 1 else ranked[0][1] - 1.0
        margins[rid] = float(ranked[0][1] - second)
        candidate_indices = [idx for idx, ans in enumerate(row_answers) if ans == best_answer]
        chosen_idx = max(
            candidate_indices,
            key=lambda idx: (float(target_rel[models[idx]]), bias[models[idx]]["global_acc"], -idx),
        )
        choices[rid] = models[int(chosen_idx)]
        answer_choices[rid] = best_answer
        if len(score_debug) < 50:
            score_debug[rid] = {key: float(value) for key, value in scores.items()}
    return choices, {
        "answer_choices": answer_choices,
        "margins": margins,
        "bias_models": bias,
        "sample_scores": score_debug,
    }


def quid_masks(
    target_rows: list[dict[str, Any]],
    target_ids: list[str],
    leaf: dict[str, Any],
    bres_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lineages = leaf["lineages"]
    # Preserve model order from target_answers by reconstructing it from lineages is unsafe; pass row support by answer only.
    target_answers = leaf["target_answers"]
    rows_by_id = {row_id(row): row for row in target_rows}
    identified: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    coverage_nulls: dict[str, float | None] = {}
    effective_lineage_support: dict[str, int] = {}
    ordered_model_to_idx = {model: idx for idx, model in enumerate(leaf["models"])}
    for rid, row_answers in zip(target_ids, target_answers):
        answer_to_lineages: dict[str, set[str]] = defaultdict(set)
        for lineage_name, lineage_models in lineages.items():
            for model in lineage_models:
                idx = ordered_model_to_idx[model]
                answer_to_lineages[row_answers[idx]].add(lineage_name)
        chosen_answer = bres_meta["answer_choices"].get(rid, leaf["answer_choices"].get(rid))
        eff = len(answer_to_lineages.get(chosen_answer, set()))
        effective_lineage_support[rid] = eff
        option_count = infer_option_count(rows_by_id[rid])
        unique_answers = {ans for ans in row_answers if ans and ans not in {"<EMPTY>", "<empty>"}}
        cov = min(1.0, len(unique_answers) / float(option_count)) if option_count else None
        coverage_nulls[rid] = cov
        leaf_margin = float(leaf["margins"].get(rid, 0.0))
        bres_margin = float(bres_meta["margins"].get(rid, 0.0))
        if eff < 2:
            identified[rid] = False
            reasons[rid] = "single_lineage_support"
        elif cov is not None and cov >= args.quid_max_coverage_null and leaf_margin < 0.35:
            identified[rid] = False
            reasons[rid] = "coverage_null_dominates"
        elif leaf_margin >= args.quid_leaf_margin or bres_margin >= args.quid_bres_margin:
            identified[rid] = True
            reasons[rid] = "margin_identified"
        else:
            identified[rid] = False
            reasons[rid] = "ambiguous_decoy_worlds"
    return {
        "identified": identified,
        "reasons": reasons,
        "coverage_nulls": coverage_nulls,
        "effective_lineage_support": effective_lineage_support,
    }


def quid_safe_choices(
    target_ids: list[str],
    leaf: dict[str, Any],
    bres_choices_map: dict[str, str],
    quid: dict[str, Any],
) -> dict[str, str]:
    base = leaf["base_model"]
    return {rid: bres_choices_map[rid] if quid["identified"].get(rid, False) else base for rid in target_ids}


def target_group_report(
    case: OrbitCase,
    rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    target_matrix: dict[str, dict[str, bool]],
    choice_maps: dict[str, dict[str, str]],
    audit: dict[str, Any],
    path: Path,
) -> None:
    target_ids = [row_id(row) for row in target_rows]
    best_model, best_acc = best_model_for_ids(target_matrix, target_ids)
    oracle_acc = instance_oracle_accuracy(target_matrix, target_ids)
    columns = sorted({group_value(row, case.group_key) for row in target_rows}) + ["Average"]
    name_width = 38
    col_width = 14
    lines = [
        "=" * 110,
        f"ORBIT-CoE cache-only validation: {case.title}",
        "=" * 110,
        "| Calibration: source labels + target unlabeled expert outputs only; target labels are final scoring only.",
        f"| Best single on target: {best_model} ({fmt_pct(best_acc)})",
        f"| Instance oracle on target: {fmt_pct(oracle_acc)}",
        f"| LEAF base: {audit.get('leaf_base_model')} ({fmt_pct(audit.get('leaf_base_lcb'))} pseudo-LCB)",
        f"| QUID identified coverage: {fmt_pct(audit.get('quid_identified_rate'))}",
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
    lines.extend(
        [
            "",
            "ORBIT audit:",
            f"- Lineages: {json.dumps(audit.get('lineages', {}), ensure_ascii=False)}",
            f"- QUID reasons: {json.dumps(audit.get('quid_reason_counts', {}), ensure_ascii=False)}",
            "",
            "Implemented cache-only ORBIT modules: PARD source audit, LEAF-CoE, BRES-CoE, QUID-CoE.",
            "Not implemented in this cache-only run: ECI, CAST, RELAY, WITNESS; these require new model calls or generated instruments.",
        ]
    )
    write_text(path, lines)


def pard_audit(source_full_raw: dict[str, dict[str, dict[str, Any]]], models: list[str], source_ids: list[str]) -> dict[str, Any]:
    rows = []
    for model in models:
        valid = 0
        empty = 0
        truncated = 0
        partial = 0
        errors = 0
        total = 0
        for rid in source_ids:
            row = source_full_raw[model][rid]
            total += 1
            pred = normalize_answer(row.get("pred", row.get("prediction")))
            if pred in {"<EMPTY>", "Z"}:
                empty += 1
            else:
                valid += 1
            if row.get("model_input_truncated") or row.get("prompt_was_truncated"):
                truncated += 1
            if row.get("has_partial_credit"):
                partial += 1
            if row.get("model_error"):
                errors += 1
        rows.append(
            {
                "model": model,
                "total": total,
                "valid_answer_rate": valid / total if total else 0.0,
                "empty_answer_rate": empty / total if total else 0.0,
                "truncated_rate": truncated / total if total else 0.0,
                "partial_credit_rate": partial / total if total else 0.0,
                "model_error_rate": errors / total if total else 0.0,
            }
        )
    return {
        "rows": rows,
        "mean_valid_answer_rate": sum(row["valid_answer_rate"] for row in rows) / len(rows) if rows else None,
        "mean_empty_answer_rate": sum(row["empty_answer_rate"] for row in rows) / len(rows) if rows else None,
    }


def run_case(case: OrbitCase, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_full_raw = load_orbit_full(case.source_kind, case.source_root)
    target_full_raw = load_orbit_full(case.target_kind, case.target_root)
    source_ids = infer_ids(source_full_raw)
    target_ids = infer_ids(target_full_raw)
    source_scores_raw = score_matrix(source_full_raw)
    target_bool_raw = bool_matrix(target_full_raw)
    source_complete = complete_score_models(source_scores_raw, source_ids)
    target_complete = complete_models(target_bool_raw, target_ids)
    if args.exclude_models is not None:
        excluded = set(args.exclude_models)
    elif is_vl_orbit_case(case):
        excluded = set(args.vl_exclude_models)
    else:
        excluded = set(args.text_exclude_models)
    models = sorted(set(source_complete).intersection(target_complete).difference(excluded))
    if not models:
        raise RuntimeError(f"No common complete models for {case.case_id}; excluded={sorted(excluded)}")

    source_full = subset_full(source_full_raw, models, source_ids)
    target_full = subset_full(target_full_raw, models, target_ids)
    source_scores = {model: source_complete[model] for model in models}
    target_matrix = {model: target_complete[model] for model in models}
    source_rows = first_complete_rows(source_full, source_ids)
    target_rows = first_complete_rows(target_full, target_ids)
    source_y = matrix_array_float(source_scores, models, source_ids)
    best_source_idx = max(range(len(models)), key=lambda idx: float(source_y[:, idx].mean()))
    best_source_model = models[best_source_idx]
    best_source_acc = float(source_y[:, best_source_idx].mean())
    best_target_model, best_target_acc = best_model_for_ids(target_matrix, target_ids)
    oracle_acc = instance_oracle_accuracy(target_matrix, target_ids)

    leaf = leaf_estimate(source_full, target_full, source_y, models, source_ids, target_ids, args)
    bres_map, bres_meta = bres_choices(source_full, target_full, source_y, models, source_ids, target_ids, leaf, args)
    quid = quid_masks(target_rows, target_ids, leaf, bres_meta, args)
    quid_map = quid_safe_choices(target_ids, leaf, bres_map, quid)

    choice_maps: dict[str, dict[str, str]] = {
        "source_global_best": {rid: best_source_model for rid in target_ids},
        "leaf_base": {rid: leaf["base_model"] for rid in target_ids},
        "leaf_posterior_vote": leaf["posterior_choices"],
        "bres_residual_evidence": bres_map,
        "orbit_quid_safe": quid_map,
    }
    metadata = {
        "source_global_best": {"source_accuracy": best_source_acc},
        "leaf_base": {"leaf_base_lcb": leaf["base_lcb"], "lineages": leaf["lineages"]},
        "leaf_posterior_vote": {"lineages": leaf["lineages"], "leaf_source_weight": args.leaf_source_weight},
        "bres_residual_evidence": {"lineage_cap": args.lineage_cap, "idea": "bias-residual answer evidence"},
        "orbit_quid_safe": {
            "identified_rate": sum(1 for value in quid["identified"].values() if value) / len(target_ids),
            "quid_reasons": dict(Counter(quid["reasons"].values())),
        },
    }

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
            metadata.get(method, {}),
        )
        row.update(paired_bootstrap_delta(choices, best_target_model, target_matrix, target_ids, args.bootstrap_iters, args.seed))
        row.update(
            {
                "case_id": case.case_id,
                "source_kind": case.source_kind,
                "target_kind": case.target_kind,
                "source_samples": len(source_ids),
                "target_samples": len(target_ids),
                "source_global_best": best_source_model,
                "source_global_best_accuracy": best_source_acc,
                "leaf_base_model": leaf["base_model"],
                "leaf_base_lcb": leaf["base_lcb"],
            }
        )
        rows.append(row)

    audit = {
        "case_id": case.case_id,
        "title": case.title,
        "source_samples": len(source_ids),
        "target_samples": len(target_ids),
        "models": models,
        "excluded_models": sorted(excluded),
        "best_source_model": best_source_model,
        "best_source_accuracy": best_source_acc,
        "best_target_model": best_target_model,
        "best_target_accuracy": best_target_acc,
        "instance_oracle": oracle_acc,
        "lineages": leaf["lineages"],
        "leaf_target_reliability": leaf["target_reliability"],
        "leaf_target_lcb": leaf["target_lcb"],
        "leaf_base_model": leaf["base_model"],
        "leaf_base_lcb": leaf["base_lcb"],
        "quid_identified_rate": sum(1 for value in quid["identified"].values() if value) / len(target_ids),
        "quid_reason_counts": dict(Counter(quid["reasons"].values())),
        "quid_effective_lineage_support_counts": dict(Counter(quid["effective_lineage_support"].values())),
        "pard_source_audit": pard_audit(source_full, models, source_ids),
        "note": "ORBIT routing uses source labels and target unlabeled outputs only; target correctness is final scoring only.",
    }

    case_dir = args.output_dir / case.case_id
    write_csv(case_dir / "orbit_results.csv", rows)
    write_json(case_dir / "orbit_results.json", rows)
    write_json(case_dir / "choices_by_method.json", choice_maps)
    write_json(case_dir / "orbit_audit.json", audit)
    write_json(case_dir / "leaf_state.json", {key: value for key, value in leaf.items() if key not in {"target_answers", "posteriors"}})
    write_json(
        case_dir / "manifest.json",
        {
            "case_id": case.case_id,
            "title": case.title,
            "source_kind": case.source_kind,
            "source_root": str(case.source_root),
            "target_kind": case.target_kind,
            "target_root": str(case.target_root),
            "models": models,
            "excluded_models": sorted(excluded),
        },
    )
    target_group_report(case, rows, target_rows, target_matrix, choice_maps, audit, case_dir / f"Bench_Harness_Result_orbit_{case.case_id}.txt")
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    lines = [
        "# ORBIT-CoE Cache-Only Validation Results",
        "",
        "Implemented cache-only modules from `Bench-CoE_ORBIT-CoE_全数据集开创性创新方案.md`:",
        "",
        "- PARD audit: source output protocol/empty/truncation audit from cached outputs.",
        "- LEAF-CoE lite: lineage-quotiented target-unlabeled reliability and base selection.",
        "- BRES-CoE lite: source-oriented bias-residual answer evidence with lineage capping.",
        "- QUID-CoE lite: fixed-threshold unroutability mask using posterior margin, residual margin, lineage support, and coverage null.",
        "",
        "Not implemented in this cache-only run: ECI, CAST, RELAY, WITNESS. They require generated instruments or new model calls.",
        "",
        "Routing policies use source labels plus target unlabeled expert outputs only. Target labels are final scoring only.",
        "",
        "| Case | Method | Target Acc | Best Single | Gain | Paired CI | LEAF Base | Models |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["case_id"], -float(item["target_accuracy"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_id']}`",
                    f"`{row['method']}`",
                    fmt_pct(row["target_accuracy"]),
                    f"{fmt_pct(row['best_single_target'])} ({row['best_single_model_target']})",
                    f"{float(row['gain_vs_best_single_target']) * 100:+.2f}%",
                    f"[{float(row.get('paired_ci_low', 0.0)) * 100:+.2f}%, {float(row.get('paired_ci_high', 0.0)) * 100:+.2f}%]",
                    str(row.get("leaf_base_model", "")),
                    str(row.get("models_used", "")),
                ]
            )
            + " |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "summary.md", lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for case in select_cases(args.cases):
        all_rows.extend(run_case(case, args))
    write_summary(args.output_dir, all_rows)


if __name__ == "__main__":
    main()
