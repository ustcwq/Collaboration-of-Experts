from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.atlas_coe_experiments import (
    DEFAULT_CASES,
    DEFAULT_TEXT_EXCLUDES,
    DEFAULT_VL_EXCLUDES,
    atlas_alignment,
    is_vl_case,
    normalize_answer,
    proposal_builders,
    select_cases,
    simultaneous_bootstrap_ci,
    source_lobo_method_weights,
)
from bench_coe.improve2_capability_routing_experiments import (
    CaseSpec,
    bool_matrix,
    complete_models,
    evaluate_choices,
    first_complete_rows,
    group_value,
    infer_ids,
    load_full_predictions,
    matrix_array,
)
from bench_coe.improve4_failure_modeling_experiments import output_bundle, subset_full
from bench_coe.materialize_innovation_strategies import fmt_pct, summarize_boolean_choices, table_row, write_text
from bench_coe.offline_router_innovation_experiments import (
    best_model_for_ids,
    instance_oracle_accuracy,
    row_id,
    write_csv,
    write_json,
)
from bench_coe.orbit_coe_experiments import leaf_estimate, lineage_key
from bench_coe.shared_eval_utils import infer_option_count, paired_bootstrap_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SPECTRA-CoE Lite cache-only validation. The router uses source labels plus "
            "target-unlabeled expert outputs only. Target labels are opened only for final scoring."
        )
    )
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated case ids, or all.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/spectra_coe_lite_source_transfer_validation"))
    parser.add_argument("--exclude-models", nargs="*", default=None)
    parser.add_argument("--text-exclude-models", nargs="*", default=list(DEFAULT_TEXT_EXCLUDES))
    parser.add_argument("--vl-exclude-models", nargs="*", default=list(DEFAULT_VL_EXCLUDES))
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--state-clusters", type=int, default=12)
    parser.add_argument("--leaf-iters", type=int, default=8)
    parser.add_argument("--leaf-source-weight", type=float, default=0.35)
    parser.add_argument("--lineage-cap", type=float, default=1.25)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--null-worlds", type=int, default=99)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--lobo-max-splits", type=int, default=6)
    parser.add_argument("--lobo-max-heldout", type=int, default=500)
    parser.add_argument("--tapestry-eta", type=float, default=8.0)
    parser.add_argument("--spectra-source-prior-weight", type=float, default=0.28)
    parser.add_argument("--spectra-null-alpha", type=float, default=0.30)
    parser.add_argument("--spectra-green-threshold", type=float, default=0.48)
    parser.add_argument("--spectra-red-threshold", type=float, default=0.31)
    parser.add_argument("--spectra-item-margin", type=float, default=0.045)
    parser.add_argument("--spectra-strong-margin", type=float, default=0.075)
    parser.add_argument("--spectra-min-eff-lineage", type=float, default=0.34)
    parser.add_argument("--spectra-fallback", choices=("source", "leaf", "posterior"), default="posterior")
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def logit(value: float) -> float:
    clipped = max(1e-5, min(1.0 - 1e-5, value))
    return math.log(clipped / (1.0 - clipped))


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    exp_scores = {key: math.exp(value - top) for key, value in scores.items()}
    denom = sum(exp_scores.values()) or 1.0
    return {key: value / denom for key, value in exp_scores.items()}


def model_answers(full: dict[str, dict[str, dict[str, Any]]], models: list[str], ids: list[str]) -> list[list[str]]:
    answers, _raw, _stats = output_bundle(full, models, ids)
    return [[normalize_answer(answer) for answer in row] for row in answers]


def answer_symbol(answer: str) -> bool:
    text = normalize_answer(answer)
    return len(text) == 1 and "A" <= text <= "J"


def target_answer_frequencies(target_answers: list[list[str]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for row in target_answers:
        for answer in row:
            if answer and answer not in {"<EMPTY>", "<EMPTY>", "<empty>"} and answer_symbol(answer):
                counts[answer] += 1
                total += 1
    return {answer: count / max(1, total) for answer, count in counts.items()}


def lineage_support_stats(
    row_answers: list[str],
    models: list[str],
    lineages: list[str],
    source_accuracy: dict[str, float],
    transported_success: dict[str, float] | None,
) -> dict[str, dict[str, float]]:
    lineage_total = max(1, len(set(lineages)))
    answer_to_lineages: dict[str, set[str]] = defaultdict(set)
    answer_to_models: dict[str, list[str]] = defaultdict(list)
    for midx, answer in enumerate(row_answers):
        if not answer or answer in {"<EMPTY>", "<empty>"}:
            continue
        answer_to_lineages[answer].add(lineages[midx])
        answer_to_models[answer].append(models[midx])
    out: dict[str, dict[str, float]] = {}
    for answer, supporters in answer_to_models.items():
        src_values = [source_accuracy.get(model, 0.0) for model in supporters]
        transported_values = [
            transported_success.get(model, source_accuracy.get(model, 0.0)) if transported_success else source_accuracy.get(model, 0.0)
            for model in supporters
        ]
        lineage_count = len(answer_to_lineages[answer])
        out[answer] = {
            "model_share": len(supporters) / max(1, len(models)),
            "lineage_share": lineage_count / lineage_total,
            "effective_lineage_count": float(lineage_count),
            "source_support": sum(src_values) / max(1, len(src_values)),
            "transported_support": sum(transported_values) / max(1, len(transported_values)),
        }
    return out


def answer_support(row_answers: list[str], lineages: list[str], answer: str) -> tuple[float, int]:
    support = {lineages[idx] for idx, value in enumerate(row_answers) if value == answer}
    lineage_total = max(1, len(set(lineages)))
    return len(support) / lineage_total, len(support)


def option_bias_penalty(
    answer: str,
    global_freq: dict[str, float],
    option_count: int | None,
    candidate_count: int,
) -> float:
    if not answer_symbol(answer):
        return 0.0
    expected = 1.0 / float(option_count or max(2, candidate_count))
    return max(0.0, global_freq.get(answer, 0.0) - expected)


def source_accuracy_map(source_y: Any, models: list[str]) -> dict[str, float]:
    return {model: float(source_y[:, midx].mean()) for midx, model in enumerate(models)}


def choose_supporting_model(
    answer: str,
    row_answers: list[str],
    models: list[str],
    source_accuracy: dict[str, float],
    transported_success: dict[str, float] | None,
    leaf_reliability: dict[str, float],
    fallback_model: str,
) -> str:
    candidates = [idx for idx, value in enumerate(row_answers) if value == answer]
    if not candidates:
        return fallback_model
    return models[
        max(
            candidates,
            key=lambda idx: (
                float(leaf_reliability.get(models[idx], 0.0)),
                float(transported_success.get(models[idx], source_accuracy.get(models[idx], 0.0)) if transported_success else 0.0),
                float(source_accuracy.get(models[idx], 0.0)),
                -idx,
            ),
        )
    ]


def static_target_majority_choices(
    target_answers: list[list[str]],
    target_ids: list[str],
    models: list[str],
    source_accuracy: dict[str, float],
    leaf_reliability: dict[str, float],
    fallback_model: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    choices: dict[str, str] = {}
    answer_choices: dict[str, str] = {}
    margins: dict[str, float] = {}
    for rid, row_answers in zip(target_ids, target_answers):
        counts = Counter(answer for answer in row_answers if answer and answer not in {"<EMPTY>", "<empty>"})
        if not counts:
            choices[rid] = fallback_model
            answer_choices[rid] = row_answers[models.index(fallback_model)] if fallback_model in models else "<empty>"
            margins[rid] = 0.0
            continue
        ranked = counts.most_common()
        top_answer, top_count = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        answer_choices[rid] = top_answer
        margins[rid] = (top_count - second) / max(1, len(models))
        choices[rid] = choose_supporting_model(
            top_answer,
            row_answers,
            models,
            source_accuracy,
            None,
            leaf_reliability,
            fallback_model,
        )
    return choices, {"answer_choices": answer_choices, "margins": margins, "module": "static target majority"}


def leaf_posterior_fallback_model(
    leaf: dict[str, Any],
    target_answers: list[list[str]],
    target_ids: list[str],
    models: list[str],
    source_accuracy: dict[str, float],
    alignment: dict[str, Any],
    source_fallback: str,
) -> tuple[str, dict[str, Any]]:
    posteriors = leaf.get("posteriors", [])
    support_sum = {model: 0.0 for model in models}
    transported_sum = {model: 0.0 for model in models}
    count = 0
    for rid, row_answers, posterior in zip(target_ids, target_answers, posteriors):
        normalized_posterior = {normalize_answer(answer): float(prob) for answer, prob in dict(posterior).items()}
        transported = alignment.get("transported_success", {}).get(rid, {})
        for midx, model in enumerate(models):
            support_sum[model] += normalized_posterior.get(row_answers[midx], 0.0)
            transported_sum[model] += float(transported.get(model, source_accuracy.get(model, 0.0)))
        count += 1
    scores: dict[str, float] = {}
    for model in models:
        posterior_support = support_sum[model] / max(1, count)
        transported_support = transported_sum[model] / max(1, count)
        scores[model] = 0.72 * posterior_support + 0.18 * source_accuracy.get(model, 0.0) + 0.10 * transported_support
    fallback = max(models, key=lambda model: (scores[model], source_accuracy.get(model, 0.0), model))
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fallback, {
        "module": "target-unlabeled posterior-supported single fallback",
        "source_fallback": source_fallback,
        "leaf_base_model": leaf.get("base_model"),
        "fallback_model": fallback,
        "ranked_scores": ranked[:10],
        "note": "Scores use source labels and target-unlabeled LEAF posteriors only; target labels are not used.",
    }


def spectra_item_posteriors(
    case: CaseSpec,
    target_rows: list[dict[str, Any]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    target_answers: list[list[str]],
    source_accuracy: dict[str, float],
    leaf: dict[str, Any],
    alignment: dict[str, Any],
    models: list[str],
    target_ids: list[str],
    fallback_model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = random.Random(args.seed + 919)
    rows_by_id = {row_id(row): row for row in target_rows}
    lineages = [lineage_key(model) for model in models]
    lineage_total = max(1, len(set(lineages)))
    answer_freq = target_answer_frequencies(target_answers)
    leaf_reliability = {model: float(value) for model, value in leaf.get("target_reliability", {}).items()}
    leaf_answers = {rid: normalize_answer(answer) for rid, answer in leaf.get("answer_choices", {}).items()}
    spectra_choices: dict[str, str] = {}
    answer_choices: dict[str, str] = {}
    item_rows: list[dict[str, Any]] = []
    item_debug: dict[str, Any] = {}

    for rid, row_answers in zip(target_ids, target_answers):
        row = rows_by_id[rid]
        transported_success = alignment.get("transported_success", {}).get(rid, {})
        local_distortion = float(alignment.get("target_local_distortion", {}).get(rid, 0.0))
        option_count = infer_option_count(row)
        candidates = sorted({answer for answer in row_answers if answer and answer not in {"<EMPTY>", "<empty>"}})
        if not candidates:
            spectra_choices[rid] = fallback_model
            answer_choices[rid] = "<empty>"
            continue
        support_stats = lineage_support_stats(row_answers, models, lineages, source_accuracy, transported_success)
        candidate_scores: dict[str, float] = {}
        candidate_records: dict[str, Any] = {}
        for answer in candidates:
            real_support, eff_count = answer_support(row_answers, lineages, answer)
            null_scores = []
            for _ in range(max(1, args.null_worlds)):
                other_answers = target_answers[rng.randrange(len(target_answers))]
                null_support, _ = answer_support(other_answers, lineages, answer)
                null_scores.append(null_support)
            null_mean = sum(null_scores) / max(1, len(null_scores))
            null_p = (1.0 + sum(1 for value in null_scores if value >= real_support)) / (1.0 + max(1, args.null_worlds))
            gap = real_support - null_mean
            stats = support_stats.get(answer, {})
            bias_penalty = option_bias_penalty(answer, answer_freq, option_count, len(candidates))
            source_prior = 0.55 * float(stats.get("transported_support", 0.0)) + 0.45 * float(stats.get("source_support", 0.0))
            source_prior = max(0.02, min(0.98, source_prior))
            q_orbit = max(0.0, min(1.0, 0.55 * real_support + 0.45 * (1.0 - null_p)))
            tau = sigmoid(
                0.35
                - 1.05 * local_distortion
                + 1.60 * max(0.0, gap)
                + 0.70 * q_orbit
                + 0.35 * float(stats.get("lineage_share", 0.0))
                - 0.60 * bias_penalty
            )
            leaf_bonus = 0.12 if leaf_answers.get(rid) == answer else 0.0
            signed_score = (
                1.20 * real_support
                + 0.75 * float(stats.get("model_share", 0.0))
                + 1.05 * gap
                + 0.42 * (1.0 - null_p)
                + args.spectra_source_prior_weight * tau * logit(source_prior)
                + leaf_bonus
                - 0.75 * bias_penalty
                - 0.10 * local_distortion
            )
            candidate_scores[answer] = signed_score
            candidate_records[answer] = {
                "real_lineage_support": real_support,
                "effective_lineage_count": eff_count,
                "model_share": float(stats.get("model_share", 0.0)),
                "source_support": float(stats.get("source_support", 0.0)),
                "transported_support": float(stats.get("transported_support", 0.0)),
                "local_distortion": local_distortion,
                "null_mean_support": null_mean,
                "null_pvalue": null_p,
                "grounded_orbit_gap_proxy": gap,
                "option_bias_penalty": bias_penalty,
                "q_orbit_proxy": q_orbit,
                "soft_ledger_temperature": tau,
                "source_prior": source_prior,
                "signed_score": signed_score,
            }
        posterior = softmax(candidate_scores)
        ranked = sorted(posterior.items(), key=lambda item: (-item[1], item[0]))
        best_answer = ranked[0][0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = ranked[0][1] - second_prob
        record = candidate_records[best_answer]
        eff_diversity_norm = math.log1p(float(record["effective_lineage_count"])) / math.log1p(float(lineage_total))
        item_phase_score = (
            0.30 * max(0.0, float(record["grounded_orbit_gap_proxy"]))
            + 0.24 * eff_diversity_norm
            + 0.22 * margin
            + 0.16 * (1.0 - float(record["null_pvalue"]))
            + 0.08 * math.exp(-local_distortion)
        )
        chosen_model = choose_supporting_model(
            best_answer,
            row_answers,
            models,
            source_accuracy,
            transported_success,
            leaf_reliability,
            fallback_model,
        )
        spectra_choices[rid] = chosen_model
        answer_choices[rid] = best_answer
        detail = {
            "id": rid,
            "chosen_model": chosen_model,
            "chosen_answer": best_answer,
            "posterior": ranked[:8],
            "posterior_margin": margin,
            "item_phase_score": item_phase_score,
            "effective_diversity_norm": eff_diversity_norm,
            **record,
        }
        item_rows.append(detail)
        if len(item_debug) < 80:
            item_debug[rid] = {**detail, "candidate_records": candidate_records}

    return {
        "choices": spectra_choices,
        "answer_choices": answer_choices,
        "item_rows": item_rows,
        "sample_debug": item_debug,
        "target_answer_symbol_frequency": answer_freq,
    }


def route_phase(item_rows: list[dict[str, Any]], alignment: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not item_rows:
        return {"phase": "Red", "routability_score": 0.0}
    n = len(item_rows)
    grounded_gap = sum(max(0.0, float(row["grounded_orbit_gap_proxy"])) for row in item_rows) / n
    effective_diversity = sum(float(row["effective_diversity_norm"]) for row in item_rows) / n
    alignment_score = math.exp(-float(alignment.get("alignment_distance", 0.0)))
    concentration = sum(float(row["posterior_margin"]) for row in item_rows) / n
    null_survival = sum(float(row["null_pvalue"]) for row in item_rows) / n
    low_null_rate = sum(1 for row in item_rows if float(row["null_pvalue"]) < args.spectra_null_alpha) / n
    certificate_rate = sum(
        1
        for row in item_rows
        if float(row["posterior_margin"]) >= args.spectra_item_margin
        and float(row["effective_diversity_norm"]) >= args.spectra_min_eff_lineage
        and float(row["null_pvalue"]) < args.spectra_null_alpha
    ) / n
    strong_certificate_rate = sum(
        1
        for row in item_rows
        if float(row["posterior_margin"]) >= args.spectra_strong_margin
        and float(row["effective_diversity_norm"]) >= args.spectra_min_eff_lineage
        and float(row["null_pvalue"]) < args.spectra_null_alpha
        and float(row["grounded_orbit_gap_proxy"]) > -0.02
    ) / n
    score = (
        0.25 * grounded_gap
        + 0.23 * effective_diversity
        + 0.17 * alignment_score
        + 0.25 * concentration
        + 0.15 * low_null_rate
        - 0.22 * null_survival
    )
    if score >= args.spectra_green_threshold and certificate_rate >= 0.18:
        phase = "Green"
    elif score <= args.spectra_red_threshold or certificate_rate <= 0.04:
        phase = "Red"
    else:
        phase = "Amber"
    return {
        "phase": phase,
        "routability_score": score,
        "G_grounded_gap": grounded_gap,
        "D_effective_diversity": effective_diversity,
        "A_alignment": alignment_score,
        "C_concentration": concentration,
        "N_null_survival": null_survival,
        "low_null_rate": low_null_rate,
        "safe_certificate_rate": certificate_rate,
        "strong_certificate_rate": strong_certificate_rate,
        "thresholds": {
            "green": args.spectra_green_threshold,
            "red": args.spectra_red_threshold,
            "null_alpha": args.spectra_null_alpha,
            "item_margin": args.spectra_item_margin,
            "strong_margin": args.spectra_strong_margin,
            "min_eff_lineage": args.spectra_min_eff_lineage,
        },
    }


def phase_safe_choices(
    spectra_choices: dict[str, str],
    target_ids: list[str],
    item_rows: list[dict[str, Any]],
    fallback_model: str,
    phase: str,
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    by_id = {str(row["id"]): row for row in item_rows}
    final: dict[str, str] = {}
    reasons: Counter[str] = Counter()
    for rid in target_ids:
        row = by_id.get(rid)
        if row is None:
            final[rid] = fallback_model
            reasons["no_candidate"] += 1
            continue
        pvalue = float(row["null_pvalue"])
        margin = float(row["posterior_margin"])
        eff = float(row["effective_diversity_norm"])
        gap = float(row["grounded_orbit_gap_proxy"])
        if phase == "Red":
            allow = (
                pvalue < args.spectra_null_alpha * 0.50
                and margin >= args.spectra_strong_margin
                and eff >= args.spectra_min_eff_lineage + 0.12
                and gap > 0.02
            )
            reason = "red_strong_certificate" if allow else "red_fallback"
        elif phase == "Amber":
            allow = (
                pvalue < args.spectra_null_alpha
                and margin >= args.spectra_item_margin
                and eff >= args.spectra_min_eff_lineage
                and gap > -0.03
            )
            reason = "amber_certificate" if allow else "amber_fallback"
        else:
            allow = (
                pvalue < min(0.45, args.spectra_null_alpha + 0.12)
                and margin >= args.spectra_item_margin * 0.65
                and eff >= max(0.20, args.spectra_min_eff_lineage - 0.08)
            )
            reason = "green_certificate" if allow else "green_fallback"
        final[rid] = spectra_choices[rid] if allow else fallback_model
        reasons[reason] += 1
    return final, {"phase": phase, "fallback_model": fallback_model, "reason_counts": dict(reasons)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_case_report(
    path: Path,
    case: CaseSpec,
    rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    target_matrix: dict[str, dict[str, bool]],
    choice_maps: dict[str, dict[str, str]],
    audit: dict[str, Any],
) -> None:
    target_ids = [row_id(row) for row in target_rows]
    columns = sorted({group_value(row, case.group_key) for row in target_rows}) + ["Average"]
    name_width = 42
    col_width = 14
    lines = [
        "=" * 114,
        f"SPECTRA-CoE Lite cache-only validation: {case.title}",
        "=" * 114,
        "| Calibration: source labels + target unlabeled expert outputs only; target labels are final scoring only.",
        f"| Phase prediction before scoring: {audit['phase']} (R={audit['routability_score']:.4f})",
        f"| Best single on target: {audit['best_target_model']} ({fmt_pct(audit['best_target_accuracy'])})",
        f"| Source base: {audit['base_model']} ({fmt_pct(audit['base_source_accuracy'])} on source)",
        f"| SPECTRA fallback model: {audit['fallback_model']}",
        "",
        table_row(name_width, col_width, "Method", columns),
        table_row(name_width, col_width, "-" * 20, ["-" * 10 for _ in columns]),
    ]
    for row in sorted(rows, key=lambda item: -float(item["target_accuracy"])):
        method = str(row["method"])
        summary = summarize_boolean_choices(target_rows, target_ids, choice_maps[method], target_matrix, case.group_key)
        values = [
            fmt_pct(summary["by_group"].get(group, {}).get("accuracy")) if group in summary["by_group"] else "N/A"
            for group in columns[:-1]
        ]
        values.append(fmt_pct(summary["accuracy"]))
        lines.append(table_row(name_width, col_width, method, values))
    lines.extend(
        [
            "",
            "Implemented SPECTRA-Lite components:",
            "- zero-call ORBIT proxy: lineage-capped answer support and target answer-symbol bias audit;",
            "- SOFT-LEDGER proxy: source/transported reliability is locally tempered by alignment distortion and orbit gap;",
            "- dynamic pseudo-base: answer clusters are selected per item before choosing a supporting expert;",
            "- null survival: question-output decoupling over cached outputs estimates whether support survives a semantic break;",
            "- ROUTE-PHASE: Green/Amber/Red is predicted from unlabeled G/D/A/C/N metrics before scoring;",
            "- refusal: Red/Amber items fall back unless item certificates pass frozen thresholds.",
            "",
            "This run does not execute new option-permutation or visual-mask probes. It validates the cache-only SPECTRA-Lite lower-cost path.",
        ]
    )
    write_text(path, lines)


def run_case(case: CaseSpec, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_full_raw = load_full_predictions(case.source_kind, case.source_root)
    target_full_raw = load_full_predictions(case.target_kind, case.target_root)
    source_ids = infer_ids(source_full_raw)
    target_ids = infer_ids(target_full_raw)
    source_bool_raw = bool_matrix(source_full_raw)
    target_bool_raw = bool_matrix(target_full_raw)
    source_complete = complete_models(source_bool_raw, source_ids)
    target_complete = complete_models(target_bool_raw, target_ids)
    if args.exclude_models is not None:
        excluded = set(args.exclude_models)
    elif is_vl_case(case):
        excluded = set(args.vl_exclude_models)
    else:
        excluded = set(args.text_exclude_models)
    models = sorted(set(source_complete).intersection(target_complete).difference(excluded))
    if not models:
        raise RuntimeError(f"No common complete models for {case.case_id}; excluded={sorted(excluded)}")

    source_full = subset_full(source_full_raw, models, source_ids)
    target_full = subset_full(target_full_raw, models, target_ids)
    source_matrix = {model: source_complete[model] for model in models}
    target_matrix = {model: target_complete[model] for model in models}
    source_rows = first_complete_rows(source_full, source_ids)
    target_rows = first_complete_rows(target_full, target_ids)
    source_y = matrix_array(source_matrix, models, source_ids)
    base_model, base_source_acc = best_model_for_ids(source_matrix, source_ids)
    best_target_model, best_target_acc = best_model_for_ids(target_matrix, target_ids)
    oracle_acc = instance_oracle_accuracy(target_matrix, target_ids)
    source_acc = source_accuracy_map(source_y, models)

    case_dir = args.output_dir / case.case_id
    alignment = atlas_alignment(source_full, target_full, models, source_ids, target_ids, source_y, args)
    builders = proposal_builders(args)
    method_weight_audit = source_lobo_method_weights(
        case,
        builders,
        source_rows,
        source_full,
        source_matrix,
        source_y,
        models,
        source_ids,
        base_model,
        args,
    )
    leaf = leaf_estimate(source_full, target_full, source_y, models, source_ids, target_ids, args)
    target_answers = model_answers(target_full, models, target_ids)
    posterior_fallback_model, posterior_fallback_meta = leaf_posterior_fallback_model(
        leaf,
        target_answers,
        target_ids,
        models,
        source_acc,
        alignment,
        base_model,
    )
    if args.spectra_fallback == "source":
        fallback_model = base_model
    elif args.spectra_fallback == "leaf":
        fallback_model = str(leaf.get("base_model", base_model))
    else:
        fallback_model = posterior_fallback_model

    majority_choices, majority_meta = static_target_majority_choices(
        target_answers,
        target_ids,
        models,
        source_acc,
        {model: float(value) for model, value in leaf.get("target_reliability", {}).items()},
        fallback_model,
    )
    spectra = spectra_item_posteriors(
        case,
        target_rows,
        target_full,
        target_answers,
        source_acc,
        leaf,
        alignment,
        models,
        target_ids,
        fallback_model,
        args,
    )
    phase_audit = route_phase(spectra["item_rows"], alignment, args)
    phase_choices, phase_meta = phase_safe_choices(
        spectra["choices"],
        target_ids,
        spectra["item_rows"],
        fallback_model,
        str(phase_audit["phase"]),
        args,
    )
    base_map = {rid: base_model for rid in target_ids}
    fallback_map = {rid: fallback_model for rid in target_ids}
    posterior_fallback_map = {rid: posterior_fallback_model for rid in target_ids}

    proposal_maps: dict[str, dict[str, str]] = {}
    proposal_meta: dict[str, Any] = {}
    for name, builder in builders.items():
        try:
            choices, meta = builder(source_rows, target_rows, source_full, target_full, source_y, models, source_ids, target_ids, args)
        except Exception as exc:
            proposal_meta[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        proposal_maps[name] = choices
        proposal_meta[name] = meta

    choice_maps: dict[str, dict[str, str]] = {
        "source_global_best": base_map,
        "spectra_predeclared_fallback": fallback_map,
        "spectra_posterior_single_fallback": posterior_fallback_map,
        "spectra_static_target_majority": majority_choices,
        "spectra_leaf_pseudo_base": leaf["posterior_choices"],
        "spectra_dynamic_pseudo_base": spectra["choices"],
        "spectra_route_phase_safe": phase_choices,
    }
    for name, choices in proposal_maps.items():
        choice_maps[f"proposal_{name}"] = choices

    sim_ci = simultaneous_bootstrap_ci(choice_maps, best_target_model, target_matrix, target_ids, args.bootstrap_iters, args.seed)
    metadata: dict[str, dict[str, Any]] = {
        "source_global_best": {"source_accuracy": base_source_acc},
        "spectra_predeclared_fallback": {
            "fallback_policy": args.spectra_fallback,
            "fallback_model": fallback_model,
            "leaf_base_model": leaf.get("base_model"),
            "posterior_fallback": posterior_fallback_meta,
        },
        "spectra_posterior_single_fallback": posterior_fallback_meta,
        "spectra_static_target_majority": majority_meta,
        "spectra_leaf_pseudo_base": {
            "module": "LEAF posterior pseudo-base",
            "base_model": leaf.get("base_model"),
            "base_lcb": leaf.get("base_lcb"),
        },
        "spectra_dynamic_pseudo_base": {
            "module": "SPECTRA-Lite dynamic pseudo-base",
            "phase": phase_audit,
            "target_answer_symbol_frequency": spectra["target_answer_symbol_frequency"],
            "note": "Uses source labels plus target-unlabeled cached expert outputs; no target labels.",
        },
        "spectra_route_phase_safe": {
            "module": "SPECTRA-Lite ROUTE-PHASE safe router",
            "phase_meta": phase_meta,
            "phase": phase_audit,
            "method_weights": method_weight_audit["weights"],
        },
    }
    for name, meta in proposal_meta.items():
        metadata[f"proposal_{name}"] = meta if isinstance(meta, dict) else {"meta": meta}

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
        row.update(sim_ci.get(method, {}))
        row.update(
            {
                "case_id": case.case_id,
                "source_kind": case.source_kind,
                "target_kind": case.target_kind,
                "source_root": str(case.source_root),
                "target_root": str(case.target_root),
                "source_samples": len(source_ids),
                "target_samples": len(target_ids),
                "base_model": base_model,
                "base_source_accuracy": base_source_acc,
                "fallback_model": fallback_model,
                "alignment_distance": alignment["alignment_distance"],
                "spectra_phase": phase_audit["phase"],
                "spectra_routability_score": phase_audit["routability_score"],
                "spectra_safe_certificate_rate": phase_audit["safe_certificate_rate"],
            }
        )
        rows.append(row)

    phase_rows = [{key: value for key, value in phase_audit.items() if key != "thresholds"}]
    audit = {
        "case_id": case.case_id,
        "title": case.title,
        "models": models,
        "excluded_models": sorted(excluded),
        "source_samples": len(source_ids),
        "target_samples": len(target_ids),
        "base_model": base_model,
        "base_source_accuracy": base_source_acc,
        "fallback_model": fallback_model,
        "best_target_model": best_target_model,
        "best_target_accuracy": best_target_acc,
        "instance_oracle": oracle_acc,
        "alignment_distance": alignment["alignment_distance"],
        "alignment_p90": alignment["alignment_p90"],
        "phase": phase_audit["phase"],
        "routability_score": phase_audit["routability_score"],
        "phase_audit": phase_audit,
        "method_weight_audit": method_weight_audit,
        "proposal_meta": proposal_meta,
        "note": "SPECTRA-Lite routing uses source labels and target unlabeled expert outputs only. Target labels are final scoring only.",
        "full_probe_status": "not_run_cache_only_validation",
    }

    write_csv(case_dir / "spectra_results.csv", rows)
    write_json(case_dir / "spectra_results.json", rows)
    write_json(case_dir / "choices_by_method.json", choice_maps)
    write_json(case_dir / "route_phase_audit.json", audit)
    write_csv(case_dir / "route_phase_summary.csv", phase_rows)
    write_jsonl(case_dir / "signed_intervention_proxy.jsonl", spectra["item_rows"])
    render_case_report(case_dir / f"Bench_Harness_Result_spectra_{case.case_id}.txt", case, rows, target_rows, target_matrix, choice_maps, audit)
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    lines = [
        "# SPECTRA-CoE Lite Source-Transfer Validation",
        "",
        "Calibration uses source labels plus target-unlabeled cached expert outputs only. Target labels are used once for final scoring.",
        "",
        "| Case | Method | Phase | R | Accuracy | Best Single | Gain | Paired CI | Selection-Aware Simul CI | Cert Rate | Models |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["case_id"]), -float(item["target_accuracy"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_id']}`",
                    f"`{row['method']}`",
                    str(row.get("spectra_phase", "")),
                    f"{float(row.get('spectra_routability_score', 0.0)):.3f}",
                    fmt_pct(row["target_accuracy"]),
                    f"{fmt_pct(row['best_single_target'])} ({row['best_single_model_target']})",
                    f"{float(row['gain_vs_best_single_target']) * 100:+.2f}%",
                    f"[{float(row.get('paired_ci_low', 0.0)) * 100:+.2f}%, {float(row.get('paired_ci_high', 0.0)) * 100:+.2f}%]",
                    f"[{float(row.get('simul_ci_low', 0.0)) * 100:+.2f}%, {float(row.get('simul_ci_high', 0.0)) * 100:+.2f}%]",
                    f"{float(row.get('spectra_safe_certificate_rate', 0.0)):.3f}",
                    str(row.get("models_used", "")),
                ]
            )
            + " |"
        )
    write_text(output_dir / "summary.md", lines)

    best_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best_rows.get(str(row["case_id"]))
        if current is None or float(row["target_accuracy"]) > float(current["target_accuracy"]):
            best_rows[str(row["case_id"])] = row
    best_lines = [
        "# SPECTRA-CoE Lite Best-by-Target Summary",
        "",
        "| Case | Best Method | Phase | R | Accuracy | Best Single | Gain | Paired CI | Selection-Aware Simul CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_id, row in sorted(best_rows.items()):
        best_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case_id}`",
                    f"`{row['method']}`",
                    str(row.get("spectra_phase", "")),
                    f"{float(row.get('spectra_routability_score', 0.0)):.3f}",
                    fmt_pct(row["target_accuracy"]),
                    f"{fmt_pct(row['best_single_target'])} ({row['best_single_model_target']})",
                    f"{float(row['gain_vs_best_single_target']) * 100:+.2f}%",
                    f"[{float(row.get('paired_ci_low', 0.0)) * 100:+.2f}%, {float(row.get('paired_ci_high', 0.0)) * 100:+.2f}%]",
                    f"[{float(row.get('simul_ci_low', 0.0)) * 100:+.2f}%, {float(row.get('simul_ci_high', 0.0)) * 100:+.2f}%]",
                ]
            )
            + " |"
        )
    write_text(output_dir / "best_by_target.md", best_lines)

    phase_lines = [
        "# SPECTRA-CoE Lite ROUTE-PHASE Audit",
        "",
        "| Case | Phase | R | G | D | A | C | N | Cert Rate | Low Null |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: str(item["case_id"])):
        case_id = str(row["case_id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        meta = json.loads(str(row.get("metadata", "{}"))) if row.get("method") == "spectra_dynamic_pseudo_base" else {}
        phase = meta.get("phase", {})
        if not phase:
            for candidate in rows:
                if candidate["case_id"] == case_id and candidate["method"] == "spectra_dynamic_pseudo_base":
                    phase = json.loads(str(candidate.get("metadata", "{}"))).get("phase", {})
                    break
        phase_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case_id}`",
                    str(row.get("spectra_phase", "")),
                    f"{float(row.get('spectra_routability_score', 0.0)):.3f}",
                    f"{float(phase.get('G_grounded_gap', 0.0)):.3f}",
                    f"{float(phase.get('D_effective_diversity', 0.0)):.3f}",
                    f"{float(phase.get('A_alignment', 0.0)):.3f}",
                    f"{float(phase.get('C_concentration', 0.0)):.3f}",
                    f"{float(phase.get('N_null_survival', 0.0)):.3f}",
                    f"{float(phase.get('safe_certificate_rate', row.get('spectra_safe_certificate_rate', 0.0))):.3f}",
                    f"{float(phase.get('low_null_rate', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    write_text(output_dir / "route_phase_summary.md", phase_lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for case in select_cases(args.cases):
        all_rows.extend(run_case(case, args))
    write_summary(args.output_dir, all_rows)


if __name__ == "__main__":
    main()
