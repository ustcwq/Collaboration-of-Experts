from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.improve2_capability_routing_experiments import (
    CASES,
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
from bench_coe.improve4_failure_modeling_experiments import (
    answer_partition_features,
    output_bundle,
    source_dataset_splits,
    subset_full,
)
from bench_coe.improve5_failure_ecology_experiments import (
    dare_reliability_choices,
    ecc_code_decoder_choices,
    ecr_correction_graph_choices,
    fate_failure_ecology_choices,
)
from bench_coe.improve6_adaptive_failure_ecology_experiments import (
    repair_chain_choices,
)
from bench_coe.materialize_innovation_strategies import (
    fmt_pct,
    summarize_boolean_choices,
    table_row,
    write_text,
)
from bench_coe.offline_router_innovation_experiments import (
    best_model_for_ids,
    instance_oracle_accuracy,
    row_id,
    write_csv,
    write_json,
)
from bench_coe.orbit_coe_experiments import bres_choices, leaf_estimate, lineage_key
from bench_coe.shared_eval_utils import paired_bootstrap_delta


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


DEFAULT_CASES = (
    "mmlu_val_to_mmlu_test",
    "mmlu_val_to_gaokao2010_2022",
    "mmlu_val_to_bbh",
    "mmlu_val_to_gpqa",
    "mmlu_val_to_mmstar",
    "mmmu_pro_val_to_cmmmu",
    "mmmu_pro_val_to_mathvista",
    "mmmu_pro_val_to_mmmu_pro_test",
)
DEFAULT_TEXT_EXCLUDES = ("Qwen3.5-9B", "DeepSeek-R1-0528-Qwen3-8B", "Qwen3-8B")
DEFAULT_VL_EXCLUDES = ("Qwen3.5-9B", "DeepSeek-R1-0528-Qwen3-8B", "Qwen3-8B", "Qwen3-VL-4B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ATLAS-CoE cache-only validation. Implements LEDGER, A-FGW-lite, TRIDENT, "
            "NULLSHIELD, TAPESTRY-lite, and selection-aware simultaneous bootstrap using "
            "source labels plus target unlabeled expert outputs only."
        )
    )
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated case ids, or all.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench_coe/atlas_coe_source_transfer_validation"))
    parser.add_argument("--exclude-models", nargs="*", default=None)
    parser.add_argument("--text-exclude-models", nargs="*", default=list(DEFAULT_TEXT_EXCLUDES))
    parser.add_argument("--vl-exclude-models", nargs="*", default=list(DEFAULT_VL_EXCLUDES))
    parser.add_argument("--knn-k", type=int, default=32)
    parser.add_argument("--state-clusters", type=int, default=12)
    parser.add_argument("--leaf-iters", type=int, default=8)
    parser.add_argument("--leaf-source-weight", type=float, default=0.35)
    parser.add_argument("--lineage-cap", type=float, default=1.25)
    parser.add_argument("--posterior-draws", type=int, default=4000)
    parser.add_argument("--ledger-alpha", type=float, default=0.10)
    parser.add_argument("--ledger-min-prob", type=float, default=0.62)
    parser.add_argument("--ledger-strict-prob", type=float, default=0.80)
    parser.add_argument("--max-item-influence", type=float, default=0.18)
    parser.add_argument("--alignment-weight", type=float, default=0.055)
    parser.add_argument("--stability-weight", type=float, default=0.045)
    parser.add_argument("--null-worlds", type=int, default=99)
    parser.add_argument("--null-alpha", type=float, default=0.20)
    parser.add_argument("--trident-threshold", type=float, default=-0.015)
    parser.add_argument("--tapestry-eta", type=float, default=8.0)
    parser.add_argument("--tapestry-base-prior", type=float, default=0.10)
    parser.add_argument("--tapestry-margin", type=float, default=0.12)
    parser.add_argument("--lobo-max-splits", type=int, default=6)
    parser.add_argument("--lobo-max-heldout", type=int, default=500)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def select_cases(value: str) -> list[CaseSpec]:
    by_id = {case.case_id: case for case in CASES}
    if value == "all":
        selected = DEFAULT_CASES
    else:
        selected = tuple(item.strip() for item in value.split(",") if item.strip())
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"Unknown case ids: {missing}; valid ids: {sorted(by_id)}")
    return [by_id[item] for item in selected]


def is_vl_case(case: CaseSpec) -> bool:
    roots = f"{case.source_root} {case.target_root}".lower()
    return "multimodal_babyvision_models" in roots or case.case_id.startswith(("mmmu", "cmmmu"))


def normalize_answer(value: Any) -> str:
    if value is None:
        return "<empty>"
    if isinstance(value, list):
        value = "".join(str(item) for item in value)
    text = str(value).strip().upper()
    if not text:
        return "<empty>"
    return " ".join(text.split())


def safe_group(row: dict[str, Any], preferred: str) -> str:
    for key in (preferred, "category", "subject", "task", "domain", "subdomain", "source_dataset", "benchmark"):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)
    return "unknown"


def proposal_builders(args: argparse.Namespace) -> dict[str, MethodBuilder]:
    return {
        "leaf_posterior_vote": lambda sr, tr, sf, tf, sy, models, sids, tids, a: (
            leaf_estimate(sf, tf, sy, models, sids, tids, a)["posterior_choices"],
            {"module": "LEAF", "source": "ORBIT-lite"},
        ),
        "bres_residual_evidence": lambda sr, tr, sf, tf, sy, models, sids, tids, a: _bres_wrapper(
            sf, tf, sy, models, sids, tids, a
        ),
        "ecr_correction_graph": lambda sr, tr, sf, tf, sy, models, sids, tids, a: ecr_correction_graph_choices(
            sf, tf, sy, models, sids, tids, a.knn_k
        ),
        "fate_failure_ecology": lambda sr, tr, sf, tf, sy, models, sids, tids, a: fate_failure_ecology_choices(
            sf, tf, sy, models, sids, tids, a.state_clusters, a.seed
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
    }


def _bres_wrapper(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    leaf = leaf_estimate(source_full, target_full, source_y, models, source_ids, target_ids, args)
    choices, meta = bres_choices(source_full, target_full, source_y, models, source_ids, target_ids, leaf, args)
    meta = dict(meta)
    meta["module"] = "BRES"
    return choices, meta


def source_splits(rows: list[dict[str, Any]], ids: list[str], case: CaseSpec) -> dict[str, list[str]]:
    splits = source_dataset_splits(rows, ids)
    if len(splits) >= 2:
        return dict(splits)
    rows_by_id = {row_id(row): row for row in rows}
    for key in (case.group_key, "category", "subject", "task", "domain", "subdomain"):
        grouped: dict[str, list[str]] = defaultdict(list)
        for rid in ids:
            grouped[safe_group(rows_by_id[rid], key)].append(rid)
        if len(grouped) >= 2:
            return dict(grouped)
    chunks: dict[str, list[str]] = defaultdict(list)
    for idx, rid in enumerate(ids):
        chunks[f"chunk_{idx % 5}"].append(rid)
    return dict(chunks)


def build_edge_ledger(
    source_y: Any,
    models: list[str],
    source_rows: list[dict[str, Any]],
    source_ids: list[str],
    group_key: str,
    draws: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows_by_id = {row_id(row): row for row in source_rows}
    groups = [safe_group(rows_by_id[rid], group_key) for rid in source_ids]
    n = int(source_y.shape[0])
    ledger_rows: list[dict[str, Any]] = []
    ledger_map: dict[tuple[str, str], dict[str, Any]] = {}
    for base_idx, base in enumerate(models):
        for cand_idx, cand in enumerate(models):
            if cand_idx == base_idx:
                continue
            repair = (source_y[:, cand_idx] > 0.5) & (source_y[:, base_idx] <= 0.5)
            harm = (source_y[:, cand_idx] <= 0.5) & (source_y[:, base_idx] > 0.5)
            repair_count = int(repair.sum())
            harm_count = int(harm.sum())
            signed = repair.astype(float) - harm.astype(float)
            mean_gain = float(signed.mean()) if n else 0.0
            if n > 1:
                total = float(signed.sum())
                leave_one = np.abs(mean_gain - (total - signed) / float(n - 1))
                influence = float(leave_one.max())
                sign_flip = bool(np.any(np.sign(mean_gain) != np.sign((total - signed) / float(n - 1))))
            else:
                influence = abs(mean_gain)
                sign_flip = False
            repair_draws = rng.beta(repair_count + 1.0, n - repair_count + 1.0, size=draws)
            harm_draws = rng.beta(harm_count + 1.0, n - harm_count + 1.0, size=draws)
            gain_draws = repair_draws - harm_draws
            prob_positive = float((gain_draws > 0.0).mean())
            lcb_gain = float(np.quantile(gain_draws, 0.05))
            supported_groups = sorted({groups[idx] for idx, value in enumerate(repair | harm) if bool(value)})
            repair_groups = sorted({groups[idx] for idx, value in enumerate(repair) if bool(value)})
            harm_groups = sorted({groups[idx] for idx, value in enumerate(harm) if bool(value)})
            row = {
                "base": base,
                "candidate": cand,
                "source_samples": n,
                "repair_count": repair_count,
                "harm_count": harm_count,
                "disagreement_count": repair_count + harm_count,
                "mean_gain": mean_gain,
                "lcb_gain": lcb_gain,
                "prob_positive": prob_positive,
                "item_influence": influence,
                "sign_flip_leave_one_item": sign_flip,
                "supported_group_count": len(supported_groups),
                "repair_group_count": len(repair_groups),
                "harm_group_count": len(harm_groups),
                "supported_groups": ";".join(supported_groups[:40]),
            }
            ledger_rows.append(row)
            ledger_map[(cand, base)] = row
    return ledger_rows, ledger_map


def behavior_features(full: dict[str, dict[str, dict[str, Any]]], models: list[str], ids: list[str]) -> tuple[Any, list[list[str]]]:
    import numpy as np

    answers, _, stats = output_bundle(full, models, ids)
    base = answer_partition_features(answers, stats)
    lineages = [lineage_key(model) for model in models]
    lineage_names = sorted(set(lineages))
    extra = []
    for row_answers in answers:
        counts = Counter(row_answers)
        lineage_support = []
        for name in lineage_names:
            idxs = [idx for idx, lineage in enumerate(lineages) if lineage == name]
            if not idxs:
                lineage_support.append(0.0)
            else:
                lineage_counts = Counter(row_answers[idx] for idx in idxs)
                lineage_support.append(max(lineage_counts.values()) / len(idxs))
        total = max(1, len(row_answers))
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(total + 1e-12)
        extra.append([len(counts) / total, entropy, max(probs) if probs else 0.0] + lineage_support)
    return np.hstack([base, np.asarray(extra, dtype=float)]), answers


def standardize_pair(x_source: Any, x_target: Any) -> tuple[Any, Any]:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    xs = scaler.fit_transform(np.asarray(x_source, dtype=float))
    xt = scaler.transform(np.asarray(x_target, dtype=float))
    return xs, xt


def pairwise_dist(x: Any, y: Any | None = None) -> Any:
    import numpy as np

    a = np.asarray(x, dtype=float)
    b = a if y is None else np.asarray(y, dtype=float)
    aa = (a * a).sum(axis=1)[:, None]
    bb = (b * b).sum(axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0))


def atlas_alignment(
    source_full: dict[str, dict[str, dict[str, Any]]],
    target_full: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    source_ids: list[str],
    target_ids: list[str],
    source_y: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    x_source_raw, _ = behavior_features(source_full, models, source_ids)
    x_target_raw, _ = behavior_features(target_full, models, target_ids)
    x_source, x_target = standardize_pair(x_source_raw, x_target_raw)
    k_source = min(max(2, args.state_clusters), max(2, len(source_ids) // 4), len(source_ids))
    k_target = min(max(2, args.state_clusters), max(2, len(target_ids) // 80), len(target_ids))
    source_clusterer = MiniBatchKMeans(n_clusters=k_source, random_state=args.seed, n_init=10, batch_size=512)
    target_clusterer = MiniBatchKMeans(n_clusters=k_target, random_state=args.seed + 17, n_init=10, batch_size=1024)
    source_labels = source_clusterer.fit_predict(x_source)
    target_labels = target_clusterer.fit_predict(x_target)
    source_centers = np.vstack([x_source[source_labels == cid].mean(axis=0) for cid in range(k_source)])
    target_centers = np.vstack([x_target[target_labels == cid].mean(axis=0) for cid in range(k_target)])
    ds = pairwise_dist(source_centers)
    dt = pairwise_dist(target_centers)
    cross = pairwise_dist(source_centers, target_centers)
    temp = max(1e-6, float(np.median(cross) or 1.0))
    weights = np.exp(-cross / temp)
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    source_rel = ds.mean(axis=1)
    target_rel = dt.mean(axis=1)
    rel_scale = float(np.std(np.concatenate([source_rel, target_rel])) or 1.0)
    state_distortion = []
    state_anchor_cost = []
    transported_source_state = []
    for tidx in range(k_target):
        w = weights[:, tidx]
        anchor = float((w * cross[:, tidx]).sum() / temp)
        rel = float((w * np.abs(source_rel - target_rel[tidx])).sum() / rel_scale)
        state_anchor_cost.append(anchor)
        state_distortion.append(0.55 * anchor + 0.45 * rel)
        transported_source_state.append(int(np.argmax(w)))
    local = np.asarray([state_distortion[int(label)] for label in target_labels], dtype=float)
    if local.size:
        local = local / float(np.percentile(local, 90) or local.max() or 1.0)
        local = np.clip(local, 0.0, 2.0)
    source_state_success = np.zeros((k_source, len(models)), dtype=float)
    for cid in range(k_source):
        mask = source_labels == cid
        source_state_success[cid] = source_y[mask].mean(axis=0) if mask.any() else source_y.mean(axis=0)
    target_transported_success = np.vstack(
        [(weights[:, int(label)] @ source_state_success) for label in target_labels]
    )
    return {
        "source_state_count": int(k_source),
        "target_state_count": int(k_target),
        "alignment_distance": float(local.mean()) if local.size else 0.0,
        "alignment_p90": float(np.percentile(local, 90)) if local.size else 0.0,
        "target_local_distortion": {rid: float(local[idx]) for idx, rid in enumerate(target_ids)},
        "target_state": {rid: int(target_labels[idx]) for idx, rid in enumerate(target_ids)},
        "state_distortion": {str(idx): float(value) for idx, value in enumerate(state_distortion)},
        "state_anchor_cost": {str(idx): float(value) for idx, value in enumerate(state_anchor_cost)},
        "transported_source_state": {str(idx): int(value) for idx, value in enumerate(transported_source_state)},
        "transported_success": {
            rid: {model: float(target_transported_success[idx, midx]) for midx, model in enumerate(models)}
            for idx, rid in enumerate(target_ids)
        },
    }


def answer_supports(
    target_full: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    target_ids: list[str],
) -> tuple[dict[tuple[str, str], float], dict[str, list[str]], dict[str, dict[str, float]]]:
    answers, _, _ = output_bundle(target_full, models, target_ids)
    lineages = [lineage_key(model) for model in models]
    lineage_total = max(1, len(set(lineages)))
    support: dict[tuple[str, str], float] = {}
    answers_by_id: dict[str, list[str]] = {}
    answer_shares: dict[str, dict[str, float]] = {}
    for rid, row_answers in zip(target_ids, answers):
        answers_by_id[rid] = [normalize_answer(ans) for ans in row_answers]
        answer_to_lineages: dict[str, set[str]] = defaultdict(set)
        answer_to_count: Counter[str] = Counter()
        for midx, answer in enumerate(answers_by_id[rid]):
            answer_to_lineages[answer].add(lineages[midx])
            answer_to_count[answer] += 1
        answer_shares[rid] = {answer: count / len(models) for answer, count in answer_to_count.items()}
        for midx, model in enumerate(models):
            answer = answers_by_id[rid][midx]
            support[(rid, model)] = len(answer_to_lineages[answer]) / lineage_total
    return support, answers_by_id, answer_shares


def nullshield_pvalues(
    candidate_choices: dict[str, str],
    target_full: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    target_ids: list[str],
    null_worlds: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    rng = random.Random(seed)
    support, answers_by_id, _ = answer_supports(target_full, models, target_ids)
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    pvalues: dict[str, float] = {}
    real_scores: dict[str, float] = {}
    null_ge_total = 0
    for rid in target_ids:
        cand = candidate_choices[rid]
        real = support.get((rid, cand), 0.0)
        real_scores[rid] = real
        cand_idx = model_to_idx[cand]
        ge = 0
        for _ in range(max(1, null_worlds)):
            other = target_ids[rng.randrange(len(target_ids))]
            other_answers = answers_by_id[other]
            cand_answer = other_answers[cand_idx]
            lineages = {lineage_key(models[idx]) for idx, ans in enumerate(other_answers) if ans == cand_answer}
            null_score = len(lineages) / max(1, len({lineage_key(model) for model in models}))
            ge += int(null_score >= real)
        null_ge_total += ge
        pvalues[rid] = (1.0 + ge) / (1.0 + max(1, null_worlds))
    return pvalues, {
        "mean_real_support": sum(real_scores.values()) / max(1, len(real_scores)),
        "mean_pvalue": sum(pvalues.values()) / max(1, len(pvalues)),
        "null_worlds": null_worlds,
        "null_ge_rate": null_ge_total / max(1, len(target_ids) * max(1, null_worlds)),
    }


def source_lobo_method_weights(
    case: CaseSpec,
    builders: dict[str, MethodBuilder],
    source_rows: list[dict[str, Any]],
    source_full: dict[str, dict[str, dict[str, Any]]],
    source_matrix: dict[str, dict[str, bool]],
    source_y: Any,
    models: list[str],
    source_ids: list[str],
    base_model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np

    splits = source_splits(source_rows, source_ids, case)
    rows_by_id = {row_id(row): row for row in source_rows}
    source_index = {rid: idx for idx, rid in enumerate(source_ids)}
    selected = sorted(splits.items(), key=lambda item: -len(item[1]))[: max(1, args.lobo_max_splits)]
    rng = random.Random(args.seed)
    method_deltas: dict[str, list[float]] = {name: [] for name in builders}
    method_errors: dict[str, list[str]] = defaultdict(list)
    for split_name, heldout_full in selected:
        heldout_ids = list(heldout_full)
        if args.lobo_max_heldout and len(heldout_ids) > args.lobo_max_heldout:
            heldout_ids = sorted(rng.sample(heldout_ids, args.lobo_max_heldout))
        train_ids = [rid for name, ids in splits.items() if name != split_name for rid in ids]
        if not train_ids or not heldout_ids:
            continue
        train_rows = [rows_by_id[rid] for rid in train_ids]
        heldout_rows = [rows_by_id[rid] for rid in heldout_ids]
        train_full = subset_full(source_full, models, train_ids)
        heldout_source_full = subset_full(source_full, models, heldout_ids)
        train_y = source_y[[source_index[rid] for rid in train_ids]]
        base_acc = sum(1 for rid in heldout_ids if source_matrix[base_model].get(rid, False)) / len(heldout_ids)
        for name, builder in builders.items():
            try:
                choices, _ = builder(train_rows, heldout_rows, train_full, heldout_source_full, train_y, models, train_ids, heldout_ids, args)
            except Exception as exc:  # keep ATLAS audit robust across heterogeneous caches
                method_errors[name].append(f"{split_name}: {type(exc).__name__}: {exc}")
                continue
            acc = sum(1 for rid in heldout_ids if source_matrix[choices[rid]].get(rid, False)) / len(heldout_ids)
            method_deltas[name].append(acc - base_acc)
    mean_delta = {name: (sum(values) / len(values) if values else -0.05) for name, values in method_deltas.items()}
    raw = {name: math.exp(args.tapestry_eta * max(-0.10, min(0.10, value))) for name, value in mean_delta.items()}
    denom = sum(raw.values()) or 1.0
    weights = {name: value / denom for name, value in raw.items()}
    return {
        "weights": weights,
        "source_lobo_mean_delta": mean_delta,
        "source_lobo_split_count": len(selected),
        "source_lobo_errors": dict(method_errors),
    }


def trident_choices(
    proposal_maps: dict[str, dict[str, str]],
    base_model: str,
    ledger: dict[tuple[str, str], dict[str, Any]],
    alignment: dict[str, Any],
    target_full: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    target_ids: list[str],
    method_weights: dict[str, float],
    args: argparse.Namespace,
    strict: bool,
) -> tuple[dict[str, str], dict[str, Any], dict[str, float]]:
    support, _, _ = answer_supports(target_full, models, target_ids)
    candidate_pool: dict[str, str] = {}
    candidate_scores: dict[str, float] = {}
    candidate_method: dict[str, str] = {}
    cert_debug: dict[str, dict[str, Any]] = {}
    for rid in target_ids:
        best_model = base_model
        best_score = -999.0
        best_method = "base"
        local_distortion = float(alignment["target_local_distortion"].get(rid, 0.0))
        for method, choices in proposal_maps.items():
            cand = choices.get(rid, base_model)
            if cand == base_model:
                continue
            edge = ledger.get((cand, base_model))
            if not edge:
                continue
            sup = float(support.get((rid, cand), 0.0))
            transported = float(alignment["transported_success"].get(rid, {}).get(cand, 0.0))
            lobo_weight = float(method_weights.get(method, 0.0))
            lobo_component = 0.035 * math.log(max(1e-6, lobo_weight * len(method_weights)))
            if strict:
                source_component = float(edge["lcb_gain"])
                pass_source = (
                    float(edge["lcb_gain"]) > 0.0
                    and float(edge["prob_positive"]) >= args.ledger_strict_prob
                    and float(edge["item_influence"]) <= args.max_item_influence
                )
            else:
                source_component = 0.65 * float(edge["mean_gain"]) + 0.35 * float(edge["lcb_gain"])
                pass_source = (
                    float(edge["prob_positive"]) >= args.ledger_min_prob
                    and float(edge["mean_gain"]) > -0.02
                    and float(edge["item_influence"]) <= args.max_item_influence
                )
            cert = (
                source_component
                + 0.08 * (float(edge["prob_positive"]) - 0.5)
                + 0.035 * transported
                + 0.05 * sup
                + lobo_component
                - args.alignment_weight * local_distortion
                - args.stability_weight * (1.0 - sup)
            )
            if pass_source and cert > best_score:
                best_score = cert
                best_model = cand
                best_method = method
        candidate_pool[rid] = best_model
        candidate_scores[rid] = best_score if best_model != base_model else -999.0
        candidate_method[rid] = best_method
    pvalues, null_meta = nullshield_pvalues(candidate_pool, target_full, models, target_ids, args.null_worlds, args.seed + (23 if strict else 37))
    final: dict[str, str] = {}
    reason_counts: Counter[str] = Counter()
    for rid in target_ids:
        cand = candidate_pool[rid]
        if cand == base_model:
            final[rid] = base_model
            reason_counts["base_no_candidate"] += 1
            continue
        if candidate_scores[rid] <= args.trident_threshold:
            final[rid] = base_model
            reason_counts["cert_below_threshold"] += 1
        elif pvalues[rid] >= args.null_alpha:
            final[rid] = base_model
            reason_counts["nullshield_rejected"] += 1
        else:
            final[rid] = cand
            reason_counts["accepted_switch"] += 1
        if len(cert_debug) < 80:
            edge = ledger.get((cand, base_model), {})
            cert_debug[rid] = {
                "candidate": cand,
                "method": candidate_method[rid],
                "score": candidate_scores[rid],
                "pvalue": pvalues[rid],
                "local_distortion": alignment["target_local_distortion"].get(rid),
                "support": support.get((rid, cand), 0.0),
                "edge_mean_gain": edge.get("mean_gain"),
                "edge_lcb_gain": edge.get("lcb_gain"),
                "edge_prob_positive": edge.get("prob_positive"),
                "accepted": final[rid] == cand,
            }
    return final, {
        "strict": strict,
        "base_model": base_model,
        "reason_counts": dict(reason_counts),
        "nullshield": null_meta,
        "candidate_method_counts": dict(Counter(candidate_method.values())),
        "sample_certificates": cert_debug,
    }, pvalues


def tapestry_weighted_vote_choices(
    proposal_maps: dict[str, dict[str, str]],
    method_weights: dict[str, float],
    base_model: str,
    target_ids: list[str],
    base_prior: float,
    margin: float | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    choices: dict[str, str] = {}
    reason_counts: Counter[str] = Counter()
    sample_scores: dict[str, Any] = {}
    for rid in target_ids:
        scores: dict[str, float] = {base_model: float(base_prior)}
        for method, method_choices in proposal_maps.items():
            model = method_choices.get(rid, base_model)
            scores[model] = scores.get(model, 0.0) + float(method_weights.get(method, 0.0))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_model, top_score = ranked[0]
        base_score = scores.get(base_model, 0.0)
        if margin is not None and top_score - base_score <= margin:
            choices[rid] = base_model
            reason_counts["fallback_margin"] += 1
        else:
            choices[rid] = top_model
            reason_counts["weighted_vote"] += 1 if top_model != base_model else 0
            reason_counts["base_top"] += 1 if top_model == base_model else 0
        if len(sample_scores) < 80:
            sample_scores[rid] = {
                "ranked": ranked[:5],
                "base_score": base_score,
                "chosen": choices[rid],
            }
    return choices, {
        "module": "TAPESTRY-lite",
        "base_model": base_model,
        "base_prior": base_prior,
        "margin": margin,
        "method_weights": method_weights,
        "reason_counts": dict(reason_counts),
        "sample_scores": sample_scores,
        "note": "Source-only method weights combine target-unlabeled proposal choices; target labels are not used.",
    }


def simultaneous_bootstrap_ci(
    choice_maps: dict[str, dict[str, str]],
    baseline_model: str,
    target_matrix: dict[str, dict[str, bool]],
    target_ids: list[str],
    iters: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    methods = sorted(choice_maps)
    deltas: dict[str, list[float]] = {}
    obs: dict[str, float] = {}
    for method in methods:
        values = []
        for rid in target_ids:
            model = choice_maps[method][rid]
            values.append(float(target_matrix[model].get(rid, False)) - float(target_matrix[baseline_model].get(rid, False)))
        deltas[method] = values
        obs[method] = sum(values) / max(1, len(values))
    max_deviation = []
    n = len(target_ids)
    if n and iters > 0:
        for _ in range(iters):
            indices = [rng.randrange(n) for _j in range(n)]
            deviations = []
            for method in methods:
                boot = sum(deltas[method][idx] for idx in indices) / n
                deviations.append(abs(boot - obs[method]))
            max_deviation.append(max(deviations))
    radius = sorted(max_deviation)[int(0.95 * (len(max_deviation) - 1))] if max_deviation else 0.0
    return {method: {"simul_ci_low": obs[method] - radius, "simul_ci_high": obs[method] + radius} for method in methods}


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
    name_width = 40
    col_width = 14
    lines = [
        "=" * 110,
        f"ATLAS-CoE cache-only validation: {case.title}",
        "=" * 110,
        "| Calibration: source labels + target unlabeled expert outputs only; target labels are final scoring only.",
        f"| Best single on target: {audit['best_target_model']} ({fmt_pct(audit['best_target_accuracy'])})",
        f"| Source base: {audit['base_model']} ({fmt_pct(audit['base_source_accuracy'])} on source)",
        f"| A-FGW-lite distance: {audit['alignment_distance']:.4f}",
        f"| LEDGER edges: {audit['ledger_edges']} total, {audit['ledger_strict_edges']} strict-positive",
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
            "ATLAS modules implemented in this cache-only run:",
            "- LEDGER: source repair/harm posterior, LCB, probability positive, leave-one-item influence.",
            "- A-FGW-lite: expert-anchored failure-state relation geometry audit and local distortion certificate.",
            "- TRIDENT: source evidence + alignment + target answer-support stability certificate.",
            "- NULLSHIELD: target-unlabeled question-output decoupling p-values for proposed switches.",
            "- TAPESTRY-lite: source-only leave-one-group-out method weighting.",
            "- PROSPECT-lite: simultaneous bootstrap interval over the candidate method family.",
            "",
            "This is an offline approximation of the ATLAS proposal, not a full external FGW solver run.",
        ]
    )
    write_text(path, lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    case_dir = args.output_dir / case.case_id
    ledger_rows, ledger = build_edge_ledger(
        source_y,
        models,
        source_rows,
        source_ids,
        case.group_key,
        args.posterior_draws,
        args.seed,
    )
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

    base_map = {rid: base_model for rid in target_ids}
    atlas_relaxed, relaxed_meta, relaxed_pvalues = trident_choices(
        proposal_maps,
        base_model,
        ledger,
        alignment,
        target_full,
        models,
        target_ids,
        method_weight_audit["weights"],
        args,
        strict=False,
    )
    atlas_strict, strict_meta, strict_pvalues = trident_choices(
        proposal_maps,
        base_model,
        ledger,
        alignment,
        target_full,
        models,
        target_ids,
        method_weight_audit["weights"],
        args,
        strict=True,
    )
    tapestry_vote, tapestry_vote_meta = tapestry_weighted_vote_choices(
        proposal_maps,
        method_weight_audit["weights"],
        base_model,
        target_ids,
        args.tapestry_base_prior,
        margin=None,
    )
    tapestry_conservative, tapestry_conservative_meta = tapestry_weighted_vote_choices(
        proposal_maps,
        method_weight_audit["weights"],
        base_model,
        target_ids,
        args.tapestry_base_prior,
        margin=args.tapestry_margin,
    )

    choice_maps: dict[str, dict[str, str]] = {
        "source_global_best": base_map,
        "atlas_tapestry_weighted_vote": tapestry_vote,
        "atlas_tapestry_conservative_vote": tapestry_conservative,
        "atlas_trident_relaxed_nullshield": atlas_relaxed,
        "atlas_trident_strict_nullshield": atlas_strict,
    }
    # Include fixed source-only proposal baselines for context. They are not picked using target labels.
    for name, choices in proposal_maps.items():
        choice_maps[f"proposal_{name}"] = choices

    sim_ci = simultaneous_bootstrap_ci(choice_maps, best_target_model, target_matrix, target_ids, args.bootstrap_iters, args.seed)
    metadata: dict[str, dict[str, Any]] = {
        "source_global_best": {"source_accuracy": base_source_acc},
        "atlas_tapestry_weighted_vote": tapestry_vote_meta,
        "atlas_tapestry_conservative_vote": tapestry_conservative_meta,
        "atlas_trident_relaxed_nullshield": {
            "atlas": "LEDGER + A-FGW-lite + TRIDENT relaxed + NULLSHIELD + TAPESTRY-lite",
            "trident": relaxed_meta,
            "method_weights": method_weight_audit["weights"],
        },
        "atlas_trident_strict_nullshield": {
            "atlas": "LEDGER strict-LCB + A-FGW-lite + TRIDENT strict + NULLSHIELD + TAPESTRY-lite",
            "trident": strict_meta,
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
                "alignment_distance": alignment["alignment_distance"],
                "ledger_strict_edges": sum(
                    1
                    for edge in ledger_rows
                    if float(edge["lcb_gain"]) > 0.0
                    and float(edge["prob_positive"]) >= args.ledger_strict_prob
                    and float(edge["item_influence"]) <= args.max_item_influence
                ),
            }
        )
        rows.append(row)

    null_rows = [
        {"id": rid, "relaxed_pvalue": relaxed_pvalues.get(rid), "strict_pvalue": strict_pvalues.get(rid)}
        for rid in target_ids
    ]
    audit = {
        "case_id": case.case_id,
        "title": case.title,
        "models": models,
        "excluded_models": sorted(excluded),
        "source_samples": len(source_ids),
        "target_samples": len(target_ids),
        "base_model": base_model,
        "base_source_accuracy": base_source_acc,
        "best_target_model": best_target_model,
        "best_target_accuracy": best_target_acc,
        "instance_oracle": oracle_acc,
        "alignment_distance": alignment["alignment_distance"],
        "alignment_p90": alignment["alignment_p90"],
        "ledger_edges": len(ledger_rows),
        "ledger_strict_edges": sum(
            1
            for edge in ledger_rows
            if float(edge["lcb_gain"]) > 0.0
            and float(edge["prob_positive"]) >= args.ledger_strict_prob
            and float(edge["item_influence"]) <= args.max_item_influence
        ),
        "method_weight_audit": method_weight_audit,
        "proposal_meta": proposal_meta,
        "note": "ATLAS routing uses source labels and target unlabeled expert outputs only. Target labels are final scoring only.",
    }
    state_rows = [
        {
            "target_state": state,
            "state_distortion": alignment["state_distortion"][state],
            "state_anchor_cost": alignment["state_anchor_cost"][state],
            "transported_source_state": alignment["transported_source_state"][state],
        }
        for state in sorted(alignment["state_distortion"], key=lambda item: int(item))
    ]

    write_csv(case_dir / "source_edge_ledger.csv", ledger_rows)
    write_csv(case_dir / "edge_leave_one_item_influence.csv", ledger_rows)
    write_csv(case_dir / "alignment_state_summary.csv", state_rows)
    write_json(case_dir / "alignment_audit.json", {key: value for key, value in alignment.items() if key != "transported_success"})
    write_jsonl(case_dir / "routing_null_pvalues.jsonl", null_rows)
    write_json(case_dir / "atlas_audit.json", audit)
    write_csv(case_dir / "atlas_results.csv", rows)
    write_json(case_dir / "atlas_results.json", rows)
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
            "models": models,
            "excluded_models": sorted(excluded),
            "implemented_modules": [
                "LEDGER",
                "A-FGW-lite",
                "TRIDENT",
                "NULLSHIELD",
                "TAPESTRY-lite",
                "PROSPECT-lite",
            ],
        },
    )
    render_case_report(case_dir / f"Bench_Harness_Result_atlas_{case.case_id}.txt", case, rows, target_rows, target_matrix, choice_maps, audit)
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    lines = [
        "# ATLAS-CoE Cache-Only Validation Results",
        "",
        "Implemented from `Bench-CoE_ATLAS-CoE_源目标关系几何开创性创新方案.md`:",
        "",
        "- LEDGER source repair-edge posterior ledger.",
        "- A-FGW-lite expert-anchored source-target behavior geometry alignment.",
        "- TRIDENT switch certificate over source evidence, alignment, and target answer-support stability.",
        "- NULLSHIELD target-unlabeled negative-control p-value for proposed switches.",
        "- TAPESTRY-lite source-only leave-one-group-out method weighting.",
        "- PROSPECT-lite simultaneous bootstrap CI over the candidate method family.",
        "",
        "Target labels are used only for final scoring and bootstrap intervals.",
        "",
        "| Case | Method | Target Acc | Best Single | Gain | Paired CI | Simul CI | Align | Strict Edges | Models |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    f"[{float(row.get('simul_ci_low', 0.0)) * 100:+.2f}%, {float(row.get('simul_ci_high', 0.0)) * 100:+.2f}%]",
                    f"{float(row.get('alignment_distance', 0.0)):.3f}",
                    str(row.get("ledger_strict_edges", "")),
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
        "# ATLAS-CoE Best-by-Target Summary",
        "",
        "| Case | Best Method | Accuracy | Best Single | Gain | Paired CI | Selection-Aware Simul CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_id, row in sorted(best_rows.items()):
        best_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case_id}`",
                    f"`{row['method']}`",
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for case in select_cases(args.cases):
        all_rows.extend(run_case(case, args))
    write_summary(args.output_dir, all_rows)


if __name__ == "__main__":
    main()
