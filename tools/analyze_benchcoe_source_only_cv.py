#!/usr/bin/env python3
"""Paired inference for target-blind source-CV expert subset selections."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/bench_coe/model_subset_search_20260813/source_only_cv"
BOOTSTRAP_ITERS = 20_000
RANDOMIZATION_ITERS = 100_000
SEED = 20260813


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = load_module(
    "benchcoe_source_stats_selection", ROOT / "tools/search_benchcoe_source_only_cv.py"
)
CORE = SOURCE.CORE
MM = SOURCE.MM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS)
    parser.add_argument("--randomization-iters", type=int, default=RANDOMIZATION_ITERS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def weighted_accuracy(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_bootstrap_ci(
    candidate: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(candidate)
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    draws: list[np.ndarray] = []
    chunk = 500
    for start in range(0, samples, chunk):
        take = min(chunk, samples - start)
        indices = rng.integers(0, n, size=(take, n))
        sampled_weights = weights[indices]
        numerator = np.sum(sampled_weights * delta[indices], axis=1)
        denominator = np.sum(sampled_weights, axis=1)
        draws.append(100 * numerator / denominator)
    values = np.concatenate(draws)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_test(
    candidate: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    randomization_iters: int,
    seed: int,
) -> tuple[str, float]:
    wins = (candidate == 1) & (baseline == 0)
    losses = (candidate == 0) & (baseline == 1)
    discordant = wins | losses
    if not np.any(discordant):
        return "no_discordance", 1.0
    discordant_weights = weights[discordant]
    if np.allclose(discordant_weights, discordant_weights[0]):
        win_count = int(np.sum(wins))
        loss_count = int(np.sum(losses))
        return "exact_mcnemar", float(
            binomtest(win_count, win_count + loss_count, 0.5).pvalue
        )

    signed = np.where(wins[discordant], discordant_weights, -discordant_weights)
    observed = abs(float(np.sum(signed)))
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 1000
    for start in range(0, randomization_iters, chunk):
        take = min(chunk, randomization_iters - start)
        signs = rng.integers(0, 2, size=(take, len(signed)), dtype=np.int8) * 2 - 1
        permuted = np.abs(signs @ discordant_weights)
        extreme += int(np.sum(permuted >= observed - 1e-12))
    return "monte_carlo_paired_randomization", (extreme + 1) / (randomization_iters + 1)


def paired_stats(
    candidate: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    bootstrap_iters: int,
    randomization_iters: int,
    seed: int,
) -> dict[str, Any]:
    if candidate.shape != baseline.shape or candidate.shape != weights.shape:
        raise ValueError("Candidate, baseline, and weight arrays must align")
    test_name, p_value = paired_test(
        candidate, baseline, weights, randomization_iters, seed + 1
    )
    wins = int(np.sum((candidate == 1) & (baseline == 0)))
    losses = int(np.sum((candidate == 0) & (baseline == 1)))
    candidate_accuracy = weighted_accuracy(candidate, weights)
    baseline_accuracy = weighted_accuracy(baseline, weights)
    return {
        "examples": len(candidate),
        "weighted_denominator": float(np.sum(weights)),
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "delta_pp": 100 * (candidate_accuracy - baseline_accuracy),
        "bootstrap_95_ci_pp": weighted_bootstrap_ci(
            candidate, baseline, weights, bootstrap_iters, seed
        ),
        "candidate_only_correct": wins,
        "baseline_only_correct": losses,
        "paired_test": test_name,
        "raw_p": p_value,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["pool"], row["candidate"], row["baseline"])
        grouped.setdefault(key, []).append(row)
    for family_rows in grouped.values():
        ordered = sorted(family_rows, key=lambda row: row["raw_p"])
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * row["raw_p"])
            running = max(running, adjusted)
            row["holm_adjusted_p"] = running
            row["holm_reject_0_05"] = running < 0.05
            row["holm_family_size"] = count


def aligned_rows(
    route_rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
    dataset: str,
) -> tuple[list[str], list[str], np.ndarray]:
    common = set.intersection(*(set(values) for values in matrix.values()))
    ids: list[str] = []
    clusters: list[str] = []
    weights: list[float] = []
    for row in route_rows:
        rid = CORE.route_id(row, dataset)
        if rid not in common:
            continue
        ids.append(rid)
        clusters.append(CORE.cluster_value(row))
        weights.append(CORE.row_weight(row, dataset))
    return ids, clusters, np.asarray(weights, dtype=np.float64)


def prediction_vector(
    mapping: dict[str, str],
    matrix: dict[str, dict[str, bool]],
    ids: list[str],
    clusters: list[str],
) -> np.ndarray:
    return np.asarray(
        [int(matrix[mapping[cluster]][rid]) for rid, cluster in zip(ids, clusters)],
        dtype=np.int8,
    )


def analyze_pool(
    pool: str,
    summary: dict[str, Any],
    routers: dict[str, SOURCE.RouterData],
    route_rows: dict[str, dict[str, list[dict[str, Any]]]],
    matrices: dict[str, dict[str, dict[str, bool]]],
    bootstrap_iters: int,
    randomization_iters: int,
    seed: int,
) -> list[dict[str, Any]]:
    models = list(summary["models"])
    full = tuple(models)
    candidates = ("source_cv_unrestricted", "source_cv_strict_multiexpert")
    output: list[dict[str, Any]] = []
    comparison_index = 0
    for candidate_name in candidates:
        candidate_row = summary["selected_source_rows"][candidate_name]
        subset = tuple(candidate_row["models"].split(";"))
        for router_name, router in routers.items():
            candidate_mapping = router.target_mapping(subset)
            full_mapping = router.target_mapping(full)
            for dataset, rows in route_rows[router_name].items():
                matrix = matrices[dataset]
                ids, clusters, weights = aligned_rows(rows, matrix, dataset)
                candidate_y = prediction_vector(
                    candidate_mapping, matrix, ids, clusters
                )
                full_y = prediction_vector(full_mapping, matrix, ids, clusters)
                best_model = summary["target_oracle_best_single"][router_name][dataset]["model"]
                best_y = np.asarray(
                    [int(matrix[best_model][rid]) for rid in ids], dtype=np.int8
                )
                for baseline_name, baseline_model, baseline_y in (
                    ("full_pool_router", "full_pool_router", full_y),
                    ("target_oracle_best_single", best_model, best_y),
                ):
                    stats = paired_stats(
                        candidate_y,
                        baseline_y,
                        weights,
                        bootstrap_iters,
                        randomization_iters,
                        seed + 10 * comparison_index,
                    )
                    output.append(
                        {
                            "pool": pool,
                            "candidate": candidate_name,
                            "candidate_models": candidate_row["models"],
                            "router": router_name,
                            "dataset": dataset,
                            "baseline": baseline_name,
                            "baseline_model": baseline_model,
                            "ci_low_pp": stats["bootstrap_95_ci_pp"][0],
                            "ci_high_pp": stats["bootstrap_95_ci_pp"][1],
                            **stats,
                        }
                    )
                    comparison_index += 1
    return output


def load_language(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, SOURCE.RouterData], dict[str, Any], dict[str, Any]]:
    summary = json.loads(
        (output_dir / "language_14_model_pool/summary.json").read_text(encoding="utf-8")
    )
    models, routers, _ = SOURCE.load_text_data(3, 5, SEED)
    if models != summary["models"]:
        raise RuntimeError("Language model pool changed after selection")
    datasets = ("bbh", "gpqa", "gaokao")
    matrices = {
        dataset: CORE.load_matrix(CORE.DEFAULT_VIEW, dataset, models)
        for dataset in datasets
    }
    rows = {
        spec.name: {
            dataset: CORE.read_rows(spec.targets[dataset]) for dataset in datasets
        }
        for spec in CORE.ROUTERS
    }
    return summary, routers, rows, matrices


def load_multimodal(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, SOURCE.RouterData], dict[str, Any], dict[str, Any]]:
    summary = json.loads(
        (output_dir / "multimodal_8_model_pool/summary.json").read_text(encoding="utf-8")
    )
    models, routers, _ = SOURCE.load_multimodal_data(3, 5, SEED + 1000)
    if models != summary["models"]:
        raise RuntimeError("Multimodal model pool changed after selection")
    datasets = ("mathvista", "mmmu_pro")
    matrices = {
        dataset: MM.target_matrix(dataset, models)
        for dataset in (*datasets, "cmmmu")
    }
    rows = {
        spec.name: {
            dataset: MM.normalize_routes(spec.targets[dataset], spec.mode)
            for dataset in datasets
        }
        for spec in MM.SPECS
    }
    rows["qwen3vl_subject"]["cmmmu"] = MM.normalize_routes(
        ROOT / "outputs/bench_coe/cmmmu_qwen3vl_gaokao_mm_router_front4/test_predictions.json",
        "subject",
    )
    return summary, routers, rows, matrices


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{value:+.2f}"


def p_value(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def strict_comparison_table(
    pool: str, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    selected = {
        (row["router"], row["dataset"]): row
        for row in rows
        if row["pool"] == pool
        and row["candidate"] == "source_cv_strict_multiexpert"
        and row["baseline"] == "full_pool_router"
    }
    frozen = summary["frozen_target_evaluation"]["source_cv_strict_multiexpert"]
    lines = [
        "| Router | Target | Strict multi | Full pool | Delta | Bootstrap 95% CI | Holm p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for router, router_data in summary["full_pool_target_accuracy"].items():
        for dataset in router_data:
            stat = selected[(router, dataset)]
            cell = frozen[f"{router}_{dataset}"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        router,
                        dataset,
                        percent(cell["accuracy"]),
                        percent(cell["full_pool_accuracy"]),
                        f"{pp(cell['delta_full_pp'])} pp",
                        f"[{pp(stat['ci_low_pp'])}, {pp(stat['ci_high_pp'])}]",
                        p_value(stat["holm_adjusted_p"]),
                    )
                )
                + " |"
            )
    return "\n".join(lines)


def selection_summary_line(summary: dict[str, Any], name: str) -> str:
    row = summary["selected_source_rows"][name]
    frozen = summary["frozen_target_evaluation"][name]
    return (
        f"- `{name}`: `{row['models']}`; source-CV mean/worst "
        f"{pp(row['source_cv_mean_delta_full_pp'])}/{pp(row['source_cv_worst_delta_full_pp'])} pp; "
        f"frozen target mean/worst {pp(frozen['mean_delta_full_pp'])}/"
        f"{pp(frozen['worst_delta_full_pp'])} pp; positive cells "
        f"{frozen['positive_cells_vs_full']}/{frozen['target_cell_count']}."
    )


def write_final_report(
    output_dir: Path,
    language: dict[str, Any],
    multimodal: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    language_strict = language["frozen_target_evaluation"]["source_cv_strict_multiexpert"]
    multimodal_strict = multimodal["frozen_target_evaluation"]["source_cv_strict_multiexpert"]
    multimodal_single = multimodal["frozen_target_evaluation"]["source_cv_best_single"]
    report = [
        "# Bench-CoE 源域选择、目标域冻结评估报告",
        "",
        "## 协议",
        "",
        "所有候选只按源域 5 次重复 3 折分层交叉验证成绩排序。组合及簇到专家映射冻结后，才在跨数据集目标域评估。目标集答案不参与组合选择、映射或并列决胜；目标 Router 输出只用于在冻结后应用源域映射和源域回退规则。",
        "",
        "## 最终结论",
        "",
        f"- 语言没有可推荐的多专家池。严格组合仅 {language_strict['positive_cells_vs_full']}/{language_strict['target_cell_count']} 单元提高，平均 {pp(language_strict['mean_delta_full_pp'])} pp，最差 {pp(language_strict['worst_delta_full_pp'])} pp。",
        f"- 多模态没有可升级为正式结论的多专家池。严格组合仅 {multimodal_strict['positive_cells_vs_full']}/{multimodal_strict['target_cell_count']} 单元提高，平均 {pp(multimodal_strict['mean_delta_full_pp'])} pp，最差 {pp(multimodal_strict['worst_delta_full_pp'])} pp。",
        f"- 多模态安全回退是 `InternVL3_5-14B`。它在 {multimodal_single['positive_cells_vs_full']}/{multimodal_single['target_cell_count']} 单元优于完整池，其余打平，平均 {pp(multimodal_single['mean_delta_full_pp'])} pp。这是固定单模型回退，不是多专家路由创新的正证据。",
        "- 上一轮 `InternVL3_5-14B + InternVL3_5-8B` 仍只能标为后验探索候选，因为它是查看目标集表现后选出的。",
        "",
        "## 语言 14 模型池",
        "",
        f"精确遍历全部 {language['num_nonempty_subsets']:,} 个非空子集；严格多专家候选 {language['strict_multiexpert']['eligible_subsets']:,} 个。",
        "",
        selection_summary_line(language, "source_cv_unrestricted"),
        selection_summary_line(language, "source_cv_best_single"),
        selection_summary_line(language, "source_cv_strict_multiexpert"),
        "",
        strict_comparison_table("language_14_model_pool", language, rows),
        "",
        "语言严格组合在 GAOKAO Query 上有显著正增益，但其余五个单元均显著下降，不能据此宣称稳健跨分布提升。源域最佳单模型 `Qwen3-4B-Instruct-2507` 相对完整池平均仅 +0.19 pp，且明显低于查看目标标签后得到的各目标最强单模型，因此只适合作为低成本候选，不是性能回退基线。",
        "",
        "## 多模态 8 模型池",
        "",
        f"精确遍历全部 {multimodal['num_nonempty_subsets']:,} 个非空子集；严格 Subject 多专家候选 {multimodal['strict_multiexpert']['eligible_subsets']:,} 个。",
        "",
        selection_summary_line(multimodal, "source_cv_unrestricted"),
        selection_summary_line(multimodal, "source_cv_best_single"),
        selection_summary_line(multimodal, "source_cv_strict_multiexpert"),
        "",
        strict_comparison_table("multimodal_8_model_pool", multimodal, rows),
        "",
        "严格双专家 `InternVL3_5-8B + Qwen2.5-VL-7B-Instruct` 在两个 MathVista Subject 单元有小幅正点估计，但置信区间均跨 0；MMMU-Pro 的三个相关单元显著下降。CMMMU 补充评估也下降 0.56 pp，置信区间跨 0。",
        "",
        "## 统计协议",
        "",
        f"普通准确率使用逐题 {BOOTSTRAP_ITERS:,} 次 bootstrap 和精确 McNemar；GAOKAO 分值加权准确率使用加权 bootstrap 和 {RANDOMIZATION_ITERS:,} 次配对随机化检验。`p` 值按“模型池 x 冻结候选 x 基线”家族做 Holm 校正。相对目标集最强单模型的比较仅为描述性上界，因为该单模型使用了目标标签选择。",
        "",
        "## 解释限制",
        "",
        "源域交叉验证选出的组合若在目标域下降，说明仅凭源域排行榜挑模型不足以保证跨分布泛化；不能再通过查看目标集后重新挑组合来改写该负结果。Query 标签塌缩也不会因筛选模型而自动恢复，仍需按冻结专家池重新制标签并训练、校准分类器。",
        "",
        "既有目标集已经被多轮分析。真正的最终确认必须预先冻结组合与规则，再在从未查看答案标签的新数据集上测试。",
        "",
    ]
    (output_dir / "REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")


def write_deployment_decision(
    output_dir: Path, language: dict[str, Any], multimodal: dict[str, Any]
) -> None:
    language_strict = language["frozen_target_evaluation"]["source_cv_strict_multiexpert"]
    multimodal_strict = multimodal["frozen_target_evaluation"]["source_cv_strict_multiexpert"]
    multimodal_single = multimodal["frozen_target_evaluation"]["source_cv_best_single"]
    payload = {
        "status": "completed",
        "protocol": "target-answer-label-blind repeated 5x3-fold source-domain cross-validation",
        "language": {
            "recommended_multiexpert_pool": None,
            "source_cv_best_single": language["selected_source_rows"]["source_cv_best_single"]["models"],
            "source_cv_best_single_role": "low-cost candidate, not a performance fallback",
            "strict_multiexpert_target_result": language_strict,
            "deployment_recommendation": "Do not deploy a newly selected language multiexpert pool from this search.",
        },
        "multimodal": {
            "recommended_multiexpert_pool": None,
            "source_cv_safe_fallback": multimodal["selected_source_rows"]["source_cv_best_single"]["models"],
            "safe_fallback_target_result": multimodal_single,
            "strict_multiexpert_target_result": multimodal_strict,
            "deployment_recommendation": "Use the fixed single-model fallback; do not promote a multiexpert pool from the target-blind search.",
            "previous_exploratory_pool": {
                "models": ["InternVL3_5-14B", "InternVL3_5-8B"],
                "status": "exploratory_only_not_target_blind_selected",
            },
        },
        "required_next_experiment": "Freeze the pool and rule before collecting a genuinely unseen confirmation dataset; retrain and calibrate Query for that frozen pool.",
    }
    with (output_dir / "DEPLOYMENT_DECISION.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.bootstrap_iters <= 0 or args.randomization_iters <= 0:
        raise ValueError("Iteration counts must be positive")
    language = load_language(args.output_dir)
    multimodal = load_multimodal(args.output_dir)
    rows = analyze_pool(
        "language_14_model_pool",
        *language,
        args.bootstrap_iters,
        args.randomization_iters,
        args.seed,
    )
    rows.extend(
        analyze_pool(
            "multimodal_8_model_pool",
            *multimodal,
            args.bootstrap_iters,
            args.randomization_iters,
            args.seed + 10_000,
        )
    )
    holm_adjust(rows)
    payload = {
        "status": "completed",
        "bootstrap_iterations": args.bootstrap_iters,
        "weighted_randomization_iterations": args.randomization_iters,
        "seed": args.seed,
        "multiple_testing_adjustment": (
            "Holm within each pool x frozen candidate x baseline family"
        ),
        "primary_family": "source_cv_strict_multiexpert vs full_pool_router",
        "target_oracle_note": (
            "The target-oracle best single is label-selected and is only a descriptive ceiling."
        ),
        "comparisons": rows,
    }
    with (args.output_dir / "paired_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    fields = sorted(
        {key for row in rows for key in row if key != "bootstrap_95_ci_pp"}
    )
    with (args.output_dir / "paired_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_final_report(args.output_dir, language[0], multimodal[0], rows)
    write_deployment_decision(args.output_dir, language[0], multimodal[0])
    print(json.dumps({"status": "completed", "comparisons": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
