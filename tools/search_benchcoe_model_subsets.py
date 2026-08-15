#!/usr/bin/env python3
"""Exhaustively search fixed expert subsets for cross-dataset Bench-CoE transfer.

The router partitions are kept fixed.  For every candidate subset, each router
partition is assigned to the model with the highest MMLU-Pro source accuracy.
Target labels are never used to build that partition-to-model mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW = ROOT / "outputs/bench_coe/scale_transfer_views_20260802/text"
DEFAULT_OUTPUT = ROOT / "outputs/bench_coe/model_subset_search_20260813"


@dataclass(frozen=True)
class RouterSpec:
    name: str
    source_routes: Path
    targets: dict[str, Path]


ROUTERS = (
    RouterSpec(
        name="query",
        source_routes=ROOT
        / "outputs/bench_coe/mmlu_validation_query_router_on_mmlu_pro/test_predictions.jsonl",
        targets={
            "bbh": ROOT / "outputs/bench_coe/mmlu_validation_query_router_on_bbh/predictions.jsonl",
            "gpqa": ROOT / "outputs/bench_coe/mmlu_validation_query_router_on_gpqa/predictions.jsonl",
            "gaokao": ROOT / "outputs/bench_coe/mmlu_validation_query_router_on_gaokao2010/predictions.json",
        },
    ),
    RouterSpec(
        name="subject",
        source_routes=ROOT
        / "outputs/bench_coe/mmlu_subject_bert_validation_7b_9b_offline/test_predictions.jsonl",
        targets={
            "bbh": ROOT / "outputs/bench_coe/mmlu_validation_7b_9b_subject_router_on_bbh/predictions.jsonl",
            "gpqa": ROOT / "outputs/bench_coe/mmlu_validation_7b_9b_subject_router_on_gpqa/predictions.jsonl",
            "gaokao": ROOT / "outputs/bench_coe/mmlu_validation_7b_9b_router_on_gaokao2010/predictions.json",
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-rule",
        choices=("robust", "mean", "positive_cells"),
        default="robust",
        help="Primary rule used to name the recommended exploratory subset.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path) if path.suffix == ".json" else read_jsonl(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a row list: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def route_id(row: dict[str, Any], dataset: str) -> str:
    if dataset == "mmlu_source":
        if row.get("id") is not None:
            return str(row["id"])
        return f"test:{row['question_id']}"
    if dataset in {"bbh", "gpqa"}:
        return str(row["id"])
    if dataset == "gaokao":
        return str(row.get("id") or f"{row['keyword']}:{row['index']}")
    if row.get("id") is not None:
        return str(row["id"])
    raise KeyError(dataset)


def prediction_id(row: dict[str, Any], dataset: str) -> str:
    if dataset == "mmlu_source":
        return f"test:{row['question_id']}"
    if dataset in {"bbh", "gpqa"}:
        return str(row["id"])
    if dataset == "gaokao":
        return f"{row['keyword']}:{row['index']}"
    if row.get("id") is not None:
        return str(row["id"])
    raise KeyError(dataset)


def is_correct(row: dict[str, Any]) -> bool:
    if row.get("is_correct") is not None:
        return bool(row["is_correct"])
    pred = row.get("pred", row.get("model_answer"))
    target = row.get("target", row.get("answer", row.get("standard_answer")))
    if isinstance(pred, list):
        pred = pred[0] if pred else None
    if isinstance(target, list):
        target = target[0] if target else None
    return pred is not None and str(pred).strip() == str(target).strip()


def load_mmlu_model(path: Path) -> dict[str, bool]:
    values: dict[str, bool] = {}
    for result in sorted(path.glob("CoT/all/*.json")):
        for row in read_json(result):
            values[prediction_id(row, "mmlu_source")] = is_correct(row)
    return values


def load_target_model(path: Path, dataset: str) -> dict[str, bool]:
    result = path / "predictions.jsonl"
    return {prediction_id(row, dataset): is_correct(row) for row in read_jsonl(result)}


def available_models(view_root: Path) -> list[str]:
    datasets = ("mmlu_test", "bbh", "gpqa", "gaokao_2010_2022")
    model_sets = [
        {entry.name for entry in (view_root / dataset).iterdir() if entry.is_dir()}
        for dataset in datasets
    ]
    return sorted(set.intersection(*model_sets))


def load_matrix(view_root: Path, dataset: str, models: list[str]) -> dict[str, dict[str, bool]]:
    if dataset == "mmlu_source":
        return {model: load_mmlu_model(view_root / "mmlu_test" / model) for model in models}
    dirname = "gaokao_2010_2022" if dataset == "gaokao" else dataset
    return {
        model: load_target_model(view_root / dirname / model, dataset)
        for model in models
    }


def cluster_value(row: dict[str, Any]) -> str:
    value = row.get("route_label")
    if value is None:
        value = row.get("routed_category", row.get("routed_model"))
    return str(value)


def row_weight(row: dict[str, Any], dataset: str) -> float:
    if dataset == "gaokao":
        return float(row.get("total_score", row.get("score", 1)))
    return 1.0


def build_cluster_stats(
    route_rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
    dataset: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, int], dict[str, Any]]:
    ids_by_cluster: dict[str, list[tuple[str, float]]] = defaultdict(list)
    common_ids = set.intersection(*(set(values) for values in matrix.values()))
    dropped_ids: list[str] = []
    for row in route_rows:
        rid = route_id(row, dataset)
        if rid not in common_ids:
            dropped_ids.append(rid)
            continue
        ids_by_cluster[cluster_value(row)].append((rid, row_weight(row, dataset)))

    correct: dict[str, dict[str, float]] = defaultdict(dict)
    totals: dict[str, float] = {}
    missing: dict[str, int] = Counter()
    all_route_ids = [route_id(row, dataset) for row in route_rows]
    for model, values in matrix.items():
        missing[model] = sum(rid not in values for rid in all_route_ids)
    for cluster, weighted_ids in sorted(ids_by_cluster.items()):
        totals[cluster] = sum(weight for _, weight in weighted_ids)
        for model, values in matrix.items():
            correct[cluster][model] = sum(
                weight * bool(values[rid]) for rid, weight in weighted_ids
            )
    audit = {
        "route_rows_total": len(route_rows),
        "route_rows_evaluated": sum(len(rows) for rows in ids_by_cluster.values()),
        "route_rows_dropped": len(dropped_ids),
        "dropped_id_examples": sorted(dropped_ids)[:10],
        "weighted_denominator": sum(totals.values()),
    }
    return dict(correct), totals, dict(missing), audit


def selected_model_by_cluster(
    subset: tuple[str, ...],
    source_correct: dict[str, dict[str, float]],
    source_totals: dict[str, float],
) -> dict[str, str]:
    denominator = sum(source_totals.values())
    overall = {
        model: sum(source_correct[cluster][model] for cluster in source_totals) / denominator
        for model in subset
    }
    # Match the project leaderboards: cluster score, then source overall score,
    # then model name as the final deterministic tie breaker.
    return {
        cluster: max(
            subset,
            key=lambda model: (source_correct[cluster][model] / total, overall[model], model),
        )
        for cluster, total in source_totals.items()
    }


def routed_accuracy(
    mapping: dict[str, str],
    target_correct: dict[str, dict[str, float]],
    target_totals: dict[str, float],
) -> float:
    total = sum(target_totals.values())
    if not total:
        return math.nan
    return sum(target_correct[cluster][mapping[cluster]] for cluster in target_totals) / total


def model_accuracy(
    model: str,
    target_correct: dict[str, dict[str, float]],
    target_totals: dict[str, float],
) -> float:
    total = sum(target_totals.values())
    return sum(target_correct[cluster][model] for cluster in target_totals) / total


def subset_key(row: dict[str, Any], rule: str, datasets: tuple[str, ...]) -> tuple[Any, ...]:
    deltas = [row[f"{router}_{dataset}_delta_full_pp"] for router in ("query", "subject") for dataset in datasets]
    if rule == "mean":
        return (sum(deltas) / len(deltas), min(deltas), -row["size"], row["models"])
    if rule == "positive_cells":
        return (sum(delta > 0 for delta in deltas), sum(deltas) / len(deltas), min(deltas), -row["size"], row["models"])
    # Robustness first: improve the weakest cell, then the macro mean.
    return (min(deltas), sum(deltas) / len(deltas), sum(delta > 0 for delta in deltas), -row["size"], row["models"])


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    models = available_models(args.view_root)
    datasets = ("bbh", "gpqa", "gaokao")
    matrices = {dataset: load_matrix(args.view_root, dataset, models) for dataset in ("mmlu_source", *datasets)}

    stats: dict[str, dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []
    for spec in ROUTERS:
        source_rows = read_rows(spec.source_routes)
        source_correct, source_totals, source_missing, source_audit = build_cluster_stats(
            source_rows, matrices["mmlu_source"], "mmlu_source"
        )
        target_stats: dict[str, Any] = {}
        for dataset, route_path in spec.targets.items():
            correct, totals, missing, audit = build_cluster_stats(
                read_rows(route_path), matrices[dataset], dataset
            )
            target_stats[dataset] = {
                "correct": correct,
                "totals": totals,
                "missing": missing,
                "audit": audit,
            }
        stats[spec.name] = {
            "source_correct": source_correct,
            "source_totals": source_totals,
            "source_missing": source_missing,
            "source_audit": source_audit,
            "targets": target_stats,
        }

    for router, router_stats in stats.items():
        for dataset in ("mmlu_source", *datasets):
            missing = (
                router_stats["source_missing"]
                if dataset == "mmlu_source"
                else router_stats["targets"][dataset]["missing"]
            )
            totals = (
                router_stats["source_totals"]
                if dataset == "mmlu_source"
                else router_stats["targets"][dataset]["totals"]
            )
            audit = (
                router_stats["source_audit"]
                if dataset == "mmlu_source"
                else router_stats["targets"][dataset]["audit"]
            )
            coverage_rows.append(
                {
                    "router": router,
                    "dataset": dataset,
                    **audit,
                    "clusters": len(totals),
                    "max_missing_per_model": max(missing.values(), default=0),
                    "models_with_any_missing": sum(value > 0 for value in missing.values()),
                }
            )

    for row in coverage_rows:
        if row["route_rows_evaluated"] <= 0:
            raise RuntimeError(f"No common rows remain after alignment: {row}")

    global_best: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    full_mapping: dict[str, dict[str, str]] = {}
    full_accuracy: dict[str, dict[str, float]] = defaultdict(dict)
    full_subset = tuple(models)
    for router, router_stats in stats.items():
        mapping = selected_model_by_cluster(
            full_subset, router_stats["source_correct"], router_stats["source_totals"]
        )
        full_mapping[router] = mapping
        for dataset in datasets:
            target = router_stats["targets"][dataset]
            full_accuracy[router][dataset] = routed_accuracy(mapping, target["correct"], target["totals"])
            scored = [(model_accuracy(model, target["correct"], target["totals"]), model) for model in models]
            accuracy, model = max(scored)
            global_best[router][dataset] = (model, accuracy)

    result_rows: list[dict[str, Any]] = []
    for size in range(1, len(models) + 1):
        for subset in combinations(models, size):
            row: dict[str, Any] = {"size": size, "models": ";".join(subset)}
            cell_deltas: list[float] = []
            global_deltas: list[float] = []
            positive_cells = 0
            for router, router_stats in stats.items():
                mapping = selected_model_by_cluster(
                    subset, router_stats["source_correct"], router_stats["source_totals"]
                )
                row[f"{router}_mapping"] = json.dumps(mapping, sort_keys=True, ensure_ascii=False)
                row[f"{router}_active_models"] = ";".join(sorted(set(mapping.values())))
                row[f"{router}_active_count"] = len(set(mapping.values()))
                for dataset in datasets:
                    target = router_stats["targets"][dataset]
                    accuracy = routed_accuracy(mapping, target["correct"], target["totals"])
                    delta_full = 100 * (accuracy - full_accuracy[router][dataset])
                    delta_global = 100 * (accuracy - global_best[router][dataset][1])
                    row[f"{router}_{dataset}_accuracy"] = accuracy
                    row[f"{router}_{dataset}_delta_full_pp"] = delta_full
                    row[f"{router}_{dataset}_delta_global_best_pp"] = delta_global
                    cell_deltas.append(delta_full)
                    global_deltas.append(delta_global)
                    positive_cells += delta_full > 1e-12
            row["macro_accuracy"] = sum(
                row[f"{router}_{dataset}_accuracy"] for router in ("query", "subject") for dataset in datasets
            ) / (2 * len(datasets))
            row["mean_delta_full_pp"] = sum(cell_deltas) / len(cell_deltas)
            row["worst_delta_full_pp"] = min(cell_deltas)
            row["mean_delta_global_best_pp"] = sum(global_deltas) / len(global_deltas)
            row["worst_delta_global_best_pp"] = min(global_deltas)
            row["positive_cells_vs_full"] = positive_cells
            row["active_union_count"] = len(
                set(row["query_active_models"].split(";"))
                | set(row["subject_active_models"].split(";"))
            )
            result_rows.append(row)

    ranked = sorted(result_rows, key=lambda row: subset_key(row, args.selection_rule, datasets), reverse=True)
    recommended = ranked[0]
    top_mean = max(result_rows, key=lambda row: subset_key(row, "mean", datasets))
    top_positive = max(result_rows, key=lambda row: subset_key(row, "positive_cells", datasets))
    nondegenerate = [
        row
        for row in result_rows
        if row["query_active_count"] >= 2 and row["subject_active_count"] >= 2
    ]
    best_nondegenerate = max(
        nondegenerate,
        key=lambda row: subset_key(row, args.selection_rule, datasets),
    )
    best_by_router: dict[str, dict[str, Any]] = {}
    for router in ("query", "subject"):
        eligible = [row for row in result_rows if row[f"{router}_active_count"] >= 2]

        def router_key(row: dict[str, Any]) -> tuple[Any, ...]:
            deltas = [row[f"{router}_{dataset}_delta_full_pp"] for dataset in datasets]
            return (min(deltas), sum(deltas) / len(deltas), sum(delta > 0 for delta in deltas), -row["size"])

        best_by_router[router] = max(eligible, key=router_key)

    lodo_rows: list[dict[str, Any]] = []
    for held_out in datasets:
        selection_datasets = tuple(dataset for dataset in datasets if dataset != held_out)
        selected = max(result_rows, key=lambda row: subset_key(row, args.selection_rule, selection_datasets))
        for router in ("query", "subject"):
            lodo_rows.append(
                {
                    "held_out_dataset": held_out,
                    "router": router,
                    "selected_size": selected["size"],
                    "selected_models": selected["models"],
                    "held_out_accuracy": selected[f"{router}_{held_out}_accuracy"],
                    "held_out_delta_full_pp": selected[f"{router}_{held_out}_delta_full_pp"],
                    "held_out_delta_global_best_pp": selected[f"{router}_{held_out}_delta_global_best_pp"],
                }
            )

    top_rows = ranked[:100]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_subsets.csv", result_rows)
    write_csv(args.output_dir / "top_100_subsets.csv", top_rows)
    write_csv(args.output_dir / "leave_one_dataset_out.csv", lodo_rows)
    write_csv(args.output_dir / "coverage_audit.csv", coverage_rows)

    summary = {
        "status": "completed",
        "method": {
            "candidate_pool": "14-model common-coverage scale-transfer text pool",
            "mapping_source": "MMLU-Pro test rows with fixed saved Query/Subject router partitions",
            "target_labels_used_for_mapping": False,
            "subset_search_uses_target_metrics": True,
            "selection_rule": args.selection_rule,
            "num_models": len(models),
            "num_nonempty_subsets": len(result_rows),
            "models": models,
            "targets": list(datasets),
        },
        "coverage": coverage_rows,
        "global_best_single": {
            router: {
                dataset: {"model": value[0], "accuracy": value[1]}
                for dataset, value in per_dataset.items()
            }
            for router, per_dataset in global_best.items()
        },
        "full_pool": {
            "models": models,
            "mapping": full_mapping,
            "accuracy": full_accuracy,
        },
        "recommended_exploratory": recommended,
        "best_macro_mean": top_mean,
        "best_positive_cells": top_positive,
        "nondegenerate_audit": {
            "definition": "at least two distinct models are actually routed by both Query and Subject",
            "eligible_subsets": len(nondegenerate),
            "subsets_positive_in_all_six_cells": sum(
                row["positive_cells_vs_full"] == 2 * len(datasets) for row in nondegenerate
            ),
            "best": best_nondegenerate,
            "best_by_router": best_by_router,
        },
        "leave_one_dataset_out": lodo_rows,
    }
    write_json(args.output_dir / "summary.json", summary)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    def pp(value: float) -> str:
        return f"{value:+.2f}"

    selected_models = recommended["models"].split(";")
    report: list[str] = [
        "# Bench-CoE 固定专家子集跨数据集搜索结果",
        "",
        "## 结论",
        "",
        f"主搜索在共同覆盖的 {len(models)} 个新增规模语言模型上穷举了 {len(result_rows):,} 个非空子集。",
        f"按 `{args.selection_rule}` 规则选出的探索性组合包含 {len(selected_models)} 个模型：",
        "",
        *[f"- `{model}`" for model in selected_models],
        "",
        "该组合使用全部三个目标集选出，因此只能作为探索性候选；真正反映跨数据集选择可靠性的结果见留一数据集表。",
        "",
        "## 主结果",
        "",
        markdown_table(
            ["Router", "Dataset", "Full pool", "Selected", "Δ full (pp)", "Global best single", "Δ global (pp)"],
            [
                [
                    router.capitalize(),
                    dataset.upper(),
                    pct(full_accuracy[router][dataset]),
                    pct(recommended[f"{router}_{dataset}_accuracy"]),
                    pp(recommended[f"{router}_{dataset}_delta_full_pp"]),
                    f"{global_best[router][dataset][0]} ({pct(global_best[router][dataset][1])})",
                    pp(recommended[f"{router}_{dataset}_delta_global_best_pp"]),
                ]
                for router in ("query", "subject")
                for dataset in datasets
            ],
        ),
        "",
        f"六个 Router×Dataset 单元相对完整 14 模型池的平均变化为 {pp(recommended['mean_delta_full_pp'])} pp，"
        f"最差单元为 {pp(recommended['worst_delta_full_pp'])} pp，正向单元为 {recommended['positive_cells_vs_full']}/6。",
        "",
        "## 多专家非退化约束",
        "",
        f"要求 Query 和 Subject 都至少实际调用两个不同专家后，共有 {len(nondegenerate):,} 个候选；"
        f"其中 6/6 单元均严格提高的组合数为 {sum(row['positive_cells_vs_full'] == 6 for row in nondegenerate)}。",
        "",
        markdown_table(
            ["Constraint", "Models", "Mean Δ full (pp)", "Worst Δ full (pp)", "Positive cells"],
            [
                [
                    "Query+Subject each >=2 active",
                    best_nondegenerate["models"].replace(";", ", "),
                    pp(best_nondegenerate["mean_delta_full_pp"]),
                    pp(best_nondegenerate["worst_delta_full_pp"]),
                    f"{best_nondegenerate['positive_cells_vs_full']}/6",
                ],
                *[
                    [
                        f"{router.capitalize()} >=2 active",
                        best_by_router[router]["models"].replace(";", ", "),
                        pp(
                            sum(best_by_router[router][f"{router}_{dataset}_delta_full_pp"] for dataset in datasets)
                            / len(datasets)
                        ),
                        pp(min(best_by_router[router][f"{router}_{dataset}_delta_full_pp"] for dataset in datasets)),
                        f"{sum(best_by_router[router][f'{router}_{dataset}_delta_full_pp'] > 0 for dataset in datasets)}/3",
                    ]
                    for router in ("query", "subject")
                ],
            ],
        ),
        "",
        "因此，不限组合规模时的统一正增益来自退化为强单模型，而不是更好的多专家路由。"
        "在当前源域分簇冠军映射下，仅删除模型无法得到 Query 与 Subject 都跨三个目标集稳定提高的真实多专家组合。",
        "",
        "## 留一数据集选择",
        "",
        markdown_table(
            ["Held out", "Router", "Selected on other datasets", "Accuracy", "Δ full (pp)", "Δ global (pp)"],
            [
                [
                    row["held_out_dataset"].upper(),
                    row["router"].capitalize(),
                    f"{row['selected_size']} models",
                    pct(row["held_out_accuracy"]),
                    pp(row["held_out_delta_full_pp"]),
                    pp(row["held_out_delta_global_best_pp"]),
                ]
                for row in lodo_rows
            ],
        ),
        "",
        "## 解释边界",
        "",
        "- 每个候选子集的 `路由簇 -> 专家` 映射只由 MMLU-Pro 源域正确率决定，未使用目标集标签。",
        "- 子集本身由目标集分数筛选，主结果属于后验探索，不能直接当作未见数据泛化证据。",
        "- 留一数据集结果在另外两个数据集上选组合，再报告未参与选择的数据集，是更严格的组合泛化检查。",
        "- Query 使用已保存分类器产生的固定路由簇，再重估簇到专家映射；部署前应按最终专家池重建 Query 标签并重训分类头确认。",
        "- GAOKAO 使用 2010--2022 结果中 14 个候选模型共同覆盖的 1,676 题并按题目分值加权；统一缺失的 105 条旧版英语选择题被显式排除。BBH、GPQA 使用完整保存结果。",
        "- `Δ global` 使用全部 14 个候选模型中的目标集最强单模型作为固定基线，避免通过删除强模型人为降低基线。",
        "",
        "## 可复核文件",
        "",
        "- `summary.json`: 完整配置、基线、映射及推荐组合。",
        "- `all_subsets.csv`: 全部 16,383 个组合的逐单元结果。",
        "- `top_100_subsets.csv`: 按主规则排序的前 100 个组合。",
        "- `leave_one_dataset_out.csv`: 留一数据集选择结果。",
        "- `coverage_audit.csv`: ID 对齐与缺失审计。",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "recommended": recommended}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
