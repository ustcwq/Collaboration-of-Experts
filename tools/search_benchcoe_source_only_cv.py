#!/usr/bin/env python3
"""Select Bench-CoE expert subsets without inspecting target-domain labels.

The older subset searches correctly learned cluster-to-expert mappings on the
source domain, but ranked candidate subsets with target-domain accuracy.  This
script closes that leakage path.  It ranks every candidate using deterministic
stratified cross-validation on the source domain, freezes the selected subset,
rebuilds its mapping on all source rows, and only then evaluates target domains.

The 14-model language pool and the 8-model multimodal pool are both exhaustively
enumerated.  Existing per-item model predictions are replayed, so no model
forward pass is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/bench_coe/model_subset_search_20260813/source_only_cv"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("benchcoe_source_only_core", ROOT / "tools/search_benchcoe_model_subsets.py")
MM = load_module("benchcoe_source_only_mm", ROOT / "tools/search_benchcoe_multimodal_subsets.py")


@dataclass
class FoldStats:
    train_correct: dict[str, dict[str, float]]
    train_totals: dict[str, float]
    validation_correct: dict[str, dict[str, float]]
    validation_totals: dict[str, float]


@dataclass
class RouterData:
    name: str
    source_correct: dict[str, dict[str, float]]
    source_totals: dict[str, float]
    folds: list[FoldStats]
    targets: dict[str, dict[str, Any]]
    target_mapping: Callable[[tuple[str, ...]], dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stratified_folds(
    route_rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
    dataset: str,
    requested_folds: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    common_ids = set.intersection(*(set(values) for values in matrix.values()))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in route_rows:
        rid = CORE.route_id(row, dataset)
        if rid in common_ids:
            groups.setdefault(CORE.cluster_value(row), []).append(row)
    if not groups:
        raise RuntimeError(f"No aligned source rows for {dataset}")
    min_cluster_size = min(len(rows) for rows in groups.values())
    effective_folds = min(requested_folds, min_cluster_size)
    if effective_folds < 2:
        raise RuntimeError(
            f"At least one source cluster has fewer than two rows: "
            f"{dict(sorted((cluster, len(rows)) for cluster, rows in groups.items()))}"
        )
    folds: list[list[dict[str, Any]]] = [[] for _ in range(effective_folds)]
    for cluster, rows in sorted(groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_order(CORE.route_id(row, dataset), seed),
        )
        for index, row in enumerate(ordered):
            folds[index % effective_folds].append(row)
    audit = {
        "requested_folds": requested_folds,
        "effective_folds": effective_folds,
        "aligned_rows": sum(len(rows) for rows in groups.values()),
        "cluster_counts": dict(sorted((cluster, len(rows)) for cluster, rows in groups.items())),
        "fold_sizes": [len(fold) for fold in folds],
    }
    return folds, audit


def build_fold_stats(
    route_rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
    dataset: str,
    requested_folds: int,
    seed: int,
) -> tuple[list[FoldStats], dict[str, Any]]:
    validation_folds, audit = stratified_folds(
        route_rows, matrix, dataset, requested_folds, seed
    )
    result: list[FoldStats] = []
    for held_out, validation_rows in enumerate(validation_folds):
        train_rows = [
            row
            for fold_index, fold_rows in enumerate(validation_folds)
            if fold_index != held_out
            for row in fold_rows
        ]
        train_correct, train_totals, _, _ = CORE.build_cluster_stats(
            train_rows, matrix, dataset
        )
        validation_correct, validation_totals, _, _ = CORE.build_cluster_stats(
            validation_rows, matrix, dataset
        )
        if set(validation_totals).difference(train_totals):
            raise RuntimeError("Stratification left a validation-only source cluster")
        result.append(
            FoldStats(
                train_correct=train_correct,
                train_totals=train_totals,
                validation_correct=validation_correct,
                validation_totals=validation_totals,
            )
        )
    return result, audit


def build_repeated_fold_stats(
    route_rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, bool]],
    dataset: str,
    requested_folds: int,
    repeats: int,
    seed: int,
) -> tuple[list[FoldStats], dict[str, Any]]:
    all_folds: list[FoldStats] = []
    repeat_audits: list[dict[str, Any]] = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat * 100_003
        folds, audit = build_fold_stats(
            route_rows,
            matrix,
            dataset,
            requested_folds,
            repeat_seed,
        )
        all_folds.extend(folds)
        repeat_audits.append({"repeat": repeat, "seed": repeat_seed, **audit})
    return all_folds, {
        "repeats": repeats,
        "total_validation_folds": len(all_folds),
        "repeat_audits": repeat_audits,
    }


def overall_source_model(
    subset: tuple[str, ...],
    source_correct: dict[str, dict[str, float]],
    source_totals: dict[str, float],
) -> str:
    denominator = sum(source_totals.values())
    return max(
        subset,
        key=lambda model: (
            sum(source_correct[cluster][model] for cluster in source_totals) / denominator,
            model,
        ),
    )


def mapping_with_fallback(
    subset: tuple[str, ...],
    source_correct: dict[str, dict[str, float]],
    source_totals: dict[str, float],
    target_clusters: set[str],
) -> dict[str, str]:
    mapping = CORE.selected_model_by_cluster(subset, source_correct, source_totals)
    fallback = overall_source_model(subset, source_correct, source_totals)
    for cluster in target_clusters:
        mapping.setdefault(cluster, fallback)
    return mapping


def cross_validated_router(
    subset: tuple[str, ...], router: RouterData
) -> tuple[float, int, int]:
    weighted_correct = 0.0
    denominator = 0.0
    active_counts: list[int] = []
    for fold in router.folds:
        mapping = CORE.selected_model_by_cluster(
            subset, fold.train_correct, fold.train_totals
        )
        active_counts.append(len(set(mapping.values())))
        fold_total = sum(fold.validation_totals.values())
        fold_accuracy = CORE.routed_accuracy(
            mapping, fold.validation_correct, fold.validation_totals
        )
        weighted_correct += fold_accuracy * fold_total
        denominator += fold_total
    return weighted_correct / denominator, min(active_counts), max(active_counts)


def source_rank(row: dict[str, Any], routers: tuple[str, ...]) -> tuple[Any, ...]:
    deltas = [row[f"{router}_source_cv_delta_full_pp"] for router in routers]
    accuracies = [row[f"{router}_source_cv_accuracy"] for router in routers]
    return (
        min(deltas),
        sum(deltas) / len(deltas),
        sum(accuracies) / len(accuracies),
        -row["size"],
        row["models"],
    )


def router_source_rank(row: dict[str, Any], router: str) -> tuple[Any, ...]:
    return (
        row[f"{router}_source_cv_delta_full_pp"],
        row[f"{router}_source_cv_accuracy"],
        -row["size"],
        row["models"],
    )


def evaluate_target_candidate(
    row: dict[str, Any],
    routers: dict[str, RouterData],
    full_target_accuracy: dict[str, dict[str, float]],
    target_best_single: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    subset = tuple(row["models"].split(";"))
    result: dict[str, Any] = {
        "size": row["size"],
        "models": row["models"],
        "selection_uses_target_labels": False,
    }
    deltas: list[float] = []
    for router_name, router in routers.items():
        mapping = router.target_mapping(subset)
        result[f"{router_name}_mapping"] = mapping
        result[f"{router_name}_active_models"] = sorted(set(mapping.values()))
        for dataset, target in router.targets.items():
            accuracy = CORE.routed_accuracy(mapping, target["correct"], target["totals"])
            delta_full = 100 * (accuracy - full_target_accuracy[router_name][dataset])
            delta_best = 100 * (
                accuracy - target_best_single[router_name][dataset]["accuracy"]
            )
            result[f"{router_name}_{dataset}"] = {
                "accuracy": accuracy,
                "full_pool_accuracy": full_target_accuracy[router_name][dataset],
                "delta_full_pp": delta_full,
                "target_oracle_best_single": target_best_single[router_name][dataset],
                "delta_target_oracle_best_single_pp": delta_best,
            }
            deltas.append(delta_full)
    result["mean_delta_full_pp"] = sum(deltas) / len(deltas)
    result["worst_delta_full_pp"] = min(deltas)
    result["positive_cells_vs_full"] = sum(delta > 1e-12 for delta in deltas)
    result["target_cell_count"] = len(deltas)
    return result


def enumerate_source_subsets(
    models: list[str], routers: dict[str, RouterData]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    router_names = tuple(routers)
    full = tuple(models)
    full_cv = {
        router_name: cross_validated_router(full, router)[0]
        for router_name, router in routers.items()
    }
    rows: list[dict[str, Any]] = []
    for size in range(1, len(models) + 1):
        for subset in combinations(models, size):
            row: dict[str, Any] = {"size": size, "models": ";".join(subset)}
            union: set[str] = set()
            for router_name, router in routers.items():
                accuracy, min_cv_active, max_cv_active = cross_validated_router(subset, router)
                # Candidate eligibility must remain target-blind.  In
                # particular, do not let unseen target clusters turn an
                # otherwise single-expert source mapping into a multi-expert
                # candidate through the fallback path.
                final_mapping = CORE.selected_model_by_cluster(
                    subset, router.source_correct, router.source_totals
                )
                final_active = sorted(set(final_mapping.values()))
                union.update(final_active)
                row[f"{router_name}_source_cv_accuracy"] = accuracy
                row[f"{router_name}_source_cv_delta_full_pp"] = 100 * (
                    accuracy - full_cv[router_name]
                )
                row[f"{router_name}_cv_min_active_count"] = min_cv_active
                row[f"{router_name}_cv_max_active_count"] = max_cv_active
                row[f"{router_name}_final_source_active_count"] = len(final_active)
                row[f"{router_name}_final_source_active_models"] = ";".join(final_active)
                row[f"{router_name}_final_source_mapping"] = json.dumps(
                    final_mapping, ensure_ascii=False, sort_keys=True
                )
            deltas = [row[f"{name}_source_cv_delta_full_pp"] for name in router_names]
            row["source_cv_mean_delta_full_pp"] = sum(deltas) / len(deltas)
            row["source_cv_worst_delta_full_pp"] = min(deltas)
            row["source_cv_positive_router_count"] = sum(delta > 1e-12 for delta in deltas)
            row["final_active_union_count"] = len(union)
            rows.append(row)
    return rows, full_cv


def target_baselines(
    models: list[str], routers: dict[str, RouterData]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, Any]]]]:
    full = tuple(models)
    full_accuracy: dict[str, dict[str, float]] = {}
    best_single: dict[str, dict[str, dict[str, Any]]] = {}
    for router_name, router in routers.items():
        full_mapping = router.target_mapping(full)
        full_accuracy[router_name] = {}
        best_single[router_name] = {}
        for dataset, target in router.targets.items():
            full_accuracy[router_name][dataset] = CORE.routed_accuracy(
                full_mapping, target["correct"], target["totals"]
            )
            accuracy, model = max(
                (
                    CORE.model_accuracy(candidate, target["correct"], target["totals"]),
                    candidate,
                )
                for candidate in models
            )
            best_single[router_name][dataset] = {"model": model, "accuracy": accuracy}
    return full_accuracy, best_single


def load_text_data(
    requested_folds: int, repeats: int, seed: int
) -> tuple[list[str], dict[str, RouterData], dict[str, Any]]:
    models = CORE.available_models(CORE.DEFAULT_VIEW)
    datasets = ("bbh", "gpqa", "gaokao")
    matrices = {
        dataset: CORE.load_matrix(CORE.DEFAULT_VIEW, dataset, models)
        for dataset in ("mmlu_source", *datasets)
    }
    routers: dict[str, RouterData] = {}
    audit: dict[str, Any] = {}
    for index, spec in enumerate(CORE.ROUTERS):
        source_rows = CORE.read_rows(spec.source_routes)
        source_correct, source_totals, _, source_audit = CORE.build_cluster_stats(
            source_rows, matrices["mmlu_source"], "mmlu_source"
        )
        folds, fold_audit = build_repeated_fold_stats(
            source_rows,
            matrices["mmlu_source"],
            "mmlu_source",
            requested_folds,
            repeats,
            seed + index,
        )
        targets: dict[str, dict[str, Any]] = {}
        for dataset in datasets:
            correct, totals, _, target_audit = CORE.build_cluster_stats(
                CORE.read_rows(spec.targets[dataset]), matrices[dataset], dataset
            )
            targets[dataset] = {
                "correct": correct,
                "totals": totals,
                "audit": target_audit,
            }

        target_clusters = {
            cluster for target in targets.values() for cluster in target["totals"]
        }

        def make_mapping(
            subset: tuple[str, ...],
            source_correct: dict[str, dict[str, float]] = source_correct,
            source_totals: dict[str, float] = source_totals,
            target_clusters: set[str] = target_clusters,
        ) -> dict[str, str]:
            return mapping_with_fallback(
                subset, source_correct, source_totals, target_clusters
            )

        routers[spec.name] = RouterData(
            name=spec.name,
            source_correct=source_correct,
            source_totals=source_totals,
            folds=folds,
            targets=targets,
            target_mapping=make_mapping,
        )
        audit[spec.name] = {"source": source_audit, "cross_validation": fold_audit}
    return models, routers, audit


def load_multimodal_data(
    requested_folds: int, repeats: int, seed: int
) -> tuple[list[str], dict[str, RouterData], dict[str, Any]]:
    improve2 = load_module(
        "benchcoe_source_only_improve2",
        ROOT / "bench_coe/improve2_capability_routing_experiments.py",
    )
    models = sorted(entry.name for entry in (MM.VIEW / "cmmmu/val").iterdir() if entry.is_dir())
    datasets = ("mathvista", "mmmu_pro")
    source_rows_by_model = improve2.load_gaokao_mm_predictions(MM.VIEW / "gaokao_mm")
    source_matrix = {
        model: {
            rid: bool(row.get("is_correct", False))
            for rid, row in source_rows_by_model[model].items()
        }
        for model in models
    }
    target_matrices = {
        dataset: MM.target_matrix(dataset, models)
        for dataset in (*datasets, "cmmmu")
    }
    routers: dict[str, RouterData] = {}
    audit: dict[str, Any] = {}
    for index, spec in enumerate(MM.SPECS):
        normalized_source = MM.normalize_routes(spec.source, spec.mode)
        source_correct, source_totals, _, source_audit = CORE.build_cluster_stats(
            normalized_source, source_matrix, "gaokao"
        )
        folds, fold_audit = build_repeated_fold_stats(
            normalized_source,
            source_matrix,
            "gaokao",
            requested_folds,
            repeats,
            seed + index,
        )
        targets: dict[str, dict[str, Any]] = {}
        for dataset in datasets:
            normalized_target = MM.normalize_routes(spec.targets[dataset], spec.mode)
            correct, totals, _, target_audit = CORE.build_cluster_stats(
                normalized_target, target_matrices[dataset], dataset
            )
            if target_audit["route_rows_evaluated"] != target_audit["route_rows_total"]:
                raise RuntimeError(f"Incomplete target alignment: {spec.name}/{dataset}")
            targets[dataset] = {
                "correct": correct,
                "totals": totals,
                "audit": target_audit,
            }
        if spec.name == "qwen3vl_subject":
            dataset = "cmmmu"
            cmmmu_routes = MM.normalize_routes(
                ROOT
                / "outputs/bench_coe/cmmmu_qwen3vl_gaokao_mm_router_front4/test_predictions.json",
                "subject",
            )
            correct, totals, _, target_audit = CORE.build_cluster_stats(
                cmmmu_routes, target_matrices[dataset], dataset
            )
            if (
                target_audit["route_rows_evaluated"]
                != target_audit["route_rows_total"]
                or target_audit["route_rows_evaluated"] != 900
            ):
                raise RuntimeError("Incomplete Qwen3-VL Subject/CMMMU-val alignment")
            targets[dataset] = {
                "correct": correct,
                "totals": totals,
                "audit": target_audit,
                "scope": "Qwen3-VL Subject only; aligned CMMMU val 900 supplement",
            }
        target_clusters = {
            cluster for target in targets.values() for cluster in target["totals"]
        }

        def make_mapping(
            subset: tuple[str, ...],
            source_correct: dict[str, dict[str, float]] = source_correct,
            source_totals: dict[str, float] = source_totals,
            target_clusters: set[str] = target_clusters,
        ) -> dict[str, str]:
            return mapping_with_fallback(
                subset, source_correct, source_totals, target_clusters
            )

        routers[spec.name] = RouterData(
            name=spec.name,
            source_correct=source_correct,
            source_totals=source_totals,
            folds=folds,
            targets=targets,
            target_mapping=make_mapping,
        )
        audit[spec.name] = {"source": source_audit, "cross_validation": fold_audit}
    return models, routers, audit


def select_and_evaluate(
    label: str,
    models: list[str],
    routers: dict[str, RouterData],
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows, full_cv = enumerate_source_subsets(models, routers)
    router_names = tuple(routers)
    unrestricted = max(rows, key=lambda row: source_rank(row, router_names))
    best_single = max(
        (row for row in rows if row["size"] == 1),
        key=lambda row: source_rank(row, router_names),
    )
    per_router_unrestricted = {
        router: max(rows, key=lambda row: router_source_rank(row, router))
        for router in router_names
    }

    if label == "language":
        strict_multi = [
            row
            for row in rows
            if all(
                row[f"{router}_final_source_active_count"] >= 2
                and row[f"{router}_cv_min_active_count"] >= 2
                for router in router_names
            )
        ]
        per_router_multi = {
            router: [
                row
                for row in rows
                if row[f"{router}_final_source_active_count"] >= 2
                and row[f"{router}_cv_min_active_count"] >= 2
            ]
            for router in router_names
        }
        strict_definition = "Query and Subject each activate >=2 experts in every CV fold and final mapping"
    else:
        subject_names = tuple(name for name in router_names if name.endswith("_subject"))
        strict_multi = [
            row
            for row in rows
            if all(
                row[f"{router}_final_source_active_count"] >= 2
                and row[f"{router}_cv_min_active_count"] >= 2
                for router in subject_names
            )
        ]
        per_router_multi = {
            router: [
                row
                for row in rows
                if row[f"{router}_final_source_active_count"] >= 2
                and row[f"{router}_cv_min_active_count"] >= 2
            ]
            for router in router_names
        }
        strict_definition = "Both Subject backbones activate >=2 experts in every CV fold and final mapping"
    if not strict_multi:
        raise RuntimeError(f"No strict multiexpert candidates for {label}")
    strict_best = max(strict_multi, key=lambda row: source_rank(row, router_names))
    per_router_best_multi = {
        router: max(eligible, key=lambda row: router_source_rank(row, router))
        for router, eligible in per_router_multi.items()
        if eligible
    }

    full_target_accuracy, target_best_single = target_baselines(models, routers)
    selected_rows: dict[str, dict[str, Any]] = {
        "source_cv_unrestricted": unrestricted,
        "source_cv_best_single": best_single,
        "source_cv_strict_multiexpert": strict_best,
    }
    selected_rows.update(
        {f"source_cv_{router}_unrestricted": row for router, row in per_router_unrestricted.items()}
    )
    selected_rows.update(
        {f"source_cv_{router}_multiexpert": row for router, row in per_router_best_multi.items()}
    )
    target_results = {
        name: evaluate_target_candidate(
            row, routers, full_target_accuracy, target_best_single
        )
        for name, row in selected_rows.items()
    }

    write_csv(output_dir / "all_source_cv_subsets.csv", rows)
    summary = {
        "status": "completed",
        "selection_protocol": {
            "target_answer_labels_used_for_subset_selection": False,
            "target_answer_labels_used_for_cluster_to_expert_mapping": False,
            "target_router_outputs_used_for_selection": False,
            "target_router_outputs_used_after_freezing": (
                "Only to apply the source-fitted mapping and source-fitted fallback"
            ),
            "selection_data": "source-domain per-item correctness only",
            "selection_method": "repeated deterministic stratified source-domain cross-validation",
            "rank_order": "worst router delta vs full pool, mean delta, macro accuracy, smaller pool",
            "target_evaluation_timing": "after subset selection and final source mapping are frozen",
        },
        "models": models,
        "num_models": len(models),
        "num_nonempty_subsets": len(rows),
        "source_audit": audit,
        "full_pool_source_cv_accuracy": full_cv,
        "strict_multiexpert": {
            "definition": strict_definition,
            "eligible_subsets": len(strict_multi),
        },
        "selected_source_rows": selected_rows,
        "full_pool_target_accuracy": full_target_accuracy,
        "target_oracle_best_single": target_best_single,
        "frozen_target_evaluation": target_results,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def selected_line(summary: dict[str, Any], name: str) -> str:
    row = summary["selected_source_rows"][name]
    result = summary["frozen_target_evaluation"][name]
    return (
        f"- `{name}`: `{row['models']}`; source-CV mean delta "
        f"{row['source_cv_mean_delta_full_pp']:+.2f} pp, worst "
        f"{row['source_cv_worst_delta_full_pp']:+.2f} pp; frozen target mean delta "
        f"{result['mean_delta_full_pp']:+.2f} pp, worst "
        f"{result['worst_delta_full_pp']:+.2f} pp, positive cells "
        f"{result['positive_cells_vs_full']}/{result['target_cell_count']}."
    )


def write_report(output_dir: Path, language: dict[str, Any], multimodal: dict[str, Any]) -> None:
    lines = [
        "# Bench-CoE 源域选择、目标域冻结评估报告",
        "",
        "## 协议",
        "",
        "本次搜索彻底隔离模型组合选择与目标集标签。所有候选只按源域分层交叉验证成绩排序；组合冻结后，才在跨数据集目标域上评估一次。目标集成绩不参与组合选择、簇到专家映射或并列决胜。",
        "",
        "这比上一轮后验穷举更严格，但仍属于对既有目标集的重复分析。只有未来新增且从未查看标签的数据集，才能作为最终确认集。",
        "",
        "## 语言 14 模型池",
        "",
        f"精确遍历全部 {language['num_nonempty_subsets']:,} 个非空子集。",
        "",
        selected_line(language, "source_cv_unrestricted"),
        selected_line(language, "source_cv_best_single"),
        selected_line(language, "source_cv_strict_multiexpert"),
        "",
        "## 多模态 8 模型池",
        "",
        f"精确遍历全部 {multimodal['num_nonempty_subsets']:,} 个非空子集。",
        "",
        selected_line(multimodal, "source_cv_unrestricted"),
        selected_line(multimodal, "source_cv_best_single"),
        selected_line(multimodal, "source_cv_strict_multiexpert"),
        selected_line(multimodal, "source_cv_qwen3vl_subject_multiexpert"),
        selected_line(multimodal, "source_cv_tinyllava_subject_multiexpert"),
        "",
        "## 解释限制",
        "",
        "源域交叉验证选出的组合若在目标域下降，说明仅凭源域排行榜挑模型仍不足以保证跨分布泛化；该负结果不能再通过查看目标集后重新挑组合来改写。Query 路由标签塌缩也不会因筛选模型而自动恢复，仍需按冻结专家池重新制标签并训练/校准 Query 分类器。",
        "",
    ]
    (output_dir / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    language_models, language_routers, language_audit = load_text_data(
        args.folds, args.repeats, args.seed
    )
    language = select_and_evaluate(
        "language",
        language_models,
        language_routers,
        language_audit,
        args.output_dir / "language_14_model_pool",
    )
    multimodal_models, multimodal_routers, multimodal_audit = load_multimodal_data(
        args.folds, args.repeats, args.seed + 1000
    )
    multimodal = select_and_evaluate(
        "multimodal",
        multimodal_models,
        multimodal_routers,
        multimodal_audit,
        args.output_dir / "multimodal_8_model_pool",
    )
    top_level = {
        "status": "completed",
        "language_summary": "language_14_model_pool/summary.json",
        "multimodal_summary": "multimodal_8_model_pool/summary.json",
        "selection_is_target_blind": True,
        "folds_requested": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    write_json(args.output_dir / "manifest.json", top_level)
    write_report(args.output_dir, language, multimodal)
    print(json.dumps(top_level, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
