#!/usr/bin/env python3
"""Exhaustive fixed-subset search over the eight scale-expanded VLMs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "outputs/bench_coe/scale_transfer_views_20260802/vision"
DEFAULT_OUTPUT = ROOT / "outputs/bench_coe/model_subset_search_20260813/multimodal_8_model_pool"


@dataclass(frozen=True)
class RouteSpec:
    name: str
    mode: str
    source: Path
    targets: dict[str, Path]


SPECS = (
    RouteSpec(
        "qwen3vl_query",
        "query",
        ROOT / "outputs/bench_coe/qwen3vl_gaokao_query_strict_holdout/predictions.json",
        {
            "cmmmu": ROOT / "outputs/bench_coe/qwen3vl_gaokao_query_strict_on_cmmmu/predictions.json",
            "mathvista": ROOT / "outputs/bench_coe/qwen3vl_gaokao_query_strict_on_mathvista/predictions.json",
            "mmmu_pro": ROOT / "outputs/bench_coe/qwen3vl_gaokao_query_strict_on_mmmu_pro/predictions.json",
        },
    ),
    RouteSpec(
        "qwen3vl_subject",
        "subject",
        ROOT / "outputs/bench_coe/qwen3vl-gaokao-mm-router-holdout/predictions.json",
        {
            "cmmmu": ROOT / "outputs/bench_coe/qwen3vl_gaokao_mm_router_on_cmmmu_dev/predictions.json",
            "mathvista": ROOT / "outputs/bench_coe/qwen3vl_gaokao_mm_router_on_mathvista_testmini/predictions.json",
            "mmmu_pro": ROOT / "outputs/bench_coe/qwen3vl_gaokao_mm_router_on_mmmu_pro_test/predictions.json",
        },
    ),
    RouteSpec(
        "tinyllava_query",
        "query",
        ROOT / "outputs/bench_coe/tinyllava_gaokao_query_strict_holdout/predictions.json",
        {
            "cmmmu": ROOT / "outputs/bench_coe/tinyllava_gaokao_query_strict_on_cmmmu/predictions.json",
            "mathvista": ROOT / "outputs/bench_coe/tinyllava_gaokao_query_strict_on_mathvista/predictions.json",
            "mmmu_pro": ROOT / "outputs/bench_coe/tinyllava_gaokao_query_strict_on_mmmu_pro/predictions.json",
        },
    ),
    RouteSpec(
        "tinyllava_subject",
        "subject",
        ROOT / "outputs/bench_coe/tinyllava-gaokao-mm-router-holdout/predictions.json",
        {
            "cmmmu": ROOT / "outputs/bench_coe/tinyllava_gaokao_mm_router_on_cmmmu_dev/predictions.json",
            "mathvista": ROOT / "outputs/bench_coe/tinyllava_gaokao_mm_router_on_mathvista_testmini/predictions.json",
            "mmmu_pro": ROOT / "outputs/bench_coe/tinyllava_gaokao_mm_router_on_mmmu_pro_test/predictions.json",
        },
    ),
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_routes(path: Path, mode: str) -> list[dict[str, Any]]:
    rows = read_json(path)
    if mode == "query":
        return rows
    return [{**row, "route_label": str(row["routed_subject"])} for row in rows]


def target_matrix(dataset: str, models: list[str]) -> dict[str, dict[str, bool]]:
    subdir = {
        "cmmmu": Path("cmmmu/val"),
        "mathvista": Path("mathvista/testmini"),
        "mmmu_pro": Path("mmmu_pro/standard_10_options/test"),
    }[dataset]
    return {
        model: {str(row["id"]): bool(row.get("is_correct", False)) for row in read_jsonl(VIEW / subdir / model / "predictions.jsonl")}
        for model in models
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank(row: dict[str, Any], specs: tuple[RouteSpec, ...], datasets: tuple[str, ...]) -> tuple[Any, ...]:
    deltas = [row[f"{spec.name}_{dataset}_delta_full_pp"] for spec in specs for dataset in datasets]
    return (min(deltas), sum(deltas) / len(deltas), sum(delta > 0 for delta in deltas), -row["size"], row["models"])


def main() -> None:
    args = parse_args()
    core = load_module("benchcoe_subset_core_mm", ROOT / "tools/search_benchcoe_model_subsets.py")
    improve2 = load_module("benchcoe_improve2_loader", ROOT / "bench_coe/improve2_capability_routing_experiments.py")
    models = sorted(entry.name for entry in (VIEW / "cmmmu/val").iterdir() if entry.is_dir())
    # CMMMU router outputs use dev IDs while expanded model results use val IDs.
    # Keep only targets with exact row alignment in the joint search.
    datasets = ("mathvista", "mmmu_pro")

    source_rows = improve2.load_gaokao_mm_predictions(VIEW / "gaokao_mm")
    source_matrix = {
        model: {rid: bool(row.get("is_correct", False)) for rid, row in source_rows[model].items()}
        for model in models
    }
    targets = {dataset: target_matrix(dataset, models) for dataset in datasets}
    stats: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    cluster_audit: dict[str, dict[str, Any]] = {}
    for spec in SPECS:
        normalized_source = normalize_routes(spec.source, spec.mode)
        source_correct, source_totals, _, source_audit = core.build_cluster_stats(
            normalized_source, source_matrix, "gaokao"
        )
        source_counts = Counter(core.cluster_value(row) for row in normalized_source)
        cluster_audit[spec.name] = {
            "source": dict(sorted(source_counts.items())),
            "targets": {},
        }
        target_stats: dict[str, Any] = {}
        coverage.append({"router": spec.name, "dataset": "gaokao_mm_holdout", **source_audit})
        for dataset in datasets:
            path = spec.targets[dataset]
            normalized_target = normalize_routes(path, spec.mode)
            correct, totals, _, audit = core.build_cluster_stats(
                normalized_target, targets[dataset], dataset
            )
            if audit["route_rows_evaluated"] != audit["route_rows_total"]:
                raise RuntimeError(f"Incomplete {spec.name}/{dataset} alignment: {audit}")
            target_stats[dataset] = {"correct": correct, "totals": totals}
            target_counts = Counter(core.cluster_value(row) for row in normalized_target)
            unseen = sorted(set(target_counts).difference(source_counts))
            cluster_audit[spec.name]["targets"][dataset] = {
                "counts": dict(sorted(target_counts.items())),
                "unseen_source_clusters": unseen,
                "fallback_rows": sum(target_counts[cluster] for cluster in unseen),
                "fallback_rate": sum(target_counts[cluster] for cluster in unseen) / len(normalized_target),
            }
            coverage.append({"router": spec.name, "dataset": dataset, **audit})
        stats[spec.name] = {
            "source_correct": source_correct,
            "source_totals": source_totals,
            "targets": target_stats,
        }

    def subset_mapping(spec_name: str, subset: tuple[str, ...]) -> dict[str, str]:
        item = stats[spec_name]
        mapping = core.selected_model_by_cluster(
            subset, item["source_correct"], item["source_totals"]
        )
        source_denominator = sum(item["source_totals"].values())
        fallback = max(
            subset,
            key=lambda model: (
                sum(item["source_correct"][cluster][model] for cluster in item["source_totals"])
                / source_denominator,
                model,
            ),
        )
        target_clusters = {
            cluster
            for target in item["targets"].values()
            for cluster in target["totals"]
        }
        for cluster in target_clusters:
            mapping.setdefault(cluster, fallback)
        return mapping

    full = tuple(models)
    full_accuracy: dict[str, dict[str, float]] = {}
    global_best: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in SPECS:
        item = stats[spec.name]
        mapping = subset_mapping(spec.name, full)
        full_accuracy[spec.name] = {}
        global_best[spec.name] = {}
        for dataset in datasets:
            target = item["targets"][dataset]
            full_accuracy[spec.name][dataset] = core.routed_accuracy(mapping, target["correct"], target["totals"])
            accuracy, model = max(
                (core.model_accuracy(candidate, target["correct"], target["totals"]), candidate)
                for candidate in models
            )
            global_best[spec.name][dataset] = {"model": model, "accuracy": accuracy}

    rows: list[dict[str, Any]] = []
    for size in range(1, len(models) + 1):
        for subset in combinations(models, size):
            row: dict[str, Any] = {"size": size, "models": ";".join(subset)}
            deltas: list[float] = []
            global_deltas: list[float] = []
            for spec in SPECS:
                item = stats[spec.name]
                mapping = subset_mapping(spec.name, subset)
                active = sorted(set(mapping.values()))
                row[f"{spec.name}_mapping"] = json.dumps(mapping, ensure_ascii=False, sort_keys=True)
                row[f"{spec.name}_active_count"] = len(active)
                row[f"{spec.name}_active_models"] = ";".join(active)
                for dataset in datasets:
                    target = item["targets"][dataset]
                    accuracy = core.routed_accuracy(mapping, target["correct"], target["totals"])
                    delta = 100 * (accuracy - full_accuracy[spec.name][dataset])
                    global_delta = 100 * (accuracy - global_best[spec.name][dataset]["accuracy"])
                    row[f"{spec.name}_{dataset}_accuracy"] = accuracy
                    row[f"{spec.name}_{dataset}_delta_full_pp"] = delta
                    row[f"{spec.name}_{dataset}_delta_global_best_pp"] = global_delta
                    deltas.append(delta)
                    global_deltas.append(global_delta)
            row["mean_delta_full_pp"] = sum(deltas) / len(deltas)
            row["worst_delta_full_pp"] = min(deltas)
            row["positive_cells_vs_full"] = sum(delta > 1e-12 for delta in deltas)
            row["mean_delta_global_best_pp"] = sum(global_deltas) / len(global_deltas)
            row["worst_delta_global_best_pp"] = min(global_deltas)
            rows.append(row)

    best = max(rows, key=lambda row: rank(row, SPECS, datasets))
    subject_multi = [
        row for row in rows
        if row["qwen3vl_subject_active_count"] >= 2 and row["tinyllava_subject_active_count"] >= 2
    ]
    best_subject_multi = max(subject_multi, key=lambda row: rank(row, SPECS, datasets))
    lodo: list[dict[str, Any]] = []
    for held_out in datasets:
        train_datasets = tuple(dataset for dataset in datasets if dataset != held_out)
        selected = max(rows, key=lambda row: rank(row, SPECS, train_datasets))
        for spec in SPECS:
            lodo.append(
                {
                    "held_out": held_out,
                    "router": spec.name,
                    "selected_models": selected["models"],
                    "accuracy": selected[f"{spec.name}_{held_out}_accuracy"],
                    "delta_full_pp": selected[f"{spec.name}_{held_out}_delta_full_pp"],
                    "delta_global_best_pp": selected[f"{spec.name}_{held_out}_delta_global_best_pp"],
                }
            )

    # Supplementary CMMMU-val analysis. Only Qwen3-VL Subject has a saved
    # 900-row route file aligned to the expanded-model val results.
    cmmmu_routes_path = ROOT / "outputs/bench_coe/cmmmu_qwen3vl_gaokao_mm_router_front4/test_predictions.json"
    cmmmu_matrix = target_matrix("cmmmu", models)
    cmmmu_routes = normalize_routes(cmmmu_routes_path, "subject")
    cmmmu_correct, cmmmu_totals, _, cmmmu_audit = core.build_cluster_stats(
        cmmmu_routes, cmmmu_matrix, "cmmmu"
    )
    if cmmmu_audit["route_rows_evaluated"] != 900:
        raise RuntimeError(f"Unexpected CMMMU supplement coverage: {cmmmu_audit}")
    source_item = stats["qwen3vl_subject"]

    def cmmmu_mapping(subset: tuple[str, ...]) -> dict[str, str]:
        mapping = core.selected_model_by_cluster(
            subset, source_item["source_correct"], source_item["source_totals"]
        )
        denominator = sum(source_item["source_totals"].values())
        fallback = max(
            subset,
            key=lambda model: (
                sum(source_item["source_correct"][cluster][model] for cluster in source_item["source_totals"])
                / denominator,
                model,
            ),
        )
        for cluster in cmmmu_totals:
            mapping.setdefault(cluster, fallback)
        return mapping

    cmmmu_full_mapping = cmmmu_mapping(full)
    cmmmu_full_accuracy = core.routed_accuracy(cmmmu_full_mapping, cmmmu_correct, cmmmu_totals)
    cmmmu_best_single_accuracy, cmmmu_best_single_model = max(
        (core.model_accuracy(model, cmmmu_correct, cmmmu_totals), model) for model in models
    )
    cmmmu_rows: list[dict[str, Any]] = []
    for row in rows:
        subset = tuple(row["models"].split(";"))
        mapping = cmmmu_mapping(subset)
        accuracy = core.routed_accuracy(mapping, cmmmu_correct, cmmmu_totals)
        cmmmu_rows.append(
            {
                "size": row["size"],
                "models": row["models"],
                "active_count": len(set(mapping.values())),
                "active_models": ";".join(sorted(set(mapping.values()))),
                "mapping": json.dumps(mapping, ensure_ascii=False, sort_keys=True),
                "accuracy": accuracy,
                "delta_full_pp": 100 * (accuracy - cmmmu_full_accuracy),
                "delta_global_best_pp": 100 * (accuracy - cmmmu_best_single_accuracy),
            }
        )
    cmmmu_best = max(
        cmmmu_rows,
        key=lambda row: (row["delta_full_pp"], row["accuracy"], -row["size"], row["models"]),
    )
    cmmmu_multi = [row for row in cmmmu_rows if row["active_count"] >= 2]
    cmmmu_best_multi = max(
        cmmmu_multi,
        key=lambda row: (row["delta_full_pp"], row["accuracy"], -row["size"], row["models"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_255_subsets.csv", rows)
    write_csv(args.output_dir / "leave_one_dataset_out.csv", lodo)
    write_csv(args.output_dir / "cmmmu_qwen3vl_subject_255_subsets.csv", cmmmu_rows)
    summary = {
        "status": "completed",
        "models": models,
        "source": "GAOKAO-MM 129-example validation holdout; official project binary scoring",
        "targets": list(datasets),
        "routers": [spec.name for spec in SPECS],
        "coverage": coverage,
        "cluster_audit": cluster_audit,
        "full_pool_accuracy": full_accuracy,
        "global_best_single": global_best,
        "best": best,
        "subject_multiexpert": {
            "definition": "both Qwen3-VL and TinyLLaVA Subject routers activate >=2 models",
            "eligible_subsets": len(subject_multi),
            "positive_all_8": sum(row["positive_cells_vs_full"] == 8 for row in subject_multi),
            "best": best_subject_multi,
        },
        "query_collapse": {
            "qwen3vl_source_cluster_counts": {"0": 129},
            "tinyllava_source_cluster_counts": {"0": 126, "1": 3},
        },
        "cmmmu_qwen3vl_subject_supplement": {
            "scope": "Qwen3-VL Subject only; CMMMU val 900; excluded from joint ranking",
            "coverage": cmmmu_audit,
            "full_pool_accuracy": cmmmu_full_accuracy,
            "global_best_single_model": cmmmu_best_single_model,
            "global_best_single_accuracy": cmmmu_best_single_accuracy,
            "best": cmmmu_best,
            "best_multiexpert": cmmmu_best_multi,
        },
        "leave_one_dataset_out": lodo,
    }
    write_json(args.output_dir / "summary.json", summary)
    report = f"""# Multimodal 8-model subset search

All 255 non-empty subsets were exhaustively evaluated using fixed saved Query/Subject partitions from two router backbones.
Partition-to-expert mappings use only the 129-example GAOKAO-MM validation holdout; aligned targets are MathVista (1,000) and MMMU-Pro (1,730).

## Result

- Unrestricted robust best: `{best['models']}`; mean delta vs full pool {best['mean_delta_full_pp']:+.2f} pp; worst {best['worst_delta_full_pp']:+.2f} pp; positive cells {best['positive_cells_vs_full']}/8.
- Subject-nondegenerate best: `{best_subject_multi['models']}`; mean delta {best_subject_multi['mean_delta_full_pp']:+.2f} pp; worst {best_subject_multi['worst_delta_full_pp']:+.2f} pp; positive cells {best_subject_multi['positive_cells_vs_full']}/8.
- Subject-nondegenerate subsets positive in all 8 cells: {sum(row['positive_cells_vs_full'] == 8 for row in subject_multi)}.

CMMMU is excluded from the joint search because saved Query/TinyLLaVA routes cover the 112-example dev split, whereas the eight expanded-model results cover the disjoint 900-example val split. Mixing them would be invalid.

As a separate aligned supplement, Qwen3-VL Subject on CMMMU val selects `{cmmmu_best_multi['models']}` as its best active multi-expert subset: {cmmmu_best_multi['accuracy']:.2%}, {cmmmu_best_multi['delta_full_pp']:+.2f} pp vs the full pool and {cmmmu_best_multi['delta_global_best_pp']:+.2f} pp vs the global best single model.

Qwen3-VL Query predicted one source cluster for all 129 holdout examples. TinyLLaVA Query predicted cluster 0 for 126/129 examples. Consequently model filtering cannot recover a genuinely diverse Query router without retraining or recalibrating the Query classifier.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
