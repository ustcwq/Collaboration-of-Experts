#!/usr/bin/env python3
"""Paired uncertainty analysis for the selected multimodal Subject subset."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "outputs/bench_coe/scale_transfer_views_20260802/vision"
OUT = ROOT / "outputs/bench_coe/model_subset_search_20260813/multimodal_8_model_pool"
SELECTED = ("InternVL3_5-14B", "InternVL3_5-8B")
BOOTSTRAP_ITERS = 20_000
SEED = 20260813


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paired_stats(candidate: np.ndarray, baseline: np.ndarray, seed: int) -> dict[str, Any]:
    if candidate.shape != baseline.shape:
        raise ValueError("Paired arrays must have identical shape.")
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    n = len(delta)
    chunk = 1000
    samples: list[np.ndarray] = []
    for start in range(0, BOOTSTRAP_ITERS, chunk):
        take = min(chunk, BOOTSTRAP_ITERS - start)
        indices = rng.integers(0, n, size=(take, n))
        samples.append(delta[indices].mean(axis=1))
    bootstrap = np.concatenate(samples)
    wins = int(np.sum((candidate == 1) & (baseline == 0)))
    losses = int(np.sum((candidate == 0) & (baseline == 1)))
    discordant = wins + losses
    p_value = float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "examples": n,
        "candidate_accuracy": float(candidate.mean()),
        "baseline_accuracy": float(baseline.mean()),
        "delta_pp": float(100 * delta.mean()),
        "bootstrap_95_ci_pp": [
            float(100 * np.quantile(bootstrap, 0.025)),
            float(100 * np.quantile(bootstrap, 0.975)),
        ],
        "candidate_only_correct": wins,
        "baseline_only_correct": losses,
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    core = load_module("subset_core_stats", ROOT / "tools/search_benchcoe_model_subsets.py")
    mm = load_module("subset_mm_stats", ROOT / "tools/search_benchcoe_multimodal_subsets.py")
    improve2 = load_module(
        "improve2_loader_stats", ROOT / "bench_coe/improve2_capability_routing_experiments.py"
    )
    summary = mm.read_json(OUT / "summary.json")
    models = list(summary["models"])
    source_rows = improve2.load_gaokao_mm_predictions(VIEW / "gaokao_mm")
    source_matrix = {
        model: {rid: bool(row.get("is_correct", False)) for rid, row in source_rows[model].items()}
        for model in models
    }
    target_matrices = {
        dataset: mm.target_matrix(dataset, models) for dataset in ("mathvista", "mmmu_pro", "cmmmu")
    }

    rows_out: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    subject_specs = [spec for spec in mm.SPECS if spec.mode == "subject"]
    for spec_idx, spec in enumerate(subject_specs):
        normalized_source = mm.normalize_routes(spec.source, spec.mode)
        source_correct, source_totals, _, _ = core.build_cluster_stats(
            normalized_source, source_matrix, "gaokao"
        )
        selected_mapping = core.selected_model_by_cluster(SELECTED, source_correct, source_totals)
        full_mapping = core.selected_model_by_cluster(tuple(models), source_correct, source_totals)
        details[spec.name] = {
            "selected_mapping": selected_mapping,
            "full_mapping": full_mapping,
            "datasets": {},
        }
        for dataset_idx, dataset in enumerate(("mathvista", "mmmu_pro")):
            route_rows = mm.normalize_routes(spec.targets[dataset], spec.mode)
            ids = [str(row["id"]) for row in route_rows]
            clusters = [core.cluster_value(row) for row in route_rows]
            matrix = target_matrices[dataset]
            selected_y = np.asarray(
                [int(matrix[selected_mapping[cluster]][rid]) for rid, cluster in zip(ids, clusters)],
                dtype=np.int8,
            )
            full_y = np.asarray(
                [int(matrix[full_mapping[cluster]][rid]) for rid, cluster in zip(ids, clusters)],
                dtype=np.int8,
            )
            best_single = summary["global_best_single"][spec.name][dataset]["model"]
            best_y = np.asarray([int(matrix[best_single][rid]) for rid in ids], dtype=np.int8)
            comparisons = {
                "full_pool_router": paired_stats(
                    selected_y, full_y, SEED + 100 * spec_idx + 10 * dataset_idx
                ),
                "global_best_single": paired_stats(
                    selected_y, best_y, SEED + 100 * spec_idx + 10 * dataset_idx + 1
                ),
            }
            details[spec.name]["datasets"][dataset] = {
                "best_single_model": best_single,
                "comparisons": comparisons,
            }
            for baseline, values in comparisons.items():
                rows_out.append(
                    {
                        "router": spec.name,
                        "dataset": dataset,
                        "baseline": baseline,
                        "baseline_model": best_single if baseline == "global_best_single" else "full_pool_router",
                        **values,
                        "ci_low_pp": values["bootstrap_95_ci_pp"][0],
                        "ci_high_pp": values["bootstrap_95_ci_pp"][1],
                    }
                )

    # Aligned CMMMU-val supplement for Qwen3-VL Subject only.
    spec = next(spec for spec in subject_specs if spec.name == "qwen3vl_subject")
    normalized_source = mm.normalize_routes(spec.source, spec.mode)
    source_correct, source_totals, _, _ = core.build_cluster_stats(
        normalized_source, source_matrix, "gaokao"
    )
    selected_mapping = core.selected_model_by_cluster(SELECTED, source_correct, source_totals)
    full_mapping = core.selected_model_by_cluster(tuple(models), source_correct, source_totals)
    route_rows = mm.normalize_routes(
        ROOT / "outputs/bench_coe/cmmmu_qwen3vl_gaokao_mm_router_front4/test_predictions.json",
        "subject",
    )
    ids = [str(row["id"]) for row in route_rows]
    clusters = [core.cluster_value(row) for row in route_rows]
    matrix = target_matrices["cmmmu"]
    selected_y = np.asarray(
        [int(matrix[selected_mapping[cluster]][rid]) for rid, cluster in zip(ids, clusters)], dtype=np.int8
    )
    full_y = np.asarray(
        [int(matrix[full_mapping[cluster]][rid]) for rid, cluster in zip(ids, clusters)], dtype=np.int8
    )
    best_single = summary["cmmmu_qwen3vl_subject_supplement"]["global_best_single_model"]
    best_y = np.asarray([int(matrix[best_single][rid]) for rid in ids], dtype=np.int8)
    cmmmu_comparisons = {
        "full_pool_router": paired_stats(selected_y, full_y, SEED + 999),
        "global_best_single": paired_stats(selected_y, best_y, SEED + 1000),
    }
    details["qwen3vl_subject"]["datasets"]["cmmmu"] = {
        "best_single_model": best_single,
        "comparisons": cmmmu_comparisons,
    }
    for baseline, values in cmmmu_comparisons.items():
        rows_out.append(
            {
                "router": "qwen3vl_subject",
                "dataset": "cmmmu",
                "baseline": baseline,
                "baseline_model": best_single if baseline == "global_best_single" else "full_pool_router",
                **values,
                "ci_low_pp": values["bootstrap_95_ci_pp"][0],
                "ci_high_pp": values["bootstrap_95_ci_pp"][1],
            }
        )

    payload = {
        "status": "completed",
        "selected_models": list(SELECTED),
        "bootstrap_iterations": BOOTSTRAP_ITERS,
        "seed": SEED,
        "multiple_testing_adjustment": None,
        "details": details,
    }
    with (OUT / "paired_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    fields = sorted({key for row in rows_out for key in row if key != "bootstrap_95_ci_pp"})
    with (OUT / "paired_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
