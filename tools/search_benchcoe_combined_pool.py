#!/usr/bin/env python3
"""Search the combined legacy and scale-expanded language expert pool.

BBH and GPQA are the two targets with complete, protocol-compatible per-item
predictions for both pools.  Subsets of size 1--6 are enumerated exactly; larger
subsets are explored with a deterministic, mapping-signature-deduplicated beam.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/bench_coe/model_subset_search_20260813/combined_28_model_pool"


def load_core() -> Any:
    path = ROOT / "tools/search_benchcoe_model_subsets.py"
    spec = importlib.util.spec_from_file_location("benchcoe_subset_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exact-max-size", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=2048)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    core = load_core()
    pool_config = core.read_json(ROOT / "bench_coe/configs/expert_pools.json")
    legacy = sorted(
        entry["name"]
        for entry in pool_config["pools"]["language_7b_9b_specialists"]["models"]
        if not entry.get("exclude_reason")
    )
    expanded = core.available_models(core.DEFAULT_VIEW)
    models = legacy + expanded
    datasets = ("bbh", "gpqa")

    matrices: dict[str, dict[str, dict[str, bool]]] = {"mmlu_source": {}}
    for model in legacy:
        matrices["mmlu_source"][model] = core.load_mmlu_model(ROOT / "MMLU-Pro/results" / model)
    for model in expanded:
        matrices["mmlu_source"][model] = core.load_mmlu_model(
            core.DEFAULT_VIEW / "mmlu_test" / model
        )
    for dataset in datasets:
        matrices[dataset] = {}
        for model in legacy:
            matrices[dataset][model] = core.load_target_model(
                ROOT / "outputs/model_benchmarks/official_code_local_models" / dataset / model,
                dataset,
            )
        for model in expanded:
            matrices[dataset][model] = core.load_target_model(
                core.DEFAULT_VIEW / dataset / model, dataset
            )

    stats: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    for spec in core.ROUTERS:
        source_correct, source_totals, source_missing, source_audit = core.build_cluster_stats(
            core.read_rows(spec.source_routes), matrices["mmlu_source"], "mmlu_source"
        )
        targets: dict[str, Any] = {}
        coverage.append({"router": spec.name, "dataset": "mmlu_source", **source_audit})
        for dataset in datasets:
            correct, totals, missing, audit = core.build_cluster_stats(
                core.read_rows(spec.targets[dataset]), matrices[dataset], dataset
            )
            targets[dataset] = {"correct": correct, "totals": totals, "missing": missing}
            coverage.append({"router": spec.name, "dataset": dataset, **audit})
        stats[spec.name] = {
            "source_correct": source_correct,
            "source_totals": source_totals,
            "source_missing": source_missing,
            "targets": targets,
        }

    full_subset = tuple(models)
    full_accuracy: dict[str, dict[str, float]] = {}
    global_best: dict[str, dict[str, dict[str, Any]]] = {}
    for router, router_stats in stats.items():
        mapping = core.selected_model_by_cluster(
            full_subset, router_stats["source_correct"], router_stats["source_totals"]
        )
        full_accuracy[router] = {}
        global_best[router] = {}
        for dataset in datasets:
            target = router_stats["targets"][dataset]
            full_accuracy[router][dataset] = core.routed_accuracy(
                mapping, target["correct"], target["totals"]
            )
            accuracy, model = max(
                (
                    core.model_accuracy(candidate, target["correct"], target["totals"]),
                    candidate,
                )
                for candidate in models
            )
            global_best[router][dataset] = {"model": model, "accuracy": accuracy}

    def evaluate(subset: tuple[str, ...], phase: str) -> dict[str, Any]:
        row: dict[str, Any] = {"size": len(subset), "models": ";".join(subset), "phase": phase}
        all_deltas: list[float] = []
        all_global_deltas: list[float] = []
        active_union: set[str] = set()
        signatures: list[tuple[tuple[str, str], ...]] = []
        for router, router_stats in stats.items():
            mapping = core.selected_model_by_cluster(
                subset, router_stats["source_correct"], router_stats["source_totals"]
            )
            active = sorted(set(mapping.values()))
            active_union.update(active)
            signatures.append(tuple(sorted(mapping.items())))
            row[f"{router}_mapping"] = json.dumps(mapping, sort_keys=True, ensure_ascii=False)
            row[f"{router}_active_count"] = len(active)
            row[f"{router}_active_models"] = ";".join(active)
            for dataset in datasets:
                target = router_stats["targets"][dataset]
                accuracy = core.routed_accuracy(mapping, target["correct"], target["totals"])
                delta = 100 * (accuracy - full_accuracy[router][dataset])
                global_delta = 100 * (accuracy - global_best[router][dataset]["accuracy"])
                row[f"{router}_{dataset}_accuracy"] = accuracy
                row[f"{router}_{dataset}_delta_full_pp"] = delta
                row[f"{router}_{dataset}_delta_global_best_pp"] = global_delta
                all_deltas.append(delta)
                all_global_deltas.append(global_delta)
        row["active_union_count"] = len(active_union)
        row["mean_delta_full_pp"] = sum(all_deltas) / len(all_deltas)
        row["worst_delta_full_pp"] = min(all_deltas)
        row["positive_cells_vs_full"] = sum(delta > 1e-12 for delta in all_deltas)
        row["mean_delta_global_best_pp"] = sum(all_global_deltas) / len(all_global_deltas)
        row["worst_delta_global_best_pp"] = min(all_global_deltas)
        row["_signature"] = tuple(signatures)
        return row

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["worst_delta_full_pp"],
            row["mean_delta_full_pp"],
            row["positive_cells_vs_full"],
            -row["size"],
            row["models"],
        )

    exact_top: list[dict[str, Any]] = []
    exact_frontier: list[dict[str, Any]] = []
    exact_size_summary: list[dict[str, Any]] = []
    exact_count = 0
    exact_nondegenerate_count = 0
    exact_positive_all_count = 0
    best_exact: dict[str, Any] | None = None
    best_exact_nondegenerate: dict[str, Any] | None = None
    for size in range(1, min(args.exact_max_size, len(models)) + 1):
        size_count = 0
        size_best: dict[str, Any] | None = None
        size_frontier: list[dict[str, Any]] = []
        for subset in combinations(models, size):
            row = evaluate(subset, "exact")
            size_count += 1
            exact_count += 1
            if size_best is None or rank_key(row) > rank_key(size_best):
                size_best = row
            if best_exact is None or rank_key(row) > rank_key(best_exact):
                best_exact = row
            is_nondegenerate = row["query_active_count"] >= 2 and row["subject_active_count"] >= 2
            if is_nondegenerate:
                exact_nondegenerate_count += 1
                exact_positive_all_count += row["positive_cells_vs_full"] == 4
                if best_exact_nondegenerate is None or rank_key(row) > rank_key(best_exact_nondegenerate):
                    best_exact_nondegenerate = row
            exact_top.append(row)
            if len(exact_top) >= 10000:
                exact_top = sorted(exact_top, key=rank_key, reverse=True)[:1000]
            if size == args.exact_max_size:
                size_frontier.append(row)
                if len(size_frontier) >= 10000:
                    size_frontier = sorted(size_frontier, key=rank_key, reverse=True)[: args.beam_width]
        if size_best is None:
            continue
        exact_size_summary.append(
            {
                "size": size,
                "subsets": size_count,
                "best_models": size_best["models"],
                "best_mean_delta_full_pp": size_best["mean_delta_full_pp"],
                "best_worst_delta_full_pp": size_best["worst_delta_full_pp"],
                "best_positive_cells": size_best["positive_cells_vs_full"],
            }
        )
        if size == args.exact_max_size:
            exact_frontier = sorted(size_frontier, key=rank_key, reverse=True)[: args.beam_width]

    if best_exact is None or best_exact_nondegenerate is None:
        raise RuntimeError("Exact search did not produce the required candidates.")
    exact_top = sorted(exact_top, key=rank_key, reverse=True)[:1000]
    frontier = exact_frontier
    beam_rows: list[dict[str, Any]] = []
    unique_subsets_evaluated = exact_count
    for size in range(args.exact_max_size + 1, len(models) + 1):
        candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
        seen_subsets: set[tuple[str, ...]] = set()
        for parent in frontier:
            parent_models = tuple(parent["models"].split(";"))
            parent_set = set(parent_models)
            for model in models:
                if model in parent_set:
                    continue
                subset = tuple(sorted((*parent_models, model), key=models.index))
                if subset in seen_subsets:
                    continue
                seen_subsets.add(subset)
                row = evaluate(subset, "beam")
                signature = row["_signature"]
                previous = candidates.get(signature)
                if previous is None or rank_key(row) > rank_key(previous):
                    candidates[signature] = row
        unique_subsets_evaluated += len(seen_subsets)
        frontier = sorted(candidates.values(), key=rank_key, reverse=True)[: args.beam_width]
        beam_rows.extend(frontier)
        if not frontier:
            break

    retained_rows = exact_top + beam_rows
    best = max(retained_rows, key=rank_key)
    retained_nondegenerate = [
        row
        for row in retained_rows
        if row["query_active_count"] >= 2 and row["subject_active_count"] >= 2
    ]
    best_nondegenerate = max(
        [best_exact_nondegenerate, *retained_nondegenerate],
        key=rank_key,
    )
    # Remove the internal hashable signature before serialization.
    for row in retained_rows:
        row.pop("_signature", None)
    best_nondegenerate.pop("_signature", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "exact_top_1000.csv", exact_top)
    write_csv(args.output_dir / "exact_size_summary.csv", exact_size_summary)
    write_csv(args.output_dir / "beam_subsets_size_7_plus.csv", beam_rows)
    write_csv(args.output_dir / "top_100.csv", sorted(retained_rows, key=rank_key, reverse=True)[:100])
    summary = {
        "status": "completed",
        "models": models,
        "legacy_models": legacy,
        "expanded_models": expanded,
        "datasets": list(datasets),
        "coverage": coverage,
        "search": {
            "exact_max_size": args.exact_max_size,
            "exact_subsets": exact_count,
            "beam_width": args.beam_width,
            "beam_rows": len(beam_rows),
            "unique_subsets_evaluated": unique_subsets_evaluated,
        },
        "full_pool_accuracy": full_accuracy,
        "global_best_single": global_best,
        "best": best,
        "nondegenerate": {
            "definition": "Query and Subject each route at least two distinct models",
            "exact_candidates_size_1_to_6": exact_nondegenerate_count,
            "exact_positive_all_four_size_1_to_6": exact_positive_all_count,
            "retained_beam_candidates": sum(
                row["phase"] == "beam" for row in retained_nondegenerate
            ),
            "retained_positive_all_four": sum(
                row["positive_cells_vs_full"] == 4 for row in retained_nondegenerate
            ),
            "best": best_nondegenerate,
        },
    }
    write_json(args.output_dir / "summary.json", summary)

    report = f"""# Combined 28-model pool search

The common pool contains {len(legacy)} legacy 7B--9B specialists and {len(expanded)} scale-expanded models.
BBH and GPQA have complete protocol-compatible predictions for all {len(models)} models.

## Search coverage

- Exact enumeration: all {exact_count:,} subsets of size 1--{args.exact_max_size}.
- Deterministic mapping-deduplicated beam: {len(beam_rows):,} retained rows above size {args.exact_max_size}.
- Total unique subsets evaluated: {unique_subsets_evaluated:,}.

## Result

- Unrestricted best: `{best['models']}`; mean delta vs full 28-model pool {best['mean_delta_full_pp']:+.2f} pp; worst delta {best['worst_delta_full_pp']:+.2f} pp; positive cells {best['positive_cells_vs_full']}/4.
- Best true multi-expert subset: `{best_nondegenerate['models']}`; mean delta {best_nondegenerate['mean_delta_full_pp']:+.2f} pp; worst delta {best_nondegenerate['worst_delta_full_pp']:+.2f} pp; positive cells {best_nondegenerate['positive_cells_vs_full']}/4.
- Exact true multi-expert candidates improving all four Query/Subject x BBH/GPQA cells (size 1--{args.exact_max_size}): {exact_positive_all_count}.

The exact claim applies to sizes 1--{args.exact_max_size}. Results for larger subsets are heuristic beam-search results.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
