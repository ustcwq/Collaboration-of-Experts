from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import (
    environment_manifest,
    manifest_sha256,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .data import CacheAdapter, load_family_map
from .evaluation import exact_mcnemar, selection_correctness
from .method_consensus import ConsensusVariant, consensus_selections
from .run_strict_positive_portfolio import (
    _dataset_jobs,
    _load_labels,
    _prior_art_rows,
)
from .source_method_weights import source_method_profiles, source_weight_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-OOF-weighted method consensus with a target-label firewall"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-source-seeds", type=int)
    parser.add_argument("--max-target-seeds", type=int)
    return parser.parse_args()


def _source_labels(source_config: dict[str, Any]) -> Any:
    source = source_config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        load_family_map(Path(source_config["family_map"])),
        [str(value) for value in source_config["experts"]],
        Path(source_config["dataset_registry"]),
        str(source_config["dataset_registry_sha256"]),
    )
    return adapter.load_source_labels()


def _full_pool_methods(job: dict[str, Any], seed: int, expected: int) -> tuple[str, ...]:
    matches = sorted(Path(job["run_root"]).glob(f"seed_{seed}_gpu*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one seed directory for {job['name']}/{seed}")
    manifest = json.loads((matches[0] / "prediction_manifest.json").read_text(encoding="utf-8"))
    target = manifest if job["package_scope"] == "source" else manifest["targets"][job["name"]]
    methods = tuple(
        sorted(method for method, group in target["method_group"].items() if group == "full_pool")
    )
    if len(methods) != expected:
        raise RuntimeError(f"Unexpected full-pool method count: {job['name']}/{seed}/{len(methods)}")
    return methods


def _score_rows(
    dataset: str,
    seed: int,
    method: str,
    candidate: list[Any],
    reference: list[Any],
    labels: Any,
) -> dict[str, Any]:
    candidate_map = selection_correctness(candidate, labels)
    reference_map = selection_correctness(reference, labels)
    if set(candidate_map) != set(reference_map):
        raise RuntimeError(f"Candidate/reference IDs differ: {dataset}/{seed}/{method}")
    ids = sorted(reference_map)
    values = np.asarray([candidate_map[qid] for qid in ids], dtype=np.int8)
    baseline = np.asarray([reference_map[qid] for qid in ids], dtype=np.int8)
    rescue, harm, p_value = exact_mcnemar(values, baseline)
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "samples": len(ids),
        "accuracy": float(values.mean()),
        "reference_accuracy": float(baseline.mean()),
        "delta": float((values - baseline).mean()),
        "rescue_count": rescue,
        "harm_count": harm,
        "exact_mcnemar_p": p_value,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)

    portfolio_path = Path(config["portfolio_config"])
    portfolio = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
    jobs = _dataset_jobs(portfolio)
    source_job = jobs[str(config["source_job"])]
    source_config_path = Path(portfolio["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_labels = _source_labels(source_config)
    expected = int(config["expected_full_pool_methods"])
    reference_method = str(config["reference_method"])
    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = set()

    source_seeds = list(source_job["seeds"])
    if args.max_source_seeds is not None:
        source_seeds = source_seeds[: args.max_source_seeds]
    source_rows: dict[int, dict[str, list[Any]]] = {}
    for seed in source_seeds:
        methods = _full_pool_methods(source_job, seed, expected)
        source_rows[seed] = {
            method: _prior_art_rows(
                source_job, seed, method, authenticated, manifest_paths
            )
            for method in methods
        }
    profiles = source_method_profiles(
        source_rows,
        source_labels,
        reference_method=reference_method,
    )
    weight_candidates = source_weight_candidates(
        profiles,
        reference_method=reference_method,
    )
    variants: list[tuple[str, ConsensusVariant, dict[str, float]]] = []
    for source_weighting, weights in sorted(weight_candidates.items()):
        for subset in config["candidate_grid"]["subsets"]:
            for structural in config["candidate_grid"]["structural_weightings"]:
                name = f"swcons__{source_weighting}__{subset}__{structural}"
                variants.append(
                    (
                        name,
                        ConsensusVariant(name, str(subset), str(structural)),
                        weights,
                    )
                )
    if args.max_candidates is not None:
        variants = variants[: args.max_candidates]
    if not variants:
        raise ValueError("No source-weighted candidates are configured")

    target_names = tuple(str(value) for value in config["targets"])
    predictions: dict[str, dict[int, dict[str, list[Any]]]] = {}
    references: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for dataset in target_names:
        job = jobs[dataset]
        seeds = list(job["seeds"])
        if args.max_target_seeds is not None:
            seeds = seeds[: args.max_target_seeds]
        predictions[dataset] = {}
        references[dataset] = {}
        output_hashes[dataset] = {}
        for seed in seeds:
            methods = _full_pool_methods(job, seed, expected)
            target_rows: dict[str, list[Any]] = {}
            for method in methods:
                rows = _prior_art_rows(job, seed, method, authenticated, manifest_paths)
                if args.max_questions is not None:
                    rows = rows[: args.max_questions]
                target_rows[method] = rows
            references[dataset][seed] = target_rows[reference_method]
            predictions[dataset][seed] = {}
            output_hashes[dataset][str(seed)] = {}
            for name, variant, weights in variants:
                rows = consensus_selections(
                    target_rows,
                    variant,
                    reference_method=reference_method,
                    external_method_weights=weights,
                )
                predictions[dataset][seed][name] = rows
                relative = Path("predictions") / dataset / f"seed_{seed}" / f"{name}.jsonl"
                digest = write_selections(args.output_dir / relative, rows)
                output_hashes[dataset][str(seed)][name] = {
                    "path": str(relative),
                    "sha256": digest,
                }

    write_json(args.output_dir / "source_method_profiles.json", profiles)
    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [args.config, portfolio_path, source_config_path, *sorted(manifest_paths)],
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "source_job": config["source_job"],
        "source_seeds": source_seeds,
        "targets": target_names,
        "reference_method": reference_method,
        "candidate_count": len(variants),
        "candidates": [name for name, _, _ in variants],
        "predictions": output_hashes,
        "target_labels_opened_during_prediction": False,
    }
    prediction_manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_target_labels": sha256_file(prediction_manifest_path),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "source_labels_used_for_method_profiles": True,
        "target_labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    evaluation_rows: list[dict[str, Any]] = []
    for dataset in target_names:
        labels = _load_labels(jobs[dataset], None)
        for seed, rows_by_method in predictions[dataset].items():
            for method, rows in rows_by_method.items():
                evaluation_rows.append(
                    _score_rows(
                        dataset,
                        seed,
                        method,
                        rows,
                        references[dataset][seed],
                        labels,
                    )
                )
    write_csv(args.output_dir / "candidate_results.csv", evaluation_rows)
    aggregate_rows: list[dict[str, Any]] = []
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in target_names:
        for method in (name for name, _, _ in variants):
            rows = [
                row for row in evaluation_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed_count": len(rows),
                    "samples_per_seed": rows[0]["samples"],
                    "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
                    "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
                    "reference_accuracy_mean": float(
                        np.mean([row["reference_accuracy"] for row in rows])
                    ),
                    "delta_mean": float(np.mean([row["delta"] for row in rows])),
                    "delta_min_seed": float(np.min([row["delta"] for row in rows])),
                    "rescue_mean": float(np.mean([row["rescue_count"] for row in rows])),
                    "harm_mean": float(np.mean([row["harm_count"] for row in rows])),
                }
            )
        ranked = sorted(
            (row for row in aggregate_rows if row["dataset"] == dataset),
            key=lambda row: (row["delta_mean"], row["delta_min_seed"], row["method"]),
            reverse=True,
        )
        best_by_dataset[dataset] = ranked[0]
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "best_by_dataset_posthoc_development_diagnostic": best_by_dataset,
            "target_label_firewall": (
                "All candidate predictions were written and hashed after source-only profiling "
                "and before any target evaluation labels were opened."
            ),
        },
    )
    artifact_hashes = {
        str(path): sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "complete_manifest.json"
    }
    write_json(
        args.output_dir / "complete_manifest.json",
        {
            "protocol": config["protocol_name"],
            "scope": config["scope"],
            "runtime_seconds": time.time() - started,
            "prediction_manifest_sha256_before_target_labels": boundary[
                "prediction_manifest_sha256_before_target_labels"
            ],
            "artifact_hashes": artifact_hashes,
            "artifact_manifest_sha256": manifest_sha256(artifact_hashes),
        },
    )


if __name__ == "__main__":
    main()
