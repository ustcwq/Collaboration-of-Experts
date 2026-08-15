from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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
from .evaluation import exact_mcnemar, selection_correctness
from .method_consensus import ConsensusVariant, consensus_selections
from .run_strict_positive_portfolio import (
    _dataset_jobs,
    _load_labels,
    _prior_art_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate target-label-free method-consensus predictions before evaluation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def configured_variants(config: Mapping[str, Any]) -> tuple[ConsensusVariant, ...]:
    grid = config["candidate_grid"]
    variants: list[ConsensusVariant] = []
    for subset in grid["subsets"]:
        for weighting in grid["global_weightings"]:
            for confidence_power in grid["confidence_powers"]:
                for fallback_share in grid["fallback_shares"]:
                    confidence = str(float(confidence_power)).replace(".", "p")
                    fallback = str(float(fallback_share)).replace(".", "p")
                    variants.append(
                        ConsensusVariant(
                            name=(
                                f"mcons__{subset}__{weighting}__c{confidence}__f{fallback}"
                            ),
                            subset=str(subset),
                            global_weighting=str(weighting),
                            confidence_power=float(confidence_power),
                            fallback_share=float(fallback_share),
                        )
                    )
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("Consensus variant names are not unique")
    return tuple(variants)


def _target_manifest(
    job: Mapping[str, Any], seed: int
) -> tuple[Path, dict[str, Any]]:
    matches = sorted(Path(job["run_root"]).glob(f"seed_{seed}_gpu*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one authenticated seed directory for {job['name']}/{seed}")
    seed_dir = matches[0]
    manifest = json.loads((seed_dir / "prediction_manifest.json").read_text(encoding="utf-8"))
    if job["package_scope"] == "source":
        return seed_dir, manifest
    return seed_dir, manifest["targets"][str(job["name"])]


def _candidate_row(
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
        raise RuntimeError(f"Consensus/reference IDs differ: {dataset}/{seed}/{method}")
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

    portfolio_config_path = Path(config["portfolio_config"])
    portfolio_config = yaml.safe_load(portfolio_config_path.read_text(encoding="utf-8"))
    jobs = _dataset_jobs(portfolio_config)
    target_names = tuple(str(value) for value in config["targets"])
    if not target_names or not set(target_names).issubset(jobs):
        raise ValueError("Consensus targets are empty or unknown")
    variants = configured_variants(config)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    if not variants:
        raise ValueError("No consensus variants are configured")

    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = set()
    predictions: dict[str, dict[int, dict[str, list[Any]]]] = {}
    references: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for dataset in target_names:
        job = jobs[dataset]
        seeds = list(job["seeds"])
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        predictions[dataset] = {}
        references[dataset] = {}
        output_hashes[dataset] = {}
        for seed in seeds:
            _, target_manifest = _target_manifest(job, seed)
            methods = tuple(
                sorted(
                    method
                    for method, group in target_manifest["method_group"].items()
                    if group == "full_pool"
                )
            )
            expected = int(config["expected_full_pool_methods"])
            if len(methods) != expected:
                raise RuntimeError(
                    f"Full-pool method count differs for {dataset}/{seed}: {len(methods)} != {expected}"
                )
            rows_by_method: dict[str, list[Any]] = {}
            for method in methods:
                rows = _prior_art_rows(
                    job,
                    seed,
                    method,
                    authenticated,
                    manifest_paths,
                )
                if args.max_questions is not None:
                    rows = rows[: args.max_questions]
                rows_by_method[method] = rows
            reference_method = str(config["reference_method"])
            references[dataset][seed] = rows_by_method[reference_method]
            predictions[dataset][seed] = {}
            output_hashes[dataset][str(seed)] = {}
            for variant in variants:
                rows = consensus_selections(
                    rows_by_method,
                    variant,
                    reference_method=reference_method,
                )
                predictions[dataset][seed][variant.name] = rows
                relative = Path("predictions") / dataset / f"seed_{seed}" / f"{variant.name}.jsonl"
                digest = write_selections(args.output_dir / relative, rows)
                output_hashes[dataset][str(seed)][variant.name] = {
                    "path": str(relative),
                    "sha256": digest,
                }

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [args.config, portfolio_config_path, *sorted(manifest_paths)],
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "targets": target_names,
        "reference_method": config["reference_method"],
        "variants": [variant.__dict__ for variant in variants],
        "predictions": output_hashes,
        "labels_opened_during_prediction": False,
    }
    prediction_manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_labels": sha256_file(prediction_manifest_path),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    evaluation_rows: list[dict[str, Any]] = []
    for dataset in target_names:
        labels = _load_labels(jobs[dataset], None)
        for seed, by_method in predictions[dataset].items():
            reference = references[dataset][seed]
            for method, rows in by_method.items():
                evaluation_rows.append(
                    _candidate_row(dataset, seed, method, rows, reference, labels)
                )
    write_csv(args.output_dir / "candidate_results.csv", evaluation_rows)
    aggregate_rows: list[dict[str, Any]] = []
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in target_names:
        for variant in variants:
            rows = [
                row
                for row in evaluation_rows
                if row["dataset"] == dataset and row["method"] == variant.name
            ]
            aggregate = {
                "dataset": dataset,
                "method": variant.name,
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
            aggregate_rows.append(aggregate)
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
            "selection_warning": (
                "Candidate ranking uses development-target labels only after every candidate "
                "prediction was written and hashed; it is not a blind-confirmation claim."
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
            "prediction_manifest_sha256_before_labels": boundary[
                "prediction_manifest_sha256_before_labels"
            ],
            "artifact_hashes": artifact_hashes,
            "artifact_manifest_sha256": manifest_sha256(artifact_hashes),
        },
    )


if __name__ == "__main__":
    main()
