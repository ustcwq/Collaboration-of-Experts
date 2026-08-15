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
from .data import CacheAdapter, EvaluationLabelAdapter, assert_disjoint, load_family_map
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap
from .expanded_expert_bridge import (
    annotate_bridge_predictions,
    cross_pool_paired_comparison,
)
from .expanded_source_transfer import leave_one_environment_out_source_best
from .run_large_gain_portfolio_v3 import _component_rows, _run_manifests
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows
from .selectors import SourceBestSelector, source_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run expanded MMMU-Pro source LOSO and validation-to-test transfer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def evaluate_acceptance(
    rows: list[Mapping[str, Any]],
    minimum_delta: float,
    v3_floors: Mapping[str, float],
) -> dict[str, Any]:
    by_dataset = {str(row["dataset"]): row for row in rows}
    checks: dict[str, Any] = {}
    for dataset, floor in v3_floors.items():
        row = by_dataset.get(dataset)
        checks[dataset] = {
            "present": row is not None,
            "at_least_large_gain_threshold": bool(
                row is not None
                and float(row["minimum_seed_delta_vs_fcrg"]) + 1e-12 >= minimum_delta
            ),
            "does_not_regress_v3": bool(
                row is not None and float(row["accuracy_mean"]) + 1e-12 >= float(floor)
            ),
            "v3_accuracy_floor": float(floor),
        }
    passed = set(v3_floors) == set(by_dataset) and all(
        check["present"]
        and check["at_least_large_gain_threshold"]
        and check["does_not_regress_v3"]
        for check in checks.values()
    )
    return {
        "passed": passed,
        "minimum_delta_vs_fcrg_at_least": minimum_delta,
        "checks": checks,
    }


def _load_batches(config: Mapping[str, Any], limit: int | None) -> tuple[Any, Any, Any]:
    family_map = load_family_map(Path(config["family_map"]))
    experts = [str(value) for value in config["experts"]]
    source = config["source"]
    source_adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        experts,
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    source_batch = source_adapter.load_observables(limit=limit)
    source_labels = source_adapter.load_source_labels(limit=limit)
    target = config["target"]
    target_adapter = CacheAdapter.from_target_observables(
        Path(target["observable_cache_path"]),
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        family_map,
        experts,
        str(target["observable_manifest_sha256"]),
    )
    target_batch = target_adapter.load_observables(limit=limit)
    assert_disjoint(source_batch, target_batch)
    return source_batch, source_labels, target_batch


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)

    source_manifest = Path(config["source"]["projection_manifest"])
    target_manifest = Path(config["target"]["projection_manifest"])
    if sha256_file(source_manifest) != str(config["source"]["projection_manifest_sha256"]):
        raise PermissionError("Expanded source projection manifest hash mismatch")
    if sha256_file(target_manifest) != str(config["target"]["projection_manifest_sha256"]):
        raise PermissionError("Expanded target projection manifest hash mismatch")

    source_batch, source_labels, target_batch = _load_batches(config, args.max_questions)
    expected_source = min(
        int(config["source"]["expected_questions"]),
        args.max_questions or int(config["source"]["expected_questions"]),
    )
    expected_target = min(
        int(config["target"]["expected_questions"]),
        args.max_questions or int(config["target"]["expected_questions"]),
    )
    if len(source_batch.question_ids) != expected_source:
        raise RuntimeError("Expanded source question count differs from config")
    if len(target_batch.question_ids) != expected_target:
        raise RuntimeError("Expanded target question count differs from config")

    source_rows, fold_audit = leave_one_environment_out_source_best(
        source_batch, source_labels
    )
    global_accuracies = source_accuracy(source_batch, source_labels)
    target_rows = annotate_bridge_predictions(
        SourceBestSelector().fit(source_batch, source_labels).predict(target_batch),
        source_dataset=str(config["source"]["dataset"]),
        source_split=str(config["source"]["split"]),
        source_accuracy=global_accuracies,
    )
    base_rows = {"source_loso": source_rows, str(config["target"]["name"]): target_rows}

    portfolio_path = Path(config["portfolio_config"])
    jobs = _dataset_jobs(yaml.safe_load(portfolio_path.read_text(encoding="utf-8")))
    v3_root = Path(config["v3_run_root"])
    v3_prediction, v3_boundary, v3_complete, v3_manifest_paths = _run_manifests(v3_root)
    manifest_paths = set(v3_manifest_paths)
    authenticated: dict[str, str] = {}
    candidates: dict[str, dict[int, list[Any]]] = {}
    fcrg_rows: dict[str, dict[int, list[Any]]] = {}
    v3_rows: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for dataset, rows in base_rows.items():
        seeds = list(jobs[dataset]["seeds"])
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        candidates[dataset] = {}
        fcrg_rows[dataset] = {}
        v3_rows[dataset] = {}
        output_hashes[dataset] = {}
        for seed in seeds:
            fcrg = _prior_art_rows(
                jobs[dataset], seed, "fcrg_full", authenticated, manifest_paths
            )
            prior_v3 = _component_rows(
                v3_root,
                v3_prediction,
                v3_boundary,
                v3_complete,
                dataset,
                seed,
                None,
                authenticated,
            )
            if args.max_questions is not None:
                fcrg = fcrg[: args.max_questions]
                prior_v3 = prior_v3[: args.max_questions]
            ids = {row.question_id for row in rows}
            if ids != {row.question_id for row in fcrg} or ids != {
                row.question_id for row in prior_v3
            }:
                raise RuntimeError(f"Expanded/V3/FCRG IDs differ for {dataset}/{seed}")
            candidates[dataset][seed] = list(rows)
            fcrg_rows[dataset][seed] = fcrg
            v3_rows[dataset][seed] = prior_v3
            relative = Path("predictions") / dataset / f"seed_{seed}.jsonl"
            digest = write_selections(args.output_dir / relative, list(rows))
            output_hashes[dataset][str(seed)] = {
                "path": str(relative),
                "sha256": digest,
            }

    input_paths = {
        args.config,
        portfolio_path,
        Path(config["family_map"]),
        Path(config["dataset_registry"]),
        Path(config["source"]["cache_path"]),
        Path(config["target"]["observable_cache_path"]),
        source_manifest,
        target_manifest,
        *manifest_paths,
    }
    environment = environment_manifest(
        sys.argv, int(config["protocol_seed"]), sorted(input_paths)
    )
    environment["authenticated_reference_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "source_selection_rule": "leave_one_environment_out_source_best",
        "target_selection_rule": "global_source_best_on_validation_id",
        "source_accuracy_by_expert": dict(sorted(global_accuracies.items())),
        "source_loso_fold_audit": fold_audit,
        "predictions": output_hashes,
        "target_labels_opened_during_prediction": False,
        "target_observables_physically_label_free": True,
    }
    prediction_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_target_labels": sha256_file(prediction_path),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "target_labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    # Isolated evaluation begins only after all source/test predictions are hashed.
    candidate_labels = {
        "source_loso": EvaluationLabelAdapter.from_registry(
            Path(config["source"]["cache_path"]),
            str(config["source"]["dataset"]),
            str(config["source"]["split"]),
            str(config["source"]["modality"]),
            [str(value) for value in config["experts"]],
            Path(config["dataset_registry"]),
            str(config["dataset_registry_sha256"]),
        ).load(limit=args.max_questions),
        str(config["target"]["name"]): EvaluationLabelAdapter.from_registry(
            Path(config["target"]["label_cache_path"]),
            str(config["target"]["dataset"]),
            str(config["target"]["split"]),
            str(config["target"]["modality"]),
            [str(value) for value in config["experts"]],
            Path(config["dataset_registry"]),
            str(config["dataset_registry_sha256"]),
        ).load(limit=args.max_questions),
    }
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for dataset in base_rows:
        reference_labels = _load_labels(jobs[dataset], args.max_questions)
        candidate_matrix: list[np.ndarray] = []
        fcrg_matrix: list[np.ndarray] = []
        v3_matrix: list[np.ndarray] = []
        for seed in candidates[dataset]:
            comparison, left, right = cross_pool_paired_comparison(
                candidates[dataset][seed],
                candidate_labels[dataset],
                fcrg_rows[dataset][seed],
                reference_labels,
                bootstrap_seed=int(config["protocol_seed"]) + seed,
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
            _, _, prior = cross_pool_paired_comparison(
                candidates[dataset][seed],
                candidate_labels[dataset],
                v3_rows[dataset][seed],
                reference_labels,
                bootstrap_seed=int(config["protocol_seed"]) + seed + 101,
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
            v3_rescue, v3_harm, v3_p = exact_mcnemar(left, prior)
            candidate_matrix.append(left)
            fcrg_matrix.append(right)
            v3_matrix.append(prior)
            seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "samples": len(left),
                    "accuracy": float(left.mean()),
                    "fcrg_full_accuracy": float(right.mean()),
                    "v3_accuracy": float(prior.mean()),
                    "delta_vs_fcrg_full": float((left - right).mean()),
                    "delta_vs_v3": float((left - prior).mean()),
                    "rescue_count_vs_fcrg": comparison["rescue_count"],
                    "harm_count_vs_fcrg": comparison["harm_count"],
                    "exact_mcnemar_p_vs_fcrg": comparison["exact_mcnemar_p"],
                    "rescue_count_vs_v3": v3_rescue,
                    "harm_count_vs_v3": v3_harm,
                    "exact_mcnemar_p_vs_v3": v3_p,
                }
            )
        candidate_array = np.stack(candidate_matrix)
        fcrg_array = np.stack(fcrg_matrix)
        v3_array = np.stack(v3_matrix)
        aggregate_rows.append(
            {
                "dataset": dataset,
                "seed_count": candidate_array.shape[0],
                "samples_per_seed": candidate_array.shape[1],
                "accuracy_mean": float(candidate_array.mean()),
                "accuracy_std": float(np.std(candidate_array.mean(axis=1))),
                "fcrg_full_accuracy_mean": float(fcrg_array.mean()),
                "v3_accuracy_mean": float(v3_array.mean()),
                "delta_vs_fcrg_full_mean": float((candidate_array - fcrg_array).mean()),
                "delta_vs_v3_mean": float((candidate_array - v3_array).mean()),
                "minimum_seed_delta_vs_fcrg": float(
                    np.min((candidate_array - fcrg_array).mean(axis=1))
                ),
                "hierarchical_paired_bootstrap_delta_vs_fcrg_ci95": list(
                    hierarchical_paired_bootstrap(
                        candidate_array,
                        fcrg_array,
                        int(config["protocol_seed"]),
                        samples=int(config["bootstrap_samples"]),
                    )
                ),
            }
        )

    acceptance = evaluate_acceptance(
        aggregate_rows,
        float(config["acceptance"]["minimum_delta_vs_fcrg"]),
        {
            str(key): float(value)
            for key, value in config["acceptance"]["v3_accuracy_floors"].items()
        },
    )
    write_csv(args.output_dir / "seed_results.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(args.output_dir / "acceptance.json", acceptance)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "results": {row["dataset"]: row for row in aggregate_rows},
            "acceptance": acceptance,
            "all_targets_pass": bool(acceptance["passed"]),
            "target_label_firewall": (
                "All test predictions were generated from a physically label-free cache and "
                "hashed before expanded-pool or reference labels were opened. Each source row "
                "was selected using labels from other source environments only."
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
            "all_targets_pass": bool(acceptance["passed"]),
        },
    )


if __name__ == "__main__":
    main()
