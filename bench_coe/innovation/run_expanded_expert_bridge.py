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
from .data import CacheAdapter, EvaluationLabelAdapter, load_family_map
from .expanded_expert_bridge import (
    annotate_bridge_predictions,
    cross_pool_paired_comparison,
)
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows
from .selectors import SourceBestSelector, source_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a validation-selected expanded expert against frozen FCRG targets"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def _source(config: Mapping[str, Any], family_map: Mapping[str, str]) -> tuple[Any, Any]:
    spec = config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(spec["cache_path"]),
        str(spec["dataset"]),
        str(spec["split"]),
        str(spec["modality"]),
        family_map,
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    batch = adapter.load_observables()
    if len(batch.question_ids) != int(spec["expected_questions"]):
        raise RuntimeError("Expanded source question count differs from the pinned config")
    return batch, adapter.load_source_labels()


def _target_batch(
    config: Mapping[str, Any],
    target: Mapping[str, Any],
    family_map: Mapping[str, str],
    limit: int | None,
) -> Any:
    adapter = CacheAdapter.from_target_observables(
        Path(target["observable_cache_path"]),
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        family_map,
        [str(value) for value in config["experts"]],
        str(target["observable_manifest_sha256"]),
    )
    batch = adapter.load_observables(limit=limit)
    expected = min(int(target["expected_questions"]), limit or int(target["expected_questions"]))
    if len(batch.question_ids) != expected:
        raise RuntimeError(f"{target['name']} observable question count differs from config")
    return batch


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
    if sha256_file(source_manifest) != str(config["source"]["projection_manifest_sha256"]):
        raise PermissionError("Expanded source projection manifest hash mismatch")
    family_map = load_family_map(Path(config["family_map"]))
    source_batch, source_labels = _source(config, family_map)
    accuracies = source_accuracy(source_batch, source_labels)
    selector = SourceBestSelector().fit(source_batch, source_labels)

    portfolio_path = Path(config["portfolio_config"])
    jobs = _dataset_jobs(yaml.safe_load(portfolio_path.read_text(encoding="utf-8")))
    targets = list(config["targets"])
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    authenticated: dict[str, str] = {}
    prior_art_manifests: set[Path] = set()
    predictions: dict[str, dict[int, list[Any]]] = {}
    references: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, str]]] = {}

    for target in targets:
        name = str(target["name"])
        if name not in jobs:
            raise KeyError(f"No frozen prior-art job for {name}")
        batch = _target_batch(config, target, family_map, args.max_questions)
        base = annotate_bridge_predictions(
            selector.predict(batch),
            source_dataset=str(config["source"]["dataset"]),
            source_split=str(config["source"]["split"]),
            source_accuracy=accuracies,
        )
        seeds = list(jobs[name]["seeds"])
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        predictions[name] = {}
        references[name] = {}
        output_hashes[name] = {}
        for seed in seeds:
            reference = _prior_art_rows(
                jobs[name], seed, "fcrg_full", authenticated, prior_art_manifests
            )
            if args.max_questions is not None:
                reference = reference[: args.max_questions]
            if {row.question_id for row in base} != {row.question_id for row in reference}:
                raise RuntimeError(f"Expanded candidate/reference IDs differ: {name}/{seed}")
            predictions[name][seed] = list(base)
            references[name][seed] = reference
            relative = Path("predictions") / name / f"seed_{seed}.jsonl"
            digest = write_selections(args.output_dir / relative, list(base))
            output_hashes[name][str(seed)] = {"path": str(relative), "sha256": digest}

    input_paths = {
        args.config,
        portfolio_path,
        Path(config["family_map"]),
        Path(config["dataset_registry"]),
        Path(config["source"]["cache_path"]),
        source_manifest,
        *prior_art_manifests,
    }
    input_paths.update(Path(target["observable_cache_path"]) for target in targets)
    environment = environment_manifest(
        sys.argv, int(config["protocol_seed"]), sorted(input_paths)
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "source_selection_rule": "highest_global_accuracy_on_mmmu_pro_validation_id",
        "source_accuracy_by_expert": dict(sorted(accuracies.items())),
        "predictions": output_hashes,
        "target_labels_opened_during_prediction": False,
        "target_observables_physically_label_free": True,
    }
    prediction_manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_target_labels": sha256_file(
            prediction_manifest_path
        ),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "target_labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    # Isolated evaluation begins only after every candidate prediction is boundary-bound.
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    acceptance: dict[str, Any] = {}
    for target in targets:
        name = str(target["name"])
        candidate_labels = EvaluationLabelAdapter.from_registry(
            Path(target["label_cache_path"]),
            str(target["dataset"]),
            str(target["split"]),
            str(target["modality"]),
            [str(value) for value in config["experts"]],
            Path(config["dataset_registry"]),
            str(config["dataset_registry_sha256"]),
        ).load(limit=args.max_questions)
        reference_labels = _load_labels(jobs[name], args.max_questions)
        comparisons: list[dict[str, Any]] = []
        candidate_matrix: list[np.ndarray] = []
        reference_matrix: list[np.ndarray] = []
        for seed in predictions[name]:
            comparison, candidate_values, reference_values = cross_pool_paired_comparison(
                predictions[name][seed],
                candidate_labels,
                references[name][seed],
                reference_labels,
                bootstrap_seed=int(config["protocol_seed"]) + int(seed),
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
            comparison.update({"dataset": name, "seed": seed})
            comparisons.append(comparison)
            seed_rows.append(comparison)
            candidate_matrix.append(candidate_values)
            reference_matrix.append(reference_values)
        candidate_array = np.stack(candidate_matrix)
        reference_array = np.stack(reference_matrix)
        accuracy = float(candidate_array.mean())
        fcrg_accuracy = float(reference_array.mean())
        delta = accuracy - fcrg_accuracy
        aggregate = {
            "dataset": name,
            "seed_count": len(comparisons),
            "samples_per_seed": int(candidate_array.shape[1]),
            "accuracy_mean": accuracy,
            "accuracy_std": float(np.std(candidate_array.mean(axis=1))),
            "fcrg_full_accuracy_mean": fcrg_accuracy,
            "delta_vs_fcrg_full_mean": delta,
            "delta_vs_v2": accuracy - float(target["v2_accuracy"]),
            "minimum_seed_delta_vs_fcrg": min(
                float(row["delta_vs_fcrg_full"]) for row in comparisons
            ),
        }
        aggregate_rows.append(aggregate)
        minimum = float(config["acceptance"]["minimum_delta_vs_fcrg"])
        acceptance[name] = {
            "passes_large_gain": bool(
                aggregate["minimum_seed_delta_vs_fcrg"] > minimum
            ),
            "does_not_regress_v2": bool(aggregate["delta_vs_v2"] >= -1e-12),
            "minimum_required_delta_vs_fcrg": minimum,
        }

    write_csv(args.output_dir / "seed_results.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "results": {row["dataset"]: row for row in aggregate_rows},
            "acceptance": acceptance,
            "all_targets_pass": all(
                row["passes_large_gain"] and row["does_not_regress_v2"]
                for row in acceptance.values()
            ),
            "target_label_firewall": (
                "All expanded-pool predictions were generated from hashed, physically label-free "
                "target observables and boundary-bound before either candidate or FCRG labels "
                "were opened by the evaluator."
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
