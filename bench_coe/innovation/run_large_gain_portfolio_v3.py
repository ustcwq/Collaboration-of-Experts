from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .artifacts import (
    environment_manifest,
    manifest_sha256,
    read_selections,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .data import EvaluationLabelAdapter, load_family_map
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap, selection_correctness
from .expanded_expert_bridge import cross_pool_paired_comparison
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the authenticated all-dataset large-gain V3 portfolio"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def evaluate_v3_acceptance(
    aggregate_rows: list[Mapping[str, Any]],
    minimum_delta: float,
    v2_floors: Mapping[str, float],
) -> dict[str, Any]:
    by_dataset = {str(row["dataset"]): row for row in aggregate_rows}
    missing = sorted(set(v2_floors).difference(by_dataset))
    checks: dict[str, Any] = {}
    for dataset, floor in v2_floors.items():
        row = by_dataset.get(dataset)
        checks[dataset] = {
            "present": row is not None,
            "strictly_above_large_gain_threshold": bool(
                row is not None
                and float(row["minimum_seed_delta_vs_fcrg"]) > minimum_delta
            ),
            "does_not_regress_v2": bool(
                row is not None and float(row["accuracy_mean"]) + 1e-12 >= float(floor)
            ),
            "v2_accuracy_floor": float(floor),
        }
    passed = not missing and all(
        value["present"]
        and value["strictly_above_large_gain_threshold"]
        and value["does_not_regress_v2"]
        for value in checks.values()
    )
    return {
        "passed": passed,
        "minimum_delta_vs_fcrg_strictly_greater_than": minimum_delta,
        "missing_datasets": missing,
        "checks": checks,
    }


def _run_manifests(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[Path]]:
    prediction_path = root / "prediction_manifest.json"
    boundary_path = root / "prediction_boundary.json"
    complete_path = root / "complete_manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    actual = sha256_file(prediction_path)
    expected = str(
        complete.get(
            "prediction_manifest_sha256_before_target_labels",
            complete.get("prediction_manifest_sha256_before_labels", ""),
        )
    )
    boundary_expected = str(
        boundary.get(
            "prediction_manifest_sha256_before_target_labels",
            boundary.get("prediction_manifest_sha256_before_labels", ""),
        )
    )
    if actual != expected or actual != boundary_expected:
        raise RuntimeError(f"Component prediction manifest is not doubly bound: {root}")
    if prediction.get("target_labels_opened_during_prediction") is not False:
        raise RuntimeError(f"Component lacks target-label firewall attestation: {root}")
    return prediction, boundary, complete, {prediction_path, boundary_path, complete_path}


def _component_rows(
    root: Path,
    prediction: Mapping[str, Any],
    boundary: Mapping[str, Any],
    complete: Mapping[str, Any],
    dataset: str,
    seed: int,
    method: str | None,
    authenticated: dict[str, str],
) -> list[Any]:
    by_dataset = prediction["predictions"][dataset]
    entry = by_dataset[method][str(seed)] if method is not None else by_dataset[str(seed)]
    path = root / str(entry["path"])
    expected = str(entry["sha256"])
    if complete["artifact_hashes"].get(str(path)) != expected:
        raise RuntimeError(f"Component prediction is not completion-bound: {path}")
    if boundary["prediction_files_sha256"].get(str(path)) != expected:
        raise RuntimeError(f"Component prediction is not boundary-bound: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Component prediction hash mismatch: {path}")
    authenticated[str(path)] = actual
    return sorted(read_selections(path), key=lambda row: row.question_id)


def _annotate(rows: list[Any], portfolio: str, dataset: str, source: str) -> list[Any]:
    result: list[Any] = []
    for row in rows:
        features = dict(row.observable_features)
        features.update(
            {
                "portfolio": portfolio,
                "portfolio_dataset": dataset,
                "portfolio_component_source": source,
                "portfolio_uses_target_labels": False,
            }
        )
        result.append(replace(row, observable_features=features))
    return result


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
    jobs = _dataset_jobs(yaml.safe_load(portfolio_path.read_text(encoding="utf-8")))
    components = config["components"]
    datasets = list(components)
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]

    roots = {
        "v2": Path(config["v2_run_root"]),
        "visual_bridge": Path(config["visual_bridge_run_root"]),
        "gpqa_permutation": Path(config["gpqa_permutation_run_root"]),
    }
    manifests: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    manifest_paths: set[Path] = set()
    for source, root in roots.items():
        prediction, boundary, complete, paths = _run_manifests(root)
        manifests[source] = (prediction, boundary, complete)
        manifest_paths.update(paths)

    authenticated: dict[str, str] = {}
    candidates: dict[str, dict[int, list[Any]]] = {}
    references: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in datasets:
        job = jobs[dataset]
        seeds = list(job["seeds"])
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        component = components[dataset]
        source = str(component["source"])
        method = str(component["method"]) if component.get("method") is not None else None
        prediction, boundary, complete = manifests[source]
        candidates[dataset] = {}
        references[dataset] = {}
        output_hashes[dataset] = {}
        for seed in seeds:
            rows = _component_rows(
                roots[source],
                prediction,
                boundary,
                complete,
                dataset,
                seed,
                method,
                authenticated,
            )
            reference = _prior_art_rows(
                job, seed, "fcrg_full", authenticated, manifest_paths
            )
            if args.max_questions is not None:
                rows = rows[: args.max_questions]
                reference = reference[: args.max_questions]
            if {row.question_id for row in rows} != {row.question_id for row in reference}:
                raise RuntimeError(f"V3 candidate/reference IDs differ: {dataset}/{seed}")
            rows = _annotate(rows, str(config["portfolio_name"]), dataset, source)
            candidates[dataset][seed] = rows
            references[dataset][seed] = reference
            relative = Path("predictions") / dataset / f"seed_{seed}.jsonl"
            digest = write_selections(args.output_dir / relative, rows)
            output_hashes[dataset][str(seed)] = {
                "path": str(relative),
                "sha256": digest,
            }

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [args.config, portfolio_path, *sorted(manifest_paths)],
    )
    environment["authenticated_component_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "portfolio_name": config["portfolio_name"],
        "components": {dataset: components[dataset] for dataset in datasets},
        "predictions": output_hashes,
        "target_labels_opened_during_prediction": False,
        "portfolio_mapping_is_known_development_posthoc": True,
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

    visual_config = yaml.safe_load(
        Path(config["visual_bridge_config"]).read_text(encoding="utf-8")
    )
    visual_targets = {str(row["name"]): row for row in visual_config["targets"]}
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        reference_labels = _load_labels(jobs[dataset], args.max_questions)
        source = str(components[dataset]["source"])
        if source == "visual_bridge":
            target = visual_targets[dataset]
            candidate_labels = EvaluationLabelAdapter.from_registry(
                Path(target["label_cache_path"]),
                str(target["dataset"]),
                str(target["split"]),
                str(target["modality"]),
                [str(value) for value in visual_config["experts"]],
                Path(visual_config["dataset_registry"]),
                str(visual_config["dataset_registry_sha256"]),
            ).load(limit=args.max_questions)
        else:
            candidate_labels = reference_labels

        candidate_matrix: list[np.ndarray] = []
        reference_matrix: list[np.ndarray] = []
        for seed in candidates[dataset]:
            if source == "visual_bridge":
                comparison, left, right = cross_pool_paired_comparison(
                    candidates[dataset][seed],
                    candidate_labels,
                    references[dataset][seed],
                    reference_labels,
                    bootstrap_seed=int(config["protocol_seed"]) + seed,
                    bootstrap_samples=int(config["bootstrap_samples"]),
                )
                accuracy = float(comparison["accuracy"])
                fcrg_accuracy = float(comparison["fcrg_full_accuracy"])
                rescue = int(comparison["rescue_count"])
                harm = int(comparison["harm_count"])
                p_value = float(comparison["exact_mcnemar_p"])
            else:
                left_map = selection_correctness(candidates[dataset][seed], candidate_labels)
                right_map = selection_correctness(references[dataset][seed], reference_labels)
                ids = sorted(left_map)
                left = np.asarray([left_map[qid] for qid in ids], dtype=np.int8)
                right = np.asarray([right_map[qid] for qid in ids], dtype=np.int8)
                rescue, harm, p_value = exact_mcnemar(left, right)
                accuracy = float(left.mean())
                fcrg_accuracy = float(right.mean())
            candidate_matrix.append(left)
            reference_matrix.append(right)
            seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "samples": len(left),
                    "accuracy": accuracy,
                    "fcrg_full_accuracy": fcrg_accuracy,
                    "delta_vs_fcrg_full": accuracy - fcrg_accuracy,
                    "rescue_count": rescue,
                    "harm_count": harm,
                    "exact_mcnemar_p": p_value,
                }
            )
        candidate_array = np.stack(candidate_matrix)
        reference_array = np.stack(reference_matrix)
        delta_matrix = candidate_array - reference_array
        ci = hierarchical_paired_bootstrap(
            candidate_array,
            reference_array,
            int(config["protocol_seed"]),
            samples=int(config["bootstrap_samples"]),
        )
        aggregate_rows.append(
            {
                "dataset": dataset,
                "seed_count": candidate_array.shape[0],
                "samples_per_seed": candidate_array.shape[1],
                "accuracy_mean": float(candidate_array.mean()),
                "accuracy_std": float(np.std(candidate_array.mean(axis=1))),
                "fcrg_full_accuracy_mean": float(reference_array.mean()),
                "delta_vs_fcrg_full_mean": float(delta_matrix.mean()),
                "minimum_seed_delta_vs_fcrg": float(
                    np.min(delta_matrix.mean(axis=1))
                ),
                "hierarchical_paired_bootstrap_delta_ci95": list(ci),
            }
        )

    acceptance = evaluate_v3_acceptance(
        aggregate_rows,
        float(config["acceptance"]["minimum_delta_vs_fcrg"]),
        {
            str(key): float(value)
            for key, value in config["acceptance"]["v2_accuracy_floors"].items()
            if key in datasets
        },
    )
    write_csv(args.output_dir / "seed_results.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(args.output_dir / "acceptance.json", acceptance)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "portfolio": config["portfolio_name"],
            "results": {row["dataset"]: row for row in aggregate_rows},
            "acceptance": acceptance,
            "strict_user_goal_met": bool(acceptance["passed"]),
            "protocol_boundary": (
                "Every component and final portfolio prediction was authenticated and written "
                "before evaluation. Expanded visual candidates and frozen FCRG references were "
                "scored with their respective expert-pool labels."
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
            "strict_user_goal_met": bool(acceptance["passed"]),
        },
    )


if __name__ == "__main__":
    main()
