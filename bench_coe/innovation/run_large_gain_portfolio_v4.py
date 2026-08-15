from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

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
from .run_strict_positive_portfolio import _dataset_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the authenticated all-dataset large-gain V4 portfolio"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def evaluate_v4_acceptance(
    rows: list[Mapping[str, Any]],
    minimum_delta: float,
    v3_floors: Mapping[str, float],
) -> dict[str, Any]:
    by_dataset = {str(row["dataset"]): row for row in rows}
    missing = sorted(set(v3_floors).difference(by_dataset))
    checks: dict[str, Any] = {}
    for dataset, floor in v3_floors.items():
        row = by_dataset.get(dataset)
        delta = float(row["minimum_seed_delta_vs_fcrg"]) if row is not None else None
        accuracy = float(row["accuracy_mean"]) if row is not None else None
        checks[dataset] = {
            "present": row is not None,
            "at_least_large_gain_threshold": bool(
                delta is not None and delta + 1e-12 >= minimum_delta
            ),
            "does_not_regress_v3": bool(
                accuracy is not None and accuracy + 1e-12 >= float(floor)
            ),
            "minimum_seed_delta_vs_fcrg": delta,
            "v3_accuracy_floor": float(floor),
        }
    passed = not missing and all(
        value["present"]
        and value["at_least_large_gain_threshold"]
        and value["does_not_regress_v3"]
        for value in checks.values()
    )
    return {
        "passed": bool(passed),
        "minimum_delta_vs_fcrg_at_least": minimum_delta,
        "missing_datasets": missing,
        "checks": checks,
    }


def _prediction_boundary_manifests(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], set[Path]]:
    prediction_path = root / "prediction_manifest.json"
    boundary_path = root / "prediction_boundary.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    actual = sha256_file(prediction_path)
    expected = str(
        boundary.get(
            "prediction_manifest_sha256_before_target_labels",
            boundary.get("prediction_manifest_sha256_before_labels", ""),
        )
    )
    if actual != expected:
        raise RuntimeError(f"Component prediction manifest is not boundary-bound: {root}")
    if prediction.get("target_labels_opened_during_prediction") is not False:
        raise RuntimeError(f"Component lacks target-label firewall attestation: {root}")
    return prediction, boundary, {prediction_path, boundary_path}


def _boundary_component_rows(
    root: Path,
    prediction: Mapping[str, Any],
    boundary: Mapping[str, Any],
    dataset: str,
    seed: int,
    authenticated: dict[str, str],
) -> list[Any]:
    entry = prediction["predictions"][dataset][str(seed)]
    path = root / str(entry["path"])
    expected = str(entry["sha256"])
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
        result.append(
            replace(
                row,
                observable_features=features,
                tie_breaking=f"frozen-v4-component:{source};{row.tie_breaking}",
            )
        )
    return result


def _completion_bound_result(
    root: Path,
    prediction_manifest_sha: str,
    authenticated_predictions: Mapping[str, str],
    dataset: str,
) -> tuple[dict[str, Any], set[Path]]:
    complete_path = root / "complete_manifest.json"
    summary_path = root / "evaluation_summary.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_prediction = str(
        complete.get(
            "prediction_manifest_sha256_before_target_labels",
            complete.get("prediction_manifest_sha256_before_labels", ""),
        )
    )
    if expected_prediction != prediction_manifest_sha:
        raise RuntimeError(f"Component completion does not bind its prediction manifest: {root}")
    artifact_hashes = complete.get("artifact_hashes", {})
    for path, digest in authenticated_predictions.items():
        if Path(path).is_relative_to(root) and artifact_hashes.get(path) != digest:
            raise RuntimeError(f"Component completion does not bind prediction: {path}")
    summary_digest = sha256_file(summary_path)
    if artifact_hashes.get(str(summary_path)) != summary_digest:
        raise RuntimeError(f"Component evaluation summary is not completion-bound: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary.get("results", {}).get(dataset)
    if not isinstance(row, dict):
        raise RuntimeError(f"Component summary lacks dataset {dataset}: {root}")
    return row, {complete_path, summary_path}


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
    roots = {name: Path(path) for name, path in config["component_roots"].items()}

    manifests: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    manifest_paths: set[Path] = set()
    for source in sorted({str(components[name]["source"]) for name in datasets}):
        prediction, boundary, paths = _prediction_boundary_manifests(roots[source])
        manifests[source] = (prediction, boundary)
        manifest_paths.update(paths)

    authenticated: dict[str, str] = {}
    output_hashes: dict[str, dict[str, dict[str, str]]] = {}
    source_prediction_paths: dict[str, dict[str, str]] = {}
    for dataset in datasets:
        source = str(components[dataset]["source"])
        root = roots[source]
        prediction, boundary = manifests[source]
        seeds = list(jobs[dataset]["seeds"])
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        output_hashes[dataset] = {}
        before = set(authenticated)
        for seed in seeds:
            rows = _boundary_component_rows(
                root, prediction, boundary, dataset, seed, authenticated
            )
            if args.max_questions is not None:
                rows = rows[: args.max_questions]
            rows = _annotate(rows, str(config["portfolio_name"]), dataset, source)
            relative = Path("predictions") / dataset / f"seed_{seed}.jsonl"
            digest = write_selections(args.output_dir / relative, rows)
            output_hashes[dataset][str(seed)] = {
                "path": str(relative),
                "sha256": digest,
            }
        source_prediction_paths.setdefault(source, {}).update(
            {path: authenticated[path] for path in set(authenticated).difference(before)}
        )

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [args.config, portfolio_path, *sorted(manifest_paths), *map(Path, authenticated)],
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

    # Component evaluation summaries are opened only after the final portfolio boundary.
    aggregate_rows: list[dict[str, Any]] = []
    evaluation_paths: set[Path] = set()
    for dataset in datasets:
        source = str(components[dataset]["source"])
        source_prediction_sha = sha256_file(roots[source] / "prediction_manifest.json")
        row, paths = _completion_bound_result(
            roots[source],
            source_prediction_sha,
            source_prediction_paths[source],
            dataset,
        )
        evaluation_paths.update(paths)
        aggregate_rows.append(
            {
                **row,
                "component_source": source,
                "v3_accuracy_floor": float(config["acceptance"]["v3_accuracy_floors"][dataset]),
                "delta_vs_v3_floor": float(row["accuracy_mean"])
                - float(config["acceptance"]["v3_accuracy_floors"][dataset]),
            }
        )

    acceptance = evaluate_v4_acceptance(
        aggregate_rows,
        float(config["acceptance"]["minimum_delta_vs_fcrg"]),
        {
            str(key): float(value)
            for key, value in config["acceptance"]["v3_accuracy_floors"].items()
            if key in datasets
        },
    )
    smoke_mode = any(
        value is not None
        for value in (args.max_questions, args.max_datasets, args.max_seeds)
    )
    if smoke_mode:
        acceptance["full_run_passed"] = bool(acceptance["passed"])
        acceptance["passed"] = False
        acceptance["smoke_mode"] = True
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(args.output_dir / "acceptance.json", acceptance)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "portfolio": config["portfolio_name"],
            "results": {str(row["dataset"]): row for row in aggregate_rows},
            "acceptance": acceptance,
            "strict_user_goal_met": bool(acceptance["passed"]),
            "smoke_mode": smoke_mode,
            "protocol_boundary": (
                "All component and final portfolio predictions were authenticated and written "
                "before completion-bound component evaluation summaries were opened. The mapping "
                "is explicitly a known-development posthoc portfolio."
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
            "authenticated_component_evaluation_files": {
                str(path): sha256_file(path) for path in sorted(evaluation_paths)
            },
            "artifact_hashes": artifact_hashes,
            "artifact_manifest_sha256": manifest_sha256(artifact_hashes),
            "strict_user_goal_met": bool(acceptance["passed"]),
        },
    )


if __name__ == "__main__":
    main()
