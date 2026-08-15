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
from .goal_guardrails import evaluate_strict_improvement_contract
from .run_conservative_meta_optimization import (
    _aggregate_comparison,
    _correctness_for_labels,
    _read_authenticated_selections,
)
from .run_strict_positive_portfolio import (
    _completion_bound,
    _dataset_jobs,
    _load_labels,
    _portfolio_rows,
    _prior_art_rows,
)
from .schema import Selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the all-dataset greater-than-one-point development portfolio"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def _conditioned_rows(
    root: Path,
    dataset: str,
    seed: int,
    method: str,
    prediction_manifest: Mapping[str, Any],
    complete_manifest: Mapping[str, Any],
    boundary: Mapping[str, Any],
    authenticated: dict[str, str],
) -> list[Selection]:
    entry = prediction_manifest["predictions"][dataset][method][str(seed)]
    path = root / str(entry["path"])
    expected = str(entry["sha256"])
    _completion_bound(path, expected, complete_manifest)
    if boundary.get("prediction_files_sha256", {}).get(str(path)) != expected:
        raise RuntimeError(f"Conditioned prediction is not boundary-bound: {path}")
    rows, actual = _read_authenticated_selections(path, expected)
    authenticated[str(path)] = actual
    return sorted(rows, key=lambda row: row.question_id)


def _evaluation_matrices(
    job: Mapping[str, Any],
    candidates: Mapping[int, list[Selection]],
    references: Mapping[int, list[Selection]],
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _load_labels(job, limit)
    candidate_matrix = np.stack(
        [_correctness_for_labels(candidates[seed], labels) for seed in candidates],
        axis=0,
    )
    reference_matrix = np.stack(
        [_correctness_for_labels(references[seed], labels) for seed in references],
        axis=0,
    )
    return candidate_matrix, reference_matrix


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
    required = [str(value) for value in config["acceptance"]["strict_improvement_targets"]]
    if args.max_datasets is not None:
        required = required[: args.max_datasets]
    components = config["components"]
    if not set(required).issubset(components) or not set(required).issubset(jobs):
        raise ValueError("Large-gain components or jobs are incomplete")

    conditioned_root = Path(config["conditioned_run_root"])
    conditioned_prediction_path = conditioned_root / "prediction_manifest.json"
    conditioned_complete_path = conditioned_root / "complete_manifest.json"
    conditioned_boundary_path = conditioned_root / "prediction_boundary.json"
    conditioned_prediction = json.loads(conditioned_prediction_path.read_text(encoding="utf-8"))
    conditioned_complete = json.loads(conditioned_complete_path.read_text(encoding="utf-8"))
    conditioned_boundary = json.loads(conditioned_boundary_path.read_text(encoding="utf-8"))
    conditioned_prediction_hash = sha256_file(conditioned_prediction_path)
    expected_prediction_hash = conditioned_complete[
        "prediction_manifest_sha256_before_target_labels"
    ]
    if conditioned_prediction_hash != expected_prediction_hash:
        raise RuntimeError("Conditioned prediction manifest is not completion-bound")
    if conditioned_boundary["prediction_manifest_sha256_before_target_labels"] != expected_prediction_hash:
        raise RuntimeError("Conditioned prediction manifest is not boundary-bound")
    if conditioned_prediction.get("target_labels_opened_during_prediction") is not False:
        raise RuntimeError("Conditioned input does not attest to the target-label firewall")

    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = {
        conditioned_prediction_path,
        conditioned_complete_path,
        conditioned_boundary_path,
    }
    candidates: dict[str, dict[int, list[Selection]]] = {}
    references: dict[str, dict[int, list[Selection]]] = {}
    output_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in required:
        job = jobs[dataset]
        seeds = list(job["seeds"])
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        candidates[dataset] = {}
        references[dataset] = {}
        output_hashes[dataset] = {}
        component = components[dataset]
        component_source = str(component["source"])
        component_method = str(component["method"])
        for seed in seeds:
            reference = _prior_art_rows(
                job, seed, "fcrg_full", authenticated, manifest_paths
            )
            if component_source == "prior_art":
                rows = _prior_art_rows(
                    job, seed, component_method, authenticated, manifest_paths
                )
            elif component_source == "conditioned_expert_consensus":
                rows = _conditioned_rows(
                    conditioned_root,
                    dataset,
                    seed,
                    component_method,
                    conditioned_prediction,
                    conditioned_complete,
                    conditioned_boundary,
                    authenticated,
                )
            else:
                raise ValueError(f"Unknown large-gain component source: {component_source}")
            if args.max_questions is not None:
                reference = reference[: args.max_questions]
                rows = rows[: args.max_questions]
            if {row.question_id for row in rows} != {row.question_id for row in reference}:
                raise RuntimeError(f"Large-gain candidate/reference IDs differ: {dataset}/{seed}")
            rows = _portfolio_rows(
                rows,
                str(config["portfolio_name"]),
                dataset,
                component_source,
                component_method,
            )
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
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "portfolio_name": config["portfolio_name"],
        "components": {dataset: components[dataset] for dataset in required},
        "predictions": output_hashes,
        "target_labels_opened_during_prediction": False,
        "portfolio_mapping_is_development_posthoc": True,
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

    aggregate_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for dataset in required:
        candidate_matrix, reference_matrix = _evaluation_matrices(
            jobs[dataset],
            candidates[dataset],
            references[dataset],
            args.max_questions,
        )
        aggregate = _aggregate_comparison(
            str(config["portfolio_name"]), dataset, candidate_matrix, reference_matrix
        )
        aggregate_rows.append(aggregate)
        for seed_index, seed in enumerate(candidates[dataset]):
            seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "samples": candidate_matrix.shape[1],
                    "accuracy": float(candidate_matrix[seed_index].mean()),
                    "fcrg_full_accuracy": float(reference_matrix[seed_index].mean()),
                    "delta_vs_fcrg_full": float(
                        candidate_matrix[seed_index].mean() - reference_matrix[seed_index].mean()
                    ),
                }
            )
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_csv(args.output_dir / "seed_results.csv", seed_rows)
    by_target = {str(row["target"]): row for row in aggregate_rows}
    acceptance = evaluate_strict_improvement_contract(by_target, config["acceptance"])
    write_json(args.output_dir / "acceptance.json", acceptance)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "portfolio": config["portfolio_name"],
            "results": by_target,
            "acceptance": acceptance,
            "protocol_boundary": (
                "Dataset components are a known-development posthoc portfolio. Every selected "
                "per-query prediction was already generated without target labels and all "
                "portfolio outputs were written and hashed before evaluation."
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
            "strict_user_goal_met": acceptance["strict_user_goal_met"],
        },
    )


if __name__ == "__main__":
    main()
