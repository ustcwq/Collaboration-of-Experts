from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import (
    files_manifest,
    innovation_code_manifest,
    manifest_sha256,
    read_selections,
    seed_gpu_map,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
)
from .data import CacheAdapter, load_family_map
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap, selection_correctness
from .run_cpi_ce import METHODS
from .schema import EvaluationLabels


PRIMARY = "cpi_ce_calibrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate authenticated categorical CPI seeds")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_hashes(manifest: dict[str, Any], root: Path) -> None:
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = Path(name)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"CPI-CE completion manifest references an external path: {path}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"CPI-CE artifact hash mismatch: {path}")


def _correctness(selections, labels) -> tuple[list[str], np.ndarray]:
    values = selection_correctness(selections, labels)
    ids = sorted(values)
    return ids, np.asarray([float(values[question_id]) for question_id in ids])


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    receipt = validate_test_receipt(receipt_path, args.config)
    current_code = manifest_sha256(innovation_code_manifest())
    if current_code != receipt["code_manifest_sha256"]:
        raise RuntimeError("Current code does not match the CPI-CE test receipt")
    seeds = [int(value) for value in config["seeds"]]
    expected_gpu = seed_gpu_map(config, "physical_gpus")
    base_gpu = seed_gpu_map(config, "base_physical_gpus")
    run_dirs = sorted(path for path in args.run_root.glob("seed_*_gpu*") if path.is_dir())
    if len(run_dirs) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} completed CPI-CE runs, found {len(run_dirs)}")

    runs: dict[int, dict[str, Any]] = {}
    for run_dir in run_dirs:
        prediction_manifest_path = run_dir / "prediction_manifest.json"
        completion_path = run_dir / "run_complete_manifest.json"
        if not prediction_manifest_path.exists() or not completion_path.exists():
            raise RuntimeError(f"Incomplete CPI-CE run: {run_dir}")
        prediction_manifest = _load(prediction_manifest_path)
        completion = _load(completion_path)
        _validate_hashes(completion, run_dir)
        seed = int(prediction_manifest["seed"])
        if seed in runs or seed not in expected_gpu:
            raise RuntimeError(f"Unexpected or duplicate CPI-CE seed: {seed}")
        if int(prediction_manifest["physical_gpu"]) != expected_gpu[seed]:
            raise RuntimeError(f"CPI-CE seed {seed} has the wrong physical GPU")
        if int(completion["seed"]) != seed or int(completion["physical_gpu"]) != expected_gpu[seed]:
            raise RuntimeError(f"CPI-CE seed {seed} completion metadata is inconsistent")
        if sha256_file(prediction_manifest_path) != completion["prediction_manifest_sha256"]:
            raise RuntimeError(f"CPI-CE seed {seed} prediction manifest hash mismatch")
        if prediction_manifest["innovation_code_manifest_sha256"] != current_code:
            raise RuntimeError(f"CPI-CE seed {seed} did not use the tested code")
        for input_path, expected_hash in prediction_manifest["input_hashes"].items():
            if sha256_file(Path(input_path)) != expected_hash:
                raise RuntimeError(f"CPI-CE seed {seed} input changed: {input_path}")
        if int(prediction_manifest["source_questions"]) != int(config["expected_source_questions"]):
            raise RuntimeError(f"CPI-CE seed {seed} did not predict all source questions")
        if int(prediction_manifest["source_environments"]) != int(config["expected_source_environments"]):
            raise RuntimeError(f"CPI-CE seed {seed} did not evaluate all source environments")
        if prediction_manifest["source_label_structure"] != {
            "none_correct": int(config["expected_none_correct_questions"]),
            "one_correct": int(config["expected_one_correct_questions"]),
        }:
            raise RuntimeError(f"CPI-CE seed {seed} has an unexpected label structure")
        selections = {}
        for method in METHODS:
            prediction_path = run_dir / "predictions" / f"{method}.jsonl"
            expected_hash = prediction_manifest["prediction_hashes_before_evaluation"][method]
            if sha256_file(prediction_path) != expected_hash:
                raise RuntimeError(f"CPI-CE seed {seed} prediction hash mismatch for {method}")
            if completion["artifact_hashes"].get(str(prediction_path)) != expected_hash:
                raise RuntimeError(f"CPI-CE seed {seed} completion manifest omits {method}")
            selections[method] = read_selections(prediction_path)
        deterministic = completion.get("determinism", {})
        if deterministic != {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(seed),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }:
            raise RuntimeError(f"CPI-CE seed {seed} lacks deterministic GPU evidence")
        base_dir = Path(config["base_run_root"]) / f"seed_{seed}_gpu{base_gpu[seed]}"
        source_best_path = base_dir / "predictions" / "source_best_single.jsonl"
        bce_path = base_dir / "predictions" / "deepsets_full.jsonl"
        expected_base = config["base_artifacts"][seed]
        if sha256_file(source_best_path) != str(expected_base["source_best_single_sha256"]):
            raise RuntimeError(f"CPI-CE seed {seed} frozen Source-Best input changed")
        if sha256_file(bce_path) != str(expected_base["deepsets_full_sha256"]):
            raise RuntimeError(f"CPI-CE seed {seed} frozen BCE input changed")
        runs[seed] = {
            "dir": run_dir,
            "manifest": prediction_manifest,
            "selections": selections,
            "baseline": read_selections(source_best_path),
            "bce": read_selections(bce_path),
            "calibration": _load(run_dir / "selected_calibration.json"),
            "resource": _load(run_dir / "resource_usage.json"),
        }
    if sorted(runs) != sorted(seeds):
        raise RuntimeError("CPI-CE seed set is incomplete")

    source = config["source"]
    family_map = load_family_map(Path(config["family_map"]))
    source_adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    source_labels = source_adapter.load_source_labels()
    labels = EvaluationLabels(source_labels.dataset, source_labels.split, dict(source_labels.correctness))

    method_vectors: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    baseline_vectors: list[np.ndarray] = []
    bce_vectors: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    seed_rows: list[dict[str, Any]] = []
    environment_deltas: dict[str, list[float]] = {method: [] for method in METHODS}
    threshold_counts: Counter[str] = Counter()
    temperature_counts: Counter[str] = Counter()
    accepted_replacements = 0
    none_wins = 0
    for seed in seeds:
        baseline_ids, baseline = _correctness(runs[seed]["baseline"], labels)
        bce_ids, bce = _correctness(runs[seed]["bce"], labels)
        if baseline_ids != bce_ids or (reference_ids is not None and baseline_ids != reference_ids):
            raise RuntimeError("CPI-CE frozen comparison predictions are not aligned")
        reference_ids = baseline_ids
        baseline_vectors.append(baseline)
        bce_vectors.append(bce)
        for calibration in runs[seed]["calibration"].values():
            temperature_counts[str(calibration["temperature"])] += 1
            threshold_counts[str(calibration["threshold"])] += 1
        accepted_replacements += sum(
            int(bool(item.observable_features.get("proposal_accepted")))
            for item in runs[seed]["selections"][PRIMARY]
        )
        for item in runs[seed]["selections"]["cpi_ce_raw"]:
            selected_probability = float(item.cluster_scores.get(str(item.selected_cluster_id), 0.0))
            none_wins += int(float(item.observable_features.get("none_correct_probability", 0.0)) >= selected_probability)

        for method in METHODS:
            candidate_ids, candidate = _correctness(runs[seed]["selections"][method], labels)
            if candidate_ids != baseline_ids:
                raise RuntimeError(f"CPI-CE {method} predictions are not aligned")
            method_vectors[method].append(candidate)
            per_environment: list[float] = []
            for environment in sorted(runs[seed]["calibration"]):
                indices = [
                    index
                    for index, question_id in enumerate(candidate_ids)
                    if source_labels.environment_by_question[question_id] == environment
                ]
                if not indices:
                    raise RuntimeError(f"CPI-CE seed {seed} references unknown environment {environment}")
                delta = float((candidate[indices] - baseline[indices]).mean())
                per_environment.append(delta)
                environment_deltas[method].append(delta)
            rescue, harm, p_value = exact_mcnemar(candidate, baseline)
            seed_rows.append(
                {
                    "seed": seed,
                    "physical_gpu": expected_gpu[seed],
                    "method": method,
                    "accuracy": float(candidate.mean()),
                    "source_best_accuracy": float(baseline.mean()),
                    "micro_delta": float((candidate - baseline).mean()),
                    "macro_delta": float(np.mean(per_environment)),
                    "worst_environment_delta": min(per_environment),
                    "nonnegative_environment_fraction": float(np.mean([value >= 0.0 for value in per_environment])),
                    "rescue_count": rescue,
                    "harm_count": harm,
                    "exact_mcnemar_p": p_value,
                }
            )
    assert reference_ids is not None
    baseline_matrix = np.stack(baseline_vectors)
    bce_matrix = np.stack(bce_vectors)
    method_results: dict[str, Any] = {}
    for method in METHODS:
        matrix = np.stack(method_vectors[method])
        rows = [row for row in seed_rows if row["method"] == method]
        ci = hierarchical_paired_bootstrap(matrix, baseline_matrix, seed=20260809, samples=10000)
        rescue, harm, p_value = exact_mcnemar(matrix.ravel(), baseline_matrix.ravel())
        method_results[method] = {
            "accuracy_mean": float(matrix.mean()),
            "source_best_accuracy_mean": float(baseline_matrix.mean()),
            "micro_delta": float((matrix - baseline_matrix).mean()),
            "macro_delta": float(np.mean([row["macro_delta"] for row in rows])),
            "worst_seed_environment_delta": min(environment_deltas[method]),
            "nonnegative_seed_environment_fraction": float(
                np.mean([value >= 0.0 for value in environment_deltas[method]])
            ),
            "crossed_seed_query_bootstrap_ci95": list(ci),
            "pooled_seed_query_rescue": rescue,
            "pooled_seed_query_harm": harm,
            "pooled_seed_query_mcnemar_p_descriptive": p_value,
        }
    raw_matrix = np.stack(method_vectors["cpi_ce_raw"])
    raw_bce_ci = hierarchical_paired_bootstrap(raw_matrix, bce_matrix, seed=20260810, samples=10000)
    raw_bce_rescue, raw_bce_harm, raw_bce_p = exact_mcnemar(raw_matrix.ravel(), bce_matrix.ravel())
    raw_vs_bce = {
        "cpi_ce_raw_accuracy_mean": float(raw_matrix.mean()),
        "frozen_bce_accuracy_mean": float(bce_matrix.mean()),
        "micro_delta": float((raw_matrix - bce_matrix).mean()),
        "crossed_seed_query_bootstrap_ci95": list(raw_bce_ci),
        "pooled_seed_query_rescue": raw_bce_rescue,
        "pooled_seed_query_harm": raw_bce_harm,
        "pooled_seed_query_mcnemar_p_descriptive": raw_bce_p,
    }
    primary = method_results[PRIMARY]
    aggregate_gate = {
        "seeds": seeds,
        "primary_method": PRIMARY,
        **primary,
        "method_results": method_results,
        "raw_ce_vs_frozen_bce_full": raw_vs_bce,
        "temperature_counts": dict(sorted(temperature_counts.items())),
        "threshold_counts": dict(sorted(threshold_counts.items())),
        "none_wins_seed_queries": none_wins,
        "accepted_replacements_seed_queries": accepted_replacements,
        "required_macro_delta": float(config["required_macro_delta"]),
        "required_worst_delta": float(config["required_worst_delta"]),
        "required_nonnegative_fraction": float(config["required_nonnegative_fraction"]),
        "property_tests": {
            "receipt": str(receipt_path),
            "test_count": receipt["test_count"],
            "exit_code": receipt["exit_code"],
        },
        "derived_from": "hashed CPI-CE predictions, frozen v5 Source-Best/BCE predictions, and registry-validated labels",
    }
    aggregate_gate["decision"] = (
        "GO"
        if aggregate_gate["macro_delta"] >= aggregate_gate["required_macro_delta"]
        and aggregate_gate["worst_seed_environment_delta"] >= aggregate_gate["required_worst_delta"]
        and aggregate_gate["nonnegative_seed_environment_fraction"] >= aggregate_gate["required_nonnegative_fraction"]
        else "NO-GO"
    )
    resource_rows = [
        {
            "seed": seed,
            "physical_gpu": runs[seed]["resource"]["physical_gpu"],
            "device_name": runs[seed]["resource"]["device_name"],
            "peak_allocated_bytes": runs[seed]["resource"]["peak_allocated_bytes"],
            "peak_reserved_bytes": runs[seed]["resource"]["peak_reserved_bytes"],
            "runtime_seconds": runs[seed]["resource"]["runtime_seconds"],
        }
        for seed in seeds
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "gate.json", aggregate_gate)
    write_csv(args.output_dir / "seed_method_results.csv", seed_rows)
    write_csv(args.output_dir / "resources.csv", resource_rows)
    write_json(
        args.output_dir / "authenticated_inputs.json",
        {
            "config": str(args.config),
            "seeds": seeds,
            "physical_gpu_by_seed": expected_gpu,
            "base_physical_gpu_by_seed": base_gpu,
            "innovation_code_manifest_sha256": current_code,
            "test_receipt": str(receipt_path),
            "base_run_root": str(config["base_run_root"]),
            "per_seed_input_manifest_sha256": {
                str(seed): runs[seed]["manifest"]["input_manifest_sha256"] for seed in seeds
            },
        },
    )
    artifacts = files_manifest([args.output_dir])
    write_json(args.output_dir / "aggregate_complete_manifest.json", {"artifact_hashes": artifacts})
    print(json.dumps({"output_dir": str(args.output_dir), "gate": aggregate_gate}, indent=2))


if __name__ == "__main__":
    main()
