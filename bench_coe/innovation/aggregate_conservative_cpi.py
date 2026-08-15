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
from .schema import EvaluationLabels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate authenticated Conservative-CPI seeds")
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
            raise RuntimeError(f"Completion manifest references an external path: {path}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Conservative-CPI artifact hash mismatch: {path}")


def _correctness(selections, labels) -> tuple[list[str], np.ndarray]:
    values = selection_correctness(selections, labels)
    ids = sorted(values)
    return ids, np.asarray([float(values[qid]) for qid in ids])


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    receipt = validate_test_receipt(receipt_path, args.config)
    current_code = manifest_sha256(innovation_code_manifest())
    if current_code != receipt["code_manifest_sha256"]:
        raise RuntimeError("Current code does not match the Conservative-CPI test receipt")
    seeds = [int(value) for value in config["seeds"]]
    expected_gpu = seed_gpu_map(config, "physical_gpus")
    base_gpu = seed_gpu_map(config, "base_physical_gpus")
    run_dirs = sorted(path for path in args.run_root.glob("seed_*_gpu*") if path.is_dir())
    if len(run_dirs) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} completed runs, found {len(run_dirs)}")

    runs: dict[int, dict[str, Any]] = {}
    for run_dir in run_dirs:
        prediction_manifest_path = run_dir / "prediction_manifest.json"
        completion_path = run_dir / "run_complete_manifest.json"
        if not prediction_manifest_path.exists() or not completion_path.exists():
            raise RuntimeError(f"Incomplete Conservative-CPI run: {run_dir}")
        prediction_manifest = _load(prediction_manifest_path)
        completion = _load(completion_path)
        _validate_hashes(completion, run_dir)
        seed = int(prediction_manifest["seed"])
        if seed in runs or seed not in expected_gpu:
            raise RuntimeError(f"Unexpected or duplicate seed: {seed}")
        if int(prediction_manifest["physical_gpu"]) != expected_gpu[seed]:
            raise RuntimeError(f"Seed {seed} has the wrong physical GPU")
        if int(completion["seed"]) != seed or int(completion["physical_gpu"]) != expected_gpu[seed]:
            raise RuntimeError(f"Seed {seed} completion metadata is inconsistent")
        if sha256_file(prediction_manifest_path) != completion["prediction_manifest_sha256"]:
            raise RuntimeError(f"Seed {seed} prediction manifest hash mismatch")
        if prediction_manifest["innovation_code_manifest_sha256"] != current_code:
            raise RuntimeError(f"Seed {seed} did not use the tested code")
        for input_path, expected_hash in prediction_manifest["input_hashes"].items():
            if sha256_file(Path(input_path)) != expected_hash:
                raise RuntimeError(f"Seed {seed} input changed: {input_path}")
        if int(prediction_manifest["source_questions"]) != int(config["expected_source_questions"]):
            raise RuntimeError(f"Seed {seed} did not predict all source questions")
        if int(prediction_manifest["source_environments"]) != int(config["expected_source_environments"]):
            raise RuntimeError(f"Seed {seed} did not evaluate all source environments")
        prediction_path = run_dir / "predictions" / "conservative_cpi.jsonl"
        expected_prediction_hash = prediction_manifest["prediction_hashes_before_evaluation"]["conservative_cpi"]
        if sha256_file(prediction_path) != expected_prediction_hash:
            raise RuntimeError(f"Seed {seed} conservative prediction hash mismatch")
        if completion["artifact_hashes"].get(str(prediction_path)) != expected_prediction_hash:
            raise RuntimeError(f"Seed {seed} prediction is not in the completion manifest")
        deterministic = completion.get("determinism", {})
        if deterministic != {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(seed),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }:
            raise RuntimeError(f"Seed {seed} lacks deterministic GPU evidence")
        base_dir = Path(config["base_run_root"]) / f"seed_{seed}_gpu{base_gpu[seed]}"
        base_path = base_dir / "predictions" / "source_best_single.jsonl"
        expected_base_hash = str(config["base_artifacts"][seed]["source_best_single_sha256"])
        if sha256_file(base_path) != expected_base_hash:
            raise RuntimeError(f"Seed {seed} frozen Source-Best input changed")
        runs[seed] = {
            "dir": run_dir,
            "manifest": prediction_manifest,
            "candidate": read_selections(prediction_path),
            "baseline": read_selections(base_path),
            "thresholds": _load(run_dir / "selected_thresholds.json"),
            "resource": _load(run_dir / "resource_usage.json"),
        }
    if sorted(runs) != sorted(seeds):
        raise RuntimeError("Conservative-CPI seed set is incomplete")

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
    full_vectors: list[np.ndarray] = []
    baseline_vectors: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    seed_rows: list[dict[str, Any]] = []
    all_environment_deltas: list[float] = []
    threshold_counts: Counter[str] = Counter()
    for seed in seeds:
        candidate_ids, candidate = _correctness(runs[seed]["candidate"], labels)
        baseline_ids, baseline = _correctness(runs[seed]["baseline"], labels)
        if candidate_ids != baseline_ids or (reference_ids is not None and candidate_ids != reference_ids):
            raise RuntimeError("Conservative-CPI seed/query predictions are not aligned")
        reference_ids = candidate_ids
        full_vectors.append(candidate)
        baseline_vectors.append(baseline)
        environment_deltas: list[float] = []
        for environment in sorted(runs[seed]["thresholds"]):
            indices = [
                index
                for index, qid in enumerate(candidate_ids)
                if source_labels.environment_by_question[qid] == environment
            ]
            if not indices:
                raise RuntimeError(f"Seed {seed} threshold references an unknown environment: {environment}")
            delta = float((candidate[indices] - baseline[indices]).mean())
            environment_deltas.append(delta)
            all_environment_deltas.append(delta)
            threshold_counts[str(runs[seed]["thresholds"][environment])] += 1
        rescue, harm, p_value = exact_mcnemar(candidate, baseline)
        seed_rows.append(
            {
                "seed": seed,
                "accuracy": float(candidate.mean()),
                "source_best_accuracy": float(baseline.mean()),
                "micro_delta": float((candidate - baseline).mean()),
                "macro_delta": float(np.mean(environment_deltas)),
                "worst_environment_delta": min(environment_deltas),
                "nonnegative_environment_fraction": float(np.mean([value >= 0.0 for value in environment_deltas])),
                "rescue_count": rescue,
                "harm_count": harm,
                "exact_mcnemar_p": p_value,
            }
        )
    assert reference_ids is not None
    candidate_matrix = np.stack(full_vectors)
    baseline_matrix = np.stack(baseline_vectors)
    crossed_ci = hierarchical_paired_bootstrap(candidate_matrix, baseline_matrix, seed=20260809, samples=10000)
    rescue, harm, p_value = exact_mcnemar(candidate_matrix.ravel(), baseline_matrix.ravel())
    aggregate_gate = {
        "seeds": seeds,
        "accuracy_mean": float(candidate_matrix.mean()),
        "source_best_accuracy_mean": float(baseline_matrix.mean()),
        "micro_delta": float((candidate_matrix - baseline_matrix).mean()),
        "macro_delta": float(np.mean([row["macro_delta"] for row in seed_rows])),
        "worst_seed_environment_delta": min(all_environment_deltas),
        "nonnegative_seed_environment_fraction": float(np.mean([value >= 0.0 for value in all_environment_deltas])),
        "crossed_seed_query_bootstrap_ci95": list(crossed_ci),
        "pooled_seed_query_rescue": rescue,
        "pooled_seed_query_harm": harm,
        "pooled_seed_query_mcnemar_p_descriptive": p_value,
        "threshold_counts": dict(sorted(threshold_counts.items())),
        "required_macro_delta": float(config["required_macro_delta"]),
        "required_worst_delta": float(config["required_worst_delta"]),
        "required_nonnegative_fraction": float(config["required_nonnegative_fraction"]),
        "property_tests": {"receipt": str(receipt_path), "test_count": receipt["test_count"], "exit_code": receipt["exit_code"]},
        "derived_from": "hashed Conservative-CPI and frozen v5 Source-Best predictions plus registry-validated labels",
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
    write_csv(args.output_dir / "seed_results.csv", seed_rows)
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
