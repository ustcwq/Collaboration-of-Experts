from __future__ import annotations

import argparse
import json
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
from .cpi_remaining import METHODS, PRIMARY_METHOD
from .data import CacheAdapter, load_family_map
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap, holm_adjust, selection_correctness
from .schema import EvaluationLabels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate authenticated remaining-source CPI experiments")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_hashes(manifest: dict[str, Any], run_dir: Path) -> None:
    for raw_path, expected in manifest.get("artifact_hashes", {}).items():
        path = Path(raw_path)
        try:
            path.resolve().relative_to(run_dir.resolve())
        except ValueError as error:
            raise RuntimeError(f"Completion manifest references an external path: {path}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Remaining-source artifact hash mismatch: {path}")


def _vector(selections, labels) -> tuple[list[str], np.ndarray]:
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
        raise RuntimeError("Current code does not match the remaining-source test receipt")
    seeds = [int(value) for value in config["seeds"]]
    gpu_by_seed = seed_gpu_map(config, "physical_gpus")
    base_gpu_by_seed = seed_gpu_map(config, "base_physical_gpus")
    run_dirs = sorted(path for path in args.run_root.glob("seed_*_gpu*") if path.is_dir())
    if len(run_dirs) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} completed runs, found {len(run_dirs)}")

    runs: dict[int, dict[str, Any]] = {}
    for run_dir in run_dirs:
        prediction_manifest_path = run_dir / "prediction_manifest.json"
        completion_path = run_dir / "run_complete_manifest.json"
        if not prediction_manifest_path.is_file() or not completion_path.is_file():
            raise RuntimeError(f"Incomplete remaining-source run: {run_dir}")
        prediction_manifest = _load(prediction_manifest_path)
        completion = _load(completion_path)
        _validate_hashes(completion, run_dir)
        seed = int(prediction_manifest["seed"])
        if seed in runs or seed not in gpu_by_seed:
            raise RuntimeError(f"Unexpected or duplicate seed: {seed}")
        if int(prediction_manifest["physical_gpu"]) != gpu_by_seed[seed]:
            raise RuntimeError(f"Seed {seed} ran on the wrong physical GPU")
        if int(completion["seed"]) != seed or int(completion["physical_gpu"]) != gpu_by_seed[seed]:
            raise RuntimeError(f"Seed {seed} completion metadata is inconsistent")
        if sha256_file(prediction_manifest_path) != completion["prediction_manifest_sha256"]:
            raise RuntimeError(f"Seed {seed} prediction manifest hash mismatch")
        if prediction_manifest["innovation_code_manifest_sha256"] != current_code:
            raise RuntimeError(f"Seed {seed} did not use the tested code")
        if tuple(prediction_manifest["active_methods"]) != METHODS:
            raise RuntimeError(f"Seed {seed} did not run the frozen method family")
        if int(prediction_manifest["source_questions"]) != int(config["expected_source_questions"]):
            raise RuntimeError(f"Seed {seed} did not predict every source question")
        if int(prediction_manifest["source_environments"]) != int(config["expected_source_environments"]):
            raise RuntimeError(f"Seed {seed} did not evaluate every source environment")
        expected_structure = {
            "none_correct": int(config["expected_none_correct_questions"]),
            "one_correct": int(config["expected_one_correct_questions"]),
        }
        if prediction_manifest["source_label_structure"] != expected_structure:
            raise RuntimeError(f"Seed {seed} source-label structure changed")
        for input_path, expected in prediction_manifest["input_hashes"].items():
            if sha256_file(Path(input_path)) != expected:
                raise RuntimeError(f"Seed {seed} input changed: {input_path}")
        selections = {}
        for method in METHODS:
            path = run_dir / "predictions" / f"{method}.jsonl"
            expected = prediction_manifest["prediction_hashes_before_evaluation"][method]
            if sha256_file(path) != expected or completion["artifact_hashes"].get(str(path)) != expected:
                raise RuntimeError(f"Seed {seed} prediction hash mismatch for {method}")
            selections[method] = read_selections(path)
        base_dir = Path(config["base_run_root"]) / f"seed_{seed}_gpu{base_gpu_by_seed[seed]}"
        source_best_path = base_dir / "predictions" / "source_best_single.jsonl"
        expected_source_best = str(config["base_artifacts"][seed]["source_best_single_sha256"])
        if sha256_file(source_best_path) != expected_source_best:
            raise RuntimeError(f"Seed {seed} frozen Source-Best input changed")
        expected_determinism = {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(seed),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
        if completion.get("determinism") != expected_determinism:
            raise RuntimeError(f"Seed {seed} lacks deterministic GPU evidence")
        runs[seed] = {
            "dir": run_dir,
            "manifest": prediction_manifest,
            "selections": selections,
            "baseline": read_selections(source_best_path),
            "resource": _load(run_dir / "resource_usage.json"),
            "invariance": run_dir / "invariance.csv",
        }
    if sorted(runs) != sorted(seeds):
        raise RuntimeError("Remaining-source seed set is incomplete")

    source = config["source"]
    family_map = load_family_map(Path(config["family_map"]))
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    source_labels = adapter.load_source_labels()
    labels = EvaluationLabels(source_labels.dataset, source_labels.split, dict(source_labels.correctness))

    baseline_vectors: list[np.ndarray] = []
    method_vectors: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    reference_ids: list[str] | None = None
    environment_rows: list[dict[str, Any]] = []
    for seed in seeds:
        ids, baseline = _vector(runs[seed]["baseline"], labels)
        if reference_ids is not None and ids != reference_ids:
            raise RuntimeError("Frozen Source-Best predictions are not aligned across seeds")
        reference_ids = ids
        baseline_vectors.append(baseline)
        for method in METHODS:
            method_ids, candidate = _vector(runs[seed]["selections"][method], labels)
            if method_ids != ids:
                raise RuntimeError(f"Seed {seed} predictions are not aligned for {method}")
            method_vectors[method].append(candidate)
            for environment in sorted(set(source_labels.environment_by_question.values())):
                indices = [
                    index
                    for index, question_id in enumerate(ids)
                    if source_labels.environment_by_question[question_id] == environment
                ]
                environment_rows.append(
                    {
                        "seed": seed,
                        "physical_gpu": gpu_by_seed[seed],
                        "method": method,
                        "environment": environment,
                        "samples": len(indices),
                        "delta": float((candidate[indices] - baseline[indices]).mean()),
                    }
                )
    baseline_matrix = np.stack(baseline_vectors)
    method_results: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for method in METHODS:
        matrix = np.stack(method_vectors[method])
        relevant = [row for row in environment_rows if row["method"] == method]
        environment_deltas = [float(row["delta"]) for row in relevant]
        ci = hierarchical_paired_bootstrap(
            matrix,
            baseline_matrix,
            seed=int(config["aggregate_bootstrap_seed"]),
            samples=int(config["aggregate_bootstrap_samples"]),
        )
        rescue, harm, p_value = exact_mcnemar(matrix.ravel(), baseline_matrix.ravel())
        raw_p_values[method] = p_value
        method_results[method] = {
            "accuracy_mean": float(matrix.mean()),
            "source_best_accuracy_mean": float(baseline_matrix.mean()),
            "micro_delta": float((matrix - baseline_matrix).mean()),
            "macro_delta": float(np.mean(environment_deltas)),
            "worst_seed_environment_delta": min(environment_deltas),
            "nonnegative_seed_environment_fraction": float(np.mean([value >= 0.0 for value in environment_deltas])),
            "crossed_seed_query_bootstrap_ci95": list(ci),
            "pooled_seed_query_rescue": rescue,
            "pooled_seed_query_harm": harm,
            "pooled_seed_query_mcnemar_p_descriptive": p_value,
        }
    holm = holm_adjust(raw_p_values)
    for method, adjusted in holm.items():
        method_results[method]["holm_vs_source_best"] = adjusted

    def contrast(candidate: str, reference: str) -> dict[str, Any]:
        candidate_matrix = np.stack(method_vectors[candidate])
        reference_matrix = np.stack(method_vectors[reference])
        ci = hierarchical_paired_bootstrap(
            candidate_matrix,
            reference_matrix,
            seed=int(config["aggregate_bootstrap_seed"]) + 1,
            samples=int(config["aggregate_bootstrap_samples"]),
        )
        rescue, harm, p_value = exact_mcnemar(candidate_matrix.ravel(), reference_matrix.ravel())
        return {
            "candidate": candidate,
            "reference": reference,
            "accuracy_delta": float((candidate_matrix - reference_matrix).mean()),
            "crossed_seed_query_bootstrap_ci95": list(ci),
            "rescue": rescue,
            "harm": harm,
            "exact_mcnemar_p_descriptive": p_value,
        }

    factorial_contrasts = {
        "mask_main_mean_fallback": contrast(
            "factor_mask_mean__none_fallback", "factor_legacy_mean__none_fallback"
        ),
        "rich_main_mean_fallback": contrast(
            "factor_rich_mean__none_fallback", "factor_legacy_mean__none_fallback"
        ),
        "dro_main_legacy_fallback": contrast(
            "factor_legacy_dro__none_fallback", "factor_legacy_mean__none_fallback"
        ),
        "dro_on_rich_mask_fallback": contrast(
            "factor_rich_mask_dro__none_fallback", "factor_rich_mask_mean__none_fallback"
        ),
        "full_combination_vs_legacy": contrast(
            PRIMARY_METHOD, "factor_legacy_mean__none_fallback"
        ),
    }
    primary = method_results[PRIMARY_METHOD]
    gate = {
        "seeds": seeds,
        "primary_method": PRIMARY_METHOD,
        **primary,
        "method_results": method_results,
        "factorial_contrasts": factorial_contrasts,
        "required_macro_delta": float(config["required_macro_delta"]),
        "required_worst_delta": float(config["required_worst_delta"]),
        "required_nonnegative_fraction": float(config["required_nonnegative_fraction"]),
        "property_tests": {
            "receipt": str(receipt_path),
            "test_count": receipt["test_count"],
            "exit_code": receipt["exit_code"],
        },
        "derived_from": "hashed source-LOSO predictions and registry-validated labels",
    }
    gate["decision"] = (
        "GO"
        if gate["macro_delta"] >= gate["required_macro_delta"]
        and gate["worst_seed_environment_delta"] >= gate["required_worst_delta"]
        and gate["nonnegative_seed_environment_fraction"] >= gate["required_nonnegative_fraction"]
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
    write_json(args.output_dir / "gate.json", gate)
    write_csv(
        args.output_dir / "method_results.csv",
        [{"method": method, **result} for method, result in method_results.items()],
    )
    write_csv(args.output_dir / "environment_results.csv", environment_rows)
    write_csv(args.output_dir / "resources.csv", resource_rows)
    write_json(
        args.output_dir / "authenticated_inputs.json",
        {
            "config": str(args.config),
            "seeds": seeds,
            "physical_gpu_by_seed": gpu_by_seed,
            "base_physical_gpu_by_seed": base_gpu_by_seed,
            "innovation_code_manifest_sha256": current_code,
            "test_receipt": str(receipt_path),
            "per_seed_input_manifest_sha256": {
                str(seed): runs[seed]["manifest"]["input_manifest_sha256"] for seed in seeds
            },
        },
    )
    write_json(
        args.output_dir / "aggregate_complete_manifest.json",
        {"artifact_hashes": files_manifest([args.output_dir])},
    )
    print(json.dumps({"output_dir": str(args.output_dir), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
