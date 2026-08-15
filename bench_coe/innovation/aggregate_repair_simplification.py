from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import (
    files_manifest,
    innovation_code_manifest,
    manifest_sha256,
    read_selections,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
)
from .data import EvaluationLabelAdapter
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap, holm_adjust, selection_correctness
from .repair_simplification import POOL_SHIFT_METHODS
from .schema import EvaluationLabels, Selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate four authenticated scoring-simplification GPU runs")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_completion(run_dir: Path, completion: dict[str, Any]) -> None:
    required = (
        run_dir / "prediction_manifest.json",
        run_dir / "summary.json",
        run_dir / "per_environment.csv",
        run_dir / "seed_gate.json",
        run_dir / "resource_usage.json",
    )
    hashes = completion.get("artifact_hashes", {})
    missing = [str(path) for path in required if str(path) not in hashes]
    if missing:
        raise RuntimeError(f"Completion manifest omits required artifacts: {missing}")
    root = run_dir.resolve()
    for path_string, expected in hashes.items():
        path = Path(path_string)
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Completion manifest references an external path: {path}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Completed artifact hash mismatch: {path}")
    expected_determinism = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": str(completion["seed"]),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if completion.get("determinism") != expected_determinism:
        raise RuntimeError(f"Incomplete deterministic CUDA evidence: {run_dir}")


def _correctness_vector(
    selections: list[Selection], labels: EvaluationLabels
) -> tuple[list[str], np.ndarray]:
    values = selection_correctness(selections, labels)
    question_ids = sorted(values)
    return question_ids, np.asarray([float(values[question_id]) for question_id in question_ids], dtype=float)


def _comparison(
    name: str,
    candidate_method: str,
    reference_method: str,
    runs: dict[int, dict[str, Any]],
    seeds: list[int],
    labels: EvaluationLabels,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    candidate_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    rescues: list[int] = []
    harms: list[int] = []
    p_values: list[float] = []
    for seed in seeds:
        candidate_ids, candidate = _correctness_vector(runs[seed]["predictions"][candidate_method], labels)
        baseline_ids, reference = _correctness_vector(runs[seed]["predictions"][reference_method], labels)
        if candidate_ids != baseline_ids or (reference_ids is not None and candidate_ids != reference_ids):
            raise RuntimeError(f"Unaligned aggregate comparison: {name}, seed={seed}")
        reference_ids = candidate_ids
        rescue, harm, p_value = exact_mcnemar(candidate, reference)
        candidate_rows.append(candidate)
        reference_rows.append(reference)
        rescues.append(rescue)
        harms.append(harm)
        p_values.append(p_value)
    candidate_matrix = np.stack(candidate_rows)
    reference_matrix = np.stack(reference_rows)
    ci = hierarchical_paired_bootstrap(
        candidate_matrix,
        reference_matrix,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    seed_deltas = (candidate_matrix - reference_matrix).mean(axis=1)
    return {
        "comparison": name,
        "candidate_method": candidate_method,
        "reference_method": reference_method,
        "seeds": len(seeds),
        "queries_per_seed": candidate_matrix.shape[1],
        "candidate_accuracy_mean": float(candidate_matrix.mean()),
        "reference_accuracy_mean": float(reference_matrix.mean()),
        "delta_mean": float(seed_deltas.mean()),
        "delta_std": float(seed_deltas.std(ddof=1)),
        "hierarchical_paired_bootstrap_delta_ci95": list(ci),
        "rescue_count_mean": float(np.mean(rescues)),
        "harm_count_mean": float(np.mean(harms)),
        "exact_mcnemar_p_max_across_seeds": float(max(p_values)),
        "seed_deltas": {str(seed): float(delta) for seed, delta in zip(seeds, seed_deltas, strict=True)},
        "note": "Seeds verify hardware/random-control reproducibility; deterministic formulas are not independent datasets.",
    }


def _aggregate_environment_rows(
    runs: dict[int, dict[str, Any]], seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    samples: dict[tuple[str, str], set[int]] = defaultdict(set)
    for seed in seeds:
        path = runs[seed]["dir"] / "per_environment.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["environment"]), str(row["method"]))
                values[key].append(float(row["accuracy"]))
                samples[key].add(int(row["samples"]))
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], float] = {}
    for key, accuracies in sorted(values.items()):
        if len(accuracies) != len(seeds) or len(samples[key]) != 1:
            raise RuntimeError(f"Incomplete environment aggregation for {key}")
        lookup[key] = float(np.mean(accuracies))
        rows.append(
            {
                "environment": key[0],
                "method": key[1],
                "samples": next(iter(samples[key])),
                "accuracy_mean": lookup[key],
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "seeds": len(seeds),
            }
        )
    return rows, lookup


def _formula_gate(
    comparison: dict[str, Any],
    environment_lookup: dict[tuple[str, str], float],
    environments: list[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    method = str(comparison["candidate_method"])
    reference = str(comparison["reference_method"])
    deltas = [
        environment_lookup[(environment, method)] - environment_lookup[(environment, reference)]
        for environment in environments
    ]
    result = {
        "method": method,
        "reference": reference,
        "micro_delta": float(comparison["delta_mean"]),
        "macro_environment_delta": float(np.mean(deltas)),
        "worst_environment_delta": float(min(deltas)),
        "nonnegative_environment_fraction": float(np.mean(np.asarray(deltas) >= 0.0)),
        "hierarchical_paired_bootstrap_delta_ci95": comparison[
            "hierarchical_paired_bootstrap_delta_ci95"
        ],
        "rescue_count_mean": float(comparison["rescue_count_mean"]),
        "harm_count_mean": float(comparison["harm_count_mean"]),
        "required_micro_delta": float(thresholds["required_micro_delta"]),
        "required_worst_environment_delta": float(thresholds["required_worst_environment_delta"]),
        "required_nonnegative_environment_fraction": float(
            thresholds["required_nonnegative_environment_fraction"]
        ),
        "ci_noninferiority_margin": float(thresholds["ci_noninferiority_margin"]),
    }
    result["decision"] = (
        "GO"
        if result["micro_delta"] >= result["required_micro_delta"]
        and result["worst_environment_delta"] >= result["required_worst_environment_delta"]
        and result["nonnegative_environment_fraction"]
        >= result["required_nonnegative_environment_fraction"]
        and float(result["hierarchical_paired_bootstrap_delta_ci95"][0])
        >= -result["ci_noninferiority_margin"]
        else "NO-GO"
    )
    return result


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    receipt = validate_test_receipt(receipt_path, args.config)
    seeds = [int(value) for value in config["seeds"]]
    physical_gpus = [int(value) for value in config["run_physical_gpus"]]
    expected_gpu = dict(zip(seeds, physical_gpus, strict=True))
    current_code_sha = manifest_sha256(innovation_code_manifest())
    source = config["source"]
    current_inputs = files_manifest(
        [
            args.config,
            Path(config["family_map"]),
            Path(config["dataset_registry"]),
            Path(source["cache_path"]),
            receipt_path,
        ]
    )
    current_input_sha = manifest_sha256(current_inputs)

    run_dirs = sorted(path for path in args.run_root.glob("seed_*_gpu*") if path.is_dir())
    if len(run_dirs) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} seed directories, found {len(run_dirs)}")
    runs: dict[int, dict[str, Any]] = {}
    for run_dir in run_dirs:
        completion = _load(run_dir / "run_complete_manifest.json")
        _validate_completion(run_dir, completion)
        manifest_path = run_dir / "prediction_manifest.json"
        manifest = _load(manifest_path)
        if sha256_file(manifest_path) != completion["prediction_manifest_sha256"]:
            raise RuntimeError(f"Prediction manifest hash mismatch: {run_dir}")
        seed = int(manifest["seed"])
        if seed in runs or seed not in expected_gpu:
            raise RuntimeError(f"Unexpected or duplicate seed: {seed}")
        if int(manifest["physical_gpu"]) != expected_gpu[seed] or int(completion["physical_gpu"]) != expected_gpu[seed]:
            raise RuntimeError(f"Seed {seed} ran on an unexpected physical GPU")
        if int(manifest["source_questions"]) != int(config["expected_source_questions"]):
            raise RuntimeError(f"Seed {seed} has an incomplete source query set")
        if int(manifest["source_environments"]) != int(config["expected_source_environments"]):
            raise RuntimeError(f"Seed {seed} has an incomplete environment set")
        if manifest["innovation_code_manifest_sha256"] != current_code_sha:
            raise RuntimeError(f"Seed {seed} code snapshot differs from current code")
        if current_code_sha != receipt["code_manifest_sha256"]:
            raise RuntimeError("Current code differs from the passing test receipt")
        if manifest["input_manifest_sha256"] != current_input_sha:
            raise RuntimeError(f"Seed {seed} input snapshot differs from authenticated inputs")
        predictions: dict[str, list[Selection]] = {}
        for method, expected_hash in manifest["prediction_hashes_before_evaluation"].items():
            path = run_dir / "predictions" / f"{method}.jsonl"
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Prediction hash mismatch: seed={seed}, method={method}")
            if completion["artifact_hashes"].get(str(path)) != expected_hash:
                raise RuntimeError(f"Prediction is not bound to completion: seed={seed}, method={method}")
            predictions[str(method)] = read_selections(path)
        runs[seed] = {
            "dir": run_dir,
            "manifest": manifest,
            "completion": completion,
            "predictions": predictions,
            "resource": _load(run_dir / "resource_usage.json"),
            "seed_gate": _load(run_dir / "seed_gate.json"),
        }
    if sorted(runs) != sorted(seeds):
        raise RuntimeError("Authenticated seed set does not match the frozen config")
    if len({run["manifest"]["source_question_ids_sha256"] for run in runs.values()}) != 1:
        raise RuntimeError("Seed query sets differ")
    method_sets = {tuple(sorted(run["predictions"])) for run in runs.values()}
    if len(method_sets) != 1:
        raise RuntimeError("Seed method sets differ")
    for method in config["deterministic_methods"]:
        hashes = {
            run["manifest"]["prediction_hashes_before_evaluation"][str(method)] for run in runs.values()
        }
        if len(hashes) != 1:
            raise RuntimeError(f"Deterministic method differs across GPUs/seeds: {method}")

    # Labels are opened only after all four completed prediction packages have been authenticated.
    evaluation_labels = EvaluationLabelAdapter.from_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    ).load()
    reference_ids: list[str] | None = None
    for seed in seeds:
        ids, _ = _correctness_vector(runs[seed]["predictions"]["source_best_single"], evaluation_labels)
        if reference_ids is not None and ids != reference_ids:
            raise RuntimeError("Source query IDs differ across seeds")
        reference_ids = ids
    assert reference_ids is not None
    question_ids_sha = hashlib.sha256("\n".join(reference_ids).encode("utf-8")).hexdigest()
    if question_ids_sha != next(iter(runs.values()))["manifest"]["source_question_ids_sha256"]:
        raise RuntimeError("Evaluator query IDs differ from the prediction manifest")

    methods = sorted(next(iter(runs.values()))["predictions"])
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        accuracies: list[float] = []
        deltas: list[float] = []
        rescues: list[int] = []
        harms: list[int] = []
        for seed in seeds:
            _, candidate = _correctness_vector(runs[seed]["predictions"][method], evaluation_labels)
            _, baseline = _correctness_vector(
                runs[seed]["predictions"]["source_best_single"], evaluation_labels
            )
            rescue, harm, _ = exact_mcnemar(candidate, baseline)
            accuracies.append(float(candidate.mean()))
            deltas.append(float((candidate - baseline).mean()))
            rescues.append(rescue)
            harms.append(harm)
        summary_rows.append(
            {
                "method": method,
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "delta_vs_source_best_mean": float(np.mean(deltas)),
                "rescue_count_mean": float(np.mean(rescues)),
                "harm_count_mean": float(np.mean(harms)),
                "seeds": len(seeds),
            }
        )

    pool_names = tuple(sorted(str(name) for name in config["pool_shift"]["pools"]))
    primary_methods = [
        method for method in methods if not any(method.startswith(f"{name}__") for name in pool_names)
    ]
    comparison_specs = [
        (f"{method}_vs_m0_full", method, "m0_full")
        for method in primary_methods
        if method != "m0_full"
    ]
    for pool_name in pool_names:
        reference = f"{pool_name}__m0_full"
        comparison_specs.extend(
            (f"{method}_vs_{reference}", method, reference)
            for method in methods
            if method.startswith(f"{pool_name}__") and method != reference
        )
    original_pool = str(config["pool_shift"]["original_pool"])
    replacement_pool = str(config["pool_shift"]["replacement_pool"])
    comparison_specs.extend(
        (
            f"pool_shift_{replacement_pool}_vs_{original_pool}__{method}",
            f"{replacement_pool}__{method}",
            f"{original_pool}__{method}",
        )
        for method in ("m0_full", "m3_h1_support", "m3_cluster_h1_support", "m4_h1", "m5_h1_h2")
    )
    comparison_specs.append(("m5_h1_h2_vs_m4_h1", "m5_h1_h2", "m4_h1"))
    comparison_specs.extend(
        [
            ("m0_full_vs_m1_no_local", "m0_full", "m1_no_local"),
            ("m1_no_local_vs_m2_no_local_no_global", "m1_no_local", "m2_no_local_no_global"),
            ("m3_h1_support_vs_m4_h1", "m3_h1_support", "m4_h1"),
            ("m4_h1_vs_m6_support", "m4_h1", "m6_support"),
        ]
    )
    seen: set[str] = set()
    comparisons: list[dict[str, Any]] = []
    for index, (name, candidate, reference) in enumerate(comparison_specs):
        if name in seen:
            continue
        seen.add(name)
        comparisons.append(
            _comparison(
                name,
                candidate,
                reference,
                runs,
                seeds,
                evaluation_labels,
                bootstrap_seed=int(config["aggregate_bootstrap_seed"]) + index,
                bootstrap_samples=int(config["aggregate_bootstrap_samples"]),
            )
        )
    holm = holm_adjust(
        {
            str(row["comparison"]): float(row["exact_mcnemar_p_max_across_seeds"])
            for row in comparisons
        }
    )
    for row in comparisons:
        row["holm"] = holm[str(row["comparison"])]
    comparison_by_name = {str(row["comparison"]): row for row in comparisons}

    environment_rows, environment_lookup = _aggregate_environment_rows(runs, seeds)
    environments = sorted({environment for environment, _ in environment_lookup})
    thresholds = config["simplification_gate"]
    m4_gate = _formula_gate(
        comparison_by_name["m4_h1_vs_m0_full"], environment_lookup, environments, thresholds
    )
    m3_gate = _formula_gate(
        comparison_by_name["m3_cluster_h1_support_vs_m0_full"],
        environment_lookup,
        environments,
        thresholds,
    )
    required_pool_delta = float(config["pool_shift"]["required_delta_vs_pool_m0"])

    def pool_gate(method: str) -> dict[str, Any]:
        deltas = {
            pool_name: float(
                comparison_by_name[
                    f"{pool_name}__{method}_vs_{pool_name}__m0_full"
                ]["delta_mean"]
            )
            for pool_name in pool_names
        }
        return {
            "method": method,
            "delta_vs_pool_m0": deltas,
            "required_delta_vs_pool_m0": required_pool_delta,
            "decision": "PASS" if min(deltas.values()) >= required_pool_delta else "FAIL",
        }

    m4_pool_gate = pool_gate("m4_h1")
    m3_pool_gate = pool_gate("m3_cluster_h1_support")
    if m4_gate["decision"] == "GO" and m4_pool_gate["decision"] == "PASS":
        decision = "GO"
        selected_formula: str | None = "m4_h1"
    elif m3_gate["decision"] == "GO" and m3_pool_gate["decision"] == "PASS":
        decision = "GO"
        selected_formula = "m3_cluster_h1_support"
    else:
        decision = "NO-GO"
        selected_formula = None

    h2_comparison = comparison_by_name["m5_h1_h2_vs_m4_h1"]
    h2_environment_deltas = [
        environment_lookup[(environment, "m5_h1_h2")]
        - environment_lookup[(environment, "m4_h1")]
        for environment in environments
    ]
    accuracy_by_method = {str(row["method"]): float(row["accuracy_mean"]) for row in summary_rows}
    controls = (
        "m5_h1_h2_randomized",
        "m5_h1_h2_symmetric",
        "m5_h1_h2_no_self",
        "m5_h1_h2_centrality",
    )
    h2_gate = {
        "comparison": "m5_h1_h2_vs_m4_h1",
        "micro_delta": float(h2_comparison["delta_mean"]),
        "worst_environment_delta": float(min(h2_environment_deltas)),
        "nonnegative_environment_fraction": float(np.mean(np.asarray(h2_environment_deltas) >= 0.0)),
        "hierarchical_paired_bootstrap_delta_ci95": h2_comparison[
            "hierarchical_paired_bootstrap_delta_ci95"
        ],
        "real_accuracy": accuracy_by_method["m5_h1_h2"],
        "control_accuracies": {method: accuracy_by_method[method] for method in controls},
    }
    h2_gate["decision"] = (
        "RETAIN"
        if h2_gate["micro_delta"] > 0.0
        and h2_gate["worst_environment_delta"] >= 0.0
        and float(h2_gate["hierarchical_paired_bootstrap_delta_ci95"][0]) > 0.0
        and all(h2_gate["real_accuracy"] > value for value in h2_gate["control_accuracies"].values())
        else "DELETE"
    )
    aggregate_gate = {
        "decision": decision,
        "selected_formula": selected_formula,
        "development_ood_authorized": decision == "GO",
        "m4_pure_h1": m4_gate,
        "m3_cluster_h1_support": m3_gate,
        "m4_pool_shift": m4_pool_gate,
        "m3_pool_shift": m3_pool_gate,
        "h2_retention": h2_gate,
        "m0_vs_legacy_answer_mismatch_counts": {
            str(seed): int(runs[seed]["seed_gate"]["m0_vs_legacy_answer_mismatch_count"])
            for seed in seeds
        },
    }

    component_specs = (
        ("L", "m0_full_vs_m1_no_local"),
        ("direct_G", "m1_no_local_vs_m2_no_local_no_global"),
        ("A_with_nested_beta", "m3_h1_support_vs_m4_h1"),
        ("H2", "m5_h1_h2_vs_m4_h1"),
        ("H1_vs_support_only", "m4_h1_vs_m6_support"),
    )
    component_rows: list[dict[str, Any]] = []
    for component, comparison_name in component_specs:
        row = comparison_by_name[comparison_name]
        ci = row["hierarchical_paired_bootstrap_delta_ci95"]
        component_rows.append(
            {
                "component": component,
                "comparison": comparison_name,
                "delta": row["delta_mean"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "rescue_count_mean": row["rescue_count_mean"],
                "harm_count_mean": row["harm_count_mean"],
                "empirically_justified": bool(
                    float(row["delta_mean"]) > 0.0
                    and float(ci[0]) > 0.0
                    and float(row["rescue_count_mean"]) > float(row["harm_count_mean"])
                ),
            }
        )

    pool_shift_rows: list[dict[str, Any]] = []
    for method in POOL_SHIFT_METHODS:
        comparison_name = f"pool_shift_{replacement_pool}_vs_{original_pool}__{method}"
        if comparison_name not in comparison_by_name:
            # The five key shift comparisons carry paired CIs; all M0-M8 rows
            # still receive exact aggregate accuracies below.
            comparison = None
        else:
            comparison = comparison_by_name[comparison_name]
        original_method = f"{original_pool}__{method}"
        replacement_method = f"{replacement_pool}__{method}"
        pool_shift_rows.append(
            {
                "method": method,
                "original_pool": original_pool,
                "replacement_pool": replacement_pool,
                "original_accuracy": accuracy_by_method[original_method],
                "replacement_accuracy": accuracy_by_method[replacement_method],
                "raw_replacement_delta": (
                    float(comparison["delta_mean"])
                    if comparison is not None
                    else accuracy_by_method[replacement_method] - accuracy_by_method[original_method]
                ),
                "paired_ci_low": (
                    comparison["hierarchical_paired_bootstrap_delta_ci95"][0]
                    if comparison is not None
                    else None
                ),
                "paired_ci_high": (
                    comparison["hierarchical_paired_bootstrap_delta_ci95"][1]
                    if comparison is not None
                    else None
                ),
                "original_delta_vs_pool_m0": (
                    accuracy_by_method[original_method]
                    - accuracy_by_method[f"{original_pool}__m0_full"]
                ),
                "replacement_delta_vs_pool_m0": (
                    accuracy_by_method[replacement_method]
                    - accuracy_by_method[f"{replacement_pool}__m0_full"]
                ),
            }
        )

    selected_parameters: Counter[tuple[str, str]] = Counter()
    for seed in seeds:
        path = runs[seed]["dir"] / "nested_parameter_search.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row["selected"]).lower() == "true":
                    selected_parameters[(str(row["parameter"]), str(row["value"]))] += 1
    parameter_rows = [
        {"parameter": key[0], "value": key[1], "outer_fold_seed_count": count}
        for key, count in sorted(selected_parameters.items())
    ]
    resource_rows = [
        {
            "seed": seed,
            "physical_gpu": runs[seed]["resource"]["physical_gpu"],
            "visible_device": runs[seed]["resource"]["visible_device"],
            "device_name": runs[seed]["resource"]["device_name"],
            "peak_allocated_bytes": runs[seed]["resource"]["peak_allocated_bytes"],
            "peak_reserved_bytes": runs[seed]["resource"]["peak_reserved_bytes"],
            "runtime_seconds": runs[seed]["resource"]["runtime_seconds"],
        }
        for seed in seeds
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "gate.json", aggregate_gate)
    write_json(args.output_dir / "aggregate_comparisons.json", comparisons)
    write_csv(
        args.output_dir / "aggregate_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in comparisons],
    )
    write_csv(args.output_dir / "aggregate_summary.csv", summary_rows)
    write_csv(args.output_dir / "aggregate_per_environment.csv", environment_rows)
    write_csv(args.output_dir / "component_decisions.csv", component_rows)
    write_csv(args.output_dir / "pool_shift_results.csv", pool_shift_rows)
    write_csv(args.output_dir / "selected_parameter_distribution.csv", parameter_rows)
    write_csv(args.output_dir / "resources.csv", resource_rows)
    write_json(
        args.output_dir / "authenticated_inputs.json",
        {
            "config": str(args.config),
            "seeds": seeds,
            "physical_gpu_by_seed": expected_gpu,
            "input_manifest_sha256": current_input_sha,
            "innovation_code_manifest_sha256": current_code_sha,
            "source_question_ids_sha256": question_ids_sha,
            "dataset_registry": str(config["dataset_registry"]),
            "dataset_registry_sha256": str(config["dataset_registry_sha256"]),
            "test_receipt": str(receipt_path),
            "evaluation_derivation": "hashed predictions authenticated before registry-validated labels were opened",
        },
    )
    write_json(
        args.output_dir / "aggregate_complete_manifest.json",
        {"artifact_hashes": files_manifest([args.output_dir])},
    )
    print(json.dumps({"output_dir": str(args.output_dir), "gate": aggregate_gate}, indent=2))


if __name__ == "__main__":
    main()
