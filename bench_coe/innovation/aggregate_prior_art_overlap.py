from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap, holm_adjust
from .schema import EvaluationLabels, Selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate authenticated Improve5/6 prior-art overlap runs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completion(run_dir: Path, completion: dict[str, Any]) -> None:
    for raw_path, expected_hash in completion.get("artifact_hashes", {}).items():
        path = Path(raw_path)
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Completion artifact hash mismatch: {path}")
        try:
            path.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError(f"Completion binds an artifact outside its run: {path}") from error


def correctness_vector(selections: list[Selection], labels: EvaluationLabels) -> tuple[list[str], np.ndarray]:
    ordered = sorted(selections, key=lambda selection: selection.question_id)
    ids = [selection.question_id for selection in ordered]
    values = np.asarray(
        [bool(labels.get(selection.question_id, selection.selected_expert_id or "")) for selection in ordered],
        dtype=int,
    )
    return ids, values


def comparison(
    name: str,
    candidate_method: str,
    reference_method: str,
    runs: dict[int, dict[str, Any]],
    seeds: list[int],
    labels: EvaluationLabels,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    candidates: list[np.ndarray] = []
    references: list[np.ndarray] = []
    rescues: list[int] = []
    harms: list[int] = []
    p_values: list[float] = []
    reference_ids: list[str] | None = None
    for seed in seeds:
        ids, candidate = correctness_vector(runs[seed]["predictions"][candidate_method], labels)
        reference_ids_for_seed, reference = correctness_vector(runs[seed]["predictions"][reference_method], labels)
        if ids != reference_ids_for_seed or (reference_ids is not None and ids != reference_ids):
            raise RuntimeError(f"Unaligned paired comparison: {name}")
        reference_ids = ids
        rescue, harm, p_value = exact_mcnemar(candidate, reference)
        candidates.append(candidate)
        references.append(reference)
        rescues.append(rescue)
        harms.append(harm)
        p_values.append(p_value)
    candidate_matrix = np.stack(candidates)
    reference_matrix = np.stack(references)
    ci = hierarchical_paired_bootstrap(
        candidate_matrix,
        reference_matrix,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    return {
        "comparison": name,
        "candidate": candidate_method,
        "reference": reference_method,
        "samples": candidate_matrix.shape[1],
        "seeds": candidate_matrix.shape[0],
        "candidate_accuracy_mean": float(candidate_matrix.mean()),
        "reference_accuracy_mean": float(reference_matrix.mean()),
        "delta_mean": float((candidate_matrix - reference_matrix).mean()),
        "rescue_count_mean": float(np.mean(rescues)),
        "harm_count_mean": float(np.mean(harms)),
        "exact_mcnemar_p_max_across_seeds": float(max(p_values)),
        "hierarchical_paired_bootstrap_delta_ci95": [float(ci[0]), float(ci[1])],
        "note": "seeds are reproducibility/null-control repeats; the query resample is shared across seeds",
    }


def aggregate_environment_rows(
    runs: dict[int, dict[str, Any]], seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    samples: dict[tuple[str, str], set[int]] = defaultdict(set)
    pools: dict[tuple[str, str], str] = {}
    for seed in seeds:
        with (runs[seed]["dir"] / "per_environment.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["environment"]), str(row["method"]))
                values[key].append(float(row["accuracy"]))
                samples[key].add(int(row["samples"]))
                pools[key] = str(row["pool"])
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], float] = {}
    for key, observed in sorted(values.items()):
        if len(observed) != len(seeds) or len(samples[key]) != 1:
            raise RuntimeError(f"Incomplete environment aggregate: {key}")
        mean = float(np.mean(observed))
        lookup[key] = mean
        rows.append(
            {
                "environment": key[0],
                "method": key[1],
                "pool": pools[key],
                "samples": next(iter(samples[key])),
                "accuracy_mean": mean,
                "accuracy_std": float(np.std(observed, ddof=1)),
            }
        )
    return rows, lookup


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    seeds = [int(value) for value in config["seeds"]]
    gpus = [int(value) for value in config["run_physical_gpus"]]
    expected_gpu = dict(zip(seeds, gpus, strict=True))
    current_code_hash = manifest_sha256(innovation_code_manifest())

    runs: dict[int, dict[str, Any]] = {}
    input_hashes: set[str] = set()
    for seed in seeds:
        gpu = expected_gpu[seed]
        run_dir = args.run_root / f"seed_{seed}_gpu{gpu}"
        completion = load_json(run_dir / "complete_manifest.json")
        validate_completion(run_dir, completion)
        manifest = load_json(run_dir / "prediction_manifest.json")
        environment = load_json(run_dir / "environment.json")
        if int(manifest["seed"]) != seed or int(manifest["physical_gpu"]) != gpu:
            raise RuntimeError(f"Seed/GPU identity mismatch: {run_dir}")
        if manifest["innovation_code_manifest_sha256"] != current_code_hash:
            raise RuntimeError(f"Run code snapshot differs from current authenticated code: {run_dir}")
        if environment["innovation_code_manifest_sha256"] != current_code_hash:
            raise RuntimeError(f"Environment code hash mismatch: {run_dir}")
        input_hashes.add(str(manifest["input_manifest_sha256"]))
        predictions: dict[str, list[Selection]] = {}
        for method, expected_hash in manifest["prediction_hashes_before_evaluation"].items():
            path = run_dir / "predictions" / f"{method}.jsonl"
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Prediction hash mismatch: seed={seed}, method={method}")
            if completion["artifact_hashes"].get(str(path)) != expected_hash:
                raise RuntimeError(f"Prediction is not bound to completion: seed={seed}, method={method}")
            predictions[str(method)] = read_selections(path)
        if int(manifest["method_count"]) != len(predictions):
            raise RuntimeError(f"Method count mismatch: {run_dir}")
        runs[seed] = {
            "dir": run_dir,
            "manifest": manifest,
            "completion": completion,
            "predictions": predictions,
            "resource": load_json(run_dir / "resource_usage.json"),
        }
    if len(input_hashes) != 1:
        raise RuntimeError("Input snapshots differ across seeds")
    method_sets = {tuple(sorted(run["predictions"])) for run in runs.values()}
    if len(method_sets) != 1:
        raise RuntimeError("Method sets differ across seeds")
    question_hashes = {str(run["manifest"]["source_question_ids_sha256"]) for run in runs.values()}
    if len(question_hashes) != 1:
        raise RuntimeError("Question sets differ across seeds")
    methods = sorted(next(iter(runs.values()))["predictions"])
    stochastic_patterns = tuple(str(value) for value in config["stochastic_method_patterns"])
    for method in methods:
        if any(pattern in method for pattern in stochastic_patterns):
            continue
        hashes = {
            run["manifest"]["prediction_hashes_before_evaluation"][method] for run in runs.values()
        }
        if len(hashes) != 1:
            raise RuntimeError(f"Deterministic method differs across GPU/seed runs: {method}")

    source = config["source"]
    # Labels are opened only after all seed packages and prediction hashes authenticate.
    labels = EvaluationLabelAdapter.from_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    ).load()
    reference_ids, _ = correctness_vector(runs[seeds[0]]["predictions"]["global_best_posthoc"], labels)
    question_hash = hashlib.sha256("\n".join(reference_ids).encode("utf-8")).hexdigest()
    if question_hash not in question_hashes:
        raise RuntimeError("Evaluator question set differs from prediction manifests")

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        accuracies: list[float] = []
        for seed in seeds:
            _, values = correctness_vector(runs[seed]["predictions"][method], labels)
            accuracies.append(float(values.mean()))
        pool = str(runs[seeds[0]]["manifest"]["method_group"][method])
        reference = "global_best_posthoc" if pool == "full_pool" else f"{pool}__global_best_posthoc"
        reference_accuracies: list[float] = []
        rescues: list[int] = []
        harms: list[int] = []
        for seed in seeds:
            _, candidate = correctness_vector(runs[seed]["predictions"][method], labels)
            _, baseline = correctness_vector(runs[seed]["predictions"][reference], labels)
            rescue, harm, _ = exact_mcnemar(candidate, baseline)
            reference_accuracies.append(float(baseline.mean()))
            rescues.append(rescue)
            harms.append(harm)
        summary_rows.append(
            {
                "method": method,
                "pool": pool,
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "delta_vs_pool_global_best_mean": float(np.mean(accuracies) - np.mean(reference_accuracies)),
                "rescue_count_mean": float(np.mean(rescues)),
                "harm_count_mean": float(np.mean(harms)),
                "seeds": len(seeds),
            }
        )
    accuracy = {str(row["method"]): float(row["accuracy_mean"]) for row in summary_rows}

    pool_names = tuple(sorted(str(name) for name in config["pool_shift"]["pools"]))
    main_methods = [method for method in methods if not any(method.startswith(f"{pool}__") for pool in pool_names)]
    specifications: list[tuple[str, str, str]] = [
        (f"{method}_vs_global_best_posthoc", method, "global_best_posthoc")
        for method in main_methods
        if method != "global_best_posthoc"
    ]
    decisive = [
        ("fcrg_full_vs_knop", "fcrg_full", "knop_output_profile"),
        ("fcrg_full_vs_oprs", "fcrg_full", "oprs_robust_output_profile"),
        ("fcrg_full_vs_mcb", "fcrg_full", "mcb_dcs_structured"),
        ("fcrg_full_vs_more", "fcrg_full", "more_style_structured"),
        ("fcrg_full_vs_more_minilm", "fcrg_full", "more_style_minilm"),
        ("fcrg_full_vs_smoothie_local", "fcrg_full", "smoothie_local_spectral"),
        (
            "fcrg_full_vs_smoothie_local_minilm",
            "fcrg_full",
            "smoothie_local_minilm",
        ),
        ("fcrg_h1_h2_vs_h1", "fcrg_h1_h2", "fcrg_h1_only"),
        ("fcrg_full_vs_symmetric", "fcrg_full", "fcrg_symmetric"),
        ("fcrg_full_vs_random_edges", "fcrg_full", "fcrg_random_edges"),
        ("fcrg_full_vs_degree_relabel", "fcrg_full", "fcrg_degree_relabel"),
        ("fcrg_full_vs_column_mean", "fcrg_full", "fcrg_column_mean_only"),
        ("fcrg_full_vs_row_normalized", "fcrg_full", "fcrg_row_normalized"),
        ("fcrg_full_vs_column_normalized", "fcrg_full", "fcrg_column_normalized"),
        ("fcrg_full_vs_row_softmax", "fcrg_full", "fcrg_row_softmax"),
        ("fcrg_learned_vs_fixed", "fcrg_learned_weights", "fcrg_full"),
    ]
    specifications.extend(decisive)
    for pool in pool_names:
        reference = f"{pool}__global_best_posthoc"
        for method in config["pool_shift"]["methods"]:
            candidate = f"{pool}__{method}"
            if candidate != reference:
                specifications.append((f"{candidate}_vs_{reference}", candidate, reference))
    seen: set[str] = set()
    comparisons: list[dict[str, Any]] = []
    for index, (name, candidate, reference) in enumerate(specifications):
        if name in seen:
            continue
        seen.add(name)
        comparisons.append(
            comparison(
                name,
                candidate,
                reference,
                runs,
                seeds,
                labels,
                bootstrap_seed=int(config["aggregate_bootstrap_seed"]) + index,
                bootstrap_samples=int(config["aggregate_bootstrap_samples"]),
            )
        )
    corrections = holm_adjust(
        {str(row["comparison"]): float(row["exact_mcnemar_p_max_across_seeds"]) for row in comparisons}
    )
    for row in comparisons:
        row["holm"] = corrections[str(row["comparison"])]
    comparison_by_name = {str(row["comparison"]): row for row in comparisons}

    environment_rows, environment_lookup = aggregate_environment_rows(runs, seeds)
    environments = sorted(
        environment for environment, method in environment_lookup if method == "fcrg_full"
    )
    environment_deltas = [
        environment_lookup[(environment, "fcrg_full")]
        - environment_lookup[(environment, "global_best_posthoc")]
        for environment in environments
    ]
    primary = comparison_by_name["fcrg_full_vs_global_best_posthoc"]
    thresholds = config["fcrg_gate"]
    pool_deltas = {
        pool: accuracy[f"{pool}__fcrg_full"] - accuracy[f"{pool}__global_best_posthoc"]
        for pool in pool_names
    }
    graph_controls = [
        "fcrg_symmetric",
        "fcrg_random_edges",
        "fcrg_degree_relabel",
        "fcrg_column_mean_only",
        "fcrg_row_normalized",
        "fcrg_column_normalized",
        "fcrg_row_softmax",
    ]
    criteria = {
        "micro_delta_vs_global": float(primary["delta_mean"]) >= float(thresholds["required_micro_delta"]),
        "full_above_knop": accuracy["fcrg_full"] > accuracy["knop_output_profile"],
        "knop_above_global": accuracy["knop_output_profile"] > accuracy["global_best_posthoc"],
        "real_graph_above_all_controls": all(
            accuracy["fcrg_full"] > accuracy[method] for method in graph_controls
        ),
        "h2_increment_positive": accuracy["fcrg_h1_h2"] > accuracy["fcrg_h1_only"],
        "worst_environment_noninferior": min(environment_deltas) >= float(
            thresholds["required_worst_environment_delta"]
        ),
        "environment_stability": float(np.mean(np.asarray(environment_deltas) >= 0.0))
        >= float(thresholds["required_nonnegative_environment_fraction"]),
        "paired_ci_noninferior": float(primary["hierarchical_paired_bootstrap_delta_ci95"][0])
        >= -float(thresholds["ci_noninferiority_margin"]),
        "pool_shift_stable": min(pool_deltas.values()) >= float(
            config["pool_shift"]["required_delta_vs_pool_global"]
        ),
    }
    decision = "GO" if all(criteria.values()) else "NO-GO"
    gate = {
        "decision": decision,
        "development_ood_authorized": decision == "GO",
        "criteria": criteria,
        "primary_comparison": primary,
        "full_accuracy": accuracy["fcrg_full"],
        "global_best_accuracy": accuracy["global_best_posthoc"],
        "knop_accuracy": accuracy["knop_output_profile"],
        "graph_control_accuracies": {method: accuracy[method] for method in graph_controls},
        "h1_only_accuracy": accuracy["fcrg_h1_only"],
        "h1_h2_accuracy": accuracy["fcrg_h1_h2"],
        "pool_delta_vs_global": pool_deltas,
        "worst_environment_delta": float(min(environment_deltas)),
        "nonnegative_environment_fraction": float(np.mean(np.asarray(environment_deltas) >= 0.0)),
        "positioning": (
            "FCRG remains an exploratory post-hoc response selector; no main-method novelty claim is authorized"
            if decision == "NO-GO"
            else "source gate passed; development OOD may be run without selecting defaults"
        ),
    }

    cost_values: dict[str, list[dict[str, float]]] = defaultdict(list)
    for seed in seeds:
        with (runs[seed]["dir"] / "inference_costs.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                cost_values[str(row["method"])].append(
                    {
                        "calls": float(row["mean_nominal_model_calls"]),
                        "latency": float(row["mean_cached_serial_latency_seconds"]),
                    }
                )
    cost_rows = [
        {
            "method": method,
            "mean_nominal_model_calls": float(np.mean([row["calls"] for row in values])),
            "mean_cached_serial_latency_seconds": float(np.mean([row["latency"] for row in values])),
            "seed_count": len(values),
        }
        for method, values in sorted(cost_values.items())
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
    write_json(args.output_dir / "gate.json", gate)
    write_csv(args.output_dir / "aggregate_summary.csv", summary_rows)
    write_json(args.output_dir / "aggregate_comparisons.json", comparisons)
    write_csv(
        args.output_dir / "aggregate_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in comparisons],
    )
    write_csv(args.output_dir / "aggregate_per_environment.csv", environment_rows)
    write_csv(args.output_dir / "aggregate_costs.csv", cost_rows)
    write_csv(args.output_dir / "resources.csv", resource_rows)
    write_json(
        args.output_dir / "coverage.json",
        {
            "authenticated_method_count": len(methods),
            "methods": methods,
            "classic_and_response_baselines": config["required_baseline_methods"],
            "fcrg_ablations": config["required_fcrg_methods"],
            "all_required_present": set(config["required_baseline_methods"]).issubset(methods)
            and set(config["required_fcrg_methods"]).issubset(methods),
        },
    )
    write_json(
        args.output_dir / "authenticated_inputs.json",
        {
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "test_receipt": str(config["test_receipt"]),
            "seeds": seeds,
            "physical_gpu_by_seed": expected_gpu,
            "input_manifest_sha256": next(iter(input_hashes)),
            "innovation_code_manifest_sha256": current_code_hash,
            "source_question_ids_sha256": question_hash,
            "evaluation_derivation": "all four prediction packages authenticated before labels were opened",
        },
    )
    write_json(
        args.output_dir / "aggregate_complete_manifest.json",
        {"artifact_hashes": files_manifest([args.output_dir])},
    )
    print(json.dumps({"output_dir": str(args.output_dir), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
