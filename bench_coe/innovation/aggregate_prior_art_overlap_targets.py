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

from .aggregate_prior_art_overlap import validate_completion
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
from .prior_art_targets import authenticate_prediction_package
from .schema import EvaluationLabels, Selection


DECISIVE_REFERENCES = (
    "global_best_posthoc",
    "majority_answer_support",
    "mcb_dcs_structured",
    "knop_output_profile",
    "oprs_robust_output_profile",
    "more_style_structured",
    "more_style_minilm",
    "smoothie_local_spectral",
    "smoothie_local_minilm",
    "uncertainty_only",
    "learned_logistic_selector",
    "fcrg_column_mean_only",
    "fcrg_symmetric",
    "fcrg_random_edges",
    "fcrg_degree_relabel",
    "fcrg_row_normalized",
    "fcrg_column_normalized",
    "fcrg_row_softmax",
    "fcrg_h1_only",
    "fcrg_h1_h2",
    "fcrg_learned_weights",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate authenticated cross-dataset prior-art runs"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _correctness(
    selections: list[Selection], labels: EvaluationLabels
) -> tuple[list[str], np.ndarray]:
    ordered = sorted(selections, key=lambda selection: selection.question_id)
    ids = [selection.question_id for selection in ordered]
    values = np.asarray(
        [bool(labels.get(item.question_id, item.selected_expert_id or "")) for item in ordered],
        dtype=int,
    )
    return ids, values


def _comparison(
    target: str,
    candidate: str,
    reference: str,
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
        ids, candidate_values = _correctness(
            runs[seed]["predictions"][target][candidate], labels
        )
        reference_question_ids, reference_values = _correctness(
            runs[seed]["predictions"][target][reference], labels
        )
        if ids != reference_question_ids or (reference_ids is not None and ids != reference_ids):
            raise RuntimeError(f"Unaligned target comparison: {target}/{candidate}/{reference}")
        reference_ids = ids
        rescue, harm, p_value = exact_mcnemar(candidate_values, reference_values)
        candidates.append(candidate_values)
        references.append(reference_values)
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
        "comparison": f"{target}__{candidate}_vs_{reference}",
        "target": target,
        "candidate": candidate,
        "reference": reference,
        "samples": candidate_matrix.shape[1],
        "seeds": candidate_matrix.shape[0],
        "candidate_accuracy_mean": float(candidate_matrix.mean()),
        "reference_accuracy_mean": float(reference_matrix.mean()),
        "delta_mean": float((candidate_matrix - reference_matrix).mean()),
        "rescue_count_mean": float(np.mean(rescues)),
        "harm_count_mean": float(np.mean(harms)),
        "exact_mcnemar_p_max_across_seeds": float(max(p_values)),
        "hierarchical_paired_bootstrap_delta_ci95": [float(ci[0]), float(ci[1])],
        "bootstrap_samples": bootstrap_samples,
        "note": "seeds repeat the same target queries; one shared query resample is used per draw",
    }


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
        completion = _load_json(run_dir / "complete_manifest.json")
        validate_completion(run_dir, completion)
        manifest = _load_json(run_dir / "prediction_manifest.json")
        environment = _load_json(run_dir / "environment.json")
        if int(manifest["seed"]) != seed or int(manifest["physical_gpu"]) != gpu:
            raise RuntimeError(f"Target seed/GPU identity mismatch: {run_dir}")
        if manifest["innovation_code_manifest_sha256"] != current_code_hash:
            raise RuntimeError(f"Target run code snapshot differs: {run_dir}")
        if environment["innovation_code_manifest_sha256"] != current_code_hash:
            raise RuntimeError(f"Target environment code snapshot differs: {run_dir}")
        input_hashes.add(str(manifest["input_manifest_sha256"]))
        predictions: dict[str, dict[str, list[Selection]]] = {}
        for target, target_manifest in sorted(manifest["targets"].items()):
            authenticate_prediction_package(run_dir, target_manifest)
            target_predictions: dict[str, list[Selection]] = {}
            for method, relative in target_manifest["prediction_paths"].items():
                path = run_dir / str(relative)
                expected_hash = target_manifest["prediction_hashes_before_evaluation"][method]
                if completion["artifact_hashes"].get(str(path)) != expected_hash:
                    raise RuntimeError(f"Target prediction is not completion-bound: {path}")
                target_predictions[str(method)] = read_selections(path)
            if len(target_predictions) != int(target_manifest["method_count"]):
                raise RuntimeError(f"Target method count mismatch: {run_dir}/{target}")
            predictions[str(target)] = target_predictions
        runs[seed] = {
            "dir": run_dir,
            "manifest": manifest,
            "completion": completion,
            "predictions": predictions,
            "resource": _load_json(run_dir / "resource_usage.json"),
        }
    if len(input_hashes) != 1:
        raise RuntimeError("Target input snapshots differ across seeds")

    target_names = tuple(sorted(runs[seeds[0]]["predictions"]))
    for seed in seeds[1:]:
        if tuple(sorted(runs[seed]["predictions"])) != target_names:
            raise RuntimeError("Target sets differ across seed packages")
    stochastic = tuple(str(value) for value in config["stochastic_method_patterns"])
    for target in target_names:
        method_sets = {
            tuple(sorted(runs[seed]["predictions"][target])) for seed in seeds
        }
        if len(method_sets) != 1:
            raise RuntimeError(f"Method sets differ across seeds: {target}")
        methods = next(iter(method_sets))
        for method in methods:
            if any(pattern in method for pattern in stochastic):
                continue
            hashes = {
                runs[seed]["manifest"]["targets"][target][
                    "prediction_hashes_before_evaluation"
                ][method]
                for seed in seeds
            }
            if len(hashes) != 1:
                raise RuntimeError(f"Deterministic target method differs: {target}/{method}")

    target_config = {str(target["name"]): target for target in config["targets"]}
    labels_by_target: dict[str, EvaluationLabels] = {}
    for target in target_names:
        definition = target_config[target]
        labels_by_target[target] = EvaluationLabelAdapter.from_registry(
            Path(definition["label_cache_path"]),
            str(definition["dataset"]),
            str(definition["split"]),
            str(definition["modality"]),
            [str(value) for value in config["experts"]],
            Path(config["dataset_registry"]),
            str(config["dataset_registry_sha256"]),
        ).load()

    summary_rows: list[dict[str, Any]] = []
    accuracy: dict[tuple[str, str], float] = {}
    for target in target_names:
        labels = labels_by_target[target]
        methods = sorted(runs[seeds[0]]["predictions"][target])
        groups = runs[seeds[0]]["manifest"]["targets"][target]["method_group"]
        for method in methods:
            values_by_seed: list[np.ndarray] = []
            reference_by_seed: list[np.ndarray] = []
            rescues: list[int] = []
            harms: list[int] = []
            pool = str(groups[method])
            reference = (
                "global_best_posthoc"
                if pool == "full_pool"
                else f"{pool}__global_best_posthoc"
            )
            for seed in seeds:
                _, values = _correctness(runs[seed]["predictions"][target][method], labels)
                _, baseline = _correctness(
                    runs[seed]["predictions"][target][reference], labels
                )
                rescue, harm, _ = exact_mcnemar(values, baseline)
                values_by_seed.append(values)
                reference_by_seed.append(baseline)
                rescues.append(rescue)
                harms.append(harm)
            candidate_matrix = np.stack(values_by_seed)
            reference_matrix = np.stack(reference_by_seed)
            mean = float(candidate_matrix.mean())
            accuracy[(target, method)] = mean
            summary_rows.append(
                {
                    "target": target,
                    "dataset": str(target_config[target]["dataset"]),
                    "split": str(target_config[target]["split"]),
                    "method": method,
                    "pool": pool,
                    "samples": candidate_matrix.shape[1],
                    "seeds": len(seeds),
                    "accuracy_mean": mean,
                    "accuracy_std": float(
                        np.std(candidate_matrix.mean(axis=1), ddof=1)
                    ),
                    "delta_vs_pool_global_best_mean": float(
                        (candidate_matrix - reference_matrix).mean()
                    ),
                    "rescue_count_mean": float(np.mean(rescues)),
                    "harm_count_mean": float(np.mean(harms)),
                }
            )

    comparisons: list[dict[str, Any]] = []
    comparison_samples = int(config.get("comparison_bootstrap_samples", 2000))
    primary_samples = int(config.get("primary_bootstrap_samples", 10000))
    for target_index, target in enumerate(target_names):
        methods = runs[seeds[0]]["predictions"][target]
        for reference_index, reference in enumerate(DECISIVE_REFERENCES):
            if reference == "fcrg_full" or reference not in methods:
                continue
            samples = primary_samples if reference == "global_best_posthoc" else comparison_samples
            comparisons.append(
                _comparison(
                    target,
                    "fcrg_full",
                    reference,
                    runs,
                    seeds,
                    labels_by_target[target],
                    bootstrap_seed=int(config["aggregate_bootstrap_seed"])
                    + 100 * target_index
                    + reference_index,
                    bootstrap_samples=samples,
                )
            )
    correction = holm_adjust(
        {
            str(row["comparison"]): float(row["exact_mcnemar_p_max_across_seeds"])
            for row in comparisons
        }
    )
    for row in comparisons:
        row["holm"] = correction[str(row["comparison"])]

    main_methods = sorted(
        method
        for method in runs[seeds[0]]["predictions"][target_names[0]]
        if not any(
            method.startswith(f"{pool}__")
            for pool in config["pool_shift"]["pools"]
        )
    )
    cross_dataset_rows: list[dict[str, Any]] = []
    for method in main_methods:
        target_values = [accuracy[(target, method)] for target in target_names]
        global_values = [accuracy[(target, "global_best_posthoc")] for target in target_names]
        weighted_correct = 0.0
        weighted_total = 0
        weighted_global = 0.0
        for target in target_names:
            samples = int(
                runs[seeds[0]]["manifest"]["targets"][target]["questions"]
            )
            weighted_correct += accuracy[(target, method)] * samples
            weighted_global += accuracy[(target, "global_best_posthoc")] * samples
            weighted_total += samples
        cross_dataset_rows.append(
            {
                "method": method,
                "targets": len(target_names),
                "macro_accuracy": float(np.mean(target_values)),
                "macro_delta_vs_global_best": float(
                    np.mean(np.asarray(target_values) - np.asarray(global_values))
                ),
                "micro_accuracy": weighted_correct / weighted_total,
                "micro_delta_vs_global_best": (weighted_correct - weighted_global)
                / weighted_total,
                "nonnegative_target_fraction": float(
                    np.mean(np.asarray(target_values) >= np.asarray(global_values))
                ),
                "worst_target_delta": float(
                    np.min(np.asarray(target_values) - np.asarray(global_values))
                ),
            }
        )

    environment_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    environment_samples: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    cost_values: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for seed in seeds:
        run_dir = runs[seed]["dir"]
        with (run_dir / "per_environment.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    str(row["target"]),
                    str(row["environment"]),
                    str(row["method"]),
                    str(row["pool"]),
                )
                environment_values[key].append(float(row["accuracy"]))
                environment_samples[key].add(int(row["samples"]))
        with (run_dir / "inference_costs.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["target"]), str(row["method"]), str(row["pool"]))
                cost_values[key].append(
                    (
                        float(row["mean_nominal_model_calls"]),
                        float(row["mean_cached_serial_latency_seconds"]),
                    )
                )
    environment_rows = [
        {
            "target": key[0],
            "environment": key[1],
            "method": key[2],
            "pool": key[3],
            "samples": next(iter(environment_samples[key])),
            "accuracy_mean": float(np.mean(values)),
            "accuracy_std": float(np.std(values, ddof=1)),
        }
        for key, values in sorted(environment_values.items())
    ]
    cost_rows = [
        {
            "target": key[0],
            "method": key[1],
            "pool": key[2],
            "mean_nominal_model_calls": float(np.mean([value[0] for value in values])),
            "mean_cached_serial_latency_seconds": float(
                np.mean([value[1] for value in values])
            ),
            "seed_count": len(values),
        }
        for key, values in sorted(cost_values.items())
    ]

    required = set(config["required_baseline_methods"]).union(
        config["required_fcrg_methods"]
    )
    coverage = {
        target: {
            "method_count": len(runs[seeds[0]]["predictions"][target]),
            "all_required_present": required.issubset(
                runs[seeds[0]]["predictions"][target]
            ),
            "missing_required": sorted(
                required.difference(runs[seeds[0]]["predictions"][target])
            ),
        }
        for target in target_names
    }
    if not all(row["all_required_present"] for row in coverage.values()):
        raise RuntimeError("Cross-dataset method coverage is incomplete")

    fcrg_dataset_rows = [
        {
            "target": target,
            "samples": int(runs[seeds[0]]["manifest"]["targets"][target]["questions"]),
            "fcrg_accuracy": accuracy[(target, "fcrg_full")],
            "global_best_accuracy": accuracy[(target, "global_best_posthoc")],
            "delta_vs_global_best": accuracy[(target, "fcrg_full")]
            - accuracy[(target, "global_best_posthoc")],
            "knop_accuracy": accuracy[(target, "knop_output_profile")],
            "mcb_accuracy": accuracy[(target, "mcb_dcs_structured")],
            "more_minilm_accuracy": accuracy[(target, "more_style_minilm")],
            "smoothie_local_minilm_accuracy": accuracy[
                (target, "smoothie_local_minilm")
            ],
        }
        for target in target_names
    ]
    diagnostic = {
        "scope": "development_ood_diagnostic_only",
        "source_gate_overridden": False,
        "source_no_go_remains_binding": True,
        "can_authorize_locked_test": False,
        "targets": fcrg_dataset_rows,
        "fcrg_cross_dataset": next(
            row for row in cross_dataset_rows if row["method"] == "fcrg_full"
        ),
        "interpretation": (
            "These datasets previously influenced development. Results complete the requested "
            "diagnostic matrix but cannot be used as blind confirmation or to reverse the source NO-GO."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "aggregate_summary.csv", summary_rows)
    write_json(args.output_dir / "aggregate_comparisons.json", comparisons)
    write_csv(
        args.output_dir / "aggregate_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in comparisons],
    )
    write_csv(args.output_dir / "aggregate_per_environment.csv", environment_rows)
    write_csv(args.output_dir / "aggregate_costs.csv", cost_rows)
    write_csv(args.output_dir / "cross_dataset_summary.csv", cross_dataset_rows)
    write_json(args.output_dir / "coverage.json", coverage)
    write_json(args.output_dir / "diagnostic_summary.json", diagnostic)
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
            "targets": list(target_names),
            "target_question_ids_sha256": {
                target: hashlib.sha256(
                    "\n".join(
                        _correctness(
                            runs[seeds[0]]["predictions"][target]["global_best_posthoc"],
                            labels_by_target[target],
                        )[0]
                    ).encode("utf-8")
                ).hexdigest()
                for target in target_names
            },
            "evaluation_derivation": (
                "all seed/target prediction packages authenticated before target labels were opened"
            ),
        },
    )
    write_csv(
        args.output_dir / "resources.csv",
        [
            {
                "seed": seed,
                "physical_gpu": runs[seed]["resource"]["physical_gpu"],
                "device_name": runs[seed]["resource"]["device_name"],
                "peak_allocated_bytes": runs[seed]["resource"]["peak_allocated_bytes"],
                "peak_reserved_bytes": runs[seed]["resource"]["peak_reserved_bytes"],
                "runtime_seconds": runs[seed]["resource"]["runtime_seconds"],
            }
            for seed in seeds
        ],
    )
    write_json(
        args.output_dir / "aggregate_complete_manifest.json",
        {"artifact_hashes": files_manifest([args.output_dir])},
    )
    print(json.dumps({"output_dir": str(args.output_dir), "diagnostic": diagnostic}, indent=2))


if __name__ == "__main__":
    main()
