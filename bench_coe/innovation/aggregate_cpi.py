from __future__ import annotations

import argparse
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
from .cpi import max_probability_difference
from .data import EvaluationLabelAdapter
from .evaluation import (
    exact_mcnemar,
    hierarchical_paired_bootstrap,
    holm_adjust,
    paired_bootstrap_delta,
    selection_correctness,
)
from .schema import EvaluationLabels, Selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate authenticated CPI GPU seeds")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_path(run_dir: Path, method: str) -> Path:
    parent = run_dir / "predictions" / "ablations" if method.startswith("fixed_ablation__") else run_dir / "predictions"
    return parent / f"{method}.jsonl"


def _correctness_vector(
    selections: list[Selection],
    labels: EvaluationLabels,
) -> tuple[list[str], np.ndarray]:
    by_question = selection_correctness(selections, labels)
    question_ids = sorted(by_question)
    return question_ids, np.asarray([float(by_question[question_id]) for question_id in question_ids])


def _score_rows(selections: list[Selection]) -> tuple[list[str], list[dict[int | str, float]]]:
    ordered = sorted(selections, key=lambda item: item.question_id)
    return [item.question_id for item in ordered], [dict(item.cluster_scores) for item in ordered]


def _validate_completion_manifest(run_dir: Path, completion: dict[str, Any]) -> None:
    root = run_dir.resolve()
    artifact_hashes = completion.get("artifact_hashes", {})
    required = [
        run_dir / "prediction_manifest.json",
        run_dir / "summary.json",
        run_dir / "fixed_ablation.csv",
        run_dir / "resource_usage.json",
        run_dir / "seed_gate.json",
    ]
    missing = [str(path) for path in required if str(path) not in artifact_hashes]
    if missing:
        raise RuntimeError(f"Completion manifest omits required artifacts: {missing}")
    for path_string, expected_hash in artifact_hashes.items():
        path = Path(path_string)
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Completion manifest references an external artifact: {path}") from error
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Completed CPI artifact hash mismatch: {path}")
    deterministic = completion.get("determinism", {})
    expected = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": str(completion.get("seed")),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if deterministic != expected:
        raise RuntimeError(f"Incomplete deterministic CUDA evidence in {run_dir}: {deterministic}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    receipt = validate_test_receipt(receipt_path, args.config)
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
    if len(run_dirs) != 4:
        raise RuntimeError(f"Expected four CPI seed directories, found {len(run_dirs)}")
    expected_seeds = [int(value) for value in config["seeds"]]
    expected_gpu_by_seed = {seed: index for index, seed in enumerate(expected_seeds)}
    runs: dict[int, dict[str, Any]] = {}
    for run_dir in run_dirs:
        completion_path = run_dir / "run_complete_manifest.json"
        prediction_manifest_path = run_dir / "prediction_manifest.json"
        if not completion_path.exists() or not prediction_manifest_path.exists():
            raise RuntimeError(f"Incomplete seed run: {run_dir}")
        completion = _load(completion_path)
        _validate_completion_manifest(run_dir, completion)
        manifest = _load(prediction_manifest_path)
        if sha256_file(prediction_manifest_path) != completion.get("prediction_manifest_sha256"):
            raise RuntimeError(f"Prediction manifest hash mismatch: {run_dir}")
        seed = int(manifest["seed"])
        if seed in runs or seed not in expected_gpu_by_seed:
            raise RuntimeError(f"Unexpected or duplicate CPI seed: {seed}")
        if int(completion.get("seed")) != seed or int(completion.get("physical_gpu")) != expected_gpu_by_seed[seed]:
            raise RuntimeError(f"Completion manifest seed/GPU mismatch: {run_dir}")
        if int(manifest["physical_gpu"]) != expected_gpu_by_seed[seed]:
            raise RuntimeError(f"Seed {seed} ran on unexpected physical GPU")
        if int(manifest["source_questions"]) != int(config["expected_source_questions"]):
            raise RuntimeError(f"Seed {seed} has an incomplete query set")
        if int(manifest["source_environments"]) != int(config["expected_source_environments"]):
            raise RuntimeError(f"Seed {seed} has an incomplete environment set")
        if len(manifest["heldout_environments"]) != int(config["expected_source_environments"]):
            raise RuntimeError(f"Seed {seed} has an incomplete heldout environment list")
        if manifest["innovation_code_manifest_sha256"] != current_code_sha or current_code_sha != receipt["code_manifest_sha256"]:
            raise RuntimeError(f"Seed {seed} code snapshot is not the tested current code")
        if manifest["input_manifest_sha256"] != current_input_sha:
            raise RuntimeError(f"Seed {seed} input snapshot does not match current authenticated inputs")
        predictions: dict[str, list[Selection]] = {}
        for method, expected_hash in manifest["prediction_hashes_before_evaluation"].items():
            path = _prediction_path(run_dir, method)
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Prediction hash mismatch: seed={seed}, method={method}")
            if completion["artifact_hashes"].get(str(path)) != expected_hash:
                raise RuntimeError(f"Prediction is not bound to the completion manifest: seed={seed}, method={method}")
            predictions[method] = read_selections(path)
        runs[seed] = {
            "dir": run_dir,
            "manifest": manifest,
            "completion": completion,
            "predictions": predictions,
            "resource": _load(run_dir / "resource_usage.json"),
        }
    if sorted(runs) != sorted(expected_seeds):
        raise RuntimeError(f"Seed set {sorted(runs)} does not match frozen config {expected_seeds}")
    if len({run["manifest"]["source_question_ids_sha256"] for run in runs.values()}) != 1:
        raise RuntimeError("CPI seed query sets are not identical")

    evaluation_labels = EvaluationLabelAdapter.from_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        [str(value) for value in config["experts"]] + [str(value) for value in config["replacement_experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    ).load()
    reference_ids: list[str] | None = None
    for seed in expected_seeds:
        ids, _ = _correctness_vector(runs[seed]["predictions"]["source_best_single"], evaluation_labels)
        if len(ids) != int(config["expected_source_questions"]):
            raise RuntimeError(f"Seed {seed} source-best prediction count is incomplete")
        if reference_ids is not None and ids != reference_ids:
            raise RuntimeError("Source-best question IDs differ across seeds")
        reference_ids = ids
    assert reference_ids is not None
    question_ids_sha = hashlib.sha256("\n".join(reference_ids).encode("utf-8")).hexdigest()
    if question_ids_sha != next(iter(runs.values()))["manifest"]["source_question_ids_sha256"]:
        raise RuntimeError("Evaluator question IDs do not match the prediction manifest")

    family_prefix = "deepsets_full__remove_family__"
    family_names = sorted(
        method.removeprefix(family_prefix)
        for method in runs[expected_seeds[0]]["predictions"]
        if method.startswith(family_prefix)
    )
    seed_gates: list[dict[str, Any]] = []
    full_vectors: list[np.ndarray] = []
    none_vectors: list[np.ndarray] = []
    family_vectors: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        family: ([], []) for family in family_names
    }
    for seed in expected_seeds:
        predictions = runs[seed]["predictions"]
        full_ids, full = _correctness_vector(predictions["deepsets_full"], evaluation_labels)
        none_ids, none = _correctness_vector(predictions["deepsets_none"], evaluation_labels)
        if full_ids != reference_ids or none_ids != reference_ids:
            raise RuntimeError(f"CPI direct comparison is not query-aligned for seed {seed}")
        full_vectors.append(full)
        none_vectors.append(none)
        leave_family_deltas: dict[str, float] = {}
        for family in family_names:
            candidate_ids, candidate = _correctness_vector(
                predictions[f"deepsets_full__remove_family__{family}"], evaluation_labels
            )
            reference_method_ids, reference = _correctness_vector(
                predictions[f"deepsets_none__remove_family__{family}"], evaluation_labels
            )
            if candidate_ids != reference_ids or reference_method_ids != reference_ids:
                raise RuntimeError(f"Family comparison {family} is not query-aligned for seed {seed}")
            family_vectors[family][0].append(candidate)
            family_vectors[family][1].append(reference)
            leave_family_deltas[family] = float((candidate - reference).mean())
        score_ids, full_scores = _score_rows(predictions["deepsets_full"])
        clone_ids, clone_scores = _score_rows(predictions["cpi_full__exact_clone"])
        permutation_ids, permutation_scores = _score_rows(predictions["cpi_full__permutation"])
        if score_ids != reference_ids or clone_ids != reference_ids or permutation_ids != reference_ids:
            raise RuntimeError(f"Invariance predictions are not query-aligned for seed {seed}")
        clone_sensitivity = max_probability_difference(full_scores, clone_scores)
        permutation_sensitivity = max_probability_difference(full_scores, permutation_scores)
        rescue, harm, p_value = exact_mcnemar(full, none)
        ci = paired_bootstrap_delta(full, none, seed=seed, samples=int(config["bootstrap_samples"]))
        gate = {
            "seed": seed,
            "full_accuracy": float(full.mean()),
            "no_intervention_accuracy": float(none.mean()),
            "delta": float((full - none).mean()),
            "worst_leave_family_delta": min(leave_family_deltas.values()),
            "leave_family_deltas": leave_family_deltas,
            "clone_probability_sensitivity": clone_sensitivity,
            "permutation_probability_sensitivity": permutation_sensitivity,
            "paired_bootstrap_delta_ci95": list(ci),
            "rescue_count": rescue,
            "harm_count": harm,
            "exact_mcnemar_p": p_value,
            "required_delta": 0.0025,
            "required_worst_leave_family_delta": -0.005,
            "clone_tolerance": float(config["clone_tolerance"]),
            "derived_from": "authenticated_predictions_plus_registry_validated_labels",
        }
        gate["decision"] = (
            "GO"
            if gate["delta"] >= gate["required_delta"]
            and gate["worst_leave_family_delta"] >= gate["required_worst_leave_family_delta"]
            and gate["clone_probability_sensitivity"] < gate["clone_tolerance"]
            and gate["permutation_probability_sensitivity"] < gate["clone_tolerance"]
            else "NO-GO"
        )
        seed_gates.append(gate)

    full_matrix = np.stack(full_vectors)
    none_matrix = np.stack(none_vectors)
    hierarchical_ci = hierarchical_paired_bootstrap(full_matrix, none_matrix, seed=20260808, samples=10000)
    pooled_rescue, pooled_harm, pooled_p = exact_mcnemar(full_matrix.ravel(), none_matrix.ravel())
    all_family_deltas = [
        float((candidate - reference).mean())
        for family in family_names
        for candidate, reference in zip(*family_vectors[family])
    ]
    family_deltas = {
        family: float((np.stack(candidate) - np.stack(reference)).mean())
        for family, (candidate, reference) in family_vectors.items()
    }
    aggregate_gate: dict[str, Any] = {
        "seeds": expected_seeds,
        "full_accuracy_mean": float(full_matrix.mean()),
        "full_accuracy_std": float(np.std(full_matrix.mean(axis=1), ddof=1)),
        "no_intervention_accuracy_mean": float(none_matrix.mean()),
        "no_intervention_accuracy_std": float(np.std(none_matrix.mean(axis=1), ddof=1)),
        "mean_paired_seed_delta": float((full_matrix - none_matrix).mean()),
        "worst_leave_family_delta": min(all_family_deltas),
        "mean_leave_family_deltas": family_deltas,
        "leave_family_deltas": family_deltas,
        "maximum_clone_probability_sensitivity": max(float(row["clone_probability_sensitivity"]) for row in seed_gates),
        "maximum_permutation_probability_sensitivity": max(float(row["permutation_probability_sensitivity"]) for row in seed_gates),
        "required_delta": 0.0025,
        "required_worst_leave_family_delta": -0.005,
        "clone_tolerance": float(config["clone_tolerance"]),
        "crossed_seed_query_bootstrap_delta_ci95": list(hierarchical_ci),
        "bootstrap_design": "crossed seeds x shared queries; one common query resample per bootstrap draw",
        "pooled_seed_query_mcnemar": {
            "rescue_count": pooled_rescue,
            "harm_count": pooled_harm,
            "exact_p_descriptive": pooled_p,
            "note": "Rows repeat shared queries across seeds; crossed bootstrap is the primary uncertainty estimate.",
        },
        "property_tests": {
            "receipt": str(receipt_path),
            "test_count": receipt["test_count"],
            "exit_code": receipt["exit_code"],
        },
        "derived_from": "authenticated_predictions_plus_registry_validated_labels",
    }

    aggregate_comparisons: list[dict[str, Any]] = []
    comparisons = [("deepsets_full_vs_none", full_matrix, none_matrix)]
    comparisons.extend(
        (
            f"remove_family_{family}__full_vs_none",
            np.stack(family_vectors[family][0]),
            np.stack(family_vectors[family][1]),
        )
        for family in family_names
    )
    for name, candidate, reference in comparisons:
        rescue, harm, p_value = exact_mcnemar(candidate.ravel(), reference.ravel())
        aggregate_comparisons.append(
            {
                "comparison": name,
                "delta": float((candidate - reference).mean()),
                "crossed_bootstrap_ci95": list(
                    hierarchical_paired_bootstrap(candidate, reference, seed=20260808, samples=10000)
                ),
                "pooled_seed_query_rescue": rescue,
                "pooled_seed_query_harm": harm,
                "pooled_seed_query_mcnemar_p_descriptive": p_value,
            }
        )
    corrections = holm_adjust(
        {row["comparison"]: float(row["pooled_seed_query_mcnemar_p_descriptive"]) for row in aggregate_comparisons}
    )
    for row in aggregate_comparisons:
        row["holm_descriptive"] = corrections[row["comparison"]]
    aggregate_gate["direct_comparison"] = aggregate_comparisons[0]
    aggregate_gate["decision"] = (
        "GO"
        if aggregate_gate["mean_paired_seed_delta"] >= aggregate_gate["required_delta"]
        and aggregate_gate["worst_leave_family_delta"] >= aggregate_gate["required_worst_leave_family_delta"]
        and aggregate_gate["maximum_clone_probability_sensitivity"] < aggregate_gate["clone_tolerance"]
        and aggregate_gate["maximum_permutation_probability_sensitivity"] < aggregate_gate["clone_tolerance"]
        else "NO-GO"
    )

    full_size_methods = sorted(
        method
        for method in runs[expected_seeds[0]]["predictions"]
        if not method.startswith("fixed_ablation__") and method != "overfit_sanity"
    )
    summary_rows: list[dict[str, Any]] = []
    for method in full_size_methods:
        accuracies: list[float] = []
        deltas: list[float] = []
        rescues: list[int] = []
        harms: list[int] = []
        switch_rates: list[float] = []
        for seed in expected_seeds:
            predictions = runs[seed]["predictions"]
            ids, candidate = _correctness_vector(predictions[method], evaluation_labels)
            baseline_ids, baseline = _correctness_vector(predictions["source_best_single"], evaluation_labels)
            if ids != reference_ids or baseline_ids != reference_ids:
                raise RuntimeError(f"Summary method {method} is not query-aligned")
            rescue, harm, _ = exact_mcnemar(candidate, baseline)
            candidate_by_id = {item.question_id: item.selected_expert_id for item in predictions[method]}
            baseline_by_id = {item.question_id: item.selected_expert_id for item in predictions["source_best_single"]}
            accuracies.append(float(candidate.mean()))
            deltas.append(float((candidate - baseline).mean()))
            rescues.append(rescue)
            harms.append(harm)
            switch_rates.append(float(np.mean([candidate_by_id[qid] != baseline_by_id[qid] for qid in reference_ids])))
        summary_rows.append(
            {
                "method": method,
                "seeds": len(expected_seeds),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "delta_vs_source_best_mean": float(np.mean(deltas)),
                "rescue_count_mean": float(np.mean(rescues)),
                "harm_count_mean": float(np.mean(harms)),
                "switch_rate_mean": float(np.mean(switch_rates)),
            }
        )

    ablation_values: dict[str, list[float]] = defaultdict(list)
    ablation_ids: dict[str, list[str]] = {}
    for seed in expected_seeds:
        for method, selections in runs[seed]["predictions"].items():
            if not method.startswith("fixed_ablation__"):
                continue
            ids, correctness = _correctness_vector(selections, evaluation_labels)
            if method in ablation_ids and ids != ablation_ids[method]:
                raise RuntimeError(f"Fixed ablation {method} differs across seeds")
            ablation_ids[method] = ids
            ablation_values[method].append(float(correctness.mean()))
    ablation_rows = [
        {
            "method": method,
            "accuracy_mean": float(np.mean(values)),
            "accuracy_std": float(np.std(values, ddof=1)),
            "seeds": len(values),
            "samples_per_seed": len(ablation_ids[method]),
        }
        for method, values in sorted(ablation_values.items())
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
        for seed in expected_seeds
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "gate.json", aggregate_gate)
    write_json(args.output_dir / "seed_gates.json", seed_gates)
    write_json(args.output_dir / "aggregate_paired_comparisons.json", aggregate_comparisons)
    write_csv(args.output_dir / "aggregate_summary.csv", summary_rows)
    write_csv(args.output_dir / "aggregate_fixed_ablation.csv", ablation_rows)
    write_csv(args.output_dir / "resources.csv", resource_rows)
    write_json(
        args.output_dir / "authenticated_inputs.json",
        {
            "config": str(args.config),
            "seeds": expected_seeds,
            "physical_gpu_by_seed": expected_gpu_by_seed,
            "input_manifest_sha256": current_input_sha,
            "innovation_code_manifest_sha256": current_code_sha,
            "source_question_ids_sha256": question_ids_sha,
            "dataset_registry": str(config["dataset_registry"]),
            "dataset_registry_sha256": config["dataset_registry_sha256"],
            "test_receipt": str(receipt_path),
            "evaluation_derivation": "all gate correctness recomputed from hashed prediction JSONL plus registry-validated labels",
        },
    )
    aggregate_artifacts = files_manifest([args.output_dir])
    write_json(args.output_dir / "aggregate_complete_manifest.json", {"artifact_hashes": aggregate_artifacts})
    print(json.dumps({"output_dir": str(args.output_dir), "gate": aggregate_gate}, indent=2))


if __name__ == "__main__":
    main()
