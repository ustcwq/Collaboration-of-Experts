from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    seed_gpu_map,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_jsonl,
    write_selections,
)
from .conservative_cpi import grouped_environment_folds
from .cpi import fit_source_fingerprints, make_pool_example, subject_folds
from .cpi_ce import (
    CategoricalScores,
    apply_categorical_gate,
    calibrate_temperature_and_threshold,
    none_fallback_selections,
    predict_categorical,
    train_categorical_scorer,
)
from .data import CacheAdapter, load_family_map
from .evaluation import evaluate, paired_selection_comparison
from .run_conservative_cpi import _configure_device, _load_base_predictions, _source_adapter, _subset
from .schema import EvaluationLabels, ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import SourceBestSelector


METHODS = ("cpi_ce_raw", "cpi_ce_none_fallback", "cpi_ce_calibrated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested-OOF categorical CPI on one physical GPU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-environments", type=int)
    return parser.parse_args()


def _validate_label_structure(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    expected_none: int,
    expected_one: int,
) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for question_id in batch.question_ids:
        correct_clusters = 0
        for cluster in batch.clusters(question_id):
            observed = [labels.get(question_id, expert) for expert in cluster.expert_ids]
            values = [bool(value) for value in observed if value is not None]
            if not values:
                raise ValueError("CPI-CE source cluster lacks correctness labels")
            if any(values) and not all(values):
                raise ValueError("CPI-CE source cluster has mixed correctness")
            correct_clusters += int(all(values))
        if correct_clusters > 1:
            raise ValueError("CPI-CE source query has multiple correct clusters")
        counts[correct_clusters] += 1
    observed_counts = {"none_correct": counts[0], "one_correct": counts[1]}
    if observed_counts != {"none_correct": expected_none, "one_correct": expected_one}:
        raise RuntimeError(f"CPI-CE source label structure changed: {observed_counts}")
    return observed_counts


def _assert_same_reference(computed: Sequence[Selection], frozen: Sequence[Selection]) -> None:
    if len(computed) != len(frozen):
        raise RuntimeError("Computed and frozen Source-Best predictions have different sizes")
    left = {item.question_id: item for item in computed}
    right = {item.question_id: item for item in frozen}
    if set(left) != set(right):
        raise RuntimeError("Computed and frozen Source-Best predictions are not aligned")
    for question_id in left:
        if (
            left[question_id].selected_cluster_id,
            left[question_id].selected_expert_id,
            left[question_id].normalized_answer,
        ) != (
            right[question_id].selected_cluster_id,
            right[question_id].selected_expert_id,
            right[question_id].normalized_answer,
        ):
            raise RuntimeError(f"Frozen Source-Best mismatch on {question_id}")


def _correct(labels: EvaluationLabels, question_id: str, selection: Selection) -> float:
    return float(bool(labels.get(question_id, selection.selected_expert_id or "")))


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    validate_test_receipt(receipt_path, args.config)
    seeds = [int(value) for value in config["seeds"]]
    run_gpu_by_seed = seed_gpu_map(config, "physical_gpus")
    base_gpu_by_seed = seed_gpu_map(config, "base_physical_gpus")
    if args.seed not in seeds or args.physical_gpu != run_gpu_by_seed[args.seed]:
        raise ValueError("Seed and physical GPU do not match the frozen CPI-CE mapping")
    device, device_manifest = _configure_device(args.physical_gpu)

    family_map_path = Path(config["family_map"])
    family_map = load_family_map(family_map_path)
    experts = [str(value) for value in config["experts"]]
    replacement_experts = [str(value) for value in config["replacement_experts"]]
    adapter = _source_adapter(config, experts, family_map)
    replacement_adapter = _source_adapter(config, replacement_experts, family_map)
    batch = adapter.load_observables()
    source_labels = adapter.load_source_labels()
    replacement_batch = replacement_adapter.load_observables()
    replacement_labels = replacement_adapter.load_source_labels()
    if replacement_batch.question_ids != batch.question_ids:
        raise ValueError("CPI-CE replacement expert cache is not aligned")
    label_structure = _validate_label_structure(
        batch,
        source_labels,
        int(config["expected_none_correct_questions"]),
        int(config["expected_one_correct_questions"]),
    )

    base_bce, base_reference, base_paths = _load_base_predictions(
        config,
        args.seed,
        base_gpu_by_seed[args.seed],
    )
    expected_ids = set(batch.question_ids)
    if {item.question_id for item in base_bce} != expected_ids:
        raise ValueError("Frozen BCE CPI predictions do not match the source cache")
    if {item.question_id for item in base_reference} != expected_ids:
        raise ValueError("Frozen Source-Best predictions do not match the source cache")

    folds = subject_folds(source_labels)
    if args.max_environments is not None:
        folds = folds[: args.max_environments]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)

    predictions: dict[str, list[Selection]] = {method: [] for method in METHODS}
    reference_predictions: list[Selection] = []
    bce_predictions: list[Selection] = []
    temperature_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    selected_calibration: dict[str, dict[str, float]] = {}
    swap_mapping = {str(key): str(value) for key, value in config["known_swaps"].items()}

    for outer_index, (environment, train_ids, test_ids) in enumerate(folds):
        outer_train_batch = batch.subset(train_ids)
        outer_train_labels = source_labels.subset(train_ids)
        inner_outputs: list[CategoricalScores] = []
        inner_labeled_examples = []
        inner_references: list[Selection] = []
        for inner_index, (heldout_environments, inner_train_ids, inner_test_ids) in enumerate(
            grouped_environment_folds(outer_train_labels, int(config["inner_groups"]))
        ):
            inner_train_batch = batch.subset(inner_train_ids)
            inner_test_batch = batch.subset(inner_test_ids)
            inner_train_labels = source_labels.subset(inner_train_ids)
            replacement_train_batch = replacement_batch.subset(inner_train_ids)
            replacement_train_labels = replacement_labels.subset(inner_train_ids)
            fingerprints = fit_source_fingerprints(
                inner_train_batch,
                inner_train_labels,
                rank=int(config["fingerprint_rank"]),
                extra_batch=replacement_train_batch,
                extra_labels=replacement_train_labels,
            )
            train_examples = [
                make_pool_example(inner_train_batch, question_id, fingerprints, inner_train_labels)
                for question_id in inner_train_batch.question_ids
            ]
            replacement_examples = {
                question_id: make_pool_example(
                    replacement_train_batch,
                    question_id,
                    fingerprints,
                    replacement_train_labels,
                )
                for question_id in replacement_train_batch.question_ids
            }
            model_seed = args.seed + outer_index * 1009 + 200_000 + inner_index * 131
            model, history = train_categorical_scorer(
                train_examples,
                fingerprints.dimension + 2,
                device,
                seed=model_seed,
                variant="full",
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                hidden_dim=int(config["hidden_dim"]),
                replacement_examples=replacement_examples,
                swap_mapping=swap_mapping,
            )
            _, outputs, predicted_examples = predict_categorical(
                model,
                inner_test_batch,
                fingerprints,
                device,
                temperature=1.0,
                method="inner_oof_cpi_ce",
            )
            inner_outputs.extend(outputs)
            inner_labeled_examples.extend(
                make_pool_example(inner_test_batch, example.question_id, fingerprints, outer_train_labels)
                for example in predicted_examples
            )
            inner_references.extend(
                SourceBestSelector().fit(inner_train_batch, inner_train_labels).predict(inner_test_batch)
            )
            training_rows.append(
                {
                    "outer_fold": outer_index,
                    "outer_environment": environment,
                    "role": "inner_calibration",
                    "inner_fold": inner_index,
                    "inner_heldout_environments": "|".join(heldout_environments),
                    "train_rows": len(inner_train_ids),
                    "prediction_rows": len(inner_test_ids),
                    "model_seed": model_seed,
                    "initialization_sha256": model.initialization_sha256,
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                }
            )
            del model

        temperature, threshold, temperature_diagnostics, threshold_diagnostics = (
            calibrate_temperature_and_threshold(
                outer_train_batch,
                inner_outputs,
                inner_labeled_examples,
                inner_references,
                outer_train_labels,
                outer_train_labels.environment_by_question,
                [float(value) for value in config["temperature_grid"]],
                [float(value) for value in config["threshold_grid"]],
                float(config["calibration_min_worst_delta"]),
                float(config["calibration_min_micro_delta"]),
                float(config["calibration_worst_weight"]),
            )
        )
        selected_calibration[environment] = {"temperature": temperature, "threshold": threshold}
        temperature_rows.extend(
            {"outer_fold": outer_index, "outer_environment": environment, **row}
            for row in temperature_diagnostics
        )
        threshold_rows.extend(
            {
                "outer_fold": outer_index,
                "outer_environment": environment,
                "selected": row.threshold == threshold,
                **asdict(row),
            }
            for row in threshold_diagnostics
        )

        replacement_outer_batch = replacement_batch.subset(train_ids)
        replacement_outer_labels = replacement_labels.subset(train_ids)
        outer_fingerprints = fit_source_fingerprints(
            outer_train_batch,
            outer_train_labels,
            rank=int(config["fingerprint_rank"]),
            extra_batch=replacement_outer_batch,
            extra_labels=replacement_outer_labels,
        )
        outer_examples = [
            make_pool_example(outer_train_batch, question_id, outer_fingerprints, outer_train_labels)
            for question_id in outer_train_batch.question_ids
        ]
        replacement_outer_examples = {
            question_id: make_pool_example(
                replacement_outer_batch,
                question_id,
                outer_fingerprints,
                replacement_outer_labels,
            )
            for question_id in replacement_outer_batch.question_ids
        }
        outer_model_seed = args.seed + outer_index * 1009 + 100_000
        outer_model, outer_history = train_categorical_scorer(
            outer_examples,
            outer_fingerprints.dimension + 2,
            device,
            seed=outer_model_seed,
            variant="full",
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]),
            replacement_examples=replacement_outer_examples,
            swap_mapping=swap_mapping,
        )
        outer_test_batch = batch.subset(test_ids)
        raw, _, _ = predict_categorical(
            outer_model,
            outer_test_batch,
            outer_fingerprints,
            device,
            temperature=temperature,
            method="cpi_ce_raw",
        )
        frozen_reference = _subset(base_reference, test_ids)
        computed_reference = SourceBestSelector().fit(outer_train_batch, outer_train_labels).predict(outer_test_batch)
        _assert_same_reference(computed_reference, frozen_reference)
        none_fallback = none_fallback_selections(raw, frozen_reference)
        calibrated = apply_categorical_gate(raw, frozen_reference, threshold)
        predictions["cpi_ce_raw"].extend(raw)
        predictions["cpi_ce_none_fallback"].extend(none_fallback)
        predictions["cpi_ce_calibrated"].extend(calibrated)
        reference_predictions.extend(frozen_reference)
        bce_predictions.extend(_subset(base_bce, test_ids))
        training_rows.append(
            {
                "outer_fold": outer_index,
                "outer_environment": environment,
                "role": "outer_prediction",
                "inner_fold": "",
                "inner_heldout_environments": "",
                "train_rows": len(train_ids),
                "prediction_rows": len(test_ids),
                "model_seed": outer_model_seed,
                "initialization_sha256": outer_model.initialization_sha256,
                "initial_loss": outer_history[0],
                "final_loss": outer_history[-1],
            }
        )
        del outer_model

    for values in [*predictions.values(), reference_predictions, bce_predictions]:
        values.sort(key=lambda item: item.question_id)
    write_csv(args.output_dir / "temperature_calibration.csv", temperature_rows)
    write_csv(args.output_dir / "threshold_calibration.csv", threshold_rows)
    write_csv(args.output_dir / "training_history.csv", training_rows)
    write_json(args.output_dir / "selected_calibration.json", selected_calibration)
    prediction_hashes = {
        method: write_selections(args.output_dir / "predictions" / f"{method}.jsonl", predictions[method])
        for method in METHODS
    }

    input_paths = [
        args.config,
        Path(config["protocol_document"]),
        family_map_path,
        Path(config["dataset_registry"]),
        Path(config["source"]["cache_path"]),
        receipt_path,
        *base_paths,
    ]
    manifest = environment_manifest(sys.argv, args.seed, input_paths)
    manifest.update(
        {
            **device_manifest,
            "started_unix": started_wall,
            "protocol": "outer source LOSO with two-fold grouped inner OOF temperature and margin calibration",
            "run_physical_gpu_by_seed": run_gpu_by_seed,
            "base_physical_gpu_by_seed": base_gpu_by_seed,
            "heldout_environments": [fold[0] for fold in folds],
            "source_questions": len(predictions["cpi_ce_calibrated"]),
            "source_environments": len(folds),
            "source_label_structure": label_structure,
            "source_question_ids_sha256": hashlib.sha256(
                "\n".join(item.question_id for item in predictions["cpi_ce_calibrated"]).encode("utf-8")
            ).hexdigest(),
            "selected_calibration": selected_calibration,
            "prediction_hashes_before_evaluation": prediction_hashes,
            "base_prediction_hashes": {
                "source_best_single": str(config["base_artifacts"][args.seed]["source_best_single_sha256"]),
                "deepsets_full": str(config["base_artifacts"][args.seed]["deepsets_full_sha256"]),
            },
        }
    )
    write_json(args.output_dir / "prediction_manifest.json", manifest)

    evaluation_labels = EvaluationLabels(source_labels.dataset, source_labels.split, dict(source_labels.correctness))
    evaluation_batch = batch.subset(item.question_id for item in predictions["cpi_ce_calibrated"])
    summaries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    per_query_paths: list[Path] = []
    for method in METHODS:
        summary, per_query = evaluate(
            method,
            predictions[method],
            reference_predictions,
            evaluation_batch,
            evaluation_labels,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=args.seed,
        )
        summaries[method] = summary
        comparisons[f"{method}_vs_source_best"] = paired_selection_comparison(
            f"{method}_vs_source_best",
            predictions[method],
            reference_predictions,
            evaluation_labels,
            seed=args.seed,
            bootstrap_samples=int(config["bootstrap_samples"]),
        )
        path = args.output_dir / "per_query" / f"{method}.jsonl"
        write_jsonl(path, per_query)
        per_query_paths.append(path)
    comparisons["cpi_ce_raw_vs_frozen_bce_full"] = paired_selection_comparison(
        "cpi_ce_raw_vs_frozen_bce_full",
        predictions["cpi_ce_raw"],
        bce_predictions,
        evaluation_labels,
        seed=args.seed,
        bootstrap_samples=int(config["bootstrap_samples"]),
    )

    primary_by_id = {item.question_id: item for item in predictions["cpi_ce_calibrated"]}
    reference_by_id = {item.question_id: item for item in reference_predictions}
    environment_rows: list[dict[str, Any]] = []
    for environment, _, test_ids in folds:
        deltas = [
            _correct(evaluation_labels, question_id, primary_by_id[question_id])
            - _correct(evaluation_labels, question_id, reference_by_id[question_id])
            for question_id in test_ids
        ]
        environment_rows.append(
            {
                "environment": environment,
                "samples": len(deltas),
                "delta": float(np.mean(deltas)) if deltas else 0.0,
                **selected_calibration[environment],
            }
        )
    environment_deltas = [float(row["delta"]) for row in environment_rows]
    primary_comparison = comparisons["cpi_ce_calibrated_vs_source_best"]
    gate = {
        "seed": args.seed,
        "primary_method": "cpi_ce_calibrated",
        "macro_delta": float(np.mean(environment_deltas)),
        "micro_delta": float(summaries["cpi_ce_calibrated"]["delta_vs_source_best_single"]),
        "worst_environment_delta": min(environment_deltas),
        "nonnegative_environment_fraction": float(np.mean([value >= 0.0 for value in environment_deltas])),
        "required_macro_delta": float(config["required_macro_delta"]),
        "required_worst_delta": float(config["required_worst_delta"]),
        "required_nonnegative_fraction": float(config["required_nonnegative_fraction"]),
        "paired_comparison": primary_comparison,
        "raw_vs_frozen_bce_full": comparisons["cpi_ce_raw_vs_frozen_bce_full"],
    }
    gate["decision"] = (
        "GO"
        if gate["macro_delta"] >= gate["required_macro_delta"]
        and gate["worst_environment_delta"] >= gate["required_worst_delta"]
        and gate["nonnegative_environment_fraction"] >= gate["required_nonnegative_fraction"]
        else "NO-GO"
    )
    torch.cuda.synchronize()
    resource_usage = {
        **device_manifest,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "runtime_seconds": time.perf_counter() - started,
        "finished_unix": time.time(),
    }
    write_json(args.output_dir / "summaries.json", summaries)
    write_json(args.output_dir / "comparisons.json", comparisons)
    write_csv(args.output_dir / "environment_results.csv", environment_rows)
    write_json(args.output_dir / "gate.json", gate)
    write_json(args.output_dir / "resource_usage.json", resource_usage)
    artifacts = files_manifest(
        [
            args.output_dir / "prediction_manifest.json",
            args.output_dir / "predictions",
            args.output_dir / "selected_calibration.json",
            args.output_dir / "temperature_calibration.csv",
            args.output_dir / "threshold_calibration.csv",
            args.output_dir / "training_history.csv",
            args.output_dir / "summaries.json",
            args.output_dir / "comparisons.json",
            args.output_dir / "per_query",
            args.output_dir / "environment_results.csv",
            args.output_dir / "gate.json",
            args.output_dir / "resource_usage.json",
        ]
    )
    write_json(
        args.output_dir / "run_complete_manifest.json",
        {
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "prediction_manifest_sha256": sha256_file(args.output_dir / "prediction_manifest.json"),
            "artifact_hashes": artifacts,
            "determinism": {
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "gate": gate, "resources": resource_usage}, indent=2))


if __name__ == "__main__":
    main()
