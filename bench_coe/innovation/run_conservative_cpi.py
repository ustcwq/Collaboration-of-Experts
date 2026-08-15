from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    read_selections,
    seed_gpu_map,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_jsonl,
    write_selections,
)
from .conservative_cpi import apply_conservative_gate, calibrate_threshold, grouped_environment_folds
from .cpi import fit_source_fingerprints, make_pool_example, predict_selections, subject_folds, train_cluster_scorer
from .data import CacheAdapter, load_family_map
from .evaluation import evaluate, paired_selection_comparison
from .schema import EvaluationLabels, Selection
from .selectors import SourceBestSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested-OOF Conservative-CPI on one physical GPU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-environments", type=int)
    return parser.parse_args()


def _configure_device(physical_gpu: int) -> tuple[torch.device, dict[str, Any]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Conservative-CPI requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0"), {
        "physical_gpu": physical_gpu,
        "visible_device": 0,
        "cuda_visible_devices": visible,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
    }


def _source_adapter(config: dict[str, Any], experts: list[str], family_map: dict[str, str]) -> CacheAdapter:
    source = config["source"]
    return CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        experts,
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )


def _load_base_predictions(
    config: dict[str, Any],
    seed: int,
    base_physical_gpu: int,
) -> tuple[list[Selection], list[Selection], list[Path]]:
    base_dir = Path(config["base_run_root"]) / f"seed_{seed}_gpu{base_physical_gpu}"
    manifest_path = base_dir / "prediction_manifest.json"
    source_best_path = base_dir / "predictions" / "source_best_single.jsonl"
    full_path = base_dir / "predictions" / "deepsets_full.jsonl"
    expected = config["base_artifacts"][seed]
    if sha256_file(manifest_path) != str(expected["prediction_manifest_sha256"]):
        raise RuntimeError(f"Base prediction manifest hash mismatch for seed {seed}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "source_best_single": (source_best_path, str(expected["source_best_single_sha256"])),
        "deepsets_full": (full_path, str(expected["deepsets_full_sha256"])),
    }
    for method, (path, expected_hash) in checks.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Frozen base prediction hash mismatch: seed={seed}, method={method}")
        if manifest["prediction_hashes_before_evaluation"][method] != expected_hash:
            raise RuntimeError(f"Base manifest does not bind {method} for seed {seed}")
    return read_selections(full_path), read_selections(source_best_path), [manifest_path, source_best_path, full_path]


def _subset(selections: list[Selection], question_ids: tuple[str, ...]) -> list[Selection]:
    by_id = {item.question_id: item for item in selections}
    if set(question_ids).difference(by_id):
        raise ValueError("Frozen base predictions do not cover an outer fold")
    return [by_id[question_id] for question_id in sorted(question_ids)]


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
        raise ValueError("Seed and physical GPU do not match the frozen mapping")
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
        raise ValueError("Replacement expert cache is not aligned")
    base_candidate, base_reference, base_paths = _load_base_predictions(
        config,
        args.seed,
        base_gpu_by_seed[args.seed],
    )
    if {item.question_id for item in base_candidate} != set(batch.question_ids):
        raise ValueError("Frozen CPI predictions do not match the source cache")
    if {item.question_id for item in base_reference} != set(batch.question_ids):
        raise ValueError("Frozen Source-Best predictions do not match the source cache")

    folds = subject_folds(source_labels)
    if args.max_environments is not None:
        folds = folds[: args.max_environments]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)
    gated_predictions: list[Selection] = []
    reference_predictions: list[Selection] = []
    calibration_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    selected_thresholds: dict[str, float] = {}
    swap_mapping = {str(key): str(value) for key, value in config["known_swaps"].items()}

    for outer_index, (environment, train_ids, test_ids) in enumerate(folds):
        outer_train_labels = source_labels.subset(train_ids)
        inner_candidates: list[Selection] = []
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
            examples = [
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
            model_seed = args.seed + outer_index * 1009 + 50_000 + inner_index * 131
            model, history = train_cluster_scorer(
                examples,
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
            candidate, _, _ = predict_selections(
                model,
                inner_test_batch,
                fingerprints,
                device,
                method="inner_oof_cpi_full",
            )
            reference = SourceBestSelector().fit(inner_train_batch, inner_train_labels).predict(inner_test_batch)
            inner_candidates.extend(candidate)
            inner_references.extend(reference)
            training_rows.append(
                {
                    "outer_fold": outer_index,
                    "outer_environment": environment,
                    "inner_fold": inner_index,
                    "inner_heldout_environments": "|".join(heldout_environments),
                    "train_rows": len(inner_train_ids),
                    "calibration_rows": len(inner_test_ids),
                    "model_seed": model_seed,
                    "initialization_sha256": model.initialization_sha256,
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                }
            )
            del model
        threshold, diagnostics = calibrate_threshold(
            inner_candidates,
            inner_references,
            outer_train_labels,
            outer_train_labels.environment_by_question,
            [float(value) for value in config["threshold_grid"]],
            min_worst_delta=float(config["calibration_min_worst_delta"]),
            min_micro_delta=float(config["calibration_min_micro_delta"]),
            worst_weight=float(config["calibration_worst_weight"]),
        )
        selected_thresholds[environment] = threshold
        calibration_rows.extend(
            {
                "outer_fold": outer_index,
                "outer_environment": environment,
                "selected": row.threshold == threshold,
                **asdict(row),
            }
            for row in diagnostics
        )
        outer_candidate = _subset(base_candidate, test_ids)
        outer_reference = _subset(base_reference, test_ids)
        gated_predictions.extend(
            apply_conservative_gate(
                outer_candidate,
                outer_reference,
                threshold,
                method="conservative_cpi_nested_oof",
            )
        )
        reference_predictions.extend(outer_reference)

    gated_predictions.sort(key=lambda item: item.question_id)
    reference_predictions.sort(key=lambda item: item.question_id)
    write_csv(args.output_dir / "calibration_thresholds.csv", calibration_rows)
    write_csv(args.output_dir / "inner_training_history.csv", training_rows)
    write_json(args.output_dir / "selected_thresholds.json", selected_thresholds)
    prediction_hash = write_selections(args.output_dir / "predictions" / "conservative_cpi.jsonl", gated_predictions)
    input_paths = [
        args.config,
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
            "protocol": "outer source LOSO with two-fold grouped inner OOF threshold calibration",
            "run_physical_gpu_by_seed": run_gpu_by_seed,
            "base_physical_gpu_by_seed": base_gpu_by_seed,
            "heldout_environments": [fold[0] for fold in folds],
            "source_questions": len(gated_predictions),
            "source_environments": len(folds),
            "source_question_ids_sha256": hashlib.sha256(
                "\n".join(item.question_id for item in gated_predictions).encode("utf-8")
            ).hexdigest(),
            "selected_thresholds": selected_thresholds,
            "prediction_hashes_before_evaluation": {"conservative_cpi": prediction_hash},
            "base_prediction_hashes": {
                "source_best_single": str(config["base_artifacts"][args.seed]["source_best_single_sha256"]),
                "deepsets_full": str(config["base_artifacts"][args.seed]["deepsets_full_sha256"]),
            },
        }
    )
    write_json(args.output_dir / "prediction_manifest.json", manifest)

    evaluation_labels = EvaluationLabels(source_labels.dataset, source_labels.split, dict(source_labels.correctness))
    evaluation_batch = batch.subset(item.question_id for item in gated_predictions)
    summary, per_query = evaluate(
        "conservative_cpi",
        gated_predictions,
        reference_predictions,
        evaluation_batch,
        evaluation_labels,
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=args.seed,
    )
    paired = paired_selection_comparison(
        "conservative_cpi_vs_source_best",
        gated_predictions,
        reference_predictions,
        evaluation_labels,
        seed=args.seed,
        bootstrap_samples=int(config["bootstrap_samples"]),
    )
    environment_rows: list[dict[str, Any]] = []
    gated_by_id = {item.question_id: item for item in gated_predictions}
    reference_by_id = {item.question_id: item for item in reference_predictions}
    for environment, _, test_ids in folds:
        deltas = [
            float(bool(evaluation_labels.get(qid, gated_by_id[qid].selected_expert_id or "")))
            - float(bool(evaluation_labels.get(qid, reference_by_id[qid].selected_expert_id or "")))
            for qid in test_ids
        ]
        environment_rows.append(
            {
                "environment": environment,
                "samples": len(deltas),
                "delta": float(np.mean(deltas)) if deltas else 0.0,
                "threshold": selected_thresholds[environment],
            }
        )
    environment_deltas = [float(row["delta"]) for row in environment_rows]
    gate = {
        "seed": args.seed,
        "macro_delta": float(np.mean(environment_deltas)),
        "micro_delta": float(summary["delta_vs_source_best_single"]),
        "worst_environment_delta": min(environment_deltas),
        "nonnegative_environment_fraction": float(np.mean([value >= 0.0 for value in environment_deltas])),
        "required_macro_delta": float(config["required_macro_delta"]),
        "required_worst_delta": float(config["required_worst_delta"]),
        "required_nonnegative_fraction": float(config["required_nonnegative_fraction"]),
        "paired_comparison": paired,
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
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "per_query.jsonl", per_query)
    write_csv(args.output_dir / "environment_results.csv", environment_rows)
    write_json(args.output_dir / "gate.json", gate)
    write_json(args.output_dir / "resource_usage.json", resource_usage)
    artifacts = files_manifest(
        [
            args.output_dir / "prediction_manifest.json",
            args.output_dir / "predictions",
            args.output_dir / "selected_thresholds.json",
            args.output_dir / "calibration_thresholds.csv",
            args.output_dir / "inner_training_history.csv",
            args.output_dir / "summary.json",
            args.output_dir / "per_query.jsonl",
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
