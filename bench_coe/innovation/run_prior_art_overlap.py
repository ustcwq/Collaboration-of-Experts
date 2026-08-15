from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_jsonl,
    write_selections,
)
from .data import CacheAdapter, EvaluationLabelAdapter, load_family_map
from .evaluation import evaluate, holm_adjust, paired_selection_comparison, selection_correctness
from .features import records_by_question
from .prior_art_overlap import (
    cascade_selections,
    fcrg_ablation_selections,
    fcrg_feature_rows,
    fit_fcrg_weight_model,
    fit_predict_prior_art_baselines,
    knop_sensitivity_selections,
    learned_fcrg_selections,
)
from .repair_simplification import fit_repair_components, subset_expert_pool
from .response_embeddings import MiniLMResponseEncoder
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import LegacyDARESelector, LegacyRepairChainSelector, correctness_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one authenticated Improve5/6 prior-art overlap seed")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--max-inner-environments", type=int)
    return parser.parse_args()


def configure_device(seed: int, physical_gpu: int) -> tuple[torch.device, dict[str, Any]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("PYTHONHASHSEED must match the pre-registered seed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The prior-art run requires exactly one visible CUDA device")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0"), {
        "physical_gpu": physical_gpu,
        "visible_device": 0,
        "cuda_visible_devices": visible,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
    }


def environment_folds(labels: SourceTrainingLabels) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    by_environment: dict[str, list[str]] = defaultdict(list)
    for question_id, environment in labels.environment_by_question.items():
        by_environment[str(environment)].append(question_id)
    all_ids = set(labels.environment_by_question)
    return [
        (environment, tuple(sorted(all_ids.difference(test_ids))), tuple(sorted(test_ids)))
        for environment, test_ids in sorted(by_environment.items())
    ]


def retag(selections: list[Selection], method: str) -> list[Selection]:
    result: list[Selection] = []
    for selection in selections:
        features = dict(selection.observable_features)
        features["method"] = method
        result.append(replace(selection, observable_features=features))
    return result


def legacy_equivalence_diagnostics(
    batch: ObservableQueryBatch,
    modern: list[Selection],
    legacy: list[Selection],
) -> dict[str, Any]:
    modern_by_id = {selection.question_id: selection for selection in modern}
    legacy_by_id = {selection.question_id: selection for selection in legacy}
    expected_ids = set(batch.question_ids)
    if set(modern_by_id) != expected_ids or set(legacy_by_id) != expected_ids:
        raise ValueError("Legacy-equivalence selections do not align with the observable batch")
    grouped = records_by_question(batch)
    complete_ids = {
        question_id
        for question_id, records in grouped.items()
        if all(
            record.valid_output and record.per_query_cluster_id is not None
            for record in records
        )
    }
    complete_mismatches = sorted(
        question_id
        for question_id in complete_ids
        if modern_by_id[question_id].normalized_answer
        != legacy_by_id[question_id].normalized_answer
    )
    incomplete_differences = sorted(
        question_id
        for question_id in expected_ids.difference(complete_ids)
        if modern_by_id[question_id].normalized_answer
        != legacy_by_id[question_id].normalized_answer
    )
    return {
        "complete_question_count": len(complete_ids),
        "incomplete_question_count": len(expected_ids.difference(complete_ids)),
        "complete_answer_mismatch_count": len(complete_mismatches),
        "complete_answer_mismatch_ids": complete_mismatches,
        "incomplete_answer_difference_count": len(incomplete_differences),
        "incomplete_answer_difference_ids": incomplete_differences,
        "scope": (
            "legacy equivalence is required only for complete expert-output rows; "
            "legacy code groups missing outputs as answers, which the observable protocol forbids"
        ),
    }


def fit_inner_fcrg_weight_model(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    *,
    neighbors: int,
    seed: int,
    device: torch.device,
    max_environments: int | None,
) -> tuple[Any | None, float, dict[str, Any]]:
    folds = environment_folds(labels)
    if max_environments is not None:
        folds = folds[:max_environments]
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    heldout_environments: list[str] = []
    for fold_index, (environment, train_ids, test_ids) in enumerate(folds):
        inner_train = batch.subset(train_ids)
        inner_test = batch.subset(test_ids)
        inner_labels = labels.subset(train_ids)
        components = fit_repair_components(
            inner_train,
            inner_labels,
            inner_test,
            neighbors=neighbors,
            seed=seed + fold_index,
            device=device,
        )
        feature_rows.append(fcrg_feature_rows(components))
        label_rows.append(correctness_matrix(inner_test, labels.subset(test_ids)).reshape(-1))
        heldout_environments.append(environment)
    if not feature_rows:
        raise ValueError("No inner environments are available for learned FCRG weights")
    x = np.concatenate(feature_rows, axis=0)
    y = np.concatenate(label_rows, axis=0)
    model, constant, diagnostics = fit_fcrg_weight_model(x, y, seed)
    diagnostics.update(
        {
            "inner_environments": heldout_environments,
            "inner_environment_count": len(heldout_environments),
            "training_rows": len(y),
            "positive_fraction": float(y.mean()) if len(y) else 0.0,
            "provenance": "each competence feature row is generated by an inner held-environment-out model",
        }
    )
    return model, constant, diagnostics


def run_fold_pool(
    train_batch: ObservableQueryBatch,
    train_labels: SourceTrainingLabels,
    test_batch: ObservableQueryBatch,
    *,
    config: Mapping[str, Any],
    seed: int,
    fold_index: int,
    device: torch.device,
    full_protocol: bool,
    max_inner_environments: int | None,
    response_encoder: MiniLMResponseEncoder | None,
) -> tuple[dict[str, list[Selection]], dict[str, Any]]:
    neighbors = int(config["knn_k"])
    baseline_bundle = fit_predict_prior_art_baselines(
        train_batch,
        train_labels,
        test_batch,
        neighbors=neighbors,
        seed=seed + fold_index,
        mcb_behavior_threshold=float(config["mcb_behavior_threshold"]),
        mcb_min_neighbors=int(config["mcb_min_neighbors"]),
        include_mlp=full_protocol,
        response_encoder=response_encoder,
    )
    selections = dict(baseline_bundle.selections)
    oprs = LegacyDARESelector(neighbors=neighbors).fit(train_batch, train_labels).predict(test_batch)
    selections["oprs_robust_output_profile"] = retag(oprs, "oprs_robust_output_profile")

    components = fit_repair_components(
        train_batch,
        train_labels,
        test_batch,
        neighbors=neighbors,
        seed=seed + fold_index,
        device=device,
    )
    fcrg, fcrg_diagnostics = fcrg_ablation_selections(
        test_batch,
        components,
        seed=seed + fold_index,
        device=device,
    )
    selections.update(fcrg)

    learned_diagnostics: dict[str, Any] | None = None
    if full_protocol:
        weight_model, weight_constant, learned_diagnostics = fit_inner_fcrg_weight_model(
            train_batch,
            train_labels,
            neighbors=neighbors,
            seed=seed + 1000 * (fold_index + 1),
            device=device,
            max_environments=max_inner_environments,
        )
        selections["fcrg_learned_weights"] = learned_fcrg_selections(
            test_batch, components, weight_model, weight_constant
        )
        selections.update(
            knop_sensitivity_selections(
                train_batch,
                train_labels,
                test_batch,
                [int(value) for value in config["knn_sensitivity"]],
            )
        )
        fast = selections["fast_global_best_single_call"]
        full = selections["fcrg_full"]
        cascade_diagnostics: dict[str, Any] = {}
        for raw_threshold in config["cascade_uncertainty_thresholds"]:
            threshold = float(raw_threshold)
            suffix = str(threshold).replace(".", "p")
            method = f"cascade_fcrg_u_gt_{suffix}"
            cascade, diagnostics = cascade_selections(test_batch, fast, full, threshold)
            selections[method] = retag(cascade, method)
            cascade_diagnostics[method] = diagnostics
    else:
        cascade_diagnostics = {}

    legacy = LegacyRepairChainSelector(neighbors=neighbors).fit(train_batch, train_labels).predict(test_batch)
    legacy_equivalence = legacy_equivalence_diagnostics(
        test_batch, selections["fcrg_full"], legacy
    )
    mismatch_ids = legacy_equivalence["complete_answer_mismatch_ids"]
    if mismatch_ids:
        raise RuntimeError(
            "FCRG full does not reproduce legacy RepairChain answers on complete rows: "
            f"{mismatch_ids[:5]}"
        )

    diagnostics = {
        "baseline": baseline_bundle.diagnostics,
        "fcrg": fcrg_diagnostics,
        "learned_fcrg": learned_diagnostics,
        "cascade": cascade_diagnostics,
        "legacy_equivalence": legacy_equivalence,
        "legacy_answer_mismatch_count": len(mismatch_ids),
    }
    return selections, diagnostics


def method_pool(method: str, pool_names: tuple[str, ...]) -> str:
    for pool_name in pool_names:
        prefix = f"{pool_name}__"
        if method.startswith(prefix):
            return pool_name
    return "full_pool"


def method_cost(
    method: str,
    selections: list[Selection],
    batch: ObservableQueryBatch,
) -> dict[str, Any]:
    grouped = records_by_question(batch)
    calls: list[float] = []
    serial_latency: list[float] = []
    for selection in selections:
        records = grouped[selection.question_id]
        by_expert = {record.expert_id: record for record in records}
        if method.endswith("fast_global_best_single_call") or method == "fast_global_best_single_call":
            called = [by_expert[selection.selected_expert_id or ""]]
        elif "cascade_fcrg" in method:
            triggered = bool(selection.observable_features.get("cascade_triggered", False))
            if triggered:
                called = list(records)
            else:
                called = [by_expert[selection.selected_expert_id or ""]]
        else:
            called = list(records)
        calls.append(float(len(called)))
        serial_latency.append(float(sum(record.inference_cost or 0.0 for record in called)))
    return {
        "method": method,
        "pool_experts": len(batch.pool.expert_ids),
        "mean_nominal_model_calls": float(np.mean(calls)),
        "mean_cached_serial_latency_seconds": float(np.mean(serial_latency)),
        "latency_note": "sum of cached per-model generation latencies; not a measured parallel wall-clock latency",
    }


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    seeds = [int(value) for value in config["seeds"]]
    gpus = [int(value) for value in config["run_physical_gpus"]]
    expected_gpu = dict(zip(seeds, gpus, strict=True))
    if expected_gpu.get(args.seed) != args.physical_gpu:
        raise ValueError("Seed and physical GPU do not match the frozen mapping")
    device, device_manifest = configure_device(args.seed, args.physical_gpu)
    encoder_config = config.get("response_embedding")
    response_encoder = (
        MiniLMResponseEncoder(
            str(encoder_config["model_id"]),
            device,
            batch_size=int(encoder_config.get("batch_size", 256)),
            max_length=int(encoder_config.get("max_length", 128)),
        )
        if encoder_config
        else None
    )

    family_map_path = Path(config["family_map"])
    registry_path = Path(config["dataset_registry"])
    source = config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        load_family_map(family_map_path),
        [str(value) for value in config["experts"]],
        registry_path,
        str(config["dataset_registry_sha256"]),
    )
    batch = adapter.load_observables()
    labels = adapter.load_source_labels()
    if args.max_environments is None and len(batch.question_ids) != int(config["expected_source_questions"]):
        raise RuntimeError("Source question count differs from the frozen protocol")

    pool_batches: dict[str, tuple[ObservableQueryBatch, SourceTrainingLabels]] = {
        "full_pool": (batch, labels)
    }
    for pool_name, expert_ids in sorted(config["pool_shift"]["pools"].items()):
        pool_batches[str(pool_name)] = subset_expert_pool(
            batch,
            labels,
            tuple(str(value) for value in expert_ids),
        )
    pool_names = tuple(sorted(name for name in pool_batches if name != "full_pool"))
    folds = environment_folds(labels)
    if args.max_environments is not None:
        folds = folds[: args.max_environments]
    if not folds:
        raise ValueError("No source environments are available")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)
    prediction_groups: dict[str, list[Selection]] = defaultdict(list)
    method_group: dict[str, str] = {}
    fold_by_question: dict[str, str] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    pool_method_filter = set(str(value) for value in config["pool_shift"]["methods"])

    for fold_index, (environment, train_ids, test_ids) in enumerate(folds):
        for question_id in test_ids:
            fold_by_question[question_id] = environment
        for pool_name, (pool_batch, pool_labels) in sorted(pool_batches.items()):
            full_protocol = pool_name == "full_pool"
            fold_selections, diagnostics = run_fold_pool(
                pool_batch.subset(train_ids),
                pool_labels.subset(train_ids),
                pool_batch.subset(test_ids),
                config=config,
                seed=args.seed,
                fold_index=fold_index,
                device=device,
                full_protocol=full_protocol,
                max_inner_environments=args.max_inner_environments,
                response_encoder=response_encoder,
            )
            if not full_protocol:
                fold_selections = {
                    method: values for method, values in fold_selections.items() if method in pool_method_filter
                }
            for method, values in fold_selections.items():
                stored_method = method if full_protocol else f"{pool_name}__{method}"
                prediction_groups[stored_method].extend(retag(values, stored_method))
                method_group[stored_method] = pool_name
            diagnostic_rows.append(
                {
                    "heldout_environment": environment,
                    "fold_index": fold_index,
                    "pool": pool_name,
                    "diagnostics": diagnostics,
                }
            )

    expected_ids = {question_id for _, _, test_ids in folds for question_id in test_ids}
    for method, values in prediction_groups.items():
        values.sort(key=lambda selection: selection.question_id)
        if {selection.question_id for selection in values} != expected_ids:
            raise RuntimeError(f"Method {method} does not cover the complete held-out query set")
    if args.max_environments is None:
        expected_methods = int(config["expected_full_pool_methods"]) + len(pool_names) * len(pool_method_filter)
        if len(prediction_groups) != expected_methods:
            raise RuntimeError(
                f"Method count differs from protocol: observed={len(prediction_groups)}, expected={expected_methods}"
            )

    write_jsonl(args.output_dir / "fold_diagnostics.jsonl", diagnostic_rows)
    prediction_hashes: dict[str, str] = {}
    for method, values in sorted(prediction_groups.items()):
        prediction_hashes[method] = write_selections(args.output_dir / "predictions" / f"{method}.jsonl", values)

    input_paths = [args.config, family_map_path, registry_path, Path(source["cache_path"])]
    if response_encoder is not None:
        input_paths.append(response_encoder.snapshot)
    run_environment = environment_manifest(sys.argv, args.seed, input_paths)
    write_json(args.output_dir / "environment.json", run_environment)
    question_ids_sha = hashlib.sha256("\n".join(sorted(expected_ids)).encode("utf-8")).hexdigest()
    prediction_manifest = {
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "source_question_ids_sha256": question_ids_sha,
        "source_questions": len(expected_ids),
        "source_environments": len(folds),
        "method_count": len(prediction_groups),
        "method_group": method_group,
        "prediction_hashes_before_evaluation": prediction_hashes,
        "input_manifest_sha256": run_environment["input_manifest_sha256"],
        "innovation_code_manifest_sha256": run_environment["innovation_code_manifest_sha256"],
        "labels_opened": False,
        "response_embedding": response_encoder.diagnostics() if response_encoder is not None else None,
    }
    write_json(args.output_dir / "prediction_manifest.json", prediction_manifest)

    # Evaluation labels are opened only after every prediction is serialized and hashed.
    evaluation_labels = EvaluationLabelAdapter.from_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        [str(value) for value in config["experts"]],
        registry_path,
        str(config["dataset_registry_sha256"]),
    ).load()
    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    per_environment_rows: list[dict[str, Any]] = []
    evaluation_query_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for index, (method, selections) in enumerate(sorted(prediction_groups.items())):
        pool_name = method_group[method]
        pool_batch = pool_batches[pool_name][0].subset(expected_ids)
        reference_name = "global_best_posthoc" if pool_name == "full_pool" else f"{pool_name}__global_best_posthoc"
        reference = prediction_groups[reference_name]
        summary, query_rows = evaluate(
            method,
            selections,
            reference,
            pool_batch,
            evaluation_labels,
            bootstrap_samples=int(config["seed_bootstrap_samples"]),
            seed=args.seed + index,
        )
        summary["pool"] = pool_name
        summaries.append(summary)
        paired_rows.append(
            paired_selection_comparison(
                f"{method}_vs_{reference_name}",
                selections,
                reference,
                evaluation_labels,
                seed=args.seed + 10000 + index,
                bootstrap_samples=int(config["seed_bootstrap_samples"]),
            )
        )
        for row in query_rows:
            evaluation_query_rows.append(
                {
                    "question_id": row["question_id"],
                    "method": method,
                    "pool": pool_name,
                    "correct": row["correct"],
                    "baseline_correct": row["baseline_correct"],
                    "rescued": row["rescued"],
                    "harmed": row["harmed"],
                }
            )
        correctness = selection_correctness(selections, evaluation_labels)
        by_environment: dict[str, list[bool]] = defaultdict(list)
        for question_id, value in correctness.items():
            by_environment[fold_by_question[question_id]].append(value)
        for environment, values in sorted(by_environment.items()):
            per_environment_rows.append(
                {
                    "environment": environment,
                    "method": method,
                    "pool": pool_name,
                    "samples": len(values),
                    "accuracy": float(np.mean(values)),
                }
            )
        cost_rows.append(method_cost(method, selections, pool_batch))

    holm = holm_adjust({str(row["comparison"]): float(row["exact_mcnemar_p"]) for row in paired_rows})
    for row in paired_rows:
        row["holm"] = holm[str(row["comparison"])]
    write_json(args.output_dir / "summary.json", summaries)
    write_csv(
        args.output_dir / "summary.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in summaries],
    )
    write_csv(args.output_dir / "per_environment.csv", per_environment_rows)
    write_json(args.output_dir / "paired_comparisons.json", paired_rows)
    write_csv(
        args.output_dir / "paired_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in paired_rows],
    )
    write_jsonl(args.output_dir / "evaluation_per_query.jsonl", evaluation_query_rows)
    write_csv(args.output_dir / "inference_costs.csv", cost_rows)

    summary_by_method = {str(row["method"]): row for row in summaries}
    full_accuracy = float(summary_by_method["fcrg_full"]["accuracy"])
    seed_gate = {
        "fcrg_full_accuracy": full_accuracy,
        "global_best_accuracy": float(summary_by_method["global_best_posthoc"]["accuracy"]),
        "knop_accuracy": float(summary_by_method["knop_output_profile"]["accuracy"]),
        "h1_only_accuracy": float(summary_by_method["fcrg_h1_only"]["accuracy"]),
        "h1_h2_accuracy": float(summary_by_method["fcrg_h1_h2"]["accuracy"]),
        "graph_control_accuracies": {
            method: float(summary_by_method[method]["accuracy"])
            for method in (
                "fcrg_column_mean_only",
                "fcrg_symmetric",
                "fcrg_random_edges",
                "fcrg_degree_relabel",
            )
        },
        "development_ood_opened": False,
        "legacy_answer_mismatch_count": int(
            sum(row["diagnostics"]["legacy_answer_mismatch_count"] for row in diagnostic_rows)
        ),
    }
    write_json(args.output_dir / "seed_gate.json", seed_gate)

    torch.cuda.synchronize()
    resource = {
        **device_manifest,
        "seed": args.seed,
        "started_unix": started_wall,
        "finished_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    write_json(args.output_dir / "resource_usage.json", resource)
    completion_paths = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "complete_manifest.json"
    ]
    write_json(
        args.output_dir / "complete_manifest.json",
        {
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "prediction_hashes_before_evaluation": prediction_hashes,
            "artifact_hashes": files_manifest(completion_paths),
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "seed_gate": seed_gate}, indent=2))


if __name__ == "__main__":
    main()
