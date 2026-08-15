from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
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
from .cpi import fit_source_fingerprints, make_pool_example, subject_folds
from .cpi_ce import none_fallback_selections
from .cpi_remaining import (
    ALL_VARIANT_NAMES,
    FITTED_VARIANTS,
    METHODS,
    PRIMARY_METHOD,
    fit_masked_source_fingerprints,
    max_remaining_invariance_difference,
    predict_remaining,
    train_remaining_scorer,
)
from .data import load_family_map
from .evaluation import evaluate, paired_selection_comparison
from .run_conservative_cpi import _configure_device, _load_base_predictions, _source_adapter, _subset
from .run_cpi_ce import _assert_same_reference, _correct, _validate_label_structure
from .schema import EvaluationLabels, Selection
from .selectors import SourceBestSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remaining source-only CPI experiments on one physical GPU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--variant", action="append", help="Run only a registered fitted variant (smoke/debug only)")
    return parser.parse_args()


def _alias_selection(selection: Selection, method: str) -> Selection:
    return replace(selection, observable_features={**dict(selection.observable_features), "method": method})


def _method_names(variant_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{variant}__{suffix}" for variant in variant_names for suffix in ("raw", "none_fallback"))


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    validate_test_receipt(receipt_path, args.config)
    run_gpu_by_seed = seed_gpu_map(config, "physical_gpus")
    base_gpu_by_seed = seed_gpu_map(config, "base_physical_gpus")
    if args.seed not in run_gpu_by_seed or args.physical_gpu != run_gpu_by_seed[args.seed]:
        raise ValueError("Seed and physical GPU do not match the frozen remaining-source mapping")
    device, device_manifest = _configure_device(args.physical_gpu)

    selected_variants = list(FITTED_VARIANTS)
    if args.variant:
        requested = set(args.variant)
        selected_variants = [variant for variant in selected_variants if variant.name in requested]
        if {variant.name for variant in selected_variants} != requested:
            raise ValueError(f"Unknown requested variants: {sorted(requested.difference(v.name for v in selected_variants))}")
    formal = args.max_environments is None and not args.variant
    variant_names = [variant.name for variant in selected_variants]
    if "int_full" in variant_names:
        variant_names.append("factor_legacy_mean")
    active_methods = _method_names(variant_names)
    if formal and tuple(active_methods) != METHODS:
        raise AssertionError("Formal remaining-source run does not include the frozen method family")

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
        raise ValueError("Remaining-source replacement cache is not aligned")
    label_structure = _validate_label_structure(
        batch,
        source_labels,
        int(config["expected_none_correct_questions"]),
        int(config["expected_one_correct_questions"]),
    )
    _, base_reference, base_paths = _load_base_predictions(config, args.seed, base_gpu_by_seed[args.seed])
    if {item.question_id for item in base_reference} != set(batch.question_ids):
        raise ValueError("Frozen Source-Best predictions do not match the remaining-source cache")

    folds = subject_folds(source_labels)
    if args.max_environments is not None:
        folds = folds[: args.max_environments]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)
    predictions: dict[str, list[Selection]] = {method: [] for method in active_methods}
    references: list[Selection] = []
    training_rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    initialization_by_fold: dict[int, dict[str, list[str]]] = {}
    swap_mapping = {str(key): str(value) for key, value in config["known_swaps"].items()}

    for fold_index, (environment, train_ids, test_ids) in enumerate(folds):
        train_batch = batch.subset(train_ids)
        test_batch = batch.subset(test_ids)
        train_labels = source_labels.subset(train_ids)
        replacement_train_batch = replacement_batch.subset(train_ids)
        replacement_train_labels = replacement_labels.subset(train_ids)
        legacy_fingerprints = fit_source_fingerprints(
            train_batch,
            train_labels,
            rank=int(config["fingerprint_rank"]),
            extra_batch=replacement_train_batch,
            extra_labels=replacement_train_labels,
        )
        masked_fingerprints = fit_masked_source_fingerprints(
            train_batch,
            train_labels,
            rank=int(config["fingerprint_rank"]),
            extra_batch=replacement_train_batch,
            extra_labels=replacement_train_labels,
        )
        fingerprint_tables = {"legacy": legacy_fingerprints, "mask": masked_fingerprints}
        train_examples = {
            mode: [make_pool_example(train_batch, question_id, table, train_labels) for question_id in train_batch.question_ids]
            for mode, table in fingerprint_tables.items()
        }
        replacement_examples = {
            mode: {
                question_id: make_pool_example(
                    replacement_train_batch,
                    question_id,
                    table,
                    replacement_train_labels,
                )
                for question_id in replacement_train_batch.question_ids
            }
            for mode, table in fingerprint_tables.items()
        }
        test_examples = {
            mode: [make_pool_example(test_batch, question_id, table) for question_id in test_batch.question_ids]
            for mode, table in fingerprint_tables.items()
        }
        frozen_reference = _subset(base_reference, test_ids)
        computed_reference = SourceBestSelector().fit(train_batch, train_labels).predict(test_batch)
        _assert_same_reference(computed_reference, frozen_reference)
        references.extend(frozen_reference)
        init_groups: dict[str, list[str]] = {}
        for variant in selected_variants:
            table = fingerprint_tables[variant.fingerprint_mode]
            model_seed = args.seed + fold_index * 1009 + 300_000
            model, history = train_remaining_scorer(
                train_examples[variant.fingerprint_mode],
                train_labels.environment_by_question,
                table.dimension + 2,
                device,
                model_seed,
                variant,
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                hidden_dim=int(config["hidden_dim"]),
                dro_alpha=float(config["dro_alpha"]),
                dro_tau=float(config["dro_tau"]),
                replacement_examples=replacement_examples[variant.fingerprint_mode],
                swap_mapping=swap_mapping,
            )
            init_group = f"{table.dimension + 2}:{variant.cluster_features}"
            init_groups.setdefault(init_group, []).append(model.initialization_sha256)
            raw_method = f"{variant.name}__raw"
            fallback_method = f"{variant.name}__none_fallback"
            raw, _ = predict_remaining(
                model,
                test_batch,
                test_examples[variant.fingerprint_mode],
                device,
                variant.cluster_features,
                raw_method,
            )
            fallback = none_fallback_selections(raw, frozen_reference, method=fallback_method)
            predictions[raw_method].extend(raw)
            predictions[fallback_method].extend(fallback)
            clone_difference = max_remaining_invariance_difference(
                model,
                test_examples[variant.fingerprint_mode],
                device,
                variant.cluster_features,
                "exact_clone",
                model_seed,
            )
            permutation_difference = max_remaining_invariance_difference(
                model,
                test_examples[variant.fingerprint_mode],
                device,
                variant.cluster_features,
                "permutation",
                model_seed,
            )
            invariance_rows.append(
                {
                    "fold": fold_index,
                    "heldout_environment": environment,
                    "variant": variant.name,
                    "exact_clone_logit_difference": clone_difference,
                    "permutation_logit_difference": permutation_difference,
                }
            )
            training_rows.append(
                {
                    "fold": fold_index,
                    "heldout_environment": environment,
                    **asdict(variant),
                    "train_rows": len(train_ids),
                    "validation_rows": len(test_ids),
                    "model_seed": model_seed,
                    "initialization_sha256": model.initialization_sha256,
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                }
            )
            if variant.name == "int_full":
                alias_raw = [_alias_selection(item, "factor_legacy_mean__raw") for item in raw]
                alias_fallback = [_alias_selection(item, "factor_legacy_mean__none_fallback") for item in fallback]
                predictions["factor_legacy_mean__raw"].extend(alias_raw)
                predictions["factor_legacy_mean__none_fallback"].extend(alias_fallback)
            del model
        for group, hashes in init_groups.items():
            if len(set(hashes)) != 1:
                raise AssertionError(f"Paired variants did not share initialization in fold {fold_index}, group {group}")
        initialization_by_fold[fold_index] = init_groups

    for selections in [*predictions.values(), references]:
        selections.sort(key=lambda item: item.question_id)
    write_csv(args.output_dir / "training_history.csv", training_rows)
    write_csv(args.output_dir / "invariance.csv", invariance_rows)
    prediction_hashes = {
        method: write_selections(args.output_dir / "predictions" / f"{method}.jsonl", predictions[method])
        for method in active_methods
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
    primary_available = PRIMARY_METHOD in predictions
    manifest.update(
        {
            **device_manifest,
            "started_unix": started_wall,
            "protocol": "complete subject LOSO remaining-source factorial and intervention training ablations",
            "physical_gpu_by_seed": run_gpu_by_seed,
            "base_physical_gpu_by_seed": base_gpu_by_seed,
            "heldout_environments": [fold[0] for fold in folds],
            "source_questions": len(predictions[PRIMARY_METHOD]) if primary_available else len(references),
            "source_environments": len(folds),
            "source_label_structure": label_structure,
            "active_variants": variant_names,
            "active_methods": list(active_methods),
            "variant_specs": [asdict(variant) for variant in selected_variants],
            "initialization_by_fold": initialization_by_fold,
            "source_question_ids_sha256": hashlib.sha256(
                "\n".join(item.question_id for item in references).encode("utf-8")
            ).hexdigest(),
            "prediction_hashes_before_evaluation": prediction_hashes,
            "base_prediction_hashes": {
                "source_best_single": str(config["base_artifacts"][args.seed]["source_best_single_sha256"]),
            },
        }
    )
    write_json(args.output_dir / "prediction_manifest.json", manifest)

    evaluation_labels = EvaluationLabels(source_labels.dataset, source_labels.split, dict(source_labels.correctness))
    evaluation_batch = batch.subset(item.question_id for item in references)
    summaries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for method in active_methods:
        summary, per_query = evaluate(
            method,
            predictions[method],
            references,
            evaluation_batch,
            evaluation_labels,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=args.seed,
        )
        summaries[method] = summary
        comparisons[method] = paired_selection_comparison(
            f"{method}_vs_source_best",
            predictions[method],
            references,
            evaluation_labels,
            seed=args.seed,
            bootstrap_samples=int(config["bootstrap_samples"]),
        )
        write_jsonl(args.output_dir / "per_query" / f"{method}.jsonl", per_query)
    environment_rows: list[dict[str, Any]] = []
    for method in active_methods:
        candidate = {item.question_id: item for item in predictions[method]}
        reference = {item.question_id: item for item in references}
        for environment, _, test_ids in folds:
            deltas = [
                _correct(evaluation_labels, question_id, candidate[question_id])
                - _correct(evaluation_labels, question_id, reference[question_id])
                for question_id in test_ids
            ]
            environment_rows.append(
                {
                    "method": method,
                    "environment": environment,
                    "samples": len(deltas),
                    "delta": float(np.mean(deltas)) if deltas else 0.0,
                }
            )
    primary_rows = [row for row in environment_rows if row["method"] == PRIMARY_METHOD]
    primary_deltas = [float(row["delta"]) for row in primary_rows]
    max_clone = max(float(row["exact_clone_logit_difference"]) for row in invariance_rows)
    max_permutation = max(float(row["permutation_logit_difference"]) for row in invariance_rows)
    gate: dict[str, Any]
    if primary_available:
        gate = {
            "seed": args.seed,
            "primary_method": PRIMARY_METHOD,
            "accuracy": float(summaries[PRIMARY_METHOD]["accuracy"]),
            "micro_delta": float(summaries[PRIMARY_METHOD]["delta_vs_source_best_single"]),
            "macro_delta": float(np.mean(primary_deltas)),
            "worst_environment_delta": min(primary_deltas),
            "nonnegative_environment_fraction": float(np.mean([value >= 0.0 for value in primary_deltas])),
            "max_exact_clone_logit_difference": max_clone,
            "max_permutation_logit_difference": max_permutation,
            "required_macro_delta": float(config["required_macro_delta"]),
            "required_worst_delta": float(config["required_worst_delta"]),
            "required_nonnegative_fraction": float(config["required_nonnegative_fraction"]),
            "required_invariance_tolerance": float(config["invariance_tolerance"]),
        }
        gate["decision"] = (
            "GO"
            if gate["macro_delta"] >= gate["required_macro_delta"]
            and gate["worst_environment_delta"] >= gate["required_worst_delta"]
            and gate["nonnegative_environment_fraction"] >= gate["required_nonnegative_fraction"]
            and gate["max_exact_clone_logit_difference"] < gate["required_invariance_tolerance"]
            and gate["max_permutation_logit_difference"] < gate["required_invariance_tolerance"]
            else "NO-GO"
        )
    else:
        gate = {"seed": args.seed, "primary_method": PRIMARY_METHOD, "decision": "SMOKE_PRIMARY_NOT_RUN"}
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
            args.output_dir / "training_history.csv",
            args.output_dir / "invariance.csv",
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
