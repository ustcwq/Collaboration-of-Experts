from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from .artifacts import environment_manifest, files_manifest, sha256_file, validate_test_receipt, write_csv, write_json, write_jsonl, write_selections
from .cpi import (
    INTERVENTIONS,
    FingerprintTable,
    PoolExample,
    _stable_rng,
    apply_intervention,
    clone_invariance_loss,
    fit_source_fingerprints,
    make_pool_example,
    max_probability_difference,
    predict_selections,
    relabel_clusters,
    remove_expert,
    remove_family,
    score_examples,
    selections_from_scores,
    subject_folds,
    subset_pool_size,
    train_cluster_scorer,
)
from .data import CacheAdapter, load_family_map
from .evaluation import evaluate, holm_adjust, paired_selection_comparison
from .schema import EvaluationLabels, ObservableQueryBatch, Selection
from .selectors import (
    FamilyBalancedVoteSelector,
    LegacyDARESelector,
    LegacyRepairChainSelector,
    MajorityVoteSelector,
    OutputProfileKNNSelector,
    SourceBestSelector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CPI seed on one visible CUDA device")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-environments", type=int)
    return parser.parse_args()


def _configure_device(physical_gpu: int) -> tuple[torch.device, dict[str, Any]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={physical_gpu} for physical GPU {physical_gpu}, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CPI GPU run requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0"), {
        "physical_gpu": physical_gpu,
        "visible_device": 0,
        "cuda_visible_devices": visible,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
    }


def _examples(
    batch: ObservableQueryBatch,
    fingerprints: FingerprintTable,
    labels=None,
) -> list[PoolExample]:
    return [make_pool_example(batch, question_id, fingerprints, labels) for question_id in batch.question_ids]


def _custom_predictions(
    model,
    batch: ObservableQueryBatch,
    fingerprints: FingerprintTable,
    device: torch.device,
    method: str,
    transform: Callable[[PoolExample], PoolExample],
) -> tuple[list[Selection], list[dict[int, float]], list[PoolExample]]:
    examples = [transform(make_pool_example(batch, question_id, fingerprints)) for question_id in batch.question_ids]
    scores = score_examples(model, examples, device)
    return selections_from_scores(batch, examples, scores, method), scores, examples


def _accuracy(selections: list[Selection], labels: EvaluationLabels) -> float:
    return float(
        np.mean(
            [
                bool(labels.get(selection.question_id, selection.selected_expert_id or ""))
                for selection in selections
            ]
        )
    ) if selections else 0.0


def _calibration_rows(
    method: str,
    selections: list[Selection],
    labels: EvaluationLabels,
    bins: int = 10,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for selection in selections:
        confidence = max(selection.cluster_scores.values(), default=0.0)
        correct = float(bool(labels.get(selection.question_id, selection.selected_expert_id or "")))
        index = min(bins - 1, int(confidence * bins))
        grouped[index].append((confidence, correct))
    return [
        {
            "method": method,
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": len(values),
            "mean_confidence": float(np.mean([value[0] for value in values])),
            "accuracy": float(np.mean([value[1] for value in values])),
        }
        for index, values in sorted(grouped.items())
    ]


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    validate_test_receipt(receipt_path, args.config)
    if args.seed not in [int(value) for value in config["seeds"]]:
        raise ValueError(f"Seed {args.seed} was not pre-registered")
    device, device_manifest = _configure_device(args.physical_gpu)
    family_map_path = Path(config["family_map"])
    family_map = load_family_map(family_map_path)
    source = config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        source["dataset"],
        source["split"],
        source["modality"],
        family_map,
        config["experts"],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    replacement_adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        source["dataset"],
        source["split"],
        source["modality"],
        family_map,
        config["replacement_experts"],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    batch = adapter.load_observables()
    source_labels = adapter.load_source_labels()
    replacement_batch = replacement_adapter.load_observables()
    replacement_labels = replacement_adapter.load_source_labels()
    if replacement_batch.question_ids != batch.question_ids:
        raise ValueError("Configured real swap experts do not align with the primary source")
    swap_mapping = {str(key): str(value) for key, value in config["known_swaps"].items()}
    folds = subject_folds(source_labels)
    if args.max_environments is not None:
        folds = folds[: args.max_environments]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)
    prediction_groups: dict[str, list[Selection]] = defaultdict(list)
    probability_groups: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    training_rows: list[dict[str, Any]] = []
    family_names = tuple(sorted(set(batch.pool.family_by_expert.values())))
    expert_ids = batch.pool.expert_ids

    baseline_factories = {
        "source_best_single": SourceBestSelector,
        "majority_vote": MajorityVoteSelector,
        "family_balanced_vote": FamilyBalancedVoteSelector,
        "output_profile_knn": lambda: OutputProfileKNNSelector(neighbors=32),
        "improve5_dare": LegacyDARESelector,
        "improve6_repair_chain": lambda: LegacyRepairChainSelector(neighbors=32),
    }

    for fold_index, (environment, train_ids, test_ids) in enumerate(folds):
        train_batch = batch.subset(train_ids)
        test_batch = batch.subset(test_ids)
        train_labels = source_labels.subset(train_ids)
        replacement_train_batch = replacement_batch.subset(train_ids)
        replacement_test_batch = replacement_batch.subset(test_ids)
        replacement_train_labels = replacement_labels.subset(train_ids)
        fingerprints = fit_source_fingerprints(
            train_batch,
            train_labels,
            rank=int(config["fingerprint_rank"]),
            extra_batch=replacement_train_batch,
            extra_labels=replacement_train_labels,
        )
        train_examples = _examples(train_batch, fingerprints, train_labels)
        replacement_train_examples = {
            example.question_id: example
            for example in _examples(replacement_train_batch, fingerprints, replacement_train_labels)
        }
        replacement_test_examples = {
            example.question_id: example for example in _examples(replacement_test_batch, fingerprints)
        }
        input_dim = fingerprints.dimension + 2

        for method, factory in baseline_factories.items():
            selector = factory().fit(train_batch, train_labels)
            prediction_groups[method].extend(selector.predict(test_batch))

        models = {}
        for variant in ("none", "full"):
            model, history = train_cluster_scorer(
                train_examples,
                input_dim,
                device,
                seed=args.seed + fold_index * 1009,
                variant=variant,
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                hidden_dim=int(config["hidden_dim"]),
                replacement_examples=replacement_train_examples,
                swap_mapping=swap_mapping,
            )
            models[variant] = model
            training_rows.append(
                {
                    "stage": "source_loso",
                    "fold": fold_index,
                    "heldout_environment": environment,
                    "variant": variant,
                    "train_rows": len(train_ids),
                    "validation_rows": len(test_ids),
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                    "initialization_sha256": model.initialization_sha256,
                }
            )
            method = f"deepsets_{variant}"
            selections, scores, _ = predict_selections(model, test_batch, fingerprints, device, method)
            prediction_groups[method].extend(selections)
            probability_groups[method].update({qid: score for qid, score in zip(test_batch.question_ids, scores)})
        if models["none"].initialization_sha256 != models["full"].initialization_sha256:
            raise AssertionError("CPI paired variants did not start from identical weights")

        for intervention in INTERVENTIONS:
            method = f"cpi_full__{intervention}"
            selections, scores, _ = predict_selections(
                models["full"],
                test_batch,
                fingerprints,
                device,
                method,
                intervention=intervention,
                seed=args.seed + fold_index,
                replacement_examples=replacement_test_examples,
                swap_mapping=swap_mapping,
            )
            prediction_groups[method].extend(selections)
            probability_groups[method].update({qid: score for qid, score in zip(test_batch.question_ids, scores)})

        for family in family_names:
            for variant in ("none", "full"):
                method = f"deepsets_{variant}__remove_family__{family}"
                selections, scores, _ = _custom_predictions(
                    models[variant], test_batch, fingerprints, device, method, lambda example, f=family: remove_family(example, f)
                )
                prediction_groups[method].extend(selections)
                probability_groups[method].update({qid: score for qid, score in zip(test_batch.question_ids, scores)})

        for expert in expert_ids:
            method = f"cpi_full__remove_expert__{expert}"
            selections, _, _ = _custom_predictions(
                models["full"], test_batch, fingerprints, device, method, lambda example, e=expert: remove_expert(example, e)
            )
            prediction_groups[method].extend(selections)

        for pool_size in config["pool_sizes"]:
            method = f"cpi_full__pool_size__{pool_size}"
            selections, _, _ = _custom_predictions(
                models["full"],
                test_batch,
                fingerprints,
                device,
                method,
                lambda example, size=int(pool_size), fold=fold_index: subset_pool_size(
                    example, size, _stable_rng(args.seed, fold, example.question_id, size)
                ),
            )
            prediction_groups[method].extend(selections)
        del models

    # Each intervention is independently fitted on one pre-registered fixed source split.
    environments = sorted(set(source_labels.environment_by_question.values()))
    stride = int(config["fixed_validation_stride"])
    validation_environments = {environment for index, environment in enumerate(environments) if index % stride == 0}
    fixed_test_ids = tuple(sorted(qid for qid, env in source_labels.environment_by_question.items() if env in validation_environments))
    fixed_train_ids = tuple(sorted(set(batch.question_ids).difference(fixed_test_ids)))
    fixed_train_batch = batch.subset(fixed_train_ids)
    fixed_test_batch = batch.subset(fixed_test_ids)
    fixed_train_labels = source_labels.subset(fixed_train_ids)
    fixed_replacement_train_batch = replacement_batch.subset(fixed_train_ids)
    fixed_replacement_test_batch = replacement_batch.subset(fixed_test_ids)
    fixed_replacement_train_labels = replacement_labels.subset(fixed_train_ids)
    fixed_fingerprints = fit_source_fingerprints(
        fixed_train_batch,
        fixed_train_labels,
        rank=int(config["fingerprint_rank"]),
        extra_batch=fixed_replacement_train_batch,
        extra_labels=fixed_replacement_train_labels,
    )
    fixed_examples = _examples(fixed_train_batch, fixed_fingerprints, fixed_train_labels)
    fixed_replacement_examples = {
        example.question_id: example
        for example in _examples(fixed_replacement_train_batch, fixed_fingerprints, fixed_replacement_train_labels)
    }
    ablation_variants = ("linear_full", "none", *INTERVENTIONS, "full")
    ablation_predictions: dict[str, list[Selection]] = {}
    for variant_index, name in enumerate(ablation_variants):
        linear = name == "linear_full"
        training_variant = "full" if linear else name
        model, history = train_cluster_scorer(
            fixed_examples,
            fixed_fingerprints.dimension + 2,
            device,
            seed=args.seed + 100_000 + (1 if linear else 0),
            variant=training_variant,
            epochs=int(config["ablation_epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]),
            linear=linear,
            replacement_examples=fixed_replacement_examples,
            swap_mapping=swap_mapping,
        )
        method = f"fixed_ablation__{name}"
        selections, _, _ = predict_selections(model, fixed_test_batch, fixed_fingerprints, device, method)
        ablation_predictions[method] = selections
        training_rows.append(
            {
                "stage": "fixed_ablation",
                "fold": 0,
                "heldout_environment": ",".join(sorted(validation_environments)),
                "variant": name,
                "train_rows": len(fixed_train_ids),
                "validation_rows": len(fixed_test_ids),
                "initial_loss": history[0],
                "final_loss": history[-1],
                "initialization_sha256": model.initialization_sha256,
            }
        )

    # The sanity run intentionally fits and scores the same 100 source rows.
    sanity_ids = tuple(batch.question_ids[:100])
    sanity_batch = batch.subset(sanity_ids)
    sanity_labels = source_labels.subset(sanity_ids)
    sanity_replacement_batch = replacement_batch.subset(sanity_ids)
    sanity_replacement_labels = replacement_labels.subset(sanity_ids)
    sanity_fingerprints = fit_source_fingerprints(
        sanity_batch,
        sanity_labels,
        rank=int(config["fingerprint_rank"]),
        extra_batch=sanity_replacement_batch,
        extra_labels=sanity_replacement_labels,
    )
    sanity_examples = _examples(sanity_batch, sanity_fingerprints, sanity_labels)
    sanity_replacement_examples = {
        example.question_id: example
        for example in _examples(sanity_replacement_batch, sanity_fingerprints, sanity_replacement_labels)
    }
    sanity_model, sanity_history = train_cluster_scorer(
        sanity_examples,
        sanity_fingerprints.dimension + 2,
        device,
        seed=args.seed + 200_000,
        variant="full",
        epochs=int(config["overfit_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        hidden_dim=int(config["hidden_dim"]),
        replacement_examples=sanity_replacement_examples,
        swap_mapping=swap_mapping,
    )
    sanity_selections, _, _ = predict_selections(
        sanity_model, sanity_batch, sanity_fingerprints, device, "overfit_sanity"
    )

    prediction_hashes: dict[str, str] = {}
    for method, selections in sorted(prediction_groups.items()):
        ordered = sorted(selections, key=lambda item: item.question_id)
        prediction_groups[method] = ordered
        prediction_hashes[method] = write_selections(args.output_dir / "predictions" / f"{method}.jsonl", ordered)
    for method, selections in sorted(ablation_predictions.items()):
        prediction_hashes[method] = write_selections(
            args.output_dir / "predictions" / "ablations" / f"{method}.jsonl",
            sorted(selections, key=lambda item: item.question_id),
        )
    prediction_hashes["overfit_sanity"] = write_selections(
        args.output_dir / "predictions" / "overfit_sanity.jsonl", sanity_selections
    )
    manifest = environment_manifest(
        sys.argv,
        args.seed,
        [
            args.config,
            family_map_path,
            Path(config["dataset_registry"]),
            Path(source["cache_path"]),
            receipt_path,
        ],
    )
    manifest.update(
        {
            **device_manifest,
            "started_unix": started_wall,
            "heldout_environments": [fold[0] for fold in folds],
            "fixed_validation_environments": sorted(validation_environments),
            "prediction_hashes_before_evaluation": prediction_hashes,
            "protocol": "source LOSO; heldout labels excluded from fingerprints, training examples, and prediction",
            "source_question_ids_sha256": hashlib.sha256("\n".join(batch.question_ids).encode("utf-8")).hexdigest(),
            "source_questions": len(batch.question_ids),
            "source_environments": len(folds),
        }
    )
    write_json(args.output_dir / "prediction_manifest.json", manifest)

    # Evaluation begins only after every prediction file and hash above is immutable.
    evaluation_labels = EvaluationLabels(
        source_labels.dataset,
        source_labels.split,
        {**dict(source_labels.correctness), **dict(replacement_labels.correctness)},
    )
    baseline = prediction_groups["source_best_single"]
    summaries: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for method, selections in sorted(prediction_groups.items()):
        summary, per_query = evaluate(
            method,
            selections,
            baseline,
            batch.subset(selection.question_id for selection in selections),
            evaluation_labels,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=args.seed,
        )
        summaries.append(summary)
        if method in {"deepsets_none", "deepsets_full"}:
            calibration.extend(_calibration_rows(method, selections, evaluation_labels))
        write_jsonl(args.output_dir / "per_query" / f"{method}.jsonl", per_query)
    ablation_rows = [
        {
            "method": method,
            "accuracy": _accuracy(selections, evaluation_labels),
            "samples": len(selections),
        }
        for method, selections in sorted(ablation_predictions.items())
    ]
    sanity_accuracy = _accuracy(sanity_selections, evaluation_labels)

    paired: list[dict[str, Any]] = [
        paired_selection_comparison(
            "deepsets_full_vs_none",
            prediction_groups["deepsets_full"],
            prediction_groups["deepsets_none"],
            evaluation_labels,
            seed=args.seed,
            bootstrap_samples=int(config["bootstrap_samples"]),
        )
    ]
    for family in family_names:
        paired.append(
            paired_selection_comparison(
                f"remove_family_{family}__full_vs_none",
                prediction_groups[f"deepsets_full__remove_family__{family}"],
                prediction_groups[f"deepsets_none__remove_family__{family}"],
                evaluation_labels,
                seed=args.seed,
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
        )
    fixed_reference = ablation_predictions["fixed_ablation__none"]
    for method, selections in sorted(ablation_predictions.items()):
        if method == "fixed_ablation__none":
            continue
        paired.append(
            paired_selection_comparison(
                f"{method}_vs_fixed_none",
                selections,
                fixed_reference,
                evaluation_labels,
                seed=args.seed,
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
        )
    corrections = holm_adjust({row["comparison"]: float(row["exact_mcnemar_p"]) for row in paired})
    for row in paired:
        row["holm"] = corrections[row["comparison"]]

    full_scores = probability_groups["deepsets_full"]
    clone_scores = probability_groups["cpi_full__exact_clone"]
    permutation_scores = probability_groups["cpi_full__permutation"]
    clone_sensitivity = max_probability_difference(
        [full_scores[qid] for qid in sorted(full_scores)], [clone_scores[qid] for qid in sorted(full_scores)]
    )
    permutation_sensitivity = max_probability_difference(
        [full_scores[qid] for qid in sorted(full_scores)], [permutation_scores[qid] for qid in sorted(full_scores)]
    )
    family_deltas = {
        family: _accuracy(prediction_groups[f"deepsets_full__remove_family__{family}"], evaluation_labels)
        - _accuracy(prediction_groups[f"deepsets_none__remove_family__{family}"], evaluation_labels)
        for family in family_names
    }
    full_accuracy = _accuracy(prediction_groups["deepsets_full"], evaluation_labels)
    none_accuracy = _accuracy(prediction_groups["deepsets_none"], evaluation_labels)
    seed_gate = {
        "seed": args.seed,
        "full_accuracy": full_accuracy,
        "no_intervention_accuracy": none_accuracy,
        "delta": full_accuracy - none_accuracy,
        "worst_leave_family_delta": min(family_deltas.values()),
        "leave_family_deltas": family_deltas,
        "clone_probability_sensitivity": clone_sensitivity,
        "permutation_probability_sensitivity": permutation_sensitivity,
        "required_delta": 0.0025,
        "required_worst_leave_family_delta": -0.005,
        "clone_tolerance": float(config["clone_tolerance"]),
    }
    direct = next(row for row in paired if row["comparison"] == "deepsets_full_vs_none")
    seed_gate["paired_bootstrap_delta_ci95"] = direct["paired_bootstrap_delta_ci95"]
    seed_gate["exact_mcnemar_p"] = direct["exact_mcnemar_p"]
    seed_gate["holm"] = direct["holm"]
    seed_gate["decision"] = (
        "GO"
        if seed_gate["delta"] >= seed_gate["required_delta"]
        and seed_gate["worst_leave_family_delta"] >= seed_gate["required_worst_leave_family_delta"]
        and seed_gate["clone_probability_sensitivity"] < seed_gate["clone_tolerance"]
        and seed_gate["permutation_probability_sensitivity"] < seed_gate["clone_tolerance"]
        else "NO-GO"
    )

    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    resource_usage = {
        **device_manifest,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "runtime_seconds": runtime,
        "finished_unix": time.time(),
    }
    write_json(args.output_dir / "summary.json", summaries)
    write_csv(
        args.output_dir / "summary.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in summaries],
    )
    write_csv(args.output_dir / "fixed_ablation.csv", ablation_rows)
    write_csv(args.output_dir / "calibration.csv", calibration)
    write_csv(args.output_dir / "training_history.csv", training_rows)
    write_json(args.output_dir / "paired_comparisons.json", paired)
    write_csv(
        args.output_dir / "paired_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in paired],
    )
    write_json(
        args.output_dir / "overfit_sanity.json",
        {
            "samples": len(sanity_ids),
            "accuracy": sanity_accuracy,
            "initial_loss": sanity_history[0],
            "final_loss": sanity_history[-1],
        },
    )
    write_json(args.output_dir / "seed_gate.json", seed_gate)
    write_json(args.output_dir / "resource_usage.json", resource_usage)
    completed_artifacts = files_manifest(
        [
            args.output_dir / "prediction_manifest.json",
            args.output_dir / "predictions",
            args.output_dir / "per_query",
            args.output_dir / "summary.json",
            args.output_dir / "summary.csv",
            args.output_dir / "fixed_ablation.csv",
            args.output_dir / "calibration.csv",
            args.output_dir / "training_history.csv",
            args.output_dir / "paired_comparisons.json",
            args.output_dir / "paired_comparisons.csv",
            args.output_dir / "overfit_sanity.json",
            args.output_dir / "seed_gate.json",
            args.output_dir / "resource_usage.json",
        ]
    )
    write_json(
        args.output_dir / "run_complete_manifest.json",
        {
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "prediction_manifest_sha256": sha256_file(args.output_dir / "prediction_manifest.json"),
            "artifact_hashes": completed_artifacts,
            "determinism": {
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "gate": seed_gate, "resources": resource_usage}, indent=2))


if __name__ == "__main__":
    main()
