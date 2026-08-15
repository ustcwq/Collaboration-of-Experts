from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    write_csv,
    write_json,
    write_jsonl,
    write_selections,
    validate_test_receipt,
)
from .data import CacheAdapter, EvaluationLabelAdapter, load_family_map
from .evaluation import evaluate, holm_adjust, paired_selection_comparison, selection_correctness
from .prior_art_targets import project_observable_pool, target_environment_by_question
from .repair_simplification import subset_expert_pool
from .response_embeddings import MiniLMResponseEncoder
from .run_prior_art_overlap import configure_device, method_cost, retag, run_fold_pool
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one authenticated cross-dataset Improve5/6 prior-art seed"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-inner-environments", type=int)
    return parser.parse_args()


def _load_target(
    target: dict[str, Any],
    family_map: dict[str, str],
    experts: list[str],
    limit: int | None,
) -> ObservableQueryBatch:
    adapter = CacheAdapter.from_target_observables(
        Path(target["observable_cache_path"]),
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        family_map,
        experts,
        str(target["observable_manifest_sha256"]),
    )
    batch = adapter.load_observables(limit=limit)
    if limit is None and len(batch.question_ids) != int(target["expected_questions"]):
        raise RuntimeError(f"Target question count differs from protocol: {target['name']}")
    return batch


def _pool_batches(
    source_batch: ObservableQueryBatch,
    source_labels: SourceTrainingLabels,
    target_batch: ObservableQueryBatch,
    config: dict[str, Any],
) -> tuple[
    dict[str, tuple[ObservableQueryBatch, SourceTrainingLabels]],
    dict[str, ObservableQueryBatch],
]:
    source_pools = {"full_pool": (source_batch, source_labels)}
    target_pools = {"full_pool": target_batch}
    for pool_name, raw_experts in sorted(config["pool_shift"]["pools"].items()):
        expert_ids = tuple(str(value) for value in raw_experts)
        source_pools[str(pool_name)] = subset_expert_pool(source_batch, source_labels, expert_ids)
        target_pools[str(pool_name)] = project_observable_pool(target_batch, expert_ids)
    return source_pools, target_pools


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    expected_gpu = dict(
        zip(
            [int(value) for value in config["seeds"]],
            [int(value) for value in config["run_physical_gpus"]],
            strict=True,
        )
    )
    if expected_gpu.get(args.seed) != args.physical_gpu:
        raise ValueError("Seed and physical GPU do not match the frozen target mapping")
    device, device_manifest = configure_device(args.seed, args.physical_gpu)
    encoder_config = config["response_embedding"]
    response_encoder = MiniLMResponseEncoder(
        str(encoder_config["model_id"]),
        device,
        batch_size=int(encoder_config.get("batch_size", 256)),
        max_length=int(encoder_config.get("max_length", 128)),
    )

    family_map_path = Path(config["family_map"])
    registry_path = Path(config["dataset_registry"])
    family_map = load_family_map(family_map_path)
    experts = [str(value) for value in config["experts"]]
    source = config["source"]
    source_adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        experts,
        registry_path,
        str(config["dataset_registry_sha256"]),
    )
    source_batch = source_adapter.load_observables()
    source_labels = source_adapter.load_source_labels()
    if len(source_batch.question_ids) != int(config["expected_source_questions"]):
        raise RuntimeError("Source question count differs from target protocol")

    targets = list(config["targets"])
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if not targets:
        raise ValueError("No cross-dataset targets are configured")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)

    predictions_by_target: dict[str, dict[str, list[Selection]]] = {}
    groups_by_target: dict[str, dict[str, str]] = {}
    target_batches: dict[str, dict[str, ObservableQueryBatch]] = {}
    diagnostics: list[dict[str, Any]] = []
    pool_filter = set(str(value) for value in config["pool_shift"]["methods"])
    for target_index, target in enumerate(targets):
        target_name = str(target["name"])
        target_batch = _load_target(target, family_map, experts, args.max_questions)
        source_pools, projected_targets = _pool_batches(
            source_batch, source_labels, target_batch, config
        )
        target_batches[target_name] = projected_targets
        predictions: dict[str, list[Selection]] = {}
        groups: dict[str, str] = {}
        for pool_name, (pool_source, pool_labels) in sorted(source_pools.items()):
            full_protocol = pool_name == "full_pool"
            pool_predictions, pool_diagnostics = run_fold_pool(
                pool_source,
                pool_labels,
                projected_targets[pool_name],
                config=config,
                seed=args.seed,
                fold_index=target_index,
                device=device,
                full_protocol=full_protocol,
                max_inner_environments=args.max_inner_environments,
                response_encoder=response_encoder,
            )
            if not full_protocol:
                pool_predictions = {
                    method: values
                    for method, values in pool_predictions.items()
                    if method in pool_filter
                }
            for method, values in pool_predictions.items():
                stored = method if full_protocol else f"{pool_name}__{method}"
                predictions[stored] = retag(values, stored)
                groups[stored] = pool_name
            diagnostics.append(
                {
                    "target": target_name,
                    "pool": pool_name,
                    "diagnostics": pool_diagnostics,
                }
            )
        expected_methods = int(config["expected_full_pool_methods"]) + len(
            config["pool_shift"]["pools"]
        ) * len(pool_filter)
        if len(predictions) != expected_methods:
            raise RuntimeError(
                f"Target method count differs from protocol: target={target_name}, "
                f"observed={len(predictions)}, expected={expected_methods}"
            )
        expected_ids = set(target_batch.question_ids)
        for method, values in predictions.items():
            values.sort(key=lambda selection: selection.question_id)
            if {selection.question_id for selection in values} != expected_ids:
                raise RuntimeError(f"Incomplete target predictions: {target_name}/{method}")
        predictions_by_target[target_name] = predictions
        groups_by_target[target_name] = groups

    write_jsonl(args.output_dir / "target_diagnostics.jsonl", diagnostics)
    target_manifests: dict[str, dict[str, Any]] = {}
    for target in targets:
        target_name = str(target["name"])
        hashes: dict[str, str] = {}
        paths: dict[str, str] = {}
        for method, selections in sorted(predictions_by_target[target_name].items()):
            relative = Path("predictions") / target_name / f"{method}.jsonl"
            hashes[method] = write_selections(args.output_dir / relative, selections)
            paths[method] = str(relative)
        question_ids = target_batches[target_name]["full_pool"].question_ids
        target_manifests[target_name] = {
            "dataset": str(target["dataset"]),
            "split": str(target["split"]),
            "questions": len(question_ids),
            "question_ids_sha256": hashlib.sha256(
                "\n".join(question_ids).encode("utf-8")
            ).hexdigest(),
            "method_count": len(hashes),
            "method_group": groups_by_target[target_name],
            "prediction_paths": paths,
            "prediction_hashes_before_evaluation": hashes,
        }

    observable_inputs = [Path(target["observable_cache_path"]) for target in targets]
    run_environment = environment_manifest(
        sys.argv,
        args.seed,
        [
            args.config,
            family_map_path,
            registry_path,
            Path(source["cache_path"]),
            response_encoder.snapshot,
            *observable_inputs,
        ],
    )
    write_json(args.output_dir / "environment.json", run_environment)
    prediction_manifest = {
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "scope": "development_ood_diagnostic_only",
        "targets": target_manifests,
        "input_manifest_sha256": run_environment["input_manifest_sha256"],
        "innovation_code_manifest_sha256": run_environment[
            "innovation_code_manifest_sha256"
        ],
        "response_embedding": response_encoder.diagnostics(),
        "labels_opened": False,
    }
    write_json(args.output_dir / "prediction_manifest.json", prediction_manifest)

    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        target_name = str(target["name"])
        # No target-label adapter is constructed until every configured target prediction is hashed.
        labels = EvaluationLabelAdapter.from_registry(
            Path(target["label_cache_path"]),
            str(target["dataset"]),
            str(target["split"]),
            str(target["modality"]),
            experts,
            registry_path,
            str(config["dataset_registry_sha256"]),
        ).load(limit=args.max_questions)
        environments = target_environment_by_question(target_batches[target_name]["full_pool"])
        methods = predictions_by_target[target_name]
        for method_index, (method, selections) in enumerate(sorted(methods.items())):
            pool = groups_by_target[target_name][method]
            reference_name = (
                "global_best_posthoc"
                if pool == "full_pool"
                else f"{pool}__global_best_posthoc"
            )
            reference = methods[reference_name]
            pool_batch = target_batches[target_name][pool]
            summary, per_query = evaluate(
                method,
                selections,
                reference,
                pool_batch,
                labels,
                bootstrap_samples=int(config["seed_bootstrap_samples"]),
                seed=args.seed + 1000 * target_index + method_index,
            )
            summary.update({"target": target_name, "pool": pool})
            summaries.append(summary)
            comparison = paired_selection_comparison(
                f"{target_name}__{method}_vs_{reference_name}",
                selections,
                reference,
                labels,
                seed=args.seed + 100000 + 1000 * target_index + method_index,
                bootstrap_samples=int(config["seed_bootstrap_samples"]),
            )
            comparison["target"] = target_name
            comparison["pool"] = pool
            paired_rows.append(comparison)
            for row in per_query:
                query_rows.append({"target": target_name, "method": method, "pool": pool, **row})
            by_environment: dict[str, list[bool]] = defaultdict(list)
            for question_id, correct in selection_correctness(selections, labels).items():
                by_environment[environments[question_id]].append(correct)
            for environment, values in sorted(by_environment.items()):
                environment_rows.append(
                    {
                        "target": target_name,
                        "environment": environment,
                        "method": method,
                        "pool": pool,
                        "samples": len(values),
                        "accuracy": float(np.mean(values)),
                    }
                )
            cost = method_cost(method, selections, pool_batch)
            cost["target"] = target_name
            cost["pool"] = pool
            cost_rows.append(cost)

    corrections = holm_adjust(
        {str(row["comparison"]): float(row["exact_mcnemar_p"]) for row in paired_rows}
    )
    for row in paired_rows:
        row["holm"] = corrections[str(row["comparison"])]
    write_json(args.output_dir / "summary.json", summaries)
    write_csv(
        args.output_dir / "summary.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in summaries],
    )
    write_json(args.output_dir / "paired_comparisons.json", paired_rows)
    write_csv(
        args.output_dir / "paired_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in paired_rows],
    )
    write_csv(args.output_dir / "per_environment.csv", environment_rows)
    write_jsonl(args.output_dir / "evaluation_per_query.jsonl", query_rows)
    write_csv(args.output_dir / "inference_costs.csv", cost_rows)
    write_json(
        args.output_dir / "scope.json",
        {
            "scope": "development_ood_diagnostic_only",
            "overrides_source_gate": False,
            "can_authorize_locked_test": False,
            "targets": [str(target["name"]) for target in targets],
        },
    )

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
            "target_prediction_hashes_before_evaluation": {
                target: manifest["prediction_hashes_before_evaluation"]
                for target, manifest in target_manifests.items()
            },
            "artifact_hashes": files_manifest(completion_paths),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "targets": {name: len(values) for name, values in predictions_by_target.items()},
                "scope": "development_ood_diagnostic_only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
