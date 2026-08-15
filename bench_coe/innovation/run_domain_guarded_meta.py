from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    read_selections,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .data import CacheAdapter, EvaluationLabelAdapter, load_family_map
from .domain_shift_gate import (
    DOMAIN_FEATURES,
    DomainPolicy,
    apply_domain_policy,
    distribution_distances,
    domain_feature_matrix,
    environment_distribution_distances,
    guarded_method_name,
    nearest_source_distance,
)
from .goal_guardrails import evaluate_acceptance_contract, validate_acceptance_contract
from .run_conservative_meta_optimization import (
    _aggregate_comparison,
    _correctness_for_labels,
    _read_authenticated_selections,
    _registered_best,
    _seed_dir,
)
from .schema import EvaluationLabels, Selection, SourceTrainingLabels


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Apply source-calibrated, target-label-free domain guards to frozen conservative "
            "meta-selectors and evaluate only after every guarded prediction is hashed"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-source-questions", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-target-questions", type=int)
    parser.add_argument("--base-candidate-count", type=int)
    parser.add_argument("--max-policies", type=int)
    return parser.parse_args()


def _select_source_only_bases(frozen: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in frozen["finalists"]
        if float(row["source_delta"]) > 0.0 and float(row["switch_count_mean"]) > 0.0
    ]
    if not eligible:
        raise RuntimeError("The upstream source-only run has no improving frozen candidate")
    orderings = [
        sorted(
            eligible,
            key=lambda row: (
                -float(row["source_accuracy"]),
                -float(row["worst_environment_delta"]),
                str(row["method"]),
            ),
        ),
        sorted(
            eligible,
            key=lambda row: (
                -float(row["worst_environment_delta"]),
                -float(row["source_delta"]),
                str(row["method"]),
            ),
        ),
        sorted(
            eligible,
            key=lambda row: (
                -float(row["switch_precision"]),
                float(row["switch_rate"]),
                str(row["method"]),
            ),
        ),
        sorted(
            eligible,
            key=lambda row: (
                float(row["switch_rate"]),
                -float(row["source_delta"]),
                str(row["method"]),
            ),
        ),
    ]
    selected: list[dict[str, Any]] = []
    signatures: set[str] = set()
    cursor = 0
    while len(selected) < count and any(cursor < len(rows) for rows in orderings):
        for rows in orderings:
            if cursor >= len(rows):
                continue
            row = rows[cursor]
            signature = str(row["prediction_signature"])
            if signature not in signatures:
                signatures.add(signature)
                selected.append(row)
                if len(selected) == count:
                    break
        cursor += 1
    return selected


def _policies(config: Mapping[str, Any]) -> list[DomainPolicy]:
    result = [DomainPolicy("same_dataset", "identity", 1.0)]
    for metric in config["dataset_metrics"]:
        for quantile in config["quantiles"]:
            result.append(DomainPolicy("dataset_distance", str(metric), float(quantile)))
    for quantile in config["query_quantiles"]:
        result.append(DomainPolicy("query_support", "nearest", float(quantile)))
    for quantile in config["combined_quantiles"]:
        result.append(DomainPolicy("combined", "combined_nearest", float(quantile)))
    ids = [value.policy_id for value in result]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Domain policy IDs are not unique")
    return result


def _source_adapter(config: Mapping[str, Any]) -> tuple[SourceTrainingLabels, dict[str, Any]]:
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source = source_config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        load_family_map(Path(source_config["family_map"])),
        [str(value) for value in source_config["experts"]],
        Path(source_config["dataset_registry"]),
        str(source_config["dataset_registry_sha256"]),
    )
    return adapter.load_source_labels(), source_config


def _load_source_predictions(
    config: Mapping[str, Any],
    upstream_manifest: Mapping[str, Any],
    base_methods: Sequence[str],
    limit: int | None,
    authenticated: dict[str, str],
) -> tuple[list[dict[str, list[Selection]]], list[Path]]:
    upstream_root = Path(config["upstream_run_root"])
    original_root = Path(config["source_run_root"])
    seeds = [int(value) for value in config["source_seeds"]]
    rows_by_seed: list[dict[str, list[Selection]]] = []
    manifests: list[Path] = []
    for seed in seeds:
        original_seed = _seed_dir(original_root, seed)
        original_manifest_path = original_seed / "prediction_manifest.json"
        manifests.append(original_manifest_path)
        original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        expected_reference = original_manifest["prediction_hashes_before_evaluation"]["fcrg_full"]
        reference_path = original_seed / "predictions" / "fcrg_full.jsonl"
        reference, actual = _read_authenticated_selections(reference_path, expected_reference)
        authenticated[str(reference_path)] = actual
        values: dict[str, list[Selection]] = {
            "fcrg_full": sorted(reference, key=lambda row: row.question_id)
        }
        for method in base_methods:
            expected = upstream_manifest["predictions"]["source"][str(seed)][method]
            path = (
                upstream_root
                / "predictions"
                / "source_loso"
                / f"seed_{seed}"
                / f"{method}.jsonl"
            )
            selections, actual = _read_authenticated_selections(path, expected)
            authenticated[str(path)] = actual
            values[method] = sorted(selections, key=lambda row: row.question_id)
        if limit is not None:
            values = {method: selections[:limit] for method, selections in values.items()}
        expected_ids = {selection.question_id for selection in values["fcrg_full"]}
        for method, selections in values.items():
            if {selection.question_id for selection in selections} != expected_ids:
                raise RuntimeError(f"Source guarded input IDs differ for {method}")
        rows_by_seed.append(values)
    return rows_by_seed, manifests


def _source_policy_masks(
    policies: Sequence[DomainPolicy],
    source_features: np.ndarray,
    environment_index: np.ndarray,
    calibration: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    environment_distances = calibration["environment_distance_by_index"]
    query_distance = np.asarray(calibration["source_query_distance"], dtype=float)
    result: dict[str, np.ndarray] = {}
    for policy in policies:
        if policy.kind == "same_dataset":
            active = np.ones(len(source_features), dtype=bool)
        elif policy.kind == "dataset_distance":
            threshold = float(calibration["dataset_thresholds"][policy.metric][str(policy.quantile)])
            active = np.asarray(
                [environment_distances[str(int(env))][policy.metric] <= threshold for env in environment_index],
                dtype=bool,
            )
        elif policy.kind == "query_support":
            threshold = float(calibration["query_thresholds"][str(policy.quantile)])
            active = query_distance <= threshold
        elif policy.kind == "combined":
            distribution_threshold = float(
                calibration["dataset_thresholds"]["combined"][str(policy.quantile)]
            )
            query_threshold = float(calibration["query_thresholds"][str(policy.quantile)])
            active = np.asarray(
                [
                    environment_distances[str(int(env))]["combined"] <= distribution_threshold
                    for env in environment_index
                ],
                dtype=bool,
            ) & (query_distance <= query_threshold)
        else:
            raise ValueError(f"Unknown domain policy kind: {policy.kind}")
        result[policy.policy_id] = active
    return result


def _calibration(
    source_features: np.ndarray,
    environment_index: np.ndarray,
    policies_config: Mapping[str, Any],
) -> dict[str, Any]:
    calibrated = environment_distribution_distances(source_features, environment_index)
    environment_rows: dict[str, dict[str, float]] = {}
    for environment in sorted(set(int(value) for value in environment_index)):
        heldout = environment_index == environment
        environment_rows[str(environment)] = distribution_distances(
            source_features[~heldout], source_features[heldout]
        )
    quantiles = sorted(
        {
            float(value)
            for value in (
                list(policies_config["quantiles"])
                + list(policies_config["combined_quantiles"])
            )
        }
    )
    dataset_thresholds = {
        metric: {
            str(quantile): float(np.quantile(values, quantile)) for quantile in quantiles
        }
        for metric, values in calibrated.items()
    }
    query_distance = nearest_source_distance(
        source_features,
        source_features,
        source_environment=environment_index,
    )
    query_quantiles = sorted(
        {
            float(value)
            for value in (
                list(policies_config["query_quantiles"])
                + list(policies_config["combined_quantiles"])
            )
        }
    )
    query_thresholds = {
        str(quantile): float(np.quantile(query_distance, quantile))
        for quantile in query_quantiles
    }
    return {
        "feature_names": list(DOMAIN_FEATURES),
        "dataset_calibration_distances": {
            metric: values.tolist() for metric, values in calibrated.items()
        },
        "dataset_thresholds": dataset_thresholds,
        "environment_distance_by_index": environment_rows,
        "source_query_distance": query_distance.tolist(),
        "query_thresholds": query_thresholds,
    }


def _target_policy_mask(
    policy: DomainPolicy,
    source_dataset: str,
    target_dataset: str,
    target_features: np.ndarray,
    source_features: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    distances = distribution_distances(source_features, target_features)
    nearest = nearest_source_distance(source_features, target_features)
    diagnostics: dict[str, Any] = {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "distribution_distances": distances,
        "target_query_nearest_source_distance_mean": float(nearest.mean()),
        "target_query_nearest_source_distance_max": float(nearest.max()),
    }
    if policy.kind == "same_dataset":
        enabled = source_dataset == target_dataset
        active = np.full(len(target_features), enabled, dtype=bool)
        diagnostics.update({"same_dataset": enabled, "threshold": None})
    elif policy.kind == "dataset_distance":
        threshold = float(calibration["dataset_thresholds"][policy.metric][str(policy.quantile)])
        enabled = distances[policy.metric] <= threshold
        active = np.full(len(target_features), enabled, dtype=bool)
        diagnostics.update(
            {"metric": policy.metric, "distance": distances[policy.metric], "threshold": threshold}
        )
    elif policy.kind == "query_support":
        threshold = float(calibration["query_thresholds"][str(policy.quantile)])
        active = nearest <= threshold
        diagnostics.update({"metric": "nearest", "threshold": threshold})
    elif policy.kind == "combined":
        distribution_threshold = float(
            calibration["dataset_thresholds"]["combined"][str(policy.quantile)]
        )
        query_threshold = float(calibration["query_thresholds"][str(policy.quantile)])
        active = (distances["combined"] <= distribution_threshold) & (nearest <= query_threshold)
        diagnostics.update(
            {
                "metric": "combined_nearest",
                "distribution_threshold": distribution_threshold,
                "query_threshold": query_threshold,
            }
        )
    else:
        raise ValueError(f"Unknown domain policy kind: {policy.kind}")
    diagnostics["active_fraction"] = float(active.mean())
    return active, diagnostics


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    acceptance = validate_acceptance_contract(config["acceptance"])
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)

    upstream_root = Path(config["upstream_run_root"])
    upstream_prediction_path = upstream_root / "prediction_manifest.json"
    upstream_complete = json.loads(
        (upstream_root / "complete_manifest.json").read_text(encoding="utf-8")
    )
    upstream_prediction_hash = sha256_file(upstream_prediction_path)
    if upstream_prediction_hash != upstream_complete["prediction_manifest_sha256_before_labels"]:
        raise RuntimeError("Upstream prediction manifest does not match its completion binding")
    upstream_manifest = json.loads(upstream_prediction_path.read_text(encoding="utf-8"))
    frozen_path = upstream_root / "frozen_finalists.json"
    if sha256_file(frozen_path) != upstream_manifest["frozen_finalists_sha256"]:
        raise RuntimeError("Upstream frozen finalists do not match the pre-label manifest")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    base_count = int(
        args.base_candidate_count
        if args.base_candidate_count is not None
        else config["base_candidate_count"]
    )
    selected_bases = _select_source_only_bases(frozen, base_count)
    base_methods = [str(row["method"]) for row in selected_bases]
    write_json(
        args.output_dir / "frozen_base_candidates.json",
        {
            "selection_scope": "upstream_source_oof_only",
            "target_results_read": False,
            "base_candidate_count": len(selected_bases),
            "base_candidates": selected_bases,
        },
    )

    policies = _policies(config["domain_policies"])
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    authenticated: dict[str, str] = {}
    source_by_seed, manifest_paths = _load_source_predictions(
        config,
        upstream_manifest,
        base_methods,
        args.max_source_questions,
        authenticated,
    )
    source_labels_all, source_config = _source_adapter(config)
    source_ids = tuple(selection.question_id for selection in source_by_seed[0]["fcrg_full"])
    source_labels = source_labels_all.subset(source_ids)
    environments = sorted({source_labels.environment_by_question[qid] for qid in source_ids})
    environment_lookup = {value: index for index, value in enumerate(environments)}
    environment_index = np.asarray(
        [environment_lookup[source_labels.environment_by_question[qid]] for qid in source_ids],
        dtype=np.int32,
    )
    source_features = domain_feature_matrix(source_by_seed[0]["fcrg_full"])
    calibration = _calibration(source_features, environment_index, config["domain_policies"])
    source_masks = _source_policy_masks(
        policies,
        source_features,
        environment_index,
        calibration,
    )
    write_json(args.output_dir / "source_domain_calibration.json", calibration)

    prediction_hashes: dict[str, Any] = {"source": {}, "targets": {}}
    source_seeds = [int(value) for value in config["source_seeds"]]
    guarded_methods: dict[tuple[str, str], str] = {}
    for seed_index, seed in enumerate(source_seeds):
        reference = source_by_seed[seed_index]["fcrg_full"]
        for base_method in base_methods:
            base = source_by_seed[seed_index][base_method]
            for policy in policies:
                method = guarded_method_name(base_method, policy)
                guarded_methods[(base_method, policy.policy_id)] = method
                selections = apply_domain_policy(
                    base,
                    reference,
                    source_masks[policy.policy_id],
                    policy,
                    base_method=base_method,
                    domain_diagnostics={
                        "scope": "source_oof",
                        "active_fraction": float(source_masks[policy.policy_id].mean()),
                    },
                )
                path = (
                    args.output_dir
                    / "predictions"
                    / "source_loso"
                    / f"seed_{seed}"
                    / f"{method}.jsonl"
                )
                prediction_hashes["source"].setdefault(str(seed), {})[method] = write_selections(
                    path, selections
                )

    target_jobs: list[dict[str, Any]] = []
    for raw_panel in config["target_panels"]:
        panel_config_path = Path(raw_panel["config"])
        panel_config = yaml.safe_load(panel_config_path.read_text(encoding="utf-8"))
        panel_name = str(raw_panel.get("name", panel_config_path.stem))
        for target in panel_config["targets"]:
            target_jobs.append(
                {
                    "panel_name": panel_name,
                    "panel_config_path": panel_config_path,
                    "panel_config": panel_config,
                    "panel_root": Path(raw_panel["run_root"]),
                    "target": target,
                    "seeds": [int(value) for value in panel_config["seeds"]],
                }
            )
    if args.max_targets is not None:
        target_jobs = target_jobs[: args.max_targets]

    target_runtime: dict[tuple[str, int, str], dict[str, Any]] = {}
    gate_rows: list[dict[str, Any]] = []
    source_dataset = str(source_config["source"]["dataset"])
    for job in target_jobs:
        target = job["target"]
        target_name = str(target["name"])
        panel_name = str(job["panel_name"])
        prediction_hashes["targets"].setdefault(target_name, {})
        for seed in job["seeds"]:
            original_seed = _seed_dir(job["panel_root"], seed)
            original_manifest_path = original_seed / "prediction_manifest.json"
            manifest_paths.append(original_manifest_path)
            original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
            target_manifest = original_manifest["targets"][target_name]
            reference_relative = target_manifest["prediction_paths"]["fcrg_full"]
            reference_expected = target_manifest["prediction_hashes_before_evaluation"]["fcrg_full"]
            reference_path = original_seed / reference_relative
            reference, actual = _read_authenticated_selections(reference_path, reference_expected)
            authenticated[str(reference_path)] = actual
            reference = sorted(reference, key=lambda row: row.question_id)
            if args.max_target_questions is not None:
                reference = reference[: args.max_target_questions]
            upstream_entries = upstream_manifest["predictions"]["targets"][target_name][str(seed)]
            base_values: dict[str, list[Selection]] = {}
            for base_method in base_methods:
                entry = upstream_entries[base_method]
                path = upstream_root / entry["path"]
                values, actual = _read_authenticated_selections(path, entry["sha256"])
                authenticated[str(path)] = actual
                values = sorted(values, key=lambda row: row.question_id)
                if args.max_target_questions is not None:
                    values = values[: args.max_target_questions]
                base_values[base_method] = values
            target_features = domain_feature_matrix(reference)
            masks: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
            for policy in policies:
                masks[policy.policy_id] = _target_policy_mask(
                    policy,
                    source_dataset,
                    str(target["dataset"]),
                    target_features,
                    source_features,
                    calibration,
                )
                active, diagnostics = masks[policy.policy_id]
                gate_rows.append(
                    {
                        "target": target_name,
                        "seed": seed,
                        "policy": policy.policy_id,
                        "active_questions": int(active.sum()),
                        "questions": len(active),
                        "active_fraction": float(active.mean()),
                        **diagnostics,
                    }
                )
            for base_method, base in base_values.items():
                for policy in policies:
                    active, diagnostics = masks[policy.policy_id]
                    method = guarded_methods[(base_method, policy.policy_id)]
                    selections = apply_domain_policy(
                        base,
                        reference,
                        active,
                        policy,
                        base_method=base_method,
                        domain_diagnostics=diagnostics,
                    )
                    relative = (
                        Path("predictions")
                        / panel_name
                        / f"seed_{seed}"
                        / target_name
                        / f"{method}.jsonl"
                    )
                    digest = write_selections(args.output_dir / relative, selections)
                    prediction_hashes["targets"][target_name].setdefault(str(seed), {})[
                        method
                    ] = {"path": str(relative), "sha256": digest}
            target_runtime[(panel_name, seed, target_name)] = {
                "original_seed": str(original_seed),
                "target_manifest": target_manifest,
            }
    write_json(args.output_dir / "target_domain_diagnostics.json", gate_rows)

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [
            args.config,
            Path(config["source_config"]),
            upstream_prediction_path,
            frozen_path,
            *[Path(panel["config"]) for panel in config["target_panels"]],
            *manifest_paths,
        ],
    )
    environment["authenticated_input_prediction_hashes"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": "development_ood_diagnostic_only",
        "upstream_prediction_manifest_sha256": upstream_prediction_hash,
        "frozen_base_candidates_sha256": sha256_file(
            args.output_dir / "frozen_base_candidates.json"
        ),
        "source_domain_calibration_sha256": sha256_file(
            args.output_dir / "source_domain_calibration.json"
        ),
        "guarded_methods": {
            f"{base}|{policy}": method
            for (base, policy), method in sorted(guarded_methods.items())
        },
        "predictions": prediction_hashes,
        "labels_opened": False,
        "written_before_any_target_label_adapter": True,
        "innovation_code_manifest_sha256": environment["innovation_code_manifest_sha256"],
    }
    write_json(args.output_dir / "prediction_manifest.json", prediction_manifest)
    prediction_manifest_hash = sha256_file(args.output_dir / "prediction_manifest.json")

    # Evaluation labels are opened only after all source and target policy predictions are hashed.
    aggregate_rows: list[dict[str, Any]] = []
    methods = sorted(guarded_methods.values())
    source_reference_by_seed: list[np.ndarray] = []
    source_candidate: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed_index, seed in enumerate(source_seeds):
        reference = source_by_seed[seed_index]["fcrg_full"]
        reference_correct = np.asarray(
            [
                bool(
                    row.selected_expert_id is not None
                    and source_labels.get(row.question_id, row.selected_expert_id)
                )
                for row in reference
            ],
            dtype=bool,
        )
        source_reference_by_seed.append(reference_correct)
        for method in methods:
            path = (
                args.output_dir
                / "predictions"
                / "source_loso"
                / f"seed_{seed}"
                / f"{method}.jsonl"
            )
            values = read_selections(path)
            source_candidate[method].append(
                np.asarray(
                    [
                        bool(
                            row.selected_expert_id is not None
                            and source_labels.get(row.question_id, row.selected_expert_id)
                        )
                        for row in values
                    ],
                    dtype=bool,
                )
            )
    source_reference_matrix = np.stack(source_reference_by_seed, axis=0)
    for method in methods:
        aggregate_rows.append(
            _aggregate_comparison(
                method,
                "source_loso",
                np.stack(source_candidate[method], axis=0),
                source_reference_matrix,
            )
        )

    for job in target_jobs:
        target = job["target"]
        target_name = str(target["name"])
        panel_name = str(job["panel_name"])
        labels: EvaluationLabels = EvaluationLabelAdapter.from_registry(
            Path(target["label_cache_path"]),
            str(target["dataset"]),
            str(target["split"]),
            str(target["modality"]),
            [str(value) for value in job["panel_config"]["experts"]],
            Path(job["panel_config"]["dataset_registry"]),
            str(job["panel_config"]["dataset_registry_sha256"]),
        ).load(limit=args.max_target_questions)
        reference_rows: list[np.ndarray] = []
        candidate_rows: dict[str, list[np.ndarray]] = defaultdict(list)
        for seed in job["seeds"]:
            runtime = target_runtime[(panel_name, seed, target_name)]
            original_seed = Path(runtime["original_seed"])
            target_manifest = runtime["target_manifest"]
            relative = target_manifest["prediction_paths"]["fcrg_full"]
            expected = target_manifest["prediction_hashes_before_evaluation"]["fcrg_full"]
            reference, _ = _read_authenticated_selections(original_seed / relative, expected)
            reference = sorted(reference, key=lambda row: row.question_id)
            if args.max_target_questions is not None:
                reference = reference[: args.max_target_questions]
            reference_rows.append(_correctness_for_labels(reference, labels))
            for method in methods:
                entry = prediction_hashes["targets"][target_name][str(seed)][method]
                path = args.output_dir / entry["path"]
                if sha256_file(path) != entry["sha256"]:
                    raise RuntimeError(f"Guarded prediction changed before evaluation: {path}")
                candidate_rows[method].append(
                    _correctness_for_labels(read_selections(path), labels)
                )
        reference_matrix = np.stack(reference_rows, axis=0)
        for method in methods:
            aggregate_rows.append(
                _aggregate_comparison(
                    method,
                    target_name,
                    np.stack(candidate_rows[method], axis=0),
                    reference_matrix,
                )
            )

    registered = _registered_best(config)
    for row in aggregate_rows:
        best = registered.get(str(row["target"]))
        row["registered_best_accuracy"] = best
        row["delta_vs_registered_best"] = (
            float(row["accuracy_mean"]) - best if best is not None else None
        )
    write_json(args.output_dir / "aggregate_summary.json", aggregate_rows)
    write_csv(
        args.output_dir / "aggregate_summary.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in aggregate_rows
        ],
    )

    method_metadata = {
        method: {"base_method": base, "policy": policy}
        for (base, policy), method in guarded_methods.items()
    }
    goal_rows: list[dict[str, Any]] = []
    for method in methods:
        by_target = {
            str(row["target"]): row for row in aggregate_rows if row["method"] == method
        }
        goal = evaluate_acceptance_contract(by_target, acceptance)
        policy_id = method_metadata[method]["policy"]
        kind = policy_id.split("__", 1)[0]
        goal_rows.append(
            {
                "method": method,
                **method_metadata[method],
                "policy_kind": kind,
                **goal,
                "mmmu_pro_test_delta_vs_fcrg_full": (
                    goal["primary_target_delta_vs_fcrg_full"]
                    if goal["primary_target"] == "mmmu_pro_test_id"
                    else None
                ),
                "other_datasets_nonnegative_vs_fcrg_full": goal[
                    "relative_non_regression_met"
                ],
                "universal_label_free_guard": kind != "same_dataset",
                "worst_delta_vs_fcrg_full": min(
                    goal["all_dataset_delta_vs_fcrg_full"].values()
                ),
            }
        )
    write_json(args.output_dir / "nonregression_matrix.json", goal_rows)
    write_csv(
        args.output_dir / "nonregression_matrix.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, dict)}
            for row in goal_rows
        ],
    )
    strict = [row for row in goal_rows if row["strict_user_goal_met"]]
    universal = [row for row in strict if row["universal_label_free_guard"]]
    decision = {
        "scope": "development_ood_diagnostic_only",
        "source_only_bases_frozen_before_target_evaluation": True,
        "target_label_free_domain_gates_frozen_before_target_evaluation": True,
        "prediction_manifest_sha256_before_labels": prediction_manifest_hash,
        "base_candidate_count": len(base_methods),
        "domain_policy_count": len(policies),
        "guarded_method_count": len(methods),
        "acceptance_contract": acceptance,
        "absolute_accuracy_floors_enforced": True,
        "strict_goal_met_count": len(strict),
        "strict_goal_met_methods": [row["method"] for row in strict],
        "universal_guard_goal_met_count": len(universal),
        "universal_guard_goal_met_methods": [row["method"] for row in universal],
        "dataset_scoped_policy_is_not_universal_generalization": True,
        "default_selected_from_target_results": False,
        "can_authorize_locked_test": False,
    }
    write_json(args.output_dir / "decision.json", decision)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "prediction_manifest_sha256": prediction_manifest_hash,
            "labels_opened_after_prediction_manifest": True,
            "targets": [str(job["target"]["name"]) for job in target_jobs],
        },
    )
    completion_paths = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "complete_manifest.json"
    ]
    write_json(
        args.output_dir / "complete_manifest.json",
        {
            "runtime_seconds": time.time() - started,
            "prediction_manifest_sha256_before_labels": prediction_manifest_hash,
            "artifact_hashes": files_manifest(completion_paths),
            "decision": decision,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), **decision}, indent=2))


if __name__ == "__main__":
    main()
