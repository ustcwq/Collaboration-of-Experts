from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

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
from .evaluation import evaluate, holm_adjust, paired_selection_comparison
from .repair_simplification import (
    ABLATION_METHODS,
    GRAPH_MODE_BY_METHOD,
    POOL_SHIFT_METHODS,
    RepairComponents,
    fit_repair_components,
    selections_from_components,
    source_accuracy_of_selections,
    subset_expert_pool,
    with_graph_mode,
)
from .schema import EvaluationLabels, ObservableQueryBatch, Selection, SourceTrainingLabels
from .selectors import (
    FamilyBalancedVoteSelector,
    GlobalLocalSelector,
    LegacyDARESelector,
    LegacyRepairChainSelector,
    MajorityVoteSelector,
    OutputProfileKNNSelector,
    RandomSelector,
    SourceBestSelector,
    SourceWeightedVoteSelector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one authenticated Improve6 scoring-simplification seed")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--max-inner-environments", type=int)
    return parser.parse_args()


def _configure_device(seed: int, physical_gpu: int) -> tuple[torch.device, dict[str, Any]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("PYTHONHASHSEED must match the pre-registered run seed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The simplification run requires exactly one visible CUDA device")
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


def _environment_folds(labels: SourceTrainingLabels) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    by_environment: dict[str, list[str]] = defaultdict(list)
    for question_id, environment in labels.environment_by_question.items():
        by_environment[str(environment)].append(question_id)
    all_ids = set(labels.environment_by_question)
    return [
        (
            environment,
            tuple(sorted(all_ids.difference(test_ids))),
            tuple(sorted(test_ids)),
        )
        for environment, test_ids in sorted(by_environment.items())
    ]


def _nested_parameters(
    batch: ObservableQueryBatch,
    labels: SourceTrainingLabels,
    *,
    beta_grid: tuple[float, ...],
    alpha_grid: tuple[float, ...],
    neighbors: int,
    seed: int,
    device: torch.device,
    max_environments: int | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    folds = _environment_folds(labels)
    if max_environments is not None:
        folds = folds[:max_environments]
    scores: dict[str, dict[float, list[float]]] = {
        "m3_h1_support_beta": {value: [] for value in beta_grid},
        "m3_cluster_h1_support_beta": {value: [] for value in beta_grid},
        "m5_h1_h2_alpha": {value: [] for value in alpha_grid},
    }
    for fold_index, (_, train_ids, test_ids) in enumerate(folds):
        train_batch = batch.subset(train_ids)
        test_batch = batch.subset(test_ids)
        train_labels = labels.subset(train_ids)
        components = fit_repair_components(
            train_batch,
            train_labels,
            test_batch,
            neighbors=neighbors,
            seed=seed + fold_index,
            device=device,
        )
        for beta in beta_grid:
            expert = selections_from_components(test_batch, components, "m3_h1_support", beta=beta)
            cluster = selections_from_components(test_batch, components, "m3_cluster_h1_support", beta=beta)
            scores["m3_h1_support_beta"][beta].append(source_accuracy_of_selections(expert, labels))
            scores["m3_cluster_h1_support_beta"][beta].append(source_accuracy_of_selections(cluster, labels))
        for alpha in alpha_grid:
            predictions = selections_from_components(test_batch, components, "m5_h1_h2", alpha=alpha)
            scores["m5_h1_h2_alpha"][alpha].append(source_accuracy_of_selections(predictions, labels))

    selected: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for parameter, by_value in scores.items():
        macro_by_value = {
            value: float(np.mean(values)) if values else 0.0 for value, values in by_value.items()
        }
        # A tie resolves toward one-hop evidence: larger beta/alpha removes more auxiliary weight.
        chosen = max(macro_by_value, key=lambda value: (macro_by_value[value], value))
        selected[parameter] = float(chosen)
        rows.extend(
            {
                "parameter": parameter,
                "value": value,
                "inner_environment_count": len(by_value[value]),
                "macro_accuracy": macro_by_value[value],
                "selected": value == chosen,
            }
            for value in sorted(by_value)
        )
    return selected, rows


def _component_rows(
    environment: str,
    components: RepairComponents,
    parameters: dict[str, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row_index, question_id in enumerate(components.question_ids):
        experts = {
            expert: {
                "local": float(components.local[row_index, col]),
                "support": float(components.support[row_index, col]),
                "uncertainty": float(components.uncertainty[row_index, col]),
                "failure_weight": float(components.failure_weights[row_index, col]),
                "hop1": float(components.hop1[row_index, col]),
                "hop2": float(components.hop2[row_index, col]),
                "global_accuracy": float(components.global_accuracy[col]),
                "valid": bool(components.valid_mask[row_index, col]),
                "cluster_id": int(components.cluster_ids[row_index, col]),
            }
            for col, expert in enumerate(components.expert_ids)
        }
        result.append(
            {
                "heldout_environment": environment,
                "question_id": question_id,
                "parameters": parameters,
                "experts": experts,
            }
        )
    return result


def _accuracy(selections: list[Selection], labels: EvaluationLabels) -> float:
    values = [bool(labels.get(item.question_id, item.selected_expert_id or "")) for item in selections]
    return float(np.mean(values)) if values else 0.0


def _environment_gate(
    method: str,
    reference: str,
    environments: list[str],
    environment_accuracy: dict[tuple[str, str], float],
    direct: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    deltas = [environment_accuracy[(environment, method)] - environment_accuracy[(environment, reference)] for environment in environments]
    thresholds = config["simplification_gate"]
    result = {
        "method": method,
        "reference": reference,
        "micro_delta": float(direct["delta"]),
        "macro_environment_delta": float(np.mean(deltas)) if deltas else 0.0,
        "worst_environment_delta": float(min(deltas)) if deltas else 0.0,
        "nonnegative_environment_fraction": float(np.mean(np.asarray(deltas) >= 0.0)) if deltas else 0.0,
        "rescue_count": int(direct["rescue_count"]),
        "harm_count": int(direct["harm_count"]),
        "paired_bootstrap_delta_ci95": direct["paired_bootstrap_delta_ci95"],
        "required_micro_delta": float(thresholds["required_micro_delta"]),
        "required_worst_environment_delta": float(thresholds["required_worst_environment_delta"]),
        "required_nonnegative_environment_fraction": float(thresholds["required_nonnegative_environment_fraction"]),
        "ci_noninferiority_margin": float(thresholds["ci_noninferiority_margin"]),
    }
    result["decision"] = (
        "GO"
        if result["micro_delta"] >= result["required_micro_delta"]
        and result["worst_environment_delta"] >= result["required_worst_environment_delta"]
        and result["nonnegative_environment_fraction"] >= result["required_nonnegative_environment_fraction"]
        and float(result["paired_bootstrap_delta_ci95"][0]) >= -result["ci_noninferiority_margin"]
        else "NO-GO"
    )
    return result


def main() -> None:
    args = parse_args()
    started_wall = time.time()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    validate_test_receipt(receipt_path, args.config)
    seeds = [int(value) for value in config["seeds"]]
    physical_gpus = [int(value) for value in config["run_physical_gpus"]]
    expected_gpu = dict(zip(seeds, physical_gpus, strict=True))
    if args.seed not in expected_gpu or expected_gpu[args.seed] != args.physical_gpu:
        raise ValueError("Seed and physical GPU do not match the pre-registered mapping")
    device, device_manifest = _configure_device(args.seed, args.physical_gpu)

    family_map_path = Path(config["family_map"])
    family_map = load_family_map(family_map_path)
    source = config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    batch = adapter.load_observables()
    source_labels = adapter.load_source_labels()
    pool_variants: dict[str, tuple[ObservableQueryBatch, SourceTrainingLabels]] = {}
    for name, experts in sorted(config["pool_shift"]["pools"].items()):
        pool_variants[str(name)] = subset_expert_pool(
            batch,
            source_labels,
            tuple(str(value) for value in experts),
        )
    folds = _environment_folds(source_labels)
    if args.max_environments is not None:
        folds = folds[: args.max_environments]
    if not folds:
        raise ValueError("No source environments are available")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", config)
    prediction_groups: dict[str, list[Selection]] = defaultdict(list)
    component_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    fold_by_question: dict[str, tuple[str, int]] = {}
    neighbors = int(config["knn_k"])
    beta_grid = tuple(float(value) for value in config["beta_grid"])
    alpha_grid = tuple(float(value) for value in config["alpha_grid"])

    baseline_factories: dict[str, Callable[[], Any]] = {
        "source_best_single": SourceBestSelector,
        "majority_vote": MajorityVoteSelector,
        "source_accuracy_weighted_vote": SourceWeightedVoteSelector,
        "family_balanced_vote": FamilyBalancedVoteSelector,
        "output_profile_knn": lambda: OutputProfileKNNSelector(neighbors=neighbors),
        "global_local_competence": lambda: GlobalLocalSelector(neighbors=neighbors),
        "dare_reliability": lambda: LegacyDARESelector(neighbors=neighbors),
        "legacy_repair_chain": lambda: LegacyRepairChainSelector(neighbors=neighbors),
        "random_expert": lambda: RandomSelector(args.seed, clusters=False),
        "random_answer_cluster": lambda: RandomSelector(args.seed, clusters=True),
    }

    for fold_index, (environment, train_ids, test_ids) in enumerate(folds):
        train_batch = batch.subset(train_ids)
        test_batch = batch.subset(test_ids)
        train_labels = source_labels.subset(train_ids)
        parameters, inner_rows = _nested_parameters(
            train_batch,
            train_labels,
            beta_grid=beta_grid,
            alpha_grid=alpha_grid,
            neighbors=neighbors,
            seed=args.seed + fold_index * 1009,
            device=device,
            max_environments=args.max_inner_environments,
        )
        parameter_rows.extend(
            {"outer_fold": fold_index, "heldout_environment": environment, **row} for row in inner_rows
        )
        base = fit_repair_components(
            train_batch,
            train_labels,
            test_batch,
            neighbors=neighbors,
            seed=args.seed + fold_index,
            device=device,
        )
        components_by_mode = {"raw": base}
        for mode in ("no_self", "randomized", "symmetric", "column_centrality"):
            components_by_mode[mode] = with_graph_mode(
                base,
                base.repair_graph,
                mode,
                seed=args.seed + fold_index * 7919,
                device=device,
            )

        for name, factory in baseline_factories.items():
            selector = factory().fit(train_batch, train_labels)
            prediction_groups[name].extend(selector.predict(test_batch))
        for method in ABLATION_METHODS:
            mode = GRAPH_MODE_BY_METHOD.get(method, "raw")
            predictions = selections_from_components(
                test_batch,
                components_by_mode[mode],
                method,
                beta=(
                    parameters["m3_cluster_h1_support_beta"]
                    if method == "m3_cluster_h1_support"
                    else parameters["m3_h1_support_beta"]
                ),
                alpha=parameters["m5_h1_h2_alpha"],
                tie_tolerance=float(config["tie_tolerance"]),
            )
            prediction_groups[method].extend(predictions)

        # The full-pool nested parameters are frozen before applying them to the
        # original and Qwen-replacement judged4 pools.
        for pool_name, (pool_batch, pool_labels) in pool_variants.items():
            pool_train_batch = pool_batch.subset(train_ids)
            pool_test_batch = pool_batch.subset(test_ids)
            pool_train_labels = pool_labels.subset(train_ids)
            pool_components = fit_repair_components(
                pool_train_batch,
                pool_train_labels,
                pool_test_batch,
                neighbors=neighbors,
                seed=args.seed + fold_index,
                device=device,
            )
            pool_source_best = SourceBestSelector().fit(pool_train_batch, pool_train_labels)
            prediction_groups[f"{pool_name}__source_best_single"].extend(
                pool_source_best.predict(pool_test_batch)
            )
            for method in POOL_SHIFT_METHODS:
                prediction_groups[f"{pool_name}__{method}"].extend(
                    selections_from_components(
                        pool_test_batch,
                        pool_components,
                        method,
                        beta=(
                            parameters["m3_cluster_h1_support_beta"]
                            if method == "m3_cluster_h1_support"
                            else parameters["m3_h1_support_beta"]
                        ),
                        alpha=parameters["m5_h1_h2_alpha"],
                        tie_tolerance=float(config["tie_tolerance"]),
                    )
                )

        component_rows.extend(_component_rows(environment, base, parameters))
        for mode, components in components_by_mode.items():
            for source_index, source_expert in enumerate(components.expert_ids):
                for target_index, target_expert in enumerate(components.expert_ids):
                    edge_rows.append(
                        {
                            "outer_fold": fold_index,
                            "heldout_environment": environment,
                            "graph_mode": mode,
                            "source_expert": source_expert,
                            "target_expert": target_expert,
                            "weight": float(components.repair_graph[source_index, target_index]),
                        }
                    )
        for question_id in test_ids:
            fold_by_question[question_id] = (environment, fold_index)

    prediction_hashes: dict[str, str] = {}
    for method, predictions in sorted(prediction_groups.items()):
        ordered = sorted(predictions, key=lambda item: item.question_id)
        prediction_groups[method] = ordered
        prediction_hashes[method] = write_selections(args.output_dir / "predictions" / f"{method}.jsonl", ordered)
    write_jsonl(args.output_dir / "components.jsonl", component_rows)
    write_csv(args.output_dir / "nested_parameter_search.csv", parameter_rows)
    write_csv(args.output_dir / "graph_edges.csv", edge_rows)

    question_ids = sorted(fold_by_question)
    question_ids_sha = hashlib.sha256("\n".join(question_ids).encode("utf-8")).hexdigest()
    prediction_manifest = environment_manifest(
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
    prediction_manifest.update(
        {
            **device_manifest,
            "phase": "source_loso_prediction_before_evaluation",
            "source_questions": len(question_ids),
            "source_environments": len(folds),
            "heldout_environments": [environment for environment, _, _ in folds],
            "source_question_ids_sha256": question_ids_sha,
            "prediction_hashes_before_evaluation": prediction_hashes,
            "component_sha256": sha256_file(args.output_dir / "components.jsonl"),
            "parameter_search_sha256": sha256_file(args.output_dir / "nested_parameter_search.csv"),
            "graph_edges_sha256": sha256_file(args.output_dir / "graph_edges.csv"),
            "protocol": "outer source LOSO with nested source-only beta/alpha selection",
        }
    )
    write_json(args.output_dir / "prediction_manifest.json", prediction_manifest)

    # Evaluation labels are opened only after every prediction has been written and hashed.
    evaluation_labels = EvaluationLabelAdapter.from_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    ).load()
    evaluation_batch = batch.subset(question_ids)
    baseline = prediction_groups["source_best_single"]
    summaries: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    for method, predictions in sorted(prediction_groups.items()):
        summary, per_query = evaluate(
            method,
            predictions,
            baseline,
            evaluation_batch,
            evaluation_labels,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=args.seed,
        )
        summaries.append(summary)
        write_jsonl(args.output_dir / "per_query" / f"{method}.jsonl", per_query)
        by_environment: dict[str, list[Selection]] = defaultdict(list)
        for selection in predictions:
            by_environment[fold_by_question[selection.question_id][0]].append(selection)
        for environment, selections in sorted(by_environment.items()):
            environment_rows.append(
                {
                    "environment": environment,
                    "fold": fold_by_question[selections[0].question_id][1],
                    "method": method,
                    "samples": len(selections),
                    "accuracy": _accuracy(selections, evaluation_labels),
                }
            )

    paired: list[dict[str, Any]] = []
    pool_names = tuple(sorted(pool_variants))
    for method, predictions in sorted(prediction_groups.items()):
        pool_name = next((name for name in pool_names if method.startswith(f"{name}__")), None)
        reference_method = f"{pool_name}__m0_full" if pool_name is not None else "m0_full"
        if method == reference_method:
            continue
        paired.append(
            paired_selection_comparison(
                f"{method}_vs_{reference_method}",
                predictions,
                prediction_groups[reference_method],
                evaluation_labels,
                seed=args.seed,
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
        )
    original_pool = str(config["pool_shift"]["original_pool"])
    replacement_pool = str(config["pool_shift"]["replacement_pool"])
    for method in ("m0_full", "m3_h1_support", "m3_cluster_h1_support", "m4_h1", "m5_h1_h2"):
        paired.append(
            paired_selection_comparison(
                f"pool_shift_{replacement_pool}_vs_{original_pool}__{method}",
                prediction_groups[f"{replacement_pool}__{method}"],
                prediction_groups[f"{original_pool}__{method}"],
                evaluation_labels,
                seed=args.seed,
                bootstrap_samples=int(config["bootstrap_samples"]),
            )
        )
    paired.append(
        paired_selection_comparison(
            "m5_h1_h2_vs_m4_h1",
            prediction_groups["m5_h1_h2"],
            prediction_groups["m4_h1"],
            evaluation_labels,
            seed=args.seed,
            bootstrap_samples=int(config["bootstrap_samples"]),
        )
    )
    corrections = holm_adjust({row["comparison"]: float(row["exact_mcnemar_p"]) for row in paired})
    for row in paired:
        row["holm"] = corrections[row["comparison"]]

    write_json(args.output_dir / "summary.json", summaries)
    write_csv(
        args.output_dir / "summary.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in summaries],
    )
    write_csv(args.output_dir / "per_environment.csv", environment_rows)
    write_json(args.output_dir / "paired_comparisons.json", paired)
    write_csv(
        args.output_dir / "paired_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in paired],
    )

    direct_by_name = {str(row["comparison"]): row for row in paired}
    environment_accuracy = {
        (str(row["environment"]), str(row["method"])): float(row["accuracy"]) for row in environment_rows
    }
    environments = [environment for environment, _, _ in folds]
    m4_gate = _environment_gate(
        "m4_h1",
        "m0_full",
        environments,
        environment_accuracy,
        direct_by_name["m4_h1_vs_m0_full"],
        config,
    )
    m3_gate = _environment_gate(
        "m3_cluster_h1_support",
        "m0_full",
        environments,
        environment_accuracy,
        direct_by_name["m3_cluster_h1_support_vs_m0_full"],
        config,
    )
    required_pool_delta = float(config["pool_shift"]["required_delta_vs_pool_m0"])

    def pool_gate(method: str) -> dict[str, Any]:
        deltas = {
            pool_name: float(
                direct_by_name[
                    f"{pool_name}__{method}_vs_{pool_name}__m0_full"
                ]["delta"]
            )
            for pool_name in pool_names
        }
        return {
            "method": method,
            "delta_vs_pool_m0": deltas,
            "required_delta_vs_pool_m0": required_pool_delta,
            "decision": "PASS" if min(deltas.values()) >= required_pool_delta else "FAIL",
        }

    m4_pool_gate = pool_gate("m4_h1")
    m3_pool_gate = pool_gate("m3_cluster_h1_support")
    h2_direct = direct_by_name["m5_h1_h2_vs_m4_h1"]
    h2_environment_deltas = [
        environment_accuracy[(environment, "m5_h1_h2")] - environment_accuracy[(environment, "m4_h1")]
        for environment in environments
    ]
    summary_accuracy = {str(row["method"]): float(row["accuracy"]) for row in summaries}
    control_methods = [
        "m5_h1_h2_randomized",
        "m5_h1_h2_symmetric",
        "m5_h1_h2_no_self",
        "m5_h1_h2_centrality",
    ]
    h2_gate = {
        "comparison": "m5_h1_h2_vs_m4_h1",
        "micro_delta": float(h2_direct["delta"]),
        "worst_environment_delta": float(min(h2_environment_deltas)),
        "nonnegative_environment_fraction": float(np.mean(np.asarray(h2_environment_deltas) >= 0.0)),
        "paired_bootstrap_delta_ci95": h2_direct["paired_bootstrap_delta_ci95"],
        "real_accuracy": summary_accuracy["m5_h1_h2"],
        "control_accuracies": {method: summary_accuracy[method] for method in control_methods},
    }
    h2_gate["decision"] = (
        "RETAIN"
        if h2_gate["micro_delta"] > 0.0
        and h2_gate["worst_environment_delta"] >= 0.0
        and float(h2_gate["paired_bootstrap_delta_ci95"][0]) > 0.0
        and all(h2_gate["real_accuracy"] > value for value in h2_gate["control_accuracies"].values())
        else "DELETE"
    )
    legacy_mismatch = sum(
        first.normalized_answer != second.normalized_answer
        for first, second in zip(prediction_groups["m0_full"], prediction_groups["legacy_repair_chain"], strict=True)
    )
    seed_gate = {
        "seed": args.seed,
        "m4_pure_h1": m4_gate,
        "m3_cluster_h1_support": m3_gate,
        "m4_pool_shift": m4_pool_gate,
        "m3_pool_shift": m3_pool_gate,
        "h2_retention": h2_gate,
        "m0_vs_legacy_answer_mismatch_count": legacy_mismatch,
    }
    if m4_gate["decision"] == "GO" and m4_pool_gate["decision"] == "PASS":
        seed_gate.update({"decision": "GO", "selected_formula": "m4_h1"})
    elif m3_gate["decision"] == "GO" and m3_pool_gate["decision"] == "PASS":
        seed_gate.update({"decision": "GO", "selected_formula": "m3_cluster_h1_support"})
    else:
        seed_gate.update({"decision": "NO-GO", "selected_formula": None})
    write_json(args.output_dir / "seed_gate.json", seed_gate)

    torch.cuda.synchronize()
    resource_usage = {
        **device_manifest,
        "started_unix": started_wall,
        "finished_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    write_json(args.output_dir / "resource_usage.json", resource_usage)
    completed_artifacts = files_manifest(
        [
            args.output_dir / "prediction_manifest.json",
            args.output_dir / "predictions",
            args.output_dir / "components.jsonl",
            args.output_dir / "nested_parameter_search.csv",
            args.output_dir / "graph_edges.csv",
            args.output_dir / "per_query",
            args.output_dir / "summary.json",
            args.output_dir / "summary.csv",
            args.output_dir / "per_environment.csv",
            args.output_dir / "paired_comparisons.json",
            args.output_dir / "paired_comparisons.csv",
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
