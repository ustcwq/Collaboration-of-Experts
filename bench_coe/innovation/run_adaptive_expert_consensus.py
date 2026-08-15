from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .adaptive_expert_consensus import (
    AdaptiveConsensusVariant,
    adaptive_expert_consensus,
)
from .artifacts import (
    environment_manifest,
    manifest_sha256,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .conditioned_expert_consensus import fit_conditioned_expert_profiles
from .method_consensus import apply_consensus_gate
from .run_conditioned_expert_consensus import _score, _source_data, _target_batch
from .run_strict_positive_portfolio import (
    _completion_bound,
    _dataset_jobs,
    _load_labels,
    _prior_art_rows,
)
from .run_conservative_meta_optimization import _read_authenticated_selections
from .schema import ObservableQueryBatch, Selection, SourceTrainingLabels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-OOF-selected adaptive expert consensus"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-target-seeds", type=int)
    parser.add_argument("--max-base-variants", type=int)
    parser.add_argument("--max-datasets", type=int)
    return parser.parse_args()


def _token(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _variants(config: Mapping[str, Any]) -> list[AdaptiveConsensusVariant]:
    grid = config["candidate_grid"]
    variants: list[AdaptiveConsensusVariant] = []
    for aggregation in grid["aggregations"]:
        for prior in grid["prior_strengths"]:
            for power in grid["reliability_powers"]:
                for temperature in grid["uncertainty_temperatures"]:
                    for validity in grid["validity_powers"]:
                        for support in grid["support_powers"]:
                            for family in grid["family_balance"]:
                                for em_mix in grid["em_mixes"]:
                                    iterations = int(grid["em_iterations"]) if float(em_mix) > 0 else 0
                                    name = (
                                        f"aecs__{aggregation}__pr{_token(prior)}__rp{_token(power)}__"
                                        f"ut{_token(temperature)}__vp{_token(validity)}__"
                                        f"sp{_token(support)}__fb{int(bool(family))}__em{_token(em_mix)}"
                                    )
                                    variants.append(
                                        AdaptiveConsensusVariant(
                                            name=name,
                                            prior_strength=float(prior),
                                            reliability_power=float(power),
                                            uncertainty_temperature=float(temperature),
                                            validity_power=float(validity),
                                            support_power=float(support),
                                            aggregation=str(aggregation),
                                            family_balance=bool(family),
                                            em_mix=float(em_mix),
                                            em_iterations=iterations,
                                        )
                                    )
    return variants


def _candidate_name(variant: AdaptiveConsensusVariant, share: float, advantage: float) -> str:
    return f"{variant.name}__sh{_token(share)}__ad{_token(advantage)}"


def _observable_majority_reference(batch: ObservableQueryBatch) -> list[Selection]:
    result: list[Selection] = []
    for question_id in batch.question_ids:
        rows = batch.for_question(question_id)
        valid = [
            row
            for row in rows
            if row.valid_output and row.normalized_answer is not None
        ]
        counts = Counter(str(row.normalized_answer) for row in valid)
        if not counts:
            result.append(
                Selection(
                    question_id, None, None, None, {}, {},
                    "no_valid_expert_output", {
                        "method": "observable_majority_reference",
                        "source_reference_uses_labels": False,
                        "valid_mask": {row.expert_id: bool(row.valid_output) for row in rows},
                        "missing_mask": {row.expert_id: not row.valid_output for row in rows},
                    },
                )
            )
            continue
        answer = sorted(counts, key=lambda value: (-counts[value], value))[0]
        selected = sorted(
            (row for row in valid if str(row.normalized_answer) == answer),
            key=lambda row: (row.uncertainty, row.expert_id),
        )[0]
        cluster_scores = {
            str(
                min(
                    row.per_query_cluster_id
                    for row in valid
                    if str(row.normalized_answer) == value
                    and row.per_query_cluster_id is not None
                )
            ): float(count)
            for value, count in counts.items()
        }
        valid_mask = {row.expert_id: bool(row.valid_output) for row in rows}
        result.append(
            Selection(
                question_id=question_id,
                selected_cluster_id=selected.per_query_cluster_id,
                selected_expert_id=selected.expert_id,
                normalized_answer=answer,
                cluster_scores=cluster_scores,
                expert_scores={row.expert_id: 1.0 for row in valid},
                fallback_reason=None,
                observable_features={
                    "method": "observable_majority_reference",
                    "source_reference_uses_labels": False,
                    "valid_mask": valid_mask,
                    "missing_mask": {
                        expert: not value for expert, value in valid_mask.items()
                    },
                },
                tie_breaking="answer_support_then_answer;uncertainty_then_expert_id",
            )
        )
    return result


def _stratified_folds(
    batch: ObservableQueryBatch,
    subject_groups: Mapping[str, str],
    fold_count: int,
) -> list[tuple[str, ...]]:
    if fold_count < 2:
        raise ValueError("Source OOF requires at least two folds")
    by_group: dict[str, list[str]] = {}
    for question_id in batch.question_ids:
        subject = batch.for_question(question_id)[0].subject
        if subject not in subject_groups:
            raise ValueError(f"Source subject lacks configured group: {subject}")
        by_group.setdefault(subject_groups[subject], []).append(question_id)
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for group, question_ids in sorted(by_group.items()):
        ordered = sorted(
            question_ids,
            key=lambda question_id: hashlib.sha256(
                f"{group}:{question_id}".encode("utf-8")
            ).hexdigest(),
        )
        for index, question_id in enumerate(ordered):
            folds[index % fold_count].append(question_id)
    if any(not fold for fold in folds):
        raise ValueError("Source OOF produced an empty fold")
    flat = [question_id for fold in folds for question_id in fold]
    if len(flat) != len(set(flat)) or set(flat) != set(batch.question_ids):
        raise RuntimeError("Source OOF folds do not cover every question exactly once")
    return [tuple(sorted(fold)) for fold in folds]


def _score_source_oof(
    method: str,
    candidate: Sequence[Selection],
    reference: Sequence[Selection],
    labels: SourceTrainingLabels,
) -> dict[str, Any]:
    candidate_by_id = {row.question_id: row for row in candidate}
    reference_by_id = {row.question_id: row for row in reference}
    if set(candidate_by_id) != set(reference_by_id):
        raise RuntimeError("Source OOF candidate/reference IDs differ")
    ids = sorted(candidate_by_id)
    candidate_values = {
        question_id: bool(
            candidate_by_id[question_id].selected_expert_id
            and labels.get(
                question_id,
                candidate_by_id[question_id].selected_expert_id or "",
            )
        )
        for question_id in ids
    }
    reference_values = {
        question_id: bool(
            reference_by_id[question_id].selected_expert_id
            and labels.get(
                question_id,
                reference_by_id[question_id].selected_expert_id or "",
            )
        )
        for question_id in ids
    }
    environments = sorted(set(labels.environment_by_question[question_id] for question_id in ids))
    environment_accuracy: list[float] = []
    environment_delta: list[float] = []
    for environment in environments:
        keep = [
            question_id
            for question_id in ids
            if labels.environment_by_question[question_id] == environment
        ]
        environment_accuracy.append(float(np.mean([candidate_values[qid] for qid in keep])))
        environment_delta.append(
            float(
                np.mean([candidate_values[qid] for qid in keep])
                - np.mean([reference_values[qid] for qid in keep])
            )
        )
    rescue = sum(candidate_values[qid] and not reference_values[qid] for qid in ids)
    harm = sum(reference_values[qid] and not candidate_values[qid] for qid in ids)
    switched = sum(
        candidate_by_id[qid].normalized_answer != reference_by_id[qid].normalized_answer
        for qid in ids
    )
    accuracy = float(np.mean(list(candidate_values.values())))
    reference_accuracy = float(np.mean(list(reference_values.values())))
    return {
        "method": method,
        "samples": len(ids),
        "accuracy": accuracy,
        "reference_accuracy": reference_accuracy,
        "delta_vs_reference": accuracy - reference_accuracy,
        "balanced_environment_accuracy": float(np.mean(environment_accuracy)),
        "worst_environment_delta": min(environment_delta),
        "nonnegative_environment_fraction": float(
            np.mean([value >= -1e-12 for value in environment_delta])
        ),
        "rescue_count": rescue,
        "harm_count": harm,
        "switch_count": switched,
        "switch_precision": rescue / max(1, rescue + harm),
        "selection_scope": "source_oof_only",
    }


def _source_oof_candidates(
    source_batch: ObservableQueryBatch,
    source_labels: SourceTrainingLabels,
    source_groups: Mapping[str, str],
    reference: Sequence[Selection],
    variants: Sequence[AdaptiveConsensusVariant],
    gates: Sequence[tuple[float, float]],
    fold_count: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[AdaptiveConsensusVariant, float, float]]]:
    folds = _stratified_folds(source_batch, source_groups, fold_count)
    all_ids = set(source_batch.question_ids)
    reference_by_id = {row.question_id: row for row in reference}
    fold_state: list[tuple[ObservableQueryBatch, Mapping[str, Any], list[Selection]]] = []
    for validation_ids in folds:
        train_ids = sorted(all_ids.difference(validation_ids))
        profiles = fit_conditioned_expert_profiles(
            source_batch.subset(train_ids),
            source_labels.subset(train_ids),
            source_groups,
        )
        fold_state.append(
            (
                source_batch.subset(validation_ids),
                profiles,
                [reference_by_id[question_id] for question_id in validation_ids],
            )
        )
    scores: list[dict[str, Any]] = []
    specs: dict[str, tuple[AdaptiveConsensusVariant, float, float]] = {}
    signatures: dict[tuple[Any, ...], str] = {}
    for variant in variants:
        ungated: list[Selection] = []
        aligned_reference: list[Selection] = []
        for validation_batch, profiles, fold_reference in fold_state:
            ungated.extend(
                adaptive_expert_consensus(
                    validation_batch,
                    profiles,
                    source_groups,
                    variant,
                    reference=fold_reference,
                )
            )
            aligned_reference.extend(fold_reference)
        ungated = sorted(ungated, key=lambda row: row.question_id)
        aligned_reference = sorted(aligned_reference, key=lambda row: row.question_id)
        for share, advantage in gates:
            name = _candidate_name(variant, share, advantage)
            candidate = apply_consensus_gate(
                ungated,
                aligned_reference,
                name=name,
                fallback_share=share,
                minimum_advantage=advantage,
            )
            signature = tuple(
                (row.question_id, row.normalized_answer, row.selected_expert_id)
                for row in candidate
            )
            if signature in signatures:
                continue
            signatures[signature] = name
            specs[name] = (variant, share, advantage)
            scores.append(
                _score_source_oof(
                    name, candidate, aligned_reference, source_labels
                )
            )
    return scores, specs


def _select_source_candidates(
    scores: Sequence[Mapping[str, Any]],
    specs: Mapping[str, tuple[AdaptiveConsensusVariant, float, float]],
    config: Mapping[str, Any],
) -> list[str]:
    selected: list[str] = []
    top_count = int(config["top_per_source_metric"])
    for metric in config["source_selection_metrics"]:
        ranked = sorted(
            scores,
            key=lambda row: (
                float(row[str(metric)]),
                float(row["accuracy"]),
                float(row["worst_environment_delta"]),
                str(row["method"]),
            ),
            reverse=True,
        )
        for row in ranked[:top_count]:
            method = str(row["method"])
            if method not in selected:
                selected.append(method)
    maximum = int(config["maximum_target_candidates"])
    selected = selected[:maximum]
    if not selected or any(method not in specs for method in selected):
        raise RuntimeError("Source-only candidate selection is empty or invalid")
    return selected


def _authenticated_v2_rows(
    root: Path,
    dataset: str,
    seed: int,
    prediction_manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
    boundary: Mapping[str, Any],
    authenticated: dict[str, str],
) -> list[Selection]:
    entry = prediction_manifest["predictions"][dataset][str(seed)]
    path = root / str(entry["path"])
    expected = str(entry["sha256"])
    _completion_bound(path, expected, completion)
    if boundary.get("prediction_files_sha256", {}).get(str(path)) != expected:
        raise RuntimeError(f"v2 input is not prediction-boundary authenticated: {path}")
    rows, actual = _read_authenticated_selections(path, expected)
    authenticated[str(path)] = actual
    return sorted(rows, key=lambda row: row.question_id)


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)

    portfolio_path = Path(config["portfolio_config"])
    portfolio = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
    jobs = _dataset_jobs(portfolio)
    variants = _variants(config)
    if args.max_base_variants is not None:
        variants = variants[: args.max_base_variants]
    gates = [
        (float(row["share"]), float(row["advantage"]))
        for row in config["candidate_grid"]["gates"]
    ]
    if not variants or not gates:
        raise ValueError("Adaptive candidate grid is empty")

    v2_root = Path(config["v2_run_root"])
    v2_prediction_path = v2_root / "prediction_manifest.json"
    v2_completion_path = v2_root / "complete_manifest.json"
    v2_boundary_path = v2_root / "prediction_boundary.json"
    v2_prediction = json.loads(v2_prediction_path.read_text(encoding="utf-8"))
    v2_completion = json.loads(v2_completion_path.read_text(encoding="utf-8"))
    v2_boundary = json.loads(v2_boundary_path.read_text(encoding="utf-8"))
    v2_prediction_hash = sha256_file(v2_prediction_path)
    if v2_prediction_hash != v2_completion["prediction_manifest_sha256_before_labels"]:
        raise RuntimeError("v2 prediction manifest is not completion-bound")
    if v2_prediction_hash != v2_boundary["prediction_manifest_sha256_before_labels"]:
        raise RuntimeError("v2 prediction manifest is not boundary-bound")
    if v2_prediction.get("target_labels_opened_during_prediction") is not False:
        raise RuntimeError("v2 input does not attest to the target-label firewall")

    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = {
        v2_prediction_path,
        v2_completion_path,
        v2_boundary_path,
    }
    predictions: dict[str, dict[int, dict[str, list[Selection]]]] = {}
    references: dict[str, dict[int, list[Selection]]] = {}
    fcrg_rows: dict[str, dict[int, list[Selection]]] = {}
    selected_by_dataset: dict[str, list[str]] = {}
    specs_by_dataset: dict[
        str, dict[str, tuple[AdaptiveConsensusVariant, float, float]]
    ] = {}
    output_hashes: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    source_rows_all: list[dict[str, Any]] = []
    source_selection_manifest: dict[str, Any] = {}
    dataset_items = list(config["datasets"].items())
    if args.max_datasets is not None:
        dataset_items = dataset_items[: args.max_datasets]

    for dataset, spec in dataset_items:
        dataset = str(dataset)
        source_path = Path(spec["source_config"])
        panel_path = Path(spec["target_panel_config"])
        source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        panel = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
        source_batch, source_labels = _source_data(source_config)
        source_groups = {
            str(key): str(value)
            for key, value in spec["source_subject_groups"].items()
        }
        target_groups = {
            str(key): str(value)
            for key, value in spec["target_subject_groups"].items()
        }
        source_reference_method = str(spec["source_reference_method"])
        if source_reference_method == "observable_majority":
            source_reference = _observable_majority_reference(source_batch)
        else:
            source_job = jobs[str(spec["source_job"])]
            source_seed = int(source_job["seeds"][0])
            source_reference = _prior_art_rows(
                source_job,
                source_seed,
                source_reference_method,
                authenticated,
                manifest_paths,
            )
        source_scores, candidate_specs = _source_oof_candidates(
            source_batch,
            source_labels,
            source_groups,
            source_reference,
            variants,
            gates,
            int(config["source_oof_folds"]),
        )
        for row in source_scores:
            source_rows_all.append({"dataset": dataset, **row})
        selected = _select_source_candidates(source_scores, candidate_specs, config)
        selected_by_dataset[dataset] = selected
        specs_by_dataset[dataset] = candidate_specs
        source_selection_manifest[dataset] = {
            "source_config": str(source_path),
            "source_reference_method": source_reference_method,
            "candidate_count_before_source_selection": len(source_scores),
            "selected_candidates": selected,
            "source_primary_candidate": selected[0],
        }

        target_batch = _target_batch(panel, dataset, args.max_questions)
        target_job = jobs[dataset]
        seeds = list(target_job["seeds"])
        if args.max_target_seeds is not None:
            seeds = seeds[: args.max_target_seeds]
        target_profiles = fit_conditioned_expert_profiles(
            source_batch, source_labels, source_groups
        )
        predictions[dataset] = {}
        references[dataset] = {}
        fcrg_rows[dataset] = {}
        output_hashes[dataset] = {}
        by_variant: dict[str, list[str]] = {}
        for method in selected:
            by_variant.setdefault(candidate_specs[method][0].name, []).append(method)
        for seed in seeds:
            reference = _authenticated_v2_rows(
                v2_root,
                dataset,
                seed,
                v2_prediction,
                v2_completion,
                v2_boundary,
                authenticated,
            )
            fcrg = _prior_art_rows(
                target_job,
                seed,
                str(config["fcrg_method"]),
                authenticated,
                manifest_paths,
            )
            if args.max_questions is not None:
                reference = reference[: args.max_questions]
                fcrg = fcrg[: args.max_questions]
            references[dataset][seed] = reference
            fcrg_rows[dataset][seed] = fcrg
            predictions[dataset][seed] = {}
            output_hashes[dataset][str(seed)] = {}
            for variant_name, methods in by_variant.items():
                variant = candidate_specs[methods[0]][0]
                ungated = adaptive_expert_consensus(
                    target_batch,
                    target_profiles,
                    target_groups,
                    variant,
                    reference=reference,
                )
                for method in methods:
                    _, share, advantage = candidate_specs[method]
                    rows = apply_consensus_gate(
                        ungated,
                        reference,
                        name=method,
                        fallback_share=share,
                        minimum_advantage=advantage,
                    )
                    predictions[dataset][seed][method] = rows
                    relative = (
                        Path("predictions") / dataset / f"seed_{seed}" / f"{method}.jsonl"
                    )
                    digest = write_selections(args.output_dir / relative, rows)
                    output_hashes[dataset][str(seed)][method] = {
                        "path": str(relative),
                        "sha256": digest,
                    }

    write_csv(args.output_dir / "source_oof_results.csv", source_rows_all)
    write_json(args.output_dir / "source_selection_manifest.json", source_selection_manifest)
    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [args.config, portfolio_path, *sorted(manifest_paths)],
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "source_oof_candidate_selection_completed_before_target_evaluation": True,
        "selected_candidates": selected_by_dataset,
        "predictions": output_hashes,
        "v2_prediction_manifest_sha256": v2_prediction_hash,
        "source_labels_used_for_oof_selection_and_profiles": True,
        "target_unlabeled_observables_used_for_em": True,
        "target_labels_opened_during_prediction": False,
    }
    prediction_manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_target_labels": sha256_file(
            prediction_manifest_path
        ),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "target_labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    evaluation_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    best_by_dataset: dict[str, dict[str, Any]] = {}
    source_primary_results: dict[str, dict[str, Any]] = {}
    for dataset, by_seed in predictions.items():
        labels = _load_labels(jobs[dataset], args.max_questions)
        for seed, by_method in by_seed.items():
            for method, rows in by_method.items():
                evaluation_rows.append(
                    _score(
                        dataset,
                        seed,
                        method,
                        rows,
                        references[dataset][seed],
                        fcrg_rows[dataset][seed],
                        labels,
                    )
                )
        for method in selected_by_dataset[dataset]:
            rows = [
                row
                for row in evaluation_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed_count": len(rows),
                    "samples_per_seed": rows[0]["samples"],
                    "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
                    "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
                    "v2_accuracy_mean": float(np.mean([row["base_accuracy"] for row in rows])),
                    "fcrg_accuracy_mean": float(np.mean([row["fcrg_accuracy"] for row in rows])),
                    "delta_vs_v2_mean": float(np.mean([row["delta_vs_base"] for row in rows])),
                    "delta_vs_fcrg_mean": float(np.mean([row["delta_vs_fcrg"] for row in rows])),
                    "delta_vs_fcrg_min_seed": float(np.min([row["delta_vs_fcrg"] for row in rows])),
                }
            )
        ranked = sorted(
            (row for row in aggregate_rows if row["dataset"] == dataset),
            key=lambda row: (
                row["delta_vs_fcrg_mean"],
                row["delta_vs_v2_mean"],
                row["method"],
            ),
            reverse=True,
        )
        best_by_dataset[dataset] = ranked[0]
        primary = source_selection_manifest[dataset]["source_primary_candidate"]
        source_primary_results[dataset] = next(
            row for row in aggregate_rows if row["dataset"] == dataset and row["method"] == primary
        )
    write_csv(args.output_dir / "candidate_results.csv", evaluation_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    threshold = float(config["acceptance"]["minimum_delta_vs_fcrg"])
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "best_by_dataset_posthoc_development_diagnostic": best_by_dataset,
            "source_primary_results": source_primary_results,
            "all_datasets_have_a_selected_candidate_above_threshold": all(
                float(row["delta_vs_fcrg_min_seed"]) > threshold
                for row in best_by_dataset.values()
            ),
            "all_source_primary_candidates_above_threshold": all(
                float(row["delta_vs_fcrg_min_seed"]) > threshold
                for row in source_primary_results.values()
            ),
            "minimum_delta_vs_fcrg_exclusive": threshold,
            "target_label_firewall": (
                "Source OOF selected the target candidate set. All target predictions were "
                "written and hashed before target labels were opened."
            ),
        },
    )
    artifact_hashes = {
        str(path): sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "complete_manifest.json"
    }
    write_json(
        args.output_dir / "complete_manifest.json",
        {
            "protocol": config["protocol_name"],
            "scope": config["scope"],
            "runtime_seconds": time.time() - started,
            "prediction_manifest_sha256_before_target_labels": boundary[
                "prediction_manifest_sha256_before_target_labels"
            ],
            "artifact_hashes": artifact_hashes,
            "artifact_manifest_sha256": manifest_sha256(artifact_hashes),
        },
    )


if __name__ == "__main__":
    main()
