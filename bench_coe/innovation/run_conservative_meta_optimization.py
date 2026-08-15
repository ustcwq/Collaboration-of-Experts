from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import binomtest

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
from .conservative_meta_selector import (
    PredictionTable,
    VoteDiagnostics,
    VoteRecipe,
    VoteScheme,
    correctness_matrix_from_selections,
    generate_recipes,
    materialize_recipe_selections,
    method_weights,
    recipe_choices,
    reliability_statistics,
    vote_diagnostics,
)
from .data import CacheAdapter, EvaluationLabelAdapter, load_family_map
from .schema import EvaluationLabels, Selection, SourceTrainingLabels


@dataclass(frozen=True)
class SourceSchemeState:
    reference_index: int
    reference_cluster: np.ndarray
    winning_cluster: np.ndarray
    chosen_method_index: np.ndarray
    winning_share: np.ndarray
    winning_margin: np.ndarray
    supporting_families: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search source-OOF conservative meta-selectors, freeze finalists, then evaluate "
            "authenticated cached targets only after every prediction is written and hashed"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-source-questions", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-target-questions", type=int)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--finalist-count", type=int)
    return parser.parse_args()


def _seed_dir(root: Path, seed: int) -> Path:
    matches = sorted(root.glob(f"seed_{seed}_gpu*"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise RuntimeError(f"Expected exactly one completed seed directory for {seed} under {root}")
    if not (matches[0] / "complete_manifest.json").is_file():
        raise RuntimeError(f"Base seed is incomplete: {matches[0]}")
    return matches[0]


def _read_authenticated_selections(path: Path, expected_hash: str) -> tuple[list[Selection], str]:
    digest = hashlib.sha256()
    rows: list[Selection] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            rows.append(Selection(**json.loads(raw_line.decode("utf-8"))))
    actual = digest.hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f"Authenticated base prediction hash mismatch: {path}")
    return rows, actual


def _load_source_tables(
    config: Mapping[str, Any],
    methods: Sequence[str],
    limit: int | None,
) -> tuple[
    list[PredictionTable],
    SourceTrainingLabels,
    np.ndarray,
    list[np.ndarray],
    dict[str, str],
    list[Path],
]:
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_root = Path(config["source_run_root"])
    seeds = [int(value) for value in config["source_seeds"]]
    tables: list[PredictionTable] = []
    authenticated: dict[str, str] = {}
    manifest_paths: list[Path] = []
    for seed in seeds:
        seed_dir = _seed_dir(source_root, seed)
        manifest_path = seed_dir / "prediction_manifest.json"
        manifest_paths.append(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = manifest.get("prediction_hashes_before_evaluation", {})
        loaded: dict[str, list[Selection]] = {}
        for method in methods:
            expected = hashes.get(method)
            if not isinstance(expected, str):
                raise RuntimeError(f"Source manifest does not authenticate {method}")
            path = seed_dir / "predictions" / f"{method}.jsonl"
            loaded[method], actual = _read_authenticated_selections(path, expected)
            authenticated[str(path)] = actual
        table = PredictionTable.from_selections(loaded, methods, limit=limit)
        if tables and table.question_ids != tables[0].question_ids:
            raise RuntimeError("Source seed prediction IDs are not aligned")
        tables.append(table)

    family_map_path = Path(source_config["family_map"])
    family_map = load_family_map(family_map_path)
    source = source_config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        [str(value) for value in source_config["experts"]],
        Path(source_config["dataset_registry"]),
        str(source_config["dataset_registry_sha256"]),
    )
    labels = adapter.load_source_labels().subset(tables[0].question_ids)
    environments = sorted({labels.environment_by_question[qid] for qid in tables[0].question_ids})
    environment_lookup = {value: index for index, value in enumerate(environments)}
    environment_index = np.asarray(
        [environment_lookup[labels.environment_by_question[qid]] for qid in tables[0].question_ids],
        dtype=np.int32,
    )
    correctness = [
        correctness_matrix_from_selections(table, labels.correctness) for table in tables
    ]
    return tables, labels, environment_index, correctness, authenticated, manifest_paths


def _reliability_cache(
    tables: Sequence[PredictionTable],
    correctness: Sequence[np.ndarray],
    environment_index: np.ndarray,
    references: Sequence[str],
) -> dict[tuple[str, int | None], Any]:
    result: dict[tuple[str, int | None], Any] = {}
    for reference in references:
        reference_index = tables[0].methods.index(reference)
        result[(reference, None)] = reliability_statistics(
            correctness,
            environment_index,
            np.ones(len(environment_index), dtype=bool),
            reference_index,
        )
        for environment in sorted(set(int(value) for value in environment_index)):
            result[(reference, environment)] = reliability_statistics(
                correctness,
                environment_index,
                environment_index != environment,
                reference_index,
            )
    return result


def _source_scheme_state(
    scheme: VoteScheme,
    tables: Sequence[PredictionTable],
    environment_index: np.ndarray,
    stats_cache: Mapping[tuple[str, int | None], Any],
    family_by_method: Mapping[str, str],
    method_pools: Mapping[str, Sequence[str]],
) -> SourceSchemeState:
    seed_count = len(tables)
    question_count = len(tables[0].question_ids)
    winner = np.full((seed_count, question_count), -1, dtype=np.int32)
    chosen = np.full((seed_count, question_count), tables[0].methods.index(scheme.reference), dtype=np.int32)
    share = np.zeros((seed_count, question_count), dtype=float)
    margin = np.zeros((seed_count, question_count), dtype=float)
    families = np.zeros((seed_count, question_count), dtype=np.int32)
    reference_index = tables[0].methods.index(scheme.reference)
    reference_cluster = np.stack(
        [table.cluster_matrix()[reference_index] for table in tables], axis=0
    )
    for environment in sorted(set(int(value) for value in environment_index)):
        indices = np.flatnonzero(environment_index == environment)
        active, weights = method_weights(
            stats_cache[(scheme.reference, environment)],
            scheme,
            tables[0].methods,
            family_by_method,
            method_pools[scheme.pool],
        )
        for seed_index, table in enumerate(tables):
            diagnostics = vote_diagnostics(
                table.subset_indices(indices),
                scheme,
                active,
                weights,
                family_by_method,
            )
            winner[seed_index, indices] = diagnostics.winning_cluster
            chosen[seed_index, indices] = diagnostics.chosen_method_index
            share[seed_index, indices] = diagnostics.winning_share
            margin[seed_index, indices] = diagnostics.winning_margin
            families[seed_index, indices] = diagnostics.supporting_families
    return SourceSchemeState(
        reference_index=reference_index,
        reference_cluster=reference_cluster,
        winning_cluster=winner,
        chosen_method_index=chosen,
        winning_share=share,
        winning_margin=margin,
        supporting_families=families,
    )


def _trial_row(
    recipe: VoteRecipe,
    state: SourceSchemeState,
    tables: Sequence[PredictionTable],
    correctness: Sequence[np.ndarray],
    environment_index: np.ndarray,
) -> dict[str, Any]:
    switch = (
        (state.winning_cluster >= 0)
        & (state.winning_cluster != state.reference_cluster)
        & (state.winning_share + 1e-12 >= recipe.min_share)
        & (state.winning_margin + 1e-12 >= recipe.min_margin)
        & (state.supporting_families >= recipe.min_families)
    )
    candidate_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    choices: list[np.ndarray] = []
    question_indices = np.arange(len(environment_index))
    for seed_index, values in enumerate(correctness):
        seed_choices = np.where(
            switch[seed_index],
            state.chosen_method_index[seed_index],
            state.reference_index,
        ).astype(np.int32)
        candidate_rows.append(values[seed_choices, question_indices])
        reference_rows.append(values[state.reference_index])
        choices.append(seed_choices)
    candidate = np.stack(candidate_rows, axis=0)
    reference = np.stack(reference_rows, axis=0)
    choice_matrix = np.stack(choices, axis=0)
    rescue = candidate & ~reference
    harm = ~candidate & reference
    environment_deltas: dict[str, float] = {}
    for environment in sorted(set(int(value) for value in environment_index)):
        mask = environment_index == environment
        environment_deltas[str(environment)] = float(
            candidate[:, mask].mean() - reference[:, mask].mean()
        )
    switch_count = int(switch.sum())
    signature = hashlib.sha256(choice_matrix.astype(np.int16).tobytes()).hexdigest()
    return {
        **recipe.to_dict(),
        "scheme_id": recipe.scheme.scheme_id,
        "source_accuracy": float(candidate.mean()),
        "reference_accuracy": float(reference.mean()),
        "source_delta": float(candidate.mean() - reference.mean()),
        "rescue_count_mean": float(rescue.sum(axis=1).mean()),
        "harm_count_mean": float(harm.sum(axis=1).mean()),
        "switch_count_mean": float(switch.sum(axis=1).mean()),
        "switch_rate": float(switch.mean()),
        "switch_precision": float(rescue.sum() / max(1, rescue.sum() + harm.sum())),
        "worst_environment_delta": min(environment_deltas.values()),
        "nonnegative_environment_fraction": float(
            np.mean([value >= -1e-12 for value in environment_deltas.values()])
        ),
        "prediction_signature": signature,
        "environment_deltas": environment_deltas,
        "source_seed_count": len(tables),
        "source_questions": len(environment_index),
        "selection_scope": "source_oof_only",
    }


def _select_finalists(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if float(row["source_delta"]) >= -1e-12 and float(row["switch_count_mean"]) > 0.0
    ]
    if not eligible:
        eligible = list(rows)
    selected: list[dict[str, Any]] = []
    signatures: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        signature = str(row["prediction_signature"])
        if signature not in signatures and len(selected) < count:
            signatures.add(signature)
            selected.append(row)

    orderings = [
        sorted(
            eligible,
            key=lambda row: (
                -float(row["source_accuracy"]),
                -float(row["worst_environment_delta"]),
                -float(row["switch_precision"]),
                str(row["method"]),
            ),
        ),
        sorted(
            eligible,
            key=lambda row: (
                -float(row["worst_environment_delta"]),
                -float(row["source_delta"]),
                float(row["switch_rate"]),
                str(row["method"]),
            ),
        ),
        sorted(
            eligible,
            key=lambda row: (
                -float(row["switch_precision"]),
                -float(row["source_delta"]),
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
    categorical_keys = [
        ("scheme", "aggregation"),
        ("scheme", "weighting"),
        ("scheme", "pool"),
    ]
    for parent, child in categorical_keys:
        categories = sorted({str(row[parent][child]) for row in eligible})
        for category in categories:
            candidates = [row for row in eligible if str(row[parent][child]) == category]
            if candidates:
                add(
                    sorted(
                        candidates,
                        key=lambda row: (-float(row["source_accuracy"]), str(row["method"])),
                    )[0]
                )
    cursor = 0
    while len(selected) < count and any(cursor < len(values) for values in orderings):
        for values in orderings:
            if cursor < len(values):
                add(values[cursor])
        cursor += 1
    for row in orderings[0]:
        add(row)
    return selected


def _flat_trial_row(row: Mapping[str, Any]) -> dict[str, Any]:
    scheme = row["scheme"]
    return {
        key: value
        for key, value in {
            "method": row["method"],
            "scheme_id": row["scheme_id"],
            "pool": scheme["pool"],
            "reference": scheme["reference"],
            "weighting": scheme["weighting"],
            "aggregation": scheme["aggregation"],
            "top_k": scheme["top_k"],
            "family_balanced": scheme["family_balanced"],
            "temperature": scheme["temperature"],
            "risk_penalty": scheme["risk_penalty"],
            "min_share": row["min_share"],
            "min_margin": row["min_margin"],
            "min_families": row["min_families"],
            "source_accuracy": row["source_accuracy"],
            "reference_accuracy": row["reference_accuracy"],
            "source_delta": row["source_delta"],
            "rescue_count_mean": row["rescue_count_mean"],
            "harm_count_mean": row["harm_count_mean"],
            "switch_count_mean": row["switch_count_mean"],
            "switch_rate": row["switch_rate"],
            "switch_precision": row["switch_precision"],
            "worst_environment_delta": row["worst_environment_delta"],
            "nonnegative_environment_fraction": row["nonnegative_environment_fraction"],
            "prediction_signature": row["prediction_signature"],
        }.items()
    }


def _source_materialized_predictions(
    recipe: VoteRecipe,
    tables: Sequence[PredictionTable],
    environment_index: np.ndarray,
    stats_cache: Mapping[tuple[str, int | None], Any],
    family_by_method: Mapping[str, str],
    method_pools: Mapping[str, Sequence[str]],
) -> list[list[Selection]]:
    result: list[list[Selection]] = [[] for _ in tables]
    for environment in sorted(set(int(value) for value in environment_index)):
        indices = np.flatnonzero(environment_index == environment)
        active, weights = method_weights(
            stats_cache[(recipe.scheme.reference, environment)],
            recipe.scheme,
            tables[0].methods,
            family_by_method,
            method_pools[recipe.scheme.pool],
        )
        for seed_index, table in enumerate(tables):
            subset = table.subset_indices(indices)
            diagnostics = vote_diagnostics(
                subset,
                recipe.scheme,
                active,
                weights,
                family_by_method,
            )
            result[seed_index].extend(materialize_recipe_selections(subset, recipe, diagnostics))
    for values in result:
        values.sort(key=lambda selection: selection.question_id)
    return result


def _load_target_table(
    panel_root: Path,
    seed: int,
    target_name: str,
    methods: Sequence[str],
    limit: int | None,
    authenticated: dict[str, str],
    manifest_paths: list[Path],
) -> tuple[PredictionTable, Path, dict[str, Any]]:
    seed_dir = _seed_dir(panel_root, seed)
    manifest_path = seed_dir / "prediction_manifest.json"
    manifest_paths.append(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_manifest = manifest.get("targets", {}).get(target_name)
    if not isinstance(target_manifest, dict):
        raise RuntimeError(f"Base target manifest is missing {target_name}")
    loaded: dict[str, list[Selection]] = {}
    for method in methods:
        expected = target_manifest.get("prediction_hashes_before_evaluation", {}).get(method)
        relative = target_manifest.get("prediction_paths", {}).get(method)
        if not isinstance(expected, str) or not isinstance(relative, str):
            raise RuntimeError(f"Base target manifest does not authenticate {target_name}/{method}")
        path = seed_dir / relative
        loaded[method], actual = _read_authenticated_selections(path, expected)
        authenticated[str(path)] = actual
    return PredictionTable.from_selections(loaded, methods, limit=limit), seed_dir, target_manifest


def _normal_paired_ci(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("Paired aggregate matrices must be aligned [seed, query]")
    query_delta = (candidate.astype(float) - reference.astype(float)).mean(axis=0)
    mean = float(query_delta.mean())
    if len(query_delta) < 2:
        return mean, mean
    standard_error = float(query_delta.std(ddof=1) / math.sqrt(len(query_delta)))
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error


def _aggregate_comparison(
    method: str,
    target: str,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    rescue_by_seed = np.sum(candidate & ~reference, axis=1)
    harm_by_seed = np.sum(~candidate & reference, axis=1)
    first_rescue = int(rescue_by_seed[0])
    first_harm = int(harm_by_seed[0])
    discordant = first_rescue + first_harm
    p_value = (
        float(binomtest(min(first_rescue, first_harm), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    ci_low, ci_high = _normal_paired_ci(candidate, reference)
    return {
        "target": target,
        "method": method,
        "samples": candidate.shape[1],
        "seeds": candidate.shape[0],
        "accuracy_mean": float(candidate.mean(axis=1).mean()),
        "accuracy_std": float(candidate.mean(axis=1).std()),
        "fcrg_full_accuracy_mean": float(reference.mean(axis=1).mean()),
        "delta_vs_fcrg_full": float(candidate.mean() - reference.mean()),
        "rescue_count_mean": float(rescue_by_seed.mean()),
        "harm_count_mean": float(harm_by_seed.mean()),
        "switch_precision": float(
            rescue_by_seed.sum() / max(1, rescue_by_seed.sum() + harm_by_seed.sum())
        ),
        "paired_normal_delta_ci95": [ci_low, ci_high],
        "exact_mcnemar_p_first_seed": p_value,
    }


def _correctness_for_labels(
    selections: Sequence[Selection],
    labels: EvaluationLabels,
) -> np.ndarray:
    prediction_ids = {selection.question_id for selection in selections}
    label_ids = {question_id for question_id, _ in labels.correctness}
    if prediction_ids != label_ids:
        raise RuntimeError(
            "Target prediction and evaluation-label question IDs differ: "
            f"predictions={len(prediction_ids)}, labels={len(label_ids)}"
        )
    return np.asarray(
        [
            bool(
                selection.selected_expert_id is not None
                and labels.get(selection.question_id, selection.selected_expert_id)
            )
            for selection in selections
        ],
        dtype=bool,
    )


def _registered_best(config: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_path in config.get("registered_aggregate_summaries", []):
        path = Path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("pool") != "full_pool":
                    continue
                target = row.get("target") or "source_loso"
                value = float(row["accuracy_mean"])
                result[target] = max(result.get(target, -math.inf), value)
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)

    search = config["search"]
    recipes = generate_recipes(search)
    if args.max_trials is not None:
        recipes = recipes[: args.max_trials]
    if not recipes:
        raise ValueError("No source-only optimization trials were generated")
    method_pools = {
        str(pool): tuple(str(method) for method in methods)
        for pool, methods in search["method_pools"].items()
    }
    methods = tuple(
        sorted(
            set().union(*[set(values) for values in method_pools.values()])
            | set(str(value) for value in search.get("references", ["fcrg_full"]))
            | set(str(value) for value in config.get("report_controls", []))
        )
    )
    family_by_method = {
        str(method): str(family) for method, family in search["family_by_method"].items()
    }
    missing_families = set(methods).difference(family_by_method)
    if missing_families:
        raise ValueError(f"Search config lacks method families: {sorted(missing_families)}")

    source_tables, source_labels, environment_index, source_correctness, authenticated, manifest_paths = (
        _load_source_tables(config, methods, args.max_source_questions)
    )
    references = sorted({recipe.scheme.reference for recipe in recipes})
    stats_cache = _reliability_cache(
        source_tables,
        source_correctness,
        environment_index,
        references,
    )
    recipes_by_scheme: dict[VoteScheme, list[VoteRecipe]] = defaultdict(list)
    for recipe in recipes:
        recipes_by_scheme[recipe.scheme].append(recipe)

    trial_rows: list[dict[str, Any]] = []
    for scheme in sorted(recipes_by_scheme, key=lambda value: value.scheme_id):
        state = _source_scheme_state(
            scheme,
            source_tables,
            environment_index,
            stats_cache,
            family_by_method,
            method_pools,
        )
        for recipe in recipes_by_scheme[scheme]:
            trial_rows.append(
                _trial_row(
                    recipe,
                    state,
                    source_tables,
                    source_correctness,
                    environment_index,
                )
            )
    write_json(args.output_dir / "source_trials.json", trial_rows)
    write_csv(args.output_dir / "source_trials.csv", [_flat_trial_row(row) for row in trial_rows])

    finalist_count = int(
        args.finalist_count if args.finalist_count is not None else config["finalist_count"]
    )
    finalists = _select_finalists(trial_rows, finalist_count)
    finalist_recipes = [VoteRecipe.from_dict(row) for row in finalists]
    frozen_payload = {
        "selection_labels": "source_oof_only",
        "target_labels_opened": False,
        "trial_count": len(trial_rows),
        "unique_prediction_count": len({row["prediction_signature"] for row in trial_rows}),
        "finalist_count": len(finalists),
        "finalists": finalists,
        "selection_rule": (
            "source-noninferior unique predictions selected by round-robin source accuracy, "
            "worst-environment delta, switch precision, conservatism, and source-only diversity"
        ),
    }
    write_json(args.output_dir / "frozen_finalists.json", frozen_payload)
    frozen_hash = sha256_file(args.output_dir / "frozen_finalists.json")

    prediction_hashes: dict[str, Any] = {"source": {}, "targets": {}}
    source_seeds = [int(value) for value in config["source_seeds"]]
    for recipe in finalist_recipes:
        by_seed = _source_materialized_predictions(
            recipe,
            source_tables,
            environment_index,
            stats_cache,
            family_by_method,
            method_pools,
        )
        for seed_index, selections in enumerate(by_seed):
            seed = source_seeds[seed_index]
            path = args.output_dir / "predictions" / "source_loso" / f"seed_{seed}" / f"{recipe.method}.jsonl"
            prediction_hashes["source"].setdefault(str(seed), {})[recipe.method] = write_selections(
                path, selections
            )

    full_stats = {
        reference: stats_cache[(reference, None)] for reference in references
    }
    target_jobs: list[dict[str, Any]] = []
    for panel_index, raw_panel in enumerate(config["target_panels"]):
        panel_config_path = Path(raw_panel["config"])
        panel_config = yaml.safe_load(panel_config_path.read_text(encoding="utf-8"))
        panel_name = str(raw_panel.get("name", panel_config_path.stem))
        seeds = [int(value) for value in panel_config["seeds"]]
        for target in panel_config["targets"]:
            target_jobs.append(
                {
                    "panel_index": panel_index,
                    "panel_name": panel_name,
                    "panel_config_path": panel_config_path,
                    "panel_config": panel_config,
                    "panel_root": Path(raw_panel["run_root"]),
                    "target": target,
                    "seeds": seeds,
                }
            )
    if args.max_targets is not None:
        target_jobs = target_jobs[: args.max_targets]
    if not target_jobs:
        raise ValueError("No diagnostic target jobs are configured")

    target_index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for job in target_jobs:
        target = job["target"]
        target_name = str(target["name"])
        panel_name = str(job["panel_name"])
        prediction_hashes["targets"].setdefault(target_name, {})
        for seed in job["seeds"]:
            table, base_seed_dir, target_manifest = _load_target_table(
                job["panel_root"],
                seed,
                target_name,
                methods,
                args.max_target_questions,
                authenticated,
                manifest_paths,
            )
            target_index[(panel_name, seed, target_name)] = {
                "base_seed_dir": str(base_seed_dir),
                "base_target_manifest": target_manifest,
                "questions": len(table.question_ids),
            }
            diagnostics_by_scheme: dict[VoteScheme, VoteDiagnostics] = {}
            for scheme in {recipe.scheme for recipe in finalist_recipes}:
                active, weights = method_weights(
                    full_stats[scheme.reference],
                    scheme,
                    table.methods,
                    family_by_method,
                    method_pools[scheme.pool],
                )
                diagnostics_by_scheme[scheme] = vote_diagnostics(
                    table,
                    scheme,
                    active,
                    weights,
                    family_by_method,
                )
            for recipe in finalist_recipes:
                selections = materialize_recipe_selections(
                    table,
                    recipe,
                    diagnostics_by_scheme[recipe.scheme],
                )
                relative = (
                    Path("predictions")
                    / panel_name
                    / f"seed_{seed}"
                    / target_name
                    / f"{recipe.method}.jsonl"
                )
                digest = write_selections(args.output_dir / relative, selections)
                prediction_hashes["targets"][target_name].setdefault(str(seed), {})[
                    recipe.method
                ] = {"path": str(relative), "sha256": digest}

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [
            args.config,
            Path(config["source_config"]),
            *[Path(panel["config"]) for panel in config["target_panels"]],
            *manifest_paths,
        ],
    )
    environment["authenticated_base_prediction_hashes"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": str(config["protocol_name"]),
        "scope": "development_ood_diagnostic_only",
        "frozen_finalists_sha256": frozen_hash,
        "source_trials": len(trial_rows),
        "predictions": prediction_hashes,
        "base_prediction_hashes": authenticated,
        "target_index": {
            "|".join((panel, str(seed), target)): value
            for (panel, seed, target), value in sorted(target_index.items())
        },
        "labels_opened": False,
        "written_before_any_target_label_adapter": True,
        "innovation_code_manifest_sha256": environment["innovation_code_manifest_sha256"],
    }
    write_json(args.output_dir / "prediction_manifest.json", prediction_manifest)
    prediction_manifest_hash = sha256_file(args.output_dir / "prediction_manifest.json")

    # Target evaluation begins only after every configured finalist prediction is on disk and hashed.
    aggregate_rows: list[dict[str, Any]] = []
    target_correctness: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    target_reference: dict[str, list[np.ndarray]] = defaultdict(list)
    target_control_correctness: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for job in target_jobs:
        panel_config = job["panel_config"]
        target = job["target"]
        target_name = str(target["name"])
        panel_name = str(job["panel_name"])
        labels = EvaluationLabelAdapter.from_registry(
            Path(target["label_cache_path"]),
            str(target["dataset"]),
            str(target["split"]),
            str(target["modality"]),
            [str(value) for value in panel_config["experts"]],
            Path(panel_config["dataset_registry"]),
            str(panel_config["dataset_registry_sha256"]),
        ).load(limit=args.max_target_questions)
        for seed in job["seeds"]:
            index_row = target_index[(panel_name, seed, target_name)]
            base_seed_dir = Path(index_row["base_seed_dir"])
            base_manifest = index_row["base_target_manifest"]
            for control in config["report_controls"]:
                relative = base_manifest["prediction_paths"][control]
                expected = base_manifest["prediction_hashes_before_evaluation"][control]
                values, _ = _read_authenticated_selections(base_seed_dir / relative, expected)
                values = sorted(values, key=lambda selection: selection.question_id)
                if args.max_target_questions is not None:
                    values = values[: args.max_target_questions]
                correctness_values = _correctness_for_labels(values, labels)
                target_control_correctness[(target_name, str(control))].append(
                    correctness_values
                )
                if control == "fcrg_full":
                    target_reference[target_name].append(correctness_values)
            for recipe in finalist_recipes:
                entry = prediction_hashes["targets"][target_name][str(seed)][recipe.method]
                path = args.output_dir / entry["path"]
                if sha256_file(path) != entry["sha256"]:
                    raise RuntimeError(f"New prediction hash changed before evaluation: {path}")
                values = read_selections(path)
                target_correctness[(target_name, recipe.method)].append(
                    _correctness_for_labels(values, labels)
                )

    for job in target_jobs:
        target_name = str(job["target"]["name"])
        reference = np.stack(target_reference[target_name], axis=0)
        for recipe in finalist_recipes:
            candidate = np.stack(target_correctness[(target_name, recipe.method)], axis=0)
            aggregate_rows.append(
                _aggregate_comparison(recipe.method, target_name, candidate, reference)
            )
        for control in config["report_controls"]:
            candidate = np.stack(
                target_control_correctness[(target_name, str(control))], axis=0
            )
            aggregate_rows.append(
                _aggregate_comparison(str(control), target_name, candidate, reference)
            )

    finalist_by_method = {str(row["method"]): row for row in finalists}
    source_reference_accuracy = float(finalists[0]["reference_accuracy"]) if finalists else 0.0
    for recipe in finalist_recipes:
        row = finalist_by_method[recipe.method]
        aggregate_rows.append(
            {
                "target": "source_loso",
                "method": recipe.method,
                "samples": len(source_tables[0].question_ids),
                "seeds": len(source_tables),
                "accuracy_mean": float(row["source_accuracy"]),
                "accuracy_std": 0.0,
                "fcrg_full_accuracy_mean": source_reference_accuracy,
                "delta_vs_fcrg_full": float(row["source_delta"]),
                "rescue_count_mean": float(row["rescue_count_mean"]),
                "harm_count_mean": float(row["harm_count_mean"]),
                "switch_precision": float(row["switch_precision"]),
                "paired_normal_delta_ci95": None,
                "exact_mcnemar_p_first_seed": None,
            }
        )
    for control in config["report_controls"]:
        method_index = source_tables[0].methods.index(str(control))
        candidate = np.stack([values[method_index] for values in source_correctness], axis=0)
        reference_index = source_tables[0].methods.index("fcrg_full")
        reference = np.stack([values[reference_index] for values in source_correctness], axis=0)
        aggregate_rows.append(
            _aggregate_comparison(str(control), "source_loso", candidate, reference)
        )

    registered_best = _registered_best(config)
    for row in aggregate_rows:
        best = registered_best.get(str(row["target"]))
        row["registered_best_accuracy"] = best
        row["delta_vs_registered_best"] = (
            float(row["accuracy_mean"]) - best if best is not None else None
        )
    write_json(args.output_dir / "aggregate_summary.json", aggregate_rows)
    write_csv(
        args.output_dir / "aggregate_summary.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
            for row in aggregate_rows
        ],
    )

    candidate_rows = [row for row in aggregate_rows if row["method"] in finalist_by_method]
    targets = sorted({str(row["target"]) for row in candidate_rows})
    goal_rows: list[dict[str, Any]] = []
    for recipe in finalist_recipes:
        by_target = {
            str(row["target"]): row
            for row in candidate_rows
            if row["method"] == recipe.method
        }
        deltas = {target: float(by_target[target]["delta_vs_fcrg_full"]) for target in targets}
        best_deltas = {
            target: by_target[target]["delta_vs_registered_best"] for target in targets
        }
        mmmu_delta = deltas.get("mmmu_pro_test_id")
        other = [value for target, value in deltas.items() if target != "mmmu_pro_test_id"]
        goal_rows.append(
            {
                "method": recipe.method,
                "mmmu_pro_test_delta_vs_fcrg_full": mmmu_delta,
                "all_dataset_delta_vs_fcrg_full": deltas,
                "all_dataset_delta_vs_registered_best": best_deltas,
                "other_datasets_nonnegative_vs_fcrg_full": bool(
                    other and min(other) >= -1e-12
                ),
                "all_datasets_nonnegative_vs_fcrg_full": bool(
                    deltas and min(deltas.values()) >= -1e-12
                ),
                "improves_mmmu_pro_test": bool(mmmu_delta is not None and mmmu_delta > 0.0),
                "strict_user_goal_met": bool(
                    mmmu_delta is not None
                    and mmmu_delta > 0.0
                    and other
                    and min(other) >= -1e-12
                ),
                "worst_delta_vs_fcrg_full": min(deltas.values()),
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
    decision = {
        "scope": "development_ood_diagnostic_only",
        "source_only_frozen_before_target_evaluation": True,
        "prediction_manifest_sha256_before_labels": prediction_manifest_hash,
        "source_trial_count": len(trial_rows),
        "unique_source_prediction_count": frozen_payload["unique_prediction_count"],
        "target_finalist_count": len(finalist_recipes),
        "strict_goal_met_count": len(strict),
        "strict_goal_met_methods": [row["method"] for row in strict],
        "default_selected_from_target_results": False,
        "can_authorize_locked_test": False,
        "note": (
            "All known targets are development diagnostics. Target outcomes may report whether a "
            "source-frozen candidate met the requested empirical guardrail, but may not select a "
            "future confirmatory default."
        ),
    }
    write_json(args.output_dir / "decision.json", decision)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "prediction_manifest_sha256": prediction_manifest_hash,
            "labels_opened_after_prediction_manifest": True,
            "evaluated_targets": [str(job["target"]["name"]) for job in target_jobs],
            "registered_best": registered_best,
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
            "frozen_finalists_sha256": frozen_hash,
            "artifact_hashes": files_manifest(completion_paths),
            "decision": decision,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), **decision}, indent=2))


if __name__ == "__main__":
    main()
