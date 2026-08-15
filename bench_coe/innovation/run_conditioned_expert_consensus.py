from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .artifacts import (
    environment_manifest,
    manifest_sha256,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .conditioned_expert_consensus import (
    ConditionedConsensusVariant,
    conditioned_expert_consensus,
    fit_conditioned_expert_profiles,
)
from .data import CacheAdapter, load_family_map
from .evaluation import exact_mcnemar, selection_correctness
from .method_consensus import apply_consensus_gate
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-subject-conditioned expert consensus before target evaluation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-target-seeds", type=int)
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--max-raw-candidates", type=int)
    return parser.parse_args()


def _source_data(config: Mapping[str, Any]) -> tuple[Any, Any]:
    source = config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        load_family_map(Path(config["family_map"])),
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    return adapter.load_observables(), adapter.load_source_labels()


def _target_batch(panel: Mapping[str, Any], target_name: str, limit: int | None) -> Any:
    target = next(row for row in panel["targets"] if str(row["name"]) == target_name)
    adapter = CacheAdapter.from_target_observables(
        Path(target["observable_cache_path"]),
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        load_family_map(Path(panel["family_map"])),
        [str(value) for value in panel["experts"]],
        str(target["observable_manifest_sha256"]),
    )
    return adapter.load_observables(limit=limit)


def _signature(rows_by_seed: Mapping[int, list[Any]]) -> tuple[Any, ...]:
    return tuple(
        (seed, row.question_id, row.normalized_answer, row.selected_expert_id)
        for seed, rows in sorted(rows_by_seed.items())
        for row in rows
    )


def _score(
    dataset: str,
    seed: int,
    method: str,
    candidate: list[Any],
    base: list[Any],
    fcrg: list[Any],
    labels: Any,
) -> dict[str, Any]:
    mappings = [selection_correctness(rows, labels) for rows in (candidate, base, fcrg)]
    if not (set(mappings[0]) == set(mappings[1]) == set(mappings[2])):
        raise RuntimeError(f"Conditioned candidate IDs differ: {dataset}/{seed}/{method}")
    ids = sorted(mappings[0])
    values = [np.asarray([mapping[qid] for qid in ids], dtype=np.int8) for mapping in mappings]
    rescue, harm, p_value = exact_mcnemar(values[0], values[2])
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "samples": len(ids),
        "accuracy": float(values[0].mean()),
        "base_accuracy": float(values[1].mean()),
        "fcrg_accuracy": float(values[2].mean()),
        "delta_vs_base": float((values[0] - values[1]).mean()),
        "delta_vs_fcrg": float((values[0] - values[2]).mean()),
        "rescue_vs_fcrg": rescue,
        "harm_vs_fcrg": harm,
        "exact_mcnemar_p_vs_fcrg": p_value,
    }


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
    fcrg_method = str(config["fcrg_method"])
    variants = [
        ConditionedConsensusVariant(
            name=(
                f"cecons__prior{str(float(prior)).replace('.', 'p')}__"
                f"power{str(float(power)).replace('.', 'p')}__"
                f"ut{str(float(temperature)).replace('.', 'p')}__"
                f"family{int(bool(family))}__valid{str(float(validity)).replace('.', 'p')}"
            ),
            prior_strength=float(prior),
            reliability_power=float(power),
            uncertainty_temperature=float(temperature),
            family_balance=bool(family),
            validity_power=float(validity),
        )
        for prior in config["candidate_grid"]["prior_strengths"]
        for power in config["candidate_grid"]["reliability_powers"]
        for temperature in config["candidate_grid"]["uncertainty_temperatures"]
        for family in config["candidate_grid"]["family_balance"]
        for validity in config["candidate_grid"]["validity_powers"]
    ]
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    gates = [
        (float(share), float(advantage))
        for share in config["candidate_grid"]["fallback_shares"]
        for advantage in config["candidate_grid"]["minimum_advantages"]
    ]
    if not variants or not gates:
        raise ValueError("Conditioned expert consensus grid is empty")

    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = set()
    input_paths: set[Path] = {args.config, portfolio_path}
    predictions: dict[str, dict[str, dict[int, list[Any]]]] = {}
    bases: dict[str, dict[int, list[Any]]] = {}
    fcrg_rows: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    profiles_by_dataset: dict[str, Any] = {}
    aliases_by_dataset: dict[str, dict[str, str]] = {}
    grid_counts: dict[str, dict[str, int]] = {}
    for dataset, spec in config["datasets"].items():
        dataset = str(dataset)
        source_path = Path(spec["source_config"])
        panel_path = Path(spec["target_panel_config"])
        source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        panel = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
        input_paths.update((source_path, panel_path, Path(source_config["source"]["cache_path"])))
        source_batch, source_labels = _source_data(source_config)
        profiles = fit_conditioned_expert_profiles(
            source_batch,
            source_labels,
            {str(key): str(value) for key, value in spec["source_subject_groups"].items()},
        )
        profiles_by_dataset[dataset] = profiles
        target_batch = _target_batch(panel, dataset, args.max_questions)
        job = jobs[dataset]
        base_method = str(spec["base_method"])
        seeds = list(job["seeds"])
        if args.max_target_seeds is not None:
            seeds = seeds[: args.max_target_seeds]
        bases[dataset] = {}
        fcrg_rows[dataset] = {}
        for seed in seeds:
            base = _prior_art_rows(job, seed, base_method, authenticated, manifest_paths)
            fcrg = _prior_art_rows(job, seed, fcrg_method, authenticated, manifest_paths)
            if args.max_questions is not None:
                base = base[: args.max_questions]
                fcrg = fcrg[: args.max_questions]
            bases[dataset][seed] = base
            fcrg_rows[dataset][seed] = fcrg

        kept: dict[str, dict[int, list[Any]]] = {}
        unique_by_signature: dict[tuple[Any, ...], str] = {}
        aliases: dict[str, str] = {}
        raw_count = 0
        stop = False
        target_groups = {str(key): str(value) for key, value in spec["target_subject_groups"].items()}
        for variant in variants:
            ungated_by_seed = {
                seed: conditioned_expert_consensus(
                    target_batch,
                    profiles,
                    target_groups,
                    variant,
                    reference=bases[dataset][seed],
                )
                for seed in seeds
            }
            for share, advantage in gates:
                share_name = str(share).replace(".", "p")
                advantage_name = str(advantage).replace(".", "p")
                name = f"{variant.name}__share{share_name}__adv{advantage_name}"
                candidate_by_seed = {
                    seed: apply_consensus_gate(
                        ungated_by_seed[seed],
                        bases[dataset][seed],
                        name=name,
                        fallback_share=share,
                        minimum_advantage=advantage,
                    )
                    for seed in seeds
                }
                signature = _signature(candidate_by_seed)
                if signature in unique_by_signature:
                    aliases[name] = unique_by_signature[signature]
                else:
                    unique_by_signature[signature] = name
                    aliases[name] = name
                    kept[name] = candidate_by_seed
                raw_count += 1
                if args.max_raw_candidates is not None and raw_count >= args.max_raw_candidates:
                    stop = True
                    break
            if stop:
                break
        predictions[dataset] = kept
        aliases_by_dataset[dataset] = aliases
        grid_counts[dataset] = {"raw": raw_count, "unique": len(kept)}
        output_hashes[dataset] = {}
        for name, candidate_by_seed in kept.items():
            output_hashes[dataset][name] = {}
            for seed, rows in candidate_by_seed.items():
                relative = Path("predictions") / dataset / name / f"seed_{seed}.jsonl"
                digest = write_selections(args.output_dir / relative, rows)
                output_hashes[dataset][name][str(seed)] = {
                    "path": str(relative),
                    "sha256": digest,
                }
    write_json(args.output_dir / "source_expert_profiles.json", profiles_by_dataset)
    write_json(args.output_dir / "candidate_aliases.json", aliases_by_dataset)

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [*sorted(input_paths), *sorted(manifest_paths)],
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "grid_counts": grid_counts,
        "predictions": output_hashes,
        "source_labels_used_for_conditioned_profiles": True,
        "target_labels_opened_during_prediction": False,
        "base_mapping_is_known_development_posthoc": True,
    }
    prediction_manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_target_labels": sha256_file(prediction_manifest_path),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "target_labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    evaluation_rows: list[dict[str, Any]] = []
    for dataset, by_method in predictions.items():
        labels = _load_labels(jobs[dataset], None)
        for method, by_seed in by_method.items():
            for seed, rows in by_seed.items():
                evaluation_rows.append(
                    _score(
                        dataset,
                        seed,
                        method,
                        rows,
                        bases[dataset][seed],
                        fcrg_rows[dataset][seed],
                        labels,
                    )
                )
    write_csv(args.output_dir / "candidate_results.csv", evaluation_rows)
    aggregate_rows: list[dict[str, Any]] = []
    best_by_dataset: dict[str, dict[str, Any]] = {}
    threshold = float(config["acceptance"]["minimum_delta_vs_fcrg"])
    for dataset, by_method in predictions.items():
        for method in by_method:
            rows = [
                row for row in evaluation_rows
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
                    "base_accuracy_mean": float(np.mean([row["base_accuracy"] for row in rows])),
                    "fcrg_accuracy_mean": float(np.mean([row["fcrg_accuracy"] for row in rows])),
                    "delta_vs_base_mean": float(np.mean([row["delta_vs_base"] for row in rows])),
                    "delta_vs_fcrg_mean": float(np.mean([row["delta_vs_fcrg"] for row in rows])),
                    "delta_vs_fcrg_min_seed": float(np.min([row["delta_vs_fcrg"] for row in rows])),
                    "meets_large_gain": bool(
                        min(row["delta_vs_fcrg"] for row in rows) + 1e-12 >= threshold
                    ),
                }
            )
        ranked = sorted(
            (row for row in aggregate_rows if row["dataset"] == dataset),
            key=lambda row: (
                row["delta_vs_fcrg_mean"],
                row["delta_vs_base_mean"],
                row["method"],
            ),
            reverse=True,
        )
        best_by_dataset[dataset] = ranked[0]
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "best_by_dataset_posthoc_development_diagnostic": best_by_dataset,
            "all_targets_meet_large_gain": all(
                bool(row["meets_large_gain"]) for row in best_by_dataset.values()
            ),
            "minimum_delta_vs_fcrg": threshold,
            "target_label_firewall": (
                "Source labels fit subject-conditioned expert profiles. All target predictions "
                "were written and hashed before target labels were opened."
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
