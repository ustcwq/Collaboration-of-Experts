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
from .evaluation import exact_mcnemar, selection_correctness
from .method_consensus import ConsensusVariant, apply_consensus_gate, consensus_selections
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine frozen development components with target-label-free consensus gates"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-target-seeds", type=int)
    parser.add_argument("--max-schemes", type=int)
    parser.add_argument("--max-raw-candidates", type=int)
    return parser.parse_args()


def _seed_dir(job: Mapping[str, Any], seed: int) -> Path:
    matches = sorted(Path(job["run_root"]).glob(f"seed_{seed}_gpu*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one seed directory for {job['name']}/{seed}")
    return matches[0]


def _full_pool_methods(job: Mapping[str, Any], seed: int, expected: int) -> tuple[str, ...]:
    manifest = json.loads(
        (_seed_dir(job, seed) / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    target = manifest if job["package_scope"] == "source" else manifest["targets"][job["name"]]
    methods = tuple(
        sorted(method for method, group in target["method_group"].items() if group == "full_pool")
    )
    if len(methods) != expected:
        raise RuntimeError(f"Unexpected full-pool method count: {job['name']}/{seed}/{len(methods)}")
    return methods


def _prediction_signature(rows_by_seed: Mapping[int, list[Any]]) -> tuple[Any, ...]:
    return tuple(
        (
            seed,
            row.question_id,
            row.normalized_answer,
            row.selected_expert_id,
        )
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
    maps = [selection_correctness(rows, labels) for rows in (candidate, base, fcrg)]
    if not (set(maps[0]) == set(maps[1]) == set(maps[2])):
        raise RuntimeError(f"Guarded candidate IDs differ: {dataset}/{seed}/{method}")
    ids = sorted(maps[0])
    values = [np.asarray([mapping[qid] for qid in ids], dtype=np.int8) for mapping in maps]
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
    jobs = _dataset_jobs(yaml.safe_load(portfolio_path.read_text(encoding="utf-8")))
    expected = int(config["expected_full_pool_methods"])
    fcrg_method = str(config["fcrg_method"])
    schemes = [
        (str(subset), str(weighting))
        for subset in config["candidate_grid"]["subsets"]
        for weighting in config["candidate_grid"]["global_weightings"]
    ]
    if args.max_schemes is not None:
        schemes = schemes[: args.max_schemes]
    gates = [
        (float(share), float(advantage))
        for share in config["candidate_grid"]["fallback_shares"]
        for advantage in config["candidate_grid"]["minimum_advantages"]
    ]
    if not schemes or not gates:
        raise ValueError("Guarded consensus grid is empty")

    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = set()
    predictions: dict[str, dict[str, dict[int, list[Any]]]] = {}
    bases: dict[str, dict[int, list[Any]]] = {}
    fcrg_rows: dict[str, dict[int, list[Any]]] = {}
    output_hashes: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    alias_maps: dict[str, dict[str, str]] = {}
    grid_counts: dict[str, dict[str, int]] = {}
    for dataset, base_method_raw in config["base_methods"].items():
        dataset = str(dataset)
        base_method = str(base_method_raw)
        job = jobs[dataset]
        seeds = list(job["seeds"])
        if args.max_target_seeds is not None:
            seeds = seeds[: args.max_target_seeds]
        rows_by_seed: dict[int, dict[str, list[Any]]] = {}
        bases[dataset] = {}
        fcrg_rows[dataset] = {}
        for seed in seeds:
            methods = _full_pool_methods(job, seed, expected)
            rows_by_seed[seed] = {}
            for method in methods:
                rows = _prior_art_rows(job, seed, method, authenticated, manifest_paths)
                if args.max_questions is not None:
                    rows = rows[: args.max_questions]
                rows_by_seed[seed][method] = rows
            bases[dataset][seed] = rows_by_seed[seed][base_method]
            fcrg_rows[dataset][seed] = rows_by_seed[seed][fcrg_method]

        unique_by_signature: dict[tuple[Any, ...], str] = {}
        kept: dict[str, dict[int, list[Any]]] = {}
        aliases: dict[str, str] = {}
        raw_count = 0
        stop = False
        for subset, weighting in schemes:
            ungated_by_seed = {
                seed: consensus_selections(
                    rows,
                    ConsensusVariant(
                        name=f"gcons_raw__{subset}__{weighting}",
                        subset=subset,
                        global_weighting=weighting,
                    ),
                    reference_method=base_method,
                )
                for seed, rows in rows_by_seed.items()
            }
            for share, advantage in gates:
                share_name = str(share).replace(".", "p")
                advantage_name = str(advantage).replace(".", "p")
                name = (
                    f"gcons__{base_method}__{subset}__{weighting}__"
                    f"share{share_name}__adv{advantage_name}"
                )
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
                signature = _prediction_signature(candidate_by_seed)
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
        alias_maps[dataset] = aliases
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
    write_json(args.output_dir / "candidate_aliases.json", alias_maps)

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
        "base_methods": config["base_methods"],
        "fcrg_method": fcrg_method,
        "grid_counts": grid_counts,
        "predictions": output_hashes,
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
            "protocol_boundary": (
                "The v1 base-method mapping is development-posthoc. Every gate prediction "
                "was generated without target labels and hashed before evaluation."
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
