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
    read_selections,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .data import CacheAdapter, load_family_map
from .evaluation import exact_mcnemar, paired_bootstrap_delta, selection_correctness
from .gpqa_permutation_consensus import (
    PermutationConsensusVariant,
    gpqa_permutation_consensus,
)
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows
from .selectors import source_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-free semantic consensus across GPQA choice permutations"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def _source(config: Mapping[str, Any]) -> tuple[Any, Any, Mapping[str, Any]]:
    source_config = yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))
    source = source_config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source["cache_path"]),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        load_family_map(Path(source_config["family_map"])),
        [str(value) for value in config["experts"]],
        Path(source_config["dataset_registry"]),
        str(source_config["dataset_registry_sha256"]),
    )
    return adapter.load_observables(), adapter.load_source_labels(), source_config


def _target(config: Mapping[str, Any], limit: int | None) -> Any:
    target = config["targets"][0]
    adapter = CacheAdapter.from_target_observables(
        Path(target["observable_cache_path"]),
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        load_family_map(Path(config["family_map"])),
        [str(value) for value in config["experts"]],
        str(target["observable_manifest_sha256"]),
    )
    batch = adapter.load_observables(limit=limit)
    expected = min(int(target["expected_questions"]), limit or int(target["expected_questions"]))
    if len(batch.question_ids) != expected:
        raise RuntimeError("GPQA target observable count differs from the pinned config")
    return batch


def _authenticated_v2(
    root: Path,
    dataset: str,
    seeds: list[int],
    authenticated: dict[str, str],
) -> tuple[dict[int, list[Any]], set[Path]]:
    prediction_path = root / "prediction_manifest.json"
    boundary_path = root / "prediction_boundary.json"
    complete_path = root / "complete_manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    prediction_hash = sha256_file(prediction_path)
    expected_manifest_hash = str(complete["prediction_manifest_sha256_before_labels"])
    if prediction_hash != expected_manifest_hash:
        raise RuntimeError("V2 prediction manifest is not completion-bound")
    if str(boundary["prediction_manifest_sha256_before_labels"]) != prediction_hash:
        raise RuntimeError("V2 prediction manifest is not boundary-bound")
    if prediction.get("target_labels_opened_during_prediction") is not False:
        raise RuntimeError("V2 input does not attest to the target-label firewall")

    result: dict[int, list[Any]] = {}
    for seed in seeds:
        entry = prediction["predictions"][dataset][str(seed)]
        path = root / str(entry["path"])
        expected = str(entry["sha256"])
        if complete["artifact_hashes"].get(str(path)) != expected:
            raise RuntimeError(f"V2 prediction is not completion-bound: {path}")
        if boundary["prediction_files_sha256"].get(str(path)) != expected:
            raise RuntimeError(f"V2 prediction is not boundary-bound: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"V2 prediction hash mismatch: {path}")
        authenticated[str(path)] = actual
        result[seed] = sorted(read_selections(path), key=lambda row: row.question_id)
    return result, {prediction_path, boundary_path, complete_path}


def _variants(config: Mapping[str, Any]) -> list[PermutationConsensusVariant]:
    grid = config["candidate_grid"]
    result: list[PermutationConsensusVariant] = []
    for mode in grid["vote_modes"]:
        for source_power in grid["source_powers"]:
            for consistency_power in grid["consistency_powers"]:
                for family in grid["family_balance"]:
                    for share, advantage in grid["gates"]:
                        token = lambda value: str(float(value)).replace(".", "p")
                        name = (
                            f"permcons__{mode}__sp{token(source_power)}__"
                            f"cp{token(consistency_power)}__fam{int(bool(family))}__"
                            f"share{token(share)}__adv{token(advantage)}"
                        )
                        result.append(
                            PermutationConsensusVariant(
                                name,
                                str(mode),
                                float(source_power),
                                float(consistency_power),
                                bool(family),
                                float(share),
                                float(advantage),
                            )
                        )
    return result


def _signature(rows_by_seed: Mapping[int, list[Any]]) -> tuple[Any, ...]:
    return tuple(
        (seed, row.question_id, row.normalized_answer, row.selected_expert_id)
        for seed, rows in sorted(rows_by_seed.items())
        for row in rows
    )


def _comparison(candidate: list[Any], reference: list[Any], labels: Any, seed: int, samples: int):
    candidate_map = selection_correctness(candidate, labels)
    reference_map = selection_correctness(reference, labels)
    if set(candidate_map) != set(reference_map):
        raise RuntimeError("GPQA candidate/reference IDs differ")
    ids = sorted(candidate_map)
    left = np.asarray([candidate_map[qid] for qid in ids], dtype=np.int8)
    right = np.asarray([reference_map[qid] for qid in ids], dtype=np.int8)
    rescue, harm, p_value = exact_mcnemar(left, right)
    ci = paired_bootstrap_delta(left, right, seed=seed, samples=samples)
    return {
        "samples": len(ids),
        "candidate_accuracy": float(left.mean()),
        "reference_accuracy": float(right.mean()),
        "delta": float((left - right).mean()),
        "rescue_count": rescue,
        "harm_count": harm,
        "exact_mcnemar_p": p_value,
        "paired_bootstrap_delta_ci95": list(ci),
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

    source_batch, source_labels, source_config = _source(config)
    accuracies = source_accuracy(source_batch, source_labels)
    target_batch = _target(config, args.max_questions)
    target_name = str(config["targets"][0]["name"])
    jobs = _dataset_jobs(
        yaml.safe_load(Path(config["portfolio_config"]).read_text(encoding="utf-8"))
    )
    job = jobs[target_name]
    seeds = list(job["seeds"])
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    authenticated: dict[str, str] = {}
    v2, v2_manifests = _authenticated_v2(
        Path(config["v2_run_root"]), target_name, seeds, authenticated
    )
    prior_art_manifests: set[Path] = set()
    fcrg: dict[int, list[Any]] = {}
    for seed in seeds:
        rows = _prior_art_rows(job, seed, "fcrg_full", authenticated, prior_art_manifests)
        if args.max_questions is not None:
            rows = rows[: args.max_questions]
            v2[seed] = v2[seed][: args.max_questions]
        if set(row.question_id for row in rows) != set(target_batch.question_ids):
            raise RuntimeError(f"GPQA FCRG IDs differ for seed {seed}")
        if set(row.question_id for row in v2[seed]) != set(target_batch.question_ids):
            raise RuntimeError(f"GPQA V2 IDs differ for seed {seed}")
        fcrg[seed] = rows

    variants = _variants(config)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    kept: dict[str, dict[int, list[Any]]] = {}
    aliases: dict[str, str] = {}
    seen: dict[tuple[Any, ...], str] = {}
    for variant in variants:
        by_seed = {
            seed: gpqa_permutation_consensus(
                target_batch, accuracies, v2[seed], variant
            )
            for seed in seeds
        }
        signature = _signature(by_seed)
        canonical = seen.get(signature)
        if canonical is None:
            seen[signature] = variant.name
            canonical = variant.name
            kept[canonical] = by_seed
        aliases[variant.name] = canonical

    output_hashes: dict[str, dict[str, str]] = {}
    for method, by_seed in kept.items():
        signatures = {
            tuple(
                (row.question_id, row.normalized_answer, row.selected_expert_id)
                for row in rows
            )
            for rows in by_seed.values()
        }
        output_hashes[method] = {}
        if len(signatures) == 1:
            relative = Path("predictions") / method / "all_seeds.jsonl"
            digest = write_selections(args.output_dir / relative, next(iter(by_seed.values())))
            for seed in seeds:
                output_hashes[method][str(seed)] = {
                    "path": str(relative),
                    "sha256": digest,
                }
        else:
            for seed, rows in by_seed.items():
                relative = Path("predictions") / method / f"seed_{seed}.jsonl"
                digest = write_selections(args.output_dir / relative, rows)
                output_hashes[method][str(seed)] = {
                    "path": str(relative),
                    "sha256": digest,
                }

    write_json(args.output_dir / "candidate_aliases.json", aliases)
    input_paths = {
        args.config,
        Path(config["source_config"]),
        Path(config["portfolio_config"]),
        Path(config["family_map"]),
        Path(config["dataset_registry"]),
        Path(source_config["source"]["cache_path"]),
        Path(config["targets"][0]["observable_cache_path"]),
        *v2_manifests,
        *prior_art_manifests,
    }
    environment = environment_manifest(
        sys.argv, int(config["protocol_seed"]), sorted(input_paths)
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "raw_candidate_count": len(variants),
        "unique_candidate_count": len(kept),
        "source_accuracy_by_expert": dict(sorted(accuracies.items())),
        "predictions": {target_name: output_hashes},
        "target_labels_opened_during_prediction": False,
        "target_observables_physically_label_free": True,
        "candidate_grid_selected_posthoc_only_after_joint_prediction_boundary": True,
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

    # Evaluation starts after every grid candidate has been written and boundary-bound.
    labels = _load_labels(job, args.max_questions)
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for method, by_seed in kept.items():
        method_rows: list[dict[str, Any]] = []
        for seed in seeds:
            vs_fcrg = _comparison(
                by_seed[seed],
                fcrg[seed],
                labels,
                int(config["protocol_seed"]) + seed,
                int(config["bootstrap_samples"]),
            )
            vs_v2 = _comparison(
                by_seed[seed],
                v2[seed],
                labels,
                int(config["protocol_seed"]) + seed + 100000,
                int(config["bootstrap_samples"]),
            )
            row = {
                "dataset": target_name,
                "method": method,
                "seed": seed,
                "samples": vs_fcrg["samples"],
                "accuracy": vs_fcrg["candidate_accuracy"],
                "fcrg_accuracy": vs_fcrg["reference_accuracy"],
                "v2_accuracy": vs_v2["reference_accuracy"],
                "delta_vs_fcrg": vs_fcrg["delta"],
                "delta_vs_v2": vs_v2["delta"],
                "rescue_vs_fcrg": vs_fcrg["rescue_count"],
                "harm_vs_fcrg": vs_fcrg["harm_count"],
                "exact_mcnemar_p_vs_fcrg": vs_fcrg["exact_mcnemar_p"],
                "paired_bootstrap_ci95_vs_fcrg": vs_fcrg[
                    "paired_bootstrap_delta_ci95"
                ],
            }
            method_rows.append(row)
            seed_rows.append(row)
        minimum = float(config["acceptance"]["minimum_delta_vs_fcrg"])
        aggregate_rows.append(
            {
                "dataset": target_name,
                "method": method,
                "seed_count": len(method_rows),
                "samples_per_seed": method_rows[0]["samples"],
                "accuracy_mean": float(np.mean([row["accuracy"] for row in method_rows])),
                "accuracy_std": float(np.std([row["accuracy"] for row in method_rows])),
                "fcrg_accuracy_mean": float(
                    np.mean([row["fcrg_accuracy"] for row in method_rows])
                ),
                "v2_accuracy_mean": float(
                    np.mean([row["v2_accuracy"] for row in method_rows])
                ),
                "delta_vs_fcrg_mean": float(
                    np.mean([row["delta_vs_fcrg"] for row in method_rows])
                ),
                "delta_vs_v2_mean": float(
                    np.mean([row["delta_vs_v2"] for row in method_rows])
                ),
                "minimum_seed_delta_vs_fcrg": min(
                    float(row["delta_vs_fcrg"]) for row in method_rows
                ),
                "meets_large_gain": bool(
                    min(float(row["delta_vs_fcrg"]) for row in method_rows) > minimum
                    and min(float(row["delta_vs_v2"]) for row in method_rows) >= -1e-12
                ),
            }
        )
    ranked = sorted(
        aggregate_rows,
        key=lambda row: (
            bool(row["meets_large_gain"]),
            float(row["accuracy_mean"]),
            float(row["delta_vs_v2_mean"]),
            str(row["method"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    write_csv(args.output_dir / "seed_results.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "best_posthoc_development_candidate": best,
            "all_targets_pass": bool(best["meets_large_gain"]),
            "minimum_delta_vs_fcrg": config["acceptance"]["minimum_delta_vs_fcrg"],
            "target_label_firewall": (
                "All cross-permutation grid predictions were written and hashed together before "
                "the GPQA evaluation labels were opened. Candidate ranking is explicitly a "
                "known-development posthoc diagnostic."
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
