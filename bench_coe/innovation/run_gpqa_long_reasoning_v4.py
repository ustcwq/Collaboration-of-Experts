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
from .data import CacheAdapter, load_family_map
from .evaluation import exact_mcnemar, hierarchical_paired_bootstrap, selection_correctness
from .gpqa_long_reasoning import (
    apply_semantic_reasoning_overrides,
    semantic_answer_by_query,
)
from .run_large_gain_portfolio_v3 import _component_rows, _run_manifests
from .run_strict_positive_portfolio import _dataset_jobs, _load_labels, _prior_art_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay source-selected label-free long reasoning on frozen GPQA V3"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-seeds", type=int)
    return parser.parse_args()


def evaluate_acceptance(
    accuracy: float, minimum_delta: float, fcrg_accuracy: float, v3_floor: float
) -> dict[str, Any]:
    delta = accuracy - fcrg_accuracy
    passed = delta + 1e-12 >= minimum_delta and accuracy + 1e-12 >= v3_floor
    return {
        "passed": bool(passed),
        "minimum_delta_vs_fcrg_at_least": minimum_delta,
        "delta_vs_fcrg": delta,
        "at_least_large_gain_threshold": bool(delta + 1e-12 >= minimum_delta),
        "v3_accuracy_floor": v3_floor,
        "does_not_regress_v3": bool(accuracy + 1e-12 >= v3_floor),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _authenticated_inference(
    config: Mapping[str, Any], authenticated: dict[str, str]
) -> tuple[list[dict[str, Any]], str, set[Path]]:
    inference = config["inference"]
    root = Path(inference["run_root"])
    manifest_path = root / "prediction_manifest.json"
    complete_path = root / "complete_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != str(inference["prediction_manifest_sha256"]):
        raise PermissionError("GPQA inference manifest differs from the frozen config")
    if manifest_digest != str(complete["prediction_manifest_sha256_before_target_labels"]):
        raise RuntimeError("GPQA inference manifest is not completion-bound")
    if manifest.get("target_labels_opened_during_prediction") is not False:
        raise RuntimeError("GPQA inference lacks a target-label firewall attestation")
    if manifest.get("protocol") != inference["protocol"]:
        raise RuntimeError("GPQA inference protocol mismatch")
    if manifest.get("model") != inference["model_id"]:
        raise RuntimeError("GPQA inference model mismatch")
    prediction_path = root / "predictions.jsonl"
    prediction_digest = sha256_file(prediction_path)
    if prediction_digest != str(manifest["prediction_sha256"]):
        raise RuntimeError("GPQA inference prediction hash mismatch")
    if prediction_digest != str(complete["prediction_file_sha256"]):
        raise RuntimeError("GPQA inference prediction is not completion-bound")
    expected = int(inference["expected_questions"])
    rows = _read_jsonl(prediction_path)
    if len(rows) != expected:
        raise RuntimeError(f"GPQA inference has {len(rows)} rows; expected {expected}")
    authenticated[str(manifest_path)] = manifest_digest
    authenticated[str(prediction_path)] = prediction_digest
    authenticated[str(complete_path)] = sha256_file(complete_path)
    return rows, manifest_digest, {manifest_path, prediction_path, complete_path}


def _load_gold(config: Mapping[str, Any], limit: int | None) -> dict[str, str]:
    target = config["target_labels"]
    path = Path(target["cache_path"]) / str(target["label_expert_id"]) / "predictions.jsonl"
    rows = sorted(_read_jsonl(path), key=lambda row: str(row["id"]))
    if limit is not None:
        rows = rows[:limit]
    return {
        f"{target['dataset']}::{target['split']}::{row['id']}": str(row["answer"]).strip().lower()
        for row in rows
    }


def _candidate_correctness(
    rows: list[Any], labels: Any, gold: Mapping[str, str], model_id: str
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for row in rows:
        if row.selected_expert_id == model_id:
            result[row.question_id] = (
                str(row.normalized_answer or "").strip().lower() == gold[row.question_id]
            )
        else:
            result[row.question_id] = bool(
                labels.get(row.question_id, row.selected_expert_id)
                if row.selected_expert_id is not None
                else False
            )
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

    target = config["target_labels"]
    experts = [str(value) for value in config["experts"]]
    observable = config["target_observables"]
    batch = CacheAdapter.from_target_observables(
        Path(observable["cache_path"]),
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        load_family_map(Path(config["family_map"])),
        experts,
        str(observable["observable_manifest_sha256"]),
    ).load_observables(limit=args.max_questions)
    expected = min(
        int(target["expected_questions"]),
        args.max_questions or int(target["expected_questions"]),
    )
    if len(batch.question_ids) != expected:
        raise RuntimeError("GPQA observable count differs from the frozen config")
    metadata_by_question = {
        question_id: dict(batch.for_question(question_id)[0].observable_metadata)
        for question_id in batch.question_ids
    }

    portfolio_path = Path(config["portfolio_config"])
    jobs = _dataset_jobs(yaml.safe_load(portfolio_path.read_text(encoding="utf-8")))
    dataset = str(target["name"])
    job = jobs[dataset]
    seeds = list(job["seeds"])
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    authenticated: dict[str, str] = {}
    inference_rows, inference_manifest_sha, inference_paths = _authenticated_inference(
        config, authenticated
    )
    semantic, semantic_query_audit = semantic_answer_by_query(inference_rows)
    model_id = str(config["inference"]["model_id"])
    v3_root = Path(config["v3_run_root"])
    v3_prediction, v3_boundary, v3_complete, v3_manifest_paths = _run_manifests(v3_root)
    manifest_paths = set(v3_manifest_paths)
    candidates: dict[int, list[Any]] = {}
    v3_rows: dict[int, list[Any]] = {}
    fcrg_rows: dict[int, list[Any]] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    override_counts: dict[str, int] | None = None
    for seed in seeds:
        prior_v3 = _component_rows(
            v3_root,
            v3_prediction,
            v3_boundary,
            v3_complete,
            dataset,
            seed,
            None,
            authenticated,
        )
        fcrg = _prior_art_rows(job, seed, "fcrg_full", authenticated, manifest_paths)
        if args.max_questions is not None:
            prior_v3 = prior_v3[: args.max_questions]
            fcrg = fcrg[: args.max_questions]
        candidate, counts = apply_semantic_reasoning_overrides(
            prior_v3,
            metadata_by_question,
            semantic,
            model_id=model_id,
            inference_manifest_sha256=inference_manifest_sha,
        )
        if override_counts is None:
            override_counts = counts
        elif counts != override_counts:
            raise RuntimeError("GPQA override coverage differs across deterministic seeds")
        ids = {row.question_id for row in candidate}
        if ids != set(batch.question_ids) or ids != {row.question_id for row in fcrg}:
            raise RuntimeError(f"GPQA candidate/reference IDs differ for seed {seed}")
        candidates[seed] = candidate
        v3_rows[seed] = prior_v3
        fcrg_rows[seed] = fcrg
        relative = Path("predictions") / dataset / f"seed_{seed}.jsonl"
        digest = write_selections(args.output_dir / relative, candidate)
        output_hashes[str(seed)] = {"path": str(relative), "sha256": digest}

    input_paths = {
        args.config,
        portfolio_path,
        Path(config["family_map"]),
        Path(config["dataset_registry"]),
        Path(observable["cache_path"]),
        *inference_paths,
        *manifest_paths,
    }
    environment = environment_manifest(
        sys.argv, int(config["protocol_seed"]), sorted(input_paths)
    )
    environment["authenticated_reference_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "model": model_id,
        "source_selection": config["source_selection"],
        "inference_prediction_manifest_sha256": inference_manifest_sha,
        "valid_semantic_predictions": len(semantic),
        "semantic_query_audit": semantic_query_audit,
        "override_counts": override_counts or {},
        "fallback": "frozen_v3_prediction",
        "predictions": {dataset: output_hashes},
        "target_labels_opened_during_prediction": False,
        "target_observables_physically_label_free": True,
    }
    prediction_path = args.output_dir / "prediction_manifest.json"
    write_json(prediction_path, prediction_manifest)
    boundary = {
        "prediction_manifest_sha256_before_target_labels": sha256_file(prediction_path),
        "prediction_files_sha256": {
            str(path): sha256_file(path)
            for path in sorted((args.output_dir / "predictions").rglob("*.jsonl"))
        },
        "target_labels_opened": False,
        "boundary_created_unix": time.time(),
    }
    write_json(args.output_dir / "prediction_boundary.json", boundary)

    # Isolated evaluation starts only after every candidate prediction is hash-bound.
    labels = _load_labels(job, args.max_questions)
    gold = _load_gold(config, args.max_questions)
    if set(gold) != set(batch.question_ids):
        raise RuntimeError("GPQA gold and observable IDs differ")

    seed_rows: list[dict[str, Any]] = []
    candidate_matrix: list[np.ndarray] = []
    fcrg_matrix: list[np.ndarray] = []
    v3_matrix: list[np.ndarray] = []
    for seed in seeds:
        candidate_map = _candidate_correctness(candidates[seed], labels, gold, model_id)
        fcrg_map = selection_correctness(fcrg_rows[seed], labels)
        prior_map = selection_correctness(v3_rows[seed], labels)
        ids = sorted(candidate_map)
        left = np.asarray([candidate_map[qid] for qid in ids], dtype=np.int8)
        right = np.asarray([fcrg_map[qid] for qid in ids], dtype=np.int8)
        prior = np.asarray([prior_map[qid] for qid in ids], dtype=np.int8)
        rescue, harm, p_value = exact_mcnemar(left, right)
        v3_rescue, v3_harm, v3_p = exact_mcnemar(left, prior)
        candidate_matrix.append(left)
        fcrg_matrix.append(right)
        v3_matrix.append(prior)
        seed_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "samples": len(left),
                "accuracy": float(left.mean()),
                "fcrg_full_accuracy": float(right.mean()),
                "v3_accuracy": float(prior.mean()),
                "delta_vs_fcrg_full": float((left - right).mean()),
                "delta_vs_v3": float((left - prior).mean()),
                "rescue_count_vs_fcrg": rescue,
                "harm_count_vs_fcrg": harm,
                "exact_mcnemar_p_vs_fcrg": p_value,
                "rescue_count_vs_v3": v3_rescue,
                "harm_count_vs_v3": v3_harm,
                "exact_mcnemar_p_vs_v3": v3_p,
            }
        )

    candidate_array = np.stack(candidate_matrix)
    fcrg_array = np.stack(fcrg_matrix)
    v3_array = np.stack(v3_matrix)
    aggregate = {
        "dataset": dataset,
        "seed_count": candidate_array.shape[0],
        "samples_per_seed": candidate_array.shape[1],
        "accuracy_mean": float(candidate_array.mean()),
        "accuracy_std": float(np.std(candidate_array.mean(axis=1))),
        "fcrg_full_accuracy_mean": float(fcrg_array.mean()),
        "v3_accuracy_mean": float(v3_array.mean()),
        "delta_vs_fcrg_full_mean": float((candidate_array - fcrg_array).mean()),
        "delta_vs_v3_mean": float((candidate_array - v3_array).mean()),
        "minimum_seed_delta_vs_fcrg": float(
            np.min((candidate_array - fcrg_array).mean(axis=1))
        ),
        "hierarchical_paired_bootstrap_delta_vs_fcrg_ci95": list(
            hierarchical_paired_bootstrap(
                candidate_array,
                fcrg_array,
                int(config["protocol_seed"]),
                samples=int(config["bootstrap_samples"]),
            )
        ),
    }
    acceptance = evaluate_acceptance(
        float(aggregate["accuracy_mean"]),
        float(config["acceptance"]["minimum_delta_vs_fcrg"]),
        float(aggregate["fcrg_full_accuracy_mean"]),
        float(config["acceptance"]["v3_accuracy_floor"]),
    )
    write_csv(args.output_dir / "seed_results.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_results.csv", [aggregate])
    write_json(args.output_dir / "acceptance.json", acceptance)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "results": {dataset: aggregate},
            "acceptance": acceptance,
            "all_targets_pass": bool(acceptance["passed"]),
            "target_label_firewall": (
                "The source-selected reasoner consumed only the question and option text. Its "
                "semantic answers were permutation-mapped and all candidate files were hashed "
                "before GPQA answers or expert correctness were opened."
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
            "all_targets_pass": bool(acceptance["passed"]),
        },
    )


if __name__ == "__main__":
    main()
