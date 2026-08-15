from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    environment_manifest,
    manifest_sha256,
    sha256_file,
    validate_test_receipt,
    write_json,
    write_selections,
)
from .data import CacheAdapter, assert_disjoint, load_family_map
from .locked_protocol import load_protocol, validate_preregistration, validate_protocol
from .prior_art_targets import target_environment_by_question
from .response_embeddings import MiniLMResponseEncoder
from .run_prior_art_overlap import configure_device, method_cost, retag, run_fold_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen selectors without opening MuSR labels")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_protocol(args.config)
    validate_protocol(config, verify_files=False)
    validate_test_receipt(Path(str(config["test_receipt"])), args.config)
    run_root = Path(str(config["output_root"]))
    prereg = validate_preregistration(args.config, run_root)
    if (run_root / "label_access_started.json").exists():
        raise RuntimeError("Target labels were already opened; predictions are permanently frozen")
    target_cache = run_root / "target_observables"
    target_manifest_path = target_cache / "observable_manifest.json"
    if not target_manifest_path.exists():
        raise FileNotFoundError("All frozen expert generations must finish before selection")
    target_manifest_hash = sha256_file(target_manifest_path)

    selector = config["selector"]
    seed = int(selector["seed"])
    device, device_manifest = configure_device(seed, args.physical_gpu)
    family_map_path = Path(str(config["family_map"]))
    family_map = load_family_map(family_map_path)
    experts = [str(value) for value in config["experts"]]
    source = config["source"]
    source_adapter = CacheAdapter.from_source_registry(
        Path(str(source["cache_path"])),
        str(source["dataset"]),
        str(source["split"]),
        str(source["modality"]),
        family_map,
        experts,
        Path(str(config["source_registry"])),
        str(config["source_registry_sha256"]),
    )
    source_batch = source_adapter.load_observables()
    source_labels = source_adapter.load_source_labels()
    if len(source_batch.question_ids) != int(source["expected_questions"]):
        raise RuntimeError("Frozen source question count mismatch")
    target_batch = CacheAdapter.from_target_observables(
        target_cache,
        str(config["target"]["dataset"]),
        str(config["target"]["split"]),
        str(config["target"]["modality"]),
        family_map,
        experts,
        target_manifest_hash,
    ).load_observables()
    if len(target_batch.question_ids) != int(config["target"]["expected_questions"]):
        raise RuntimeError("Frozen target question count mismatch")
    assert_disjoint(source_batch, target_batch)

    encoder_config = selector["response_embedding"]
    response_encoder = MiniLMResponseEncoder(
        str(encoder_config["model_id"]),
        device,
        batch_size=int(encoder_config["batch_size"]),
        max_length=int(encoder_config["max_length"]),
    )
    run_config: dict[str, Any] = {
        "knn_k": int(selector["knn_k"]),
        "mcb_behavior_threshold": float(selector["mcb_behavior_threshold"]),
        "mcb_min_neighbors": int(selector["mcb_min_neighbors"]),
    }
    started = time.time()
    all_predictions, diagnostics = run_fold_pool(
        source_batch,
        source_labels,
        target_batch,
        config=run_config,
        seed=seed,
        fold_index=0,
        device=device,
        full_protocol=False,
        max_inner_environments=None,
        response_encoder=response_encoder,
    )
    missing_methods = sorted(
        str(method["internal"])
        for method in config["methods"].values()
        if str(method["internal"]) not in all_predictions
    )
    if missing_methods:
        raise RuntimeError(f"Preregistered selector outputs are missing: {missing_methods}")
    prediction_dir = run_root / "selection_predictions"
    if prediction_dir.exists():
        raise FileExistsError(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    costs: list[dict[str, Any]] = []
    method_mapping: dict[str, str] = {}
    expected_ids = set(target_batch.question_ids)
    for public_name, method in config["methods"].items():
        internal_name = str(method["internal"])
        if internal_name not in all_predictions:
            raise RuntimeError(f"Preregistered method was not produced: {internal_name}")
        selections = retag(all_predictions[internal_name], public_name)
        if {selection.question_id for selection in selections} != expected_ids:
            raise RuntimeError(f"Method {public_name} does not cover all frozen questions")
        path = prediction_dir / f"{public_name}.jsonl"
        hashes[public_name] = write_selections(path, selections)
        paths[public_name] = str(path.relative_to(run_root))
        cost = method_cost(internal_name, selections, target_batch)
        cost["method"] = public_name
        cost["preregistered_nominal_model_calls"] = int(method["nominal_model_calls"])
        costs.append(cost)
        method_mapping[public_name] = internal_name

    run_environment = environment_manifest(
        sys.argv,
        seed,
        [
            args.config,
            run_root / "preregistration.json",
            target_manifest_path,
            family_map_path,
            Path(str(config["source_registry"])),
            Path(str(source["cache_path"])),
        ],
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "all_preregistered_predictions_complete_before_evaluation",
        "seed": seed,
        "physical_gpu": args.physical_gpu,
        "questions": len(target_batch.question_ids),
        "experts": experts,
        "method_mapping": method_mapping,
        "prediction_paths": paths,
        "prediction_hashes_before_evaluation": hashes,
        "target_observable_manifest_sha256": target_manifest_hash,
        "preregistration_sha256": sha256_file(run_root / "preregistration.json"),
        "test_receipt_sha256": sha256_file(Path(str(config["test_receipt"]))),
        "innovation_code_manifest_sha256": run_environment[
            "innovation_code_manifest_sha256"
        ],
        "target_environment_counts": {
            environment: list(target_environment_by_question(target_batch).values()).count(environment)
            for environment in sorted(set(target_environment_by_question(target_batch).values()))
        },
        "costs": costs,
        "diagnostics": diagnostics,
        "response_encoder": response_encoder.diagnostics(),
        "environment": run_environment,
        "started_unix": started,
        "finished_unix": time.time(),
        "target_labels_opened": False,
    }
    manifest_path = run_root / "selection_manifest.json"
    write_json(manifest_path, manifest)
    seal = {
        "experiment_id": config["experiment_id"],
        "status": "prediction_boundary_sealed",
        "sealed_unix": time.time(),
        "protocol_sha256": prereg["protocol_sha256"],
        "preregistration_sha256": sha256_file(run_root / "preregistration.json"),
        "selection_manifest_sha256": sha256_file(manifest_path),
        "target_observable_manifest_sha256": target_manifest_hash,
        "prediction_hashes": hashes,
        "prediction_manifest_sha256": manifest_sha256(hashes),
        "methods": list(config["methods"]),
        "questions": len(target_batch.question_ids),
        "target_labels_opened": False,
        "mutable_after_seal": False,
    }
    seal_path = run_root / "prediction_seal.json"
    if seal_path.exists():
        raise FileExistsError(seal_path)
    write_json(seal_path, seal)
    print(f"Prediction boundary sealed: {seal_path}")


if __name__ == "__main__":
    main()
