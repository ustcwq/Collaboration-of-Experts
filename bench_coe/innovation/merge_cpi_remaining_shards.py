from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from .artifacts import (
    files_manifest,
    innovation_code_manifest,
    manifest_sha256,
    seed_gpu_map,
    sha256_file,
    validate_test_receipt,
    write_json,
)
from .cpi_remaining import ALL_VARIANT_NAMES, METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge authenticated remaining-source prediction shards")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_artifacts(manifest: dict[str, Any], shard_dir: Path) -> None:
    for raw_path, expected in manifest.get("artifact_hashes", {}).items():
        path = Path(raw_path)
        try:
            path.resolve().relative_to(shard_dir.resolve())
        except ValueError as error:
            raise RuntimeError(f"Shard manifest references an external artifact: {path}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Shard artifact hash mismatch: {path}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt = validate_test_receipt(Path(config["test_receipt"]), args.config)
    current_code = manifest_sha256(innovation_code_manifest())
    gpu_by_seed = seed_gpu_map(config, "physical_gpus")
    if gpu_by_seed.get(args.seed) != args.physical_gpu:
        raise ValueError("Merge seed and physical GPU do not match the frozen mapping")
    shard_dirs = sorted(path for path in args.shard_root.glob("shard_*") if path.is_dir())
    if len(shard_dirs) != 4:
        raise RuntimeError(f"Expected four remaining-source shards, found {len(shard_dirs)}")
    manifests: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    method_sources: dict[str, tuple[Path, str]] = {}
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "prediction_manifest.json"
        completion_path = shard_dir / "run_complete_manifest.json"
        if not manifest_path.is_file() or not completion_path.is_file():
            raise RuntimeError(f"Incomplete remaining-source shard: {shard_dir}")
        manifest = _load(manifest_path)
        completion = _load(completion_path)
        _validate_artifacts(completion, shard_dir)
        if int(manifest["seed"]) != args.seed or int(manifest["physical_gpu"]) != args.physical_gpu:
            raise RuntimeError(f"Shard metadata does not match requested seed/GPU: {shard_dir}")
        if manifest["innovation_code_manifest_sha256"] != current_code:
            raise RuntimeError(f"Shard did not use the tested code: {shard_dir}")
        if sha256_file(manifest_path) != completion["prediction_manifest_sha256"]:
            raise RuntimeError(f"Shard prediction manifest changed: {shard_dir}")
        for method in manifest["active_methods"]:
            if method in method_sources:
                raise RuntimeError(f"Prediction method appears in multiple shards: {method}")
            prediction_path = shard_dir / "predictions" / f"{method}.jsonl"
            expected = manifest["prediction_hashes_before_evaluation"][method]
            if sha256_file(prediction_path) != expected:
                raise RuntimeError(f"Shard prediction changed for {method}")
            method_sources[method] = (prediction_path, expected)
        manifests.append(manifest)
        completions.append(completion)
    if tuple(sorted(method_sources)) != tuple(sorted(METHODS)):
        missing = sorted(set(METHODS).difference(method_sources))
        extra = sorted(set(method_sources).difference(METHODS))
        raise RuntimeError(f"Shard method partition is incomplete: missing={missing}, extra={extra}")
    reference = manifests[0]
    invariant_fields = (
        "input_hashes",
        "input_manifest_sha256",
        "source_questions",
        "source_environments",
        "source_label_structure",
        "source_question_ids_sha256",
        "heldout_environments",
        "base_prediction_hashes",
    )
    for manifest in manifests[1:]:
        for field in invariant_fields:
            if manifest[field] != reference[field]:
                raise RuntimeError(f"Shard manifests disagree on {field}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    prediction_hashes: dict[str, str] = {}
    for method in METHODS:
        source_path, expected = method_sources[method]
        destination = predictions_dir / source_path.name
        shutil.copyfile(source_path, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"Copied prediction changed for {method}")
        prediction_hashes[method] = expected
    merged_manifest = {
        **{key: value for key, value in reference.items() if key not in {"command", "active_methods", "active_variants", "variant_specs", "prediction_hashes_before_evaluation", "started_unix"}},
        "command": sys.argv,
        "started_unix": min(float(manifest["started_unix"]) for manifest in manifests),
        "protocol": "hash-only merge of four concurrently executed remaining-source variant shards",
        "active_variants": list(ALL_VARIANT_NAMES),
        "active_methods": list(METHODS),
        "prediction_hashes_before_evaluation": prediction_hashes,
        "derived_from_shards": {
            str(shard_dir): {
                "prediction_manifest_sha256": sha256_file(shard_dir / "prediction_manifest.json"),
                "completion_manifest_sha256": sha256_file(shard_dir / "run_complete_manifest.json"),
                "active_methods": manifest["active_methods"],
            }
            for shard_dir, manifest in zip(shard_dirs, manifests)
        },
    }
    write_json(args.output_dir / "prediction_manifest.json", merged_manifest)
    resources = [_load(shard_dir / "resource_usage.json") for shard_dir in shard_dirs]
    resource_usage = {
        "physical_gpu": args.physical_gpu,
        "visible_device": 0,
        "cuda_visible_devices": str(args.physical_gpu),
        "device_name": resources[0]["device_name"],
        "cuda_runtime": resources[0]["cuda_runtime"],
        "concurrent_shards": len(shard_dirs),
        "peak_allocated_bytes": sum(int(row["peak_allocated_bytes"]) for row in resources),
        "peak_reserved_bytes": sum(int(row["peak_reserved_bytes"]) for row in resources),
        "runtime_seconds": max(float(row["finished_unix"]) for row in resources)
        - min(float(manifest["started_unix"]) for manifest in manifests),
        "sum_shard_runtime_seconds": sum(float(row["runtime_seconds"]) for row in resources),
        "finished_unix": max(float(row["finished_unix"]) for row in resources),
        "shard_resources": {shard_dir.name: row for shard_dir, row in zip(shard_dirs, resources)},
    }
    write_json(args.output_dir / "resource_usage.json", resource_usage)
    artifacts = files_manifest(
        [args.output_dir / "prediction_manifest.json", predictions_dir, args.output_dir / "resource_usage.json"]
    )
    write_json(
        args.output_dir / "run_complete_manifest.json",
        {
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "prediction_manifest_sha256": sha256_file(args.output_dir / "prediction_manifest.json"),
            "artifact_hashes": artifacts,
            "determinism": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": str(args.seed),
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
            "test_receipt_code_manifest_sha256": receipt["code_manifest_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "seed": args.seed,
                "physical_gpu": args.physical_gpu,
                "methods": len(METHODS),
                "runtime_seconds": resource_usage["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
