from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import manifest_sha256, sha256_file, write_json, write_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge authenticated GPQA inference shards")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    shard_count = int(config["generation"]["shard_count"])
    rows: list[dict[str, Any]] = []
    authenticated: dict[str, str] = {}
    seen: set[str] = set()
    for shard_index in range(shard_count):
        root = args.shard_root / f"shard_{shard_index}"
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["shard_index"]) != shard_index or int(manifest["shard_count"]) != shard_count:
            raise RuntimeError(f"Invalid shard manifest: {manifest_path}")
        if manifest.get("protocol") != config["protocol_name"]:
            raise RuntimeError(f"Shard protocol mismatch: {manifest_path}")
        if manifest.get("model") != config["source_selection"]["selected_model"]:
            raise RuntimeError(f"Shard model mismatch: {manifest_path}")
        config_path = str(args.config)
        expected_config_hash = manifest.get("input_hashes", {}).get(config_path)
        if expected_config_hash != sha256_file(args.config):
            raise RuntimeError(f"Shard config hash mismatch: {manifest_path}")
        if manifest.get("target_labels_opened") is not False:
            raise RuntimeError(f"Shard lacks label-firewall attestation: {manifest_path}")
        prediction_path = root / "predictions.jsonl"
        digest = sha256_file(prediction_path)
        if digest != str(manifest["prediction_sha256"]):
            raise RuntimeError(f"Shard prediction hash mismatch: {prediction_path}")
        authenticated[str(prediction_path)] = digest
        authenticated[str(manifest_path)] = sha256_file(manifest_path)
        for row in _read_jsonl(prediction_path):
            row_id = str(row["id"])
            if row_id in seen:
                raise RuntimeError(f"Duplicate GPQA inference ID: {row_id}")
            seen.add(row_id)
            rows.append(row)
    expected = int(config["target_observables"]["expected_epoch0_questions"])
    if len(rows) != expected:
        raise RuntimeError(f"Merged {len(rows)} rows; expected {expected}")
    rows.sort(key=lambda row: str(row["id"]))
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, rows)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "model": str(config["source_selection"]["selected_model"]),
        "questions": len(rows),
        "valid_explicit_predictions": sum(row.get("prediction") is not None for row in rows),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "authenticated_shards": authenticated,
        "authenticated_shards_manifest_sha256": manifest_sha256(authenticated),
        "target_labels_opened_during_prediction": False,
    }
    manifest_path = args.output_dir / "prediction_manifest.json"
    write_json(manifest_path, prediction_manifest)
    write_json(
        args.output_dir / "complete_manifest.json",
        {
            "prediction_manifest_sha256_before_target_labels": sha256_file(manifest_path),
            "prediction_file_sha256": sha256_file(prediction_path),
            "target_labels_opened": False,
        },
    )
    print(json.dumps(prediction_manifest, indent=2))


if __name__ == "__main__":
    main()
