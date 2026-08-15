#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_coe.assets.locking import write_json
from bench_coe.assets.paths import AssetPaths
from bench_coe.assets.registry import Registry
from bench_coe.data.preprocessing import ID_KEYS, QUESTION_KEYS, convert_row, detect_cross_dataset_overlaps, discover_data_files, first_value, read_rows, write_processed_dataset


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Convert ModelScope dataset snapshots to benchcoe_unified_v1")
    value.add_argument("--dataset", required=True, help="Logical dataset name")
    value.add_argument("--split", required=True)
    value.add_argument("--role", choices=["source_calibration", "source_validation", "target_locked_test", "secondary_test"])
    value.add_argument("--revision", default="resolved")
    value.add_argument("--input", type=Path, help="Raw snapshot/file; defaults to asset root datasets_raw/<dataset>")
    value.add_argument("--asset-root")
    value.add_argument("--modelscope-cache")
    value.add_argument("--datasets-config", type=Path, default=REPO_ROOT / "configs/modelscope_datasets.yaml")
    value.add_argument("--roles-config", type=Path, default=REPO_ROOT / "configs/data_roles.yaml")
    value.add_argument("--smoke", action="store_true", help="Write deterministic maximum-8 sample slice")
    return value


def main() -> int:
    args = parser().parse_args()
    paths = AssetPaths.from_env(args.asset_root, args.modelscope_cache)
    paths.ensure()
    registry = Registry.load(args.datasets_config)
    specs = {item.logical_name: item for item in registry.resources}
    if args.dataset not in specs:
        raise SystemExit(f"Unknown dataset {args.dataset!r}")
    spec = specs[args.dataset]
    roles = yaml.safe_load(args.roles_config.read_text(encoding="utf-8"))
    role_config = roles.get("datasets", {}).get(args.dataset)
    if not role_config:
        raise SystemExit(f"Dataset {args.dataset!r} has no locked role")
    role = args.role or role_config["role"]
    if role != role_config["role"] or args.split not in [str(item) for item in role_config.get("splits", [])]:
        raise SystemExit(f"Role/split violates data_roles.yaml for {args.dataset}")
    raw = (args.input or paths.resource_dir("dataset", args.dataset)).resolve()
    files = [raw] if raw.is_file() else discover_data_files(raw)
    if not files:
        raise SystemExit(f"No supported data files found under {raw}")
    raw_rows = []
    for file in files:
        for row in read_rows(file):
            native_id = str(first_value(row, ID_KEYS, ""))
            stable_text = native_id or str(first_value(row, QUESTION_KEYS, ""))
            key = hashlib.sha256(f"{args.dataset}\0{args.split}\0{stable_text}".encode()).hexdigest()
            raw_rows.append((key, file, row))
    if args.smoke:
        raw_rows = sorted(raw_rows, key=lambda item: item[0])[:8]
    samples = []
    for _, file, row in raw_rows:
        try:
            samples.append(convert_row(row, args.dataset, args.revision, args.split, role, spec.modality, str(spec.metadata.get("task_type", "mixed")), spec.license, file.parent, paths.directories()["image_assets"]))
        except ValueError as error:
            raise SystemExit(f"{file}: {error}") from error
    output = paths.directories()["datasets_processed"] / args.dataset
    manifest = write_processed_dataset(
        samples,
        output,
        files,
        smoke=args.smoke,
        preselected=args.smoke,
        selection="sha256(dataset, split, native_id-or-question) ascending" if args.smoke else "all converted rows",
    )
    overlaps = detect_cross_dataset_overlaps(paths.directories()["datasets_processed"])
    overlap_path = paths.directories()["manifests"] / "overlap_report.jsonl"
    with overlap_path.open("w", encoding="utf-8") as handle:
        for item in overlaps:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": manifest, "cross_dataset_overlaps": len(overlaps), "output": str(output)}, ensure_ascii=False, indent=2))
    return 1 if any(item.get("role_conflict") for item in overlaps) else 0


if __name__ == "__main__":
    raise SystemExit(main())
