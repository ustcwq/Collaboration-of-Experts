from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import export_label_free_observables, load_family_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a physically label-free target observable cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target = config.get("target_labels", config.get("target"))
    if target is None:
        raise KeyError("Config must define target_labels or target")
    raw_cache_path = target.get("cache_path") or target.get("label_cache_path")
    if raw_cache_path is None:
        raise KeyError("Target config must define cache_path or label_cache_path")
    family_map = load_family_map(Path(config["family_map"]))
    manifest = export_label_free_observables(
        Path(raw_cache_path),
        args.output_dir,
        str(target["dataset"]),
        str(target["split"]),
        str(target["modality"]),
        family_map,
        [str(value) for value in config["experts"]],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
