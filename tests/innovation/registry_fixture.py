from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def write_registry(
    directory: Path,
    cache_path: Path,
    dataset: str,
    split: str,
    modality: str,
    role: str = "source",
) -> tuple[Path, str]:
    path = directory / f"registry_{dataset}_{split}.yaml"
    payload = {
        "version": 1,
        "datasets": [
            {
                "dataset": dataset,
                "split": split,
                "modality": modality,
                "source_or_target": role,
                "cache_path": str(cache_path.resolve()),
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest
