from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .schema import Selection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_manifest(paths: Iterable[Path]) -> dict[str, str]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
    return {str(path): sha256_file(path) for path in sorted(files)}


def manifest_sha256(manifest: dict[str, str]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_gpu_map(config: dict[str, Any], key: str) -> dict[int, int]:
    seeds = [int(value) for value in config["seeds"]]
    raw_gpus = config.get(key, list(range(len(seeds))))
    if not isinstance(raw_gpus, list) or len(raw_gpus) != len(seeds):
        raise ValueError(f"{key} must contain exactly one GPU for each seed")
    gpus = [int(value) for value in raw_gpus]
    if any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError(f"{key} must contain unique non-negative physical GPU indices")
    return dict(zip(seeds, gpus, strict=True))


def innovation_code_manifest() -> dict[str, str]:
    roots = [Path("bench_coe/innovation"), Path("tests/innovation"), Path("scripts")]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".sh"}
        )
    return {str(path): sha256_file(path) for path in sorted(files)}


def validate_test_receipt(path: Path, config_path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if int(receipt.get("exit_code", 1)) != 0:
        raise RuntimeError(f"Test receipt is not passing: {path}")
    current_code = innovation_code_manifest()
    if receipt.get("code_manifest_sha256") != manifest_sha256(current_code):
        raise RuntimeError("Test receipt code hash does not match the current innovation source")
    expected_config_hash = receipt.get("config_hashes", {}).get(str(config_path))
    if expected_config_hash != sha256_file(config_path):
        raise RuntimeError(f"Test receipt does not authenticate config {config_path}")
    return receipt


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_selections(path: Path, selections: list[Selection]) -> str:
    write_jsonl(path, [asdict(selection) for selection in selections])
    return sha256_file(path)


def read_selections(path: Path) -> list[Selection]:
    result: list[Selection] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result.append(Selection(**row))
    return result


def environment_manifest(command: list[str], seed: int, input_paths: Iterable[Path]) -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL).splitlines()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_commit = "UNKNOWN"
        git_status = ["UNKNOWN: workspace root is not a Git work tree"]
    versions: dict[str, str] = {}
    for module in ("numpy", "scipy", "sklearn", "torch"):
        try:
            imported = __import__(module)
            versions[module] = str(imported.__version__)
        except Exception:
            versions[module] = "UNAVAILABLE"
    input_hashes = files_manifest(input_paths)
    code_hashes = innovation_code_manifest()
    return {
        "command": command,
        "seed": seed,
        "git_commit": git_commit,
        "git_status": git_status,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_hashes": input_hashes,
        "input_manifest_sha256": manifest_sha256(input_hashes),
        "innovation_code_manifest_sha256": manifest_sha256(code_hashes),
    }
