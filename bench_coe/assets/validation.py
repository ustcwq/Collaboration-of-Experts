from __future__ import annotations

import csv
import json
import mimetypes
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageStat

from .locking import canonical_json_bytes, sha256_bytes, sha256_file, write_json, atomic_write
from .paths import AssetPaths
from .registry import ResourceSpec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def adoption_path(paths: AssetPaths) -> Path:
    return paths.directories()["manifests"] / "local_adoptions.json"


def load_adoptions(paths: AssetPaths) -> dict[str, dict[str, Any]]:
    path = adoption_path(paths)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["logical_name"]: item for item in payload.get("resources", [])}


def adopt_existing_resources(specs: list[ResourceSpec], paths: AssetPaths, repo_root: Path) -> dict[str, dict[str, Any]]:
    existing = {
        name: item for name, item in load_adoptions(paths).items()
        if item.get("source") != "configured_existing_path"
    }
    for spec in specs:
        if spec.logical_name in existing:
            continue
        for configured in spec.metadata.get("existing_paths", []):
            candidate = Path(str(configured))
            candidate = (repo_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            required_globs = [str(item) for item in spec.metadata.get("existing_required_globs", [])]
            has_required_files = all(any(candidate.glob(pattern)) for pattern in required_globs)
            if candidate.exists() and has_required_files:
                existing[spec.logical_name] = {
                    "logical_name": spec.logical_name,
                    "kind": spec.kind,
                    "absolute_path": str(candidate),
                    "status": "adopted_existing",
                    "provenance": "preexisting_repository_asset",
                    "provenance_verified_modelscope": False,
                    "source": "configured_existing_path",
                    "restriction": spec.metadata.get("existing_restriction"),
                    "adopted_at": utc_now(),
                }
                break
    write_json(adoption_path(paths), {"schema_version": "benchcoe_local_adoptions_v1", "resources": sorted(existing.values(), key=lambda item: item["logical_name"])})
    return existing


def effective_resource_path(spec: ResourceSpec, paths: AssetPaths) -> tuple[Path, dict[str, Any] | None]:
    adoption = load_adoptions(paths).get(spec.logical_name)
    if adoption:
        return Path(adoption["absolute_path"]), adoption
    return paths.resource_dir(spec.kind, spec.logical_name), None


def disk_report(paths: AssetPaths, specs: list[ResourceSpec], resolutions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    usage = shutil.disk_usage(paths.root.parent if paths.root.parent.exists() else Path.cwd())
    known = 0
    unknown: list[str] = []
    adoptions = load_adoptions(paths)
    skipped_existing: list[str] = []
    for spec in specs:
        local = paths.resource_dir(spec.kind, spec.logical_name)
        if spec.logical_name in adoptions or (local / ".benchcoe_ready.json").exists():
            skipped_existing.append(spec.logical_name)
            continue
        value = resolutions.get(spec.logical_name, {}).get("download_bytes")
        if isinstance(value, int):
            known += value
        else:
            unknown.append(spec.logical_name)
    required = int(known * 1.25)
    return {
        "generated_at": utc_now(),
        "asset_root": str(paths.root),
        "filesystem_total_bytes": usage.total,
        "filesystem_free_bytes": usage.free,
        "known_download_bytes": known,
        "required_free_bytes_with_margin": required,
        "unknown_size_resources": unknown,
        "skipped_existing_resources": skipped_existing,
        "space_sufficient_for_known_sizes": usage.free >= required,
        "download_allowed": usage.free >= required and not unknown,
    }


def inspect_image(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Image path escapes allowed root: {path}") from error
    result: dict[str, Any] = {"relative_path": str(resolved.relative_to(root.resolve())), "sha256": sha256_file(resolved)}
    with Image.open(resolved) as image:
        image.verify()
    with Image.open(resolved) as image:
        image.load()
        result.update({"width": image.width, "height": image.height, "mime": Image.MIME.get(image.format) or mimetypes.guess_type(path.name)[0] or "application/octet-stream", "format": image.format, "decode_status": "ok"})
        grayscale = image.convert("L")
        extrema = ImageStat.Stat(grayscale).extrema[0]
        result["all_black"] = extrema == (0, 0)
    return result


def verify_resource(spec: ResourceSpec, paths: AssetPaths) -> dict[str, Any]:
    local, adoption = effective_resource_path(spec, paths)
    marker = local / ".benchcoe_ready.json"
    errors: list[str] = []
    if not local.is_dir():
        errors.append("missing_local_directory")
    if not marker.is_file() and adoption is None:
        errors.append("missing_ready_marker")
    if adoption and adoption.get("restriction"):
        errors.append(f"restricted:{adoption['restriction']}")
    files = [path for path in local.rglob("*") if path.is_file()] if local.exists() else []
    if not files:
        errors.append("empty_snapshot")
    image_count = 0
    broken_images = 0
    if spec.kind == "dataset" and spec.modality in {"vision_language", "mixed"}:
        for path in files:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
                image_count += 1
                try:
                    inspect_image(path, local)
                except Exception:
                    broken_images += 1
        if image_count == 0:
            errors.append("no_decodable_image_assets_found")
        if broken_images:
            errors.append(f"broken_images:{broken_images}")
    return {
        "logical_name": spec.logical_name,
        "kind": spec.kind,
        "status": "ready" if not errors else "invalid",
        "local_path": str(local),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "image_count": image_count,
        "errors": errors,
        "adoption": adoption,
    }


def file_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def build_asset_lock(
    profile: str,
    specs: list[ResourceSpec],
    resolutions: dict[str, dict[str, Any]],
    paths: AssetPaths,
    validation_rows: list[dict[str, Any]],
    code_revision: str | None,
) -> dict[str, Any]:
    by_validation = {row["logical_name"]: row for row in validation_rows}
    resources = []
    for spec in specs:
        local, adoption = effective_resource_path(spec, paths)
        resolution = resolutions.get(spec.logical_name, {})
        validation = by_validation.get(spec.logical_name, {})
        processed_manifest = paths.directories()["datasets_processed"] / spec.logical_name / "dataset_manifest.json"
        resources.append({
            "logical_name": spec.logical_name,
            "kind": spec.kind,
            "resolved_modelscope_id": resolution.get("resolved_modelscope_id"),
            "revision": resolution.get("revision"),
            "absolute_path": str(local),
            "portable_path": paths.relative_to_root(local),
            "files": file_inventory(local) if local.exists() else [],
            "license": resolution.get("license", spec.license),
            "license_status": "confirmed" if resolution.get("license") not in {None, "unknown_until_resolved", "unresolved"} else "unconfirmed",
            "download_status": resolution.get("status", "unresolved"),
            "local_adoption": adoption,
            "qa_status": validation.get("status", "not_verified"),
            "processed_manifest_sha256": sha256_file(processed_manifest) if processed_manifest.exists() else None,
            "code_revision": code_revision,
            "allowed_for_profile": validation.get("status") == "ready" and resolution.get("status") == "resolved",
        })
    return {"schema_version": "benchcoe_asset_lock_v1", "profile": profile, "asset_root": str(paths.root), "resources": resources}


def write_lock_with_hash(path: Path, payload: Any, digest_path: Path | None = None) -> tuple[Path, Path]:
    data = canonical_json_bytes(payload)
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"Refusing to overwrite immutable lock {path}; choose a new asset root or archive the existing lock")
    atomic_write(path, data)
    digest_path = digest_path or (path.with_suffix(path.suffix + ".sha256") if path.suffix else path.with_name(path.name + ".sha256"))
    atomic_write(digest_path, f"{sha256_bytes(data)}  {path.name}\n".encode())
    return path, digest_path


def verify_lock(path: Path, digest_path: Path | None = None) -> dict[str, Any]:
    digest_path = digest_path or path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not digest_path.exists():
        return {"status": "invalid", "error": "lock_or_digest_missing"}
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        return {"status": "invalid", "error": "lock_hash_mismatch", "expected": expected, "actual": actual}
    payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
    mismatches = []
    for resource in payload.get("resources", []) if isinstance(payload, dict) else []:
        root = Path(resource["absolute_path"])
        for item in resource.get("files", []):
            candidate = root / item["path"]
            if not candidate.is_file() or candidate.stat().st_size != item["size"] or sha256_file(candidate) != item["sha256"]:
                mismatches.append(f"{resource['logical_name']}:{item['path']}")
    return {"status": "valid" if not mismatches else "invalid", "mismatches": mismatches, "sha256": actual}


def write_resource_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["logical_name", "kind", "status", "local_path", "file_count", "bytes", "image_count", "errors"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["errors"] = json.dumps(item.get("errors", []), ensure_ascii=False)
            writer.writerow({key: item.get(key) for key in fields})


def validate_role_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    seen: dict[tuple[str, str], str] = {}
    conflicts = []
    for dataset, config in payload.get("datasets", {}).items():
        role = config["role"]
        for split in config.get("splits", []):
            key = (dataset, str(split))
            if key in seen and seen[key] != role:
                conflicts.append({"dataset": dataset, "split": split, "roles": [seen[key], role]})
            seen[key] = role
    if conflicts:
        raise ValueError(f"Source/target role conflicts: {conflicts}")
    return payload
