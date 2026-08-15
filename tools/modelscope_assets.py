#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_coe.assets.locking import atomic_write, canonical_json_bytes, sha256_bytes, write_json
from bench_coe.assets.modelscope_backend import ModelScopeBackend, download_resource, load_resolutions
from bench_coe.assets.paths import AssetPaths
from bench_coe.assets.registry import Registry
from bench_coe.assets.validation import (
    adopt_existing_resources,
    build_asset_lock,
    disk_report,
    validate_role_config,
    verify_lock,
    verify_resource,
    write_lock_with_hash,
    write_resource_status,
)


LOG = logging.getLogger("benchcoe.modelscope_assets")
DEFAULT_MODELS = REPO_ROOT / "configs/modelscope_models.yaml"
DEFAULT_DATASETS = REPO_ROOT / "configs/modelscope_datasets.yaml"
DEFAULT_ROLES = REPO_ROOT / "configs/data_roles.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_names(values: list[str] | None) -> set[str]:
    result = set()
    for value in values or []:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=["smoke", "core", "full"], default=os.environ.get("BENCHCOE_PROFILE", "smoke"))
    parser.add_argument("--asset-root", help="Override BENCHCOE_ASSET_ROOT")
    parser.add_argument("--modelscope-cache", help="Override MODELSCOPE_CACHE")
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--roles-config", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--only", action="append", help="Comma-separated logical resource names")
    parser.add_argument("--exclude", action="append", help="Comma-separated logical resource names")
    parser.add_argument("--revision", help="Explicit revision override")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable ModelScope-only assets for Bench-CoE")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    common = common_parser()
    resolve = sub.add_parser("resolve", parents=[common], help="Verify configured IDs with ModelScope SDK")
    resolve.add_argument("--max-workers", type=int, default=4)
    resolve.add_argument("--dry-run", action="store_true", help="Validate selection without network calls")
    sub.add_parser("estimate", parents=[common], help="Estimate disk requirement from resolved metadata")
    download = sub.add_parser("download", parents=[common], help="Download resolved snapshots atomically")
    download.add_argument("--max-workers", type=int, default=1)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--timeout", type=int, default=86400)
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--allow-unknown-size", action="store_true", help="Explicitly bypass unknown-size block")
    sub.add_parser("verify", parents=[common], help="Verify local snapshots and existing locks")
    lock = sub.add_parser("lock", parents=[common], help="Create immutable asset and protocol locks")
    lock.add_argument("--reason", help="Required if writing a versioned protocol lock after changes")
    sub.add_parser("report", parents=[common], help="Generate preparation status report")
    return parser


def context(args: argparse.Namespace):
    paths = AssetPaths.from_env(args.asset_root, args.modelscope_cache)
    registry = Registry.load(args.models_config, args.datasets_config)
    specs = registry.select(args.profile, parse_names(args.only), parse_names(args.exclude))
    return paths, registry, specs


def resolution_path(paths: AssetPaths, profile: str) -> Path:
    return paths.directories()["manifests"] / f"modelscope_resolution_{profile}.json"


def environment_payload(paths: AssetPaths) -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "modelscope", "evalscope", "vllm", "yaml", "PIL"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as error:
            versions[name] = f"missing:{type(error).__name__}"
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.strip() or None
        status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.splitlines()
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.strip() or None
    except OSError:
        branch, status, revision = None, [], None
    return {"generated_at": utc_now(), "python": sys.version, "platform": platform.platform(), "versions": versions, "asset_root": str(paths.root), "modelscope_cache": str(paths.modelscope_cache), "git_branch": branch, "git_revision": revision, "git_status": status}


def command_resolve(args: argparse.Namespace) -> int:
    paths, _, specs = context(args)
    paths.ensure()
    validate_role_config(args.roles_config)
    if args.dry_run:
        payload = {"schema_version": "benchcoe_resolution_v1", "profile": args.profile, "generated_at": utc_now(), "dry_run": True, "resources": [{"logical_name": spec.logical_name, "kind": spec.kind, "status": "not_queried_dry_run", "candidates": list(spec.candidate_modelscope_ids), "search_queries": list(spec.search_queries), "required": spec.required_for(args.profile)} for spec in specs]}
    else:
        backend = ModelScopeBackend()
        results = []
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            futures = {pool.submit(backend.resolve, spec, args.revision): spec for spec in specs}
            for future in as_completed(futures):
                result = future.result().to_dict()
                result["required"] = futures[future].required_for(args.profile)
                results.append(result)
                LOG.info("resolved %s -> %s", result["logical_name"], result["status"])
        payload = {"schema_version": "benchcoe_resolution_v1", "profile": args.profile, "generated_at": utc_now(), "dry_run": False, "resources": sorted(results, key=lambda item: item["logical_name"])}
    write_json(resolution_path(paths, args.profile), payload)
    write_json(paths.directories()["manifests"] / "model_registry.resolved.json", {
        "schema_version": "benchcoe_resolved_model_registry_v1",
        "profile": args.profile,
        "resources": [item for item in payload["resources"] if item["kind"] == "model"],
    })
    write_json(paths.directories()["manifests"] / "dataset_registry.resolved.json", {
        "schema_version": "benchcoe_resolved_dataset_registry_v1",
        "profile": args.profile,
        "resources": [item for item in payload["resources"] if item["kind"] == "dataset"],
    })
    write_json(paths.directories()["manifests"] / "environment_audit.json", environment_payload(paths))
    required_failures = [item for item in payload["resources"] if item.get("required") and item["status"] != "resolved"]
    print(json.dumps({"path": str(resolution_path(paths, args.profile)), "resources": len(specs), "required_unresolved": len(required_failures)}, ensure_ascii=False, indent=2))
    return 1 if required_failures and not args.dry_run else 0


def require_resolutions(paths: AssetPaths, profile: str) -> dict[str, dict[str, Any]]:
    path = resolution_path(paths, profile)
    if not path.exists():
        raise FileNotFoundError(f"Run resolve first: {path}")
    return load_resolutions(path)


def command_estimate(args: argparse.Namespace) -> int:
    paths, _, specs = context(args)
    paths.ensure()
    resolutions = require_resolutions(paths, args.profile)
    report = disk_report(paths, specs, resolutions)
    write_json(paths.directories()["manifests"] / "disk_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["download_allowed"] else 2


def command_download(args: argparse.Namespace) -> int:
    paths, _, specs = context(args)
    paths.ensure()
    adoptions = adopt_existing_resources(specs, paths, REPO_ROOT)
    resolutions = require_resolutions(paths, args.profile)
    report = disk_report(paths, specs, resolutions)
    write_json(paths.directories()["manifests"] / "disk_report.json", report)
    if not report["space_sufficient_for_known_sizes"]:
        print("Insufficient disk space for known resource sizes with 1.25x margin", file=sys.stderr)
        return 2
    if report["unknown_size_resources"] and not args.allow_unknown_size:
        print("Unknown resource sizes block downloads; resolve file metadata or pass --allow-unknown-size explicitly", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "resources": [item.logical_name for item in specs], "disk_report": report}, ensure_ascii=False, indent=2))
        return 0
    backend = ModelScopeBackend()
    rows = []
    for spec in specs:
        if spec.logical_name in adoptions:
            result = {"logical_name": spec.logical_name, "status": "adopted_existing", "local_path": adoptions[spec.logical_name]["absolute_path"], "retries": 0}
        else:
            result = download_resource(spec, resolutions.get(spec.logical_name, {}), paths, backend, max_workers=args.max_workers, retries=args.retries, timeout_seconds=args.timeout)
        rows.append(result)
        LOG.info("download %s -> %s", spec.logical_name, result["status"])
    write_json(paths.directories()["manifests"] / f"download_status_{args.profile}.json", {"profile": args.profile, "generated_at": utc_now(), "resources": rows})
    failures = [row for row, spec in zip(rows, specs) if spec.required_for(args.profile) and row["status"] not in {"downloaded", "already_ready", "adopted_existing"}]
    return 1 if failures else 0


def command_verify(args: argparse.Namespace) -> int:
    paths, _, specs = context(args)
    paths.ensure()
    adopt_existing_resources(specs, paths, REPO_ROOT)
    rows = [verify_resource(spec, paths) for spec in specs]
    write_json(paths.directories()["manifests"] / f"verification_{args.profile}.json", {"profile": args.profile, "generated_at": utc_now(), "resources": rows})
    write_resource_status(paths.directories()["manifests"] / "resource_status.csv", rows)
    lock_result = verify_lock(paths.directories()["manifests"] / "asset_lock.json", paths.directories()["manifests"] / "asset_lock.sha256")
    protocol_result = verify_lock(paths.directories()["manifests"] / "protocol_lock.yaml", paths.directories()["manifests"] / "protocol_lock.sha256")
    print(json.dumps({"resources": rows, "asset_lock": lock_result, "protocol_lock": protocol_result}, ensure_ascii=False, indent=2))
    failures = [row for row, spec in zip(rows, specs) if spec.required_for(args.profile) and row["status"] != "ready"]
    return 1 if failures or (lock_result["status"] == "invalid" and (paths.directories()["manifests"] / "asset_lock.json").exists()) else 0


def _protocol_payload(args: argparse.Namespace, specs) -> dict[str, Any]:
    roles = validate_role_config(args.roles_config)
    return {"schema_version": "benchcoe_protocol_lock_v1", "profile": args.profile, "data_roles": roles["datasets"], "models": [spec.logical_name for spec in specs if spec.kind == "model"], **roles.get("protocol", {})}


def command_lock(args: argparse.Namespace) -> int:
    paths, _, specs = context(args)
    paths.ensure()
    adopt_existing_resources(specs, paths, REPO_ROOT)
    resolutions = require_resolutions(paths, args.profile)
    rows = [verify_resource(spec, paths) for spec in specs]
    audit = environment_payload(paths)
    lock = build_asset_lock(args.profile, specs, resolutions, paths, rows, audit.get("git_revision"))
    manifest_dir = paths.directories()["manifests"]
    write_lock_with_hash(manifest_dir / "asset_lock.json", lock, manifest_dir / "asset_lock.sha256")
    protocol = _protocol_payload(args, specs)
    protocol_path = manifest_dir / "protocol_lock.yaml"
    if protocol_path.exists() and protocol_path.read_bytes() != canonical_json_bytes(protocol):
        if not args.reason:
            raise FileExistsError("Protocol changed; pass --reason to create a new immutable version")
        digest = sha256_bytes(canonical_json_bytes(protocol))[:12]
        protocol["change_reason"] = args.reason
        protocol_path = manifest_dir / f"protocol_lock.{digest}.yaml"
        digest_path = manifest_dir / f"protocol_lock.{digest}.sha256"
    else:
        digest_path = manifest_dir / "protocol_lock.sha256"
    write_lock_with_hash(protocol_path, protocol, digest_path)
    print(json.dumps({"asset_lock": str(manifest_dir / "asset_lock.json"), "protocol_lock": str(protocol_path)}, ensure_ascii=False, indent=2))
    required_not_ready = [row for row, spec in zip(rows, specs) if spec.required_for(args.profile) and row["status"] != "ready"]
    return 1 if required_not_ready else 0


def command_report(args: argparse.Namespace) -> int:
    paths, _, specs = context(args)
    paths.ensure()
    adopt_existing_resources(specs, paths, REPO_ROOT)
    resolutions = require_resolutions(paths, args.profile)
    disk = disk_report(paths, specs, resolutions)
    rows = [verify_resource(spec, paths) for spec in specs]
    resolved = sum(resolutions.get(spec.logical_name, {}).get("status") == "resolved" for spec in specs)
    ready = sum(row["status"] == "ready" for row in rows)
    restricted = sum(bool(resolutions.get(spec.logical_name, {}).get("gated")) for spec in specs)
    required_missing = [spec.logical_name for spec, row in zip(specs, rows) if spec.required_for(args.profile) and row["status"] != "ready"]
    status = "COMPLETE" if not required_missing else "PARTIAL"
    report_path = paths.directories()["manifests"] / "preparation_report.md"
    report_path.write_text(
        "# Bench-CoE ModelScope Resource Preparation\n\n"
        f"- Status: **{status}**\n- Generated: {utc_now()}\n- Profile: `{args.profile}`\n"
        f"- Asset root: `{paths.root}`\n- ModelScope cache: `{paths.modelscope_cache}`\n"
        f"- Selected resources: {len(specs)}\n- Resolved: {resolved}\n- Locally ready: {ready}\n- Restricted/gated: {restricted}\n"
        f"- Free disk: {disk['filesystem_free_bytes']} bytes\n- Known download estimate: {disk['known_download_bytes']} bytes\n"
        f"- Unknown-size resources: {', '.join(disk['unknown_size_resources']) or 'none'}\n"
        f"- Required resources not ready: {', '.join(required_missing) or 'none'}\n\n"
        "## Safety\n\nNo target labels are exposed by this preparation report. No non-ModelScope fallback is used.\n",
        encoding="utf-8",
    )
    licenses = paths.directories()["manifests"] / "licenses_report.md"
    licenses.write_text("# Licenses\n\n" + "\n".join(f"- `{spec.logical_name}`: {resolutions.get(spec.logical_name, {}).get('license', spec.license)}" for spec in specs) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path), "required_not_ready": required_missing}, ensure_ascii=False, indent=2))
    return 0 if status == "COMPLETE" else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    commands = {"resolve": command_resolve, "estimate": command_estimate, "download": command_download, "verify": command_verify, "lock": command_lock, "report": command_report}
    try:
        return commands[args.command](args)
    except Exception as error:
        LOG.error("%s: %s", type(error).__name__, error)
        if args.verbose:
            LOG.exception("command failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
