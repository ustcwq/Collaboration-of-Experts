from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .locking import file_lock, write_json
from .paths import AssetPaths
from .registry import ResourceSpec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_size(item: dict[str, Any]) -> int | None:
    for key in ("Size", "size", "FileSize", "file_size"):
        value = item.get(key)
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


@dataclass
class Resolution:
    logical_name: str
    kind: str
    status: str
    resolved_modelscope_id: str | None
    revision: str | None
    files: list[dict[str, Any]]
    download_bytes: int | None
    publisher: str | None
    license: str
    gated: bool
    candidates_checked: list[dict[str, Any]]
    search_queries: list[str]
    resolved_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelScopeBackend:
    """Thin, mockable adapter over the installed ModelScope SDK."""

    def __init__(self, api: Any | None = None):
        os.environ.setdefault("MODELSCOPE_DOMAIN", "modelscope.cn")
        if api is None:
            from modelscope.hub.api import HubApi

            api = HubApi()
            token = os.environ.get("MODELSCOPE_API_TOKEN")
            if token:
                api.login(token)
        self.api = api

    @staticmethod
    def _rest_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        import requests

        response = requests.get(f"https://modelscope.cn{path}", params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("Code") != 200:
            raise RuntimeError(f"ModelScope REST error {payload.get('Code')}: {payload.get('Message')}")
        return payload["Data"]

    def _resolve_via_rest(self, spec: ResourceSpec, candidate: str, revision: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        revision_value = revision or "master"
        if spec.kind == "model":
            metadata = self._rest_get(f"/api/v1/models/{candidate}", {"Revision": revision_value})
            file_data = self._rest_get(
                f"/api/v1/models/{candidate}/repo/files",
                {"Revision": revision_value, "Recursive": "true"},
            )
            return metadata, list(file_data.get("Files") or [])
        metadata = self._rest_get(f"/api/v1/datasets/{candidate}", {"Revision": revision_value})
        dataset_id = metadata.get("Id")
        if dataset_id is None:
            raise RuntimeError("Dataset metadata did not include numeric Id")
        file_data = self._rest_get(
            f"/api/v1/datasets/{dataset_id}/repo/tree",
            {"Revision": revision_value, "Root": "/", "Recursive": "True", "PageNumber": 1, "PageSize": 10000},
        )
        return metadata, list(file_data.get("Files") or [])

    def resolve(self, spec: ResourceSpec, revision_override: str | None = None) -> Resolution:
        checked: list[dict[str, Any]] = []
        revision = revision_override or spec.revision
        for candidate in spec.candidate_modelscope_ids:
            try:
                try:
                    if spec.kind == "model":
                        metadata = self.api.get_model(candidate, revision=revision)
                        files = self.api.get_model_files(candidate, revision=revision, recursive=True)
                    else:
                        metadata = self.api.get_dataset(candidate, revision=revision or "master")
                        files = self.api.get_dataset_files(candidate, revision=revision or "master", recursive=True)
                except Exception:
                    metadata, files = self._resolve_via_rest(spec, candidate, revision)
                normalized_files = [dict(item) for item in (files or [])]
                sizes = [_file_size(item) for item in normalized_files]
                total = sum(size for size in sizes if size is not None) if sizes and all(size is not None for size in sizes) else None
                meta = metadata if isinstance(metadata, dict) else {}
                publisher = candidate.split("/", 1)[0] if "/" in candidate else None
                resolved_revision = str(meta.get("Revision") or meta.get("revision") or revision or "master")
                checked.append({"id": candidate, "status": "resolved"})
                return Resolution(
                    logical_name=spec.logical_name,
                    kind=spec.kind,
                    status="resolved",
                    resolved_modelscope_id=candidate,
                    revision=resolved_revision,
                    files=normalized_files,
                    download_bytes=total,
                    publisher=publisher,
                    license=str(meta.get("License") or meta.get("license") or spec.license),
                    gated=bool(meta.get("Gated") or meta.get("gated") or spec.metadata.get("gated", False)),
                    candidates_checked=checked,
                    search_queries=list(spec.search_queries),
                    resolved_at=utc_now(),
                )
            except Exception as error:
                checked.append({"id": candidate, "status": "unavailable", "error": f"{type(error).__name__}: {error}"})
        network_markers = ("connectionerror", "proxyerror", "timeout", "connection refused", "network is unreachable")
        had_network_failure = any(
            any(marker in item.get("error", "").casefold() for marker in network_markers)
            for item in checked
        )
        status = "resolution_failed_network" if had_network_failure else str(
            spec.metadata.get("discovery_status") or (
                "not_available_on_modelscope" if spec.candidate_modelscope_ids else "unresolved_search_required"
            )
        )
        return Resolution(
            logical_name=spec.logical_name,
            kind=spec.kind,
            status=status,
            resolved_modelscope_id=None,
            revision=revision,
            files=[],
            download_bytes=None,
            publisher=None,
            license=spec.license,
            gated=bool(spec.metadata.get("gated", False)),
            candidates_checked=checked,
            search_queries=list(spec.search_queries),
            resolved_at=utc_now(),
            error=str(spec.metadata.get("impact") or "No configured candidate could be verified with the installed ModelScope SDK/API"),
        )

    def snapshot_download(
        self,
        kind: str,
        resource_id: str,
        revision: str | None,
        cache_dir: Path,
        local_dir: Path,
        max_workers: int,
    ) -> str:
        if kind == "model":
            from modelscope import snapshot_download

            return snapshot_download(
                resource_id,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=str(local_dir),
                max_workers=max_workers,
                enable_file_lock=True,
            )
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        return dataset_snapshot_download(
            resource_id,
            revision=revision or "master",
            cache_dir=cache_dir,
            local_dir=str(local_dir),
            max_workers=max_workers,
            enable_file_lock=True,
        )

    def cli_snapshot_download(
        self,
        kind: str,
        resource_id: str,
        revision: str | None,
        cache_dir: Path,
        local_dir: Path,
        max_workers: int,
    ) -> str:
        command = ["modelscope", "download", "--model" if kind == "model" else "--dataset", resource_id]
        if revision:
            command.extend(["--revision", revision])
        command.extend([
            "--cache_dir", str(cache_dir),
            "--local_dir", str(local_dir),
            "--max-workers", str(max_workers),
        ])
        token = os.environ.get("MODELSCOPE_API_TOKEN")
        if token:
            command.extend(["--token", token])
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"ModelScope CLI failed ({result.returncode}): {result.stderr[-2000:]}")
        return str(local_dir)


def download_resource(
    spec: ResourceSpec,
    resolution: dict[str, Any],
    paths: AssetPaths,
    backend: ModelScopeBackend,
    max_workers: int = 1,
    retries: int = 3,
    timeout_seconds: int = 86400,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if resolution.get("status") != "resolved" or not resolution.get("resolved_modelscope_id"):
        return {"logical_name": spec.logical_name, "status": "blocked_unresolved", "error": resolution.get("error")}
    paths.ensure()
    final = paths.resource_dir(spec.kind, spec.logical_name)
    marker = final / ".benchcoe_ready.json"
    if marker.exists():
        return {"logical_name": spec.logical_name, "status": "already_ready", "local_path": str(final), "retries": 0}
    lock_path = paths.directories()["locks"] / f"download-{spec.logical_name}.lock"
    with file_lock(lock_path, timeout=float(timeout_seconds)):
        if marker.exists():
            return {"logical_name": spec.logical_name, "status": "already_ready", "local_path": str(final), "retries": 0}
        staging = paths.directories()["tmp"] / f"{spec.logical_name}.incomplete"
        staging.mkdir(parents=True, exist_ok=True)
        started = utc_now()
        last_error = None
        for attempt in range(retries + 1):
            try:
                try:
                    backend.snapshot_download(
                        spec.kind,
                        str(resolution["resolved_modelscope_id"]),
                        resolution.get("revision"),
                        paths.modelscope_cache,
                        staging,
                        max_workers=max_workers if spec.kind == "dataset" else 1,
                    )
                except Exception:
                    if not hasattr(backend, "cli_snapshot_download"):
                        raise
                    backend.cli_snapshot_download(
                        spec.kind,
                        str(resolution["resolved_modelscope_id"]),
                        resolution.get("revision"),
                        paths.modelscope_cache,
                        staging,
                        max_workers=max_workers if spec.kind == "dataset" else 1,
                    )
                if not any(path.is_file() for path in staging.rglob("*")):
                    raise RuntimeError("ModelScope returned an empty snapshot")
                if final.exists():
                    raise RuntimeError(f"Final path appeared without ready marker: {final}")
                write_json(staging / ".benchcoe_ready.json", {
                    "logical_name": spec.logical_name,
                    "modelscope_id": resolution["resolved_modelscope_id"],
                    "revision": resolution.get("revision"),
                    "started_at": started,
                    "completed_at": utc_now(),
                })
                os.replace(staging, final)
                return {"logical_name": spec.logical_name, "status": "downloaded", "local_path": str(final), "retries": attempt}
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < retries:
                    sleep(min(60.0, 2.0 ** attempt))
        quarantine = paths.directories()["quarantine"] / f"{spec.logical_name}-{int(time.time())}"
        if staging.exists():
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(quarantine))
        return {"logical_name": spec.logical_name, "status": "download_failed", "error": last_error, "quarantine_path": str(quarantine), "retries": retries}


def load_resolutions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["logical_name"]: item for item in payload.get("resources", [])}
