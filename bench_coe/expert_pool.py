from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_expert_pool(path: Path, pool_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    pools = payload.get("pools", {})
    if pool_name not in pools:
        raise ValueError(
            f"Unknown expert pool {pool_name!r}; available pools: {sorted(pools)}"
        )
    pool = dict(pools[pool_name])
    pool["name"] = pool_name
    pool["config_path"] = str(path)
    return pool


def select_expert_pool_models(
    available_models: list[str],
    pool: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    available = set(available_models)
    entries = pool.get("models", [])
    metadata = {str(entry["name"]): dict(entry) for entry in entries}
    scale_group = pool.get("scale_group")

    eligible: list[str] = []
    excluded: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in entries:
        model_name = str(entry["name"])
        entry_scale = entry.get("scale_group", scale_group)
        if scale_group is not None and entry_scale != scale_group:
            raise ValueError(
                f"Model {model_name!r} uses scale group {entry_scale!r}, "
                f"expected {scale_group!r}"
            )
        reason = entry.get("exclude_reason")
        if reason:
            excluded.append({"model": model_name, "reason": str(reason)})
            continue
        if model_name not in available:
            missing.append(model_name)
            continue
        eligible.append(model_name)

    if pool.get("require_all_models") and missing:
        raise ValueError(f"Expert pool is missing evaluated models: {missing}")
    if not eligible:
        raise ValueError("The selected expert pool has no evaluated eligible models.")

    report = {
        "name": pool["name"],
        "config_path": pool["config_path"],
        "modality": pool.get("modality"),
        "scale_group": scale_group,
        "eligible_models": eligible,
        "excluded_models": excluded,
        "missing_evaluations": missing,
    }
    return eligible, report, metadata
