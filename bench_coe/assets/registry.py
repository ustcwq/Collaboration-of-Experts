from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import PROFILE_ORDER


VALID_KINDS = {"model", "dataset"}
VALID_MODALITIES = {"text", "vision_language", "mixed"}


@dataclass(frozen=True)
class ResourceSpec:
    logical_name: str
    kind: str
    candidate_modelscope_ids: tuple[str, ...]
    profiles: tuple[str, ...]
    required_profiles: tuple[str, ...]
    modality: str
    license: str
    revision: str | None = None
    search_queries: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def required(self) -> bool:
        return bool(self.required_profiles)

    def applies_to(self, profile: str) -> bool:
        return profile in self.profiles

    def required_for(self, profile: str) -> bool:
        return profile in self.required_profiles


class Registry:
    def __init__(self, resources: Iterable[ResourceSpec]):
        self.resources = tuple(resources)
        names = [item.logical_name for item in self.resources]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate logical_name entries in registry")

    @classmethod
    def load(cls, *paths: Path) -> "Registry":
        resources: list[ResourceSpec] = []
        for path in paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not str(payload.get("schema_version", "")).startswith("benchcoe_"):
                raise ValueError(f"Unsupported registry schema in {path}")
            for raw in payload.get("resources", []):
                resources.append(cls._parse(raw, path))
        return cls(resources)

    @staticmethod
    def _parse(raw: dict[str, Any], path: Path) -> ResourceSpec:
        required = {"logical_name", "kind", "candidate_modelscope_ids", "profiles", "modality", "license"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"{path}: registry entry missing {missing}")
        kind = str(raw["kind"])
        modality = str(raw["modality"])
        profiles = tuple(str(item) for item in raw["profiles"])
        required_profiles = tuple(str(item) for item in raw.get("required_profiles", []))
        if kind not in VALID_KINDS:
            raise ValueError(f"{path}: invalid kind {kind!r}")
        if modality not in VALID_MODALITIES:
            raise ValueError(f"{path}: invalid modality {modality!r}")
        if not profiles or any(item not in PROFILE_ORDER for item in profiles + required_profiles):
            raise ValueError(f"{path}: invalid profiles for {raw['logical_name']}")
        if not set(required_profiles).issubset(profiles):
            raise ValueError(f"{path}: required_profiles must be a subset of profiles")
        known = {
            "logical_name", "kind", "candidate_modelscope_ids", "profiles", "required_profiles",
            "modality", "license", "revision", "search_queries",
        }
        return ResourceSpec(
            logical_name=str(raw["logical_name"]),
            kind=kind,
            candidate_modelscope_ids=tuple(str(item) for item in raw["candidate_modelscope_ids"]),
            profiles=profiles,
            required_profiles=required_profiles,
            modality=modality,
            license=str(raw["license"]),
            revision=str(raw["revision"]) if raw.get("revision") is not None else None,
            search_queries=tuple(str(item) for item in raw.get("search_queries", [])),
            metadata={key: value for key, value in raw.items() if key not in known},
        )

    def select(self, profile: str, only: set[str] | None = None, exclude: set[str] | None = None) -> list[ResourceSpec]:
        if profile not in PROFILE_ORDER:
            raise ValueError(f"Unknown profile {profile!r}")
        only = only or set()
        exclude = exclude or set()
        unknown = (only | exclude) - {item.logical_name for item in self.resources}
        if unknown:
            raise ValueError(f"Unknown logical resources: {sorted(unknown)}")
        return [item for item in self.resources if item.applies_to(profile) and (not only or item.logical_name in only) and item.logical_name not in exclude]

