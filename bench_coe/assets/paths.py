from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROFILE_ORDER = {"smoke": 0, "core": 1, "full": 2}


@dataclass(frozen=True)
class AssetPaths:
    root: Path
    modelscope_cache: Path
    dataset_root: Path | None = None

    @classmethod
    def from_env(cls, asset_root: str | None = None, cache: str | None = None) -> "AssetPaths":
        root_value = asset_root or os.environ.get("BENCHCOE_ASSET_ROOT")
        cache_value = cache or os.environ.get("MODELSCOPE_CACHE")
        if not root_value:
            root_value = str(Path.cwd().parent / "benchcoe_assets")
        root = Path(root_value).expanduser().resolve()
        if root == Path(root.anchor) or root == Path.home().resolve():
            raise ValueError("BENCHCOE_ASSET_ROOT may not be the filesystem root or home directory")
        cache_path = Path(cache_value).expanduser().resolve() if cache_value else root / "modelscope_cache"
        if cache_path == Path(cache_path.anchor) or cache_path == Path.home().resolve():
            raise ValueError("MODELSCOPE_CACHE may not be the filesystem root or home directory")
        dataset_value = os.environ.get("BENCHCOE_DATA_ROOT")
        dataset_root = Path(dataset_value).expanduser().resolve() if dataset_value else None
        return cls(root=root, modelscope_cache=cache_path, dataset_root=dataset_root)

    def ensure(self) -> None:
        for path in self.directories().values():
            path.mkdir(parents=True, exist_ok=True)

    def directories(self) -> dict[str, Path]:
        return {
            "models": self.root / "models",
            "datasets_raw": self.dataset_root or self.root / "datasets_raw",
            "datasets_processed": self.root / "datasets_processed",
            "image_assets": self.root / "image_assets",
            "locks": self.root / "locks",
            "manifests": self.root / "manifests",
            "logs": self.root / "logs",
            "quarantine": self.root / "quarantine",
            "tmp": self.root / ".tmp",
            "modelscope_cache": self.modelscope_cache,
        }

    def resource_dir(self, kind: str, logical_name: str) -> Path:
        parent = "models" if kind == "model" else "datasets_raw"
        return self.directories()[parent] / logical_name

    def relative_to_root(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.root))
        except ValueError:
            if self.dataset_root is not None:
                try:
                    return str(Path("data") / resolved.relative_to(self.dataset_root))
                except ValueError:
                    pass
            return str(resolved)
