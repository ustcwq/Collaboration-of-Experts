from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench_coe.assets.registry import Registry


ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_repository_registries_validate(self):
        registry = Registry.load(ROOT / "configs/modelscope_models.yaml", ROOT / "configs/modelscope_datasets.yaml")
        self.assertGreaterEqual(len(registry.resources), 30)
        smoke = registry.select("smoke")
        self.assertIn("qwen3_1_7b", {item.logical_name for item in smoke})
        self.assertNotIn("qwen3_32b", {item.logical_name for item in smoke})

    def test_invalid_registry_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("schema_version: benchcoe_bad_v1\nresources:\n  - logical_name: broken\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Registry.load(path)


if __name__ == "__main__":
    unittest.main()

