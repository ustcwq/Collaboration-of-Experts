from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench_coe.assets.modelscope_backend import ModelScopeBackend, download_resource
from bench_coe.assets.paths import AssetPaths
from bench_coe.assets.registry import ResourceSpec


class FakeApi:
    def get_model(self, model_id, revision=None):
        return {"revision": revision or "abc123", "license": "test-license"}

    def get_model_files(self, model_id, revision=None, recursive=True):
        return [{"Path": "config.json", "Size": 12}, {"Path": "weights.bin", "Size": 20}]


class FakeDownloadBackend:
    def __init__(self):
        self.calls = 0

    def snapshot_download(self, kind, resource_id, revision, cache_dir, local_dir, max_workers):
        self.calls += 1
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "payload.bin").write_bytes(b"resume-safe")
        return str(local_dir)


def spec() -> ResourceSpec:
    return ResourceSpec("tiny", "model", ("Org/Tiny",), ("smoke",), ("smoke",), "text", "unknown")


class BackendTests(unittest.TestCase):
    def test_mock_resolution(self):
        result = ModelScopeBackend(FakeApi()).resolve(spec())
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.download_bytes, 32)
        self.assertEqual(result.resolved_modelscope_id, "Org/Tiny")

    def test_existing_ready_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AssetPaths(Path(directory) / "assets", Path(directory) / "cache")
            final = paths.resource_dir("model", "tiny")
            final.mkdir(parents=True)
            (final / ".benchcoe_ready.json").write_text("{}")
            backend = FakeDownloadBackend()
            result = download_resource(spec(), {"status": "resolved", "resolved_modelscope_id": "Org/Tiny", "revision": "abc"}, paths, backend)
            self.assertEqual(result["status"], "already_ready")
            self.assertEqual(backend.calls, 0)

    def test_incomplete_staging_is_resumed_and_published(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = AssetPaths(Path(directory) / "assets", Path(directory) / "cache")
            paths.ensure()
            staging = paths.directories()["tmp"] / "tiny.incomplete"
            staging.mkdir()
            (staging / "partial.bin").write_bytes(b"partial")
            backend = FakeDownloadBackend()
            result = download_resource(spec(), {"status": "resolved", "resolved_modelscope_id": "Org/Tiny", "revision": "abc"}, paths, backend, retries=0)
            self.assertEqual(result["status"], "downloaded")
            self.assertTrue((paths.resource_dir("model", "tiny") / ".benchcoe_ready.json").exists())
            self.assertEqual(backend.calls, 1)


if __name__ == "__main__":
    unittest.main()

