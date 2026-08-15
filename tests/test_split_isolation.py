from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bench_coe.data.preprocessing import detect_cross_dataset_overlaps


class SplitIsolationTests(unittest.TestCase):
    def test_exact_source_target_overlap_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"content_hash": "abc", "question": "same question", "context": "", "split": "x"}
            for name, role in (("source", "source_calibration"), ("target", "target_locked_test")):
                path = root / name
                path.mkdir()
                row = {**common, "dataset": name, "role": role}
                (path / "samples.jsonl").write_text(json.dumps(row) + "\n")
            overlaps = detect_cross_dataset_overlaps(root)
            self.assertTrue(any(item["type"] == "exact_content_hash" and item["role_conflict"] for item in overlaps))


if __name__ == "__main__":
    unittest.main()

