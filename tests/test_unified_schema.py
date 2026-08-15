from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bench_coe.data.preprocessing import convert_row, qa_samples, stable_smoke_subset, write_processed_dataset
from bench_coe.data.schema import canonicalize_answer, canonicalize_choices


class UnifiedSchemaTests(unittest.TestCase):
    def test_choices_and_answer_mapping_preserve_order(self):
        choices = canonicalize_choices(["zero", "one", "two"])
        self.assertEqual([item["label"] for item in choices], ["A", "B", "C"])
        self.assertEqual(canonicalize_answer("one", choices), "B")
        self.assertEqual(canonicalize_answer(2, choices), "C")

    def test_numpy_values_are_normalized(self):
        import numpy as np
        choices = canonicalize_choices(np.array(["zero", "one"], dtype=object))
        self.assertEqual([item["text"] for item in choices], ["zero", "one"])
        self.assertEqual(canonicalize_answer(np.int64(1), choices), "B")

    def test_multimage_order_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
                Image.new("RGB", (4 + index, 5 + index), color).save(root / f"{index}.png")
            sample = convert_row(
                {"id": "x", "question": "Which?", "choices": ["A1", "A2"], "answer": "A", "images": ["0.png", "1.png"]},
                "vision", "rev", "test", "target_locked_test", "vision_language", "multiple_choice", "test", root, root / "assets",
            )
            self.assertEqual([item["index"] for item in sample.images], [0, 1])
            self.assertNotEqual(sample.images[0]["sha256"], sample.images[1]["sha256"])
            self.assertEqual(sample.images[0]["width"], 4)

    def test_embedded_image_bytes(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            Image.new("RGB", (3, 2), (1, 2, 3)).save(buffer, format="PNG")
            sample = convert_row(
                {"id": "bytes", "question": "Q", "answer": "ok", "image": {"bytes": buffer.getvalue(), "path": None}},
                "vision", "rev", "test", "secondary_test", "vision_language", "exact_match", "test", root, root / "assets",
            )
            self.assertEqual(sample.images[0]["width"], 3)
            self.assertEqual(sample.images[0]["height"], 2)

    def test_duplicate_detection_and_stable_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"id": "same", "question": "Q", "choices": ["x", "y"], "answer": "A"}
            samples = [convert_row(row, "d", "r", "validation", "source_calibration", "text", "multiple_choice", "test", root, root / "images") for _ in range(2)]
            qa, overlaps = qa_samples(samples)
            self.assertEqual(qa["duplicate_sample_ids"], 1)
            self.assertTrue(overlaps)
            self.assertEqual([item.sample_id for item in stable_smoke_subset(samples, 1)], [samples[0].sample_id])

    def test_smoke_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [convert_row({"id": i, "question": f"Q{i}", "choices": ["x", "y"], "answer": "A"}, "d", "r", "validation", "source_calibration", "text", "multiple_choice", "test", root, root / "images") for i in range(12)]
            raw = root / "raw.jsonl"
            raw.write_text("{}\n")
            manifest = write_processed_dataset(samples, root / "out", [raw], smoke=True)
            self.assertEqual(manifest["sample_count"], 8)
            self.assertTrue((root / "out/dataset_manifest.json").exists())
            self.assertEqual(len((root / "out/samples.jsonl").read_text().splitlines()), 8)
            self.assertIn("Smoke subset: true", (root / "out/README.generated.md").read_text())

    def test_preselected_smoke_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [convert_row({"id": i, "question": f"Q{i}", "answer": "ok"}, "d", "r", "validation", "source_calibration", "text", "exact_match", "test", root, root / "images") for i in range(3)]
            raw = root / "raw.jsonl"
            raw.write_text("{}\n")
            manifest = write_processed_dataset(samples, root / "out", [raw], smoke=True, preselected=True, selection="raw-row-hash")
            self.assertEqual(manifest["sample_count"], 3)
            self.assertEqual(manifest["selection"], "raw-row-hash")


if __name__ == "__main__":
    unittest.main()
