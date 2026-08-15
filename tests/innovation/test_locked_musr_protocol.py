from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from bench_coe.innovation.evaluate_locked_musr import _plot_results, _wilson_interval
from bench_coe.innovation.locked_protocol import (
    FORBIDDEN_TARGET_KEYS,
    audit_gpqa_units,
    create_label_access_marker,
    extract_musr_choice,
    load_protocol,
    parse_choices,
    read_musr_observable_rows,
    stratified_paired_bootstrap_delta,
    validate_protocol,
)
from bench_coe.innovation.locked_musr_labels import load_musr_evaluation_answers


class LockedMusrProtocolTests(unittest.TestCase):
    def _write_musr(
        self,
        path: Path,
        *,
        answer_index: int = 1,
        answer_choice: str = "right",
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["narrative", "question", "choices", "answer_index", "answer_choice"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "narrative": "A compact story.",
                    "question": "Which choice follows?",
                    "choices": json.dumps(["left", "right"]),
                    "answer_index": answer_index,
                    "answer_choice": answer_choice,
                }
            )

    def test_observable_materialization_is_invariant_to_target_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.csv"
            self._write_musr(path)
            spec = [{"task": "toy", "path": str(path), "expected_questions": 1}]
            before = read_musr_observable_rows(spec)
            self._write_musr(path, answer_index=0, answer_choice="left")
            after = read_musr_observable_rows(spec)
            self.assertEqual(before, after)
            self.assertTrue(FORBIDDEN_TARGET_KEYS.isdisjoint(before[0]))

    def test_evaluator_maps_answer_only_in_label_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.csv"
            self._write_musr(path)
            answers = load_musr_evaluation_answers(
                [{"task": "toy", "path": str(path), "expected_questions": 1}]
            )
            self.assertEqual(answers, {"musr::test::toy:0000": "b"})

    def test_label_access_marker_is_atomic_and_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = create_label_access_marker(root, "a" * 64)
            self.assertTrue(marker.exists())
            with self.assertRaises(FileExistsError):
                create_label_access_marker(root, "a" * 64)

    def test_frozen_parser_does_not_guess_from_arbitrary_letters(self) -> None:
        self.assertEqual(parse_choices('["one", "two"]'), ["one", "two"])
        self.assertEqual(extract_musr_choice("Work. Final answer: (B)", ["A", "B"]), "B")
        self.assertEqual(extract_musr_choice("Work.\nA", ["A", "B"]), "A")
        self.assertIsNone(extract_musr_choice("Because alpha follows beta.", ["A", "B"]))

    def test_stratified_bootstrap_is_deterministic_and_paired(self) -> None:
        candidate = {"a": True, "b": True, "c": False, "d": True}
        reference = {"a": False, "b": True, "c": False, "d": False}
        strata = {"a": "x", "b": "x", "c": "y", "d": "y"}
        first = stratified_paired_bootstrap_delta(
            candidate, reference, strata, seed=9, samples=500
        )
        second = stratified_paired_bootstrap_delta(
            candidate, reference, strata, seed=9, samples=500
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0.0)

    def test_real_preregistration_has_equal_budget_primary_and_valid_hashes(self) -> None:
        config = load_protocol(Path("configs/innovation/locked_musr_paper_v1.yaml"))
        validate_protocol(config)
        primary = config["hypotheses"]["primary"]
        methods = config["methods"]
        self.assertEqual(
            methods[primary["candidate"]]["nominal_model_calls"],
            methods[primary["reference"]]["nominal_model_calls"],
        )
        self.assertFalse(config["claim_boundary"]["pretraining_contamination_excluded"])

    def test_prediction_clis_do_not_import_evaluator_label_reader(self) -> None:
        for path in (
            Path("bench_coe/innovation/run_locked_musr_generation.py"),
            Path("bench_coe/innovation/run_locked_musr_selection.py"),
        ):
            self.assertNotIn("load_musr_evaluation_answers", path.read_text(encoding="utf-8"))

    def test_wilson_interval_contains_point_estimate(self) -> None:
        low, high = _wilson_interval(60, 100)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_result_plot_writes_png_and_pdf_before_label_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [
                {"method": "primary", "accuracy": 0.6, "wilson_ci95": [0.55, 0.65]},
                {"method": "reference", "accuracy": 0.5, "wilson_ci95": [0.45, 0.55]},
            ]
            comparisons = [
                {
                    "candidate": "primary",
                    "reference": "reference",
                    "delta": 0.1,
                    "stratified_paired_bootstrap_delta_ci95": [0.04, 0.16],
                }
            ]
            _plot_results(output, rows, comparisons, "primary", "reference")
            self.assertGreater((output / "locked_musr_results.png").stat().st_size, 0)
            self.assertGreater((output / "locked_musr_results.pdf").stat().st_size, 0)

    def test_gpqa_audit_counts_unique_units_and_permutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            cache = root / "cache" / "expert"
            raw.mkdir()
            cache.mkdir(parents=True)
            ids = {
                "diamond": ["a"],
                "main": ["a", "b"],
                "extended": ["a", "b", "c"],
            }
            for name, values in ids.items():
                with (raw / f"gpqa_{name}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=["Record ID"])
                    writer.writeheader()
                    for value in values:
                        writer.writerow({"Record ID": value})
            rows = [
                {"record_id": value, "epoch": epoch}
                for value in ("a", "b", "c")
                for epoch in range(2)
            ]
            (cache / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = audit_gpqa_units(root / "cache", raw)
            self.assertEqual(result["unique_record_ids_union"], 3)
            self.assertEqual(result["cached_rows"], 6)
            self.assertEqual(result["cached_unique_record_ids"], 3)
            self.assertEqual(result["pairwise_overlap"]["diamond_and_main"], 1)


if __name__ == "__main__":
    unittest.main()
