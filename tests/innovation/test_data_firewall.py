from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from bench_coe.innovation.data import (
    CacheAdapter,
    EvaluationLabelAdapter,
    assert_disjoint,
    export_label_free_observables,
    normalize_answer,
)
from bench_coe.innovation.schema import CanonicalPredictionRecord, EvaluationLabels
from bench_coe.innovation.schema import SourceTrainingLabels
from bench_coe.innovation.selectors import SourceBestSelector
from registry_fixture import write_registry


FAMILIES = {"expert-a": "family-a", "expert-b": "family-b"}


class CacheFixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def write(self, expert: str, rows: list[dict]) -> None:
        model_dir = self.root / expert
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")

    def close(self) -> None:
        self.temp.cleanup()


def row(question_id: str, prediction: str, correct: bool, subject: str = "s") -> dict:
    return {
        "id": question_id,
        "subject": subject,
        "answer": "a",
        "prediction": prediction,
        "response": prediction,
        "is_correct": correct,
        "model_error": None,
        "model_latency_seconds": 1.0,
    }


class DataFirewallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = CacheFixture()
        self.fixture.write("expert-a", [row("q1", "A", True), row("q2", "B", False)])
        self.fixture.write("expert-b", [row("q1", " a. ", True)])
        self.source_registry, self.source_registry_hash = write_registry(
            self.fixture.root, self.fixture.root, "toy", "source", "language", "source"
        )
        self.target_registry, self.target_registry_hash = write_registry(
            self.fixture.root, self.fixture.root, "toy", "target", "language", "target"
        )

    def tearDown(self) -> None:
        self.fixture.close()

    def adapter(self, experts: list[str] | None = None) -> CacheAdapter:
        return CacheAdapter.from_source_registry(
            self.fixture.root,
            "toy",
            "source",
            "language",
            FAMILIES,
            experts or ["expert-a", "expert-b"],
            self.source_registry,
            self.source_registry_hash,
        )

    def test_observable_schema_contains_no_correctness_or_gold(self) -> None:
        names = {item.name for item in fields(CanonicalPredictionRecord)}
        self.assertTrue({"correctness", "is_correct", "gold", "target"}.isdisjoint(names))
        batch = self.adapter().load_observables()
        for record in batch.records:
            self.assertTrue({"answer", "is_correct", "gold", "target"}.isdisjoint(record.observable_metadata))

    def test_safe_question_and_option_metadata_survives_projection(self) -> None:
        self.fixture.write(
            "expert-a",
            [
                {
                    **row("q1", "A", True),
                    "question": "Which option?",
                    "options": ["first", "second"],
                    "base_question_id": 7,
                    "epoch": 1,
                }
            ],
        )
        record = self.adapter(experts=["expert-a"]).load_observables().records[0]
        self.assertEqual(record.observable_metadata["question"], "Which option?")
        self.assertEqual(record.observable_metadata["options"], ["first", "second"])
        self.assertEqual(record.observable_metadata["base_question_id"], 7)
        self.assertEqual(record.observable_metadata["epoch"], 1)

    def test_target_evaluation_labels_cannot_fit_selector(self) -> None:
        adapter = self.adapter()
        batch = adapter.load_observables()
        target_labels = EvaluationLabelAdapter.from_registry(
            self.fixture.root,
            "toy",
            "source",
            "language",
            FAMILIES,
            self.source_registry,
            self.source_registry_hash,
        ).load()
        self.assertIsInstance(target_labels, EvaluationLabels)
        with self.assertRaises(TypeError):
            SourceBestSelector().fit(batch, target_labels)  # type: ignore[arg-type]

    def test_target_cache_cannot_be_forged_as_source_by_caller_role(self) -> None:
        with self.assertRaises(PermissionError):
            CacheAdapter.from_source_registry(
                self.fixture.root,
                "toy",
                "target",
                "language",
                FAMILIES,
                FAMILIES,
                self.target_registry,
                self.target_registry_hash,
            )
        with self.assertRaises(PermissionError):
            CacheAdapter.from_target_observables(
                self.fixture.root,
                "toy",
                "target",
                "language",
                FAMILIES,
                FAMILIES,
                "0" * 64,
            )
        with self.assertRaises(PermissionError):
            CacheAdapter(
                self.fixture.root,
                "toy",
                "target",
                "language",
                FAMILIES,
                FAMILIES,
                _validated_role="source",
                _capability=object(),
            )

    def test_source_training_labels_cannot_be_forged_directly(self) -> None:
        with self.assertRaises(ValueError):
            SourceTrainingLabels("toy", "target", {}, {})

    def test_present_null_prediction_stays_invalid(self) -> None:
        self.fixture.write("expert-a", [{**row("q1", "A", False), "prediction": None, "response": "A"}])
        record = self.adapter(experts=["expert-a"]).load_observables().records[0]
        self.assertFalse(record.valid_output)
        self.assertIsNone(record.normalized_answer)

    def test_exported_target_cache_is_physically_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observables"
            export_label_free_observables(
                self.fixture.root,
                output,
                "toy",
                "target",
                "language",
                FAMILIES,
                FAMILIES,
                self.target_registry,
                self.target_registry_hash,
            )
            for path in output.glob("*/observables.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    self.assertTrue({"answer", "gold", "target", "is_correct", "score"}.isdisjoint(json.loads(line)))
            manifest_path = output / "observable_manifest.json"
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            target = CacheAdapter.from_target_observables(
                output, "toy", "target", "language", FAMILIES, FAMILIES, manifest_hash
            )
            self.assertEqual(len(target.load_observables().question_ids), 2)
            with self.assertRaises(PermissionError):
                target.load_source_labels()

            observable = output / "expert-a" / "observables.jsonl"
            observable.write_text(observable.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                CacheAdapter.from_target_observables(
                    output, "toy", "target", "language", FAMILIES, FAMILIES, manifest_hash
                )

    def test_expert_order_does_not_change_clusters(self) -> None:
        first = self.adapter(experts=["expert-a", "expert-b"]).load_observables()
        second = self.adapter(experts=["expert-b", "expert-a"]).load_observables()
        first_clusters = [(cluster.question_id, cluster.cluster_id, cluster.expert_ids) for qid in first.question_ids for cluster in first.clusters(qid)]
        second_clusters = [(cluster.question_id, cluster.cluster_id, cluster.expert_ids) for qid in second.question_ids for cluster in second.clusters(qid)]
        self.assertEqual(first_clusters, second_clusters)

    def test_normalization_and_query_local_clusters(self) -> None:
        self.assertEqual(normalize_answer(" A. "), "a")
        batch = self.adapter().load_observables()
        q1, q2 = batch.question_ids
        self.assertEqual(len(batch.clusters(q1)), 1)
        self.assertEqual(batch.clusters(q1)[0].expert_ids, ("expert-a", "expert-b"))
        self.assertEqual(batch.clusters(q2)[0].question_id, q2)
        self.assertNotEqual(batch.clusters(q1)[0].question_id, batch.clusters(q2)[0].question_id)

    def test_missing_output_is_not_an_answer_cluster(self) -> None:
        batch = self.adapter().load_observables()
        q2 = batch.question_ids[1]
        missing = next(record for record in batch.for_question(q2) if record.expert_id == "expert-b")
        self.assertFalse(missing.valid_output)
        self.assertIsNone(missing.normalized_answer)
        self.assertIsNone(missing.per_query_cluster_id)
        self.assertEqual(len(batch.clusters(q2)), 1)

    def test_deterministic_loading_and_variable_pool(self) -> None:
        first = self.adapter().load_observables()
        second = self.adapter().load_observables()
        self.assertEqual(first, second)
        smaller = self.adapter(experts=["expert-a"]).load_observables()
        self.assertEqual(smaller.pool.expert_ids, ("expert-a",))
        self.assertEqual(len(smaller.records), 2)

    def test_nested_per_environment_source_cache_is_loaded(self) -> None:
        expert = "nested-expert"
        nested = self.fixture.root / expert / "CoT" / "source"
        nested.mkdir(parents=True)
        (nested / "part-a.json").write_text(
            json.dumps([row("nested-q1", "A", True, "subject-a")]),
            encoding="utf-8",
        )
        (nested / "part-b.json").write_text(
            json.dumps([row("nested-q2", "B", False, "subject-b")]),
            encoding="utf-8",
        )
        adapter = CacheAdapter.from_source_registry(
            self.fixture.root,
            "toy",
            "source",
            "language",
            {expert: "nested-family"},
            [expert],
            self.source_registry,
            self.source_registry_hash,
        )
        batch = adapter.load_observables()
        labels = adapter.load_source_labels()
        self.assertEqual(
            batch.question_ids,
            ("toy::source::nested-q1", "toy::source::nested-q2"),
        )
        self.assertTrue(labels.get("toy::source::nested-q1", expert))
        self.assertFalse(labels.get("toy::source::nested-q2", expert))

    def test_duplicate_question_id_is_rejected(self) -> None:
        self.fixture.write("expert-a", [row("q1", "A", True), row("q1", "B", False)])
        with self.assertRaises(ValueError):
            self.adapter().load_observables()

    def test_source_target_raw_overlap_within_dataset_is_rejected(self) -> None:
        source = self.adapter().load_observables()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observables"
            export_label_free_observables(
                self.fixture.root,
                output,
                "toy",
                "target",
                "language",
                FAMILIES,
                FAMILIES,
                self.target_registry,
                self.target_registry_hash,
            )
            manifest_hash = hashlib.sha256((output / "observable_manifest.json").read_bytes()).hexdigest()
            target = CacheAdapter.from_target_observables(
                output, "toy", "target", "language", FAMILIES, FAMILIES, manifest_hash
            ).load_observables()
            with self.assertRaises(ValueError):
                assert_disjoint(source, target)


if __name__ == "__main__":
    unittest.main()
