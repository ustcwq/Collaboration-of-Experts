from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench_coe.innovation.data import CacheAdapter, EvaluationLabelAdapter
from bench_coe.innovation.repair_simplification import (
    ABLATION_METHODS,
    POOL_SHIFT_METHODS,
    RepairComponents,
    expert_score_matrix,
    fit_repair_components,
    graph_variant,
    selections_from_components,
    subset_expert_pool,
)
from bench_coe.innovation.schema import ExpertPool, ObservableQueryBatch
from bench_coe.innovation.selectors import LegacyRepairChainSelector
from registry_fixture import write_registry


class RepairSimplificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        experts = ("e1", "e2", "e3")
        for expert_index, expert in enumerate(experts):
            model_dir = root / expert
            model_dir.mkdir()
            rows = []
            for index in range(18):
                gold = "a" if index % 2 == 0 else "b"
                prediction = gold if (index + expert_index) % 4 else ("b" if gold == "a" else "a")
                rows.append(
                    {
                        "id": str(index),
                        "subject": f"s{index % 3}",
                        "answer": gold,
                        "prediction": prediction,
                        "response": prediction,
                        "is_correct": prediction == gold,
                    }
                )
            (model_dir / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")
        self.registry, self.registry_hash = write_registry(root, root, "toy", "source", "language")
        self.adapter = CacheAdapter.from_source_registry(
            root,
            "toy",
            "source",
            "language",
            {"e1": "f1", "e2": "f2", "e3": "f3"},
            experts,
            self.registry,
            self.registry_hash,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_m0_matches_legacy_repair_chain_on_complete_pool(self) -> None:
        batch = self.adapter.load_observables()
        labels = self.adapter.load_source_labels()
        train_ids = batch.question_ids[:12]
        test_ids = batch.question_ids[12:]
        train_batch = batch.subset(train_ids)
        test_batch = batch.subset(test_ids)
        train_labels = labels.subset(train_ids)
        components = fit_repair_components(train_batch, train_labels, test_batch, neighbors=4)
        simplified = selections_from_components(test_batch, components, "m0_full")
        legacy = LegacyRepairChainSelector(neighbors=4).fit(train_batch, train_labels).predict(test_batch)
        self.assertEqual(
            [item.normalized_answer for item in simplified],
            [item.normalized_answer for item in legacy],
        )

    def test_formula_ablation_matrices_are_exact(self) -> None:
        components = RepairComponents(
            question_ids=("q",),
            expert_ids=("e1", "e2"),
            local=np.asarray([[0.2, 0.8]]),
            support=np.asarray([[0.5, 0.5]]),
            uncertainty=np.zeros((1, 2)),
            global_accuracy=np.asarray([0.4, 0.6]),
            repair_graph=np.asarray([[0.0, 0.3], [0.2, 0.0]]),
            failure_weights=np.asarray([[0.5, 0.5]]),
            hop1=np.asarray([[0.1, 0.3]]),
            hop2=np.asarray([[0.4, 0.2]]),
            valid_mask=np.ones((1, 2), dtype=bool),
            cluster_ids=np.asarray([[0, 0]]),
        )
        expected_m0 = (
            0.30 * components.local
            + 0.25 * components.hop1
            + 0.18 * components.hop2
            + 0.16 * components.support
            + 0.11 * components.global_accuracy[None, :]
        )
        self.assertTrue(np.allclose(expert_score_matrix("m0_full", components), expected_m0))
        self.assertTrue(
            np.allclose(
                expert_score_matrix("m3_h1_support", components, beta=0.75),
                0.75 * components.hop1 + 0.25 * components.support,
            )
        )

    def test_all_registered_ablations_produce_finite_selections(self) -> None:
        batch = self.adapter.load_observables()
        labels = self.adapter.load_source_labels()
        train_ids = batch.question_ids[:12]
        test_batch = batch.subset(batch.question_ids[12:])
        components = fit_repair_components(
            batch.subset(train_ids), labels.subset(train_ids), test_batch, neighbors=4
        )
        for method in ABLATION_METHODS:
            predictions = selections_from_components(test_batch, components, method, beta=0.6, alpha=0.7)
            self.assertEqual(len(predictions), len(test_batch.question_ids), method)
            self.assertTrue(
                all(np.isfinite(list(prediction.expert_scores.values())).all() for prediction in predictions),
                method,
            )

    def test_cluster_formula_uses_mean_h1_and_pool_denominator(self) -> None:
        batch = self.adapter.load_observables().subset(("toy::source::0",))
        components = RepairComponents(
            question_ids=batch.question_ids,
            expert_ids=batch.pool.expert_ids,
            local=np.zeros((1, 3)),
            support=np.asarray([[2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]]),
            uncertainty=np.zeros((1, 3)),
            global_accuracy=np.asarray([0.4, 0.5, 0.6]),
            repair_graph=np.zeros((3, 3)),
            failure_weights=np.full((1, 3), 1.0 / 3.0),
            hop1=np.asarray([[0.2, 0.4, 0.5]]),
            hop2=np.zeros((1, 3)),
            valid_mask=np.ones((1, 3), dtype=bool),
            cluster_ids=np.asarray([[0, 0, 1]]),
        )
        prediction = selections_from_components(
            batch, components, "m3_cluster_h1_support", beta=0.5
        )[0]
        self.assertAlmostEqual(prediction.cluster_scores["0"], 0.5 * 0.3 + 0.5 * (2.0 / 3.0))
        self.assertAlmostEqual(prediction.cluster_scores["1"], 0.5 * 0.5 + 0.5 * (1.0 / 3.0))
        self.assertEqual(prediction.selected_cluster_id, 0)

    def test_missing_output_has_zero_support_and_no_cluster(self) -> None:
        batch = self.adapter.load_observables()
        question_id = batch.question_ids[12]
        changed_records = tuple(
            replace(
                record,
                raw_answer=None,
                normalized_answer=None,
                per_query_cluster_id=None,
                valid_output=False,
                missing_reason="test_missing",
            )
            if record.question_id == question_id and record.expert_id == "e1"
            else record
            for record in batch.records
        )
        changed = ObservableQueryBatch(batch.dataset, batch.split, batch.modality, batch.pool, changed_records)
        labels = self.adapter.load_source_labels()
        train_ids = batch.question_ids[:12]
        components = fit_repair_components(
            batch.subset(train_ids), labels.subset(train_ids), changed.subset((question_id,)), neighbors=4
        )
        missing_index = components.expert_ids.index("e1")
        self.assertEqual(components.support[0, missing_index], 0.0)
        self.assertEqual(components.cluster_ids[0, missing_index], -1)
        self.assertFalse(components.valid_mask[0, missing_index])

    def test_pool_projection_supports_all_shift_methods(self) -> None:
        batch = self.adapter.load_observables()
        labels = self.adapter.load_source_labels()
        projected, projected_labels = subset_expert_pool(batch, labels, ("e1", "e3"))
        self.assertEqual(projected.pool.expert_ids, ("e1", "e3"))
        train_ids = projected.question_ids[:12]
        test_batch = projected.subset(projected.question_ids[12:])
        components = fit_repair_components(
            projected.subset(train_ids), projected_labels.subset(train_ids), test_batch, neighbors=4
        )
        for method in POOL_SHIFT_METHODS:
            predictions = selections_from_components(test_batch, components, method, beta=0.5, alpha=0.5)
            self.assertEqual(len(predictions), len(test_batch.question_ids), method)
        self.assertTrue(np.array_equal(expert_score_matrix("m4_h1", components), components.hop1))
        self.assertTrue(
            np.allclose(
                expert_score_matrix("m5_h1_h2", components, alpha=0.25),
                0.25 * components.hop1 + 0.75 * components.hop2,
            )
        )

    def test_graph_controls_are_deterministic_and_preserve_randomized_rows(self) -> None:
        graph = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        first = graph_variant(graph, "randomized", seed=13)
        second = graph_variant(graph, "randomized", seed=13)
        self.assertTrue(np.array_equal(first, second))
        for source in range(len(graph)):
            destinations = [index for index in range(len(graph)) if index != source]
            self.assertEqual(sorted(first[source, destinations]), sorted(graph[source, destinations]))
            self.assertEqual(first[source, source], graph[source, source])
        no_self = graph_variant(graph, "no_self", seed=0)
        self.assertTrue(np.array_equal(np.diag(no_self), np.zeros(3)))
        symmetric = graph_variant(graph, "symmetric", seed=0)
        self.assertTrue(np.allclose(symmetric, symmetric.T))
        centrality = graph_variant(graph, "column_centrality", seed=0)
        self.assertTrue(np.allclose(centrality[0], centrality[1]))

    def test_cluster_scoring_is_permutation_invariant(self) -> None:
        batch = self.adapter.load_observables()
        labels = self.adapter.load_source_labels()
        train_ids = batch.question_ids[:12]
        test_ids = batch.question_ids[12:]
        first_components = fit_repair_components(
            batch.subset(train_ids), labels.subset(train_ids), batch.subset(test_ids), neighbors=4
        )
        first = selections_from_components(
            batch.subset(test_ids), first_components, "m3_cluster_h1_support", beta=0.5
        )

        reverse_experts = tuple(reversed(batch.pool.expert_ids))
        reverse_pool = ExpertPool(reverse_experts, batch.pool.family_by_expert)
        reverse_batch = ObservableQueryBatch(batch.dataset, batch.split, batch.modality, reverse_pool, batch.records)
        reverse_components = fit_repair_components(
            reverse_batch.subset(train_ids),
            labels.subset(train_ids),
            reverse_batch.subset(test_ids),
            neighbors=4,
        )
        second = selections_from_components(
            reverse_batch.subset(test_ids), reverse_components, "m3_cluster_h1_support", beta=0.5
        )
        self.assertEqual(
            [item.normalized_answer for item in first],
            [item.normalized_answer for item in second],
        )

    def test_evaluation_labels_are_rejected(self) -> None:
        batch = self.adapter.load_observables()
        labels = EvaluationLabelAdapter.from_registry(
            self.adapter.cache_path,
            "toy",
            "source",
            "language",
            self.adapter.expert_ids,
            self.registry,
            self.registry_hash,
        ).load()
        with self.assertRaises(TypeError):
            fit_repair_components(batch, labels, batch, neighbors=4)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
