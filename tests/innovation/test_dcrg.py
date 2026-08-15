from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench_coe.innovation.data import CacheAdapter, EvaluationLabelAdapter
from bench_coe.innovation.dcrg import (
    DCRGSelector,
    cross_fitted_expected_correctness,
    estimate_rescue_graphs,
    make_oof_splits,
    randomize_outgoing_edges,
    validate_oof_splits,
)
from registry_fixture import write_registry


class DCRGCoreTests(unittest.TestCase):
    def test_oof_row_correctness_cannot_change_its_nuisance_prediction(self) -> None:
        correctness = np.asarray([[row % 2, (row + 1) % 2, row % 3 == 0] for row in range(12)], dtype=float)
        environments = np.asarray(["a"] * 6 + ["b"] * 6)
        observables = np.asarray([[row / 12, (row % 4) / 4] for row in range(12)], dtype=float)
        expected, splits = cross_fitted_expected_correctness(correctness, environments, observables, folds=3, seed=7)
        row = int(splits[0][1][0])
        modified = correctness.copy()
        modified[row] = 1.0 - modified[row]
        changed, _ = cross_fitted_expected_correctness(modified, environments, observables, folds=3, seed=7)
        self.assertTrue(np.allclose(expected[:, :, row], changed[:, :, row]))

    def test_oof_splits_cover_once_and_overlap_fails(self) -> None:
        environments = np.asarray(["a"] * 6 + ["b"] * 6)
        splits = make_oof_splits(environments, folds=3, seed=7)
        validate_oof_splits(splits, 12)
        with self.assertRaises(ValueError):
            validate_oof_splits([(np.asarray([0, 1]), np.asarray([1, 2]))], 3)

    def test_graph_has_zero_self_loops_and_extremes_are_finite(self) -> None:
        correctness = np.asarray([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=float)
        environments = np.asarray(["a", "a", "b", "b", "c", "c"])
        observable = np.asarray([[index / 6, (index % 2)] for index in range(6)], dtype=float)
        expected, _ = cross_fitted_expected_correctness(correctness, environments, observable, folds=2, seed=3)
        graphs, edges = estimate_rescue_graphs(
            correctness,
            expected,
            environments,
            ("x", "y", "z"),
            min_support=1,
            min_environments=1,
        )
        self.assertTrue(edges)
        for graph in graphs.values():
            self.assertTrue(np.isfinite(graph).all())
            self.assertTrue(np.array_equal(np.diag(graph), np.zeros(3)))

    def test_random_graph_preserves_out_degree_and_weight_multiset(self) -> None:
        graph = np.asarray([[0, 0.1, 0.2, 0], [0.4, 0, 0, 0.3], [0, 0.5, 0, 0], [0.7, 0, 0.8, 0]], dtype=float)
        randomized = randomize_outgoing_edges(graph, seed=11)
        for source in range(len(graph)):
            self.assertEqual(int(np.sum(graph[source] > 0)), int(np.sum(randomized[source] > 0)))
            self.assertEqual(sorted(graph[source][graph[source] > 0]), sorted(randomized[source][randomized[source] > 0]))
            self.assertEqual(randomized[source, source], 0.0)


class DCRGSelectorTests(unittest.TestCase):
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
                prediction = gold if (index + expert_index) % 3 else ("b" if gold == "a" else "a")
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

    def test_seed_is_reproducible_and_scores_are_finite(self) -> None:
        batch = self.adapter.load_observables()
        labels = self.adapter.load_source_labels()
        first = DCRGSelector(seed=9, folds=2, min_support=1, min_environments=1).fit(batch, labels)
        second = DCRGSelector(seed=9, folds=2, min_support=1, min_environments=1).fit(batch, labels)
        self.assertTrue(np.array_equal(first.graphs_["stable"], second.graphs_["stable"]))
        predictions = first.predict(batch)
        self.assertEqual(len(predictions), len(batch.question_ids))
        self.assertTrue(all(np.isfinite(list(item.expert_scores.values())).all() for item in predictions))

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
            DCRGSelector(folds=2).fit(batch, labels)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
