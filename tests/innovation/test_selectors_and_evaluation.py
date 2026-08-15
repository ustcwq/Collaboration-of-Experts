from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench_coe.innovation.data import CacheAdapter, EvaluationLabelAdapter
from bench_coe.innovation.evaluation import evaluate, hierarchical_paired_bootstrap, holm_adjust
from bench_coe.innovation.selectors import MajorityVoteSelector, SourceBestSelector
from registry_fixture import write_registry


class SelectorEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        rows = {
            "e1": [
                {"id": "1", "subject": "x", "answer": "a", "prediction": "a", "response": "a", "is_correct": True},
                {"id": "2", "subject": "y", "answer": "b", "prediction": "a", "response": "a", "is_correct": False},
            ],
            "e2": [
                {"id": "1", "subject": "x", "answer": "a", "prediction": "b", "response": "b", "is_correct": False},
                {"id": "2", "subject": "y", "answer": "b", "prediction": "b", "response": "b", "is_correct": True},
            ],
            "e3": [
                {"id": "1", "subject": "x", "answer": "a", "prediction": "a", "response": "a", "is_correct": True},
                {"id": "2", "subject": "y", "answer": "b", "prediction": "b", "response": "b", "is_correct": True},
            ],
        }
        for expert, expert_rows in rows.items():
            model_dir = root / expert
            model_dir.mkdir()
            (model_dir / "predictions.json").write_text(json.dumps(expert_rows), encoding="utf-8")
        self.registry, self.registry_hash = write_registry(root, root, "toy", "eval", "language")
        self.adapter = CacheAdapter.from_source_registry(
            root,
            "toy",
            "eval",
            "language",
            {"e1": "f1", "e2": "f2", "e3": "f3"},
            ("e1", "e2", "e3"),
            self.registry,
            self.registry_hash,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_permutation_invariant_majority_and_metrics(self) -> None:
        batch = self.adapter.load_observables()
        source_labels = self.adapter.load_source_labels()
        evaluation_labels = EvaluationLabelAdapter.from_registry(
            self.adapter.cache_path,
            "toy",
            "eval",
            "language",
            self.adapter.expert_ids,
            self.registry,
            self.registry_hash,
        ).load()
        baseline = SourceBestSelector().fit(batch, source_labels).predict(batch)
        majority = MajorityVoteSelector().fit(batch, source_labels).predict(batch)
        summary, rows = evaluate("majority", majority, baseline, batch, evaluation_labels, bootstrap_samples=100, seed=7)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(len(rows), 2)
        self.assertIn("rescue_count", summary)
        self.assertIn("exact_mcnemar_p", summary)

    def test_holm_is_monotone_and_bounded(self) -> None:
        adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})
        values = [adjusted[key]["holm_adjusted_p"] for key in ("a", "b", "c")]
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in values))
        self.assertEqual(values, sorted(values))

    def test_crossed_bootstrap_uses_one_shared_query_resample(self) -> None:
        candidate = np.asarray([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=float)
        reference = np.asarray([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=float)
        observed = hierarchical_paired_bootstrap(candidate, reference, seed=17, samples=50)
        rng = np.random.default_rng(17)
        draws = []
        delta = candidate - reference
        for _ in range(50):
            seeds = rng.integers(0, 3, size=3)
            shared_queries = rng.integers(0, 4, size=4)
            draws.append(float(delta[np.ix_(seeds, shared_queries)].mean()))
        expected = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
