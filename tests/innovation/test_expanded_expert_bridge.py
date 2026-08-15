from __future__ import annotations

import unittest

from bench_coe.innovation.expanded_expert_bridge import (
    cross_pool_paired_comparison,
    filter_rows_by_id_prefix,
)
from bench_coe.innovation.schema import EvaluationLabels, Selection


def selection(question_id: str, expert: str) -> Selection:
    return Selection(question_id, 0, expert, "a", {}, {}, None, {})


class ExpandedExpertBridgeTests(unittest.TestCase):
    def test_source_projection_is_prefix_defined_sorted_and_unique(self) -> None:
        rows = [
            {"id": "test_b", "is_correct": True},
            {"id": "validation_z", "is_correct": False},
            {"id": "validation_a", "is_correct": True},
        ]
        projected = filter_rows_by_id_prefix(rows, "validation_")
        self.assertEqual([row["id"] for row in projected], ["validation_a", "validation_z"])
        with self.assertRaises(ValueError):
            filter_rows_by_id_prefix(
                [{"id": "validation_a"}, {"id": "validation_a"}], "validation_"
            )

    def test_cross_pool_comparison_uses_each_pools_own_labels(self) -> None:
        ids = ["toy::target::q1", "toy::target::q2", "toy::target::q3"]
        candidates = [selection(qid, "large") for qid in ids]
        references = [selection(qid, "small") for qid in ids]
        candidate_labels = EvaluationLabels(
            "toy",
            "target",
            {(qid, "large"): value for qid, value in zip(ids, [True, True, False])},
        )
        reference_labels = EvaluationLabels(
            "toy",
            "target",
            {(qid, "small"): value for qid, value in zip(ids, [False, True, False])},
        )
        summary, candidate, reference = cross_pool_paired_comparison(
            candidates,
            candidate_labels,
            references,
            reference_labels,
            bootstrap_seed=1,
            bootstrap_samples=100,
        )
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["fcrg_full_accuracy"], 1 / 3)
        self.assertEqual(summary["rescue_count"], 1)
        self.assertEqual(summary["harm_count"], 0)
        self.assertEqual(candidate.tolist(), [1, 1, 0])
        self.assertEqual(reference.tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
