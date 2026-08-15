from __future__ import annotations

import unittest
from unittest.mock import patch

from bench_coe.innovation.run_large_gain_portfolio import _evaluation_matrices
from bench_coe.innovation.schema import EvaluationLabels, Selection


def selection(question_id: str, expert: str = "e1") -> Selection:
    return Selection(
        question_id=question_id,
        selected_cluster_id=0,
        selected_expert_id=expert,
        normalized_answer="a",
        cluster_scores={"0": 1.0},
        expert_scores={expert: 1.0},
        fallback_reason=None,
        observable_features={},
    )


class LargeGainPortfolioTests(unittest.TestCase):
    def test_smoke_limit_is_applied_to_evaluation_labels(self) -> None:
        labels = EvaluationLabels(
            "dataset",
            "split",
            {("q0", "e1"): True, ("q1", "e1"): False},
        )
        candidates = {7: [selection("q0"), selection("q1")]}
        references = {7: [selection("q0"), selection("q1")]}
        job = {"name": "dataset"}
        with patch(
            "bench_coe.innovation.run_large_gain_portfolio._load_labels",
            return_value=labels,
        ) as load_labels:
            candidate_matrix, reference_matrix = _evaluation_matrices(
                job, candidates, references, 2
            )
        load_labels.assert_called_once_with(job, 2)
        self.assertEqual(candidate_matrix.shape, (1, 2))
        self.assertEqual(reference_matrix.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
