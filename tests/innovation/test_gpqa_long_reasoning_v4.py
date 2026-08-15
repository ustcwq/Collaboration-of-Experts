from __future__ import annotations

import unittest

from bench_coe.innovation.run_gpqa_long_reasoning_v4 import (
    _candidate_correctness,
    evaluate_acceptance,
)
from bench_coe.innovation.schema import EvaluationLabels, Selection


class GPQALongReasoningV4Tests(unittest.TestCase):
    def test_acceptance_requires_large_gain_and_v3_floor(self) -> None:
        self.assertTrue(evaluate_acceptance(0.39, 0.05, 0.33, 0.355)["passed"])
        self.assertFalse(evaluate_acceptance(0.37, 0.05, 0.33, 0.355)["passed"])
        self.assertFalse(evaluate_acceptance(0.35, 0.01, 0.33, 0.355)["passed"])

    def test_synthetic_reasoner_is_scored_against_post_boundary_gold(self) -> None:
        rows = [
            Selection("q1", -2, "reasoner", "c", {}, {}, None, {}),
            Selection("q2", 0, "fallback", "a", {}, {}, None, {}),
        ]
        labels = EvaluationLabels("gpqa", "cached_eval", {("q2", "fallback"): True})
        result = _candidate_correctness(rows, labels, {"q1": "c", "q2": "b"}, "reasoner")
        self.assertEqual(result, {"q1": True, "q2": True})


if __name__ == "__main__":
    unittest.main()
