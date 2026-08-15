from __future__ import annotations

import unittest

from bench_coe.innovation.run_bbh_symbolic_v4 import (
    evaluate_acceptance as evaluate_bbh_acceptance,
)
from bench_coe.innovation.run_expanded_mmmu_source_test_v4 import (
    evaluate_acceptance as evaluate_mmmu_acceptance,
)


class LargeGainV4Tests(unittest.TestCase):
    def test_mmmu_requires_five_points_and_v3_floor(self) -> None:
        rows = [
            {
                "dataset": "a",
                "accuracy_mean": 0.55,
                "minimum_seed_delta_vs_fcrg": 0.05,
            }
        ]
        self.assertTrue(evaluate_mmmu_acceptance(rows, 0.05, {"a": 0.54})["passed"])
        rows[0]["accuracy_mean"] = 0.53
        self.assertFalse(evaluate_mmmu_acceptance(rows, 0.05, {"a": 0.54})["passed"])

    def test_bbh_requires_both_guards(self) -> None:
        self.assertTrue(evaluate_bbh_acceptance(0.82, 0.05, 0.76, 0.78)["passed"])
        self.assertFalse(evaluate_bbh_acceptance(0.80, 0.05, 0.76, 0.78)["passed"])


if __name__ == "__main__":
    unittest.main()
