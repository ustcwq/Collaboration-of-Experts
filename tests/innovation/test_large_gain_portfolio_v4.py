from __future__ import annotations

import unittest

from bench_coe.innovation.run_large_gain_portfolio_v4 import evaluate_v4_acceptance


class LargeGainPortfolioV4Tests(unittest.TestCase):
    def test_every_dataset_requires_five_points_and_v3_floor(self) -> None:
        rows = [
            {
                "dataset": "a",
                "accuracy_mean": 0.40,
                "minimum_seed_delta_vs_fcrg": 0.05,
            },
            {
                "dataset": "b",
                "accuracy_mean": 0.50,
                "minimum_seed_delta_vs_fcrg": 0.10,
            },
        ]
        self.assertTrue(evaluate_v4_acceptance(rows, 0.05, {"a": 0.4, "b": 0.45})["passed"])
        self.assertFalse(evaluate_v4_acceptance(rows, 0.051, {"a": 0.4, "b": 0.45})["passed"])
        self.assertFalse(evaluate_v4_acceptance(rows, 0.05, {"a": 0.41, "b": 0.45})["passed"])

    def test_missing_dataset_fails_closed(self) -> None:
        rows = [{"dataset": "a", "accuracy_mean": 0.5, "minimum_seed_delta_vs_fcrg": 0.1}]
        result = evaluate_v4_acceptance(rows, 0.05, {"a": 0.4, "b": 0.4})
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_datasets"], ["b"])


if __name__ == "__main__":
    unittest.main()
