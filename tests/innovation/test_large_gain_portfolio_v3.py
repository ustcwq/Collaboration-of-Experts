from __future__ import annotations

import unittest

from bench_coe.innovation.run_large_gain_portfolio_v3 import evaluate_v3_acceptance


class LargeGainPortfolioV3Tests(unittest.TestCase):
    def test_all_datasets_must_strictly_exceed_threshold_and_v2(self) -> None:
        rows = [
            {"dataset": "a", "accuracy_mean": 0.5, "minimum_seed_delta_vs_fcrg": 0.021},
            {"dataset": "b", "accuracy_mean": 0.6, "minimum_seed_delta_vs_fcrg": 0.03},
        ]
        self.assertTrue(evaluate_v3_acceptance(rows, 0.02, {"a": 0.5, "b": 0.59})["passed"])
        rows[0]["minimum_seed_delta_vs_fcrg"] = 0.02
        self.assertFalse(evaluate_v3_acceptance(rows, 0.02, {"a": 0.5, "b": 0.59})["passed"])

    def test_missing_or_v2_regressed_dataset_fails(self) -> None:
        rows = [{"dataset": "a", "accuracy_mean": 0.49, "minimum_seed_delta_vs_fcrg": 0.03}]
        result = evaluate_v3_acceptance(rows, 0.02, {"a": 0.5, "b": 0.4})
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_datasets"], ["b"])


if __name__ == "__main__":
    unittest.main()
