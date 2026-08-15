from __future__ import annotations

import unittest

from bench_coe.innovation.goal_guardrails import (
    evaluate_acceptance_contract,
    evaluate_strict_improvement_contract,
    validate_acceptance_contract,
)


CONTRACT = {
    "primary_target": "test",
    "minimum_primary_delta": 0.0,
    "relative_non_regression_targets": ["math", "bbh", "gpqa", "mmstar"],
    "absolute_accuracy_floors": {
        "math": 0.626,
        "bbh": 0.759638,
        "gpqa": 0.330117,
        "mmstar": 0.226,
    },
    "tolerance": 1e-12,
}


def aggregate(accuracy: float, delta: float) -> dict[str, float]:
    return {"accuracy_mean": accuracy, "delta_vs_fcrg_full": delta}


class GoalGuardrailTests(unittest.TestCase):
    def test_goal_requires_primary_gain_relative_guards_and_absolute_floors(self) -> None:
        rows = {
            "test": aggregate(0.31, 0.005),
            "math": aggregate(0.626, 0.0),
            "bbh": aggregate(0.759638, 0.0),
            "gpqa": aggregate(0.330117, 0.0),
            "mmstar": aggregate(0.226, 0.0),
        }
        result = evaluate_acceptance_contract(rows, CONTRACT)
        self.assertTrue(result["strict_user_goal_met"])
        self.assertTrue(result["absolute_accuracy_floors_met"])
        self.assertTrue(result["relative_non_regression_met"])
        self.assertEqual(result["missing_required_targets"], [])

    def test_absolute_floor_cannot_be_hidden_by_nonnegative_delta(self) -> None:
        rows = {
            "test": aggregate(0.31, 0.005),
            "math": aggregate(0.625, 0.001),
            "bbh": aggregate(0.759638, 0.0),
            "gpqa": aggregate(0.330117, 0.0),
            "mmstar": aggregate(0.226, 0.0),
        }
        result = evaluate_acceptance_contract(rows, CONTRACT)
        self.assertTrue(result["relative_non_regression_met"])
        self.assertFalse(result["absolute_accuracy_floor_checks"]["math"]["met"])
        self.assertFalse(result["strict_user_goal_met"])

    def test_missing_or_regressed_target_fails_closed(self) -> None:
        rows = {
            "test": aggregate(0.31, 0.005),
            "math": aggregate(0.626, 0.0),
            "bbh": aggregate(0.759638, 0.0),
            "gpqa": aggregate(0.330117, -0.001),
        }
        result = evaluate_acceptance_contract(rows, CONTRACT)
        self.assertEqual(result["missing_required_targets"], ["mmstar"])
        self.assertFalse(result["relative_non_regression_met"])
        self.assertFalse(result["absolute_accuracy_floors_met"])
        self.assertFalse(result["strict_user_goal_met"])

    def test_floor_targets_must_also_be_relative_guards(self) -> None:
        contract = dict(CONTRACT)
        contract["relative_non_regression_targets"] = ["math"]
        with self.assertRaisesRegex(ValueError, "must also require relative"):
            validate_acceptance_contract(contract)

    def test_strict_contract_rejects_zero_delta(self) -> None:
        rows = {
            "source": aggregate(0.29, 0.01),
            "test": aggregate(0.31, 0.0),
        }
        result = evaluate_strict_improvement_contract(
            rows,
            {
                "strict_improvement_targets": ["source", "test"],
                "absolute_accuracy_floors": {"source": 0.28, "test": 0.30},
            },
        )
        self.assertFalse(result["strict_improvement_checks"]["test"]["met"])
        self.assertFalse(result["strict_user_goal_met"])

    def test_strict_contract_requires_every_target_and_floor(self) -> None:
        contract = {
            "strict_improvement_targets": ["source", "test"],
            "absolute_accuracy_floors": {"source": 0.30, "test": 0.30},
        }
        below_floor = evaluate_strict_improvement_contract(
            {
                "source": aggregate(0.29, 0.01),
                "test": aggregate(0.31, 0.01),
            },
            contract,
        )
        self.assertTrue(below_floor["all_datasets_strictly_improve"])
        self.assertFalse(below_floor["absolute_accuracy_floors_met"])
        self.assertFalse(below_floor["strict_user_goal_met"])
        missing = evaluate_strict_improvement_contract(
            {"source": aggregate(0.31, 0.01)},
            contract,
        )
        self.assertEqual(missing["missing_required_targets"], ["test"])
        self.assertFalse(missing["strict_user_goal_met"])

    def test_strict_contract_accepts_only_all_positive_results(self) -> None:
        result = evaluate_strict_improvement_contract(
            {
                "source": aggregate(0.31, 0.01),
                "test": aggregate(0.32, 0.02),
            },
            {
                "strict_improvement_targets": ["source", "test"],
                "absolute_accuracy_floors": {"source": 0.30, "test": 0.30},
            },
        )
        self.assertTrue(result["strict_user_goal_met"])
        self.assertGreater(result["worst_required_delta_vs_fcrg_full"], 0.0)


if __name__ == "__main__":
    unittest.main()
