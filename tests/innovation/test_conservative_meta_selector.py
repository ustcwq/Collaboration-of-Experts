from __future__ import annotations

import unittest

import numpy as np

from bench_coe.innovation.conservative_meta_selector import (
    PredictionTable,
    ReliabilityStats,
    VoteRecipe,
    VoteScheme,
    generate_recipes,
    materialize_recipe_selections,
    method_weights,
    recipe_choices,
    reliability_statistics,
    vote_diagnostics,
)
from bench_coe.innovation.schema import Selection


def selection(question: str, cluster: int, expert: str, scores: dict[str, float]) -> Selection:
    return Selection(
        question_id=question,
        selected_cluster_id=cluster,
        selected_expert_id=expert,
        normalized_answer=chr(ord("a") + cluster),
        cluster_scores=scores,
        expert_scores={"e0": 0.2, "e1": 0.8},
        fallback_reason=None,
        observable_features={
            "valid_mask": {"e0": True, "e1": True},
            "missing_mask": {"e0": False, "e1": False},
            "valid_fraction": 1.0,
        },
        tie_breaking="fixture",
    )


class ConservativeMetaSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.methods = ("ref", "alt_a", "alt_b")
        self.questions = ("q0", "q1")
        self.table = PredictionTable.from_selections(
            {
                "ref": [
                    selection("q0", 0, "e0", {"0": 0.7, "1": 0.3}),
                    selection("q1", 0, "e0", {"0": 0.7, "1": 0.3}),
                ],
                "alt_a": [
                    selection("q0", 1, "e1", {"0": 0.2, "1": 0.8}),
                    selection("q1", 1, "e1", {"0": 0.2, "1": 0.8}),
                ],
                "alt_b": [
                    selection("q0", 1, "e1", {"0": 0.4, "1": 0.6}),
                    selection("q1", 0, "e0", {"0": 0.6, "1": 0.4}),
                ],
            },
            self.methods,
        )
        self.stats = ReliabilityStats(
            accuracy=np.asarray([0.5, 0.8, 0.7]),
            beta_lcb=np.asarray([0.3, 0.6, 0.5]),
            rescue_rate=np.asarray([0.0, 0.3, 0.2]),
            harm_rate=np.asarray([0.0, 0.05, 0.1]),
            worst_environment_delta=np.asarray([0.0, 0.1, -0.1]),
        )
        self.families = {"ref": "anchor", "alt_a": "a", "alt_b": "b"}

    def test_reliability_excludes_heldout_environment(self) -> None:
        correctness = [
            np.asarray(
                [
                    [True, True, False, False],
                    [False, False, True, True],
                    [True, False, True, False],
                ],
                dtype=bool,
            )
        ]
        environments = np.asarray([0, 0, 1, 1])
        stats = reliability_statistics(
            correctness,
            environments,
            environments != 1,
            reference_index=0,
        )
        np.testing.assert_allclose(stats.accuracy, [1.0, 0.0, 0.5])
        np.testing.assert_allclose(stats.harm_rate, [0.0, 1.0, 0.5])
        np.testing.assert_allclose(stats.rescue_rate, [0.0, 0.0, 0.0])

    def test_family_balancing_equalizes_correlated_family_mass(self) -> None:
        scheme = VoteScheme(
            pool="all",
            reference="ref",
            weighting="equal",
            aggregation="hard",
            top_k=0,
            family_balanced=True,
        )
        families = {"ref": "anchor", "alt_a": "repair", "alt_b": "repair"}
        active, weights = method_weights(
            self.stats,
            scheme,
            self.methods,
            families,
            self.methods,
        )
        mass = {
            family: float(
                sum(
                    weights[offset]
                    for offset, method_index in enumerate(active)
                    if families[self.methods[method_index]] == family
                )
            )
            for family in set(families.values())
        }
        self.assertAlmostEqual(mass["anchor"], 0.5)
        self.assertAlmostEqual(mass["repair"], 0.5)

    def test_conservative_family_gate_switches_only_with_independent_support(self) -> None:
        scheme = VoteScheme(
            pool="all",
            reference="ref",
            weighting="equal",
            aggregation="hard",
            top_k=0,
            family_balanced=False,
        )
        active, weights = method_weights(
            self.stats,
            scheme,
            self.methods,
            self.families,
            self.methods,
        )
        diagnostics = vote_diagnostics(self.table, scheme, active, weights, self.families)
        recipe = VoteRecipe(scheme, min_share=0.5, min_margin=0.0, min_families=2)
        choices = recipe_choices(self.table, recipe, diagnostics)
        self.assertEqual(self.methods[int(choices[0])], "alt_a")
        self.assertEqual(self.methods[int(choices[1])], "ref")

        rows = materialize_recipe_selections(self.table, recipe, diagnostics)
        self.assertEqual(rows[0].normalized_answer, "b")
        self.assertEqual(rows[1].normalized_answer, "a")
        self.assertTrue(rows[0].observable_features["switched_from_reference"])
        self.assertFalse(rows[1].observable_features["switched_from_reference"])
        self.assertEqual(set(rows[1].observable_features["valid_mask"]), {"e0", "e1"})

    def test_cluster_rank_uses_normalized_per_method_cluster_scores(self) -> None:
        scheme = VoteScheme(
            pool="all",
            reference="ref",
            weighting="equal",
            aggregation="cluster_rank",
            top_k=0,
            family_balanced=False,
        )
        active, weights = method_weights(
            self.stats,
            scheme,
            self.methods,
            self.families,
            self.methods,
        )
        diagnostics = vote_diagnostics(self.table, scheme, active, weights, self.families)
        self.assertTrue(np.isfinite(diagnostics.winning_share).all())
        for votes in diagnostics.cluster_votes:
            self.assertAlmostEqual(sum(votes.values()), 1.0)

    def test_recipe_grid_is_deterministic_and_unique(self) -> None:
        config = {
            "references": ["ref"],
            "method_pools": {"all": list(self.methods)},
            "weightings": ["equal", {"name": "safe_utility", "risk_penalty": 2.0}],
            "aggregations": ["hard", "cluster_rank"],
            "top_k": [0, 2],
            "family_balanced": [False, True],
            "min_share": [0.0, 0.5],
            "min_margin": [0.0],
            "min_families": [1, 2],
        }
        first = generate_recipes(config)
        second = generate_recipes(config)
        self.assertEqual([row.method for row in first], [row.method for row in second])
        self.assertEqual(len(first), 64)
        self.assertEqual(len({row.method for row in first}), len(first))


if __name__ == "__main__":
    unittest.main()
