from __future__ import annotations

import unittest

from bench_coe.innovation.method_consensus import (
    ConsensusVariant,
    apply_consensus_gate,
    consensus_selections,
    default_consensus_variants,
    method_subsets,
)
from bench_coe.innovation.schema import Selection


def row(question: str, answer: str, cluster: int, expert: str, score: float = 1.0) -> Selection:
    return Selection(
        question_id=question,
        selected_cluster_id=cluster,
        selected_expert_id=expert,
        normalized_answer=answer,
        cluster_scores={str(cluster): score, str(1 - cluster): 0.0},
        expert_scores={expert: score},
        fallback_reason=None,
        observable_features={"valid_mask": {expert: True}, "missing_mask": {expert: False}},
    )


class MethodConsensusTest(unittest.TestCase):
    def test_query_local_majority_and_reference_tie(self) -> None:
        rows = {
            "fcrg_full": [row("q1", "a", 0, "e1"), row("q2", "b", 1, "e1")],
            "smoothie_local": [row("q1", "b", 1, "e2"), row("q2", "a", 0, "e2")],
            "more_style": [row("q1", "b", 1, "e3"), row("q2", "b", 1, "e3")],
        }
        variant = ConsensusVariant("test", "all", "equal")
        result = consensus_selections(rows, variant)
        self.assertEqual([value.normalized_answer for value in result], ["b", "b"])
        self.assertTrue(all(value.observable_features["consensus_uses_target_labels"] is False for value in result))
        self.assertIn(str(result[0].selected_cluster_id), result[0].cluster_scores)

    def test_family_weighting_prevents_duplicate_family_domination(self) -> None:
        rows = {
            "fcrg_full": [row("q", "a", 0, "e1")],
            "fcrg_no_g": [row("q", "a", 0, "e1")],
            "smoothie_local": [row("q", "b", 1, "e2")],
            "more_style": [row("q", "b", 1, "e3")],
        }
        variant = ConsensusVariant("test", "all", "family")
        result = consensus_selections(rows, variant)
        self.assertEqual(result[0].normalized_answer, "b")

    def test_external_source_weights_override_vote_mass(self) -> None:
        rows = {
            "fcrg_full": [row("q", "a", 0, "e1")],
            "smoothie_local": [row("q", "b", 1, "e2")],
            "more_style": [row("q", "b", 1, "e3")],
        }
        variant = ConsensusVariant("test", "all", "equal")
        result = consensus_selections(
            rows,
            variant,
            external_method_weights={"fcrg_full": 10.0, "smoothie_local": 1.0, "more_style": 1.0},
        )
        self.assertEqual(result[0].normalized_answer, "a")
        self.assertTrue(result[0].observable_features["consensus_external_method_weights"])

    def test_variants_and_subsets_are_stable(self) -> None:
        methods = ("fcrg_full", "fcrg_random_edges", "knop_k8", "smoothie_local")
        subsets = method_subsets(methods)
        self.assertNotIn("fcrg_random_edges", subsets["nonstochastic"])
        variants = default_consensus_variants()
        self.assertEqual(len(variants), 80)
        self.assertEqual(len({variant.name for variant in variants}), 80)

    def test_guarded_consensus_falls_back_without_labels(self) -> None:
        reference = [row("q", "a", 0, "e1")]
        rows = {
            "fcrg_full": reference,
            "smoothie_local": [row("q", "b", 1, "e2")],
            "more_style": [row("q", "b", 1, "e3")],
        }
        ungated = consensus_selections(rows, ConsensusVariant("raw", "all", "equal"))
        guarded = apply_consensus_gate(
            ungated,
            reference,
            name="guarded",
            fallback_share=0.8,
            minimum_advantage=0.0,
        )
        self.assertEqual(guarded[0].normalized_answer, "a")
        self.assertFalse(guarded[0].observable_features["consensus_gate_uses_target_labels"])


if __name__ == "__main__":
    unittest.main()
