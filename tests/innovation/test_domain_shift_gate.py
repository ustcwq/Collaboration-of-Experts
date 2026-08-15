from __future__ import annotations

import unittest

import numpy as np

from bench_coe.innovation.domain_shift_gate import (
    DomainPolicy,
    apply_domain_policy,
    distribution_distances,
    domain_feature_matrix,
    environment_distribution_distances,
    nearest_source_distance,
)
from bench_coe.innovation.schema import Selection


def row(question: str, cluster: int, method: str, shift: float = 0.0) -> Selection:
    return Selection(
        question_id=question,
        selected_cluster_id=cluster,
        selected_expert_id=f"e{cluster}",
        normalized_answer=chr(ord("a") + cluster),
        cluster_scores={"0": 0.6 - shift, "1": 0.4 + shift},
        expert_scores={"e0": 0.6 - shift, "e1": 0.4 + shift},
        fallback_reason=None,
        observable_features={
            "method": method,
            "valid_fraction": 1.0 - shift,
            "cluster_fraction": 0.5 + shift,
            "partition_entropy": 0.7,
            "top1_share": 0.6,
            "top2_share": 0.4,
            "cluster_margin": 0.2,
            "mean_uncertainty": 0.0,
            "std_uncertainty": 0.0,
            "missing_fraction": shift,
            "answer_clusters": 2,
            "valid_experts": 4,
            "top_cluster_family_breadth": 2,
            "valid_mask": {"e0": True, "e1": True},
            "missing_mask": {"e0": False, "e1": False},
        },
        tie_breaking="fixture",
    )


class DomainShiftGateTests(unittest.TestCase):
    def test_features_and_distribution_distances_are_finite(self) -> None:
        source_rows = [row(f"q{index}", index % 2, "ref", shift=0.01 * index) for index in range(8)]
        source = domain_feature_matrix(source_rows)
        same = distribution_distances(source, source.copy())
        shifted = distribution_distances(source, source + 0.5)
        self.assertEqual(source.shape, (8, 12))
        self.assertTrue(np.isfinite(source).all())
        self.assertTrue(all(abs(value) < 1e-12 for value in same.values()))
        self.assertGreater(shifted["combined"], same["combined"])

    def test_environment_distance_and_nearest_support_exclude_same_group(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
                [2.0, 2.0],
                [2.1, 2.0],
            ],
            dtype=float,
        )
        environments = np.asarray([0, 0, 1, 1, 2, 2])
        calibrated = environment_distribution_distances(source, environments)
        self.assertEqual(set(calibrated), {"mean", "scale", "quantile", "combined"})
        distance = nearest_source_distance(
            source,
            source,
            source_environment=environments,
        )
        self.assertTrue((distance > 0.0).all())

    def test_policy_fallback_is_exact_and_preserves_masks(self) -> None:
        base = [row("q0", 1, "base"), row("q1", 1, "base")]
        reference = [row("q0", 0, "ref"), row("q1", 0, "ref")]
        policy = DomainPolicy("query_support", "nearest", 0.9)
        result = apply_domain_policy(
            base,
            reference,
            np.asarray([True, False]),
            policy,
            domain_diagnostics={"threshold": 0.3},
        )
        self.assertEqual([value.normalized_answer for value in result], ["b", "a"])
        self.assertTrue(result[0].observable_features["domain_guard_active"])
        self.assertTrue(result[1].observable_features["domain_guard_fallback"])
        self.assertEqual(set(result[1].observable_features["valid_mask"]), {"e0", "e1"})


if __name__ == "__main__":
    unittest.main()
