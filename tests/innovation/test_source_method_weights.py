from __future__ import annotations

import unittest

from bench_coe.innovation.schema import EvaluationLabels, Selection, SourceTrainingLabels
from bench_coe.innovation.source_method_weights import (
    source_method_profiles,
    source_weight_candidates,
)


def selection(question: str, expert: str, answer: str) -> Selection:
    return Selection(question, 0, expert, answer, {"0": 1.0}, {expert: 1.0}, None, {})


class SourceMethodWeightsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = SourceTrainingLabels._from_source_adapter(
            "source",
            "validation",
            {
                ("q1", "good"): True,
                ("q1", "bad"): False,
                ("q2", "good"): True,
                ("q2", "bad"): False,
            },
            {"q1": "a", "q2": "b"},
        )
        self.rows = {
            1: {
                "fcrg_full": [selection("q1", "bad", "a"), selection("q2", "bad", "a")],
                "better": [selection("q1", "good", "b"), selection("q2", "good", "b")],
            }
        }

    def test_profiles_and_candidate_weights_use_source_only(self) -> None:
        profiles = source_method_profiles(self.rows, self.labels)
        self.assertEqual(profiles["better"]["accuracy"], 1.0)
        self.assertGreater(profiles["better"]["net_delta"], 0.0)
        candidates = source_weight_candidates(profiles)
        self.assertGreater(candidates["accuracy_p2p0"]["better"], candidates["accuracy_p2p0"]["fcrg_full"])
        self.assertEqual(candidates["accuracy_top1"]["better"], 1.0)

    def test_evaluation_labels_are_rejected(self) -> None:
        target = EvaluationLabels("target", "test", {})
        with self.assertRaises(TypeError):
            source_method_profiles(self.rows, target)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
