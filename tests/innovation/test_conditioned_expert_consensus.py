from __future__ import annotations

import unittest

from bench_coe.innovation.conditioned_expert_consensus import (
    ConditionedConsensusVariant,
    conditioned_expert_consensus,
    fit_conditioned_expert_profiles,
)
from bench_coe.innovation.schema import (
    CanonicalPredictionRecord,
    EvaluationLabels,
    ExpertPool,
    ObservableQueryBatch,
    Selection,
    SourceTrainingLabels,
)


def batch(dataset: str, split: str, subjects: tuple[str, ...]) -> ObservableQueryBatch:
    pool = ExpertPool(("e1", "e2"), {"e1": "f1", "e2": "f2"})
    records = []
    for index, subject in enumerate(subjects):
        for expert, answer, cluster in (("e1", "a", 0), ("e2", "b", 1)):
            records.append(
                CanonicalPredictionRecord(
                    dataset, split, f"{dataset}::{split}::q{index}", f"q{index}", subject,
                    "language", expert, pool.family_by_expert[expert], answer, answer,
                    answer, cluster, 0.1, True, None,
                )
            )
    return ObservableQueryBatch(dataset, split, "language", pool, tuple(records))


class ConditionedExpertConsensusTest(unittest.TestCase):
    def test_source_group_reliability_selects_expected_answer(self) -> None:
        source = batch("source", "validation", ("physics", "chemistry"))
        labels = SourceTrainingLabels._from_source_adapter(
            "source", "validation",
            {
                ("source::validation::q0", "e1"): True,
                ("source::validation::q0", "e2"): False,
                ("source::validation::q1", "e1"): False,
                ("source::validation::q1", "e2"): True,
            },
            {"source::validation::q0": "physics", "source::validation::q1": "chemistry"},
        )
        profiles = fit_conditioned_expert_profiles(
            source, labels, {"physics": "science", "chemistry": "other"}
        )
        target = batch("target", "test", ("Physics",))
        reference = [Selection("target::test::q0", 1, "e2", "b", {"1": 1.0}, {"e2": 1.0}, None, {})]
        variant = ConditionedConsensusVariant("conditioned", 0.0, 1.0, 0.0, False, 0.0)
        result = conditioned_expert_consensus(
            target, profiles, {"Physics": "science"}, variant, reference=reference
        )
        self.assertEqual(result[0].normalized_answer, "a")
        self.assertFalse(result[0].observable_features["conditioned_consensus_uses_target_labels"])

    def test_evaluation_labels_are_rejected(self) -> None:
        source = batch("source", "validation", ("physics",))
        with self.assertRaises(TypeError):
            fit_conditioned_expert_profiles(
                source,
                EvaluationLabels("source", "validation", {}),  # type: ignore[arg-type]
                {"physics": "science"},
            )


if __name__ == "__main__":
    unittest.main()
