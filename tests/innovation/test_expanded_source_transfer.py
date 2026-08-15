from __future__ import annotations

import unittest

from bench_coe.innovation.expanded_source_transfer import (
    leave_one_environment_out_source_best,
)
from bench_coe.innovation.schema import (
    CanonicalPredictionRecord,
    ExpertPool,
    ObservableQueryBatch,
    SourceTrainingLabels,
)


def _fixture() -> tuple[ObservableQueryBatch, SourceTrainingLabels]:
    pool = ExpertPool(("a", "b"), {"a": "fa", "b": "fb"})
    records = []
    environments = {"q1": "e1", "q2": "e1", "q3": "e2", "q4": "e2"}
    correctness = {
        ("q1", "a"): True,
        ("q1", "b"): False,
        ("q2", "a"): True,
        ("q2", "b"): False,
        ("q3", "a"): False,
        ("q3", "b"): True,
        ("q4", "a"): False,
        ("q4", "b"): True,
    }
    for question_id in environments:
        for expert, answer, cluster in (("a", "x", 0), ("b", "y", 1)):
            records.append(
                CanonicalPredictionRecord(
                    "source",
                    "validation",
                    question_id,
                    question_id,
                    environments[question_id],
                    "multimodal",
                    expert,
                    pool.family_by_expert[expert],
                    answer,
                    answer,
                    answer,
                    cluster,
                    0.0,
                    True,
                    None,
                )
            )
    batch = ObservableQueryBatch(
        "source", "validation", "multimodal", pool, tuple(records)
    )
    labels = SourceTrainingLabels._from_source_adapter(
        "source", "validation", correctness, environments
    )
    return batch, labels


class ExpandedSourceTransferTests(unittest.TestCase):
    def test_each_heldout_environment_is_excluded_from_its_fit(self) -> None:
        batch, labels = _fixture()
        rows, audit = leave_one_environment_out_source_best(batch, labels)
        self.assertEqual(len(rows), 4)
        self.assertEqual(audit["environments"], 2)
        by_id = {row.question_id: row for row in rows}
        self.assertEqual(by_id["q1"].selected_expert_id, "b")
        self.assertEqual(by_id["q3"].selected_expert_id, "a")
        self.assertFalse(
            by_id["q1"].observable_features["heldout_environment_labels_used"]
        )


if __name__ == "__main__":
    unittest.main()
