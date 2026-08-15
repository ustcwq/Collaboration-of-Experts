from __future__ import annotations

import unittest

from bench_coe.innovation.gpqa_permutation_consensus import (
    PermutationConsensusVariant,
    gpqa_permutation_consensus,
)
from bench_coe.innovation.schema import (
    CanonicalPredictionRecord,
    ExpertPool,
    ObservableQueryBatch,
    Selection,
)


def record(qid: str, epoch: int, expert: str, answer: str, options: list[str]):
    return CanonicalPredictionRecord(
        dataset="gpqa",
        split="cached_eval",
        question_id=qid,
        raw_question_id=qid,
        subject="Physics",
        modality="language",
        expert_id=expert,
        expert_family=expert,
        raw_answer=answer,
        raw_output=answer,
        normalized_answer=answer.lower(),
        per_query_cluster_id=ord(answer) - ord("A"),
        uncertainty=0.0,
        valid_output=True,
        missing_reason=None,
        observable_metadata={
            "base_question_id": 7,
            "epoch": epoch,
            "options": options,
        },
    )


class GPQAPermutationConsensusTests(unittest.TestCase):
    def test_semantic_vote_maps_back_through_each_permutation(self) -> None:
        q0 = "gpqa::cached_eval::q0"
        q1 = "gpqa::cached_eval::q1"
        records = (
            record(q0, 0, "strong", "A", ["cat", "dog", "bird", "fish"]),
            record(q0, 0, "weak", "B", ["cat", "dog", "bird", "fish"]),
            record(q1, 1, "strong", "B", ["dog", "cat", "fish", "bird"]),
            record(q1, 1, "weak", "A", ["dog", "cat", "fish", "bird"]),
        )
        batch = ObservableQueryBatch(
            "gpqa",
            "cached_eval",
            "language",
            ExpertPool(("strong", "weak"), {"strong": "strong", "weak": "weak"}),
            records,
        )
        reference = [
            Selection(q0, 1, "weak", "b", {}, {}, None, {}),
            Selection(q1, 0, "weak", "a", {}, {}, None, {}),
        ]
        variant = PermutationConsensusVariant(
            "semantic",
            "expert_normalized",
            source_power=2.0,
            consistency_power=1.0,
            family_balance=False,
            minimum_share=0.0,
            minimum_advantage=0.0,
        )
        result = gpqa_permutation_consensus(
            batch, {"strong": 0.9, "weak": 0.1}, reference, variant
        )
        self.assertEqual([row.normalized_answer for row in result], ["a", "b"])
        self.assertTrue(all(row.observable_features["permutation_consensus_switched"] for row in result))

    def test_missing_permutation_metadata_fails_closed(self) -> None:
        broken = record(
            "gpqa::cached_eval::q0", 0, "strong", "A", ["a", "b", "c", "d"]
        )
        broken = CanonicalPredictionRecord(
            **{**broken.__dict__, "observable_metadata": {"epoch": 0, "options": ["a", "b"]}}
        )
        batch = ObservableQueryBatch(
            "gpqa",
            "cached_eval",
            "language",
            ExpertPool(("strong",), {"strong": "strong"}),
            (broken,),
        )
        with self.assertRaises(ValueError):
            gpqa_permutation_consensus(
                batch,
                {"strong": 1.0},
                [Selection(broken.question_id, 0, "strong", "a", {}, {}, None, {})],
                PermutationConsensusVariant("x", "raw", 1, 0, False, 0, 0),
            )


if __name__ == "__main__":
    unittest.main()
