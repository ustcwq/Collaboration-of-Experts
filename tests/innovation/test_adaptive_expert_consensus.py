from __future__ import annotations

import unittest
from dataclasses import replace

from bench_coe.innovation.adaptive_expert_consensus import (
    AdaptiveConsensusVariant,
    adaptive_expert_consensus,
    adapt_target_reliability,
)
from bench_coe.innovation.conditioned_expert_consensus import (
    fit_conditioned_expert_profiles,
)
from bench_coe.innovation.run_adaptive_expert_consensus import (
    _observable_majority_reference,
    _select_source_candidates,
    _stratified_folds,
)
from bench_coe.innovation.schema import (
    CanonicalPredictionRecord,
    ExpertPool,
    ObservableQueryBatch,
    Selection,
    SourceTrainingLabels,
)


def batch(dataset: str, answers: tuple[tuple[str, str], ...]) -> ObservableQueryBatch:
    experts = tuple(expert for expert, _ in answers)
    pool = ExpertPool(experts, {expert: expert for expert in experts})
    records = tuple(
        CanonicalPredictionRecord(
            dataset=dataset,
            split="split",
            question_id=f"{dataset}::split::q0",
            raw_question_id="q0",
            subject="subject",
            modality="language",
            expert_id=expert,
            expert_family=expert,
            raw_answer=answer,
            raw_output=answer,
            normalized_answer=answer,
            per_query_cluster_id=index,
            uncertainty=0.1,
            valid_output=True,
            missing_reason=None,
        )
        for index, (expert, answer) in enumerate(answers)
    )
    return ObservableQueryBatch(dataset, "split", "language", pool, records)


class AdaptiveExpertConsensusTests(unittest.TestCase):
    def test_source_reliability_can_override_raw_majority(self) -> None:
        source = batch("source", (("strong", "a"), ("weak1", "b"), ("weak2", "b")))
        labels = SourceTrainingLabels._from_source_adapter(
            "source",
            "split",
            {
                ("source::split::q0", "strong"): True,
                ("source::split::q0", "weak1"): False,
                ("source::split::q0", "weak2"): False,
            },
            {"source::split::q0": "subject"},
        )
        profiles = fit_conditioned_expert_profiles(source, labels, {"subject": "group"})
        target = batch("target", (("strong", "a"), ("weak1", "b"), ("weak2", "b")))
        reference = [
            Selection(
                "target::split::q0", 1, "weak1", "b", {"1": 1.0},
                {"weak1": 1.0}, None, {},
            )
        ]
        variant = AdaptiveConsensusVariant(
            "adaptive", 0.01, 2.0, 0.0, 0.0, 0.0, "sum", False, 0.0, 0
        )
        rows = adaptive_expert_consensus(
            target, profiles, {"subject": "group"}, variant, reference=reference
        )
        self.assertEqual(rows[0].normalized_answer, "a")
        self.assertFalse(rows[0].observable_features["adaptive_consensus_uses_target_labels"])

    def test_unlabeled_em_is_deterministic_and_bounded(self) -> None:
        source = batch("source", (("e1", "a"), ("e2", "a"), ("e3", "b")))
        labels = SourceTrainingLabels._from_source_adapter(
            "source",
            "split",
            {
                ("source::split::q0", "e1"): True,
                ("source::split::q0", "e2"): True,
                ("source::split::q0", "e3"): False,
            },
            {"source::split::q0": "subject"},
        )
        profiles = fit_conditioned_expert_profiles(source, labels, {"subject": "group"})
        target = batch("target", (("e1", "a"), ("e2", "a"), ("e3", "b")))
        variant = AdaptiveConsensusVariant(
            "em", 1.0, 1.0, 0.0, 0.0, 0.0, "noisy_or", False, 0.5, 2
        )
        first, _ = adapt_target_reliability(target, profiles, {"subject": "group"}, variant)
        second, _ = adapt_target_reliability(target, profiles, {"subject": "group"}, variant)
        self.assertEqual(first, second)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first["group"].values()))

    def test_source_oof_folds_are_deterministic_and_cover_once(self) -> None:
        template = batch("source", (("e1", "a"), ("e2", "b")))
        records = []
        for index in range(6):
            for row in template.records:
                records.append(
                    replace(
                        row,
                        question_id=f"source::split::q{index}",
                        raw_question_id=f"q{index}",
                    )
                )
        source = ObservableQueryBatch(
            template.dataset,
            template.split,
            template.modality,
            template.pool,
            tuple(records),
        )
        first = _stratified_folds(source, {"subject": "group"}, 3)
        second = _stratified_folds(source, {"subject": "group"}, 3)
        self.assertEqual(first, second)
        flat = [question_id for fold in first for question_id in fold]
        self.assertEqual(set(flat), set(source.question_ids))
        self.assertEqual(len(flat), len(set(flat)))

    def test_observable_majority_reference_is_label_free(self) -> None:
        source = batch("source", (("e1", "a"), ("e2", "b"), ("e3", "b")))
        rows = _observable_majority_reference(source)
        self.assertEqual(rows[0].normalized_answer, "b")
        self.assertFalse(rows[0].observable_features["source_reference_uses_labels"])

    def test_target_candidates_are_selected_only_from_source_metrics(self) -> None:
        variant = AdaptiveConsensusVariant(
            "variant", 1.0, 1.0, 0.0, 0.0, 0.0, "sum", False, 0.0, 0
        )
        specs = {
            name: (variant, 0.0, 0.0) for name in ("accuracy", "balanced", "worst")
        }
        rows = [
            {
                "method": "accuracy", "accuracy": 0.9,
                "balanced_environment_accuracy": 0.4, "worst_environment_delta": -0.2,
            },
            {
                "method": "balanced", "accuracy": 0.8,
                "balanced_environment_accuracy": 0.95, "worst_environment_delta": -0.1,
            },
            {
                "method": "worst", "accuracy": 0.7,
                "balanced_environment_accuracy": 0.7, "worst_environment_delta": 0.1,
            },
        ]
        selected = _select_source_candidates(
            rows,
            specs,
            {
                "top_per_source_metric": 1,
                "source_selection_metrics": [
                    "accuracy", "balanced_environment_accuracy", "worst_environment_delta"
                ],
                "maximum_target_candidates": 3,
            },
        )
        self.assertEqual(selected, ["accuracy", "balanced", "worst"])


if __name__ == "__main__":
    unittest.main()
