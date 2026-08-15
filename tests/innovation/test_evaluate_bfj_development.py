from __future__ import annotations

import unittest

from bench_coe.innovation.blind_falsification_jury import (
    AuditObservation,
    BFJVariant,
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
from bench_coe.innovation.evaluate_bfj_development import (
    leave_one_environment_out,
    select_variant_nested,
    variants_from_config,
)
from bench_coe.innovation.schema import SourceTrainingLabels


class BFJDevelopmentEvaluationTests(unittest.TestCase):
    def test_leave_one_environment_out_is_disjoint_and_complete(self) -> None:
        environments = {
            "q1": "a",
            "q2": "a",
            "q3": "b",
            "q4": "c",
        }
        folds = leave_one_environment_out(environments, environments)
        self.assertEqual([row[0] for row in folds], ["a", "b", "c"])
        heldout = []
        for _, train_ids, heldout_ids in folds:
            self.assertFalse(set(train_ids).intersection(heldout_ids))
            self.assertEqual(set(train_ids).union(heldout_ids), set(environments))
            heldout.extend(heldout_ids)
        self.assertEqual(sorted(heldout), sorted(environments))

    def test_variant_grid_order_is_frozen(self) -> None:
        config = {
            "variant_grid": {
                "prior_strength": [0.5, 1.0],
                "evidence_strength": [1.0],
                "smoothing": [1.0],
                "use_confidence_bins": [False, True],
                "calibrate_self_bias": [True],
                "open_option_set": [True],
                "intervention_margin": [0.0, 0.1],
                "max_abs_log_likelihood_ratio": 2.5,
                "confidence_threshold": 67,
            }
        }
        variants = variants_from_config(config)
        self.assertEqual(len(variants), 8)
        self.assertEqual(variants[0].name, "bfj_grid_000")
        self.assertEqual(variants[-1].name, "bfj_grid_007")

    def test_nested_selection_prefers_source_predictive_audits(self) -> None:
        questions = []
        answers = {}
        base = []
        audits = []
        environments = {}
        correctness = {}
        for index in range(6):
            question_id = f"q{index}"
            answer = "A" if index % 2 == 0 else "B"
            environment = f"env{index // 2}"
            question = FalsificationQuestion(
                question_id,
                "synthetic",
                environment,
                f"Question {index}",
                ("first", "second"),
                ("A", "B"),
            )
            questions.append(question)
            answers[question_id] = answer
            environments[question_id] = environment
            for expert in ("auditor", "weak"):
                base.append(BasePrediction(question_id, expert, "A"))
                correctness[(question_id, expert)] = answer == "A"
            for candidate in ("A", "B"):
                correctness[(question_id, candidate_label_key(candidate))] = candidate == answer
                audits.append(
                    AuditObservation(
                        question_id,
                        "auditor",
                        candidate,
                        "SURVIVES" if candidate == answer else "FALSIFIED",
                        90,
                        answer,
                    )
                )
        labels = SourceTrainingLabels._from_source_adapter(
            "synthetic", "development", correctness, environments
        )
        variants = (
            BFJVariant(
                "no_evidence",
                prior_strength=1.0,
                evidence_strength=0.0,
                use_confidence_bins=False,
            ),
            BFJVariant(
                "predictive_evidence",
                prior_strength=0.1,
                evidence_strength=2.0,
                smoothing=0.5,
                use_confidence_bins=False,
            ),
        )
        selected, rows = select_variant_nested(
            questions, base, audits, labels, answers, variants
        )
        self.assertEqual(selected.name, "predictive_evidence")
        selected_rows = [row for row in rows if row["selected"]]
        self.assertEqual(len(selected_rows), 1)
        self.assertEqual(selected_rows[0]["name"], "predictive_evidence")


if __name__ == "__main__":
    unittest.main()
