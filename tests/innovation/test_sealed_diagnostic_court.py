from __future__ import annotations

import inspect
import unittest

from bench_coe.innovation.blind_falsification_jury import (
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
from bench_coe.innovation.schema import EvaluationLabels, SourceTrainingLabels
from bench_coe.innovation.sealed_diagnostic_court import (
    DiagnosticProbe,
    DiagnosticProbeCheck,
    SDBVariant,
    SealedDiagnosticBijectionCourt,
)


def _question(question_id: str) -> FalsificationQuestion:
    return FalsificationQuestion(
        question_id,
        "synthetic",
        f"env::{int(question_id.removeprefix('q') or 0) % 3}",
        "Which result follows?",
        ("one", "two"),
        ("A", "B"),
    )


def _probe(question_id: str, answer: str) -> DiagnosticProbe:
    return DiagnosticProbe(
        probe_id=f"{question_id}::author",
        question_id=question_id,
        author_id="author",
        first_candidate="A",
        second_candidate="B",
        author_stage0_prediction="A",
        confidence=95,
        parse_error=None,
        abstained=False,
        left_candidate="A",
        right_candidate="B",
    )


def _check(question_id: str, answer: str) -> DiagnosticProbeCheck:
    rejected = "B" if answer == "A" else "A"
    return DiagnosticProbeCheck(
        check_id=f"{question_id}::author::checker",
        probe_id=f"{question_id}::author",
        question_id=question_id,
        author_id="author",
        checker_id="checker",
        outcome_side="LEFT" if answer == "A" else "RIGHT",
        selected_candidate=answer,
        rejected_candidate=rejected,
        confidence=95,
        parse_error=None,
        uncertain=False,
    )


class SealedDiagnosticCourtTests(unittest.TestCase):
    def test_delayed_reveal_evidence_can_rescue_unproposed_candidate(self) -> None:
        questions = [_question(f"q{index}") for index in range(12)]
        answers = {
            question.question_id: ("A" if index % 2 == 0 else "B")
            for index, question in enumerate(questions)
        }
        experts = ("author", "checker", "weak")
        base = [
            BasePrediction(question.question_id, expert, "A")
            for question in questions
            for expert in experts
        ]
        probes = [_probe(question.question_id, answers[question.question_id]) for question in questions]
        checks = [_check(question.question_id, answers[question.question_id]) for question in questions]
        correctness = {}
        environments = {}
        for question in questions:
            answer = answers[question.question_id]
            environments[question.question_id] = question.environment
            for expert in experts:
                correctness[(question.question_id, expert)] = answer == "A"
            for candidate in question.option_labels:
                correctness[(question.question_id, candidate_label_key(candidate))] = (
                    candidate == answer
                )
        labels = SourceTrainingLabels._from_source_adapter(
            "synthetic", "development", correctness, environments
        )
        model = SealedDiagnosticBijectionCourt(
            SDBVariant("sdb", regularization_c=10.0), seed=9
        ).fit(questions, base, probes, checks, labels)

        heldout = FalsificationQuestion(
            "heldout", "synthetic", "env::heldout", "q", ("one", "two"), ("A", "B")
        )
        heldout_base = [BasePrediction("heldout", expert, "A") for expert in experts]
        decision = model.predict(
            [heldout], heldout_base, [_probe("heldout", "B")], [_check("heldout", "B")]
        )[0]
        self.assertEqual(decision.answer, "B")
        self.assertTrue(decision.open_set_rescue)
        self.assertIsNone(decision.selected_expert_id)

    def test_evaluation_labels_cannot_fit_or_predict(self) -> None:
        self.assertNotIn(
            "labels", inspect.signature(SealedDiagnosticBijectionCourt.predict).parameters
        )
        with self.assertRaises(TypeError):
            SealedDiagnosticBijectionCourt(SDBVariant("reject")).fit(
                [_question("q0")],
                [BasePrediction("q0", "expert", "A")],
                [],
                [],
                EvaluationLabels("synthetic", "test", {}),  # type: ignore[arg-type]
            )

    def test_probe_and_check_contracts_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            DiagnosticProbe(
                "p", "q", "author", "A", "A", "A", 90, None, False, "A", "A"
            )
        with self.assertRaises(ValueError):
            DiagnosticProbeCheck(
                "c", "p", "q", "same", "same", "LEFT", "A", "B", 90, None, False
            )


if __name__ == "__main__":
    unittest.main()
