from __future__ import annotations

import inspect
import unittest

from bench_coe.innovation import run_bfj_audits
from bench_coe.innovation.blind_falsification_jury import (
    AuditObservation,
    BFJVariant,
    BasePrediction,
    BlindFalsificationJury,
    FalsificationQuestion,
    build_falsification_prompt,
    candidate_label_key,
    parse_audit_output,
)
from bench_coe.innovation.schema import EvaluationLabels, SourceTrainingLabels


def _question(question_id: str, environment: str = "env") -> FalsificationQuestion:
    return FalsificationQuestion(
        question_id=question_id,
        dataset="synthetic",
        environment=environment,
        question=f"Which option is correct for {question_id}?",
        options=("first", "second"),
        option_labels=("A", "B"),
    )


def _source_labels(
    answers: dict[str, str],
    base: list[BasePrediction],
) -> SourceTrainingLabels:
    correctness: dict[tuple[str, str], bool] = {}
    for question_id, answer in answers.items():
        for candidate in ("A", "B"):
            correctness[(question_id, candidate_label_key(candidate))] = candidate == answer
    for row in base:
        correctness[(row.question_id, row.expert_id)] = row.answer == answers[row.question_id]
    return SourceTrainingLabels._from_source_adapter(
        "synthetic",
        "development",
        correctness,
        {question_id: f"env::{index % 2}" for index, question_id in enumerate(answers)},
    )


class BlindFalsificationJuryTests(unittest.TestCase):
    def test_line_parser_fails_closed(self) -> None:
        parsed = parse_audit_output(
            "VERDICT: SURVIVES\nCONFIDENCE: 81\n"
            "ALTERNATIVE: (B)\nREASON: check",
            ("A", "B"),
        )
        self.assertEqual(parsed, ("SURVIVES", 81, "B", None))
        self.assertEqual(
            parse_audit_output("I think B", ("A", "B")),
            ("INCONCLUSIVE", 0, None, "missing_required_field"),
        )
        self.assertEqual(
            parse_audit_output(
                "VERDICT: FALSIFIED\nCONFIDENCE: 101\n"
                "ALTERNATIVE: A\nREASON: concrete check",
                ("A", "B"),
            )[-1],
            "confidence_out_of_range",
        )
        self.assertEqual(
            parse_audit_output(
                "VERDICT: FALSIFIED\nVERDICT: SURVIVES\nCONFIDENCE: 80\n"
                "ALTERNATIVE: A\nREASON: check",
                ("A", "B"),
            )[-1],
            "duplicate_required_field",
        )

    def test_prompt_is_identity_and_popularity_blind(self) -> None:
        prompt = build_falsification_prompt(_question("q"), "B")
        self.assertIn("Candidate under audit: (B) second", prompt)
        self.assertNotIn("Qwen", prompt)
        self.assertNotIn("3 votes", prompt)
        self.assertNotIn("correct answer is", prompt.lower())

    def test_open_option_can_rescue_an_unproposed_answer(self) -> None:
        train_questions = [_question(f"q{index}") for index in range(8)]
        train_answers = {
            question.question_id: ("A" if index % 2 == 0 else "B")
            for index, question in enumerate(train_questions)
        }
        base = [
            BasePrediction(question.question_id, expert, "A")
            for question in train_questions
            for expert in ("auditor1", "auditor2", "weak")
        ]
        audits: list[AuditObservation] = []
        for question in train_questions:
            answer = train_answers[question.question_id]
            for auditor in ("auditor1", "auditor2"):
                for candidate in ("A", "B"):
                    audits.append(
                        AuditObservation(
                            question.question_id,
                            auditor,
                            candidate,
                            "SURVIVES" if candidate == answer else "FALSIFIED",
                            90,
                            answer,
                        )
                    )
        model = BlindFalsificationJury(
            BFJVariant(
                "bfj_test",
                prior_strength=0.5,
                evidence_strength=1.0,
                use_confidence_bins=False,
                open_option_set=True,
            )
        ).fit(train_questions, base, audits, _source_labels(train_answers, base))

        target = _question("heldout")
        target_base = [
            BasePrediction("heldout", expert, "A")
            for expert in ("auditor1", "auditor2", "weak")
        ]
        target_audits = [
            AuditObservation("heldout", auditor, candidate, verdict, 90, "B")
            for auditor in ("auditor1", "auditor2")
            for candidate, verdict in (("A", "FALSIFIED"), ("B", "SURVIVES"))
        ]
        decision = model.predict([target], target_base, target_audits)[0]
        self.assertEqual(decision.answer, "B")
        self.assertTrue(decision.open_set_rescue)
        self.assertIsNone(decision.selected_expert_id)

    def test_prediction_is_order_invariant_and_has_no_label_argument(self) -> None:
        questions = [_question("q1"), _question("q2")]
        answers = {"q1": "A", "q2": "B"}
        base = [
            BasePrediction(question.question_id, expert, answer)
            for question, answer in zip(questions, ("A", "B"), strict=True)
            for expert in ("auditor1", "expert2")
        ]
        audits = [
            AuditObservation(question.question_id, "auditor1", candidate, verdict, 80, answer)
            for question, answer in zip(questions, ("A", "B"), strict=True)
            for candidate, verdict in (
                (answer, "SURVIVES"),
                ("B" if answer == "A" else "A", "FALSIFIED"),
            )
        ]
        model = BlindFalsificationJury(
            BFJVariant("order", use_confidence_bins=False)
        ).fit(questions, base, audits, _source_labels(answers, base))
        first = model.predict(questions, base, audits)
        second = model.predict(list(reversed(questions)), list(reversed(base)), list(reversed(audits)))
        self.assertEqual(
            [(row.question_id, row.answer) for row in first],
            [(row.question_id, row.answer) for row in second],
        )
        self.assertNotIn("labels", inspect.signature(model.predict).parameters)

    def test_evaluation_labels_cannot_fit_bfj(self) -> None:
        question = _question("q")
        base = [BasePrediction("q", "auditor", "A")]
        audits = [AuditObservation("q", "auditor", "A", "SURVIVES", 80, "A")]
        with self.assertRaises(TypeError):
            BlindFalsificationJury(BFJVariant("reject")).fit(
                [question],
                base,
                audits,
                EvaluationLabels("synthetic", "test", {}),  # type: ignore[arg-type]
            )

    def test_parse_failures_contribute_no_calibration_or_prediction_evidence(self) -> None:
        questions = [_question("q1"), _question("q2")]
        answers = {"q1": "A", "q2": "B"}
        base = [
            BasePrediction(question.question_id, "auditor", "A")
            for question in questions
        ]
        failed = [
            AuditObservation(
                question.question_id,
                "auditor",
                candidate,
                "INCONCLUSIVE",
                0,
                None,
                "missing_required_field",
            )
            for question in questions
            for candidate in ("A", "B")
        ]
        model = BlindFalsificationJury(BFJVariant("failed")).fit(
            questions, base, failed, _source_labels(answers, base)
        )
        self.assertEqual(sum(model.event_totals_.values()), 0)
        decision = model.predict([_question("heldout")], [BasePrediction("heldout", "auditor", "A")], [
            AuditObservation("heldout", "auditor", "B", "SURVIVES", 100, "B", "parse_error")
        ])[0]
        self.assertEqual(decision.diagnostics["audit_count_by_candidate"], {})

    def test_generation_module_does_not_import_label_reader(self) -> None:
        source = inspect.getsource(run_bfj_audits)
        self.assertNotIn("development_labels", source)
        self.assertNotIn("SourceTrainingLabels", source)
        self.assertNotIn("EvaluationLabels", source)


if __name__ == "__main__":
    unittest.main()
