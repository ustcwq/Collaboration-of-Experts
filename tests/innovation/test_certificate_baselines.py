from __future__ import annotations

import inspect
import unittest

from bench_coe.innovation.blind_falsification_jury import (
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
from bench_coe.innovation.certificate_baselines import (
    MinorityVetoCourt,
    MinorityVetoVariant,
    MinoritySentinelStyleCourt,
    MinoritySentinelStyleVariant,
    StaticCalibrationVariant,
    StaticCheckerCalibrationCourt,
)
from bench_coe.innovation.cross_examined_certificates import CertificateCheck
from bench_coe.innovation.schema import EvaluationLabels, SourceTrainingLabels


def _question(question_id: str, environment: str) -> FalsificationQuestion:
    return FalsificationQuestion(
        question_id,
        "synthetic",
        environment,
        f"Question {question_id}",
        ("first", "second"),
        ("A", "B"),
    )


def _check(question_id: str, candidate: str, correct: bool) -> CertificateCheck:
    return CertificateCheck(
        certificate_id=f"{question_id}::generator::{candidate}",
        question_id=question_id,
        generator_id="generator",
        checker_id="checker",
        candidate=candidate,
        status="VALID_SUPPORT" if correct else "VALID_REFUTATION",
        confidence=95,
        independent_answer="A",
        first_flaw="NONE",
        logic_status="VALID",
        eliminated_options=() if correct else (candidate,),
        supported_options=(candidate,) if correct else (),
        target_was_hidden=True,
    )


class CertificateBaselineTests(unittest.TestCase):
    def _training_data(self):
        questions = [_question(f"q{index}", f"env{index % 3}") for index in range(12)]
        answers = {
            question.question_id: ("A" if index % 2 == 0 else "B")
            for index, question in enumerate(questions)
        }
        base = [
            BasePrediction(question.question_id, expert, "A")
            for question in questions
            for expert in ("generator", "checker", "weak")
        ]
        checks = [
            _check(question.question_id, candidate, candidate == answers[question.question_id])
            for question in questions
            for candidate in question.option_labels
        ]
        correctness = {}
        environments = {}
        for question in questions:
            answer = answers[question.question_id]
            environments[question.question_id] = question.environment
            for expert in ("generator", "checker", "weak"):
                correctness[(question.question_id, expert)] = answer == "A"
            for candidate in question.option_labels:
                correctness[(question.question_id, candidate_label_key(candidate))] = (
                    candidate == answer
                )
        labels = SourceTrainingLabels._from_source_adapter(
            "synthetic", "development", correctness, environments
        )
        return questions, base, checks, labels

    def test_static_calibration_uses_checker_confusion_without_target_labels(self) -> None:
        questions, base, checks, labels = self._training_data()
        model = StaticCheckerCalibrationCourt(
            StaticCalibrationVariant("static", 0.0, 0.0)
        ).fit(questions, base, checks, labels)
        target = _question("heldout", "heldout")
        target_base = [
            BasePrediction("heldout", expert, "A")
            for expert in ("generator", "checker", "weak")
        ]
        target_checks = [_check("heldout", "A", False), _check("heldout", "B", True)]
        self.assertEqual(
            model.predict([target], target_base, target_checks)["heldout"], "B"
        )
        self.assertNotIn("labels", inspect.signature(model.predict).parameters)

    def test_minority_veto_only_overrides_after_valid_veto(self) -> None:
        questions, base, _checks, labels = self._training_data()
        model = MinorityVetoCourt(MinorityVetoVariant("veto", 1)).fit(
            questions, base, labels
        )
        target = _question("heldout", "heldout")
        target_base = [
            BasePrediction("heldout", expert, "A")
            for expert in ("generator", "checker", "weak")
        ]
        self.assertEqual(
            model.predict(
                [target],
                target_base,
                [_check("heldout", "A", False), _check("heldout", "B", True)],
            )["heldout"],
            "B",
        )
        self.assertEqual(model.predict([target], target_base, [])["heldout"], "A")

    def test_evaluation_labels_are_rejected(self) -> None:
        question = _question("q", "env")
        base = [BasePrediction("q", "expert", "A")]
        with self.assertRaises(TypeError):
            StaticCheckerCalibrationCourt(
                StaticCalibrationVariant("reject", 0.0, 0.0)
            ).fit(
                [question],
                base,
                [],
                EvaluationLabels("synthetic", "test", {}),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            MinorityVetoCourt(MinorityVetoVariant("reject", 1)).fit(
                [question],
                base,
                EvaluationLabels("synthetic", "test", {}),  # type: ignore[arg-type]
            )

    def test_minority_sentinel_style_learns_conservative_runner_flips(self) -> None:
        questions = [_question(f"sentinel-{index}", f"env{index % 4}") for index in range(40)]
        base = [
            BasePrediction(question.question_id, expert, answer)
            for question in questions
            for expert, answer in (
                ("generator", "A"),
                ("checker", "A"),
                ("weak", "B"),
            )
        ]
        checks = []
        correctness = {}
        environments = {}
        for index, question in enumerate(questions):
            answer = "B" if index % 2 == 0 else "A"
            environments[question.question_id] = question.environment
            for expert, prediction in (
                ("generator", "A"),
                ("checker", "A"),
                ("weak", "B"),
            ):
                correctness[(question.question_id, expert)] = prediction == answer
            for candidate in question.option_labels:
                correctness[(question.question_id, candidate_label_key(candidate))] = (
                    candidate == answer
                )
                checks.append(
                    _check(question.question_id, candidate, candidate == answer)
                )
        labels = SourceTrainingLabels._from_source_adapter(
            "synthetic", "development", correctness, environments
        )
        model = MinoritySentinelStyleCourt(
            MinoritySentinelStyleVariant(
                "sentinel-style", flip_threshold=0.6, n_estimators=30
            ),
            seed=7,
        ).fit(questions, base, checks, labels)

        positive = _question("sentinel-positive", "heldout")
        negative = _question("sentinel-negative", "heldout")
        target_base = [
            BasePrediction(question.question_id, expert, answer)
            for question in (positive, negative)
            for expert, answer in (
                ("generator", "A"),
                ("checker", "A"),
                ("weak", "B"),
            )
        ]
        target_checks = [
            _check(positive.question_id, candidate, candidate == "B")
            for candidate in positive.option_labels
        ] + [
            _check(negative.question_id, candidate, candidate == "A")
            for candidate in negative.option_labels
        ]
        predicted = model.predict(
            (positive, negative), target_base, target_checks
        )
        self.assertEqual(predicted[positive.question_id], "B")
        self.assertEqual(predicted[negative.question_id], "A")
        self.assertNotIn("labels", inspect.signature(model.predict).parameters)
        with self.assertRaises(TypeError):
            MinoritySentinelStyleCourt(
                MinoritySentinelStyleVariant("reject", 0.9)
            ).fit(
                [positive],
                target_base[:3],
                target_checks[:2],
                EvaluationLabels("synthetic", "test", {}),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
