from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer

from .blind_falsification_jury import (
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
from .cross_examined_certificates import CertificateCheck
from .schema import SourceTrainingLabels


_SIGNALS = ("eliminate", "support", "neutral", "invalid")


def checker_signal(check: CertificateCheck) -> str:
    if check.parse_error is not None:
        return "invalid"
    if check.status == "VALID_REFUTATION":
        return "eliminate"
    if check.status == "VALID_SUPPORT":
        return "support"
    if check.status in {"INVALID_REFUTATION", "INVALID_SUPPORT"}:
        return "invalid"
    return "neutral"


def _softmax(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    exponentials = {
        key: math.exp(float(np.clip(value - maximum, -60.0, 0.0)))
        for key, value in values.items()
    }
    total = sum(exponentials.values())
    return {key: value / max(total, 1e-12) for key, value in exponentials.items()}


def _base_by_question(
    base_predictions: Sequence[BasePrediction],
) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in base_predictions:
        if row.expert_id in result[row.question_id]:
            raise ValueError("Duplicate certificate-baseline prediction")
        result[row.question_id][row.expert_id] = row.answer
    return result


def _fit_expert_reference(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    labels: SourceTrainingLabels,
) -> tuple[tuple[str, ...], dict[str, float], str]:
    expert_ids = tuple(sorted({row.expert_id for row in base_predictions}))
    if not expert_ids:
        raise ValueError("Certificate baselines require base experts")
    question_ids = tuple(row.question_id for row in questions)
    accuracy: dict[str, float] = {}
    for expert in expert_ids:
        outcomes = [labels.get(question_id, expert) for question_id in question_ids]
        if any(value is None for value in outcomes):
            raise ValueError("Certificate baseline lacks source expert correctness")
        successes = sum(bool(value) for value in outcomes)
        accuracy[expert] = (successes + 1.0) / (len(outcomes) + 2.0)
    reference = sorted(expert_ids, key=lambda expert: (-accuracy[expert], expert))[0]
    return expert_ids, accuracy, reference


@dataclass(frozen=True)
class StaticCalibrationVariant:
    name: str
    base_prior_strength: float
    intervention_margin: float

    def __post_init__(self) -> None:
        if self.base_prior_strength < 0.0:
            raise ValueError("Static calibration prior strength must be nonnegative")
        if not 0.0 <= self.intervention_margin <= 1.0:
            raise ValueError("Static calibration margin must be in [0, 1]")


class StaticCheckerCalibrationCourt:
    """Checker-specific static confusion calibration without pair/answer dependence."""

    def __init__(self, variant: StaticCalibrationVariant) -> None:
        self.variant = variant

    def fit(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        checks: Sequence[CertificateCheck],
        labels: SourceTrainingLabels,
    ) -> "StaticCheckerCalibrationCourt":
        if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
            raise TypeError("Static checker calibration requires SourceTrainingLabels")
        question_by_id = {row.question_id: row for row in questions}
        self.expert_ids_, self.expert_accuracy_, self.reference_expert_ = (
            _fit_expert_reference(questions, base_predictions, labels)
        )
        self.checker_ids_ = tuple(sorted({row.checker_id for row in checks}))
        counts: dict[str, dict[bool, Counter[str]]] = {
            checker: {False: Counter(), True: Counter()}
            for checker in self.checker_ids_
        }
        pooled = {False: Counter(), True: Counter()}
        for check in checks:
            question = question_by_id.get(check.question_id)
            if question is None or check.candidate not in question.option_labels:
                raise ValueError("Static calibration check is outside the source questions")
            truth = labels.get(
                check.question_id, candidate_label_key(check.candidate)
            )
            if truth is None:
                raise ValueError("Static calibration lacks candidate correctness")
            signal = checker_signal(check)
            counts[check.checker_id][bool(truth)][signal] += 1
            pooled[bool(truth)][signal] += 1
        self.log_likelihood_ratio_: dict[tuple[str, str], float] = {}
        for checker in self.checker_ids_:
            for signal in _SIGNALS:
                positive = (counts[checker][True][signal] + 1.0) / (
                    sum(counts[checker][True].values()) + len(_SIGNALS)
                )
                negative = (counts[checker][False][signal] + 1.0) / (
                    sum(counts[checker][False].values()) + len(_SIGNALS)
                )
                self.log_likelihood_ratio_[(checker, signal)] = math.log(
                    positive / negative
                )
        self.pooled_log_likelihood_ratio_: dict[str, float] = {}
        for signal in _SIGNALS:
            positive = (pooled[True][signal] + 1.0) / (
                sum(pooled[True].values()) + len(_SIGNALS)
            )
            negative = (pooled[False][signal] + 1.0) / (
                sum(pooled[False].values()) + len(_SIGNALS)
            )
            self.pooled_log_likelihood_ratio_[signal] = math.log(positive / negative)
        return self

    def predict(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        checks: Sequence[CertificateCheck],
    ) -> dict[str, str]:
        if not hasattr(self, "reference_expert_"):
            raise RuntimeError("Static checker calibration must be fitted")
        base_by_question = _base_by_question(base_predictions)
        checks_by_key: dict[tuple[str, str], list[CertificateCheck]] = defaultdict(list)
        for check in checks:
            checks_by_key[(check.question_id, check.candidate)].append(check)
        answers: dict[str, str] = {}
        for question in questions:
            base = base_by_question.get(question.question_id, {})
            reference = base.get(self.reference_expert_)
            if reference not in question.option_labels:
                reference = question.option_labels[0]
            scores: dict[str, float] = {}
            total_weight = sum(self.expert_accuracy_.values())
            for candidate in question.option_labels:
                weighted_vote = sum(
                    self.expert_accuracy_[expert]
                    for expert, answer in base.items()
                    if expert in self.expert_accuracy_ and answer == candidate
                ) / max(total_weight, 1e-12)
                score = self.variant.base_prior_strength * weighted_vote
                for check in checks_by_key.get((question.question_id, candidate), ()):
                    signal = checker_signal(check)
                    score += self.log_likelihood_ratio_.get(
                        (check.checker_id, signal),
                        self.pooled_log_likelihood_ratio_[signal],
                    )
                scores[candidate] = score
            probabilities = _softmax(scores)
            ranked = sorted(
                question.option_labels,
                key=lambda candidate: (
                    -probabilities[candidate],
                    candidate != reference,
                    candidate,
                ),
            )
            chosen = ranked[0]
            runner = probabilities[ranked[1]] if len(ranked) > 1 else 0.0
            if (
                chosen != reference
                and probabilities[chosen] - runner + 1e-12
                < self.variant.intervention_margin
            ):
                chosen = str(reference)
            answers[question.question_id] = chosen
        return answers


@dataclass(frozen=True)
class MinorityVetoVariant:
    name: str
    veto_threshold: int

    def __post_init__(self) -> None:
        if self.veto_threshold <= 0:
            raise ValueError("Minority veto threshold must be positive")


class MinorityVetoCourt:
    """Rejects a candidate only after a source-selected checker minority veto."""

    def __init__(self, variant: MinorityVetoVariant) -> None:
        self.variant = variant

    def fit(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        labels: SourceTrainingLabels,
    ) -> "MinorityVetoCourt":
        if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
            raise TypeError("Minority veto requires SourceTrainingLabels")
        self.expert_ids_, self.expert_accuracy_, self.reference_expert_ = (
            _fit_expert_reference(questions, base_predictions, labels)
        )
        return self

    def predict(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        checks: Sequence[CertificateCheck],
    ) -> dict[str, str]:
        if not hasattr(self, "reference_expert_"):
            raise RuntimeError("Minority veto must be fitted")
        base_by_question = _base_by_question(base_predictions)
        checks_by_key: dict[tuple[str, str], list[CertificateCheck]] = defaultdict(list)
        for check in checks:
            checks_by_key[(check.question_id, check.candidate)].append(check)
        answers: dict[str, str] = {}
        for question in questions:
            base = base_by_question.get(question.question_id, {})
            reference = base.get(self.reference_expert_)
            if reference not in question.option_labels:
                reference = question.option_labels[0]
            eliminated: set[str] = set()
            support_count: Counter[str] = Counter()
            for candidate in question.option_labels:
                candidate_checks = checks_by_key.get((question.question_id, candidate), ())
                vetoes = {
                    check.checker_id
                    for check in candidate_checks
                    if checker_signal(check) == "eliminate"
                }
                supports = {
                    check.checker_id
                    for check in candidate_checks
                    if checker_signal(check) == "support"
                }
                if len(vetoes) >= self.variant.veto_threshold:
                    eliminated.add(candidate)
                support_count[candidate] = len(supports)
            survivors = [
                candidate
                for candidate in question.option_labels
                if candidate not in eliminated
            ]
            if not survivors:
                answers[question.question_id] = str(reference)
                continue
            if reference in survivors:
                answers[question.question_id] = str(reference)
                continue
            answers[question.question_id] = sorted(
                survivors,
                key=lambda candidate: (
                    -sum(
                        self.expert_accuracy_.get(expert, 0.0)
                        for expert, answer in base.items()
                        if answer == candidate
                    ),
                    -support_count[candidate],
                    candidate,
                ),
            )[0]
        return answers


@dataclass(frozen=True)
class MinoritySentinelStyleVariant:
    """A source-fitted approximation, not an official Minority Sentinel reproduction."""

    name: str
    flip_threshold: float
    n_estimators: int = 50
    learning_rate: float = 0.05
    max_depth: int = 2

    def __post_init__(self) -> None:
        if not 0.5 <= self.flip_threshold <= 1.0:
            raise ValueError("Minority-sentinel threshold must be in [0.5, 1]")
        if self.n_estimators <= 0 or self.learning_rate <= 0.0 or self.max_depth <= 0:
            raise ValueError("Minority-sentinel tree parameters must be positive")


class MinoritySentinelStyleCourt:
    """Conservative plurality-to-runner flip gate over frozen observable evidence.

    This adapts the diagnosis/cure shape of Minority Sentinel to the available C3
    artifacts. It has no debate rounds or official semantic-audit features and is
    therefore deliberately labeled ``style`` rather than a reproduction.
    """

    def __init__(
        self, variant: MinoritySentinelStyleVariant, seed: int = 20260815
    ) -> None:
        self.variant = variant
        self.seed = int(seed)

    @staticmethod
    def _ranked_answers(
        question: FalsificationQuestion,
        base: Mapping[str, str | None],
        expert_accuracy: Mapping[str, float],
    ) -> tuple[str, str | None]:
        counts = Counter(
            answer for answer in base.values() if answer in question.option_labels
        )
        weighted = {
            candidate: sum(
                expert_accuracy.get(expert, 0.0)
                for expert, answer in base.items()
                if answer == candidate
            )
            for candidate in question.option_labels
        }
        proposed = [candidate for candidate in question.option_labels if counts[candidate] > 0]
        if not proposed:
            return question.option_labels[0], None
        ranked = sorted(
            proposed,
            key=lambda candidate: (-counts[candidate], -weighted[candidate], candidate),
        )
        return ranked[0], ranked[1] if len(ranked) > 1 else None

    def _features(
        self,
        question: FalsificationQuestion,
        base: Mapping[str, str | None],
        checks: Sequence[CertificateCheck],
        majority: str,
        runner: str,
    ) -> dict[str, float]:
        valid = [answer for answer in base.values() if answer in question.option_labels]
        counts = Counter(valid)
        total_weight = sum(self.expert_accuracy_.values())
        weighted = {
            candidate: sum(
                self.expert_accuracy_.get(expert, 0.0)
                for expert, answer in base.items()
                if answer == candidate
            )
            / max(total_weight, 1e-12)
            for candidate in (majority, runner)
        }
        features: dict[str, float] = {
            "vote::majority_fraction": counts[majority] / max(1, len(valid)),
            "vote::runner_fraction": counts[runner] / max(1, len(valid)),
            "vote::margin_fraction": (counts[majority] - counts[runner])
            / max(1, len(valid)),
            "vote::unique_answers": float(len(set(valid))),
            "vote::missing_fraction": 1.0 - len(valid) / max(1, len(base)),
            "weighted::majority": weighted[majority],
            "weighted::runner": weighted[runner],
            "weighted::runner_minus_majority": weighted[runner] - weighted[majority],
        }
        for expert in self.expert_ids_:
            answer = base.get(expert)
            features[f"expert::{expert}::majority"] = float(answer == majority)
            features[f"expert::{expert}::runner"] = float(answer == runner)

        paired: dict[tuple[str, str, str], list[CertificateCheck]] = defaultdict(list)
        for check in checks:
            camp = (
                "majority"
                if check.candidate == majority
                else "runner"
                if check.candidate == runner
                else "other"
            )
            signal = checker_signal(check)
            features[f"audit::{camp}::{signal}"] = features.get(
                f"audit::{camp}::{signal}", 0.0
            ) + 1.0
            features[f"audit::{camp}::confidence_sum"] = features.get(
                f"audit::{camp}::confidence_sum", 0.0
            ) + check.confidence / 100.0
            features[
                f"audit::{camp}::checker::{check.checker_id}::{signal}"
            ] = features.get(
                f"audit::{camp}::checker::{check.checker_id}::{signal}", 0.0
            ) + 1.0
            if check.independent_answer == majority:
                features[f"audit::{camp}::commitment_majority"] = features.get(
                    f"audit::{camp}::commitment_majority", 0.0
                ) + 1.0
            elif check.independent_answer == runner:
                features[f"audit::{camp}::commitment_runner"] = features.get(
                    f"audit::{camp}::commitment_runner", 0.0
                ) + 1.0
            if check.counterfactual_pair:
                paired[(check.certificate_id, check.checker_id, camp)].append(check)
        for (_certificate_id, _checker_id, camp), members in paired.items():
            by_view = {
                member.orientation: member
                for member in members
                if member.orientation in {"trace_1", "trace_2"}
            }
            if set(by_view) != {"trace_1", "trace_2"}:
                continue
            complete = all(member.parse_error is None for member in by_view.values())
            exactly_one_valid = complete and sorted(
                member.logic_status for member in by_view.values()
            ) == ["INVALID", "VALID"]
            features[f"audit::{camp}::isolated_pair_complete"] = features.get(
                f"audit::{camp}::isolated_pair_complete", 0.0
            ) + float(complete)
            features[f"audit::{camp}::one_valid_one_invalid"] = features.get(
                f"audit::{camp}::one_valid_one_invalid", 0.0
            ) + float(exactly_one_valid)
        for key in (
            "support",
            "eliminate",
            "invalid",
            "neutral",
            "confidence_sum",
            "commitment_majority",
            "commitment_runner",
            "isolated_pair_complete",
            "one_valid_one_invalid",
        ):
            features[f"contrast::runner_minus_majority::{key}"] = features.get(
                f"audit::runner::{key}", 0.0
            ) - features.get(f"audit::majority::{key}", 0.0)
        return features

    def fit(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        checks: Sequence[CertificateCheck],
        labels: SourceTrainingLabels,
    ) -> "MinoritySentinelStyleCourt":
        if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
            raise TypeError("Minority-sentinel style fitting requires SourceTrainingLabels")
        self.expert_ids_, self.expert_accuracy_, self.reference_expert_ = (
            _fit_expert_reference(questions, base_predictions, labels)
        )
        base_by_question = _base_by_question(base_predictions)
        checks_by_question: dict[str, list[CertificateCheck]] = defaultdict(list)
        for check in checks:
            checks_by_question[check.question_id].append(check)
        feature_rows: list[dict[str, float]] = []
        targets: list[int] = []
        for question in questions:
            base = base_by_question.get(question.question_id, {})
            majority, runner = self._ranked_answers(
                question, base, self.expert_accuracy_
            )
            if runner is None:
                continue
            runner_correct = labels.get(
                question.question_id, candidate_label_key(runner)
            )
            if runner_correct is None:
                raise ValueError("Minority-sentinel style fitting lacks candidate labels")
            feature_rows.append(
                self._features(
                    question,
                    base,
                    checks_by_question.get(question.question_id, ()),
                    majority,
                    runner,
                )
            )
            targets.append(int(bool(runner_correct)))
        self.vectorizer_ = DictVectorizer(sparse=False)
        if not feature_rows:
            self.vectorizer_.fit([{}])
            self.model_ = None
            self.constant_probability_ = 0.0
            return self
        matrix = self.vectorizer_.fit_transform(feature_rows)
        self.constant_probability_ = (sum(targets) + 1.0) / (len(targets) + 2.0)
        if len(set(targets)) < 2:
            self.model_ = None
            return self
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.variant.n_estimators,
            learning_rate=self.variant.learning_rate,
            max_depth=self.variant.max_depth,
            random_state=self.seed,
        ).fit(matrix, np.asarray(targets, dtype=np.int64))
        return self

    def predict(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        checks: Sequence[CertificateCheck],
    ) -> dict[str, str]:
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("Minority-sentinel style court must be fitted")
        base_by_question = _base_by_question(base_predictions)
        checks_by_question: dict[str, list[CertificateCheck]] = defaultdict(list)
        for check in checks:
            checks_by_question[check.question_id].append(check)
        answers: dict[str, str] = {}
        for question in questions:
            base = base_by_question.get(question.question_id, {})
            majority, runner = self._ranked_answers(
                question, base, self.expert_accuracy_
            )
            if runner is None:
                answers[question.question_id] = majority
                continue
            feature_row = self._features(
                question,
                base,
                checks_by_question.get(question.question_id, ()),
                majority,
                runner,
            )
            if self.model_ is None:
                probability = self.constant_probability_
            else:
                matrix = self.vectorizer_.transform([feature_row])
                positive_index = list(self.model_.classes_).index(1)
                probability = float(self.model_.predict_proba(matrix)[0, positive_index])
            answers[question.question_id] = (
                runner if probability >= self.variant.flip_threshold else majority
            )
        return answers
