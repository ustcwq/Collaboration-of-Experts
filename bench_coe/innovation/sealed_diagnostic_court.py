from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .blind_falsification_jury import (
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
from .schema import SourceTrainingLabels


@dataclass(frozen=True)
class DiagnosticProbe:
    probe_id: str
    question_id: str
    author_id: str
    first_candidate: str
    second_candidate: str
    author_stage0_prediction: str | None
    confidence: int
    parse_error: str | None
    abstained: bool
    left_candidate: str | None
    right_candidate: str | None

    def __post_init__(self) -> None:
        if self.first_candidate == self.second_candidate:
            raise ValueError("An SDB probe needs two candidates")
        if not 0 <= self.confidence <= 100:
            raise ValueError("SDB probe confidence must be in [0, 100]")
        if self.parse_error is None and not self.abstained:
            if {self.left_candidate, self.right_candidate} != {
                self.first_candidate,
                self.second_candidate,
            }:
                raise ValueError("SDB presented mapping is not the committed bijection")


@dataclass(frozen=True)
class DiagnosticProbeCheck:
    check_id: str
    probe_id: str
    question_id: str
    author_id: str
    checker_id: str
    outcome_side: str | None
    selected_candidate: str | None
    rejected_candidate: str | None
    confidence: int
    parse_error: str | None
    uncertain: bool

    def __post_init__(self) -> None:
        if self.author_id == self.checker_id:
            raise ValueError("An SDB author may not check its own probe")
        if not 0 <= self.confidence <= 100:
            raise ValueError("SDB check confidence must be in [0, 100]")
        if self.outcome_side not in {None, "LEFT", "RIGHT"}:
            raise ValueError("Unknown SDB outcome side")
        decided = self.parse_error is None and not self.uncertain
        if decided and (
            self.outcome_side is None
            or self.selected_candidate is None
            or self.rejected_candidate is None
            or self.selected_candidate == self.rejected_candidate
        ):
            raise ValueError("A decided SDB check needs a revealed candidate bijection")
        if not decided and any(
            value is not None
            for value in (
                self.selected_candidate,
                self.rejected_candidate,
            )
        ):
            raise ValueError("An undecided SDB check cannot reveal candidate evidence")


@dataclass(frozen=True)
class SDBVariant:
    name: str
    regularization_c: float = 1.0
    intervention_margin: float = 0.0
    minimum_probe_confidence: int = 0
    use_author_identity: bool = True
    use_checker_identity: bool = True
    use_author_checker_interaction: bool = True
    use_checker_stage0_relation: bool = True
    open_option_set: bool = True

    def __post_init__(self) -> None:
        if self.regularization_c <= 0.0:
            raise ValueError("SDB regularization C must be positive")
        if not 0.0 <= self.intervention_margin <= 1.0:
            raise ValueError("SDB intervention margin must be in [0, 1]")
        if not 0 <= self.minimum_probe_confidence <= 100:
            raise ValueError("SDB minimum probe confidence must be in [0, 100]")


@dataclass(frozen=True)
class SDBDecision:
    question_id: str
    answer: str
    reference_answer: str
    selected_expert_id: str | None
    candidate_logits: Mapping[str, float]
    candidate_probabilities: Mapping[str, float]
    fallback_reason: str | None
    open_set_rescue: bool
    diagnostics: Mapping[str, Any]


def _softmax(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    exponentials = {
        key: math.exp(float(np.clip(value - maximum, -60.0, 0.0)))
        for key, value in values.items()
    }
    total = sum(exponentials.values())
    return {key: value / max(total, 1e-12) for key, value in exponentials.items()}


class SealedDiagnosticBijectionCourt:
    """Source-trained candidate arbitration over delayed-reveal diagnostic outcomes."""

    def __init__(self, variant: SDBVariant, seed: int = 20260816) -> None:
        self.variant = variant
        self.seed = int(seed)

    @staticmethod
    def _base_by_question(
        base_predictions: Sequence[BasePrediction],
    ) -> dict[str, dict[str, str | None]]:
        result: dict[str, dict[str, str | None]] = defaultdict(dict)
        for row in base_predictions:
            if row.expert_id in result[row.question_id]:
                raise ValueError("Duplicate SDB base prediction")
            result[row.question_id][row.expert_id] = row.answer
        return dict(result)

    @staticmethod
    def _group_probes(
        probes: Sequence[DiagnosticProbe],
    ) -> tuple[
        dict[str, list[DiagnosticProbe]],
        dict[str, DiagnosticProbe],
    ]:
        by_question: dict[str, list[DiagnosticProbe]] = defaultdict(list)
        by_id: dict[str, DiagnosticProbe] = {}
        for probe in probes:
            if probe.probe_id in by_id:
                raise ValueError("Duplicate SDB probe")
            by_id[probe.probe_id] = probe
            by_question[probe.question_id].append(probe)
        return dict(by_question), by_id

    @staticmethod
    def _group_checks(
        checks: Sequence[DiagnosticProbeCheck],
        probe_by_id: Mapping[str, DiagnosticProbe],
    ) -> dict[str, list[DiagnosticProbeCheck]]:
        by_probe: dict[str, list[DiagnosticProbeCheck]] = defaultdict(list)
        seen: set[str] = set()
        for check in checks:
            if check.check_id in seen:
                raise ValueError("Duplicate SDB check")
            seen.add(check.check_id)
            probe = probe_by_id.get(check.probe_id)
            if probe is None:
                raise ValueError("SDB check refers to an unknown probe")
            if (
                check.question_id != probe.question_id
                or check.author_id != probe.author_id
            ):
                raise ValueError("SDB check metadata differs from its probe")
            if check.selected_candidate is not None and {
                check.selected_candidate,
                check.rejected_candidate,
            } != {probe.first_candidate, probe.second_candidate}:
                raise ValueError("SDB revealed check differs from its candidate pair")
            by_probe[check.probe_id].append(check)
        return dict(by_probe)

    def _candidate_features(
        self,
        question: FalsificationQuestion,
        candidate: str,
        base_by_question: Mapping[str, Mapping[str, str | None]],
        probes_by_question: Mapping[str, Sequence[DiagnosticProbe]],
        checks_by_probe: Mapping[str, Sequence[DiagnosticProbeCheck]],
    ) -> dict[str, float]:
        base = base_by_question.get(question.question_id, {})
        valid_votes = [answer for answer in base.values() if answer in question.option_labels]
        vote_counts = Counter(valid_votes)
        vote_total = max(1, len(valid_votes))
        vote_probabilities = [count / vote_total for count in vote_counts.values()]
        vote_entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in vote_probabilities
        )
        features: dict[str, float] = {
            "base::vote_fraction": vote_counts[candidate] / vote_total,
            "base::proposed": float(candidate in valid_votes),
            "base::valid_vote_fraction": len(valid_votes) / max(1, len(base)),
            "base::vote_entropy": vote_entropy,
            "base::reference_candidate": float(
                base.get(self.reference_expert_) == candidate
            ),
        }
        source_weighted = 0.0
        for expert in self.expert_ids_:
            relation = base.get(expert) == candidate
            features[f"base_vote::{expert}"] = float(relation)
            if relation:
                source_weighted += self.expert_accuracy_[expert]
        features["base::source_weighted_vote"] = source_weighted

        candidate_probes = [
            probe
            for probe in probes_by_question.get(question.question_id, ())
            if candidate in {probe.first_candidate, probe.second_candidate}
            and probe.parse_error is None
            and not probe.abstained
        ]
        features["probe::coverage"] = float(len(candidate_probes))
        for probe in candidate_probes:
            author_relation = (
                "same" if probe.author_stage0_prediction == candidate else "different"
            )
            features[f"probe::author_relation::{author_relation}"] = features.get(
                f"probe::author_relation::{author_relation}", 0.0
            ) + 1.0
            features["probe::candidate_presented_left"] = features.get(
                "probe::candidate_presented_left", 0.0
            ) + float(probe.left_candidate == candidate)
            features["probe::author_confidence_sum"] = features.get(
                "probe::author_confidence_sum", 0.0
            ) + probe.confidence / 100.0
            if self.variant.use_author_identity:
                features[f"probe::author::{probe.author_id}::coverage"] = 1.0
                features[
                    f"probe::author::{probe.author_id}::relation::{author_relation}"
                ] = 1.0
            if probe.confidence < self.variant.minimum_probe_confidence:
                features["probe::below_confidence_threshold"] = features.get(
                    "probe::below_confidence_threshold", 0.0
                ) + 1.0
                continue

            decided = [
                check
                for check in checks_by_probe.get(probe.probe_id, ())
                if check.parse_error is None and not check.uncertain
            ]
            uncertain = [
                check
                for check in checks_by_probe.get(probe.probe_id, ())
                if check.parse_error is None and check.uncertain
            ]
            failed = [
                check
                for check in checks_by_probe.get(probe.probe_id, ())
                if check.parse_error is not None
            ]
            features["check::decided_count"] = features.get(
                "check::decided_count", 0.0
            ) + len(decided)
            features["check::uncertain_count"] = features.get(
                "check::uncertain_count", 0.0
            ) + len(uncertain)
            features["check::parse_failure_count"] = features.get(
                "check::parse_failure_count", 0.0
            ) + len(failed)
            support_count = sum(
                check.selected_candidate == candidate for check in decided
            )
            reject_count = sum(
                check.rejected_candidate == candidate for check in decided
            )
            features["check::mapped_support"] = features.get(
                "check::mapped_support", 0.0
            ) + support_count
            features["check::mapped_reject"] = features.get(
                "check::mapped_reject", 0.0
            ) + reject_count
            if decided:
                features["check::probe_support_fraction_sum"] = features.get(
                    "check::probe_support_fraction_sum", 0.0
                ) + support_count / len(decided)
                features["check::probe_unanimous_support"] = features.get(
                    "check::probe_unanimous_support", 0.0
                ) + float(support_count == len(decided))
                features["check::probe_unanimous_reject"] = features.get(
                    "check::probe_unanimous_reject", 0.0
                ) + float(reject_count == len(decided))
            for check in decided:
                relation = (
                    "support"
                    if check.selected_candidate == candidate
                    else "reject"
                )
                confidence = check.confidence / 100.0
                features[f"check::mapped::{relation}"] = features.get(
                    f"check::mapped::{relation}", 0.0
                ) + 1.0
                features[f"check::mapped::{relation}::confidence_sum"] = features.get(
                    f"check::mapped::{relation}::confidence_sum", 0.0
                ) + confidence
                features[
                    f"check::mapped::{relation}::presented_side::{check.outcome_side}"
                ] = features.get(
                    f"check::mapped::{relation}::presented_side::{check.outcome_side}",
                    0.0,
                ) + 1.0
                if self.variant.use_checker_identity:
                    features[
                        f"check::checker::{check.checker_id}::mapped::{relation}"
                    ] = features.get(
                        f"check::checker::{check.checker_id}::mapped::{relation}",
                        0.0,
                    ) + 1.0
                if self.variant.use_author_checker_interaction:
                    features[
                        f"check::pair::{probe.author_id}::{check.checker_id}::mapped::{relation}"
                    ] = features.get(
                        f"check::pair::{probe.author_id}::{check.checker_id}::mapped::{relation}",
                        0.0,
                    ) + 1.0
                if self.variant.use_checker_stage0_relation:
                    checker_relation = (
                        "same"
                        if base.get(check.checker_id) == candidate
                        else "different"
                    )
                    features[
                        f"check::checker_stage0::{checker_relation}::mapped::{relation}"
                    ] = features.get(
                        f"check::checker_stage0::{checker_relation}::mapped::{relation}",
                        0.0,
                    ) + 1.0
        return features

    def fit(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        probes: Sequence[DiagnosticProbe],
        checks: Sequence[DiagnosticProbeCheck],
        labels: SourceTrainingLabels,
    ) -> "SealedDiagnosticBijectionCourt":
        if not isinstance(labels, SourceTrainingLabels):
            raise TypeError("SDB fitting requires SourceTrainingLabels")
        question_ids = {question.question_id for question in questions}
        base_by_question = self._base_by_question(base_predictions)
        if set(base_by_question) != question_ids:
            raise ValueError("SDB fitting base-prediction scope differs from questions")
        expert_sets = {tuple(sorted(rows)) for rows in base_by_question.values()}
        if len(expert_sets) != 1:
            raise ValueError("SDB fitting has an inconsistent expert pool")
        self.expert_ids_ = tuple(sorted(next(iter(expert_sets))))
        self.expert_accuracy_ = {}
        for expert in self.expert_ids_:
            values = [
                labels.get(question_id, expert) for question_id in sorted(question_ids)
            ]
            observed = [bool(value) for value in values if value is not None]
            if not observed:
                raise ValueError(f"SDB lacks source correctness for expert {expert}")
            self.expert_accuracy_[expert] = float(np.mean(observed))
        self.reference_expert_ = sorted(
            self.expert_ids_,
            key=lambda expert: (-self.expert_accuracy_[expert], expert),
        )[0]
        probes_by_question, probe_by_id = self._group_probes(probes)
        checks_by_probe = self._group_checks(checks, probe_by_id)

        feature_rows: list[dict[str, float]] = []
        targets: list[int] = []
        for question in questions:
            for candidate in question.option_labels:
                correctness = labels.get(
                    question.question_id, candidate_label_key(candidate)
                )
                if correctness is None:
                    raise PermissionError(
                        "SDB source fitting lacks candidate-level correctness"
                    )
                feature_rows.append(
                    self._candidate_features(
                        question,
                        candidate,
                        base_by_question,
                        probes_by_question,
                        checks_by_probe,
                    )
                )
                targets.append(int(correctness))
        if len(set(targets)) != 2:
            raise ValueError("SDB source fitting needs positive and negative candidates")
        self.vectorizer_ = DictVectorizer(sparse=True)
        matrix = self.vectorizer_.fit_transform(feature_rows)
        self.model_ = LogisticRegression(
            C=self.variant.regularization_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=self.seed,
        ).fit(matrix, np.asarray(targets, dtype=np.int64))
        return self

    def with_variant(self, variant: SDBVariant) -> "SealedDiagnosticBijectionCourt":
        if variant.regularization_c != self.variant.regularization_c:
            raise ValueError("A fitted SDB court cannot change regularization C")
        if replace(variant, name=self.variant.name, intervention_margin=self.variant.intervention_margin) != replace(
            self.variant,
            intervention_margin=self.variant.intervention_margin,
        ):
            raise ValueError("A fitted SDB court can only change intervention margin and name")
        clone = SealedDiagnosticBijectionCourt(variant, self.seed)
        clone.expert_ids_ = self.expert_ids_
        clone.expert_accuracy_ = self.expert_accuracy_
        clone.reference_expert_ = self.reference_expert_
        clone.vectorizer_ = self.vectorizer_
        clone.model_ = self.model_
        return clone

    def predict(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        probes: Sequence[DiagnosticProbe],
        checks: Sequence[DiagnosticProbeCheck],
    ) -> list[SDBDecision]:
        base_by_question = self._base_by_question(base_predictions)
        probes_by_question, probe_by_id = self._group_probes(probes)
        checks_by_probe = self._group_checks(checks, probe_by_id)
        decisions: list[SDBDecision] = []
        for question in questions:
            base = base_by_question.get(question.question_id, {})
            if set(base) != set(self.expert_ids_):
                raise ValueError("SDB prediction has an inconsistent expert pool")
            candidates = tuple(question.option_labels)
            features = [
                self._candidate_features(
                    question,
                    candidate,
                    base_by_question,
                    probes_by_question,
                    checks_by_probe,
                )
                for candidate in candidates
            ]
            matrix = self.vectorizer_.transform(features)
            logits_array = self.model_.decision_function(matrix)
            logits = {
                candidate: float(logit)
                for candidate, logit in zip(candidates, logits_array, strict=True)
            }
            probabilities = _softmax(logits)
            option_order = {candidate: index for index, candidate in enumerate(candidates)}
            winner = sorted(
                candidates,
                key=lambda candidate: (-probabilities[candidate], option_order[candidate]),
            )[0]
            reference = base.get(self.reference_expert_)
            if reference not in candidates:
                valid_votes = [value for value in base.values() if value in candidates]
                counts = Counter(valid_votes)
                reference = sorted(
                    candidates,
                    key=lambda candidate: (-counts[candidate], option_order[candidate]),
                )[0]
            reference = str(reference)
            margin = probabilities[winner] - probabilities[reference]
            fallback_reason = None
            answer = winner
            if winner != reference and margin < self.variant.intervention_margin:
                answer = reference
                fallback_reason = "posterior_margin_below_source_selected_threshold"
            supporters = [
                expert for expert in self.expert_ids_ if base.get(expert) == answer
            ]
            selected_expert = (
                sorted(
                    supporters,
                    key=lambda expert: (-self.expert_accuracy_[expert], expert),
                )[0]
                if supporters
                else None
            )
            question_probes = probes_by_question.get(question.question_id, ())
            question_checks = [
                check
                for probe in question_probes
                for check in checks_by_probe.get(probe.probe_id, ())
            ]
            decisions.append(
                SDBDecision(
                    question_id=question.question_id,
                    answer=answer,
                    reference_answer=reference,
                    selected_expert_id=selected_expert,
                    candidate_logits=logits,
                    candidate_probabilities=probabilities,
                    fallback_reason=fallback_reason,
                    open_set_rescue=answer not in base.values(),
                    diagnostics={
                        "reference_expert": self.reference_expert_,
                        "posterior_winner": winner,
                        "posterior_margin_vs_reference": margin,
                        "valid_base_votes": sum(
                            value in candidates for value in base.values()
                        ),
                        "parsed_nonabstaining_probes": sum(
                            probe.parse_error is None and not probe.abstained
                            for probe in question_probes
                        ),
                        "parsed_decided_checks": sum(
                            check.parse_error is None and not check.uncertain
                            for check in question_checks
                        ),
                        "tie_breaking": "posterior_then_query_local_option_order",
                    },
                )
            )
        return decisions
