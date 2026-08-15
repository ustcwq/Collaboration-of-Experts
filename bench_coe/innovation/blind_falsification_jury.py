from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .schema import SourceTrainingLabels


VERDICTS = ("FALSIFIED", "INCONCLUSIVE", "SURVIVES")
FORBIDDEN_AUDIT_KEYS = frozenset(
    {"answer", "gold", "target", "correct", "correctness", "is_correct"}
)


@dataclass(frozen=True)
class FalsificationQuestion:
    question_id: str
    dataset: str
    environment: str
    question: str
    options: tuple[str, ...]
    option_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question_id or not self.question:
            raise ValueError("Falsification questions require an ID and text")
        if len(self.options) < 2 or len(self.options) != len(self.option_labels):
            raise ValueError("Question options and labels are not aligned")
        if len(set(self.option_labels)) != len(self.option_labels):
            raise ValueError("Question option labels must be unique")


@dataclass(frozen=True)
class BasePrediction:
    question_id: str
    expert_id: str
    answer: str | None


@dataclass(frozen=True)
class AuditObservation:
    question_id: str
    auditor_id: str
    candidate: str
    verdict: str
    confidence: int
    alternative: str | None
    parse_error: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"Unknown falsification verdict: {self.verdict}")
        if not 0 <= self.confidence <= 100:
            raise ValueError("Audit confidence must be in [0, 100]")


@dataclass(frozen=True)
class BFJVariant:
    name: str
    prior_strength: float = 1.0
    evidence_strength: float = 1.0
    smoothing: float = 1.0
    max_abs_log_likelihood_ratio: float = 2.5
    confidence_threshold: int = 67
    use_confidence_bins: bool = True
    calibrate_self_bias: bool = True
    open_option_set: bool = True
    intervention_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_strength < 0.0 or self.evidence_strength < 0.0:
            raise ValueError("BFJ score strengths must be non-negative")
        if self.smoothing <= 0.0:
            raise ValueError("BFJ smoothing must be positive")
        if self.max_abs_log_likelihood_ratio <= 0.0:
            raise ValueError("BFJ likelihood-ratio cap must be positive")
        if not 0 <= self.confidence_threshold <= 100:
            raise ValueError("BFJ confidence threshold must be in [0, 100]")
        if not 0.0 <= self.intervention_margin <= 1.0:
            raise ValueError("BFJ intervention margin must be in [0, 1]")


@dataclass(frozen=True)
class BFJDecision:
    question_id: str
    answer: str
    reference_answer: str
    selected_expert_id: str | None
    candidate_scores: Mapping[str, float]
    candidate_probabilities: Mapping[str, float]
    fallback_reason: str | None
    open_set_rescue: bool
    diagnostics: Mapping[str, object]


_VERDICT_RE = re.compile(
    r"^\s*VERDICT\s*:\s*(FALSIFIED|INCONCLUSIVE|SURVIVES)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_RE = re.compile(
    r"^\s*CONFIDENCE\s*:\s*(\d{1,3})\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ALTERNATIVE_RE = re.compile(
    r"^\s*ALTERNATIVE\s*:\s*\(?([A-Z]|NONE)\)?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_REASON_RE = re.compile(
    r"^\s*REASON\s*:\s*(\S.*?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def build_falsification_prompt(question: FalsificationQuestion, candidate: str) -> str:
    if candidate not in question.option_labels:
        raise ValueError(f"Candidate {candidate!r} is not an option for {question.question_id}")
    option_lines = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    candidate_text = question.options[question.option_labels.index(candidate)]
    return (
        "You are one seat in an anonymous falsification jury. No model identity, "
        "vote count, popularity signal, or gold answer is available to you.\n\n"
        "Independently solve the problem, then audit the single candidate below. "
        "Actively search for a decisive contradiction, failed assumption, numerical "
        "counterexample, or option mismatch. Do not defend the candidate merely "
        "because it was presented. Mark SURVIVES only when you can positively verify "
        "the candidate and find no decisive flaw; mark FALSIFIED only for a concrete "
        "fatal flaw; otherwise mark INCONCLUSIVE.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{option_lines}\n\n"
        f"Candidate under audit: ({candidate}) {candidate_text}\n\n"
        "Return exactly four single-line fields in the format below. Do not use XML "
        "tags, parentheses, bullets, or any text before or after the fields. "
        "ALTERNATIVE is your independently preferred option label, or NONE if "
        "unresolved.\n"
        "VERDICT: FALSIFIED|INCONCLUSIVE|SURVIVES\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "ALTERNATIVE: option label or NONE\n"
        "REASON: one concise decisive check"
    )


def parse_audit_output(
    text: str,
    option_labels: Sequence[str],
) -> tuple[str, int, str | None, str | None]:
    labels = {str(label).upper() for label in option_labels}
    verdict_matches = _VERDICT_RE.findall(text)
    confidence_matches = _CONFIDENCE_RE.findall(text)
    alternative_matches = _ALTERNATIVE_RE.findall(text)
    reason_matches = _REASON_RE.findall(text)
    match_counts = (
        len(verdict_matches),
        len(confidence_matches),
        len(alternative_matches),
        len(reason_matches),
    )
    if any(count == 0 for count in match_counts):
        return "INCONCLUSIVE", 0, None, "missing_required_field"
    if any(count != 1 for count in match_counts):
        return "INCONCLUSIVE", 0, None, "duplicate_required_field"
    confidence = int(confidence_matches[0])
    if not 0 <= confidence <= 100:
        return "INCONCLUSIVE", 0, None, "confidence_out_of_range"
    alternative_raw = alternative_matches[0].upper()
    if alternative_raw != "NONE" and alternative_raw not in labels:
        return "INCONCLUSIVE", 0, None, "alternative_outside_option_set"
    return (
        verdict_matches[0].upper(),
        confidence,
        None if alternative_raw == "NONE" else alternative_raw,
        None,
    )


def candidate_label_key(candidate: str) -> str:
    return f"candidate::{candidate}"


def _relation(
    auditor_id: str,
    candidate: str,
    base_by_question: Mapping[str, Mapping[str, str | None]],
    question_id: str,
    calibrate_self_bias: bool,
) -> str:
    if not calibrate_self_bias:
        return "pooled"
    answer = base_by_question.get(question_id, {}).get(auditor_id)
    if answer is None:
        return "missing"
    return "self" if answer == candidate else "other"


def _event(observation: AuditObservation, variant: BFJVariant) -> str:
    if not variant.use_confidence_bins:
        return observation.verdict
    confidence_bin = (
        "high" if observation.confidence >= variant.confidence_threshold else "low"
    )
    return f"{observation.verdict}:{confidence_bin}"


def _event_vocabulary(variant: BFJVariant) -> tuple[str, ...]:
    if not variant.use_confidence_bins:
        return VERDICTS
    return tuple(f"{verdict}:{level}" for verdict in VERDICTS for level in ("low", "high"))


def _softmax(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    exponentials = {
        key: math.exp(float(np.clip(value - maximum, -60.0, 0.0)))
        for key, value in values.items()
    }
    scale = sum(exponentials.values())
    return {key: value / max(scale, 1e-12) for key, value in exponentials.items()}


class BlindFalsificationJury:
    """Source-calibrated, identity-blind option-falsification aggregation."""

    def __init__(self, variant: BFJVariant) -> None:
        self.variant = variant

    def fit(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        audits: Sequence[AuditObservation],
        labels: SourceTrainingLabels,
    ) -> "BlindFalsificationJury":
        if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
            raise TypeError("BFJ may be fitted only with SourceTrainingLabels")
        question_by_id = {row.question_id: row for row in questions}
        if len(question_by_id) != len(questions):
            raise ValueError("BFJ training questions contain duplicate IDs")
        if set(question_by_id).difference(labels.environment_by_question):
            raise ValueError("BFJ training labels lack question environments")
        self.question_ids_ = tuple(sorted(question_by_id))
        self.expert_ids_ = tuple(sorted({row.expert_id for row in base_predictions}))
        self.auditor_ids_ = tuple(sorted({row.auditor_id for row in audits}))
        if not self.expert_ids_ or not self.auditor_ids_:
            raise ValueError("BFJ training requires base experts and auditors")
        base_by_question: dict[str, dict[str, str | None]] = defaultdict(dict)
        for row in base_predictions:
            if row.question_id not in question_by_id:
                raise ValueError(f"Unknown BFJ base-prediction question: {row.question_id}")
            if row.expert_id in base_by_question[row.question_id]:
                raise ValueError("Duplicate BFJ question/expert base prediction")
            base_by_question[row.question_id][row.expert_id] = row.answer
        self.expert_accuracy_: dict[str, float] = {}
        for expert in self.expert_ids_:
            observed = [
                labels.get(question_id, expert)
                for question_id in self.question_ids_
                if labels.get(question_id, expert) is not None
            ]
            successes = sum(bool(value) for value in observed)
            self.expert_accuracy_[expert] = (successes + 1.0) / (len(observed) + 2.0)
        self.reference_expert_ = sorted(
            self.expert_ids_, key=lambda expert: (-self.expert_accuracy_[expert], expert)
        )[0]

        vocabulary = _event_vocabulary(self.variant)
        counts: dict[tuple[str, str, bool], Counter[str]] = defaultdict(Counter)
        totals: Counter[tuple[str, str, bool]] = Counter()
        seen: set[tuple[str, str, str]] = set()
        for observation in audits:
            question = question_by_id.get(observation.question_id)
            if question is None:
                raise ValueError(f"Unknown BFJ audit question: {observation.question_id}")
            if observation.candidate not in question.option_labels:
                raise ValueError("BFJ audit candidate is outside its question option set")
            identity = (
                observation.question_id,
                observation.auditor_id,
                observation.candidate,
            )
            if identity in seen:
                raise ValueError(f"Duplicate BFJ audit observation: {identity}")
            seen.add(identity)
            if observation.parse_error is not None:
                continue
            truth = labels.get(
                observation.question_id, candidate_label_key(observation.candidate)
            )
            if truth is None:
                raise ValueError("BFJ source labels lack a candidate correctness entry")
            relation = _relation(
                observation.auditor_id,
                observation.candidate,
                base_by_question,
                observation.question_id,
                self.variant.calibrate_self_bias,
            )
            event = _event(observation, self.variant)
            if event not in vocabulary:
                raise AssertionError("BFJ constructed an event outside its vocabulary")
            key = (observation.auditor_id, relation, bool(truth))
            counts[key][event] += 1
            totals[key] += 1
        self.event_vocabulary_ = vocabulary
        self.event_counts_ = {key: Counter(value) for key, value in counts.items()}
        self.event_totals_ = Counter(totals)
        return self

    def _event_probability(
        self,
        auditor_id: str,
        relation: str,
        truth: bool,
        event: str,
    ) -> float:
        key = (auditor_id, relation, truth)
        counts = self.event_counts_.get(key, Counter())
        total = self.event_totals_.get(key, 0)
        smoothing = self.variant.smoothing
        return float(
            (counts.get(event, 0) + smoothing)
            / (total + smoothing * len(self.event_vocabulary_))
        )

    def predict(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        audits: Sequence[AuditObservation],
    ) -> list[BFJDecision]:
        if not hasattr(self, "event_counts_"):
            raise RuntimeError("BFJ must be fitted before prediction")
        question_by_id = {row.question_id: row for row in questions}
        base_by_question: dict[str, dict[str, str | None]] = defaultdict(dict)
        supporters: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in base_predictions:
            base_by_question[row.question_id][row.expert_id] = row.answer
            if row.answer is not None:
                supporters[row.question_id][row.answer].append(row.expert_id)
        audits_by_question: dict[str, list[AuditObservation]] = defaultdict(list)
        for row in audits:
            audits_by_question[row.question_id].append(row)

        decisions: list[BFJDecision] = []
        for question_id in sorted(question_by_id):
            question = question_by_id[question_id]
            base = base_by_question.get(question_id, {})
            proposed = {
                answer for answer in base.values() if answer in set(question.option_labels)
            }
            candidates = (
                tuple(question.option_labels)
                if self.variant.open_option_set
                else tuple(label for label in question.option_labels if label in proposed)
            )
            if not candidates:
                candidates = tuple(question.option_labels)
            prior_mass: dict[str, float] = {candidate: 1.0 for candidate in candidates}
            for expert, answer in base.items():
                if answer in prior_mass:
                    prior_mass[str(answer)] += self.expert_accuracy_.get(expert, 0.5)
            prior_scale = sum(prior_mass.values())
            scores = {
                candidate: self.variant.prior_strength
                * math.log(prior_mass[candidate] / max(prior_scale, 1e-12))
                for candidate in candidates
            }
            evidence_counts = Counter()
            for observation in audits_by_question.get(question_id, ()):
                if observation.candidate not in scores:
                    continue
                if observation.parse_error is not None:
                    continue
                relation = _relation(
                    observation.auditor_id,
                    observation.candidate,
                    base_by_question,
                    question_id,
                    self.variant.calibrate_self_bias,
                )
                event = _event(observation, self.variant)
                positive = self._event_probability(
                    observation.auditor_id, relation, True, event
                )
                negative = self._event_probability(
                    observation.auditor_id, relation, False, event
                )
                log_ratio = float(
                    np.clip(
                        math.log(max(positive, 1e-12) / max(negative, 1e-12)),
                        -self.variant.max_abs_log_likelihood_ratio,
                        self.variant.max_abs_log_likelihood_ratio,
                    )
                )
                scores[observation.candidate] += (
                    self.variant.evidence_strength * log_ratio
                )
                evidence_counts[observation.candidate] += 1
            probabilities = _softmax(scores)
            reference = base.get(self.reference_expert_)
            if reference not in candidates:
                reference = sorted(
                    candidates,
                    key=lambda candidate: (-prior_mass[candidate], candidate),
                )[0]
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    -probabilities[candidate],
                    candidate != reference,
                    candidate,
                ),
            )
            chosen = ranked[0]
            runner_probability = probabilities[ranked[1]] if len(ranked) > 1 else 0.0
            margin = probabilities[chosen] - runner_probability
            fallback_reason: str | None = None
            if chosen != reference and margin + 1e-12 < self.variant.intervention_margin:
                chosen = str(reference)
                fallback_reason = "bfj_margin_below_source_fitted_threshold"
            chosen_supporters = supporters[question_id].get(chosen, [])
            selected_expert = (
                sorted(
                    chosen_supporters,
                    key=lambda expert: (-self.expert_accuracy_.get(expert, 0.0), expert),
                )[0]
                if chosen_supporters
                else None
            )
            decisions.append(
                BFJDecision(
                    question_id=question_id,
                    answer=chosen,
                    reference_answer=str(reference),
                    selected_expert_id=selected_expert,
                    candidate_scores=dict(scores),
                    candidate_probabilities=probabilities,
                    fallback_reason=fallback_reason,
                    open_set_rescue=chosen not in proposed,
                    diagnostics={
                        "method": self.variant.name,
                        "reference_expert": self.reference_expert_,
                        "proposed_candidates": sorted(proposed),
                        "scored_candidates": list(candidates),
                        "audit_count_by_candidate": dict(evidence_counts),
                        "posterior_margin": float(margin),
                        "calibrate_self_bias": self.variant.calibrate_self_bias,
                        "open_option_set": self.variant.open_option_set,
                        "uses_target_labels": False,
                    },
                )
            )
        return decisions


def uncalibrated_falsification_vote(
    question: FalsificationQuestion,
    observations: Sequence[AuditObservation],
    reference_answer: str,
) -> str:
    scores = {label: 0.0 for label in question.option_labels}
    verdict_value = {"FALSIFIED": -1.0, "INCONCLUSIVE": 0.0, "SURVIVES": 1.0}
    for row in observations:
        if row.question_id == question.question_id and row.candidate in scores:
            scores[row.candidate] += verdict_value[row.verdict] * max(row.confidence, 1) / 100.0
    return sorted(
        scores,
        key=lambda candidate: (-scores[candidate], candidate != reference_answer, candidate),
    )[0]
