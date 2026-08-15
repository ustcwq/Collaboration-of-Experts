from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .blind_falsification_jury import BasePrediction, FalsificationQuestion, candidate_label_key
from .schema import SourceTrainingLabels
from .sealed_counterfactual_parity import (
    CHALLENGE_EFFECTS,
    ISOLATED_TRACE_VIEWS,
    PARITY_ORIENTATIONS,
)


CERTIFICATE_VERDICTS = ("FALSIFIED", "INCONCLUSIVE", "SURVIVES")
CHECK_STATUSES = (
    "VALID_REFUTATION",
    "INVALID_REFUTATION",
    "VALID_SUPPORT",
    "INVALID_SUPPORT",
    "VALID_IRRELEVANT",
    "INCONCLUSIVE",
)


@dataclass(frozen=True)
class CounterexampleCertificate:
    question_id: str
    generator_id: str
    candidate: str
    verdict: str
    confidence: int
    alternative: str | None
    premise: str | None
    check: str | None
    failure: str | None
    parse_error: str | None = None
    witness_id: str | None = None
    claimed_eliminated_options: tuple[str, ...] = ()
    claimed_supported_options: tuple[str, ...] = ()
    claim_was_sealed: bool = False
    counterfactual_pair: bool = False
    challenge_rule: str | None = None
    trace_1: str | None = None
    trace_2: str | None = None
    first_differing_step: str | None = None
    sealed_valid_trace: int | None = None
    sealed_effect: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in CERTIFICATE_VERDICTS:
            raise ValueError(f"Unknown certificate verdict: {self.verdict}")
        if not 0 <= self.confidence <= 100:
            raise ValueError("Certificate confidence must be in [0, 100]")
        if set(self.claimed_eliminated_options).intersection(
            self.claimed_supported_options
        ):
            raise ValueError("A sealed witness cannot both eliminate and support an option")
        if self.claim_was_sealed:
            expected_verdict = (
                "FALSIFIED"
                if self.candidate in self.claimed_eliminated_options
                else "SURVIVES"
                if self.candidate in self.claimed_supported_options
                else "INCONCLUSIVE"
            )
            if self.verdict != expected_verdict:
                raise ValueError("Candidate verdict differs from the sealed effect set")
            if self.witness_id is None:
                raise ValueError("A sealed effect claim requires a witness ID")
        if self.counterfactual_pair:
            if not self.claim_was_sealed or self.witness_id is None:
                raise ValueError("A counterfactual pair must be a sealed witness")
            signed_effect_count = len(self.claimed_eliminated_options) + len(
                self.claimed_supported_options
            )
            if signed_effect_count > 1:
                raise ValueError("A counterfactual challenge may affect at most one option")
            present_claim = self.sealed_valid_trace is not None
            if present_claim != (self.sealed_effect is not None):
                raise ValueError("Counterfactual trace and effect claims must be paired")
            if self.sealed_valid_trace not in {None, 1, 2}:
                raise ValueError("A sealed valid trace must be 1, 2, or None")
            if self.sealed_effect not in {None, *CHALLENGE_EFFECTS}:
                raise ValueError("Unknown sealed counterfactual effect")
            if present_claim and signed_effect_count != 1:
                raise ValueError("A non-abstaining challenge needs one signed option effect")
            if not present_claim and signed_effect_count != 0:
                raise ValueError("An abstaining challenge cannot claim an option effect")
            if self.sealed_effect == "ELIMINATES" and len(
                self.claimed_eliminated_options
            ) != 1:
                raise ValueError("ELIMINATES must bind one eliminated option")
            if self.sealed_effect == "SUPPORTS" and len(
                self.claimed_supported_options
            ) != 1:
                raise ValueError("SUPPORTS must bind one supported option")
            if self.parse_error is None:
                if any(
                    value is None
                    for value in (
                        self.challenge_rule,
                        self.trace_1,
                        self.trace_2,
                        self.first_differing_step,
                    )
                ):
                    raise ValueError("A parsed counterfactual pair lacks visible content")
                if present_claim and self.trace_1 == self.trace_2:
                    raise ValueError("A counterfactual pair needs distinct traces")

    @property
    def certificate_id(self) -> str:
        return f"{self.question_id}::{self.generator_id}::{self.candidate}"


@dataclass(frozen=True)
class CertificateCheck:
    certificate_id: str
    question_id: str
    generator_id: str
    checker_id: str
    candidate: str
    status: str
    confidence: int
    independent_answer: str | None
    first_flaw: str | None
    parse_error: str | None = None
    logic_status: str | None = None
    eliminated_options: tuple[str, ...] = ()
    supported_options: tuple[str, ...] = ()
    target_was_hidden: bool = False
    counterfactual_pair: bool = False
    orientation: str = "single"
    presented_valid_trace: int | None = None
    canonical_valid_trace: int | None = None
    reconstructed_effect: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"Unknown certificate-check status: {self.status}")
        if not 0 <= self.confidence <= 100:
            raise ValueError("Certificate-check confidence must be in [0, 100]")
        if self.generator_id == self.checker_id:
            raise ValueError("A certificate generator may not check its own certificate")
        if self.logic_status not in {None, "VALID", "INVALID", "INCONCLUSIVE"}:
            raise ValueError(f"Unknown certificate logic status: {self.logic_status}")
        if set(self.eliminated_options).intersection(self.supported_options):
            raise ValueError("A reconstructed option cannot be both eliminated and supported")
        if self.counterfactual_pair:
            if self.orientation not in {
                *PARITY_ORIENTATIONS,
                *ISOLATED_TRACE_VIEWS,
            }:
                raise ValueError("Unknown counterfactual audit view")
            if self.presented_valid_trace not in {None, 1, 2}:
                raise ValueError("Presented valid trace must be 1, 2, or None")
            if self.canonical_valid_trace not in {None, 1, 2}:
                raise ValueError("Canonical valid trace must be 1, 2, or None")
            if self.reconstructed_effect not in {None, *CHALLENGE_EFFECTS}:
                raise ValueError("Unknown reconstructed counterfactual effect")
            signed_effect_count = len(self.eliminated_options) + len(
                self.supported_options
            )
            if self.logic_status == "VALID":
                required_trace_fields = (
                    self.presented_valid_trace is not None
                    and self.canonical_valid_trace is not None
                    if self.orientation in PARITY_ORIENTATIONS
                    else self.presented_valid_trace is None
                    and self.canonical_valid_trace
                    == 1 + ISOLATED_TRACE_VIEWS.index(self.orientation)
                )
                if (
                    not required_trace_fields
                    or self.reconstructed_effect is None
                    or signed_effect_count != 1
                ):
                    raise ValueError("A valid counterfactual audit needs a complete reconstruction")
            elif any(
                value is not None
                for value in (
                    self.presented_valid_trace,
                    self.canonical_valid_trace,
                    self.reconstructed_effect,
                )
            ) or signed_effect_count:
                raise ValueError("A non-valid counterfactual audit cannot reconstruct an effect")


@dataclass(frozen=True)
class C3Variant:
    name: str
    regularization_c: float = 1.0
    intervention_margin: float = 0.0
    open_option_set: bool = True
    use_certificates: bool = True
    use_checks: bool = True
    use_generator_answer_dependence: bool = True
    use_checker_answer_dependence: bool = True
    use_generator_checker_pair_effects: bool = True
    use_sealed_set_agreement: bool = True
    use_counterfactual_parity: bool = True

    def __post_init__(self) -> None:
        if self.regularization_c <= 0.0:
            raise ValueError("C3 regularization C must be positive")
        if not 0.0 <= self.intervention_margin <= 1.0:
            raise ValueError("C3 intervention margin must be in [0, 1]")
        if self.use_checks and not self.use_certificates:
            raise ValueError("C3 certificate checks require certificate features")


@dataclass(frozen=True)
class C3Decision:
    question_id: str
    answer: str
    reference_answer: str
    selected_expert_id: str | None
    candidate_logits: Mapping[str, float]
    candidate_probabilities: Mapping[str, float]
    fallback_reason: str | None
    open_set_rescue: bool
    diagnostics: Mapping[str, Any]


def _single_line_field(name: str, choices: str | None = None) -> re.Pattern[str]:
    value = choices if choices is not None else r"(\S.*?)"
    return re.compile(
        rf"^\s*{re.escape(name)}\s*:\s*{value}\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )


_CERTIFICATE_PATTERNS = {
    "verdict": _single_line_field("VERDICT", r"(FALSIFIED|INCONCLUSIVE|SURVIVES)"),
    "confidence": _single_line_field("CONFIDENCE", r"(\d{1,3})"),
    "alternative": _single_line_field("ALTERNATIVE", r"\(?([A-Z]|NONE)\)?"),
    "premise": _single_line_field("PREMISE"),
    "check": _single_line_field("CHECK"),
    "failure": _single_line_field("FAILURE"),
}

_CHECK_PATTERNS = {
    "status": _single_line_field(
        "CERTIFICATE_STATUS",
        r"(VALID_REFUTATION|INVALID_REFUTATION|VALID_SUPPORT|INVALID_SUPPORT|INCONCLUSIVE)",
    ),
    "confidence": _single_line_field("CONFIDENCE", r"(\d{1,3})"),
    "independent_answer": _single_line_field(
        "INDEPENDENT_ANSWER", r"\(?([A-Z]|NONE)\)?"
    ),
    "first_flaw": _single_line_field("FIRST_FLAW"),
}

_CERTIFICATE_V2_PATTERNS = {
    "verdict": _single_line_field("VERDICT", r"(FALSIFIED|INCONCLUSIVE|SURVIVES)"),
    "confidence": _single_line_field("CONFIDENCE", r"(\d{1,3})"),
    "alternative": _single_line_field("ALTERNATIVE"),
    "premise": _single_line_field("PREMISE"),
    "check": _single_line_field("CHECK"),
    "failure": _single_line_field("FAILURE"),
}

_TARGET_BLIND_CHECK_PATTERNS = {
    "logic_status": _single_line_field(
        "LOGIC_STATUS", r"(VALID|INVALID|INCONCLUSIVE)"
    ),
    "confidence": _single_line_field("CONFIDENCE", r"(\d{1,3})"),
    "eliminated_options": _single_line_field("ELIMINATED_OPTIONS"),
    "supported_options": _single_line_field("SUPPORTED_OPTIONS"),
    "first_invalid_step": _single_line_field("FIRST_INVALID_STEP"),
}

_SEALED_EFFECT_WITNESS_PATTERNS = {
    "invariant": _single_line_field("INVARIANT"),
    "derivation": _single_line_field("DERIVATION"),
    "boundary": _single_line_field("BOUNDARY"),
    "eliminated_options": _single_line_field("ELIMINATED_OPTIONS"),
    "supported_options": _single_line_field("SUPPORTED_OPTIONS"),
    "confidence": _single_line_field("CONFIDENCE", r"(\d{1,3})"),
}


def _unique_matches(
    text: str, patterns: Mapping[str, re.Pattern[str]]
) -> tuple[dict[str, str], str | None]:
    values: dict[str, str] = {}
    for name, pattern in patterns.items():
        matches = pattern.findall(text)
        if not matches:
            return {}, "missing_required_field"
        if len(matches) != 1:
            return {}, "duplicate_required_field"
        values[name] = str(matches[0]).strip()
    return values, None


def build_certificate_prompt(question: FalsificationQuestion, candidate: str) -> str:
    if candidate not in question.option_labels:
        raise ValueError(f"Unknown certificate candidate: {candidate}")
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    candidate_text = question.options[question.option_labels.index(candidate)]
    return (
        "Act as an independent counterexample-certificate generator. No model identity, "
        "answer author, vote count, popularity, source score, or gold answer is available. "
        "Solve the question yourself, then inspect only the named candidate. Search for one "
        "minimal, checkable contradiction. A plausible objection is not enough. Mark "
        "FALSIFIED only when PREMISE and CHECK establish a fatal FAILURE. Mark SURVIVES only "
        "when CHECK positively verifies the candidate and FAILURE is NONE. Otherwise mark "
        "INCONCLUSIVE.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{options}\n\n"
        f"Candidate under inspection: ({candidate}) {candidate_text}\n\n"
        "Return exactly six single-line fields. Use no XML, bullets, or surrounding prose.\n"
        "VERDICT: FALSIFIED|INCONCLUSIVE|SURVIVES\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "ALTERNATIVE: option label or NONE\n"
        "PREMISE: one atomic fact, rule, or assumption used by the check\n"
        "CHECK: one calculation, implication, or concrete counterexample\n"
        "FAILURE: exact fatal failure, or NONE"
    )


def parse_certificate_output(
    text: str, option_labels: Sequence[str]
) -> tuple[str, int, str | None, str | None, str | None, str | None, str | None]:
    values, error = _unique_matches(text, _CERTIFICATE_PATTERNS)
    if error is not None:
        return "INCONCLUSIVE", 0, None, None, None, None, error
    confidence = int(values["confidence"])
    if not 0 <= confidence <= 100:
        return "INCONCLUSIVE", 0, None, None, None, None, "confidence_out_of_range"
    labels = {str(label).upper() for label in option_labels}
    alternative = values["alternative"].upper()
    if alternative != "NONE" and alternative not in labels:
        return "INCONCLUSIVE", 0, None, None, None, None, "alternative_outside_option_set"
    verdict = values["verdict"].upper()
    failure = values["failure"].strip()
    failure_is_none = failure.upper() == "NONE"
    if verdict == "FALSIFIED" and failure_is_none:
        return "INCONCLUSIVE", 0, None, None, None, None, "falsified_without_failure"
    if verdict == "SURVIVES" and not failure_is_none:
        return "INCONCLUSIVE", 0, None, None, None, None, "survives_with_failure"
    return (
        verdict,
        confidence,
        None if alternative == "NONE" else alternative,
        values["premise"],
        values["check"],
        failure,
        None,
    )


def build_certificate_prompt_v2(
    question: FalsificationQuestion, candidate: str
) -> str:
    if candidate not in question.option_labels:
        raise ValueError(f"Unknown certificate candidate: {candidate}")
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    candidate_text = question.options[question.option_labels.index(candidate)]
    return (
        "Act as an independent two-sided certificate generator. No model identity, "
        "answer author, vote count, popularity, source score, or gold answer is available. "
        "Solve the question yourself. For the named candidate, first attempt a positive "
        "derivation and then attempt a concrete counterexample. Keep only the direction "
        "with the stronger check. Do not assume that every inspected candidate is false. "
        "Use FALSIFIED only for a fatal contradiction that survives your own stress test, "
        "SURVIVES only when the check positively derives the exact candidate, and "
        "INCONCLUSIVE otherwise. PREMISE, CHECK, and FAILURE must be understandable later "
        "without mentioning the candidate label or saying 'this option'.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{options}\n\n"
        f"Candidate under inspection: ({candidate}) {candidate_text}\n\n"
        "Return exactly six single-line fields. Use no XML, bullets, or surrounding prose. "
        "ALTERNATIVE must contain one label only, such as B, or NONE; never append option "
        "text.\n"
        "VERDICT: FALSIFIED|INCONCLUSIVE|SURVIVES\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "ALTERNATIVE: one option label or NONE\n"
        "PREMISE: one atomic fact, rule, or assumption used by the check\n"
        "CHECK: one calculation, implication, or concrete counterexample\n"
        "FAILURE: exact fatal failure, or NONE"
    )


def _leading_option_label(value: str, option_labels: Sequence[str]) -> str | None:
    stripped = value.strip().upper()
    if stripped == "NONE":
        return None
    match = re.match(r"^\(?([A-Z])\)?(?:\s+.*)?$", stripped)
    if match is None:
        raise ValueError("malformed_option_label")
    label = match.group(1)
    if label not in {str(item).upper() for item in option_labels}:
        raise ValueError("option_label_outside_set")
    return label


def parse_certificate_output_v2(
    text: str, option_labels: Sequence[str]
) -> tuple[str, int, str | None, str | None, str | None, str | None, str | None]:
    values, error = _unique_matches(text, _CERTIFICATE_V2_PATTERNS)
    if error is not None:
        return "INCONCLUSIVE", 0, None, None, None, None, error
    confidence = int(values["confidence"])
    if not 0 <= confidence <= 100:
        return "INCONCLUSIVE", 0, None, None, None, None, "confidence_out_of_range"
    try:
        alternative = _leading_option_label(values["alternative"], option_labels)
    except ValueError as exc:
        return "INCONCLUSIVE", 0, None, None, None, None, str(exc)
    verdict = values["verdict"].upper()
    failure = values["failure"].strip()
    failure_is_none = failure.upper() == "NONE"
    if verdict == "FALSIFIED" and failure_is_none:
        return "INCONCLUSIVE", 0, None, None, None, None, "falsified_without_failure"
    if verdict == "SURVIVES" and not failure_is_none:
        return "INCONCLUSIVE", 0, None, None, None, None, "survives_with_failure"
    return (
        verdict,
        confidence,
        alternative,
        values["premise"],
        values["check"],
        failure,
        None,
    )


def build_certificate_check_prompt(
    question: FalsificationQuestion, certificate: CounterexampleCertificate
) -> str:
    if certificate.question_id != question.question_id:
        raise ValueError("Certificate and question IDs do not match")
    if certificate.parse_error is not None:
        raise ValueError("Unparsed certificates may not be cross-examined")
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    candidate_text = question.options[question.option_labels.index(certificate.candidate)]
    return (
        "Act as a cold, independent certificate checker. The certificate author, model "
        "identity, answer votes, popularity, other certificates, and gold answer are hidden. "
        "First solve the problem independently. Then check whether every premise is relevant "
        "and whether the stated check logically entails the claimed support or refutation. "
        "Do not reward confident wording. VALID_REFUTATION requires a concrete fatal flaw; "
        "VALID_SUPPORT requires a positive verification; use INVALID_* when the certificate's "
        "logic fails, and INCONCLUSIVE when it cannot be decided.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{options}\n\n"
        f"Candidate: ({certificate.candidate}) {candidate_text}\n"
        f"Claimed verdict: {certificate.verdict}\n"
        f"Premise: {certificate.premise}\n"
        f"Check: {certificate.check}\n"
        f"Claimed failure: {certificate.failure}\n\n"
        "Return exactly four single-line fields. Use no XML, bullets, or surrounding prose.\n"
        "CERTIFICATE_STATUS: VALID_REFUTATION|INVALID_REFUTATION|VALID_SUPPORT|INVALID_SUPPORT|INCONCLUSIVE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "INDEPENDENT_ANSWER: option label or NONE\n"
        "FIRST_FLAW: first invalid inference, or NONE"
    )


def parse_certificate_check_output(
    text: str, option_labels: Sequence[str]
) -> tuple[str, int, str | None, str | None, str | None]:
    values, error = _unique_matches(text, _CHECK_PATTERNS)
    if error is not None:
        return "INCONCLUSIVE", 0, None, None, error
    confidence = int(values["confidence"])
    if not 0 <= confidence <= 100:
        return "INCONCLUSIVE", 0, None, None, "confidence_out_of_range"
    labels = {str(label).upper() for label in option_labels}
    independent = values["independent_answer"].upper()
    if independent != "NONE" and independent not in labels:
        return "INCONCLUSIVE", 0, None, None, "answer_outside_option_set"
    return (
        values["status"].upper(),
        confidence,
        None if independent == "NONE" else independent,
        values["first_flaw"],
        None,
    )


def build_target_blind_check_prompt_v2(
    question: FalsificationQuestion, certificate: CounterexampleCertificate
) -> str:
    if certificate.question_id != question.question_id:
        raise ValueError("Certificate and question IDs do not match")
    if certificate.parse_error is not None:
        raise ValueError("Unparsed certificates may not be reconstructed")
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    return (
        "Act as an independent target-blind certificate reconstructor. The certificate's "
        "author, intended option, claimed verdict, alternative answer, answer votes, "
        "popularity, other certificates, and gold answer are hidden. Check the supplied "
        "premise and calculation without guessing their intended target. If the reasoning "
        "is valid, apply its consequence independently to every listed option and report "
        "all options it eliminates and all options it positively supports. A merely "
        "plausible narrative is not valid evidence. INVALID requires the first incorrect "
        "premise, computation, or inference. INCONCLUSIVE means validity cannot be "
        "determined.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{options}\n\n"
        "Target-blind certificate content:\n"
        f"Premise: {certificate.premise}\n"
        f"Check: {certificate.check}\n"
        f"Stated failure or boundary: {certificate.failure}\n\n"
        "Return exactly five single-line fields. Use no XML, bullets, or surrounding prose. "
        "Option sets must contain labels only, separated by commas, or NONE. Do not copy "
        "option text. For VALID, FIRST_INVALID_STEP must be NONE. For INVALID it must name "
        "the first error.\n"
        "LOGIC_STATUS: VALID|INVALID|INCONCLUSIVE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "ELIMINATED_OPTIONS: comma-separated option labels or NONE\n"
        "SUPPORTED_OPTIONS: comma-separated option labels or NONE\n"
        "FIRST_INVALID_STEP: first invalid premise/computation/inference, or NONE"
    )


def _parse_option_set(value: str, option_labels: Sequence[str]) -> tuple[str, ...]:
    stripped = value.strip().upper()
    if stripped == "NONE":
        return ()
    labels = {str(item).upper() for item in option_labels}
    parsed: list[str] = []
    for token in stripped.split(","):
        match = re.fullmatch(r"\s*\(?([A-Z])\)?\s*", token)
        if match is None:
            raise ValueError("malformed_option_set")
        label = match.group(1)
        if label not in labels:
            raise ValueError("option_set_outside_options")
        if label in parsed:
            raise ValueError("duplicate_option_in_set")
        parsed.append(label)
    return tuple(sorted(parsed))


def build_sealed_effect_witness_prompt_v3(question: FalsificationQuestion) -> str:
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    return (
        "Act as an independent set-valued reasoning witness. No model identity, prior "
        "answer, answer author, vote count, popularity, source score, or gold answer is "
        "available. Do not audit a named candidate. Solve the question once, then isolate "
        "one atomic discriminator: a fact, constraint, calculation, or counterexample whose "
        "consequence can be applied uniformly to every option. State the applicability "
        "boundary that prevents overgeneralizing the discriminator.\n\n"
        "Put an option in ELIMINATED_OPTIONS only when the stated derivation contradicts "
        "that exact option. Put an option in SUPPORTED_OPTIONS only when the derivation "
        "positively entails it. A preference, plausibility judgment, or unsupported final "
        "answer has no effect. Use NONE for both sets when no rigorous effect is available. "
        "Because exactly one listed option is correct, never eliminate every option or "
        "support every option.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{options}\n\n"
        "Return exactly six single-line fields. Use no XML, bullets, or surrounding prose. "
        "Option sets contain labels only, separated by commas, or NONE.\n"
        "INVARIANT: one atomic fact, rule, constraint, or assumption\n"
        "DERIVATION: one checkable calculation, implication, or concrete counterexample\n"
        "BOUNDARY: when the derivation applies and the first condition that would invalidate it\n"
        "ELIMINATED_OPTIONS: comma-separated option labels or NONE\n"
        "SUPPORTED_OPTIONS: comma-separated option labels or NONE\n"
        "CONFIDENCE: integer from 0 to 100"
    )


def parse_sealed_effect_witness_output_v3(
    text: str, option_labels: Sequence[str]
) -> tuple[
    int,
    str | None,
    str | None,
    str | None,
    tuple[str, ...],
    tuple[str, ...],
    str | None,
]:
    values, error = _unique_matches(text, _SEALED_EFFECT_WITNESS_PATTERNS)
    if error is not None:
        return 0, None, None, None, (), (), error
    confidence = int(values["confidence"])
    if not 0 <= confidence <= 100:
        return 0, None, None, None, (), (), "confidence_out_of_range"
    try:
        eliminated = _parse_option_set(values["eliminated_options"], option_labels)
        supported = _parse_option_set(values["supported_options"], option_labels)
    except ValueError as exc:
        return 0, None, None, None, (), (), str(exc)
    if set(eliminated).intersection(supported):
        return 0, None, None, None, (), (), "contradictory_option_sets"
    labels = {str(label).upper() for label in option_labels}
    if set(eliminated) == labels:
        return 0, None, None, None, (), (), "all_options_eliminated"
    if set(supported) == labels:
        return 0, None, None, None, (), (), "all_options_supported"
    if values["invariant"].strip().upper() == "NONE":
        return 0, None, None, None, (), (), "missing_invariant"
    if values["derivation"].strip().upper() == "NONE":
        return 0, None, None, None, (), (), "missing_derivation"
    return (
        confidence,
        values["invariant"],
        values["derivation"],
        values["boundary"],
        eliminated,
        supported,
        None,
    )


def sealed_witness_candidate_fields(
    candidate: str,
    option_labels: Sequence[str],
    eliminated_options: Sequence[str],
    supported_options: Sequence[str],
) -> tuple[str, str | None]:
    labels = tuple(str(label).upper() for label in option_labels)
    candidate = str(candidate).upper()
    if candidate not in labels:
        raise ValueError("Sealed witness candidate is outside its option set")
    eliminated = {str(label).upper() for label in eliminated_options}
    supported = {str(label).upper() for label in supported_options}
    if not eliminated.issubset(labels) or not supported.issubset(labels):
        raise ValueError("Sealed witness effect is outside its option set")
    if eliminated.intersection(supported):
        raise ValueError("Sealed witness effect sets overlap")
    verdict = (
        "FALSIFIED"
        if candidate in eliminated
        else "SURVIVES"
        if candidate in supported
        else "INCONCLUSIVE"
    )
    survivors = [label for label in labels if label not in eliminated]
    alternative = (
        sorted(supported)[0]
        if len(supported) == 1
        else survivors[0]
        if len(survivors) == 1
        else None
    )
    return verdict, alternative


def build_sealed_effect_reconstruction_prompt_v3(
    question: FalsificationQuestion, certificate: CounterexampleCertificate
) -> str:
    if certificate.question_id != question.question_id:
        raise ValueError("Witness and question IDs do not match")
    if certificate.parse_error is not None:
        raise ValueError("Unparsed witnesses may not be reconstructed")
    if not certificate.claim_was_sealed:
        raise ValueError("The v3 reconstruction protocol requires a sealed effect claim")
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(question.option_labels, question.options, strict=True)
    )
    return (
        "Act as an independent blind effect-set reconstructor. The witness author, the "
        "author's prior answer, claimed eliminated and supported sets, confidence, answer "
        "votes, popularity, other witnesses, and gold answer are hidden. Validate only the "
        "visible invariant, derivation, and applicability boundary. If valid, apply the same "
        "consequence independently to every option and reconstruct every option it eliminates "
        "and every option it positively supports. A plausible narrative or preferred answer "
        "is not evidence. INVALID requires the first false premise, calculation, inference, "
        "or boundary violation. INCONCLUSIVE means validity cannot be determined.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{options}\n\n"
        "Claim-blind witness content:\n"
        f"Invariant: {certificate.premise}\n"
        f"Derivation: {certificate.check}\n"
        f"Applicability boundary: {certificate.failure}\n\n"
        "Return exactly five single-line fields. Use no XML, bullets, or surrounding prose. "
        "Option sets contain labels only, separated by commas, or NONE. For VALID, "
        "FIRST_INVALID_STEP must be NONE. For INVALID it must name the first error.\n"
        "LOGIC_STATUS: VALID|INVALID|INCONCLUSIVE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "ELIMINATED_OPTIONS: comma-separated option labels or NONE\n"
        "SUPPORTED_OPTIONS: comma-separated option labels or NONE\n"
        "FIRST_INVALID_STEP: first invalid premise/computation/inference, or NONE"
    )


def parse_target_blind_check_output_v2(
    text: str, option_labels: Sequence[str]
) -> tuple[
    str,
    int,
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    str | None,
]:
    values, error = _unique_matches(text, _TARGET_BLIND_CHECK_PATTERNS)
    if error is not None:
        return "INCONCLUSIVE", 0, (), (), None, error
    confidence = int(values["confidence"])
    if not 0 <= confidence <= 100:
        return "INCONCLUSIVE", 0, (), (), None, "confidence_out_of_range"
    try:
        eliminated = _parse_option_set(values["eliminated_options"], option_labels)
        supported = _parse_option_set(values["supported_options"], option_labels)
    except ValueError as exc:
        return "INCONCLUSIVE", 0, (), (), None, str(exc)
    if set(eliminated).intersection(supported):
        return "INCONCLUSIVE", 0, (), (), None, "contradictory_option_sets"
    logic_status = values["logic_status"].upper()
    first_invalid_step = values["first_invalid_step"].strip()
    flaw_is_none = first_invalid_step.upper() == "NONE"
    if logic_status == "VALID" and not flaw_is_none:
        return "INCONCLUSIVE", 0, (), (), None, "valid_with_invalid_step"
    if logic_status == "INVALID" and flaw_is_none:
        return "INCONCLUSIVE", 0, (), (), None, "invalid_without_invalid_step"
    return logic_status, confidence, eliminated, supported, first_invalid_step, None


def reconstructed_check_status(
    certificate: CounterexampleCertificate,
    logic_status: str,
    eliminated_options: Sequence[str],
    supported_options: Sequence[str],
) -> str:
    if logic_status == "INCONCLUSIVE":
        return "INCONCLUSIVE"
    if logic_status == "INVALID":
        if certificate.verdict == "FALSIFIED":
            return "INVALID_REFUTATION"
        if certificate.verdict == "SURVIVES":
            return "INVALID_SUPPORT"
        return "INCONCLUSIVE"
    if certificate.candidate in eliminated_options:
        return "VALID_REFUTATION"
    if certificate.candidate in supported_options:
        return "VALID_SUPPORT"
    return "VALID_IRRELEVANT"


def _softmax(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    exponentials = {
        key: math.exp(float(np.clip(value - maximum, -60.0, 0.0)))
        for key, value in values.items()
    }
    total = sum(exponentials.values())
    return {key: value / max(total, 1e-12) for key, value in exponentials.items()}


class CrossExaminedCertificateCourt:
    """Source-trained candidate selection from blind certificates and cross-checks."""

    def __init__(self, variant: C3Variant, seed: int = 20260815) -> None:
        self.variant = variant
        self.seed = int(seed)

    @staticmethod
    def _base_by_question(
        base_predictions: Sequence[BasePrediction],
    ) -> dict[str, dict[str, str | None]]:
        result: dict[str, dict[str, str | None]] = defaultdict(dict)
        for row in base_predictions:
            if row.expert_id in result[row.question_id]:
                raise ValueError("Duplicate C3 base prediction")
            result[row.question_id][row.expert_id] = row.answer
        return result

    def _candidate_features(
        self,
        question: FalsificationQuestion,
        candidate: str,
        base_by_question: Mapping[str, Mapping[str, str | None]],
        certificates_by_key: Mapping[tuple[str, str], Sequence[CounterexampleCertificate]],
        checks_by_certificate: Mapping[str, Sequence[CertificateCheck]],
    ) -> dict[str, float]:
        base = base_by_question.get(question.question_id, {})
        valid_votes = [answer for answer in base.values() if answer in question.option_labels]
        features: dict[str, float] = {
            "base::vote_fraction": valid_votes.count(candidate) / max(1, len(valid_votes)),
            "base::proposed": float(candidate in valid_votes),
        }
        for expert in self.expert_ids_:
            answer = base.get(expert)
            features[f"base_vote::{expert}"] = float(answer == candidate)
            if answer == candidate:
                features["base::source_weighted_vote"] = features.get(
                    "base::source_weighted_vote", 0.0
                ) + self.expert_accuracy_[expert]

        if not self.variant.use_certificates:
            return features
        certificates = certificates_by_key.get((question.question_id, candidate), ())
        for certificate in certificates:
            if certificate.parse_error is not None:
                continue
            generator_relation = (
                "same" if base.get(certificate.generator_id) == candidate else "different"
            )
            claimed_relation = (
                "eliminated"
                if candidate in certificate.claimed_eliminated_options
                else "supported"
                if candidate in certificate.claimed_supported_options
                else "irrelevant"
            )
            confidence = certificate.confidence / 100.0
            features[f"certificate::verdict::{certificate.verdict}"] = features.get(
                f"certificate::verdict::{certificate.verdict}", 0.0
            ) + 1.0
            features[f"certificate::generator::{certificate.generator_id}::verdict::{certificate.verdict}"] = 1.0
            features[f"certificate::generator::{certificate.generator_id}::confidence"] = confidence
            if certificate.alternative == candidate:
                features["certificate::alternative_support"] = features.get(
                    "certificate::alternative_support", 0.0
                ) + 1.0
            if certificate.claim_was_sealed:
                features["certificate::sealed_effect_claim"] = features.get(
                    "certificate::sealed_effect_claim", 0.0
                ) + 1.0
                features[
                    f"certificate::sealed_relation::{claimed_relation}"
                ] = features.get(
                    f"certificate::sealed_relation::{claimed_relation}", 0.0
                ) + 1.0
                features["certificate::claimed_elimination_size"] = features.get(
                    "certificate::claimed_elimination_size", 0.0
                ) + len(certificate.claimed_eliminated_options)
                features["certificate::claimed_support_size"] = features.get(
                    "certificate::claimed_support_size", 0.0
                ) + len(certificate.claimed_supported_options)
            if certificate.counterfactual_pair:
                features["certificate::counterfactual_pair"] = features.get(
                    "certificate::counterfactual_pair", 0.0
                ) + 1.0
                effect_name = certificate.sealed_effect or "abstain"
                features[
                    f"certificate::counterfactual_effect::{effect_name}"
                ] = features.get(
                    f"certificate::counterfactual_effect::{effect_name}", 0.0
                ) + 1.0
            if self.variant.use_generator_answer_dependence:
                features[f"certificate::generator_relation::{generator_relation}::verdict::{certificate.verdict}"] = features.get(
                    f"certificate::generator_relation::{generator_relation}::verdict::{certificate.verdict}",
                    0.0,
                ) + 1.0
                features[f"certificate::generator::{certificate.generator_id}::relation::{generator_relation}"] = 1.0

            if not self.variant.use_checks:
                continue
            certificate_checks = tuple(
                checks_by_certificate.get(certificate.certificate_id, ())
            )
            if certificate.counterfactual_pair and self.variant.use_counterfactual_parity:
                checks_by_checker: dict[str, list[CertificateCheck]] = defaultdict(list)
                for member in certificate_checks:
                    if member.orientation in ISOLATED_TRACE_VIEWS:
                        checks_by_checker[member.checker_id].append(member)
                for checker_id, members in checks_by_checker.items():
                    by_view = {member.orientation: member for member in members}
                    complete = (
                        set(by_view) == set(ISOLATED_TRACE_VIEWS)
                        and all(member.parse_error is None for member in by_view.values())
                    )
                    features[
                        f"check::isolated_pair_complete::{str(complete).lower()}"
                    ] = features.get(
                        f"check::isolated_pair_complete::{str(complete).lower()}",
                        0.0,
                    ) + 1.0
                    if not complete:
                        continue
                    valid = [
                        member
                        for member in by_view.values()
                        if member.logic_status == "VALID"
                    ]
                    invalid = [
                        member
                        for member in by_view.values()
                        if member.logic_status == "INVALID"
                    ]
                    one_valid_one_invalid = len(valid) == 1 and len(invalid) == 1
                    features[
                        "check::isolated_one_valid_one_invalid::"
                        f"{str(one_valid_one_invalid).lower()}"
                    ] = features.get(
                        "check::isolated_one_valid_one_invalid::"
                        f"{str(one_valid_one_invalid).lower()}",
                        0.0,
                    ) + 1.0
                    features[
                        f"check::checker::{checker_id}::isolated_one_valid_one_invalid::"
                        f"{str(one_valid_one_invalid).lower()}"
                    ] = 1.0
                    if not one_valid_one_invalid:
                        continue
                    valid_check = valid[0]
                    valid_relation_matches = (
                        certificate.verdict == "FALSIFIED"
                        and candidate in valid_check.eliminated_options
                    ) or (
                        certificate.verdict == "SURVIVES"
                        and candidate in valid_check.supported_options
                    ) or (
                        certificate.verdict == "INCONCLUSIVE"
                        and candidate not in valid_check.eliminated_options
                        and candidate not in valid_check.supported_options
                    )
                    triple_match = (
                        valid_check.canonical_valid_trace
                        == certificate.sealed_valid_trace
                        and valid_check.reconstructed_effect
                        == certificate.sealed_effect
                        and valid_relation_matches
                    )
                    if self.variant.use_sealed_set_agreement:
                        features[
                            "check::isolated_pair_sealed_triple_match::"
                            f"{str(triple_match).lower()}"
                        ] = features.get(
                            "check::isolated_pair_sealed_triple_match::"
                            f"{str(triple_match).lower()}",
                            0.0,
                        ) + 1.0
                        features[
                            f"check::checker::{checker_id}::"
                            "isolated_pair_sealed_triple_match::"
                            f"{str(triple_match).lower()}"
                        ] = 1.0
            for check in certificate_checks:
                if check.parse_error is not None:
                    continue
                if (
                    check.counterfactual_pair
                    and not self.variant.use_counterfactual_parity
                    and check.orientation in {"mirrored", "trace_2"}
                ):
                    continue
                checker_relation = (
                    "same" if base.get(check.checker_id) == candidate else "different"
                )
                features[f"check::status::{check.status}"] = features.get(
                    f"check::status::{check.status}", 0.0
                ) + 1.0
                features[f"check::checker::{check.checker_id}::status::{check.status}"] = features.get(
                    f"check::checker::{check.checker_id}::status::{check.status}", 0.0
                ) + 1.0
                features[f"check::checker::{check.checker_id}::confidence_sum"] = features.get(
                    f"check::checker::{check.checker_id}::confidence_sum", 0.0
                ) + check.confidence / 100.0
                if check.independent_answer == candidate:
                    features["check::independent_answer_support"] = features.get(
                        "check::independent_answer_support", 0.0
                    ) + 1.0
                if check.target_was_hidden:
                    features["check::target_blind"] = features.get(
                        "check::target_blind", 0.0
                    ) + 1.0
                    features[f"check::logic_status::{check.logic_status}"] = features.get(
                        f"check::logic_status::{check.logic_status}", 0.0
                    ) + 1.0
                    reconstructed_relation = (
                        "eliminated"
                        if candidate in check.eliminated_options
                        else "supported"
                        if candidate in check.supported_options
                        else "irrelevant"
                    )
                    features[
                        f"check::target_reconstruction::{reconstructed_relation}"
                    ] = features.get(
                        f"check::target_reconstruction::{reconstructed_relation}", 0.0
                    ) + 1.0
                    features["check::off_target_eliminations"] = features.get(
                        "check::off_target_eliminations", 0.0
                    ) + sum(
                        option != candidate for option in check.eliminated_options
                    )
                    features["check::off_target_supports"] = features.get(
                        "check::off_target_supports", 0.0
                    ) + sum(
                        option != candidate for option in check.supported_options
                    )
                    claim_direction_match = (
                        certificate.verdict == "FALSIFIED"
                        and candidate in check.eliminated_options
                    ) or (
                        certificate.verdict == "SURVIVES"
                        and candidate in check.supported_options
                    ) or (
                        certificate.verdict == "INCONCLUSIVE"
                        and candidate not in check.eliminated_options
                        and candidate not in check.supported_options
                    )
                    if (
                        not certificate.claim_was_sealed
                        or self.variant.use_sealed_set_agreement
                    ):
                        features[
                            f"check::sealed_claim_match::{str(claim_direction_match).lower()}"
                        ] = features.get(
                            f"check::sealed_claim_match::{str(claim_direction_match).lower()}",
                            0.0,
                        ) + 1.0
                    if (
                        certificate.claim_was_sealed
                        and self.variant.use_sealed_set_agreement
                    ):
                        claimed_eliminated = set(
                            certificate.claimed_eliminated_options
                        )
                        claimed_supported = set(
                            certificate.claimed_supported_options
                        )
                        reconstructed_eliminated = set(check.eliminated_options)
                        reconstructed_supported = set(check.supported_options)
                        exact_set_match = (
                            check.logic_status == "VALID"
                            and claimed_eliminated == reconstructed_eliminated
                            and claimed_supported == reconstructed_supported
                        )
                        signed_claim = {
                            ("eliminate", option)
                            for option in claimed_eliminated
                        }.union(
                            ("support", option) for option in claimed_supported
                        )
                        signed_reconstruction = {
                            ("eliminate", option)
                            for option in reconstructed_eliminated
                        }.union(
                            ("support", option) for option in reconstructed_supported
                        )
                        union = signed_claim.union(signed_reconstruction)
                        jaccard = (
                            len(signed_claim.intersection(signed_reconstruction))
                            / len(union)
                            if union
                            else 1.0
                        )
                        agreement_bin = (
                            "exact"
                            if exact_set_match
                            else "high"
                            if check.logic_status == "VALID" and jaccard >= 0.5
                            else "low"
                            if check.logic_status == "VALID" and jaccard > 0.0
                            else "none"
                        )
                        features["check::sealed_set_jaccard_sum"] = features.get(
                            "check::sealed_set_jaccard_sum", 0.0
                        ) + jaccard
                        features[
                            f"check::sealed_set_agreement::{agreement_bin}"
                        ] = features.get(
                            f"check::sealed_set_agreement::{agreement_bin}", 0.0
                        ) + 1.0
                        features[
                            f"check::sealed_set_agreement::{agreement_bin}::candidate::{claimed_relation}"
                        ] = features.get(
                            f"check::sealed_set_agreement::{agreement_bin}::candidate::{claimed_relation}",
                            0.0,
                        ) + 1.0
                    if check.counterfactual_pair:
                        trace_match = (
                            check.logic_status == "VALID"
                            and check.canonical_valid_trace
                            == certificate.sealed_valid_trace
                        )
                        effect_match = (
                            check.logic_status == "VALID"
                            and check.reconstructed_effect
                            == certificate.sealed_effect
                            and claim_direction_match
                        )
                        triple_match = trace_match and effect_match
                        features[
                            f"check::parity_orientation::{check.orientation}"
                        ] = features.get(
                            f"check::parity_orientation::{check.orientation}", 0.0
                        ) + 1.0
                        if self.variant.use_sealed_set_agreement:
                            features[
                                f"check::sealed_trace_match::{str(trace_match).lower()}"
                            ] = features.get(
                                f"check::sealed_trace_match::{str(trace_match).lower()}",
                                0.0,
                            ) + 1.0
                            features[
                                f"check::sealed_effect_match::{str(effect_match).lower()}"
                            ] = features.get(
                                f"check::sealed_effect_match::{str(effect_match).lower()}",
                                0.0,
                            ) + 1.0
                            features[
                                f"check::sealed_triple_match::{str(triple_match).lower()}"
                            ] = features.get(
                                f"check::sealed_triple_match::{str(triple_match).lower()}",
                                0.0,
                            ) + 1.0
                        if (
                            self.variant.use_counterfactual_parity
                            and check.orientation == "canonical"
                        ):
                            siblings = [
                                row
                                for row in checks_by_certificate.get(
                                    certificate.certificate_id, ()
                                )
                                if row.counterfactual_pair
                                and row.checker_id == check.checker_id
                                and row.parse_error is None
                            ]
                            by_orientation = {
                                row.orientation: row for row in siblings
                            }
                            if set(by_orientation) == set(PARITY_ORIENTATIONS):
                                canonical = by_orientation["canonical"]
                                mirrored = by_orientation["mirrored"]
                                position_invariant = (
                                    canonical.logic_status
                                    == mirrored.logic_status
                                    and canonical.canonical_valid_trace
                                    == mirrored.canonical_valid_trace
                                    and canonical.reconstructed_effect
                                    == mirrored.reconstructed_effect
                                    and canonical.eliminated_options
                                    == mirrored.eliminated_options
                                    and canonical.supported_options
                                    == mirrored.supported_options
                                )
                                presented_flip = (
                                    canonical.presented_valid_trace is not None
                                    and mirrored.presented_valid_trace is not None
                                    and canonical.presented_valid_trace
                                    == 3 - mirrored.presented_valid_trace
                                )
                                parity_match = position_invariant and (
                                    canonical.logic_status != "VALID"
                                    or presented_flip
                                )
                                features[
                                    f"check::position_invariant::{str(parity_match).lower()}"
                                ] = features.get(
                                    f"check::position_invariant::{str(parity_match).lower()}",
                                    0.0,
                                ) + 1.0
                if self.variant.use_checker_answer_dependence:
                    features[f"check::checker_relation::{checker_relation}::status::{check.status}"] = features.get(
                        f"check::checker_relation::{checker_relation}::status::{check.status}",
                        0.0,
                    ) + 1.0
                if self.variant.use_generator_answer_dependence and self.variant.use_checker_answer_dependence:
                    features[f"check::relations::{generator_relation}::{checker_relation}::status::{check.status}"] = features.get(
                        f"check::relations::{generator_relation}::{checker_relation}::status::{check.status}",
                        0.0,
                    ) + 1.0
                if self.variant.use_generator_checker_pair_effects:
                    features[f"check::pair::{certificate.generator_id}::{check.checker_id}::status::{check.status}"] = 1.0
        return features

    @staticmethod
    def _validate_observations(
        question_by_id: Mapping[str, FalsificationQuestion],
        certificates: Sequence[CounterexampleCertificate],
        checks: Sequence[CertificateCheck],
    ) -> None:
        certificate_by_id: dict[str, CounterexampleCertificate] = {}
        for certificate in certificates:
            question = question_by_id.get(certificate.question_id)
            if question is None or certificate.candidate not in question.option_labels:
                raise ValueError("C3 certificate is outside its question option set")
            if certificate.certificate_id in certificate_by_id:
                raise ValueError("Duplicate C3 certificate")
            certificate_by_id[certificate.certificate_id] = certificate
        seen_checks: set[tuple[str, str, str]] = set()
        for check in checks:
            certificate = certificate_by_id.get(check.certificate_id)
            if certificate is None:
                raise ValueError("C3 check references an unknown certificate")
            if (
                check.question_id != certificate.question_id
                or check.generator_id != certificate.generator_id
                or check.candidate != certificate.candidate
            ):
                raise ValueError("C3 check metadata differs from its certificate")
            identity = (
                check.certificate_id,
                check.checker_id,
                check.orientation if check.counterfactual_pair else "single",
            )
            if identity in seen_checks:
                raise ValueError("Duplicate C3 certificate check")
            seen_checks.add(identity)

    def fit(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        certificates: Sequence[CounterexampleCertificate],
        checks: Sequence[CertificateCheck],
        labels: SourceTrainingLabels,
    ) -> "CrossExaminedCertificateCourt":
        if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
            raise TypeError("C3 may be fitted only with SourceTrainingLabels")
        question_by_id = {row.question_id: row for row in questions}
        if len(question_by_id) != len(questions):
            raise ValueError("C3 training questions contain duplicate IDs")
        self._validate_observations(question_by_id, certificates, checks)
        base_by_question = self._base_by_question(base_predictions)
        self.expert_ids_ = tuple(sorted({row.expert_id for row in base_predictions}))
        if not self.expert_ids_:
            raise ValueError("C3 requires at least one base expert")
        self.expert_accuracy_: dict[str, float] = {}
        for expert in self.expert_ids_:
            outcomes = [labels.get(question_id, expert) for question_id in question_by_id]
            if any(value is None for value in outcomes):
                raise ValueError("C3 source labels lack base-expert correctness")
            successes = sum(bool(value) for value in outcomes)
            self.expert_accuracy_[expert] = (successes + 1.0) / (len(outcomes) + 2.0)
        self.reference_expert_ = sorted(
            self.expert_ids_, key=lambda expert: (-self.expert_accuracy_[expert], expert)
        )[0]

        certificates_by_key: dict[tuple[str, str], list[CounterexampleCertificate]] = defaultdict(list)
        for row in certificates:
            certificates_by_key[(row.question_id, row.candidate)].append(row)
        checks_by_certificate: dict[str, list[CertificateCheck]] = defaultdict(list)
        for row in checks:
            checks_by_certificate[row.certificate_id].append(row)
        feature_rows: list[dict[str, float]] = []
        targets: list[int] = []
        for question_id in sorted(question_by_id):
            question = question_by_id[question_id]
            for candidate in question.option_labels:
                truth = labels.get(question_id, candidate_label_key(candidate))
                if truth is None:
                    raise ValueError("C3 source labels lack candidate correctness")
                feature_rows.append(
                    self._candidate_features(
                        question,
                        candidate,
                        base_by_question,
                        certificates_by_key,
                        checks_by_certificate,
                    )
                )
                targets.append(int(bool(truth)))
        self.vectorizer_ = DictVectorizer(sparse=True)
        matrix = self.vectorizer_.fit_transform(feature_rows)
        if len(set(targets)) != 2:
            raise ValueError("C3 training requires correct and incorrect candidates")
        self.classifier_ = LogisticRegression(
            C=self.variant.regularization_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=self.seed,
        ).fit(matrix, np.asarray(targets, dtype=int))
        return self

    def predict(
        self,
        questions: Sequence[FalsificationQuestion],
        base_predictions: Sequence[BasePrediction],
        certificates: Sequence[CounterexampleCertificate],
        checks: Sequence[CertificateCheck],
    ) -> list[C3Decision]:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("C3 must be fitted before prediction")
        question_by_id = {row.question_id: row for row in questions}
        self._validate_observations(question_by_id, certificates, checks)
        base_by_question = self._base_by_question(base_predictions)
        certificates_by_key: dict[tuple[str, str], list[CounterexampleCertificate]] = defaultdict(list)
        for row in certificates:
            certificates_by_key[(row.question_id, row.candidate)].append(row)
        checks_by_certificate: dict[str, list[CertificateCheck]] = defaultdict(list)
        for row in checks:
            checks_by_certificate[row.certificate_id].append(row)
        decisions: list[C3Decision] = []
        for question_id in sorted(question_by_id):
            question = question_by_id[question_id]
            base = base_by_question.get(question_id, {})
            proposed = {
                answer for answer in base.values() if answer in question.option_labels
            }
            candidates = (
                tuple(question.option_labels)
                if self.variant.open_option_set
                else tuple(label for label in question.option_labels if label in proposed)
            )
            if not candidates:
                candidates = tuple(question.option_labels)
            feature_rows = [
                self._candidate_features(
                    question,
                    candidate,
                    base_by_question,
                    certificates_by_key,
                    checks_by_certificate,
                )
                for candidate in candidates
            ]
            matrix = self.vectorizer_.transform(feature_rows)
            raw_logits = self.classifier_.decision_function(matrix)
            if np.ndim(raw_logits) == 0:
                raw_logits = np.asarray([float(raw_logits)])
            logits = {
                candidate: float(value)
                for candidate, value in zip(candidates, raw_logits, strict=True)
            }
            probabilities = _softmax(logits)
            reference = base.get(self.reference_expert_)
            if reference not in candidates:
                reference = sorted(
                    candidates,
                    key=lambda candidate: (
                        -sum(answer == candidate for answer in base.values()),
                        candidate,
                    ),
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
                fallback_reason = "c3_margin_below_source_selected_threshold"
            supporters = sorted(
                expert for expert, answer in base.items() if answer == chosen
            )
            selected_expert = (
                sorted(
                    supporters,
                    key=lambda expert: (-self.expert_accuracy_.get(expert, 0.0), expert),
                )[0]
                if supporters
                else None
            )
            decisions.append(
                C3Decision(
                    question_id=question_id,
                    answer=chosen,
                    reference_answer=str(reference),
                    selected_expert_id=selected_expert,
                    candidate_logits=logits,
                    candidate_probabilities=probabilities,
                    fallback_reason=fallback_reason,
                    open_set_rescue=chosen not in proposed,
                    diagnostics={
                        "method": self.variant.name,
                        "reference_expert": self.reference_expert_,
                        "proposed_candidates": sorted(proposed),
                        "scored_candidates": list(candidates),
                        "posterior_margin": float(margin),
                        "parsed_certificates": sum(
                            row.parse_error is None
                            for row in certificates_by_key.get((question_id, chosen), ())
                        ),
                        "parsed_checks": sum(
                            check.parse_error is None
                            for certificate in certificates_by_key.get((question_id, chosen), ())
                            for check in checks_by_certificate.get(certificate.certificate_id, ())
                        ),
                        "uses_target_labels": False,
                    },
                )
            )
        return decisions

    def with_variant(self, variant: C3Variant) -> "CrossExaminedCertificateCourt":
        model = copy.copy(self)
        model.variant = variant
        return model
