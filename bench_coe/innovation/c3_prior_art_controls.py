from __future__ import annotations

from typing import Sequence

from .blind_falsification_jury import FalsificationQuestion
from .cross_examined_certificates import CounterexampleCertificate
from .sealed_counterfactual_parity import (
    ISOLATED_TRACE_VIEWS,
    ParsedProofObligationAudit,
    bounded_private_response,
    parse_commitment_conditioned_proof_audit_output_v8,
)


CANDIDATE_VISIBLE_PROMPT_VERSION = (
    "candidate_visible_commit_first_proof_audit_v8_control"
)
UNSEALED_PROMPT_VERSION = "unsealed_set_aware_proof_audit_v8_control"
CANDIDATE_VISIBLE_PARSER_VERSION = (
    "candidate_visible_proof_obligation_fields_v8_control"
)
UNSEALED_PARSER_VERSION = "unsealed_proof_obligation_fields_v8_control"
ISOLATED_PRIOR_ART_CONTROL_PROMPTS = {
    CANDIDATE_VISIBLE_PROMPT_VERSION,
    UNSEALED_PROMPT_VERSION,
}


def _options_text(question: FalsificationQuestion) -> str:
    return "\n".join(
        f"({label}) {text}"
        for label, text in zip(
            question.option_labels, question.options, strict=True
        )
    )


def _sealed_option(certificate: CounterexampleCertificate) -> str:
    options = tuple(
        dict.fromkeys(
            (
                *certificate.claimed_eliminated_options,
                *certificate.claimed_supported_options,
            )
        )
    )
    if (
        not certificate.claim_was_sealed
        or certificate.sealed_valid_trace not in (1, 2)
        or certificate.sealed_effect is None
        or len(options) != 1
    ):
        raise ValueError("Prior-art prompt control requires one complete sealed claim")
    return options[0]


def _proof_contract() -> str:
    return (
        "Use one internally consistent branch:\n"
        "- VALID: COUNTERTEST_RESULT SURVIVES; one EFFECT and OPTION; flaw fields NONE.\n"
        "- INVALID: COUNTERTEST_RESULT BREAKS; EFFECT and OPTION NONE; concrete flaw code/detail.\n"
        "- INCONCLUSIVE: COUNTERTEST_RESULT UNCERTAIN; EFFECT and OPTION NONE; FLAW_CODE "
        "UNCERTAIN, confidence at most 50, and concrete uncertainty detail.\n\n"
        "Return exactly ten single-line fields with no surrounding prose.\n"
        "TRACE_STATUS: VALID|INVALID|INCONCLUSIVE\n"
        "COUNTERTEST: concrete attempted falsification\n"
        "COUNTERTEST_RESULT: SURVIVES|BREAKS|UNCERTAIN\n"
        "RECOMPUTATION: independently derived concrete result\n"
        "COMMITMENT_RELATION: CONSISTENT|CONFLICTS|UNRELATED|UNCERTAIN\n"
        "EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "FLAW_CODE: RULE_MISAPPLIED|ARITHMETIC|UNSUPPORTED_PREMISE|QUESTION_MISMATCH|OTHER|UNCERTAIN|NONE\n"
        "FLAW_DETAIL: concrete first flaw, uncertainty, or NONE"
    )


def _common_instruction() -> str:
    return (
        "Audit the one shown trace by actively trying to break it. COUNTERTEST must "
        "substitute a value, check a premise or unit, apply a boundary case, or derive "
        "the relevant rule independently. RECOMPUTATION must report the concrete result. "
        "A generic plausibility statement is not evidence. Mark VALID only if every local "
        "step survives and directly establishes exactly one signed option effect. Mark "
        "INVALID only with a concrete first fatal flaw, and otherwise mark INCONCLUSIVE. "
        "Do not infer all remaining options from uniqueness.\n\n"
    )


def build_candidate_visible_commit_first_prompt_v8_control(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    trace_view: str,
    private_stage0_response: str,
    certificate: CounterexampleCertificate,
) -> str:
    if trace_view not in ISOLATED_TRACE_VIEWS:
        raise ValueError(f"Unknown isolated trace view: {trace_view}")
    del first_differing_step
    trace = trace_1 if trace_view == "trace_1" else trace_2
    candidate = _sealed_option(certificate)
    candidate_text = question.options[question.option_labels.index(candidate)]
    return (
        "Act as a commit-first, candidate-aware proof auditor. Before seeing the candidate "
        "or trace you made the frozen Stage-0 attempt below; it may be wrong. This control "
        "now discloses the candidate under verification, but still hides the author's "
        "validity judgment, signed effect, trace index, sibling trace, pair existence, "
        "identity, votes, other audits, and gold answer. Do not accept the trace merely "
        "because it agrees with either the disclosed candidate or your commitment.\n\n"
        + _common_instruction()
        + f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private frozen Stage-0 attempt (untrusted precommitment):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Candidate under verification: ({candidate}) {candidate_text}\n"
        f"Atomic rule to verify:\n{rule}\n"
        f"Single trace under audit:\n{trace}\n\n"
        + _proof_contract()
    )


def build_unsealed_set_aware_prompt_v8_control(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    trace_view: str,
    private_stage0_response: str,
    certificate: CounterexampleCertificate,
) -> str:
    if trace_view not in ISOLATED_TRACE_VIEWS:
        raise ValueError(f"Unknown isolated trace view: {trace_view}")
    del first_differing_step
    trace_index = 1 + ISOLATED_TRACE_VIEWS.index(trace_view)
    trace = trace_1 if trace_view == "trace_1" else trace_2
    candidate = _sealed_option(certificate)
    claimed_valid = trace_index == certificate.sealed_valid_trace
    shown_claim = (
        f"VALID; {certificate.sealed_effect} option ({candidate})"
        if claimed_valid
        else "INVALID; no signed option effect"
    )
    return (
        "Act as an unsealed, set-aware proof auditor. Before this audit you made the "
        "frozen Stage-0 attempt below; it may be wrong. This control deliberately reveals "
        "the author's validity and signed option-effect claim for the one shown trace. "
        "The sibling trace itself, pair existence, author identity, votes, other audits, "
        "and gold answer remain hidden. Treat the revealed claim as untrusted testimony, "
        "not evidence, and try to falsify it independently.\n\n"
        + _common_instruction()
        + f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private frozen Stage-0 attempt (untrusted precommitment):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Author's unsealed claim for this trace: {shown_claim}\n"
        f"Atomic rule to verify:\n{rule}\n"
        f"Single trace under audit:\n{trace}\n\n"
        + _proof_contract()
    )


def parse_candidate_visible_proof_output_v8_control(
    text: str, option_labels: Sequence[str]
) -> ParsedProofObligationAudit:
    return parse_commitment_conditioned_proof_audit_output_v8(text, option_labels)


def parse_unsealed_proof_output_v8_control(
    text: str, option_labels: Sequence[str]
) -> ParsedProofObligationAudit:
    return parse_commitment_conditioned_proof_audit_output_v8(text, option_labels)
