from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .blind_falsification_jury import FalsificationQuestion


CHALLENGE_EFFECTS = ("ELIMINATES", "SUPPORTS")
PARITY_ORIENTATIONS = ("canonical", "mirrored")
ISOLATED_TRACE_VIEWS = ("trace_1", "trace_2")
ISOLATED_FLAW_CODES = (
    "RULE_MISAPPLIED",
    "ARITHMETIC",
    "UNSUPPORTED_PREMISE",
    "QUESTION_MISMATCH",
    "OTHER",
)
C3_V8_MECHANISM_ABLATION_PROTOCOLS = {
    "no_checker_private_precommitment": {
        "changed_factor": "checker_private_stage0_response_removed_from_prompt",
        "prompt_version": "blind_isolated_trace_audit_v7",
        "parser_version": "isolated_trace_audit_fields_v7",
        "check_calls_per_witness": 2,
    },
    "pair_visible_with_precommitment": {
        "changed_factor": "sibling_trace_and_pair_existence_revealed",
        "prompt_version": "commitment_conditioned_pair_audit_v8_ablation",
        "parser_version": "pair_proof_obligation_audit_fields_v8_ablation",
        "check_calls_per_witness": 2,
    },
    "candidate_visible_commit_first": {
        "changed_factor": "sealed_target_option_revealed_after_checker_precommitment",
        "prompt_version": "candidate_visible_commit_first_proof_audit_v8_control",
        "parser_version": "candidate_visible_proof_obligation_fields_v8_control",
        "check_calls_per_witness": 2,
    },
    "unsealed_set_aware": {
        "changed_factor": "author_validity_effect_and_option_claim_revealed",
        "prompt_version": "unsealed_set_aware_proof_audit_v8_control",
        "parser_version": "unsealed_proof_obligation_fields_v8_control",
        "check_calls_per_witness": 2,
    },
}


def validate_c3_v8_mechanism_ablation(
    base_config: Mapping[str, Any],
    ablation_config: Mapping[str, Any],
) -> str:
    ablation = ablation_config.get("mechanism_ablation")
    if not isinstance(ablation, Mapping):
        raise TypeError("C3 mechanism ablation config lacks its boundary block")
    name = str(ablation.get("name", ""))
    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in name
    ):
        raise ValueError("C3 mechanism ablation name must be lowercase snake_case")
    expected = C3_V8_MECHANISM_ABLATION_PROTOCOLS.get(name)
    if expected is None:
        raise PermissionError(f"Unregistered C3 v8 mechanism ablation: {name}")
    if ablation.get("changed_factor") != expected["changed_factor"]:
        raise PermissionError("C3 mechanism ablation changed-factor declaration differs")
    if ablation.get("unchanged_call_budget") is not True:
        raise PermissionError("C3 mechanism ablation does not preserve call budget")
    if ablation.get("target_labels_control_stopping_or_selection") is not False:
        raise PermissionError("C3 mechanism ablation lacks a label-free boundary")

    for key in (
        "protocol_version",
        "seed",
        "models_dir",
        "physical_gpus",
        "experts",
        "certificate_models",
        "checker_models",
        "datasets",
        "certificate_generation",
    ):
        if ablation_config.get(key) != base_config.get(key):
            raise PermissionError(
                f"C3 mechanism ablation changed a fixed field: {key}"
            )

    base_check = base_config.get("check_generation")
    ablation_check = ablation_config.get("check_generation")
    if not isinstance(base_check, Mapping) or not isinstance(
        ablation_check, Mapping
    ):
        raise TypeError("C3 mechanism ablation lacks check generation settings")
    if (
        base_check.get("prompt_version")
        != "commitment_conditioned_proof_audit_v8"
        or base_check.get("parser_version")
        != "proof_obligation_audit_fields_v8"
    ):
        raise PermissionError("C3 mechanism ablation base is not the v8 audit protocol")
    if (
        ablation_check.get("prompt_version") != expected["prompt_version"]
        or ablation_check.get("parser_version") != expected["parser_version"]
    ):
        raise PermissionError("C3 mechanism ablation protocol pair differs")
    allowed_protocol_keys = {"prompt_version", "parser_version"}
    generation_keys = set(base_check).union(ablation_check)
    for key in sorted(generation_keys.difference(allowed_protocol_keys)):
        if ablation_check.get(key) != base_check.get(key):
            raise PermissionError(
                f"C3 mechanism ablation changed check generation field: {key}"
            )
    if int(expected["check_calls_per_witness"]) != 2:
        raise PermissionError("C3 mechanism ablation call multiplier differs")
    return name


@dataclass(frozen=True)
class ParsedCounterfactualChallenge:
    rule: str | None
    trace_1: str | None
    trace_2: str | None
    first_differing_step: str | None
    valid_trace: int | None
    effect: str | None
    option: str | None
    confidence: int
    parse_error: str | None

    @property
    def abstained(self) -> bool:
        return self.parse_error is None and self.valid_trace is None


@dataclass(frozen=True)
class ParsedParityAudit:
    pair_status: str
    presented_valid_trace: int | None
    effect: str | None
    option: str | None
    confidence: int
    first_flaw: str | None
    parse_error: str | None


@dataclass(frozen=True)
class ParsedIsolatedTraceAudit:
    trace_status: str
    effect: str | None
    option: str | None
    confidence: int
    flaw_code: str
    flaw_detail: str | None
    parse_error: str | None


@dataclass(frozen=True)
class ParsedProofObligationAudit:
    trace_status: str
    effect: str | None
    option: str | None
    confidence: int
    flaw_code: str
    flaw_detail: str | None
    countertest: str | None
    countertest_result: str
    recomputation: str | None
    commitment_relation: str
    parse_error: str | None


@dataclass(frozen=True)
class ParsedPairProofObligationAudit:
    pair_status: str
    presented_valid_trace: int | None
    effect: str | None
    option: str | None
    confidence: int
    first_flaw: str | None
    countertest: str | None
    countertest_result: str
    recomputation: str | None
    commitment_relation: str
    parse_error: str | None


def counterfactual_trace_slot(seed: int, question_id: str, generator_id: str) -> int:
    payload = f"{int(seed)}\0{question_id}\0{generator_id}".encode("utf-8")
    return 1 + (hashlib.sha256(payload).digest()[0] & 1)


def bounded_private_response(response: str, limit: int = 2400) -> str:
    clean = " ".join(str(response or "").split())
    if len(clean) <= limit:
        return clean or "[EMPTY_STAGE0_TRACE]"
    half = (limit - len(" [TRUNCATED_MIDDLE] ")) // 2
    return f"{clean[:half]} [TRUNCATED_MIDDLE] {clean[-half:]}"


def _options_text(question: FalsificationQuestion) -> str:
    return "\n".join(
        f"({label}) {text}"
        for label, text in zip(
            question.option_labels, question.options, strict=True
        )
    )


def build_sealed_counterfactual_challenge_prompt_v4(
    question: FalsificationQuestion,
    private_stage0_response: str,
    required_valid_trace: int,
) -> str:
    if required_valid_trace not in (1, 2):
        raise ValueError("A counterfactual challenge valid-trace slot must be 1 or 2")
    return (
        "Act as a counterfactual-challenge author. You receive one private Stage-0 "
        "reasoning trace from your own independent solve. No other model output, model "
        "identity, vote count, popularity, source score, or gold answer is available. "
        "The private trace is never shown to a checker. Recheck it against the question.\n\n"
        "If one local rule or calculation rigorously supports or eliminates exactly one "
        "listed option, make a paired challenge. TRACE_1 and TRACE_2 must share the same "
        "setup and differ at exactly the named first decisive step. Exactly one trace must "
        "be logically valid; the other must contain one minimal, plausible counterfactual "
        "error. The valid trace must have exactly one signed effect: it either ELIMINATES "
        "or SUPPORTS one option. Do not infer all other options from uniqueness. Do not "
        "write option labels, answer labels, votes, or model names in RULE, either TRACE, "
        "or FIRST_DIFFERING_STEP. Put the valid trace in the required slot below.\n\n"
        "If no such rigorous single-effect pair can be made, abstain by using NONE for "
        "SEALED_VALID_TRACE, SEALED_EFFECT, and SEALED_OPTION, confidence at most 50, and "
        "state the limitation in the visible fields.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private Stage-0 trace (not shown to checkers):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Required valid-trace slot for a non-abstaining pair: {required_valid_trace}\n\n"
        "Return exactly eight single-line fields with no surrounding prose.\n"
        "RULE: one atomic fact, constraint, or calculation rule\n"
        "TRACE_1: first compact derivation\n"
        "TRACE_2: minimally different compact derivation\n"
        "FIRST_DIFFERING_STEP: the first and only decisive difference\n"
        "SEALED_VALID_TRACE: 1|2|NONE\n"
        "SEALED_EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "SEALED_OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100"
    )


def _field(name: str, choices: str | None = None) -> re.Pattern[str]:
    value = choices if choices is not None else r"(\S.*?)"
    return re.compile(
        rf"^\s*{re.escape(name)}\s*:\s*{value}\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )


_CHALLENGE_PATTERNS = {
    "rule": _field("RULE"),
    "trace_1": _field("TRACE_1"),
    "trace_2": _field("TRACE_2"),
    "first_differing_step": _field("FIRST_DIFFERING_STEP"),
    "valid_trace": _field("SEALED_VALID_TRACE", r"(1|2|NONE)"),
    "effect": _field("SEALED_EFFECT", r"(ELIMINATES|SUPPORTS|NONE)"),
    "option": _field("SEALED_OPTION", r"\(?([A-Z]|NONE)\)?"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
}

_VISIBLE_LABEL_LEAK = re.compile(
    r"(?:\b(?:option|answer|choice)\s*\(?[A-Z]\)?\b|\([A-Z]\))",
    flags=re.IGNORECASE,
)


def _unique_fields(
    text: str, patterns: dict[str, re.Pattern[str]]
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


def parse_sealed_counterfactual_challenge_output_v4(
    text: str,
    option_labels: Sequence[str],
    expected_valid_trace: int,
) -> ParsedCounterfactualChallenge:
    if expected_valid_trace not in (1, 2):
        raise ValueError("Expected valid-trace slot must be 1 or 2")
    values, error = _unique_fields(text, _CHALLENGE_PATTERNS)
    if error is not None:
        return ParsedCounterfactualChallenge(
            None, None, None, None, None, None, None, 0, error
        )
    confidence = int(values["confidence"])
    valid_value = values["valid_trace"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    none_pattern = (
        valid_value == "NONE",
        effect_value == "NONE",
        option_value == "NONE",
    )
    if any(none_pattern) and not all(none_pattern):
        error = "partial_abstention"
    elif all(none_pattern) and confidence > 50:
        error = "overconfident_abstention"
    elif not all(none_pattern) and int(valid_value) != expected_valid_trace:
        error = "valid_trace_slot_mismatch"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    elif not all(none_pattern) and values["trace_1"].strip() == values["trace_2"].strip():
        error = "identical_counterfactual_traces"
    elif not all(none_pattern) and any(
        _VISIBLE_LABEL_LEAK.search(values[name])
        for name in ("rule", "trace_1", "trace_2", "first_differing_step")
    ):
        error = "visible_option_label_leak"
    else:
        error = None
    if error is not None:
        return ParsedCounterfactualChallenge(
            None, None, None, None, None, None, None, 0, error
        )
    return ParsedCounterfactualChallenge(
        rule=values["rule"],
        trace_1=values["trace_1"],
        trace_2=values["trace_2"],
        first_differing_step=values["first_differing_step"],
        valid_trace=None if valid_value == "NONE" else int(valid_value),
        effect=None if effect_value == "NONE" else effect_value,
        option=None if option_value == "NONE" else option_value,
        confidence=confidence,
        parse_error=None,
    )


def build_hardened_counterfactual_challenge_prompt_v5(
    question: FalsificationQuestion,
    private_stage0_response: str,
    required_valid_trace: int,
) -> str:
    if required_valid_trace not in (1, 2):
        raise ValueError("A counterfactual challenge valid-trace slot must be 1 or 2")
    return (
        "Act as a meticulous counterfactual unit-test author. You receive only your own "
        "frozen Stage-0 solve; treat it as an untrusted hypothesis and recompute every local "
        "fact or operation you use. No other response, identity, vote, score, or gold answer "
        "is available, and your private solve is never shown to an auditor.\n\n"
        "Either abstain or construct a minimal paired unit test for one listed option. Start "
        "from one atomic rule whose applicability you have checked. TRACE_1 and TRACE_2 must "
        "use the same premises and differ in exactly one explicitly quoted value, operator, "
        "constraint, or inference at FIRST_DIFFERING_STEP. Re-evaluate both traces from "
        "scratch: exactly one must be valid and the other must fail because of that one local "
        "mutation. The valid trace must directly ELIMINATE or SUPPORT exactly one option. "
        "Do not infer all remaining options from uniqueness.\n\n"
        "Keep option identifiers out of RULE, both TRACE fields, and FIRST_DIFFERING_STEP. "
        "In particular, do not write phrases such as option A or bare parenthesized option "
        "letters; use descriptive mathematical variable names when needed. Put the valid "
        "trace in the required slot. Before returning, verify that all three SEALED fields "
        "are either a complete claim or all NONE. If a unit, exponent, rule premise, option "
        "mapping, or single-mutation condition is uncertain, abstain.\n\n"
        "For abstention, use NONE for SEALED_VALID_TRACE, SEALED_EFFECT, and SEALED_OPTION, "
        "confidence at most 50, and describe the limitation in the visible fields.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private frozen Stage-0 solve (untrusted; hidden from auditors):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Required valid-trace slot for a non-abstaining pair: {required_valid_trace}\n\n"
        "Return exactly eight single-line fields with no surrounding prose.\n"
        "RULE: one checked atomic fact, constraint, or calculation rule\n"
        "TRACE_1: first compact derivation with units or premises when applicable\n"
        "TRACE_2: minimally mutated compact derivation\n"
        "FIRST_DIFFERING_STEP: quote the exact before-versus-after local mutation\n"
        "SEALED_VALID_TRACE: 1|2|NONE\n"
        "SEALED_EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "SEALED_OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100"
    )


def _visible_option_label_leak_v5(
    text: str, option_labels: Sequence[str]
) -> bool:
    labels = "|".join(
        sorted((re.escape(str(label)) for label in option_labels), key=len, reverse=True)
    )
    if not labels:
        return False
    explicit = re.compile(
        rf"(?i:\b(?:option|answer|choice)\b)\s*\(?(?:{labels})\)?(?=\s|[.,;:!?)]|$)"
    )
    parenthesized = re.compile(rf"\((?:{labels})\)")
    return explicit.search(text) is not None or parenthesized.search(text) is not None


def parse_hardened_counterfactual_challenge_output_v5(
    text: str,
    option_labels: Sequence[str],
    expected_valid_trace: int,
) -> ParsedCounterfactualChallenge:
    if expected_valid_trace not in (1, 2):
        raise ValueError("Expected valid-trace slot must be 1 or 2")
    values, error = _unique_fields(text, _CHALLENGE_PATTERNS)
    if error is not None:
        return ParsedCounterfactualChallenge(
            None, None, None, None, None, None, None, 0, error
        )
    confidence = int(values["confidence"])
    valid_value = values["valid_trace"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    none_pattern = (
        valid_value == "NONE",
        effect_value == "NONE",
        option_value == "NONE",
    )
    if any(none_pattern) and not all(none_pattern):
        error = "partial_abstention"
    elif all(none_pattern) and confidence > 50:
        error = "overconfident_abstention"
    elif not all(none_pattern) and int(valid_value) != expected_valid_trace:
        error = "valid_trace_slot_mismatch"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    elif not all(none_pattern) and values["trace_1"].strip() == values["trace_2"].strip():
        error = "identical_counterfactual_traces"
    elif not all(none_pattern) and any(
        _visible_option_label_leak_v5(values[name], option_labels)
        for name in ("rule", "trace_1", "trace_2", "first_differing_step")
    ):
        error = "visible_option_label_leak"
    else:
        error = None
    if error is not None:
        return ParsedCounterfactualChallenge(
            None, None, None, None, None, None, None, 0, error
        )
    return ParsedCounterfactualChallenge(
        rule=values["rule"],
        trace_1=values["trace_1"],
        trace_2=values["trace_2"],
        first_differing_step=values["first_differing_step"],
        valid_trace=None if valid_value == "NONE" else int(valid_value),
        effect=None if effect_value == "NONE" else effect_value,
        option=None if option_value == "NONE" else option_value,
        confidence=confidence,
        parse_error=None,
    )


def build_committed_counterfactual_challenge_prompt_v6(
    question: FalsificationQuestion,
    private_stage0_response: str,
) -> str:
    return (
        "Act as a meticulous counterfactual unit-test author. You receive only your own "
        "frozen Stage-0 solve; treat it as an untrusted hypothesis and recompute every local "
        "fact or operation. No other response, identity, vote, score, or gold answer is "
        "available, and your private solve is never shown to an auditor.\n\n"
        "Either abstain or construct a minimal paired unit test for one listed option. Start "
        "from one checked atomic rule. TRACE_1 and TRACE_2 must use the same premises and "
        "differ in exactly one explicitly quoted value, operator, constraint, or inference "
        "at FIRST_DIFFERING_STEP. Re-evaluate both traces: exactly one must be valid and the "
        "other must fail because of that local mutation. The valid trace must directly "
        "ELIMINATE or SUPPORT exactly one option; do not infer all other options from "
        "uniqueness.\n\n"
        "Commit honestly to whichever trace is valid. After your output is frozen, trusted "
        "deterministic code may swap TRACE_1 and TRACE_2 using a hidden hash-balanced bit and "
        "will update the sealed trace index consistently. Thus trace position conveys no "
        "information to the later auditor.\n\n"
        "Keep option identifiers out of RULE, both TRACE fields, and FIRST_DIFFERING_STEP. "
        "Do not write phrases such as option A or bare parenthesized option letters; use "
        "descriptive mathematical variable names. If a unit, exponent, premise, option "
        "mapping, or single-mutation condition is uncertain, abstain. For abstention, all "
        "three SEALED fields must be NONE and confidence must be at most 50.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private frozen Stage-0 solve (untrusted; hidden from auditors):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        "Return exactly eight single-line fields with no surrounding prose.\n"
        "RULE: one checked atomic fact, constraint, or calculation rule\n"
        "TRACE_1: first compact derivation with units or premises when applicable\n"
        "TRACE_2: minimally mutated compact derivation\n"
        "FIRST_DIFFERING_STEP: quote the exact before-versus-after local mutation\n"
        "SEALED_VALID_TRACE: 1|2|NONE\n"
        "SEALED_EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "SEALED_OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100"
    )


def parse_committed_counterfactual_challenge_output_v6(
    text: str,
    option_labels: Sequence[str],
) -> ParsedCounterfactualChallenge:
    values, error = _unique_fields(text, _CHALLENGE_PATTERNS)
    if error is not None:
        return ParsedCounterfactualChallenge(
            None, None, None, None, None, None, None, 0, error
        )
    confidence = int(values["confidence"])
    valid_value = values["valid_trace"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    none_pattern = (
        valid_value == "NONE",
        effect_value == "NONE",
        option_value == "NONE",
    )
    if any(none_pattern) and not all(none_pattern):
        error = "partial_abstention"
    elif all(none_pattern) and confidence > 50:
        error = "overconfident_abstention"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    elif not all(none_pattern) and values["trace_1"].strip() == values["trace_2"].strip():
        error = "identical_counterfactual_traces"
    elif not all(none_pattern) and any(
        _visible_option_label_leak_v5(values[name], option_labels)
        for name in ("rule", "trace_1", "trace_2", "first_differing_step")
    ):
        error = "visible_option_label_leak"
    else:
        error = None
    if error is not None:
        return ParsedCounterfactualChallenge(
            None, None, None, None, None, None, None, 0, error
        )
    return ParsedCounterfactualChallenge(
        rule=values["rule"],
        trace_1=values["trace_1"],
        trace_2=values["trace_2"],
        first_differing_step=values["first_differing_step"],
        valid_trace=None if valid_value == "NONE" else int(valid_value),
        effect=None if effect_value == "NONE" else effect_value,
        option=None if option_value == "NONE" else option_value,
        confidence=confidence,
        parse_error=None,
    )


def permute_committed_counterfactual_challenge(
    challenge: ParsedCounterfactualChallenge,
    required_valid_trace: int,
) -> tuple[ParsedCounterfactualChallenge, bool]:
    if required_valid_trace not in (1, 2):
        raise ValueError("Required valid-trace slot must be 1 or 2")
    if challenge.parse_error is not None or challenge.valid_trace is None:
        return challenge, False
    if challenge.valid_trace == required_valid_trace:
        return challenge, False
    return (
        replace(
            challenge,
            trace_1=challenge.trace_2,
            trace_2=challenge.trace_1,
            valid_trace=required_valid_trace,
        ),
        True,
    )


def effect_option_sets(
    effect: str | None, option: str | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if effect is None and option is None:
        return (), ()
    if effect not in CHALLENGE_EFFECTS or option is None:
        raise ValueError("Counterfactual challenge effect and option must be paired")
    if effect == "ELIMINATES":
        return (option,), ()
    return (), (option,)


def signed_effect_option(
    effect: str | None,
    eliminated_options: Sequence[str],
    supported_options: Sequence[str],
) -> str | None:
    eliminated = tuple(str(value) for value in eliminated_options)
    supported = tuple(str(value) for value in supported_options)
    if effect == "ELIMINATES" and len(eliminated) == 1 and not supported:
        return eliminated[0]
    if effect == "SUPPORTS" and len(supported) == 1 and not eliminated:
        return supported[0]
    return None


def sealed_triple_matches(
    sealed_valid_trace: int | None,
    sealed_effect: str | None,
    sealed_eliminated_options: Sequence[str],
    sealed_supported_options: Sequence[str],
    reconstructed_valid_trace: int | None,
    reconstructed_effect: str | None,
    reconstructed_eliminated_options: Sequence[str],
    reconstructed_supported_options: Sequence[str],
) -> bool:
    sealed_option = signed_effect_option(
        sealed_effect, sealed_eliminated_options, sealed_supported_options
    )
    reconstructed_option = signed_effect_option(
        reconstructed_effect,
        reconstructed_eliminated_options,
        reconstructed_supported_options,
    )
    return (
        sealed_valid_trace in (1, 2)
        and reconstructed_valid_trace == sealed_valid_trace
        and reconstructed_effect == sealed_effect
        and reconstructed_option is not None
        and reconstructed_option == sealed_option
    )


def build_blind_counterfactual_parity_prompt_v4(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    orientation: str,
) -> str:
    if orientation not in PARITY_ORIENTATIONS:
        raise ValueError(f"Unknown counterfactual parity orientation: {orientation}")
    shown_1, shown_2 = (
        (trace_1, trace_2)
        if orientation == "canonical"
        else (trace_2, trace_1)
    )
    return (
        "Act as a blind counterfactual-pair auditor. The author, model identity, private "
        "Stage-0 response, claimed valid trace, claimed effect, claimed option, confidence, "
        "answer votes, popularity, other audits, and gold answer are hidden. Inspect only "
        "the question, rule, and two presented traces. Do not choose by writing quality or "
        "trace position.\n\n"
        "Return ONE_VALID only when exactly one trace is logically valid under the stated "
        "rule. Then identify that presented trace and the single signed option effect that "
        "its local derivation rigorously establishes. Do not infer effects on all remaining "
        "options from uniqueness. Return BOTH_INVALID when both traces have fatal errors, "
        "or INCONCLUSIVE when the pair cannot be resolved.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        f"Rule:\n{rule}\n"
        f"Presented trace 1:\n{shown_1}\n"
        f"Presented trace 2:\n{shown_2}\n"
        f"Claimed location of the first difference (not its validity):\n{first_differing_step}\n\n"
        "Return exactly six single-line fields with no surrounding prose.\n"
        "PAIR_STATUS: ONE_VALID|BOTH_INVALID|INCONCLUSIVE\n"
        "VALID_TRACE: 1|2|NONE\n"
        "EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "FIRST_FLAW: first fatal error in the rejected trace, or NONE"
    )


def build_hardened_blind_counterfactual_parity_prompt_v5(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    orientation: str,
) -> str:
    if orientation not in PARITY_ORIENTATIONS:
        raise ValueError(f"Unknown counterfactual parity orientation: {orientation}")
    shown_1, shown_2 = (
        (trace_1, trace_2)
        if orientation == "canonical"
        else (trace_2, trace_1)
    )
    return (
        "Act as a blind counterfactual unit-test auditor. The author, private Stage-0 "
        "solve, claimed valid trace, claimed effect, claimed option, confidence, answer "
        "votes, other audits, and gold answer are hidden. Audit only the question, atomic "
        "rule, and two presented traces. Trace position and writing style are not evidence.\n\n"
        "First independently check that the rule applies to this question. Then recompute "
        "every value, unit, operator, and inference in both traces. Verify that the stated "
        "first difference is truly local and decisive. Return ONE_VALID only if exactly one "
        "trace is valid and the other has one identifiable fatal error at that local "
        "difference. Then reconstruct the valid presented trace and exactly one option it "
        "directly ELIMINATES or SUPPORTS. Do not infer effects on all remaining options. "
        "Return BOTH_INVALID when both traces have fatal defects, and INCONCLUSIVE whenever "
        "the rule, arithmetic, locality, or option mapping cannot be established.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        f"Atomic rule:\n{rule}\n"
        f"Presented trace 1:\n{shown_1}\n"
        f"Presented trace 2:\n{shown_2}\n"
        "Claimed location of the first difference (not its validity):\n"
        f"{first_differing_step}\n\n"
        "Return exactly six single-line fields with no surrounding prose.\n"
        "PAIR_STATUS: ONE_VALID|BOTH_INVALID|INCONCLUSIVE\n"
        "VALID_TRACE: 1|2|NONE\n"
        "EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "FIRST_FLAW: first fatal error in the rejected trace, or NONE"
    )


_AUDIT_PATTERNS = {
    "pair_status": _field(
        "PAIR_STATUS", r"(ONE_VALID|BOTH_INVALID|INCONCLUSIVE)"
    ),
    "valid_trace": _field("VALID_TRACE", r"(1|2|NONE)"),
    "effect": _field("EFFECT", r"(ELIMINATES|SUPPORTS|NONE)"),
    "option": _field("OPTION", r"\(?([A-Z]|NONE)\)?"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
    "first_flaw": _field("FIRST_FLAW"),
}


def parse_blind_counterfactual_parity_output_v4(
    text: str, option_labels: Sequence[str]
) -> ParsedParityAudit:
    values, error = _unique_fields(text, _AUDIT_PATTERNS)
    if error is not None:
        return ParsedParityAudit(
            "INCONCLUSIVE", None, None, None, 0, None, error
        )
    status = values["pair_status"].upper()
    valid_value = values["valid_trace"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    confidence = int(values["confidence"])
    none_pattern = (
        valid_value == "NONE",
        effect_value == "NONE",
        option_value == "NONE",
    )
    flaw_is_none = values["first_flaw"].strip().upper() == "NONE"
    if status == "ONE_VALID" and any(none_pattern):
        error = "one_valid_without_complete_reconstruction"
    elif status != "ONE_VALID" and not all(none_pattern):
        error = "nonvalid_pair_with_effect"
    elif status == "ONE_VALID" and flaw_is_none:
        error = "one_valid_without_rejected_trace_flaw"
    elif status == "BOTH_INVALID" and flaw_is_none:
        error = "both_invalid_without_flaw"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    else:
        error = None
    if error is not None:
        return ParsedParityAudit(
            "INCONCLUSIVE", None, None, None, 0, None, error
        )
    return ParsedParityAudit(
        pair_status=status,
        presented_valid_trace=(
            None if valid_value == "NONE" else int(valid_value)
        ),
        effect=None if effect_value == "NONE" else effect_value,
        option=None if option_value == "NONE" else option_value,
        confidence=confidence,
        first_flaw=values["first_flaw"],
        parse_error=None,
    )


def parse_hardened_blind_counterfactual_parity_output_v5(
    text: str, option_labels: Sequence[str]
) -> ParsedParityAudit:
    return parse_blind_counterfactual_parity_output_v4(text, option_labels)


def canonical_trace_index(
    presented_index: int | None, orientation: str
) -> int | None:
    if orientation not in PARITY_ORIENTATIONS:
        raise ValueError(f"Unknown counterfactual parity orientation: {orientation}")
    if presented_index is None:
        return None
    if presented_index not in (1, 2):
        raise ValueError("Presented trace index must be 1, 2, or None")
    return presented_index if orientation == "canonical" else 3 - presented_index


def build_blind_isolated_trace_audit_prompt_v7(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    trace_view: str,
) -> str:
    if trace_view not in ISOLATED_TRACE_VIEWS:
        raise ValueError(f"Unknown isolated trace view: {trace_view}")
    del first_differing_step
    trace = trace_1 if trace_view == "trace_1" else trace_2
    return (
        "Act as a blind pointwise reasoning auditor. You receive exactly one proposed "
        "trace. The other trace, whether a pair exists, the author, model identity, private "
        "Stage-0 solve, claimed validity, claimed effect, confidence, votes, other audits, "
        "and gold answer are hidden. Never assume that this trace has a valid or invalid "
        "counterpart. Judge it independently from the question.\n\n"
        "First verify that the supplied atomic rule is true and applicable here. Then check "
        "every premise, value, unit, operator, and inference in the trace. Return VALID only "
        "when the entire local derivation is sound and it directly establishes exactly one "
        "signed option effect. Do not infer all remaining options from uniqueness. Return "
        "INVALID only with a concrete first fatal flaw. Return INCONCLUSIVE if validity or "
        "the option mapping cannot be established. Writing style is not evidence.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        f"Atomic rule to verify:\n{rule}\n"
        f"Single trace under audit:\n{trace}\n\n"
        "Use one internally consistent output branch:\n"
        "- VALID: one non-NONE EFFECT and OPTION; FLAW_CODE and FLAW_DETAIL are NONE.\n"
        "- INVALID: EFFECT and OPTION are NONE; use a non-NONE FLAW_CODE and concrete detail.\n"
        "- INCONCLUSIVE: EFFECT and OPTION are NONE; FLAW_CODE is UNCERTAIN, confidence is "
        "at most 50, and detail states what cannot be verified.\n\n"
        "Return exactly six single-line fields with no surrounding prose.\n"
        "TRACE_STATUS: VALID|INVALID|INCONCLUSIVE\n"
        "EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "FLAW_CODE: RULE_MISAPPLIED|ARITHMETIC|UNSUPPORTED_PREMISE|QUESTION_MISMATCH|OTHER|UNCERTAIN|NONE\n"
        "FLAW_DETAIL: concrete first flaw, uncertainty, or NONE"
    )


_ISOLATED_AUDIT_PATTERNS = {
    "trace_status": _field("TRACE_STATUS", r"(VALID|INVALID|INCONCLUSIVE)"),
    "effect": _field("EFFECT", r"(ELIMINATES|SUPPORTS|NONE)"),
    "option": _field("OPTION", r"\(?([A-Z]|NONE)\)?"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
    "flaw_code": _field(
        "FLAW_CODE",
        r"(RULE_MISAPPLIED|ARITHMETIC|UNSUPPORTED_PREMISE|QUESTION_MISMATCH|OTHER|UNCERTAIN|NONE)",
    ),
    "flaw_detail": _field("FLAW_DETAIL"),
}


def parse_blind_isolated_trace_audit_output_v7(
    text: str, option_labels: Sequence[str]
) -> ParsedIsolatedTraceAudit:
    values, error = _unique_fields(text, _ISOLATED_AUDIT_PATTERNS)
    if error is not None:
        return ParsedIsolatedTraceAudit(
            "INCONCLUSIVE", None, None, 0, "UNCERTAIN", None, error
        )
    status = values["trace_status"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    confidence = int(values["confidence"])
    flaw_code = values["flaw_code"].upper()
    flaw_detail = values["flaw_detail"].strip()
    effect_is_none = effect_value == "NONE"
    option_is_none = option_value == "NONE"
    detail_is_none = flaw_detail.upper() == "NONE"
    if effect_is_none != option_is_none:
        error = "partial_effect_reconstruction"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    elif status == "VALID" and (effect_is_none or flaw_code != "NONE" or not detail_is_none):
        error = "invalid_valid_branch"
    elif status == "INVALID" and (
        not effect_is_none
        or flaw_code not in ISOLATED_FLAW_CODES
        or detail_is_none
    ):
        error = "invalid_invalid_branch"
    elif status == "INCONCLUSIVE" and (
        not effect_is_none
        or flaw_code != "UNCERTAIN"
        or detail_is_none
        or confidence > 50
    ):
        error = "invalid_inconclusive_branch"
    else:
        error = None
    if error is not None:
        return ParsedIsolatedTraceAudit(
            "INCONCLUSIVE", None, None, 0, "UNCERTAIN", None, error
        )
    return ParsedIsolatedTraceAudit(
        trace_status=status,
        effect=None if effect_is_none else effect_value,
        option=None if option_is_none else option_value,
        confidence=confidence,
        flaw_code=flaw_code,
        flaw_detail=None if detail_is_none else flaw_detail,
        parse_error=None,
    )


def combine_isolated_trace_audits(
    trace_1_audit: ParsedIsolatedTraceAudit | ParsedProofObligationAudit,
    trace_2_audit: ParsedIsolatedTraceAudit | ParsedProofObligationAudit,
) -> ParsedParityAudit:
    audits = (trace_1_audit, trace_2_audit)
    if any(audit.parse_error is not None for audit in audits):
        return ParsedParityAudit(
            "INCONCLUSIVE", None, None, None, 0, None, "incomplete_isolated_pair"
        )
    valid_slots = [
        index
        for index, audit in enumerate(audits, start=1)
        if audit.trace_status == "VALID"
    ]
    invalid_slots = [
        index
        for index, audit in enumerate(audits, start=1)
        if audit.trace_status == "INVALID"
    ]
    confidence = min(audit.confidence for audit in audits)
    if len(valid_slots) == 1 and len(invalid_slots) == 1:
        valid = audits[valid_slots[0] - 1]
        invalid = audits[invalid_slots[0] - 1]
        return ParsedParityAudit(
            pair_status="ONE_VALID",
            presented_valid_trace=valid_slots[0],
            effect=valid.effect,
            option=valid.option,
            confidence=confidence,
            first_flaw=invalid.flaw_detail,
            parse_error=None,
        )
    if len(invalid_slots) == 2:
        return ParsedParityAudit(
            pair_status="BOTH_INVALID",
            presented_valid_trace=None,
            effect=None,
            option=None,
            confidence=confidence,
            first_flaw=" | ".join(
                audit.flaw_detail or "unspecified" for audit in audits
            ),
            parse_error=None,
        )
    return ParsedParityAudit(
        pair_status="INCONCLUSIVE",
        presented_valid_trace=None,
        effect=None,
        option=None,
        confidence=confidence,
        first_flaw="isolated verdicts do not establish exactly one valid trace",
        parse_error=None,
    )


def build_commitment_conditioned_proof_audit_prompt_v8(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    trace_view: str,
    private_stage0_response: str,
) -> str:
    if trace_view not in ISOLATED_TRACE_VIEWS:
        raise ValueError(f"Unknown isolated trace view: {trace_view}")
    del first_differing_step
    trace = trace_1 if trace_view == "trace_1" else trace_2
    return (
        "Act as a falsification-first proof-obligation auditor. Before this audit you made "
        "an independent Stage-0 attempt. It is included only as a private, frozen "
        "precommitment and may itself be wrong. Do not copy it or accept/reject the new trace "
        "merely because their final choices agree. The sibling trace, pair existence, author, "
        "model identity, sealed validity/effect/option, votes, other audits, and gold answer "
        "are hidden.\n\n"
        "Audit the one shown trace by actively trying to break it. COUNTERTEST must substitute "
        "a value, check a premise or unit, apply a boundary case, or derive the relevant rule "
        "independently. RECOMPUTATION must report the concrete result of that test. A generic "
        "claim such as 'looks correct' is not a proof obligation. Mark SURVIVES only if the "
        "countertest leaves every local step sound and the trace directly establishes exactly "
        "one signed option effect. Mark BREAKS when the test exposes a first fatal flaw. Mark "
        "UNCERTAIN if the needed premise or mapping cannot be verified.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private frozen Stage-0 attempt (untrusted precommitment):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Atomic rule to verify:\n{rule}\n"
        f"Single trace under audit:\n{trace}\n\n"
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


_PROOF_AUDIT_PATTERNS = {
    "trace_status": _field("TRACE_STATUS", r"(VALID|INVALID|INCONCLUSIVE)"),
    "countertest": _field("COUNTERTEST"),
    "countertest_result": _field(
        "COUNTERTEST_RESULT", r"(SURVIVES|BREAKS|UNCERTAIN)"
    ),
    "recomputation": _field("RECOMPUTATION"),
    "commitment_relation": _field(
        "COMMITMENT_RELATION", r"(CONSISTENT|CONFLICTS|UNRELATED|UNCERTAIN)"
    ),
    "effect": _field("EFFECT", r"(ELIMINATES|SUPPORTS|NONE)"),
    "option": _field("OPTION", r"\(?([A-Z]|NONE)\)?"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
    "flaw_code": _field(
        "FLAW_CODE",
        r"(RULE_MISAPPLIED|ARITHMETIC|UNSUPPORTED_PREMISE|QUESTION_MISMATCH|OTHER|UNCERTAIN|NONE)",
    ),
    "flaw_detail": _field("FLAW_DETAIL"),
}


def parse_commitment_conditioned_proof_audit_output_v8(
    text: str, option_labels: Sequence[str]
) -> ParsedProofObligationAudit:
    values, error = _unique_fields(text, _PROOF_AUDIT_PATTERNS)
    if error is not None:
        return ParsedProofObligationAudit(
            "INCONCLUSIVE", None, None, 0, "UNCERTAIN", None,
            None, "UNCERTAIN", None, "UNCERTAIN", error
        )
    status = values["trace_status"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    confidence = int(values["confidence"])
    flaw_code = values["flaw_code"].upper()
    flaw_detail = values["flaw_detail"].strip()
    countertest = values["countertest"].strip()
    countertest_result = values["countertest_result"].upper()
    recomputation = values["recomputation"].strip()
    commitment_relation = values["commitment_relation"].upper()
    effect_is_none = effect_value == "NONE"
    option_is_none = option_value == "NONE"
    detail_is_none = flaw_detail.upper() == "NONE"
    proof_is_empty = any(
        not value or value.upper() == "NONE"
        for value in (countertest, recomputation)
    )
    if proof_is_empty:
        error = "missing_proof_obligation"
    elif effect_is_none != option_is_none:
        error = "partial_effect_reconstruction"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    elif status == "VALID" and (
        countertest_result != "SURVIVES"
        or effect_is_none
        or flaw_code != "NONE"
        or not detail_is_none
    ):
        error = "invalid_valid_proof_branch"
    elif status == "INVALID" and (
        countertest_result != "BREAKS"
        or not effect_is_none
        or flaw_code not in ISOLATED_FLAW_CODES
        or detail_is_none
    ):
        error = "invalid_invalid_proof_branch"
    elif status == "INCONCLUSIVE" and (
        countertest_result != "UNCERTAIN"
        or not effect_is_none
        or flaw_code != "UNCERTAIN"
        or detail_is_none
        or confidence > 50
    ):
        error = "invalid_inconclusive_proof_branch"
    else:
        error = None
    if error is not None:
        return ParsedProofObligationAudit(
            "INCONCLUSIVE", None, None, 0, "UNCERTAIN", None,
            None, "UNCERTAIN", None, "UNCERTAIN", error
        )
    return ParsedProofObligationAudit(
        trace_status=status,
        effect=None if effect_is_none else effect_value,
        option=None if option_is_none else option_value,
        confidence=confidence,
        flaw_code=flaw_code,
        flaw_detail=None if detail_is_none else flaw_detail,
        countertest=countertest,
        countertest_result=countertest_result,
        recomputation=recomputation,
        commitment_relation=commitment_relation,
        parse_error=None,
    )


def build_commitment_conditioned_pair_audit_prompt_v8_ablation(
    question: FalsificationQuestion,
    rule: str,
    trace_1: str,
    trace_2: str,
    first_differing_step: str,
    orientation: str,
    private_stage0_response: str,
) -> str:
    if orientation not in PARITY_ORIENTATIONS:
        raise ValueError(f"Unknown counterfactual parity orientation: {orientation}")
    del first_differing_step
    shown_1, shown_2 = (
        (trace_1, trace_2)
        if orientation == "canonical"
        else (trace_2, trace_1)
    )
    return (
        "Act as a falsification-first pairwise proof-obligation auditor. Before this "
        "audit you made an independent Stage-0 attempt. It is included only as a private, "
        "frozen precommitment and may itself be wrong. Do not copy it or choose a trace "
        "merely because its final choice agrees. The author, model identity, sealed "
        "validity/effect/option, claimed first difference, votes, other audits, and gold "
        "answer are hidden. This ablation deliberately shows both traces.\n\n"
        "Actively try to break both traces under the supplied atomic rule. COUNTERTEST "
        "must substitute a value, check a premise or unit, apply a boundary case, or "
        "derive the relevant rule independently. RECOMPUTATION must report the concrete "
        "result. Return ONE_VALID only when exactly one trace survives and the other has "
        "a concrete first fatal flaw; then reconstruct exactly one signed option effect "
        "from the surviving trace. Do not infer all remaining options from uniqueness. "
        "Return BOTH_INVALID when both break, otherwise return INCONCLUSIVE.\n\n"
        f"Question:\n{question.question}\n\nOptions:\n{_options_text(question)}\n\n"
        "Private frozen Stage-0 attempt (untrusted precommitment):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Atomic rule to verify:\n{rule}\n"
        f"Presented trace 1:\n{shown_1}\n"
        f"Presented trace 2:\n{shown_2}\n\n"
        "Use one internally consistent branch:\n"
        "- ONE_VALID: result ONE_SURVIVES_ONE_BREAKS; complete VALID_TRACE, EFFECT, "
        "OPTION, and a concrete FIRST_FLAW.\n"
        "- BOTH_INVALID: result BOTH_BREAK; VALID_TRACE, EFFECT, and OPTION NONE; "
        "concrete FIRST_FLAW.\n"
        "- INCONCLUSIVE: result UNCERTAIN; VALID_TRACE, EFFECT, and OPTION NONE; "
        "confidence at most 50 and concrete uncertainty in FIRST_FLAW.\n\n"
        "Return exactly ten single-line fields with no surrounding prose.\n"
        "PAIR_STATUS: ONE_VALID|BOTH_INVALID|INCONCLUSIVE\n"
        "COUNTERTEST: concrete attempted falsification of both traces\n"
        "COUNTERTEST_RESULT: ONE_SURVIVES_ONE_BREAKS|BOTH_BREAK|UNCERTAIN\n"
        "RECOMPUTATION: independently derived concrete result\n"
        "COMMITMENT_RELATION: CONSISTENT|CONFLICTS|UNRELATED|UNCERTAIN\n"
        "VALID_TRACE: 1|2|NONE\n"
        "EFFECT: ELIMINATES|SUPPORTS|NONE\n"
        "OPTION: one option label or NONE\n"
        "CONFIDENCE: integer from 0 to 100\n"
        "FIRST_FLAW: first fatal error or concrete uncertainty"
    )


_PAIR_PROOF_AUDIT_PATTERNS = {
    "pair_status": _field(
        "PAIR_STATUS", r"(ONE_VALID|BOTH_INVALID|INCONCLUSIVE)"
    ),
    "countertest": _field("COUNTERTEST"),
    "countertest_result": _field(
        "COUNTERTEST_RESULT",
        r"(ONE_SURVIVES_ONE_BREAKS|BOTH_BREAK|UNCERTAIN)",
    ),
    "recomputation": _field("RECOMPUTATION"),
    "commitment_relation": _field(
        "COMMITMENT_RELATION", r"(CONSISTENT|CONFLICTS|UNRELATED|UNCERTAIN)"
    ),
    "valid_trace": _field("VALID_TRACE", r"(1|2|NONE)"),
    "effect": _field("EFFECT", r"(ELIMINATES|SUPPORTS|NONE)"),
    "option": _field("OPTION", r"\(?([A-Z]|NONE)\)?"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
    "first_flaw": _field("FIRST_FLAW"),
}


def parse_commitment_conditioned_pair_audit_output_v8_ablation(
    text: str, option_labels: Sequence[str]
) -> ParsedPairProofObligationAudit:
    values, error = _unique_fields(text, _PAIR_PROOF_AUDIT_PATTERNS)
    if error is not None:
        return ParsedPairProofObligationAudit(
            "INCONCLUSIVE", None, None, None, 0, None,
            None, "UNCERTAIN", None, "UNCERTAIN", error
        )
    status = values["pair_status"].upper()
    result = values["countertest_result"].upper()
    valid_value = values["valid_trace"].upper()
    effect_value = values["effect"].upper()
    option_value = values["option"].upper()
    confidence = int(values["confidence"])
    first_flaw = values["first_flaw"].strip()
    countertest = values["countertest"].strip()
    recomputation = values["recomputation"].strip()
    commitment_relation = values["commitment_relation"].upper()
    none_pattern = (
        valid_value == "NONE",
        effect_value == "NONE",
        option_value == "NONE",
    )
    proof_is_empty = any(
        not value or value.upper() == "NONE"
        for value in (countertest, recomputation, first_flaw)
    )
    if proof_is_empty:
        error = "missing_pair_proof_obligation"
    elif option_value != "NONE" and option_value not in {
        str(label).upper() for label in option_labels
    }:
        error = "option_outside_set"
    elif status == "ONE_VALID" and (
        result != "ONE_SURVIVES_ONE_BREAKS" or any(none_pattern)
    ):
        error = "invalid_one_valid_proof_branch"
    elif status == "BOTH_INVALID" and (
        result != "BOTH_BREAK" or not all(none_pattern)
    ):
        error = "invalid_both_invalid_proof_branch"
    elif status == "INCONCLUSIVE" and (
        result != "UNCERTAIN" or not all(none_pattern) or confidence > 50
    ):
        error = "invalid_inconclusive_pair_proof_branch"
    else:
        error = None
    if error is not None:
        return ParsedPairProofObligationAudit(
            "INCONCLUSIVE", None, None, None, 0, None,
            None, "UNCERTAIN", None, "UNCERTAIN", error
        )
    return ParsedPairProofObligationAudit(
        pair_status=status,
        presented_valid_trace=(
            None if valid_value == "NONE" else int(valid_value)
        ),
        effect=None if effect_value == "NONE" else effect_value,
        option=None if option_value == "NONE" else option_value,
        confidence=confidence,
        first_flaw=first_flaw,
        countertest=countertest,
        countertest_result=result,
        recomputation=recomputation,
        commitment_relation=commitment_relation,
        parse_error=None,
    )
