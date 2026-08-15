from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .blind_falsification_jury import FalsificationQuestion


SDB_GENERATED_CONTROLS = (
    "cove_march_style_isolated_probe",
    "mapping_visible_equal_call",
    "direct_two_candidate_equal_call",
)


@dataclass(frozen=True)
class ParsedControlSelection:
    selected_candidate: str | None
    rejected_candidate: str | None
    derivation: str | None
    confidence: int
    parse_error: str | None

    @property
    def uncertain(self) -> bool:
        return self.parse_error is None and self.selected_candidate is None


def _options_text(question: FalsificationQuestion) -> str:
    return "\n".join(
        f"({label}) {text}"
        for label, text in zip(
            question.option_labels, question.options, strict=True
        )
    )


def _selection_contract() -> str:
    return (
        "Return exactly three single-line fields with no surrounding prose.\n"
        "SELECTION: one of the two shown candidate labels or UNCERTAIN\n"
        "DERIVATION: concrete independent derivation, or NONE only when uncertain\n"
        "CONFIDENCE: integer from 0 to 100"
    )


def build_cove_march_style_isolated_probe_prompt(
    question: FalsificationQuestion,
    probe: str,
    authored_outcome_1: str,
    authored_outcome_2: str,
    mapped_candidate_1: str,
    mapped_candidate_2: str,
) -> str:
    """Strong prior-art control: ordinary unsealed atomic verification."""
    pair = {mapped_candidate_1, mapped_candidate_2}
    if len(pair) != 2 or not pair.issubset(question.option_labels):
        raise ValueError("The isolated-probe control needs a valid candidate pair")
    return (
        "Act as an independent atomic verifier. The original task and an author's "
        "untrusted diagnostic instrument are shown. Independently solve the diagnostic, "
        "then use its disclosed outcome-to-candidate relation to select between exactly "
        "the two candidates. Do not defer to the author or use model identity, votes, "
        "source reliability, other responses, or a gold answer. This is an ordinary "
        "information-asymmetric verification control: there is no sealed mapping and no "
        "post-commit outcome renaming.\n\n"
        f"Original question:\n{question.question}\n\n"
        f"Options:\n{_options_text(question)}\n\n"
        f"Diagnostic question:\n{probe}\n"
        f"Diagnostic outcome 1: {authored_outcome_1}\n"
        f"Disclosed relation 1: outcome 1 implies ({mapped_candidate_1})\n"
        f"Diagnostic outcome 2: {authored_outcome_2}\n"
        f"Disclosed relation 2: outcome 2 implies ({mapped_candidate_2})\n\n"
        + _selection_contract()
    )


def build_mapping_visible_equal_call_prompt(
    probe: str,
    left_text: str,
    right_text: str,
    left_candidate: str,
    right_candidate: str,
) -> str:
    pair = {left_candidate, right_candidate}
    if len(pair) != 2:
        raise ValueError("The mapping-visible control needs two candidates")
    return (
        "Resolve the self-contained diagnostic question independently. The original "
        "task, author identity, private trace, votes, source reliability, and gold answer "
        "remain hidden. Unlike the sealed mechanism, this control discloses the candidate "
        "mapping before you reason. Treat that mapping as an untrusted routing rule, not "
        "as evidence.\n\n"
        f"Diagnostic question:\n{probe}\n\n"
        f"LEFT outcome: {left_text}\n"
        f"LEFT maps to candidate ({left_candidate})\n"
        f"RIGHT outcome: {right_text}\n"
        f"RIGHT maps to candidate ({right_candidate})\n\n"
        + _selection_contract()
    )


def build_direct_two_candidate_equal_call_prompt(
    question: FalsificationQuestion,
    first_candidate: str,
    second_candidate: str,
    first_rationale: str,
    second_rationale: str,
) -> str:
    pair = {first_candidate, second_candidate}
    if len(pair) != 2 or not pair.issubset(question.option_labels):
        raise ValueError("The direct-pair control needs a valid candidate pair")
    option_by_label = dict(
        zip(question.option_labels, question.options, strict=True)
    )
    return (
        "Act as a direct two-candidate deliberation judge. Solve the original task and "
        "compare exactly the two candidates. Two candidate-conditioned rationales from "
        "the shared author call are untrusted testimony; check every relevant step. Do "
        "not use author identity, votes, source reliability, other responses, or a gold "
        "answer.\n\n"
        f"Original question:\n{question.question}\n\n"
        f"Options:\n{_options_text(question)}\n\n"
        f"Candidate ({first_candidate}): {option_by_label[first_candidate]}\n"
        f"Untrusted rationale: {first_rationale}\n"
        f"Candidate ({second_candidate}): {option_by_label[second_candidate]}\n"
        f"Untrusted rationale: {second_rationale}\n\n"
        + _selection_contract()
    )


_CONTROL_PATTERN = re.compile(
    r"\A\s*SELECTION\s*:\s*\(?([A-Z]|UNCERTAIN)\)?\s*\n"
    r"DERIVATION\s*:\s*(\S.*?)\s*\n"
    r"CONFIDENCE\s*:\s*(100|[0-9]{1,2})\s*\Z",
    flags=re.IGNORECASE,
)


def parse_control_selection_output(
    text: str, candidate_pair: Sequence[str]
) -> ParsedControlSelection:
    pair = tuple(candidate_pair)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("A control selection needs exactly two distinct candidates")
    match = _CONTROL_PATTERN.fullmatch(text)
    if match is None:
        return ParsedControlSelection(None, None, None, 0, "invalid_output_contract")
    selection = match.group(1).upper()
    derivation = match.group(2).strip()
    confidence = int(match.group(3))
    if selection == "UNCERTAIN":
        if derivation.upper() != "NONE":
            return ParsedControlSelection(
                None, None, None, 0, "uncertain_with_derivation"
            )
        if confidence > 50:
            return ParsedControlSelection(
                None, None, None, 0, "overconfident_uncertainty"
            )
        return ParsedControlSelection(None, None, None, confidence, None)
    if selection not in pair:
        return ParsedControlSelection(
            None, None, None, 0, "selection_outside_assigned_pair"
        )
    if derivation.upper() == "NONE":
        return ParsedControlSelection(
            None, None, None, 0, "selection_without_derivation"
        )
    rejected = pair[1] if selection == pair[0] else pair[0]
    return ParsedControlSelection(
        selection, rejected, derivation, confidence, None
    )
