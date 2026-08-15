from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .blind_falsification_jury import FalsificationQuestion
from .sealed_counterfactual_parity import bounded_private_response


PRESENTED_SIDES = ("LEFT", "RIGHT")


@dataclass(frozen=True)
class CandidatePairAssignment:
    first: str
    second: str
    author_answer: str | None
    reason: str


@dataclass(frozen=True)
class ParsedDiagnosticProbe:
    probe: str | None
    outcome_1: str | None
    outcome_2: str | None
    map_outcome_1: str | None
    map_outcome_2: str | None
    bridge_1: str | None
    bridge_2: str | None
    confidence: int
    parse_error: str | None

    @property
    def abstained(self) -> bool:
        return self.parse_error is None and self.probe is None


@dataclass(frozen=True)
class PresentedDiagnosticProbe:
    probe: str
    left_text: str
    right_text: str
    left_candidate: str
    right_candidate: str
    left_authored_outcome: int
    post_commit_permutation_applied: bool


@dataclass(frozen=True)
class ParsedProbeCheck:
    outcome_side: str | None
    derivation: str | None
    confidence: int
    parse_error: str | None

    @property
    def uncertain(self) -> bool:
        return self.parse_error is None and self.outcome_side is None


def _stable_byte(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()[0]


def presented_left_authored_outcome(
    seed: int, question_id: str, author_id: str
) -> int:
    return 1 + (_stable_byte(int(seed), question_id, author_id, "sdb-left") & 1)


def assign_candidate_pairs(
    question: FalsificationQuestion,
    author_ids: Sequence[str],
    base_answers: Mapping[str, str | None],
    expert_order: Sequence[str],
) -> dict[str, CandidatePairAssignment]:
    """Assign query-local pairs while spreading challengers across the option set."""
    labels = tuple(question.option_labels)
    if len(labels) < 2:
        raise ValueError("A diagnostic bijection needs at least two options")
    if len(set(author_ids)) != len(author_ids):
        raise ValueError("Diagnostic authors must be unique")
    if not set(author_ids).issubset(base_answers):
        raise ValueError("Every diagnostic author needs a Stage-0 answer")

    votes = Counter(answer for answer in base_answers.values() if answer in labels)
    first_support = {
        label: next(
            (
                index
                for index, expert in enumerate(expert_order)
                if base_answers.get(expert) == label
            ),
            len(expert_order),
        )
        for label in labels
    }
    label_order = {label: index for index, label in enumerate(labels)}
    ranked = sorted(
        labels,
        key=lambda label: (
            -votes[label],
            first_support[label],
            label_order[label],
        ),
    )

    exposure = Counter()
    edge_exposure = Counter()
    result: dict[str, CandidatePairAssignment] = {}
    for author in author_ids:
        raw_answer = base_answers.get(author)
        author_answer = raw_answer if raw_answer in labels else None
        first = author_answer or ranked[0]
        alternatives = [label for label in ranked if label != first]
        second = min(
            alternatives,
            key=lambda label: (
                edge_exposure[tuple(sorted((first, label)))],
                exposure[label],
                ranked.index(label),
            ),
        )
        result[author] = CandidatePairAssignment(
            first=first,
            second=second,
            author_answer=author_answer,
            reason=(
                "author_stage0_vs_coverage_ranked_challenger"
                if author_answer is not None
                else "vote_ranked_pair_after_invalid_author_stage0"
            ),
        )
        exposure[first] += 1
        exposure[second] += 1
        edge_exposure[tuple(sorted((first, second)))] += 1
    return result


def _options_text(question: FalsificationQuestion) -> str:
    return "\n".join(
        f"({label}) {text}"
        for label, text in zip(
            question.option_labels, question.options, strict=True
        )
    )


def build_diagnostic_probe_prompt(
    question: FalsificationQuestion,
    private_stage0_response: str,
    assignment: CandidatePairAssignment,
) -> str:
    pair = {assignment.first, assignment.second}
    if len(pair) != 2 or not pair.issubset(question.option_labels):
        raise ValueError("Diagnostic candidate pair is outside the query option set")
    option_by_label = dict(
        zip(question.option_labels, question.options, strict=True)
    )
    return (
        "Act as the author of a sealed binary diagnostic instrument. You receive the "
        "original multiple-choice task, one private Stage-0 trace from your own solve, "
        "and exactly two assigned candidates. No other response, model identity, vote "
        "count, popularity, source score, or gold answer is available.\n\n"
        "Construct one smaller, self-contained diagnostic question whose resolution "
        "distinguishes the assigned candidates. PROBE must contain every premise needed "
        "to solve it without the original task. It must not merely repeat the original "
        "question. OUTCOME_1 and OUTCOME_2 must be distinct, mutually exclusive answers "
        "to PROBE. Map each outcome bijectively to one assigned candidate. BRIDGE_1 and "
        "BRIDGE_2 must privately justify why each outcome implies its mapping. The "
        "bridges and mapping are sealed from checkers. Do not write option, answer, "
        "choice, or candidate labels in PROBE or either OUTCOME. A checker will see only "
        "PROBE and the two outcome texts after their positions are randomly renamed by "
        "code; it will not see this task, either assigned candidate, either bridge, your "
        "identity, or your private trace.\n\n"
        "If no rigorous self-contained binary diagnostic exists, abstain with NONE in "
        "all seven text/mapping fields and confidence at most 50.\n\n"
        f"Original question:\n{question.question}\n\n"
        f"Options:\n{_options_text(question)}\n\n"
        "Private Stage-0 trace (never shown to checkers):\n"
        f"{bounded_private_response(private_stage0_response)}\n\n"
        f"Assigned candidate X: ({assignment.first}) "
        f"{option_by_label[assignment.first]}\n"
        f"Assigned candidate Y: ({assignment.second}) "
        f"{option_by_label[assignment.second]}\n\n"
        "Return exactly eight single-line fields with no surrounding prose.\n"
        "PROBE: self-contained diagnostic question or NONE\n"
        "OUTCOME_1: first possible result or NONE\n"
        "OUTCOME_2: second possible result or NONE\n"
        "MAP_OUTCOME_1: one assigned option label or NONE\n"
        "MAP_OUTCOME_2: the other assigned option label or NONE\n"
        "BRIDGE_1: private implication from outcome 1 to its mapped candidate or NONE\n"
        "BRIDGE_2: private implication from outcome 2 to its mapped candidate or NONE\n"
        "CONFIDENCE: integer from 0 to 100"
    )


def _field(name: str, choices: str | None = None) -> re.Pattern[str]:
    value = choices if choices is not None else r"(\S.*?)"
    return re.compile(
        rf"^\s*{re.escape(name)}\s*:\s*{value}\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )


_PROBE_PATTERNS = {
    "probe": _field("PROBE"),
    "outcome_1": _field("OUTCOME_1"),
    "outcome_2": _field("OUTCOME_2"),
    "map_outcome_1": _field("MAP_OUTCOME_1", r"\(?([A-Z]|NONE)\)?"),
    "map_outcome_2": _field("MAP_OUTCOME_2", r"\(?([A-Z]|NONE)\)?"),
    "bridge_1": _field("BRIDGE_1"),
    "bridge_2": _field("BRIDGE_2"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
}

_VISIBLE_LABEL_LEAK = re.compile(
    r"\b(?:option|answer|choice|candidate)\s*\(?[A-Z]\)?\b",
    flags=re.IGNORECASE,
)


def _unique_fields(
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


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_diagnostic_probe_output(
    text: str,
    assignment: CandidatePairAssignment,
    original_question: str,
) -> ParsedDiagnosticProbe:
    values, error = _unique_fields(text, _PROBE_PATTERNS)
    if error is not None:
        return ParsedDiagnosticProbe(
            None, None, None, None, None, None, None, 0, error
        )
    confidence = int(values["confidence"])
    text_fields = (
        values["probe"],
        values["outcome_1"],
        values["outcome_2"],
        values["bridge_1"],
        values["bridge_2"],
    )
    mappings = (
        values["map_outcome_1"].upper(),
        values["map_outcome_2"].upper(),
    )
    all_none = all(value.upper() == "NONE" for value in (*text_fields, *mappings))
    any_none = any(value.upper() == "NONE" for value in (*text_fields, *mappings))
    if any_none and not all_none:
        error = "partial_abstention"
    elif all_none and confidence > 50:
        error = "overconfident_abstention"
    elif not all_none and set(mappings) != {assignment.first, assignment.second}:
        error = "mapping_is_not_assigned_bijection"
    elif not all_none and _normalized_text(values["outcome_1"]) == _normalized_text(
        values["outcome_2"]
    ):
        error = "identical_outcomes"
    elif not all_none and any(
        _VISIBLE_LABEL_LEAK.search(value)
        for value in (
            values["probe"],
            values["outcome_1"],
            values["outcome_2"],
        )
    ):
        error = "visible_candidate_label_leak"
    elif not all_none and len(_normalized_text(values["probe"])) < 20:
        error = "probe_too_short"
    elif not all_none and _normalized_text(values["probe"]) == _normalized_text(
        original_question
    ):
        error = "probe_repeats_original_question"
    else:
        error = None
    if error is not None:
        return ParsedDiagnosticProbe(
            None, None, None, None, None, None, None, 0, error
        )
    if all_none:
        return ParsedDiagnosticProbe(
            None, None, None, None, None, None, None, confidence, None
        )
    return ParsedDiagnosticProbe(
        probe=values["probe"],
        outcome_1=values["outcome_1"],
        outcome_2=values["outcome_2"],
        map_outcome_1=mappings[0],
        map_outcome_2=mappings[1],
        bridge_1=values["bridge_1"],
        bridge_2=values["bridge_2"],
        confidence=confidence,
        parse_error=None,
    )


def present_diagnostic_probe(
    parsed: ParsedDiagnosticProbe,
    left_authored_outcome: int,
) -> PresentedDiagnosticProbe:
    if parsed.parse_error is not None or parsed.abstained:
        raise ValueError("Only parsed non-abstaining probes can be presented")
    if left_authored_outcome not in (1, 2):
        raise ValueError("Presented outcome index must be 1 or 2")
    values = {
        1: (parsed.outcome_1, parsed.map_outcome_1),
        2: (parsed.outcome_2, parsed.map_outcome_2),
    }
    left_text, left_candidate = values[left_authored_outcome]
    right_text, right_candidate = values[3 - left_authored_outcome]
    if any(
        value is None
        for value in (parsed.probe, left_text, right_text, left_candidate, right_candidate)
    ):
        raise AssertionError("Parsed diagnostic probe lost a required field")
    return PresentedDiagnosticProbe(
        probe=str(parsed.probe),
        left_text=str(left_text),
        right_text=str(right_text),
        left_candidate=str(left_candidate),
        right_candidate=str(right_candidate),
        left_authored_outcome=left_authored_outcome,
        post_commit_permutation_applied=left_authored_outcome == 2,
    )


def build_blind_probe_check_prompt(
    probe: str,
    left_outcome: str,
    right_outcome: str,
) -> str:
    return (
        "Solve the isolated binary diagnostic below on its own terms. The original task, "
        "answer candidates, author, outcome-to-candidate mapping, private reasoning, votes, "
        "and gold label are unavailable. Do not guess that hidden context. Determine which "
        "presented outcome follows from the stated probe. If the probe lacks enough premises "
        "or neither outcome is justified, return UNCERTAIN with confidence at most 50.\n\n"
        f"PROBE:\n{probe}\n\n"
        f"LEFT OUTCOME:\n{left_outcome}\n\n"
        f"RIGHT OUTCOME:\n{right_outcome}\n\n"
        "Return exactly three single-line fields with no surrounding prose.\n"
        "OUTCOME: LEFT|RIGHT|UNCERTAIN\n"
        "DERIVATION: independent calculation or decisive fact; use NONE only for UNCERTAIN\n"
        "CONFIDENCE: integer from 0 to 100"
    )


_CHECK_PATTERNS = {
    "outcome": _field("OUTCOME", r"(LEFT|RIGHT|UNCERTAIN)"),
    "derivation": _field("DERIVATION"),
    "confidence": _field("CONFIDENCE", r"(100|[0-9]{1,2})"),
}


def parse_blind_probe_check_output(text: str) -> ParsedProbeCheck:
    values, error = _unique_fields(text, _CHECK_PATTERNS)
    if error is not None:
        return ParsedProbeCheck(None, None, 0, error)
    outcome = values["outcome"].upper()
    derivation = values["derivation"]
    confidence = int(values["confidence"])
    if outcome == "UNCERTAIN" and confidence > 50:
        error = "overconfident_uncertainty"
    elif outcome in PRESENTED_SIDES and derivation.upper() == "NONE":
        error = "selected_outcome_without_derivation"
    elif outcome == "UNCERTAIN" and derivation.upper() != "NONE":
        error = "uncertainty_with_asserted_derivation"
    else:
        error = None
    if error is not None:
        return ParsedProbeCheck(None, None, 0, error)
    return ParsedProbeCheck(
        outcome_side=None if outcome == "UNCERTAIN" else outcome,
        derivation=None if outcome == "UNCERTAIN" else derivation,
        confidence=confidence,
        parse_error=None,
    )


def reveal_probe_candidate(
    parsed_check: ParsedProbeCheck,
    presentation: PresentedDiagnosticProbe,
) -> tuple[str | None, str | None]:
    if parsed_check.parse_error is not None or parsed_check.outcome_side is None:
        return None, None
    if parsed_check.outcome_side == "LEFT":
        return presentation.left_candidate, presentation.right_candidate
    if parsed_check.outcome_side == "RIGHT":
        return presentation.right_candidate, presentation.left_candidate
    raise AssertionError("Parsed probe check contains an unknown side")
