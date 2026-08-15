from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .blind_falsification_jury import FalsificationQuestion


POINTWISE_ORIGINAL_METHOD = "PRePair (Jeong et al., BlackboxNLP 2025)"
POINTWISE_ADAPTATION_NAME = "prepair_style_order_audited_budget_matched_top3"
PAIRWISE_ORIENTATIONS = ("primary_left", "primary_right")

_POINTWISE_PATTERN = re.compile(r"\AANALYSIS: ([^\n]+)\Z")
_PAIRWISE_PATTERN = re.compile(
    r"\AREASON: ([^\n]+)\nWINNER: (LEFT|RIGHT|TIE)\Z"
)


def _option_lines(question: FalsificationQuestion) -> str:
    return "\n".join(
        f"{label}. {value}"
        for label, value in zip(
            question.option_labels, question.options, strict=True
        )
    )


def candidate_value(question: FalsificationQuestion, candidate: str) -> str:
    try:
        index = question.option_labels.index(candidate)
    except ValueError as error:
        raise ValueError("PRePair-style candidate is outside the option set") from error
    return question.options[index]


def rank_candidate_slate(
    question: FalsificationQuestion,
    base_answers: Mapping[str, str | None],
    expert_order: Sequence[str],
    max_challengers: int,
) -> tuple[str, ...]:
    """Select plurality and challengers without labels or cross-query identities."""

    if max_challengers <= 0:
        raise ValueError("PRePair-style control requires at least one challenger")
    if len(set(expert_order)) != len(expert_order):
        raise ValueError("PRePair-style expert order contains duplicates")
    unknown = set(base_answers).difference(expert_order)
    if unknown:
        raise ValueError(f"PRePair-style base answers contain unknown experts: {unknown}")
    counts = Counter(
        answer for answer in base_answers.values() if answer in question.option_labels
    )
    first_support = {
        candidate: min(
            (
                index
                for index, expert in enumerate(expert_order)
                if base_answers.get(expert) == candidate
            ),
            default=len(expert_order),
        )
        for candidate in question.option_labels
    }
    option_order = {
        candidate: index for index, candidate in enumerate(question.option_labels)
    }
    ranked = sorted(
        question.option_labels,
        key=lambda candidate: (
            -counts[candidate],
            first_support[candidate],
            option_order[candidate],
        ),
    )
    return tuple(ranked[: min(len(ranked), 1 + max_challengers)])


def candidate_vote_counts(
    question: FalsificationQuestion,
    base_answers: Mapping[str, str | None],
) -> dict[str, int]:
    counts = Counter(
        answer for answer in base_answers.values() if answer in question.option_labels
    )
    return {candidate: counts[candidate] for candidate in question.option_labels}


def build_pre_pair_pointwise_prompt(
    question: FalsificationQuestion, candidate: str
) -> str:
    value = candidate_value(question, candidate)
    return (
        "You are performing the pointwise-reasoning stage of an adapted PRePair "
        "evaluation. Assess only the specified candidate on its own merits. Do not "
        "compare it with another candidate, infer vote counts, or make a final pairwise "
        "choice. Check instruction fit, factual or logical validity, and the decisive "
        "calculation. No answer key or correctness signal is available.\n\n"
        f"Question:\n{question.question}\n\n"
        f"Answer choices:\n{_option_lines(question)}\n\n"
        f"Candidate under pointwise analysis: {candidate}. {value}\n\n"
        "Return exactly one nonempty line:\n"
        "ANALYSIS: <concise candidate-specific reasoning>"
    )


def parse_pre_pair_pointwise_output(raw_output: str) -> tuple[str | None, str | None]:
    match = _POINTWISE_PATTERN.fullmatch(raw_output.strip())
    if match is None:
        return None, "format_mismatch"
    analysis = match.group(1).strip()
    if not analysis:
        return None, "empty_analysis"
    return analysis, None


def build_pre_pair_pairwise_prompt(
    question: FalsificationQuestion,
    left_candidate: str,
    right_candidate: str,
    left_pointwise_output: str,
    right_pointwise_output: str,
) -> str:
    if left_candidate == right_candidate:
        raise ValueError("PRePair-style comparison requires distinct candidates")
    left_value = candidate_value(question, left_candidate)
    right_value = candidate_value(question, right_candidate)
    return (
        "You are performing the final pairwise stage of an adapted PRePair evaluation. "
        "Use the two independently generated pointwise analyses as untrusted reasoning "
        "aids, verify their claims against the question, and select the better candidate. "
        "Candidate order must not affect the decision. No answer key, expert identity, or "
        "vote count is available.\n\n"
        f"Question:\n{question.question}\n\n"
        f"Answer choices:\n{_option_lines(question)}\n\n"
        f"LEFT candidate: {left_candidate}. {left_value}\n"
        f"LEFT pointwise analysis:\n{left_pointwise_output}\n\n"
        f"RIGHT candidate: {right_candidate}. {right_value}\n"
        f"RIGHT pointwise analysis:\n{right_pointwise_output}\n\n"
        "Return exactly two nonempty lines:\n"
        "REASON: <concise comparison grounded in the question>\n"
        "WINNER: <LEFT, RIGHT, or TIE>"
    )


def parse_pre_pair_pairwise_output(
    raw_output: str,
) -> tuple[str | None, str | None, str | None]:
    match = _PAIRWISE_PATTERN.fullmatch(raw_output.strip())
    if match is None:
        return None, None, "format_mismatch"
    reason = match.group(1).strip()
    winner = match.group(2)
    if not reason:
        return None, None, "empty_reason"
    return winner, reason, None


def mapped_pairwise_winner(row: Mapping[str, Any]) -> str | None:
    if row.get("parse_error") is not None:
        return None
    winner = row.get("winner")
    if winner == "LEFT":
        return str(row["left_candidate"])
    if winner == "RIGHT":
        return str(row["right_candidate"])
    if winner == "TIE":
        return "TIE"
    return None


def aggregate_order_audited_pre_pair(
    question: FalsificationQuestion,
    slate: Sequence[str],
    models: Sequence[str],
    pairwise_rows: Sequence[Mapping[str, Any]],
    *,
    challenger_limit: int | None = None,
) -> dict[str, Any]:
    if not slate or any(candidate not in question.option_labels for candidate in slate):
        raise ValueError("PRePair-style slate is empty or outside the option set")
    if len(set(slate)) != len(slate):
        raise ValueError("PRePair-style slate contains duplicate candidates")
    if len(set(models)) != len(models):
        raise ValueError("PRePair-style model pool contains duplicates")
    primary = str(slate[0])
    challengers = tuple(str(value) for value in slate[1:])
    if challenger_limit is not None:
        if challenger_limit <= 0:
            raise ValueError("PRePair-style challenger limit must be positive")
        challengers = challengers[:challenger_limit]

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in pairwise_rows:
        if str(row.get("question_id")) != question.question_id:
            raise ValueError("PRePair-style aggregation crossed question identities")
        model = str(row.get("model"))
        challenger = str(row.get("challenger"))
        orientation = str(row.get("orientation"))
        if model not in models or challenger not in challengers:
            continue
        if orientation not in PAIRWISE_ORIENTATIONS:
            raise ValueError("PRePair-style aggregation saw an unknown orientation")
        key = (model, challenger)
        if orientation in grouped[key]:
            raise ValueError("Duplicate PRePair-style pairwise orientation")
        grouped[key][orientation] = row

    per_challenger: dict[str, dict[str, Any]] = {}
    for challenger in challengers:
        outcomes: dict[str, str] = {}
        for model in models:
            orientations = grouped.get((model, challenger), {})
            if set(orientations) != set(PAIRWISE_ORIENTATIONS):
                outcome = "ABSTAIN"
            else:
                mapped = [
                    mapped_pairwise_winner(orientations[orientation])
                    for orientation in PAIRWISE_ORIENTATIONS
                ]
                outcome = (
                    str(mapped[0])
                    if mapped[0] is not None and mapped[0] == mapped[1]
                    else "ABSTAIN"
                )
            if outcome not in {primary, challenger, "TIE", "ABSTAIN"}:
                outcome = "ABSTAIN"
            outcomes[model] = outcome
        primary_wins = sum(value == primary for value in outcomes.values())
        challenger_wins = sum(value == challenger for value in outcomes.values())
        ties = sum(value == "TIE" for value in outcomes.values())
        abstentions = sum(value == "ABSTAIN" for value in outcomes.values())
        decisive = primary_wins + challenger_wins
        per_challenger[challenger] = {
            "model_outcomes": outcomes,
            "primary_wins": primary_wins,
            "challenger_wins": challenger_wins,
            "ties": ties,
            "abstentions": abstentions,
            "decisive_judges": decisive,
            "challenger_margin": (
                (challenger_wins - primary_wins) / decisive if decisive else 0.0
            ),
        }

    eligible = [
        challenger
        for challenger in challengers
        if per_challenger[challenger]["challenger_wins"]
        > per_challenger[challenger]["primary_wins"]
    ]
    slate_order = {candidate: index for index, candidate in enumerate(slate)}
    if eligible:
        prediction = sorted(
            eligible,
            key=lambda challenger: (
                -float(per_challenger[challenger]["challenger_margin"]),
                -int(per_challenger[challenger]["challenger_wins"]),
                slate_order[challenger],
            ),
        )[0]
        fallback_reason = "order_consistent_challenger_majority"
    else:
        prediction = primary
        fallback_reason = "plurality_fallback_no_challenger_majority"
    return {
        "prediction": prediction,
        "primary": primary,
        "challengers": list(challengers),
        "per_challenger": per_challenger,
        "fallback_reason": fallback_reason,
        "tie_breaking": "challenger_margin_then_wins_then_frozen_slate_order",
    }
