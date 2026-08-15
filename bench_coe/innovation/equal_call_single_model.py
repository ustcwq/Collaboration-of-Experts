from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from .blind_falsification_jury import FalsificationQuestion


_OUTPUT_RE = re.compile(
    r"\AREASON:\s*([^\n]+)\nFINAL:\s*([A-Z])\s*\Z",
    re.ASCII,
)


def _question_block(question: FalsificationQuestion) -> str:
    options = "\n".join(
        f"{label}. {option}"
        for label, option in zip(question.option_labels, question.options, strict=True)
    )
    return f"QUESTION:\n{question.question}\n\nOPTIONS:\n{options}"


def build_independent_solution_prompt(question: FalsificationQuestion) -> str:
    allowed = ", ".join(question.option_labels)
    return (
        "Solve this multiple-choice question independently. Reason from the question and "
        "options only. Do not assume that another solver's answer exists. Check the most "
        "plausible alternative before committing.\n\n"
        f"{_question_block(question)}\n\n"
        "Return exactly two lines. Keep REASON on one line and choose one allowed label.\n"
        "REASON: <concise decisive reasoning>\n"
        f"FINAL: <one of {allowed}>"
    )


def build_self_revision_prompt(
    question: FalsificationQuestion, initial_output: str
) -> str:
    allowed = ", ".join(question.option_labels)
    return (
        "Re-solve this multiple-choice question as a strict reviewer of your own earlier "
        "attempt. Locate its first material error if one exists, recompute the decisive "
        "step, and retain the answer only when it survives that check. The earlier attempt "
        "is untrusted evidence, not an authority.\n\n"
        f"{_question_block(question)}\n\n"
        "EARLIER_ATTEMPT:\n"
        f"{initial_output.strip()}\n\n"
        "Return exactly two lines. Keep REASON on one line and choose one allowed label.\n"
        "REASON: <concise corrected reasoning>\n"
        f"FINAL: <one of {allowed}>"
    )


def parse_equal_call_answer(
    raw_output: str, option_labels: Sequence[str]
) -> tuple[str | None, str | None, str | None]:
    match = _OUTPUT_RE.fullmatch(raw_output.strip())
    if match is None:
        return None, None, "format_mismatch"
    reason = match.group(1).strip()
    answer = match.group(2)
    if not reason:
        return None, None, "empty_reason"
    if answer not in option_labels:
        return None, reason, "answer_outside_option_set"
    return answer, reason, None


def aggregate_equal_call_answers(
    answers: Sequence[str | None], option_labels: Sequence[str]
) -> tuple[str | None, dict[str, int]]:
    allowed = set(option_labels)
    valid = [answer for answer in answers if answer in allowed]
    counts = Counter(str(answer) for answer in valid)
    if not counts:
        return None, {label: 0 for label in option_labels}
    maximum = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == maximum}
    first_tied = next(answer for answer in valid if answer in tied)
    return first_tied, {label: int(counts[label]) for label in option_labels}
