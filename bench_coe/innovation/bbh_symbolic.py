from __future__ import annotations

import ast
import re
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from .schema import Selection


_BOOLEAN_TOKEN_RE = re.compile(r"(?:True|False|not|and|or|\(|\)|\s)+\Z")
_ARITHMETIC_TOKEN_RE = re.compile(r"(?:\d|\s|\+|\-|\*|\(|\))+\Z")
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _eval_boolean_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Expression):
        return _eval_boolean_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_boolean_node(node.operand)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_eval_boolean_node(value) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    raise ValueError(f"Unsupported Boolean expression node: {type(node).__name__}")


def solve_boolean_expression(text: str) -> str | None:
    expression = text.strip()
    if expression.endswith(" is"):
        expression = expression[:-3].strip()
    if not expression or not _BOOLEAN_TOKEN_RE.fullmatch(expression):
        return None
    try:
        value = _eval_boolean_node(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError):
        return None
    return "True" if value else "False"


def _eval_arithmetic_node(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    raise ValueError(f"Unsupported arithmetic expression node: {type(node).__name__}")


def solve_multistep_arithmetic(text: str) -> str | None:
    expression = text.strip()
    if expression.endswith("="):
        expression = expression[:-1].strip()
    if not expression or not _ARITHMETIC_TOKEN_RE.fullmatch(expression):
        return None
    try:
        return str(_eval_arithmetic_node(ast.parse(expression, mode="eval")))
    except (SyntaxError, ValueError):
        return None


def solve_dyck_completion(text: str) -> str | None:
    marker = "Input:"
    if marker not in text:
        return None
    tokens = text.rsplit(marker, 1)[1].strip().split()
    if not tokens:
        return None
    stack: list[str] = []
    closing = set(_OPEN_TO_CLOSE.values())
    for token in tokens:
        if token in _OPEN_TO_CLOSE:
            stack.append(token)
        elif token in closing:
            if not stack or _OPEN_TO_CLOSE[stack.pop()] != token:
                return None
        else:
            return None
    return " ".join(_OPEN_TO_CLOSE[token] for token in reversed(stack))


def solve_word_sorting(text: str) -> str | None:
    marker = "List:"
    if marker not in text:
        return None
    words = text.rsplit(marker, 1)[1].strip().split()
    if not words or any(not re.fullmatch(r"[^\s]+", word) for word in words):
        return None
    return " ".join(sorted(words))


SOLVERS: Mapping[str, Callable[[str], str | None]] = {
    "boolean_expressions": solve_boolean_expression,
    "dyck_languages": solve_dyck_completion,
    "multistep_arithmetic_two": solve_multistep_arithmetic,
    "word_sorting": solve_word_sorting,
}


def solve_bbh(task: str, text: str) -> str | None:
    solver = SOLVERS.get(task)
    return solver(text) if solver is not None else None


def apply_symbolic_overrides(
    base: Sequence[Selection],
    metadata_by_question: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Selection], dict[str, int]]:
    result: list[Selection] = []
    counts = {task: 0 for task in SOLVERS}
    for row in base:
        metadata = metadata_by_question.get(row.question_id, {})
        task = str(metadata.get("task", ""))
        text = str(metadata.get("input", ""))
        answer = solve_bbh(task, text)
        if answer is None:
            result.append(row)
            continue
        features = dict(row.observable_features)
        features.update(
            {
                "bbh_symbolic_override": True,
                "bbh_symbolic_task": task,
                "bbh_symbolic_parser_succeeded": True,
                "bbh_symbolic_uses_target_labels": False,
            }
        )
        result.append(
            replace(
                row,
                selected_cluster_id=-1,
                selected_expert_id=f"symbolic::{task}",
                normalized_answer=answer,
                cluster_scores={"symbolic": 1.0},
                expert_scores={f"symbolic::{task}": 1.0},
                fallback_reason=None,
                observable_features=features,
                tie_breaking="deterministic_task_parser_then_v3_fallback",
            )
        )
        counts[task] += 1
    return result, counts


def canonical_task_answer(task: str, value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if task == "boolean_expressions":
        return text.casefold()
    if task == "multistep_arithmetic_two":
        return text.replace(",", "")
    return text
