from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUBJECT_TASKS: dict[str, list[str]] = {
    "English": [
        "2010-2013_English_MCQs",
        "2010-2022_English_Fill_in_Blanks",
        "2012-2022_English_Cloze_Test",
        "2010-2022_English_Reading_Comp",
    ],
    "Math": [
        "2010-2022_Math_I_MCQs",
        "2010-2022_Math_II_MCQs",
    ],
    "Chinese": [
        "2010-2022_Chinese_Modern_Lit",
        "2010-2022_Chinese_Lang_and_Usage_MCQs",
    ],
    "Physics": ["2010-2022_Physics_MCQs"],
    "Chemistry": ["2010-2022_Chemistry_MCQs"],
    "Biology": ["2010-2022_Biology_MCQs"],
    "History": ["2010-2022_History_MCQs"],
    "Geography": ["2010-2022_Geography_MCQs"],
    "Politics": ["2010-2022_Political_Science_MCQs"],
}

SUBJECT_ORDER = list(SUBJECT_TASKS)
TASK_TO_SUBJECT = {
    task: subject for subject, tasks in SUBJECT_TASKS.items() for task in tasks
}
RESULT_NAME_RE = re.compile(r"^(.+)_((?:2010|2012)-\d{4}_.+)$")


@dataclass(frozen=True)
class Score:
    correct_score: float
    total_score: float
    question_num: float
    empty_answer_num: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_score / self.total_score if self.total_score else 0.0


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_result_name(path: Path) -> tuple[str, str] | None:
    stem = path.stem if path.suffix == ".json" else path.name
    match = RESULT_NAME_RE.match(stem)
    if not match:
        return None
    model_name, keyword = match.groups()
    if keyword not in TASK_TO_SUBJECT:
        return None
    return model_name, keyword


def discover_result_models(data_dir: Path) -> list[str]:
    models: set[str] = set()
    for path in data_dir.iterdir():
        if path.name in {"Objective_Questions", "Subjective_Questions"}:
            continue
        parsed = parse_result_name(path)
        if parsed:
            models.add(parsed[0])
    return sorted(models)


def discover_local_model_names(models_dir: Path) -> set[str]:
    if not models_dir.exists():
        return set()
    return {path.name for path in models_dir.iterdir() if path.is_dir()}


def filter_local_models(model_names: Iterable[str], models_dir: Path) -> list[str]:
    local_names = discover_local_model_names(models_dir)
    return [name for name in model_names if name in local_names]


def load_result_payload(data_dir: Path, model_name: str, keyword: str) -> dict[str, Any] | None:
    merged_file = data_dir / f"{model_name}_{keyword}.json"
    if merged_file.exists():
        return read_json(merged_file)

    split_dir = data_dir / f"{model_name}_{keyword}"
    if not split_dir.is_dir():
        return None

    examples: list[dict[str, Any]] = []
    for split_file in sorted(split_dir.glob("*.json")):
        payload = read_json(split_file)
        examples.extend(payload.get("example", []))
    if not examples:
        return None
    return {"keyword": keyword, "model_name": model_name, "example": examples}


def normalize_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        return [cleaned]
    return [str(value).strip()]


def reextract_choices(model_output: str, answer_len: int) -> list[str]:
    if not model_output:
        return []
    tail = model_output[-800:]
    answer_pos = max(tail.rfind("【答案】"), tail.lower().rfind("answer"))
    search_space = tail[answer_pos:] if answer_pos >= 0 else tail[-200:]
    choices = re.findall(r"[A-G]", search_space)
    if answer_len <= 1:
        return [choices[-1]] if choices else []
    return choices[:answer_len]


def get_standard_answer(example: dict[str, Any]) -> list[str]:
    return normalize_answer_list(example.get("standard_answer", example.get("answer", [])))


def get_model_answer(
    example: dict[str, Any],
    answer_len: int,
    reextract_empty: bool = False,
) -> list[str]:
    model_answer = normalize_answer_list(example.get("model_answer", []))
    if not model_answer and reextract_empty:
        model_answer = reextract_choices(str(example.get("model_output", "")), answer_len)
    if len(model_answer) != answer_len:
        return ["Z"] * answer_len
    return model_answer


def score_examples(
    keyword: str,
    examples: list[dict[str, Any]],
    reextract_empty: bool = False,
) -> Score:
    total_score = 0.0
    correct_score = 0.0
    question_num = 0.0
    empty_answer_num = 0

    for example in examples:
        standard_answer = get_standard_answer(example)
        if not example.get("model_answer"):
            empty_answer_num += 1
        model_answer = get_model_answer(example, len(standard_answer), reextract_empty)
        score = float(example.get("score", 0.0))

        if keyword == "2010-2022_Physics_MCQs":
            total_score += len(standard_answer) * score
            for idx, expected in enumerate(standard_answer):
                predicted = model_answer[idx]
                if predicted == expected:
                    correct_score += 6
                else:
                    has_wrong_choice = any(choice not in expected for choice in predicted)
                    correct_score += 0 if has_wrong_choice else 3
        else:
            total_score += len(standard_answer) * score
            for idx, expected in enumerate(standard_answer):
                if model_answer[idx] == expected:
                    correct_score += score

        question_num += len(standard_answer)

    return Score(correct_score, total_score, question_num, empty_answer_num)


def build_gaokao_scores(
    data_dir: Path,
    model_names: list[str] | None = None,
    reextract_empty: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model_names is None:
        model_names = discover_result_models(data_dir)

    task_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        subject_scores: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        for subject in SUBJECT_ORDER:
            for keyword in SUBJECT_TASKS[subject]:
                payload = load_result_payload(data_dir, model_name, keyword)
                if payload is None:
                    continue
                score = score_examples(
                    keyword,
                    payload.get("example", []),
                    reextract_empty=reextract_empty,
                )
                task_rows.append(
                    {
                        "model": model_name,
                        "subject": subject,
                        "task": keyword,
                        "correct_score": score.correct_score,
                        "total_score": score.total_score,
                        "question_num": score.question_num,
                        "empty_answer_num": score.empty_answer_num,
                        "accuracy": score.accuracy,
                    }
                )
                subject_scores[subject][0] += score.correct_score
                subject_scores[subject][1] += score.total_score
                subject_scores[subject][2] += score.question_num
                subject_scores[subject][3] += score.empty_answer_num

        for subject in SUBJECT_ORDER:
            correct, total, q_num, empty = subject_scores[subject]
            if total <= 0:
                continue
            subject_rows.append(
                {
                    "model": model_name,
                    "subject": subject,
                    "correct_score": correct,
                    "total_score": total,
                    "question_num": q_num,
                    "empty_answer_num": int(empty),
                    "accuracy": correct / total,
                }
            )

    return task_rows, subject_rows


def build_subject_winners(
    subject_rows: list[dict[str, Any]],
    model_to_index: dict[str, int],
    tie_tol: float = 1e-12,
) -> dict[str, dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        by_subject[row["subject"]].append(row)

    winners: dict[str, dict[str, Any]] = {}
    for subject in SUBJECT_ORDER:
        rows = by_subject.get(subject, [])
        if not rows:
            continue
        best_acc = max(float(row["accuracy"]) for row in rows)
        best_rows = [
            row for row in rows if abs(float(row["accuracy"]) - best_acc) <= tie_tol
        ]
        best_rows = sorted(best_rows, key=lambda row: model_to_index[row["model"]])
        selected = best_rows[0]
        winners[subject] = {
            "subject": subject,
            "best_accuracy": best_acc,
            "best_models": [
                {
                    "model": row["model"],
                    "model_index": model_to_index[row["model"]],
                    "accuracy": float(row["accuracy"]),
                }
                for row in best_rows
            ],
            "selected_model": selected["model"],
            "selected_model_index": model_to_index[selected["model"]],
        }
    return winners


def load_objective_questions(objective_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(objective_dir.glob("*.json")):
        payload = read_json(path)
        keyword = payload.get("keyword") or payload.get("keywords") or path.stem
        subject = TASK_TO_SUBJECT.get(keyword)
        if subject is None:
            continue
        for example in payload.get("example", []):
            question_id = f"{keyword}:{example.get('index')}"
            rows.append(
                {
                    "id": question_id,
                    "task": keyword,
                    "subject": subject,
                    "index": example.get("index"),
                    "year": example.get("year"),
                    "category": example.get("category"),
                    "question": str(example.get("question", "")).strip(),
                    "answer": get_standard_answer(example),
                }
            )
    return rows


def format_router_text(question: str, options: list[str] | None = None) -> str:
    if options:
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)
        )
        return f"Question:\n{question.strip()}\nOptions:\n{option_text}"
    return f"Question:\n{question.strip()}"


def make_router_samples(
    objective_dir: Path,
    winners: dict[str, dict[str, Any]],
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    winning_model_indices = sorted(
        {int(winner["selected_model_index"]) for winner in winners.values()}
    )
    model_index_to_route_label = {
        model_index: label for label, model_index in enumerate(winning_model_indices)
    }
    route_label_to_model_index = {
        label: model_index for model_index, label in model_index_to_route_label.items()
    }

    samples: list[dict[str, Any]] = []
    for row in load_objective_questions(objective_dir):
        winner = winners[row["subject"]]
        target_model_index = int(winner["selected_model_index"])
        route_label = model_index_to_route_label[target_model_index]
        target_model = model_names[target_model_index]
        samples.append(
            {
                **row,
                "text": format_router_text(row["question"]),
                "target_model": target_model,
                "target_model_index": target_model_index,
                "label": route_label,
            }
        )

    label_manifest = {
        "num_route_labels": len(winning_model_indices),
        "model_index_to_route_label": {
            str(key): value for key, value in model_index_to_route_label.items()
        },
        "route_label_to_model_index": {
            str(key): value for key, value in route_label_to_model_index.items()
        },
        "route_label_to_model": {
            str(label): model_names[model_index]
            for label, model_index in route_label_to_model_index.items()
        },
    }
    return samples, label_manifest


def make_subject_router_samples(
    objective_dir: Path,
    winners: dict[str, dict[str, Any]],
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subject_to_route_label = {subject: idx for idx, subject in enumerate(SUBJECT_ORDER)}
    route_label_to_subject = {
        label: subject for subject, label in subject_to_route_label.items()
    }
    subject_to_model = {
        subject: winners[subject]["selected_model"]
        for subject in SUBJECT_ORDER
        if subject in winners
    }
    subject_to_model_index = {
        subject: int(winners[subject]["selected_model_index"])
        for subject in SUBJECT_ORDER
        if subject in winners
    }

    samples: list[dict[str, Any]] = []
    for row in load_objective_questions(objective_dir):
        subject = row["subject"]
        label = subject_to_route_label[subject]
        samples.append(
            {
                **row,
                "text": format_router_text(row["question"]),
                "target_subject": subject,
                "target_model": subject_to_model.get(subject),
                "target_model_index": subject_to_model_index.get(subject),
                "label": label,
            }
        )

    label_manifest = {
        "label_mode": "subject",
        "num_route_labels": len(SUBJECT_ORDER),
        "subject_to_route_label": subject_to_route_label,
        "route_label_to_subject": {
            str(label): subject for label, subject in route_label_to_subject.items()
        },
        "subject_to_model": subject_to_model,
        "subject_to_model_index": subject_to_model_index,
        "route_label_to_model": {
            str(label): subject_to_model.get(subject)
            for label, subject in route_label_to_subject.items()
        },
        "route_label_to_model_index": {
            str(label): subject_to_model_index.get(subject)
            for label, subject in route_label_to_subject.items()
        },
    }
    return samples, label_manifest


def _gaokao_scalar_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(item).strip() for item in value)
    return str(value).strip()


def score_example_slots(
    keyword: str,
    example: dict[str, Any],
    reextract_empty: bool = False,
) -> list[dict[str, Any]]:
    standard_answer = get_standard_answer(example)
    model_answer = get_model_answer(example, len(standard_answer), reextract_empty)
    if not standard_answer:
        return []

    max_points = float(example.get("score", 1.0) or 1.0)
    rows: list[dict[str, Any]] = []
    for answer_idx, expected_raw in enumerate(standard_answer):
        predicted_raw = model_answer[answer_idx] if answer_idx < len(model_answer) else "Z"
        expected = _gaokao_scalar_answer(expected_raw)
        predicted = _gaokao_scalar_answer(predicted_raw)
        correct_points = 0.0
        has_partial_credit = False

        if keyword == "2010-2022_Physics_MCQs":
            if predicted == expected:
                correct_points = 6.0
            else:
                has_wrong_choice = any(choice not in expected for choice in predicted)
                if predicted and not has_wrong_choice:
                    correct_points = 3.0
                    has_partial_credit = True
            denominator = max(max_points, 1.0)
        else:
            if predicted == expected:
                correct_points = max_points
            denominator = max(max_points, 1.0)

        normalized_score = max(0.0, min(1.0, correct_points / denominator))
        rows.append(
            {
                "answer_idx": answer_idx,
                "expected": expected,
                "predicted": predicted,
                "score": normalized_score,
                "is_correct": bool(normalized_score >= 1.0 - 1e-12),
                "has_partial_credit": has_partial_credit,
                "correct_points": correct_points,
                "total_points": denominator,
            }
        )
    return rows


def load_gaokao2010_2022_full_predictions(
    data_dir: Path,
    model_names: list[str] | None = None,
    reextract_empty: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load GAOKAO-Bench-2010-2022 objective results as a model -> row matrix.

    Rows are answer slots rather than whole exams so binary routing metrics keep
    the same weighted choice granularity used by the GAOKAO objective scorer.
    Physics partial credit is preserved in the numeric ``score`` field for
    score-aware methods; boolean methods count only full-credit answers.
    """
    if model_names is None:
        model_names = discover_result_models(data_dir)

    full: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name in sorted(model_names):
        rows_by_id: dict[str, dict[str, Any]] = {}
        for subject in SUBJECT_ORDER:
            for keyword in SUBJECT_TASKS[subject]:
                payload = load_result_payload(data_dir, model_name, keyword)
                if payload is None:
                    continue
                for example in payload.get("example", []):
                    index = example.get("index")
                    for scored in score_example_slots(keyword, example, reextract_empty):
                        rid = f"{keyword}:{index}:{scored['answer_idx']}"
                        rows_by_id[rid] = {
                            "id": rid,
                            "benchmark": "gaokao_bench_2010_2022",
                            "subject": subject,
                            "task": keyword,
                            "category": example.get("category"),
                            "year": example.get("year"),
                            "index": index,
                            "question": str(example.get("question", "")).strip(),
                            "answer": scored["expected"],
                            "target": scored["expected"],
                            "pred": scored["predicted"],
                            "prediction": scored["predicted"],
                            "response": example.get("model_output", ""),
                            "model_outputs": example.get("model_output", ""),
                            "score": scored["score"],
                            "is_correct": scored["is_correct"],
                            "has_partial_credit": scored["has_partial_credit"],
                            "correct_points": scored["correct_points"],
                            "total_points": scored["total_points"],
                            "empty_answer": not bool(example.get("model_answer")),
                        }
        if rows_by_id:
            full[model_name] = rows_by_id
    return full
