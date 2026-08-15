from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bench_coe.gaokao_utils import dump_jsonl, write_json


MMLU_CATEGORY_ORDER = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

CHOICES = "ABCDEFGHIJ"
SUMMARY_CATEGORY_RE = re.compile(r"Average accuracy\s+([0-9.]+)\s+-\s+(.+)")
SUMMARY_OVERALL_RE = re.compile(r"Average accuracy:\s+([0-9.]+)")


def summary_path_to_model_name(path: Path) -> str:
    stem = path.name
    if stem.endswith("_summary.txt"):
        stem = stem[: -len("_summary.txt")]
    if "-CoT-" in stem:
        return stem.split("-CoT-", 1)[0]
    return stem


def read_mmlu_summary(path: Path) -> dict[str, Any]:
    category_accuracy: dict[str, float] = {}
    overall_accuracy: float | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            category_match = SUMMARY_CATEGORY_RE.search(line)
            if category_match:
                accuracy, category = category_match.groups()
                category_accuracy[category.strip()] = float(accuracy)
                continue
            overall_match = SUMMARY_OVERALL_RE.search(line)
            if overall_match:
                overall_accuracy = float(overall_match.group(1))

    return {
        "model": summary_path_to_model_name(path),
        "path": str(path),
        "overall_accuracy": overall_accuracy,
        "category_accuracy": category_accuracy,
    }


def discover_mmlu_summaries(summary_dir: Path) -> dict[str, Path]:
    summaries: dict[str, Path] = {}
    for path in sorted(summary_dir.glob("*_summary.txt")):
        summaries[summary_path_to_model_name(path)] = path
    return summaries


def build_mmlu_category_rows(
    summary_dir: Path,
    model_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = discover_mmlu_summaries(summary_dir)
    if model_names is None:
        model_names = sorted(summaries)

    category_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        path = summaries.get(model_name)
        if path is None:
            continue
        parsed = read_mmlu_summary(path)
        if parsed["overall_accuracy"] is not None:
            overall_rows.append(
                {
                    "model": model_name,
                    "accuracy": float(parsed["overall_accuracy"]),
                    "summary_path": parsed["path"],
                }
            )
        for category in MMLU_CATEGORY_ORDER:
            accuracy = parsed["category_accuracy"].get(category)
            if accuracy is None:
                continue
            category_rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "accuracy": float(accuracy),
                    "summary_path": parsed["path"],
                }
            )

    return category_rows, overall_rows


def discover_mmlu_evaluation_summaries(summary_root: Path) -> dict[str, Path]:
    summaries: dict[str, Path] = {}
    for path in sorted(summary_root.glob("*/summary_validation.json")):
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        model_name = str(payload.get("model") or path.parent.name)
        summaries[model_name] = path
    return summaries


def build_mmlu_evaluation_rows(
    summary_root: Path,
    model_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = discover_mmlu_evaluation_summaries(summary_root)
    if model_names is None:
        model_names = sorted(summaries)

    category_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        path = summaries.get(model_name)
        if path is None:
            continue
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        overall_rows.append(
            {
                "model": model_name,
                "accuracy": float(payload["accuracy"]),
                "summary_path": str(path),
                "split": str(payload.get("split", "validation")),
                "examples": int(payload.get("examples", 0)),
            }
        )
        for category in MMLU_CATEGORY_ORDER:
            stats = payload.get("category", {}).get(category)
            if stats is None:
                continue
            category_rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "accuracy": float(stats["accuracy"]),
                    "correct": int(stats.get("correct", 0)),
                    "wrong": int(stats.get("wrong", 0)),
                    "summary_path": str(path),
                    "split": str(payload.get("split", "validation")),
                }
            )
    return category_rows, overall_rows


def build_category_winners(
    category_rows: list[dict[str, Any]],
    model_to_index: dict[str, int],
    overall_accuracy: dict[str, float] | None = None,
    tie_tol: float = 1e-12,
) -> dict[str, dict[str, Any]]:
    overall_accuracy = overall_accuracy or {}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in category_rows:
        by_category[row["category"]].append(row)

    winners: dict[str, dict[str, Any]] = {}
    for category in MMLU_CATEGORY_ORDER:
        rows = by_category.get(category, [])
        if not rows:
            continue
        best_acc = max(float(row["accuracy"]) for row in rows)
        best_rows = [
            row for row in rows if abs(float(row["accuracy"]) - best_acc) <= tie_tol
        ]
        best_rows = sorted(
            best_rows,
            key=lambda row: (
                -overall_accuracy.get(row["model"], float("-inf")),
                model_to_index[row["model"]],
            ),
        )
        selected = best_rows[0]
        winners[category] = {
            "category": category,
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
            "selected_model_overall_accuracy": overall_accuracy.get(
                selected["model"]
            ),
            "tie_break_policy": "highest_overall_accuracy_then_model_index",
        }
    return winners


def load_mmlu_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        options = [str(option) for option in row["options"] if str(option) != "N/A"]
        rows.append(
            {
                "id": f"{split}:{int(row['question_id'])}",
                "split": split,
                "question_id": int(row["question_id"]),
                "question": str(row["question"]).strip(),
                "options": options,
                "answer": str(row["answer"]).strip(),
                "answer_index": int(row["answer_index"]),
                "category": str(row["category"]).replace("_", " ").strip(),
                "src": str(row.get("src", "")),
            }
        )
    return rows


def load_mmlu_samples(data_dir: Path, splits: Iterable[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in splits:
        samples.extend(load_mmlu_split(data_dir, split))
    return samples


def format_mmlu_router_text(question: str, options: list[str]) -> str:
    option_text = "\n".join(
        f"{CHOICES[idx]}. {option}" for idx, option in enumerate(options)
    )
    return f"Question:\n{question.strip()}\nOptions:\n{option_text}"


def make_mmlu_category_router_samples(
    data_dir: Path,
    splits: Iterable[str],
    winners: dict[str, dict[str, Any]],
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category_to_route_label = {
        category: idx for idx, category in enumerate(MMLU_CATEGORY_ORDER)
    }
    route_label_to_category = {
        label: category for category, label in category_to_route_label.items()
    }
    category_to_model = {
        category: winners[category]["selected_model"]
        for category in MMLU_CATEGORY_ORDER
        if category in winners
    }
    category_to_model_index = {
        category: int(winners[category]["selected_model_index"])
        for category in MMLU_CATEGORY_ORDER
        if category in winners
    }

    samples: list[dict[str, Any]] = []
    for row in load_mmlu_samples(data_dir, splits):
        category = row["category"]
        if category not in category_to_route_label:
            continue
        samples.append(
            {
                **row,
                "text": format_mmlu_router_text(row["question"], row["options"]),
                "target_category": category,
                "target_model": category_to_model.get(category),
                "target_model_index": category_to_model_index.get(category),
                "label": category_to_route_label[category],
            }
        )

    label_manifest = {
        "label_mode": "mmlu_category",
        "num_route_labels": len(MMLU_CATEGORY_ORDER),
        "category_to_route_label": category_to_route_label,
        "route_label_to_category": {
            str(label): category for label, category in route_label_to_category.items()
        },
        "category_to_model": category_to_model,
        "category_to_model_index": category_to_model_index,
        "route_label_to_model": {
            str(label): category_to_model.get(category)
            for label, category in route_label_to_category.items()
        },
        "route_label_to_model_index": {
            str(label): category_to_model_index.get(category)
            for label, category in route_label_to_category.items()
        },
        "model_names": model_names,
    }
    return samples, label_manifest


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    dump_jsonl(path, rows)


def write_manifest(path: Path, obj: Any) -> None:
    write_json(path, obj)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
