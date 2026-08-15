from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench_coe.expert_pool import load_expert_pool, select_expert_pool_models
from bench_coe.gaokao_utils import write_json
from bench_coe.prepare_gaokao_mm_qwen3vl_router import FILE_SUBJECTS, SUBJECT_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize GAOKAO-MM cached model results.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--expert-pool-config", type=Path, required=True)
    parser.add_argument("--expert-pool", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalized_answers(value: Any) -> set[str]:
    values = value if isinstance(value, list) else [value]
    return {str(item).strip().upper() for item in values if str(item).strip()}


def load_model_rows(root: Path, model: str) -> list[dict[str, Any]]:
    rows = []
    for keyword, subject in FILE_SUBJECTS.items():
        payload = read_json(root / model / f"{model}_{keyword}.json")
        for item in payload["example"]:
            sample_id = f"{keyword}:{item['index']}"
            answer = normalized_answers(item.get("standard_answer"))
            prediction = normalized_answers(item.get("model_answer"))
            rows.append(
                {
                    "id": sample_id,
                    "subject": subject,
                    "question": item.get("question", ""),
                    "raw": {"question": item.get("question", ""), "subject": subject},
                    "image_path": item.get("combined_image"),
                    "answer": sorted(answer),
                    "prediction": sorted(prediction),
                    "is_correct": bool(answer) and prediction == answer,
                    "year": item.get("year"),
                    "category": item.get("category"),
                }
            )
    return sorted(rows, key=lambda row: (SUBJECT_ORDER.index(row["subject"]), row["id"]))


def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    by_subject = {}
    for subject in SUBJECT_ORDER:
        subset = [row for row in rows if row["subject"] == subject]
        correct = sum(int(row["is_correct"]) for row in subset)
        by_subject[subject] = {
            "correct": correct,
            "total": len(subset),
            "accuracy": correct / len(subset),
        }
    correct = sum(int(row["is_correct"]) for row in rows)
    return {"status": "completed", "model": model, "count": len(rows), "accuracy": correct / len(rows), "by_subject": by_subject}


def main() -> None:
    args = parse_args()
    available = [path.name for path in args.results_root.iterdir() if path.is_dir()]
    pool = load_expert_pool(args.expert_pool_config, args.expert_pool)
    models, report, _ = select_expert_pool_models(available, pool)
    args.output_root.mkdir(parents=True, exist_ok=True)
    canonical = None
    for model in models:
        rows = load_model_rows(args.results_root, model)
        ids = [row["id"] for row in rows]
        if canonical is None:
            canonical = rows
        elif ids != [row["id"] for row in canonical]:
            raise ValueError(f"GAOKAO-MM IDs differ for {model}")
        output_dir = args.output_root / model
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(output_dir / "summary.json", summarize(rows, model))
    rng = random.Random(args.seed)
    by_subject: dict[str, list[str]] = defaultdict(list)
    for row in canonical or []:
        by_subject[row["subject"]].append(row["id"])
    train_ids, validation_ids = [], []
    for subject in SUBJECT_ORDER:
        ids = by_subject[subject]
        rng.shuffle(ids)
        validation_count = min(len(ids) - 1, max(1, round(len(ids) * args.validation_fraction)))
        validation_ids.extend(ids[:validation_count])
        train_ids.extend(ids[validation_count:])
    write_json(args.output_root / "split_manifest.json", {"train_ids": train_ids, "validation_ids": validation_ids, "source": "GAOKAO-MM full set with internal stratified holdout"})
    write_json(args.output_root / "pool_report.json", report)
    print(json.dumps({"models": models, "train": len(train_ids), "validation": len(validation_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
