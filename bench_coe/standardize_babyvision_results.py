from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench_coe.gaokao_utils import write_json


RUN_DIRS = {
    "InternVL3_5-2B": "InternVL3_5-2B__judge_skipped",
    "LFM2.5-VL-1.6B": "LFM2.5-VL-1.6B__judge_skipped",
    "Qwen3.5-2B": "Qwen3.5-2B__judge_skipped",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B-Instruct__judge_skipped",
    "gemma-4-E2B-it": "gemma-4-E2B-it__judge_skipped",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize completely judged BabyVision expert caches.")
    parser.add_argument("--babyvision-root", type=Path, default=Path("BabyVision"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_run(run_root: Path) -> list[dict[str, Any]]:
    merged_path = run_root / "merged_results_judge_by_Qwen3.5-9B.jsonl"
    if merged_path.exists():
        return read_jsonl(merged_path)
    predictions = {str(row["sample_id"]): row for row in read_jsonl(run_root / "predictions.jsonl")}
    judgments = {}
    for row in read_jsonl(run_root / "judgments_judge_by_Qwen3.5-9B.jsonl"):
        judgments[str(row["sample_id"])] = row
    return [
        {
            "sample_id": sample_id,
            "prediction": prediction,
            "judgment": judgments.get(sample_id, {}),
        }
        for sample_id, prediction in predictions.items()
    ]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    canonical_ids = None
    for model, run_dir in RUN_DIRS.items():
        run_root = args.babyvision_root / "outputs/rerun_local_skip_judge_fast" / run_dir
        merged = load_run(run_root)
        rows = []
        for item in merged:
            prediction = item["prediction"]
            judgment = item["judgment"]
            verdict = judgment.get("local_judge_json") or judgment.get("judge_json") or {}
            image_path = Path(str(prediction["image_path"]))
            if not image_path.is_absolute():
                image_path = args.babyvision_root / image_path
            rows.append(
                {
                    "id": str(item["sample_id"]),
                    "question": prediction["question"],
                    "raw": {"question": prediction["question"]},
                    "image_path": str(image_path.resolve()),
                    "answer": prediction.get("reference_answer"),
                    "prediction": prediction.get("model_final_answer"),
                    "is_correct": bool(verdict.get("is_correct", False)),
                    "category": prediction.get("task_type"),
                    "task": prediction.get("subtype"),
                    "question_type": "open",
                    "judge_model": judgment.get("judge_model_name"),
                }
            )
        rows.sort(key=lambda row: int(row["id"]))
        if len(rows) != 388:
            raise ValueError(f"{model} has {len(rows)} merged rows, expected 388")
        ids = [row["id"] for row in rows]
        if canonical_ids is None:
            canonical_ids = ids
        elif ids != canonical_ids:
            raise ValueError(f"Sample IDs differ for {model}")
        output_dir = args.output_root / model
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        by_category: dict[str, list[int]] = defaultdict(list)
        by_task: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            by_category[str(row["category"])].append(int(row["is_correct"]))
            by_task[str(row["task"])].append(int(row["is_correct"]))
        correct = sum(int(row["is_correct"]) for row in rows)
        write_json(
            output_dir / "summary.json",
            {
                "status": "completed",
                "model": model,
                "count": len(rows),
                "accuracy": correct / len(rows),
                "by_category": {key: {"count": len(values), "accuracy": sum(values) / len(values)} for key, values in sorted(by_category.items())},
                "by_task": {key: {"count": len(values), "accuracy": sum(values) / len(values)} for key, values in sorted(by_task.items())},
                "judge": "Qwen3.5-9B local_text_judge",
            },
        )
    print(json.dumps({"models": list(RUN_DIRS), "samples": len(canonical_ids or [])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
