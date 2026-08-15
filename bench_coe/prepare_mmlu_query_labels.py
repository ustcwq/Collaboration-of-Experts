from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bench_coe.gaokao_utils import write_json
from bench_coe.mmlu_utils import format_mmlu_router_text, load_mmlu_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MMLU-Pro query-level router labels from stored expert predictions."
    )
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--mmlu-data-dir", type=Path, default=Path("MMLU-Pro/data"))
    parser.add_argument(
        "--expert-results-root",
        type=Path,
        default=Path("outputs/bench_coe/mmlu_pro_validation_single_models"),
    )
    parser.add_argument("--result-subdir", default="CoT/validation")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def selected_experts(mapping: dict[str, Any]) -> list[str]:
    experts = {
        str(winner["selected_model"])
        for winner in mapping["category_winners"].values()
    }
    return sorted(experts)


def load_model_predictions(
    root: Path, result_subdir: str, model_name: str
) -> dict[int, bool]:
    result_dir = root / model_name / result_subdir
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    predictions: dict[int, bool] = {}
    for path in sorted(result_dir.glob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, list):
            continue
        for row in payload:
            predictions[int(row["question_id"])] = row.get("pred") == row.get("answer")
    return predictions


def main() -> None:
    args = parse_args()
    mapping = read_json(args.mapping_json)
    experts = selected_experts(mapping)
    model_to_index = {model: index for index, model in enumerate(experts)}
    prediction_matrix = {
        model: load_model_predictions(args.expert_results_root, args.result_subdir, model)
        for model in experts
    }
    rows = load_mmlu_split(args.mmlu_data_dir, args.split)
    source_question_ids = [int(row["question_id"]) for row in rows]
    overall_accuracy = {
        model: sum(
            int(prediction_matrix[model].get(question_id, False))
            for question_id in source_question_ids
        )
        / len(source_question_ids)
        for model in experts
    }
    pending_samples: list[dict[str, Any]] = []
    dropped_no_correct = 0
    label_counts: Counter[str] = Counter()
    for row in rows:
        question_id = int(row["question_id"])
        correct_models = [
            model for model in experts if prediction_matrix[model].get(question_id, False)
        ]
        if not correct_models:
            dropped_no_correct += 1
            continue
        selected_model = min(
            correct_models,
            key=lambda model: (-overall_accuracy[model], model_to_index[model]),
        )
        label_counts[selected_model] += 1
        pending_samples.append(
            {
                **row,
                "text": format_mmlu_router_text(row["question"], row["options"]),
                "correct_models": correct_models,
                "target_model": selected_model,
            }
        )

    active_experts = sorted(label_counts)
    route_label_to_model = {
        index: model for index, model in enumerate(active_experts)
    }
    model_to_route_label = {
        model: index for index, model in route_label_to_model.items()
    }
    samples = [
        {**sample, "label": model_to_route_label[sample["target_model"]]}
        for sample in pending_samples
    ]

    manifest = {
        "label_mode": "query_expert",
        "num_route_labels": len(active_experts),
        "candidate_model_names": experts,
        "model_names": active_experts,
        "inactive_zero_label_models": sorted(set(experts).difference(active_experts)),
        "model_to_route_label": model_to_route_label,
        "route_label_to_model": {
            str(label): model for label, model in route_label_to_model.items()
        },
        "overall_accuracy": overall_accuracy,
        "tie_break_policy": "highest_overall_accuracy_then_stable_model_index",
        "source_split": args.split,
        "total_source_examples": len(rows),
        "labeled_examples": len(samples),
        "dropped_no_correct": dropped_no_correct,
        "label_counts": dict(label_counts),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "query_router_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "route_label_manifest.json", manifest)
    write_json(
        args.output_dir / "preparation_manifest.json",
        {
            "mapping_json": str(args.mapping_json),
            "mmlu_data_dir": str(args.mmlu_data_dir),
            "expert_results_root": str(args.expert_results_root),
            "result_subdir": args.result_subdir,
            "sample_path": str(sample_path),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
