from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench_coe.gaokao_utils import (
    dump_jsonl,
    load_gaokao2010_2022_full_predictions,
    load_objective_questions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GAOKAO query-level labels from stored expert predictions."
    )
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("GAOKAO-Bench-2010-2022/Data"))
    parser.add_argument(
        "--objective-dir",
        type=Path,
        default=Path("GAOKAO-Bench-2010-2022/Data/Objective_Questions"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reextract-empty", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    mapping = read_json(args.mapping_json)
    candidate_models = sorted(
        {
            str(winner["selected_model"])
            for winner in mapping["subject_winners"].values()
        }
    )
    model_to_index = {
        model: index for index, model in enumerate(candidate_models)
    }
    slot_matrix = load_gaokao2010_2022_full_predictions(
        args.data_dir, candidate_models, reextract_empty=args.reextract_empty
    )
    grouped_correct: dict[str, dict[str, list[bool]]] = {
        model: defaultdict(list) for model in candidate_models
    }
    overall_accuracy: dict[str, float] = {}
    for model in candidate_models:
        rows = slot_matrix[model]
        total_points = sum(float(row["total_points"]) for row in rows.values())
        correct_points = sum(float(row["correct_points"]) for row in rows.values())
        overall_accuracy[model] = correct_points / total_points if total_points else 0.0
        for row in rows.values():
            query_id = f"{row['task']}:{row['index']}"
            grouped_correct[model][query_id].append(bool(row["is_correct"]))

    objective_rows = load_objective_questions(args.objective_dir)
    pending_samples: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    dropped_no_correct = 0
    for row in objective_rows:
        query_id = str(row["id"])
        correct_models = [
            model
            for model in candidate_models
            if grouped_correct[model].get(query_id)
            and all(grouped_correct[model][query_id])
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
                "text": f"Question:\n{row['question']}",
                "correct_models": correct_models,
                "target_model": selected_model,
            }
        )

    active_models = sorted(label_counts)
    model_to_route_label = {
        model: index for index, model in enumerate(active_models)
    }
    samples = [
        {**row, "label": model_to_route_label[row["target_model"]]}
        for row in pending_samples
    ]
    manifest = {
        "label_mode": "query_expert",
        "num_route_labels": len(active_models),
        "candidate_model_names": candidate_models,
        "model_names": active_models,
        "inactive_zero_label_models": sorted(
            set(candidate_models).difference(active_models)
        ),
        "model_to_route_label": model_to_route_label,
        "route_label_to_model": {
            str(label): model for model, label in model_to_route_label.items()
        },
        "overall_accuracy": overall_accuracy,
        "tie_break_policy": "highest_overall_accuracy_then_stable_model_index",
        "total_source_examples": len(objective_rows),
        "labeled_examples": len(samples),
        "dropped_no_correct": dropped_no_correct,
        "label_counts": dict(label_counts),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "query_router_samples.jsonl", samples)
    write_json(args.output_dir / "route_label_manifest.json", manifest)
    write_json(
        args.output_dir / "preparation_manifest.json",
        {
            "mapping_json": str(args.mapping_json),
            "data_dir": str(args.data_dir),
            "objective_dir": str(args.objective_dir),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
