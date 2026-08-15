from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build query-router labels from cached multimodal experts.")
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-key", default="train_ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    mapping = read_json(args.mapping_json)
    split = read_json(args.split_manifest)
    include_ids = {str(value) for value in split[args.split_key]}
    selected_models = sorted(
        {winner["selected_model"] for winner in mapping["subject_winners"].values()},
        key=lambda model: (-float(mapping["overall_accuracy"][model]), model),
    )
    rows_by_model = {
        model: {str(row["id"]): row for row in read_jsonl(args.predictions_root / model / "predictions.jsonl")}
        for model in selected_models
    }
    reference = rows_by_model[selected_models[0]]
    samples = []
    dropped = []
    for sample_id in sorted(include_ids):
        correct_models = [
            model for model in selected_models if bool(rows_by_model[model][sample_id].get("is_correct", False))
        ]
        if not correct_models:
            dropped.append(sample_id)
            continue
        target_model = correct_models[0]
        row = reference[sample_id]
        samples.append(
            {
                "id": sample_id,
                "subject": row.get("subject"),
                "target_model": target_model,
                "correct_models": correct_models,
            }
        )
    active_models = [model for model in selected_models if any(row["target_model"] == model for row in samples)]
    model_to_label = {model: index for index, model in enumerate(active_models)}
    for row in samples:
        row["label"] = model_to_label[row["target_model"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "query_labels.jsonl").open("w", encoding="utf-8") as file:
        for row in samples:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        args.output_dir / "route_label_manifest.json",
        {
            "label_mode": "expert",
            "num_route_labels": len(active_models),
            "model_names": active_models,
            "route_label_to_model": {str(index): model for model, index in model_to_label.items()},
            "model_to_route_label": model_to_label,
            "tie_break_policy": "correct_expert_with_highest_source_overall_accuracy",
            "source_mapping": str(args.mapping_json),
            "source_split": str(args.split_manifest),
            "source_split_key": args.split_key,
        },
    )
    write_json(
        args.output_dir / "statistics.json",
        {
            "source_count": len(include_ids),
            "labeled_count": len(samples),
            "dropped_no_correct": len(dropped),
            "candidate_models": selected_models,
            "active_models": active_models,
            "label_counts": dict(Counter(row["target_model"] for row in samples)),
        },
    )
    print(json.dumps(read_json(args.output_dir / "statistics.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
