from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from bench_coe.evaluate_tinyllava_subject_router import (
    format_text,
    grouped_accuracy,
    load_expert_rows,
    load_ids,
    read_json,
)
from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-VL frozen-feature subject router.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--route-label-manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-ids-json", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def route_prompt(row: dict[str, Any], subjects: list[str]) -> str:
    return (
        "Classify this multimodal problem into exactly one subject.\n"
        f"Allowed subjects: {', '.join(subjects)}.\n"
        "Return only the subject label.\n\n"
        f"Problem:\n{format_text(row)}"
    )


@torch.inference_mode()
def extract_features(
    args: argparse.Namespace, rows: list[dict[str, Any]], subjects: list[str]
) -> torch.Tensor:
    cache_path = args.output_dir / "qwen3vl_features.pt"
    ids = [str(row["id"]) for row in rows]
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("ids") == ids:
            return payload["features"]
    processor = AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    features = []
    for row in tqdm(rows, desc="Qwen3-VL eval features"):
        content = [
            {"type": "image", "image": str(Path(row["image_path"]).resolve())},
            {"type": "text", "text": route_prompt(row, subjects)},
        ]
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        output = model.model(**inputs, return_dict=True)
        mask = inputs["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
        pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        features.append(pooled.float().cpu())
    result = torch.cat(features, dim=0)
    torch.save({"ids": ids, "features": result}, cache_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.route_label_manifest)
    model_names = [str(value) for value in manifest["model_names"]]
    if args.reference_model not in model_names:
        raise ValueError("reference-model must belong to manifest model_names")
    include_ids = load_ids(args.include_ids_json)
    rows, expert_rows = load_expert_rows(args.predictions_root, model_names, include_ids)
    subjects = [manifest["route_label_to_subject"][str(i)] for i in range(manifest["num_route_labels"])]
    features = extract_features(args, rows, subjects)
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    classifier = nn.Sequential(
        nn.LayerNorm(int(checkpoint["feature_dim"])),
        nn.Dropout(0.15),
        nn.Linear(int(checkpoint["feature_dim"]), int(checkpoint["num_labels"])),
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    with torch.inference_mode():
        route_labels = classifier(features).argmax(dim=-1).tolist()

    output_rows = []
    single_correct = Counter()
    subject_correct = 0
    subject_total = 0
    oracle_subject_correct = 0
    oracle_subject_total = 0
    oracle_any_correct = 0
    for row, route_label in zip(rows, route_labels):
        sample_id = str(row["id"])
        routed_subject = str(manifest["route_label_to_subject"][str(route_label)])
        routed_model = str(manifest["subject_to_model"][routed_subject])
        routed_expert = expert_rows[routed_model][sample_id]
        correctness = {
            model: bool(expert_rows[model][sample_id].get("is_correct", False))
            for model in model_names
        }
        for model, correct in correctness.items():
            single_correct[model] += int(correct)
        true_subject = row.get("subject")
        if true_subject is None and isinstance(row.get("raw"), dict):
            true_subject = row["raw"].get("subject")
        true_subject = str(true_subject) if true_subject is not None else None
        if true_subject in manifest["subject_to_route_label"]:
            subject_total += 1
            subject_correct += int(routed_subject == true_subject)
            oracle_model = str(manifest["subject_to_model"][true_subject])
            oracle_subject_total += 1
            oracle_subject_correct += int(expert_rows[oracle_model][sample_id].get("is_correct", False))
        oracle_any_correct += int(any(correctness.values()))
        output_rows.append(
            {
                "id": sample_id,
                "true_subject": true_subject,
                "routed_subject": routed_subject,
                "routed_model": routed_model,
                "prediction": routed_expert.get("prediction"),
                "answer": routed_expert.get("answer"),
                "is_correct": bool(routed_expert.get("is_correct", False)),
                "category": row.get("category"),
                "task": row.get("task"),
                "question_type": row.get("question_type"),
            }
        )
    total = len(output_rows)
    single_accuracies = {model: single_correct[model] / total for model in model_names}
    best_single_model = max(
        model_names, key=lambda model: (single_accuracies[model], -model_names.index(model))
    )
    summary = {
        "count": total,
        "routed_accuracy": sum(int(row["is_correct"]) for row in output_rows) / total,
        "subject_accuracy": subject_correct / subject_total if subject_total else None,
        "subject_accuracy_count": subject_total,
        "oracle_subject_mapping_accuracy": oracle_subject_correct / oracle_subject_total if oracle_subject_total else None,
        "oracle_subject_mapping_count": oracle_subject_total,
        "oracle_any_expert_accuracy": oracle_any_correct / total,
        "single_model_accuracies": single_accuracies,
        "best_single_model": best_single_model,
        "best_single_accuracy": single_accuracies[best_single_model],
        "routed_model_counts": dict(Counter(row["routed_model"] for row in output_rows)),
        "routed_subject_counts": dict(Counter(row["routed_subject"] for row in output_rows)),
        "by_true_subject": grouped_accuracy(output_rows, "true_subject"),
        "by_category": grouped_accuracy(output_rows, "category"),
        "by_task": grouped_accuracy(output_rows, "task"),
        "by_question_type": grouped_accuracy(output_rows, "question_type"),
        "predictions_root": str(args.predictions_root),
        "include_ids_json": str(args.include_ids_json) if args.include_ids_json else None,
        "route_label_manifest": str(args.route_label_manifest),
    }
    write_json(args.output_dir / "predictions.json", output_rows)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
