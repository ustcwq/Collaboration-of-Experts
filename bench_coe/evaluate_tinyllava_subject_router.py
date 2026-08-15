from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen-feature TinyLLaVA subject router from cached expert predictions."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--route-label-manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-ids-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = read_json(path)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get("validation_ids"), list):
        values = payload["validation_ids"]
    elif isinstance(payload, dict) and isinstance(payload.get("ids"), list):
        values = payload["ids"]
    else:
        raise ValueError(f"Unsupported include-ID payload: {path}")
    return {str(value) for value in values}


def normalize_options(row: dict[str, Any]) -> list[str]:
    raw = row.get("options")
    if raw is None:
        raw = row.get("choices")
    if raw is None and isinstance(row.get("raw"), dict):
        raw = row["raw"].get("options", row["raw"].get("choices"))
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [str(value) for value in raw] if isinstance(raw, list) else []


def format_text(row: dict[str, Any]) -> str:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    question = str(raw.get("question", row.get("question", row.get("prompt", ""))))
    for token in ("<image 1>", "<image>", "[Image 1]"):
        question = question.replace(token, "")
    options = normalize_options(row)
    if not options:
        return f"Question:\n{question.strip()}"
    option_text = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    return f"Question:\n{question.strip()}\nOptions:\n{option_text}"


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def extract_features(args: argparse.Namespace, rows: list[dict[str, Any]]) -> torch.Tensor:
    cache_path = args.output_dir / "tinyllava_features.pt"
    ids = [str(row["id"]) for row in rows]
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("ids") == ids:
            return payload["features"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    processor = model.vision_tower._image_processor
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="TinyLLaVA eval features"):
        batch = rows[start : start + args.batch_size]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(device=device, dtype=model.dtype)
        encoded = tokenizer(
            [format_text(row) for row in batch],
            truncation=True,
            padding=True,
            max_length=args.max_text_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            vision = model.encode_images(pixels).mean(dim=1)
            text_output = model.language_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            text = masked_mean(text_output.last_hidden_state, attention_mask)
        chunks.append(torch.cat([vision.float(), text.float()], dim=-1).cpu())
    features = torch.cat(chunks, dim=0)
    torch.save({"ids": ids, "features": features}, cache_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features


def load_expert_rows(
    root: Path, model_names: list[str], include_ids: set[str] | None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    ordered_ids: list[str] | None = None
    reference_rows: list[dict[str, Any]] | None = None
    for model_name in model_names:
        rows = read_jsonl(root / model_name / "predictions.jsonl")
        if include_ids is not None:
            rows = [row for row in rows if str(row["id"]) in include_ids]
        ids = [str(row["id"]) for row in rows]
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise ValueError(f"Prediction IDs/order differ for {model_name}")
        by_model[model_name] = {str(row["id"]): row for row in rows}
        if reference_rows is None:
            reference_rows = rows
    if reference_rows is None or not reference_rows:
        raise RuntimeError(f"No prediction rows found under {root}")
    if include_ids is not None and len(reference_rows) != len(include_ids):
        raise ValueError(f"Matched {len(reference_rows)} rows, expected {len(include_ids)}")
    return reference_rows, by_model


def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups[str(value)].append(int(row["is_correct"]))
    return {
        value: {"count": len(scores), "accuracy": sum(scores) / len(scores)}
        for value, scores in sorted(groups.items())
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.route_label_manifest)
    model_names = [str(value) for value in manifest["model_names"]]
    if args.reference_model not in model_names:
        raise ValueError("reference-model must belong to manifest model_names")
    include_ids = load_ids(args.include_ids_json)
    reference_rows, expert_rows = load_expert_rows(
        args.predictions_root, model_names, include_ids
    )
    reference_lookup = {str(row["id"]): row for row in reference_rows}
    features = extract_features(args, reference_rows)
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    classifier = nn.Sequential(
        nn.LayerNorm(int(checkpoint["feature_dim"])),
        nn.Dropout(0.15),
        nn.Linear(int(checkpoint["feature_dim"]), int(checkpoint["num_labels"])),
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    with torch.inference_mode():
        labels = classifier(features).argmax(dim=-1).tolist()

    output_rows = []
    single_correct = Counter()
    subject_correct = 0
    subject_total = 0
    oracle_subject_correct = 0
    oracle_subject_total = 0
    oracle_any_correct = 0
    for row, label in zip(reference_rows, labels):
        sample_id = str(row["id"])
        routed_subject = str(manifest["route_label_to_subject"][str(label)])
        routed_model = str(manifest["subject_to_model"][routed_subject])
        routed_expert = expert_rows[routed_model][sample_id]
        model_correctness = {
            model: bool(expert_rows[model][sample_id].get("is_correct", False))
            for model in model_names
        }
        for model, correct in model_correctness.items():
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
            oracle_subject_correct += int(
                expert_rows[oracle_model][sample_id].get("is_correct", False)
            )
        oracle_any_correct += int(any(model_correctness.values()))
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
    routed_accuracy = sum(int(row["is_correct"]) for row in output_rows) / total
    single_accuracies = {
        model: single_correct[model] / total for model in model_names
    }
    best_single_model = max(
        model_names, key=lambda model: (single_accuracies[model], -model_names.index(model))
    )
    summary = {
        "count": total,
        "routed_accuracy": routed_accuracy,
        "subject_accuracy": subject_correct / subject_total if subject_total else None,
        "subject_accuracy_count": subject_total,
        "oracle_subject_mapping_accuracy": (
            oracle_subject_correct / oracle_subject_total if oracle_subject_total else None
        ),
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
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "model_path": str(args.model_path),
            "classifier": str(args.classifier),
            "reference_model": args.reference_model,
            "feature_cache": str(args.output_dir / "tinyllava_features.pt"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
