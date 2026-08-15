from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from bench_coe.evaluate_tinyllava_subject_router import format_text
from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-feature Qwen3-VL subject router.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


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
) -> dict[str, Any]:
    cache_path = args.output_dir / "qwen3vl_features.pt"
    ids = [str(row["id"]) for row in rows]
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("ids") == ids:
            return payload
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
    for row in tqdm(rows, desc="Qwen3-VL features"):
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
    payload = {
        "ids": ids,
        "subjects": [str(row["subject"]) for row in rows],
        "features": torch.cat(features, dim=0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


@torch.no_grad()
def evaluate(classifier: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = classifier(features)
        losses.append(float(criterion(logits, labels).cpu()))
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += int(labels.numel())
    return {"loss": sum(losses) / max(1, len(losses)), "accuracy": correct / total}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.predictions_jsonl)
    mapping = read_json(args.mapping_json)
    split = read_json(args.split_manifest)
    subjects = list(mapping["subjects"])
    subject_to_label = {subject: index for index, subject in enumerate(subjects)}
    payload = extract_features(args, rows, subjects)
    id_to_index = {sample_id: index for index, sample_id in enumerate(payload["ids"])}
    train_indices = [id_to_index[str(value)] for value in split["train_ids"]]
    validation_indices = [id_to_index[str(value)] for value in split["validation_ids"]]
    labels = torch.tensor(
        [subject_to_label[subject] for subject in payload["subjects"]], dtype=torch.long
    )
    features = payload["features"]
    train_loader = DataLoader(
        TensorDataset(features[train_indices], labels[train_indices]),
        batch_size=args.classifier_batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(features[validation_indices], labels[validation_indices]),
        batch_size=args.classifier_batch_size,
        shuffle=False,
    )
    device = torch.device(args.device)
    classifier = nn.Sequential(
        nn.LayerNorm(features.shape[1]), nn.Dropout(0.15), nn.Linear(features.shape[1], len(subjects))
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    no_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        losses = []
        correct = 0
        total = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=-1) == batch_labels).sum().item())
            total += int(batch_labels.numel())
        validation = evaluate(classifier, validation_loader, device)
        metrics = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "train_accuracy": correct / total,
            "validation": validation,
        }
        history.append(metrics)
        print(metrics)
        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in classifier.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch >= args.min_epochs and no_improvement >= args.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("No classifier checkpoint selected")
    torch.save(
        {"state_dict": best_state, "feature_dim": features.shape[1], "num_labels": len(subjects)},
        args.output_dir / "classifier.pt",
    )
    route_label_to_subject = {str(index): subject for subject, index in subject_to_label.items()}
    subject_to_model = {
        subject: mapping["subject_winners"][subject]["selected_model"] for subject in subjects
    }
    write_json(
        args.output_dir / "route_label_manifest.json",
        {
            "router_type": "qwen3vl_frozen_feature_classifier",
            "label_mode": "subject",
            "num_route_labels": len(subjects),
            "subject_to_route_label": subject_to_label,
            "route_label_to_subject": route_label_to_subject,
            "subject_to_model": subject_to_model,
            "route_label_to_model": {
                label: subject_to_model[subject] for label, subject in route_label_to_subject.items()
            },
            "model_names": mapping["model_names"],
            "qwen3vl_model_path": str(args.model_path),
            "classifier_path": str(args.output_dir / "classifier.pt"),
            "expert_mapping_source": str(args.mapping_json),
        },
    )
    write_json(
        args.output_dir / "train_metrics.json",
        {
            "train_size": len(train_indices),
            "validation_size": len(validation_indices),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs": history,
        },
    )


if __name__ == "__main__":
    main()
