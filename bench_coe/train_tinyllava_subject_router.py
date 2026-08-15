from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-feature TinyLLaVA subject classifier."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_text(row: dict[str, Any]) -> str:
    options = row.get("options", [])
    option_text = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    question = str(row.get("raw", {}).get("question", row.get("question", "")))
    question = question.replace("<image 1>", "").replace("<image>", "").strip()
    return f"Question:\n{question}\nOptions:\n{option_text}" if option_text else f"Question:\n{question}"


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def extract_features(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache_path = args.output_dir / "tinyllava_features.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

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
    image_processor = model.vision_tower._image_processor

    features: list[torch.Tensor] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="TinyLLaVA features"):
        batch = rows[start : start + args.batch_size]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        pixels = image_processor(images=images, return_tensors="pt")["pixel_values"]
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
            text_outputs = model.language_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            text = masked_mean(text_outputs.last_hidden_state, attention_mask)
        joint = torch.cat([vision.float(), text.float()], dim=-1).cpu()
        features.append(joint)

    payload = {
        "features": torch.cat(features, dim=0),
        "ids": [str(row["id"]) for row in rows],
        "subjects": [str(row["subject"]) for row in rows],
        "feature_dim": int(features[0].shape[-1]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def stratified_indices(
    labels: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    train: list[int] = []
    validation: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * validation_fraction)),
        )
        validation.extend(indices[:validation_count])
        train.extend(indices[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


@torch.no_grad()
def evaluate(
    classifier: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = classifier(features)
        losses.append(float(criterion(logits, labels).cpu()))
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += int(labels.numel())
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": correct / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.predictions_jsonl)
    mapping = read_json(args.mapping_json)
    subjects = list(mapping["subjects"])
    subject_to_label = {subject: index for index, subject in enumerate(subjects)}
    route_label_to_subject = {
        str(index): subject for subject, index in subject_to_label.items()
    }
    subject_to_model = {
        subject: mapping["subject_winners"][subject]["selected_model"]
        for subject in subjects
    }
    payload = extract_features(args, rows)
    features = payload["features"]
    labels = [subject_to_label[subject] for subject in payload["subjects"]]
    if args.split_manifest:
        split = read_json(args.split_manifest)
        id_to_index = {sample_id: index for index, sample_id in enumerate(payload["ids"])}
        train_indices = [id_to_index[str(value)] for value in split["train_ids"]]
        validation_indices = [id_to_index[str(value)] for value in split["validation_ids"]]
    else:
        train_indices, validation_indices = stratified_indices(
            labels, args.validation_fraction, args.seed
        )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(features[train_indices], label_tensor[train_indices]),
        batch_size=args.classifier_batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(features[validation_indices], label_tensor[validation_indices]),
        batch_size=args.classifier_batch_size,
        shuffle=False,
    )
    device = torch.device(args.device)
    classifier = nn.Sequential(
        nn.LayerNorm(features.shape[1]),
        nn.Dropout(0.15),
        nn.Linear(features.shape[1], len(subjects)),
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
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "train_accuracy": correct / total if total else 0.0,
            "validation": validation,
        }
        history.append(epoch_metrics)
        print(epoch_metrics)
        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in classifier.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch >= args.min_epochs and no_improvement >= args.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("No classifier checkpoint was selected.")
    classifier.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "feature_dim": int(features.shape[1]),
            "num_labels": len(subjects),
        },
        args.output_dir / "classifier.pt",
    )
    manifest = {
        "router_type": "tinyllava_frozen_feature_classifier",
        "label_mode": "subject",
        "num_route_labels": len(subjects),
        "subject_to_route_label": subject_to_label,
        "route_label_to_subject": route_label_to_subject,
        "subject_to_model": subject_to_model,
        "route_label_to_model": {
            label: subject_to_model[subject]
            for label, subject in route_label_to_subject.items()
        },
        "model_names": mapping["model_names"],
        "tinyllava_model_path": str(args.model_path),
        "classifier_path": str(args.output_dir / "classifier.pt"),
        "feature_cache": str(args.output_dir / "tinyllava_features.pt"),
    }
    write_json(args.output_dir / "route_label_manifest.json", manifest)
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
    write_json(
        args.output_dir / "split_manifest.json",
        {
            "train_ids": [payload["ids"][index] for index in train_indices],
            "validation_ids": [payload["ids"][index] for index in validation_indices],
        },
    )


if __name__ == "__main__":
    main()
