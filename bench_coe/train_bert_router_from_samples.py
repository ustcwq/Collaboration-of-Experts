from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import write_json
from bench_coe.train_bert_router import (
    RouterDataset,
    evaluate,
    jsonable_args,
    make_linear_warmup_scheduler,
    move_to_device,
    set_seed,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT from prepared router samples.")
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, default=Path("models/bert-base-uncased"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    samples = read_jsonl(args.samples_jsonl)
    manifest = read_json(args.label_manifest)
    if not samples:
        raise SystemExit("No router samples found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_samples, validation_samples = stratified_split(
        samples, args.validation_fraction, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.bert_model), use_fast=True)
    id2label = {
        int(label): str(model)
        for label, model in manifest["route_label_to_model"].items()
    }
    label2id = {model: label for label, model in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.bert_model),
        num_labels=int(manifest["num_route_labels"]),
        id2label=id2label,
        label2id=label2id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        RouterDataset(train_samples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            RouterDataset(validation_samples, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_samples
        else None
    )
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(item in name for item in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(item in name for item in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    scheduler = make_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    metrics: dict[str, Any] = {
        "train_size": len(train_samples),
        "validation_size": len(validation_samples),
        "num_route_labels": manifest["num_route_labels"],
        "epochs": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.fp16 and device.type == "cuda",
            ):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            predictions = outputs.logits.detach().argmax(dim=-1)
            correct += int((predictions == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "train_accuracy": correct / total if total else 0.0,
        }
        if validation_loader is not None:
            epoch_metrics["validation"] = evaluate(model, validation_loader, device)
            validation_loss = float(epoch_metrics["validation"]["loss"])
            if validation_loss < best_validation_loss - 1e-6:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
        metrics["epochs"].append(epoch_metrics)
        print(epoch_metrics)
        if (
            validation_loader is not None
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            metrics["stopped_early"] = True
            metrics["stopped_epoch"] = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics["best_epoch"] = best_epoch
    metrics["best_validation_loss"] = (
        best_validation_loss if best_epoch is not None else None
    )

    model_dir = args.output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    write_json(args.output_dir / "route_label_manifest.json", manifest)
    write_json(args.output_dir / "train_metrics.json", metrics)
    write_json(
        args.output_dir / "training_manifest.json",
        {
            "samples_jsonl": str(args.samples_jsonl),
            "label_manifest": str(args.label_manifest),
            "model_dir": str(model_dir),
            "training_args": jsonable_args(args),
        },
    )
    print(f"Saved router model to {model_dir}")


if __name__ == "__main__":
    main()
