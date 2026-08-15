from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import write_json
from bench_coe.mmlu_utils import (
    make_mmlu_category_router_samples,
    read_json,
    write_jsonl,
)
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
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT-base as an MMLU-Pro category router."
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("outputs/bench_coe/mmlu/all_expert_category_mapping.json"),
    )
    parser.add_argument("--mmlu-data-dir", type=Path, default=Path("MMLU-Pro/data"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation"],
        choices=["validation", "test"],
        help="Source splits used for subject-classifier labels. Default avoids test leakage.",
    )
    parser.add_argument("--bert-model", type=Path, default=Path("models/bert-base-uncased"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/router/bert-base-mmlu-category"),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Optional stratified holdout fraction. Default trains on all MMLU-Pro samples.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA fp16 autocast for faster training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write router samples and the category-to-expert manifest without training.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = read_json(args.mapping_json)
    model_names: list[str] = mapping["model_names"]
    samples, label_manifest = make_mmlu_category_router_samples(
        args.mmlu_data_dir,
        args.splits,
        mapping["category_winners"],
        model_names,
    )
    if not samples:
        raise SystemExit("No MMLU-Pro router samples were created.")

    sample_file = args.output_dir / "mmlu_router_samples.jsonl"
    label_manifest_path = args.output_dir / "route_label_manifest.json"
    write_jsonl(sample_file, samples)
    write_json(label_manifest_path, label_manifest)

    if args.prepare_only:
        run_manifest = {
            "mapping_json": str(args.mapping_json),
            "mmlu_data_dir": str(args.mmlu_data_dir),
            "splits": args.splits,
            "sample_file": str(sample_file),
            "label_manifest": str(label_manifest_path),
            "all_model_names": model_names,
            "category_winners": mapping["category_winners"],
            "training_args": jsonable_args(args),
            "status": "prepared_only",
        }
        write_json(args.output_dir / "training_manifest.json", run_manifest)
        print(f"Saved {len(samples)} labeled samples to {sample_file}")
        print(f"Saved route mapping to {label_manifest_path}")
        return

    train_samples, val_samples = stratified_split(
        samples, args.validation_fraction, args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.bert_model), use_fast=True)
    id2label = {
        int(label): category
        for label, category in label_manifest["route_label_to_category"].items()
    }
    label2id = {category: label for label, category in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.bert_model),
        num_labels=int(label_manifest["num_route_labels"]),
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
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            RouterDataset(val_samples, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    scheduler = make_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    metrics: dict[str, Any] = {
        "train_size": len(train_samples),
        "validation_size": len(val_samples),
        "num_route_labels": label_manifest["num_route_labels"],
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
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
            preds = outputs.logits.detach().argmax(dim=-1)
            correct += int((preds == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
            progress.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "train_accuracy": correct / total if total else 0.0,
        }
        if val_loader is not None:
            epoch_metrics["validation"] = evaluate(model, val_loader, device)
        metrics["epochs"].append(epoch_metrics)
        print(epoch_metrics)

    model_dir = args.output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    run_manifest = {
        "mapping_json": str(args.mapping_json),
        "mmlu_data_dir": str(args.mmlu_data_dir),
        "splits": args.splits,
        "bert_model": str(args.bert_model),
        "model_dir": str(model_dir),
        "sample_file": str(sample_file),
        "label_manifest": str(label_manifest_path),
        "all_model_names": model_names,
        "category_winners": mapping["category_winners"],
        "training_args": jsonable_args(args),
    }
    write_json(args.output_dir / "training_manifest.json", run_manifest)
    write_json(args.output_dir / "train_metrics.json", metrics)
    print(f"Saved router model to {model_dir}")
    print(f"Saved {len(samples)} labeled samples to {sample_file}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
