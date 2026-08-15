from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from bench_coe.gaokao_utils import (
    dump_jsonl,
    make_router_samples,
    make_subject_router_samples,
    read_json,
    write_json,
)


class RouterDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(sample["label"]), dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT-base as the GAOKAO expert router."
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("outputs/bench_coe/gaokao/local_expert_subject_mapping.json"),
    )
    parser.add_argument(
        "--objective-dir",
        type=Path,
        default=Path("GAOKAO-Bench-2010-2022/Data/Objective_Questions"),
    )
    parser.add_argument("--bert-model", type=Path, default=Path("models/bert-base-uncased"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/router/bert-base-gaokao"),
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-mode",
        choices=["subject", "expert"],
        default="subject",
        help=(
            "subject trains one class per GAOKAO subject and maps subjects to experts "
            "at inference time. expert reproduces the older direct expert-label router."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Optional stratified holdout fraction. Default trains on all GAOKAO samples.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA fp16 autocast for faster training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def stratified_split(
    samples: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_fraction <= 0:
        return samples, []
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_label.setdefault(int(sample["label"]), []).append(sample)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        if len(label_samples) < 2:
            train.extend(label_samples)
            continue
        val_count = min(
            len(label_samples) - 1,
            max(1, int(round(len(label_samples) * validation_fraction))),
        )
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def make_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
        preds = outputs.logits.argmax(dim=-1)
        correct += int((preds == batch["labels"]).sum().item())
        total += int(batch["labels"].numel())
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": correct / total if total else 0.0,
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = read_json(args.mapping_json)
    model_names: list[str] = mapping["model_names"]
    if args.label_mode == "subject":
        samples, label_manifest = make_subject_router_samples(
            args.objective_dir,
            mapping["subject_winners"],
            model_names,
        )
    else:
        samples, label_manifest = make_router_samples(
            args.objective_dir,
            mapping["subject_winners"],
            model_names,
        )
        label_manifest["label_mode"] = "expert"
    if not samples:
        raise SystemExit("No router samples were created.")

    dump_jsonl(args.output_dir / "gaokao_router_samples.jsonl", samples)
    write_json(args.output_dir / "route_label_manifest.json", label_manifest)

    train_samples, val_samples = stratified_split(
        samples, args.validation_fraction, args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.bert_model), use_fast=True)
    if label_manifest.get("label_mode") == "subject":
        id2label = {
            int(label): subject
            for label, subject in label_manifest["route_label_to_subject"].items()
        }
        label2id = {subject: label for label, subject in id2label.items()}
    else:
        id2label = {
            int(label): model_name
            for label, model_name in label_manifest["route_label_to_model"].items()
        }
        label2id = {model_name: label for label, model_name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.bert_model),
        num_labels=int(label_manifest["num_route_labels"]),
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = RouterDataset(train_samples, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
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
        "objective_dir": str(args.objective_dir),
        "bert_model": str(args.bert_model),
        "model_dir": str(model_dir),
        "sample_file": str(args.output_dir / "gaokao_router_samples.jsonl"),
        "label_manifest": str(args.output_dir / "route_label_manifest.json"),
        "all_model_names": model_names,
        "subject_winners": mapping["subject_winners"],
        "training_args": jsonable_args(args),
    }
    write_json(args.output_dir / "training_manifest.json", run_manifest)
    write_json(args.output_dir / "train_metrics.json", metrics)
    print(f"Saved router model to {model_dir}")
    print(f"Saved {len(samples)} labeled samples to {args.output_dir / 'gaokao_router_samples.jsonl'}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
