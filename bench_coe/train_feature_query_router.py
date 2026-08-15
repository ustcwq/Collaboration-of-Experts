from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an early-stopped query router on cached features.")
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--query-labels", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-type", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def stratified(labels: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    train, validation = [], []
    for indices in groups.values():
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction))) if len(indices) > 1 else 0
        validation.extend(indices[:count])
        train.extend(indices[count:])
    rng.shuffle(train); rng.shuffle(validation)
    return train, validation


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); criterion = nn.CrossEntropyLoss(); losses=[]; correct=total=0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits=model(features); losses.append(float(criterion(logits,labels).cpu()))
        correct += int((logits.argmax(-1)==labels).sum()); total += labels.numel()
    return {"loss":sum(losses)/max(1,len(losses)),"accuracy":correct/total if total else 0.0}


def main() -> None:
    args=parse_args(); random.seed(args.seed); torch.manual_seed(args.seed); args.output_dir.mkdir(parents=True,exist_ok=True)
    cache=torch.load(args.feature_cache,map_location="cpu",weights_only=False); lookup={str(v):i for i,v in enumerate(cache["ids"])}
    rows=read_jsonl(args.query_labels); rows=[row for row in rows if str(row["id"]) in lookup]
    features=cache["features"][[lookup[str(row["id"])] for row in rows]]; labels=[int(row["label"]) for row in rows]
    train_idx,val_idx=stratified(labels,args.validation_fraction,args.seed); label_tensor=torch.tensor(labels)
    train_loader=DataLoader(TensorDataset(features[train_idx],label_tensor[train_idx]),batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(TensorDataset(features[val_idx],label_tensor[val_idx]),batch_size=args.batch_size)
    manifest=read_json(args.label_manifest); device=torch.device(args.device)
    model=nn.Sequential(nn.LayerNorm(features.shape[1]),nn.Dropout(.15),nn.Linear(features.shape[1],manifest["num_route_labels"])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-3); criterion=nn.CrossEntropyLoss()
    best_loss=float("inf"); best_epoch=None; best_state=None; stale=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]; correct=total=0
        for batch_features,batch_labels in train_loader:
            batch_features,batch_labels=batch_features.to(device),batch_labels.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(batch_features); loss=criterion(logits,batch_labels); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu())); correct+=int((logits.argmax(-1)==batch_labels).sum()); total+=batch_labels.numel()
        validation=evaluate(model,val_loader,device); item={"epoch":epoch,"train_loss":sum(losses)/len(losses),"train_accuracy":correct/total,"validation":validation}; history.append(item); print(item)
        if validation["loss"] < best_loss-1e-6:
            best_loss=validation["loss"]; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if epoch>=args.min_epochs and stale>=args.patience: break
    torch.save({"state_dict":best_state,"feature_dim":features.shape[1],"num_labels":manifest["num_route_labels"]},args.output_dir/"classifier.pt")
    manifest.update({"router_type":args.router_type,"classifier_path":str(args.output_dir/"classifier.pt"),"feature_cache":str(args.feature_cache)})
    write_json(args.output_dir/"route_label_manifest.json",manifest)
    write_json(args.output_dir/"train_metrics.json",{"source_labeled_count":len(rows),"train_size":len(train_idx),"validation_size":len(val_idx),"best_epoch":best_epoch,"best_validation_loss":best_loss,"epochs":history})
    write_json(args.output_dir/"split_manifest.json",{"train_ids":[rows[i]["id"] for i in train_idx],"validation_ids":[rows[i]["id"] for i in val_idx]})


if __name__ == "__main__": main()
