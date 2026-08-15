from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import write_json
from bench_coe.mmlu_utils import format_mmlu_router_text, load_mmlu_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an MMLU-Pro subject router using stored expert predictions."
    )
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--route-label-manifest", type=Path, required=True)
    parser.add_argument("--mmlu-data-dir", type=Path, default=Path("MMLU-Pro/data"))
    parser.add_argument("--expert-results-root", type=Path, default=Path("MMLU-Pro/results"))
    parser.add_argument("--result-subdir", default="CoT/all")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--query-holdout-json",
        type=Path,
        default=None,
        help="Optional JSON with question_ids and target_model_by_question_id.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def route_maps(
    manifest: dict[str, Any],
) -> tuple[dict[int, str], dict[str, str], str]:
    if "route_label_to_category" in manifest:
        label_to_route = {
            int(label): str(category)
            for label, category in manifest["route_label_to_category"].items()
        }
        route_to_model = {
            str(category): str(model)
            for category, model in manifest["category_to_model"].items()
        }
        return label_to_route, route_to_model, "category"
    if "route_label_to_subject" in manifest:
        label_to_route = {
            int(label): str(subject)
            for label, subject in manifest["route_label_to_subject"].items()
        }
        route_to_model = {
            str(subject): str(model)
            for subject, model in manifest["subject_to_model"].items()
        }
        return label_to_route, route_to_model, "subject"
    if "route_label_to_model" in manifest:
        label_to_route = {
            int(label): str(model)
            for label, model in manifest["route_label_to_model"].items()
        }
        route_to_model = {model: model for model in label_to_route.values()}
        return label_to_route, route_to_model, "expert"
    raise ValueError("Unsupported route manifest: expected category or subject labels.")


def load_expert_predictions(
    root: Path,
    result_subdir: str,
    model_names: list[str],
) -> dict[str, dict[int, dict[str, Any]]]:
    predictions: dict[str, dict[int, dict[str, Any]]] = {}
    for model_name in model_names:
        result_dir = root / model_name / result_subdir
        if not result_dir.is_dir():
            raise FileNotFoundError(result_dir)
        model_predictions: dict[int, dict[str, Any]] = {}
        for path in sorted(result_dir.glob("*.json")):
            payload = read_json(path)
            if not isinstance(payload, list):
                continue
            for row in payload:
                model_predictions[int(row["question_id"])] = {
                    "pred": str(row.get("pred", "")),
                    "answer": str(row.get("answer", "")),
                    "correct": row.get("pred") == row.get("answer"),
                }
        predictions[model_name] = model_predictions
    return predictions


@torch.no_grad()
def predict_categories(
    rows: list[dict[str, Any]],
    router_dir: Path,
    label_to_route: dict[int, str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(router_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(router_dir))
    model.to(device)
    model.eval()
    routed: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="routing"):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [format_mmlu_router_text(row["question"], row["options"]) for row in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        probabilities = model(**encoded).logits.softmax(dim=-1)
        labels = probabilities.argmax(dim=-1).cpu().tolist()
        confidences = probabilities.max(dim=-1).values.cpu().tolist()
        for row, label, confidence in zip(batch, labels, confidences):
            routed.append(
                {
                    **row,
                    "route_label": int(label),
                    "routed_label_name": label_to_route[int(label)],
                    "route_confidence": float(confidence),
                }
            )
    return routed


def safe_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.route_label_manifest)
    label_to_route, route_to_model, label_kind = route_maps(manifest)
    rows = load_mmlu_split(args.mmlu_data_dir, args.split)
    query_targets: dict[str, str] = {}
    if args.query_holdout_json is not None:
        holdout = read_json(args.query_holdout_json)
        wanted_ids = {int(question_id) for question_id in holdout["question_ids"]}
        rows = [row for row in rows if int(row["question_id"]) in wanted_ids]
        query_targets = {
            str(question_id): str(model)
            for question_id, model in holdout.get(
                "target_model_by_question_id", {}
            ).items()
        }
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    expert_models = sorted(set(route_to_model.values()))
    pool_models = [str(model) for model in manifest.get("model_names", expert_models)]
    predictions = load_expert_predictions(
        args.expert_results_root, args.result_subdir, pool_models
    )
    routed_rows = predict_categories(
        rows,
        args.router_dir,
        label_to_route,
        choose_device(args.router_device),
        args.batch_size,
        args.max_length,
    )

    subject_correct = 0
    routed_correct = 0
    oracle_subject_correct = 0
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    routed_model_counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    query_route_correct = 0
    query_route_total = 0
    for row in routed_rows:
        question_id = int(row["question_id"])
        true_category = str(row["category"])
        routed_label_name = str(row["routed_label_name"])
        routed_model = route_to_model[routed_label_name]
        target_route_name = true_category if label_kind == "category" else None
        oracle_model = route_to_model[target_route_name] if target_route_name else None
        routed_prediction = predictions[routed_model].get(question_id)
        oracle_prediction = (
            predictions[oracle_model].get(question_id) if oracle_model else None
        )
        if routed_prediction is None:
            raise KeyError(f"Missing {routed_model} prediction for question {question_id}")
        if oracle_model is not None and oracle_prediction is None:
            raise KeyError(f"Missing {oracle_model} prediction for question {question_id}")

        is_subject_correct = (
            routed_label_name == target_route_name if target_route_name else None
        )
        is_routed_correct = bool(routed_prediction["correct"])
        is_oracle_subject_correct = (
            bool(oracle_prediction["correct"]) if oracle_prediction else None
        )
        subject_correct += int(bool(is_subject_correct))
        routed_correct += int(is_routed_correct)
        oracle_subject_correct += int(bool(is_oracle_subject_correct))
        routed_model_counts[routed_model] += 1
        target_query_model = query_targets.get(str(question_id))
        if target_query_model is not None:
            query_route_total += 1
            query_route_correct += int(routed_model == target_query_model)
        stats = category_stats[true_category]
        stats["total"] += 1
        stats["subject_correct"] += int(bool(is_subject_correct))
        stats["routed_correct"] += int(is_routed_correct)
        stats["oracle_subject_correct"] += int(bool(is_oracle_subject_correct))
        output_rows.append(
            {
                **row,
                "routed_model": routed_model,
                "routed_pred": routed_prediction["pred"],
                "routed_correct": is_routed_correct,
                "oracle_subject_model": oracle_model,
                "oracle_subject_pred": oracle_prediction["pred"]
                if oracle_prediction
                else None,
                "oracle_subject_correct": is_oracle_subject_correct,
                "subject_correct": is_subject_correct,
                "target_query_model": target_query_model,
                "query_route_correct": routed_model == target_query_model
                if target_query_model is not None
                else None,
            }
        )

    total = len(output_rows)
    single_model_accuracy = {}
    for model_name in pool_models:
        correct = sum(
            int(bool(predictions[model_name][int(row["question_id"])]["correct"]))
            for row in rows
        )
        single_model_accuracy[model_name] = safe_accuracy(correct, total)
    best_single_model = max(single_model_accuracy, key=single_model_accuracy.get)

    per_category = {}
    for category, stats in sorted(category_stats.items()):
        category_total = stats["total"]
        per_category[category] = {
            "examples": category_total,
            "subject_accuracy": safe_accuracy(stats["subject_correct"], category_total)
            if label_kind == "category"
            else None,
            "routed_accuracy": safe_accuracy(stats["routed_correct"], category_total),
            "oracle_subject_accuracy": safe_accuracy(
                stats["oracle_subject_correct"], category_total
            )
            if label_kind == "category"
            else None,
        }

    summary = {
        "status": "completed",
        "split": args.split,
        "examples": total,
        "label_kind": label_kind,
        "subject_accuracy": safe_accuracy(subject_correct, total)
        if label_kind == "category"
        else None,
        "routed_accuracy": safe_accuracy(routed_correct, total),
        "query_route_accuracy": safe_accuracy(query_route_correct, query_route_total)
        if query_route_total
        else None,
        "oracle_subject_accuracy": safe_accuracy(oracle_subject_correct, total)
        if label_kind == "category"
        else None,
        "best_single_model": best_single_model,
        "best_single_accuracy": single_model_accuracy[best_single_model],
        "single_model_accuracy": single_model_accuracy,
        "routed_model_counts": dict(routed_model_counts),
        "route_to_model": route_to_model,
        "per_category": per_category,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.split}_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output_dir / f"{args.split}_summary.json", summary)
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "router_dir": str(args.router_dir),
            "route_label_manifest": str(args.route_label_manifest),
            "mmlu_data_dir": str(args.mmlu_data_dir),
            "expert_results_root": str(args.expert_results_root),
            "result_subdir": args.result_subdir,
            "split": args.split,
            "output_predictions": str(predictions_path),
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
