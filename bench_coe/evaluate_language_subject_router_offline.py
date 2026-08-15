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
from bench_coe.mmlu_utils import CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an MMLU subject router on stored BBH or GPQA predictions."
    )
    parser.add_argument("--benchmark", choices=["bbh", "gpqa"], required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--route-label-manifest", type=Path, required=True)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("outputs/model_benchmarks/official_code_local_models"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id", row["question_id"]))


def router_text(row: dict[str, Any], benchmark: str) -> str:
    if benchmark == "gpqa":
        options = row.get("options", [])
        option_text = "\n".join(
            f"{CHOICES[index]}. {option}" for index, option in enumerate(options)
        )
        return f"Question:\n{row['question']}\nOptions:\n{option_text}"
    return f"Question:\n{row['input']}"


def load_prediction_matrix(
    root: Path,
    benchmark: str,
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    source_rows: list[dict[str, Any]] | None = None
    for model_name in model_names:
        path = root / benchmark / model_name / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if source_rows is None:
            source_rows = rows
        matrix[model_name] = {
            row_id(row): {
                "pred": row.get("pred"),
                "correct": bool(row.get("is_correct", False)),
            }
            for row in rows
        }
    return source_rows or [], matrix


@torch.no_grad()
def route_rows(
    rows: list[dict[str, Any]],
    benchmark: str,
    router_dir: Path,
    label_to_category: dict[int, str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(router_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(router_dir))
    model.to(device)
    model.eval()
    routed: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc=f"routing {benchmark}"):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [router_text(row, benchmark) for row in batch],
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
                    "routed_category": label_to_category[int(label)],
                    "route_confidence": float(confidence),
                }
            )
    return routed


def safe_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def true_category(row: dict[str, Any], benchmark: str) -> str | None:
    if benchmark != "gpqa":
        return None
    domain = str(row.get("domain", "")).strip().lower()
    return domain if domain in {"biology", "chemistry", "physics"} else None


def pool_model_names(
    manifest: dict[str, Any], router_dir: Path, route_to_model: dict[str, str]
) -> list[str]:
    names = manifest.get("model_names")
    if names:
        return [str(model) for model in names]
    training_manifest_path = router_dir.parent / "training_manifest.json"
    if training_manifest_path.exists():
        training_manifest = read_json(training_manifest_path)
        names = training_manifest.get("all_model_names")
        if names:
            return [str(model) for model in names]
    return sorted(set(route_to_model.values()))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.route_label_manifest)
    if "route_label_to_category" in manifest:
        label_to_category = {
            int(label): str(category)
            for label, category in manifest["route_label_to_category"].items()
        }
        category_to_model = {
            str(category): str(model)
            for category, model in manifest["category_to_model"].items()
        }
    elif "route_label_to_subject" in manifest:
        label_to_category = {
            int(label): str(subject)
            for label, subject in manifest["route_label_to_subject"].items()
        }
        category_to_model = {
            str(subject): str(model)
            for subject, model in manifest["subject_to_model"].items()
        }
    elif "route_label_to_model" in manifest:
        label_to_category = {
            int(label): str(model)
            for label, model in manifest["route_label_to_model"].items()
        }
        category_to_model = {model: model for model in label_to_category.values()}
    else:
        raise ValueError("Unsupported route manifest: expected category or subject labels.")
    pool_models = pool_model_names(manifest, args.router_dir, category_to_model)
    rows, matrix = load_prediction_matrix(
        args.predictions_root, args.benchmark, pool_models
    )
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    routed = route_rows(
        rows,
        args.benchmark,
        args.router_dir,
        label_to_category,
        choose_device(args.router_device),
        args.batch_size,
        args.max_length,
    )

    routed_correct = 0
    subject_correct = 0
    subject_total = 0
    oracle_subject_correct = 0
    routed_model_counts: Counter[str] = Counter()
    group_stats: dict[str, Counter[str]] = defaultdict(Counter)
    output_rows: list[dict[str, Any]] = []
    for row in routed:
        identifier = row_id(row)
        routed_category = str(row["routed_category"])
        routed_model = category_to_model[routed_category]
        expert_prediction = matrix[routed_model].get(identifier)
        if expert_prediction is None:
            raise KeyError(f"Missing {routed_model} prediction for {identifier}")
        is_correct = bool(expert_prediction["correct"])
        routed_correct += int(is_correct)
        routed_model_counts[routed_model] += 1
        group = str(row.get("task", row.get("domain", "all")))
        group_stats[group]["total"] += 1
        group_stats[group]["routed_correct"] += int(is_correct)

        target_category = true_category(row, args.benchmark)
        if target_category is not None and target_category not in category_to_model:
            title_target = target_category.title()
            target_category = title_target if title_target in category_to_model else None
        oracle_model = None
        oracle_correct = None
        is_subject_correct = None
        if target_category is not None:
            subject_total += 1
            is_subject_correct = routed_category == target_category
            subject_correct += int(is_subject_correct)
            oracle_model = category_to_model[target_category]
            oracle_correct = bool(matrix[oracle_model][identifier]["correct"])
            oracle_subject_correct += int(oracle_correct)
            group_stats[group]["subject_correct"] += int(is_subject_correct)
            group_stats[group]["oracle_subject_correct"] += int(oracle_correct)

        output_rows.append(
            {
                **row,
                "routed_model": routed_model,
                "routed_pred": expert_prediction["pred"],
                "routed_correct": is_correct,
                "target_category": target_category,
                "subject_correct": is_subject_correct,
                "oracle_subject_model": oracle_model,
                "oracle_subject_correct": oracle_correct,
            }
        )

    total = len(output_rows)
    single_model_accuracy = {
        model_name: safe_accuracy(
            sum(int(matrix[model_name][row_id(row)]["correct"]) for row in rows), total
        )
        for model_name in pool_models
    }
    best_single_model = max(single_model_accuracy, key=single_model_accuracy.get)
    by_group = {}
    for group, stats in sorted(group_stats.items()):
        group_total = stats["total"]
        payload = {
            "examples": group_total,
            "routed_accuracy": safe_accuracy(stats["routed_correct"], group_total),
        }
        if subject_total:
            payload["subject_accuracy"] = safe_accuracy(
                stats["subject_correct"], group_total
            )
            payload["oracle_subject_accuracy"] = safe_accuracy(
                stats["oracle_subject_correct"], group_total
            )
        by_group[group] = payload

    summary = {
        "status": "completed",
        "benchmark": args.benchmark,
        "examples": total,
        "routed_accuracy": safe_accuracy(routed_correct, total),
        "subject_accuracy": safe_accuracy(subject_correct, subject_total)
        if subject_total
        else None,
        "oracle_subject_accuracy": safe_accuracy(oracle_subject_correct, subject_total)
        if subject_total
        else None,
        "best_single_model": best_single_model,
        "best_single_accuracy": single_model_accuracy[best_single_model],
        "single_model_accuracy": single_model_accuracy,
        "routed_model_counts": dict(routed_model_counts),
        "category_to_model": category_to_model,
        "by_group": by_group,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "benchmark": args.benchmark,
            "router_dir": str(args.router_dir),
            "route_label_manifest": str(args.route_label_manifest),
            "predictions_root": str(args.predictions_root),
            "output_dir": str(args.output_dir),
        },
    )
    return summary


def main() -> None:
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
