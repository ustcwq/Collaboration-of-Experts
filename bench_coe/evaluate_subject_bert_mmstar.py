from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import format_router_text, read_json, write_json
from bench_coe.run_official_model_benchmarks import (
    extract_choice_letter,
    load_mmstar_rows,
    run_mmstar_eval_on_dataframe,
    summarize_mmstar_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GAOKAO-subject BERT router on MMStar text-only using cached expert outputs."
    )
    parser.add_argument(
        "--router-dir",
        type=Path,
        default=Path("outputs/bench_coe/router/bert-base-gaokao-subject-10epoch/model"),
    )
    parser.add_argument(
        "--route-label-manifest",
        type=Path,
        default=Path(
            "outputs/bench_coe/router/bert-base-gaokao-subject-10epoch/route_label_manifest.json"
        ),
    )
    parser.add_argument(
        "--subject-expert-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON override for subject-label routers. Accepts either "
            "{subject: model} or {'subject_to_model': {...}, 'subject_to_model_index': {...}}."
        ),
    )
    parser.add_argument(
        "--single-results-dir",
        type=Path,
        default=Path("outputs/model_benchmarks/official_code_local_models/mmstar_text_only"),
    )
    parser.add_argument("--mmstar-tsv", type=Path, default=Path("data/MMStar/MMStar.tsv"))
    parser.add_argument("--mmstar-eval-dir", type=Path, default=Path("MMStar/eval"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/mmstar_text_only_subject_bert_bench_coe_gaokao10epoch_front4"),
    )
    parser.add_argument("--router-max-length", type=int, default=256)
    parser.add_argument("--router-batch-size", type=int, default=512)
    parser.add_argument("--router-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def choose_router_device(router_device: str) -> torch.device:
    if router_device == "cpu":
        return torch.device("cpu")
    if router_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("router-device=cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_subject_expert_map(
    label_manifest: dict[str, Any],
    override_path: Path | None,
) -> dict[str, Any]:
    if override_path is None:
        return label_manifest
    if label_manifest.get("label_mode") != "subject":
        raise ValueError("--subject-expert-map can only be used with subject-label routers.")

    override = read_json(override_path)
    subject_to_model = override.get("subject_to_model", override)
    subject_to_model_index = override.get("subject_to_model_index", {})
    if not isinstance(subject_to_model, dict):
        raise ValueError("--subject-expert-map must contain a subject->model mapping.")

    merged = dict(label_manifest)
    merged["subject_to_model"] = {
        **label_manifest.get("subject_to_model", {}),
        **subject_to_model,
    }
    merged_model_index = {
        **label_manifest.get("subject_to_model_index", {}),
        **subject_to_model_index,
    }
    original_subject_to_model = label_manifest.get("subject_to_model", {})
    for subject, model_name in subject_to_model.items():
        if (
            subject not in subject_to_model_index
            and original_subject_to_model.get(subject) != model_name
        ):
            merged_model_index[subject] = -1
    merged["subject_to_model_index"] = merged_model_index
    return merged


def resolve_route_label(label_manifest: dict[str, Any], label: int) -> tuple[str, str, int]:
    if label_manifest.get("label_mode") != "subject":
        raise ValueError("MMStar evaluator expects a subject-label router manifest.")
    label_key = str(label)
    subject = label_manifest["route_label_to_subject"][label_key]
    model_name = label_manifest.get("subject_to_model", {}).get(subject)
    if not model_name:
        model_name = label_manifest.get("route_label_to_model", {}).get(label_key)
    if not model_name:
        raise KeyError(f"Missing subject_to_model entry for routed subject: {subject}")
    model_index_raw = label_manifest.get("subject_to_model_index", {}).get(subject)
    if model_index_raw is None:
        model_index_raw = label_manifest.get("route_label_to_model_index", {}).get(label_key, -1)
    return subject, model_name, int(model_index_raw)


@torch.no_grad()
def route_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    label_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(args.router_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(args.router_dir))
    device = choose_router_device(args.router_device)
    model.to(device)
    model.eval()

    routed_rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), args.router_batch_size), desc="routing"):
        batch_rows = rows[start : start + args.router_batch_size]
        texts = [format_router_text(str(row["question"])) for row in batch_rows]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=args.router_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        probs = model(**encoded).logits.softmax(dim=-1)
        labels = probs.argmax(dim=-1).detach().cpu().tolist()
        confidences = probs.max(dim=-1).values.detach().cpu().tolist()
        for row, label, confidence in zip(batch_rows, labels, confidences):
            subject, model_name, model_index = resolve_route_label(label_manifest, int(label))
            routed = dict(row)
            routed["route_label"] = int(label)
            routed["route_confidence"] = float(confidence)
            routed["routed_subject"] = subject
            routed["routed_model"] = model_name
            routed["routed_model_index"] = model_index
            routed_rows.append(routed)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return routed_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cached_predictions(single_results_dir: Path, model_names: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    caches: dict[str, dict[int, dict[str, Any]]] = {}
    missing: list[Path] = []
    for model_name in sorted(set(model_names)):
        path = single_results_dir / model_name / "predictions.jsonl"
        if not path.exists():
            missing.append(path)
            continue
        by_qid: dict[int, dict[str, Any]] = {}
        for row in read_jsonl(path):
            by_qid[int(row["question_id"])] = row
        caches[model_name] = by_qid
    if missing:
        raise FileNotFoundError(
            "Missing cached MMStar predictions for routed expert(s):\n"
            + "\n".join(str(path) for path in missing)
        )
    return caches


def combine_cached_predictions(
    routed_rows: list[dict[str, Any]],
    caches: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for row in routed_rows:
        model_name = row["routed_model"]
        qid = int(row["question_id"])
        cached = caches[model_name].get(qid)
        if cached is None:
            raise KeyError(f"Missing cached prediction for model={model_name}, question_id={qid}")
        pred = cached.get("pred")
        if pred is None:
            pred = extract_choice_letter(str(cached.get("model_outputs", "")))
        result = dict(row)
        result["pred"] = pred
        result["is_correct"] = pred == row["answer"]
        result["model_outputs"] = cached.get("model_outputs", "")
        result["expert_prediction_file"] = str(
            Path(model_name) / "predictions.jsonl"
        )
        result["expert_prompt_was_truncated"] = cached.get("prompt_was_truncated")
        result["expert_prompt_token_count"] = cached.get("prompt_token_count")
        combined.append(result)
    return combined


def stats_dict() -> dict[str, float]:
    return {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}


def finalize_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for item in stats.values():
        denom = item["correct"] + item["wrong"]
        item["accuracy"] = item["correct"] / denom if denom else 0.0
    return dict(sorted(stats.items()))


def add_routing_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    subject_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    for row in rows:
        targets = [model_stats[row["routed_model"]], subject_stats[row["routed_subject"]]]
        if row.get("is_correct"):
            for stats in targets:
                stats["correct"] += 1
        else:
            for stats in targets:
                stats["wrong"] += 1
    summary["routed_model"] = finalize_stats(model_stats)
    summary["routed_subject"] = finalize_stats(subject_stats)
    return summary


def build_mmstar_result_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "l2_category": row["l2_category"],
                "bench": row["bench"],
                "prediction": row.get("pred") or "",
                "model_outputs": row.get("model_outputs", ""),
                "routed_model": row.get("routed_model", ""),
                "routed_subject": row.get("routed_subject", ""),
            }
            for row in rows
        ]
    )


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def row_line(name_width: int, col_width: int, name: str, values: list[str]) -> str:
    cells = [name.ljust(name_width)]
    cells.extend(value.ljust(col_width) for value in values)
    return "| " + " | ".join(cells) + " |"


def total_count(stats: dict[str, Any]) -> int:
    return int(float(stats.get("correct", 0)) + float(stats.get("wrong", 0)))


def single_model_summaries(single_results_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(single_results_dir.glob("*/summary.json")):
        summary = read_json(summary_path)
        if summary.get("status") == "completed" and summary.get("accuracy") is not None:
            summaries.append(summary)
    summaries.sort(key=lambda item: (-float(item["accuracy"]), str(item["model"])))
    return summaries


def render_bench_harness_txt(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    single_summaries: list[dict[str, Any]],
    single_results_dir: Path,
) -> None:
    category_order = [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "math",
        "science & technology",
    ]
    available = set(summary["by_category"])
    categories = [category for category in category_order if category in available]
    categories.extend(sorted(available.difference(categories)))
    columns = categories + ["Average"]
    col_width = 20
    name_width = 34
    best = single_summaries[0] if single_summaries else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(summary["accuracy"])

    counts = [str(total_count(summary["by_category"][category])) for category in categories]
    counts.append(str(int(summary["num_examples"])))
    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO prior -> MMStar",
        "=" * 100,
        "| Routing Mode: bert_gaokao_subject",
        "| Benchmark: MMStar",
        "| Split: test",
        "| Mode: text-only",
        f"| Samples: {int(summary['num_examples'])}",
        f"| Single model source: {single_results_dir}",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for single in single_summaries:
        model_name = str(single["model"])
        prefix = "* " if model_name == best_model else "  "
        values = []
        for category in categories:
            stats = single.get("by_category", {}).get(category)
            values.append(format_percent(float(stats["accuracy"])) if stats else "N/A")
        values.append(format_percent(float(single["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))

    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(float(summary["by_category"][category]["accuracy"])) for category in categories]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, "GAOKAO-Bert-Bench-CoE", coe_values))
    gain_values = [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", gain_values))
    lines.append("")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in rows).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO subjects:")
    for subject, count in Counter(row["routed_subject"] for row in rows).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {path.parent / 'test_predictions.json'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leaderboard(path: Path, summary: dict[str, Any], single_summaries: list[dict[str, Any]]) -> None:
    rows = [
        {
            "model": single["model"],
            "benchmark": "MMStar",
            "mode": "text_only",
            "accuracy": single["accuracy"],
            "correct": single["correct"],
            "total": single["num_examples"],
        }
        for single in single_summaries
    ]
    rows.append(
        {
            "model": "GAOKAO-Bert-Bench-CoE",
            "benchmark": "MMStar",
            "mode": "text_only_routed",
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["num_examples"],
        }
    )
    rows.sort(key=lambda row: (-float(row["accuracy"]), str(row["model"])))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output_dir / "test_predictions.json"
    if args.resume and output_path.exists() and (args.output_dir / "test_summary.json").exists():
        rows = read_json(output_path)
        summary = read_json(args.output_dir / "test_summary.json")
        render_bench_harness_txt(
            args.output_dir / "Bench_Harness_Result_gaokao_router_mmstar.txt",
            rows,
            summary,
            single_model_summaries(args.single_results_dir),
            args.single_results_dir,
        )
        return summary

    label_manifest = apply_subject_expert_map(read_json(args.route_label_manifest), args.subject_expert_map)
    rows = load_mmstar_rows(args)
    routed_rows = route_rows(rows, args, label_manifest)
    caches = load_cached_predictions(
        args.single_results_dir,
        [row["routed_model"] for row in routed_rows],
    )
    final_rows = combine_cached_predictions(routed_rows, caches)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, final_rows)
    result_df = build_mmstar_result_dataframe(final_rows)
    csv_path = args.output_dir / "GAOKAO-Bert-Bench-CoE_MMStar.csv"
    xlsx_path = args.output_dir / "GAOKAO-Bert-Bench-CoE_MMStar.xlsx"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        result_df.to_excel(xlsx_path, index=False)
    except Exception:
        pass

    summary = summarize_mmstar_fallback("GAOKAO-Bert-Bench-CoE", final_rows, args)
    summary = add_routing_summary(summary, final_rows)
    summary["split"] = "test"
    summary["mode"] = "text_only_routed"
    summary["examples"] = len(final_rows)
    summary["prediction_file"] = str(output_path)
    summary["single_results_dir"] = str(args.single_results_dir)
    try:
        eval_file = xlsx_path if xlsx_path.exists() else csv_path
        score_payload, score_file = run_mmstar_eval_on_dataframe(args, result_df, eval_file)
        summary["official_mmstar_result_file"] = str(csv_path)
        if xlsx_path.exists():
            summary["official_mmstar_xlsx_file"] = str(xlsx_path)
        summary["official_mmstar_score_file"] = str(score_file)
        summary["official_mmstar_scores"] = score_payload
        summary["accuracy"] = float(score_payload.get("final score", summary["accuracy"]))
        summary["evaluation"] = "MMStar_eval"
    except Exception as exc:
        summary["evaluation_warning"] = f"MMStar_eval failed; fallback scores kept: {exc}"

    write_json(args.output_dir / "test_summary.json", summary)
    pd.DataFrame(summary.get("by_category", {})).T.to_csv(args.output_dir / "category_summary.csv")
    single_summaries = single_model_summaries(args.single_results_dir)
    render_bench_harness_txt(
        args.output_dir / "Bench_Harness_Result_gaokao_router_mmstar.txt",
        final_rows,
        summary,
        single_summaries,
        args.single_results_dir,
    )
    write_leaderboard(args.output_dir / "mmstar_text_only_leaderboard.csv", summary, single_summaries)
    return summary


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    summary = evaluate(args)
    print(
        "MMStar Bench-CoE completed: "
        f"accuracy={summary['accuracy'] * 100:.2f}% "
        f"correct={int(summary['correct'])}/{int(summary['num_examples'])}"
    )
    print(f"Outputs are under {args.output_dir}")


if __name__ == "__main__":
    main()
