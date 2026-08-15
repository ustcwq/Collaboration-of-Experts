from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Bench-Harness-style BBH/GPQA comparison files with single models and Bench-CoE."
    )
    parser.add_argument(
        "--single-root",
        type=Path,
        default=Path("outputs/model_benchmarks/official_code_local_models"),
    )
    parser.add_argument(
        "--bbh-coe-dir",
        type=Path,
        default=Path("outputs/bench_coe/bbh_subject_bert_bench_coe_gaokao10epoch_front4"),
    )
    parser.add_argument(
        "--gpqa-coe-dir",
        type=Path,
        default=Path("outputs/bench_coe/gpqa_diamond_subject_bert_bench_coe_gaokao10epoch_front4"),
    )
    parser.add_argument(
        "--mmstar-output-dir",
        type=Path,
        default=Path("outputs/bench_coe/mmstar_text_only_model_comparison"),
    )
    parser.add_argument("--gpqa-config", default="diamond")
    parser.add_argument("--coe-name", default="GAOKAO-Bert-Bench-CoE")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


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


def render_bbh(args: argparse.Namespace) -> None:
    coe_dir = args.bbh_coe_dir
    coe_summary = read_json(coe_dir / "test_summary.json")
    coe_predictions = read_json(coe_dir / "test_predictions.json")
    tasks = sorted(coe_summary["task"])
    columns = tasks + ["Average"]
    col_width = 15
    name_width = 34

    single_rows: list[dict[str, Any]] = []
    single_bbh_dir = args.single_root / "bbh"
    for summary_path in sorted(single_bbh_dir.glob("*/summary.json")):
        summary = read_json(summary_path)
        if summary.get("status") != "completed" or summary.get("accuracy") is None:
            continue
        single_rows.append(summary)
    single_rows.sort(key=lambda item: float(item["accuracy"]), reverse=True)
    best = single_rows[0] if single_rows else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(coe_summary["total"]["accuracy"])

    question_counts = [str(total_count(coe_summary["task"][task])) for task in tasks]
    question_counts.append(str(int(coe_summary["examples"])))

    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO prior -> BBH",
        "=" * 100,
        "| Routing Mode: bert_gaokao_subject",
        "| Benchmark: BBH",
        "| Split: test",
        f"| Samples: {int(coe_summary['examples'])}",
        f"| Single model source: {single_bbh_dir}",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", question_counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for summary in single_rows:
        model_name = str(summary["model"])
        prefix = "* " if model_name == best_model else "  "
        values = [
            format_percent(float(summary.get("by_task", {}).get(task, {}).get("accuracy")))
            if task in summary.get("by_task", {})
            else "N/A"
            for task in tasks
        ]
        values.append(format_percent(float(summary["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))

    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(float(coe_summary["task"][task]["accuracy"])) for task in tasks]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    gain_values = [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", gain_values))
    lines.append("")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in coe_predictions).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO subjects:")
    for subject, count in Counter(row["routed_subject"] for row in coe_predictions).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {coe_dir / 'test_predictions.json'}")
    write_text(coe_dir / "Bench_Harness_Result_gaokao_router_bbh.txt", lines)


def aggregate_gpqa_diamond(path: Path, config: str) -> dict[str, Any] | None:
    rows = [row for row in read_jsonl(path) if row.get("config") == config]
    if not rows:
        return None
    by_domain: dict[str, dict[str, float]] = defaultdict(lambda: {"correct": 0.0, "wrong": 0.0})
    correct = 0.0
    wrong = 0.0
    unique_questions = set()
    for row in rows:
        domain = str(row.get("domain", "Unknown"))
        unique_questions.add(row.get("base_question_id", row.get("record_id", row.get("question_id"))))
        is_correct = bool(row.get("is_correct"))
        if is_correct:
            correct += 1.0
            by_domain[domain]["correct"] += 1.0
        else:
            wrong += 1.0
            by_domain[domain]["wrong"] += 1.0
    for stats in by_domain.values():
        denom = stats["correct"] + stats["wrong"]
        stats["accuracy"] = stats["correct"] / denom if denom else 0.0
    model = path.parent.name
    denom = correct + wrong
    return {
        "model": model,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / denom if denom else None,
        "num_examples": int(denom),
        "unique_questions": len(unique_questions),
        "by_domain": dict(by_domain),
    }


def render_gpqa(args: argparse.Namespace) -> None:
    coe_dir = args.gpqa_coe_dir
    coe_summary = read_json(coe_dir / "test_summary.json")
    coe_predictions = read_json(coe_dir / "test_predictions.json")
    domains = sorted(coe_summary["domain"])
    columns = domains + ["Average"]
    col_width = 15
    name_width = 34

    single_rows: list[dict[str, Any]] = []
    single_gpqa_dir = args.single_root / "gpqa"
    for predictions_path in sorted(single_gpqa_dir.glob("*/predictions.jsonl")):
        summary = aggregate_gpqa_diamond(predictions_path, args.gpqa_config)
        if summary is not None and summary.get("accuracy") is not None:
            single_rows.append(summary)
    single_rows.sort(key=lambda item: float(item["accuracy"]), reverse=True)
    best = single_rows[0] if single_rows else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(coe_summary["total"]["accuracy"])

    counts = [str(total_count(coe_summary["domain"][domain])) for domain in domains]
    counts.append(str(int(coe_summary["examples"])))

    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO prior -> GPQA",
        "=" * 100,
        "| Routing Mode: bert_gaokao_subject",
        "| Benchmark: GPQA",
        f"| Configs: {', '.join(sorted(coe_summary['config']))}",
        f"| Samples: {int(coe_summary['examples'])}",
        f"| Unique Questions: {int(coe_summary['unique_questions'])}",
        f"| Single model source: {single_gpqa_dir} (filtered to config={args.gpqa_config})",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for summary in single_rows:
        model_name = str(summary["model"])
        prefix = "* " if model_name == best_model else "  "
        values = []
        for domain in domains:
            stats = summary["by_domain"].get(domain)
            values.append(format_percent(float(stats["accuracy"])) if stats else "N/A")
        values.append(format_percent(float(summary["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))

    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(float(coe_summary["domain"][domain]["accuracy"])) for domain in domains]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    gain_values = [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", gain_values))
    lines.append("")
    lines.append("Config accuracy:")
    for config, stats in coe_summary["config"].items():
        lines.append(f"- {config}: {format_percent(float(stats['accuracy']))}")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in coe_predictions).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO subjects:")
    for subject, count in Counter(row["routed_subject"] for row in coe_predictions).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {coe_dir / 'test_predictions.json'}")
    write_text(coe_dir / "Bench_Harness_Result_gaokao_router_gpqa.txt", lines)


def render_mmstar(args: argparse.Namespace) -> None:
    single_mmstar_dir = args.single_root / "mmstar_text_only"
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(single_mmstar_dir.glob("*/summary.json")):
        summary = read_json(summary_path)
        if summary.get("status") != "completed" or summary.get("accuracy") is None:
            continue
        summaries.append(summary)
    summaries.sort(key=lambda item: float(item["accuracy"]), reverse=True)
    if not summaries:
        return

    category_order = [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "math",
        "science & technology",
    ]
    available = set().union(*(summary.get("by_category", {}).keys() for summary in summaries))
    categories = [category for category in category_order if category in available]
    categories.extend(sorted(available.difference(categories)))
    columns = categories + ["Average"]
    col_width = 20
    name_width = 34
    best_model = summaries[0]["model"]
    best = summaries[0]

    question_counts = []
    for category in categories:
        stats = best.get("by_category", {}).get(category, {})
        question_counts.append(str(total_count(stats)) if stats else "N/A")
    question_counts.append(str(int(best["num_examples"])))

    lines = [
        "=" * 100,
        "Bench-Harness: Local models -> MMStar",
        "=" * 100,
        "| Routing Mode: N/A (single-model text-only evaluation)",
        "| Benchmark: MMStar",
        "| Split: test",
        f"| Samples: {int(best['num_examples'])}",
        f"| Single model source: {single_mmstar_dir}",
        "| Note: no MMStar Bench-CoE routed result was found under outputs/bench_coe; this file reports the existing single-model MMStar results.",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", question_counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for summary in summaries:
        model_name = str(summary["model"])
        prefix = "* " if model_name == best_model else "  "
        values = []
        for category in categories:
            stats = summary.get("by_category", {}).get(category)
            values.append(format_percent(float(stats["accuracy"])) if stats else "N/A")
        values.append(format_percent(float(summary["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))
    lines.append("")
    lines.append("Per-model detailed outputs:")
    lines.append(f"- Summary JSONs: {single_mmstar_dir}/<model>/summary.json")
    lines.append(f"- Official MMStar CSVs: {single_mmstar_dir}/<model>/<model>_MMStar.csv")

    out_dir = args.mmstar_output_dir
    write_text(out_dir / "Bench_Harness_Result_mmstar_text_only.txt", lines)
    rows = [
        {
            "model": summary["model"],
            "benchmark": "MMStar",
            "mode": "text_only",
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["num_examples"],
        }
        for summary in summaries
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "mmstar_text_only_leaderboard.csv").open("w", encoding="utf-8") as f:
        f.write("model,benchmark,mode,accuracy,correct,total\n")
        for row in rows:
            f.write(
                f"{row['model']},{row['benchmark']},{row['mode']},"
                f"{row['accuracy']},{row['correct']},{row['total']}\n"
            )


def main() -> None:
    args = parse_args()
    render_bbh(args)
    render_gpqa(args)
    render_mmstar(args)


if __name__ == "__main__":
    main()
