from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bench_coe.mmlu_utils import (
    MMLU_CATEGORY_ORDER,
    discover_mmlu_summaries,
    read_mmlu_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Bench-Harness-style txt comparison for MMLU-Pro CoE results."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument(
        "--single-summary-dir",
        type=Path,
        default=Path("MMLU-Pro/results/summary"),
    )
    parser.add_argument("--output-txt", type=Path, required=True)
    parser.add_argument("--routing-mode", default="bert_gaokao_subject")
    parser.add_argument("--coe-name", default="GAOKAO-Bert-Bench-CoE")
    parser.add_argument("--benchmark", default="MMLU-Pro")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def category_accuracy(summary: dict[str, Any], category: str) -> float | None:
    stats = summary.get("category", {}).get(category)
    if stats is None:
        return None
    return float(stats["accuracy"])


def single_category_accuracy(summary: dict[str, Any], category: str) -> float | None:
    value = summary.get("category_accuracy", {}).get(category)
    if value is None:
        return None
    return float(value)


def count_rows(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "Unknown")) for row in rows)


def main() -> None:
    args = parse_args()
    coe_summary = read_json(args.summary_json)
    predictions = read_json(args.predictions_json)

    summaries: list[dict[str, Any]] = []
    for path in discover_mmlu_summaries(args.single_summary_dir).values():
        summary = read_mmlu_summary(path)
        if summary["overall_accuracy"] is not None:
            summaries.append(summary)
    summaries.sort(key=lambda item: float(item["overall_accuracy"]), reverse=True)

    categories = [category for category in MMLU_CATEGORY_ORDER if category in coe_summary["category"]]
    columns = categories + ["Average"]
    col_width = 17
    name_width = 36

    def row_line(name: str, values: list[str]) -> str:
        cells = [name.ljust(name_width)]
        cells.extend(value.ljust(col_width) for value in values)
        return "| " + " | ".join(cells) + " |"

    category_counts = [
        str(int(coe_summary["category"][category]["correct"] + coe_summary["category"][category]["wrong"]))
        for category in categories
    ]
    category_counts.append(str(int(coe_summary["examples"])))

    best_summary = summaries[0] if summaries else None
    best_model = best_summary["model"] if best_summary else None
    best_acc = float(best_summary["overall_accuracy"]) if best_summary else 0.0
    coe_acc = float(coe_summary["total"]["accuracy"])

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Bench-Harness: GAOKAO prior -> MMLU-Pro")
    lines.append("=" * 100)
    lines.append(f"| Routing Mode: {args.routing_mode}")
    lines.append(f"| Benchmark: {args.benchmark}")
    lines.append(f"| Split: {args.split}")
    lines.append(f"| Samples: {int(coe_summary['examples'])}")
    lines.append("")
    lines.append(row_line("Model / Metric", columns))
    lines.append(row_line("-" * 30, ["-" * 12 for _ in columns]))
    lines.append(row_line("Qs (Count)", category_counts))
    lines.append(row_line("-" * 30, ["-" * 12 for _ in columns]))

    for summary in summaries:
        model_name = str(summary["model"])
        prefix = "* " if model_name == best_model else "  "
        values = [format_percent(single_category_accuracy(summary, category)) for category in categories]
        values.append(format_percent(float(summary["overall_accuracy"])))
        lines.append(row_line(prefix + model_name, values))

    lines.append(row_line("-" * 30, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(category_accuracy(coe_summary, category)) for category in categories]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(args.coe_name, coe_values))
    gain_values = [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]
    lines.append(row_line("Gain (vs Best Exp)", gain_values))
    lines.append("")

    lines.append("Routed models:")
    for model_name, count in count_rows(predictions, "routed_model").most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO subjects:")
    for subject, count in count_rows(predictions, "routed_subject").most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {args.predictions_json}")

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


if __name__ == "__main__":
    main()
