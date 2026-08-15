from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from bench_coe.expert_pool import load_expert_pool, select_expert_pool_models
from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multimodal subject leaderboard from per-model summaries."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--expert-pool-config", type=Path, required=True)
    parser.add_argument("--expert-pool", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--include-ids-json",
        type=Path,
        help="Optional ID list or split manifest whose train_ids define the leaderboard subset.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_summaries(root: Path) -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(root.glob("*/summary.json"))
        if path.is_file()
    }


def load_include_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = read_json(path)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get("train_ids"), list):
        values = payload["train_ids"]
    elif isinstance(payload, dict) and isinstance(payload.get("ids"), list):
        values = payload["ids"]
    else:
        raise ValueError(f"Unsupported include-ID payload: {path}")
    return {str(value) for value in values}


def filtered_statistics(
    predictions_path: Path, include_ids: set[str]
) -> tuple[float, dict[str, dict[str, float | int]]]:
    total = 0
    correct = 0
    by_subject: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with predictions_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["id"]) not in include_ids:
                continue
            subject = str(row["subject"])
            is_correct = bool(row.get("is_correct", False))
            total += 1
            correct += int(is_correct)
            by_subject[subject][0] += 1
            by_subject[subject][1] += int(is_correct)
    if total != len(include_ids):
        raise ValueError(
            f"{predictions_path} matched {total} rows, expected {len(include_ids)} IDs"
        )
    return correct / total, {
        subject: {
            "accuracy": counts[1] / counts[0],
            "correct": counts[1],
            "total": counts[0],
        }
        for subject, counts in by_subject.items()
    }


def build_winners(
    long_rows: list[dict[str, Any]],
    overall: dict[str, float],
    model_to_index: dict[str, int],
) -> dict[str, dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        by_subject[row["subject"]].append(row)
    winners = {}
    for subject in sorted(by_subject):
        rows = by_subject[subject]
        best_accuracy = max(float(row["accuracy"]) for row in rows)
        tied = [row for row in rows if abs(float(row["accuracy"]) - best_accuracy) <= 1e-12]
        tied.sort(
            key=lambda row: (-overall[row["model"]], model_to_index[row["model"]])
        )
        selected = tied[0]
        winners[subject] = {
            "subject": subject,
            "best_accuracy": best_accuracy,
            "best_models": [
                {
                    "model": row["model"],
                    "model_index": model_to_index[row["model"]],
                    "accuracy": float(row["accuracy"]),
                    "overall_accuracy": overall[row["model"]],
                }
                for row in tied
            ],
            "selected_model": selected["model"],
            "selected_model_index": model_to_index[selected["model"]],
            "selected_model_overall_accuracy": overall[selected["model"]],
            "tie_break_policy": "highest_overall_accuracy_then_model_index",
        }
    return winners


def render(path: Path, table: pd.DataFrame, subjects: list[str], winners: dict[str, Any]) -> None:
    values = table.set_index("model")[["overall", *subjects]].fillna(0.0)
    fig, ax = plt.subplots(
        figsize=(max(14.0, 0.55 * len(subjects) + 5.0), max(5.0, 0.55 * len(values) + 2.0))
    )
    image = ax.imshow(values.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(values.columns)), labels=values.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(values.index)), labels=values.index)
    winner_cells = {(item["selected_model"], subject) for subject, item in winners.items()}
    for row_index, model in enumerate(values.index):
        for column_index, metric in enumerate(values.columns):
            value = float(values.iloc[row_index, column_index])
            is_winner = (model, metric) in winner_cells
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 0.58 else "black",
                fontweight="bold" if is_winner else "normal",
            )
            if is_winner:
                ax.add_patch(
                    plt.Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d62728",
                        linewidth=2,
                    )
                )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02).set_label("Accuracy")
    ax.set_title("Multimodal Subject Performance Leaderboard")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summaries = discover_summaries(args.results_root)
    include_ids = load_include_ids(args.include_ids_json)
    pool = load_expert_pool(args.expert_pool_config, args.expert_pool)
    model_names, pool_report, metadata = select_expert_pool_models(
        sorted(summaries), pool
    )
    overall: dict[str, float] = {}
    long_rows: list[dict[str, Any]] = []
    subjects: set[str] = set()
    for model in model_names:
        if include_ids is None:
            payload = read_json(summaries[model])
            model_overall = float(payload["accuracy"])
            by_subject = payload.get("by_subject", {})
        else:
            model_overall, by_subject = filtered_statistics(
                summaries[model].parent / "predictions.jsonl", include_ids
            )
        overall[model] = model_overall
        for subject, stats in by_subject.items():
            subjects.add(str(subject))
            long_rows.append(
                {
                    "model": model,
                    "subject": str(subject),
                    "accuracy": float(stats["accuracy"]),
                    "correct": int(stats["correct"]),
                    "total": int(stats["total"]),
                }
            )
    subject_order = sorted(subjects)
    model_names = sorted(model_names)
    model_to_index = {model: index for index, model in enumerate(model_names)}
    winners = build_winners(long_rows, overall, model_to_index)
    pivot = pd.DataFrame(long_rows).pivot(index="model", columns="subject", values="accuracy")
    pivot = pivot.reindex(index=model_names, columns=subject_order)
    pivot.insert(0, "overall", pd.Series(overall))
    table = pivot.reset_index()
    table.insert(1, "model_size_b", table["model"].map(lambda model: metadata[model].get("parameters_b")))
    table.insert(2, "data_source", table["model"].map(lambda model: metadata[model].get("data_source", "unknown")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"{args.prefix}_leaderboard.csv", index=False)
    (args.output_dir / f"{args.prefix}_leaderboard.md").write_text(
        table.to_markdown(index=False, floatfmt=".3f") + "\n", encoding="utf-8"
    )
    pd.DataFrame(long_rows).to_csv(
        args.output_dir / f"{args.prefix}_accuracy_by_subject_long.csv", index=False
    )
    render(args.output_dir / f"{args.prefix}_leaderboard.png", table, subject_order, winners)
    write_json(
        args.output_dir / f"{args.prefix}_expert_subject_mapping.json",
        {
            "results_root": str(args.results_root),
            "include_ids_json": str(args.include_ids_json) if args.include_ids_json else None,
            "sample_count": len(include_ids) if include_ids is not None else None,
            "expert_pool": pool_report,
            "model_names": model_names,
            "model_to_index": model_to_index,
            "overall_accuracy": overall,
            "subject_winners": winners,
            "subjects": subject_order,
        },
    )
    print(f"Models: {len(model_names)}")
    print(f"Subjects: {len(subject_order)}")


if __name__ == "__main__":
    main()
