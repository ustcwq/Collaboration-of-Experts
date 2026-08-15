from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from bench_coe.gaokao_utils import (
    SUBJECT_ORDER,
    build_gaokao_scores,
    build_subject_winners,
    discover_result_models,
    filter_local_models,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GAOKAO-Bench-2010-2022 subject accuracy tables."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("GAOKAO-Bench-2010-2022/Data"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/bench_coe/gaokao")
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model names. Defaults to all discovered result models.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Keep only models that also have local weights under --models-dir.",
    )
    parser.add_argument(
        "--reextract-empty",
        action="store_true",
        help="Try to extract choices from model_output when model_answer is empty.",
    )
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def write_markdown_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(table.to_markdown(floatfmt=".3f"))
        f.write("\n")


def render_heatmap(path: Path, table: pd.DataFrame, winners: dict[str, Any]) -> None:
    values = table[SUBJECT_ORDER].fillna(0.0)
    n_rows, n_cols = values.shape
    fig_w = max(12.0, 1.15 * n_cols + 5.0)
    fig_h = max(7.0, 0.45 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_cols), labels=values.columns, rotation=35, ha="right")
    ax.set_yticks(range(n_rows), labels=values.index)
    ax.set_title("GAOKAO-Bench-2010-2022 Subject Accuracy")
    ax.tick_params(axis="both", labelsize=9)

    winner_cells = {
        (winner["selected_model"], subject) for subject, winner in winners.items()
    }
    for row_idx, model in enumerate(values.index):
        for col_idx, subject in enumerate(values.columns):
            val = values.iloc[row_idx, col_idx]
            is_winner = (model, subject) in winner_cells
            color = "white" if val >= 0.62 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                fontweight="bold" if is_winner else "normal",
            )
            if is_winner:
                rect = plt.Rectangle(
                    (col_idx - 0.5, row_idx - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#d62728",
                    linewidth=2,
                )
                ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Accuracy", rotation=270, labelpad=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_names = args.models or discover_result_models(args.data_dir)
    if args.local_only:
        model_names = filter_local_models(model_names, args.models_dir)
    model_names = sorted(model_names)
    if not model_names:
        raise SystemExit("No models found.")

    task_rows, subject_rows = build_gaokao_scores(
        args.data_dir, model_names=model_names, reextract_empty=args.reextract_empty
    )
    model_to_index = {name: idx for idx, name in enumerate(model_names)}
    winners = build_subject_winners(subject_rows, model_to_index)

    prefix = args.prefix
    if prefix is None:
        prefix = "local" if args.local_only else "all"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_df = pd.DataFrame(task_rows)
    subject_df = pd.DataFrame(subject_rows)
    subject_df.insert(0, "model_index", subject_df["model"].map(model_to_index))
    task_df.insert(0, "model_index", task_df["model"].map(model_to_index))

    task_df.to_csv(args.output_dir / f"{prefix}_accuracy_by_task.csv", index=False)
    subject_df.to_csv(args.output_dir / f"{prefix}_accuracy_by_subject_long.csv", index=False)

    pivot = (
        subject_df.pivot(index="model", columns="subject", values="accuracy")
        .reindex(index=model_names, columns=SUBJECT_ORDER)
        .reset_index()
    )
    pivot.insert(0, "model_index", pivot["model"].map(model_to_index))
    pivot.to_csv(args.output_dir / f"{prefix}_accuracy_by_subject.csv", index=False)
    write_markdown_table(
        args.output_dir / f"{prefix}_accuracy_by_subject.md",
        pivot.set_index(["model_index", "model"]),
    )
    render_heatmap(
        args.output_dir / f"{prefix}_accuracy_by_subject.png",
        pivot.set_index("model"),
        winners,
    )

    manifest = {
        "data_dir": str(args.data_dir),
        "models_dir": str(args.models_dir),
        "local_only": args.local_only,
        "reextract_empty": args.reextract_empty,
        "model_names": model_names,
        "model_to_index": model_to_index,
        "subject_winners": winners,
        "outputs": {
            "task_csv": str(args.output_dir / f"{prefix}_accuracy_by_task.csv"),
            "subject_long_csv": str(
                args.output_dir / f"{prefix}_accuracy_by_subject_long.csv"
            ),
            "subject_csv": str(args.output_dir / f"{prefix}_accuracy_by_subject.csv"),
            "subject_md": str(args.output_dir / f"{prefix}_accuracy_by_subject.md"),
            "subject_png": str(args.output_dir / f"{prefix}_accuracy_by_subject.png"),
        },
    }
    write_json(args.output_dir / f"{prefix}_expert_subject_mapping.json", manifest)

    print(f"Models: {len(model_names)}")
    print(f"Subjects: {len(winners)}")
    print(f"Wrote {args.output_dir / f'{prefix}_accuracy_by_subject.csv'}")
    print(f"Wrote {args.output_dir / f'{prefix}_expert_subject_mapping.json'}")


if __name__ == "__main__":
    main()

