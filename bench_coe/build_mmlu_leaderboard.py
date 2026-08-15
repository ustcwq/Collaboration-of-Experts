from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from bench_coe.gaokao_utils import filter_local_models, write_json
from bench_coe.expert_pool import load_expert_pool, select_expert_pool_models
from bench_coe.mmlu_utils import (
    MMLU_CATEGORY_ORDER,
    build_category_winners,
    build_mmlu_evaluation_rows,
    build_mmlu_category_rows,
    discover_mmlu_evaluation_summaries,
    discover_mmlu_summaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MMLU-Pro category accuracy tables for expert routing."
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("MMLU-Pro/results/summary"),
    )
    parser.add_argument(
        "--evaluation-summary-root",
        type=Path,
        default=None,
        help="Use per-model summary_validation.json files instead of benchmark test summaries.",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/mmlu"),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model names. Defaults to all discovered summaries.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Keep only models that also have local weights under --models-dir.",
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--expert-pool-config", type=Path, default=None)
    parser.add_argument("--expert-pool", default=None)
    return parser.parse_args()


def write_markdown_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(table.to_markdown(floatfmt=".3f"))
        f.write("\n")


def render_heatmap(path: Path, table: pd.DataFrame, winners: dict[str, Any]) -> None:
    metric_columns = ["overall", *MMLU_CATEGORY_ORDER]
    values = table[metric_columns].fillna(0.0)
    n_rows, n_cols = values.shape
    fig_w = max(13.0, 0.95 * n_cols + 5.0)
    fig_h = max(7.0, 0.45 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_cols), labels=values.columns, rotation=35, ha="right")
    ax.set_yticks(range(n_rows), labels=values.index)
    ax.set_title("MMLU-Pro Subject Performance Leaderboard")
    ax.tick_params(axis="both", labelsize=9)

    winner_cells = {
        (winner["selected_model"], category) for category, winner in winners.items()
    }
    for row_idx, model in enumerate(values.index):
        for col_idx, category in enumerate(values.columns):
            val = values.iloc[row_idx, col_idx]
            is_winner = (model, category) in winner_cells
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
    if bool(args.expert_pool_config) != bool(args.expert_pool):
        raise SystemExit("--expert-pool-config and --expert-pool must be used together.")

    if args.evaluation_summary_root is not None:
        discovered = sorted(
            discover_mmlu_evaluation_summaries(args.evaluation_summary_root)
        )
        source_type = "evaluation_summary"
    else:
        discovered = sorted(discover_mmlu_summaries(args.summary_dir))
        source_type = "benchmark_summary"
    model_names = args.models or discovered
    if args.local_only:
        model_names = filter_local_models(model_names, args.models_dir)
    pool_report = None
    model_metadata: dict[str, dict[str, Any]] = {}
    if args.expert_pool_config is not None:
        pool = load_expert_pool(args.expert_pool_config, args.expert_pool)
        model_names, pool_report, model_metadata = select_expert_pool_models(
            model_names, pool
        )
    model_names = sorted(model_names)
    if not model_names:
        raise SystemExit("No MMLU-Pro summary models found.")

    if args.evaluation_summary_root is not None:
        category_rows, overall_rows = build_mmlu_evaluation_rows(
            args.evaluation_summary_root, model_names=model_names
        )
    else:
        category_rows, overall_rows = build_mmlu_category_rows(
            args.summary_dir, model_names=model_names
        )
    model_names = sorted({row["model"] for row in category_rows})
    model_to_index = {name: idx for idx, name in enumerate(model_names)}
    overall_accuracy = {
        row["model"]: float(row["accuracy"]) for row in overall_rows
    }
    winners = build_category_winners(
        category_rows, model_to_index, overall_accuracy=overall_accuracy
    )

    prefix = args.prefix
    if prefix is None:
        prefix = "local" if args.local_only else "all"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_df = pd.DataFrame(category_rows)
    overall_df = pd.DataFrame(overall_rows)
    category_df.insert(0, "model_index", category_df["model"].map(model_to_index))
    if not overall_df.empty:
        overall_df.insert(0, "model_index", overall_df["model"].map(model_to_index))

    category_df.to_csv(
        args.output_dir / f"{prefix}_accuracy_by_category_long.csv", index=False
    )
    overall_df.to_csv(args.output_dir / f"{prefix}_accuracy_overall.csv", index=False)

    pivot = (
        category_df.pivot(index="model", columns="category", values="accuracy")
        .reindex(index=model_names, columns=MMLU_CATEGORY_ORDER)
    )
    pivot.insert(0, "overall", pd.Series(overall_accuracy))
    pivot = pivot.reset_index()
    pivot.insert(
        1,
        "model_size_b",
        pivot["model"].map(
            lambda name: model_metadata.get(name, {}).get("parameters_b")
        ),
    )
    pivot.insert(
        2,
        "data_source",
        pivot["model"].map(
            lambda name: model_metadata.get(name, {}).get("data_source", "unknown")
        ),
    )
    pivot.insert(0, "model_index", pivot["model"].map(model_to_index))
    pivot.to_csv(args.output_dir / f"{prefix}_accuracy_by_category.csv", index=False)
    write_markdown_table(
        args.output_dir / f"{prefix}_accuracy_by_category.md",
        pivot.set_index(["model_index", "model"]),
    )
    render_heatmap(
        args.output_dir / f"{prefix}_accuracy_by_category.png",
        pivot.set_index("model")[["overall", *MMLU_CATEGORY_ORDER]],
        winners,
    )

    manifest = {
        "source_type": source_type,
        "summary_dir": str(args.summary_dir),
        "evaluation_summary_root": (
            str(args.evaluation_summary_root)
            if args.evaluation_summary_root is not None
            else None
        ),
        "models_dir": str(args.models_dir),
        "local_only": args.local_only,
        "expert_pool": pool_report,
        "model_metadata": {
            name: model_metadata.get(name, {}) for name in model_names
        },
        "model_names": model_names,
        "model_to_index": model_to_index,
        "category_winners": winners,
        "outputs": {
            "category_long_csv": str(
                args.output_dir / f"{prefix}_accuracy_by_category_long.csv"
            ),
            "category_csv": str(args.output_dir / f"{prefix}_accuracy_by_category.csv"),
            "category_md": str(args.output_dir / f"{prefix}_accuracy_by_category.md"),
            "category_png": str(args.output_dir / f"{prefix}_accuracy_by_category.png"),
            "overall_csv": str(args.output_dir / f"{prefix}_accuracy_overall.csv"),
        },
    }
    write_json(args.output_dir / f"{prefix}_expert_category_mapping.json", manifest)

    print(f"Models: {len(model_names)}")
    print(f"Categories: {len(winners)}")
    print(f"Wrote {args.output_dir / f'{prefix}_accuracy_by_category.csv'}")
    print(f"Wrote {args.output_dir / f'{prefix}_expert_category_mapping.json'}")


if __name__ == "__main__":
    main()
