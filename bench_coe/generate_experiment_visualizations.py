from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path("outputs/bench_coe")
OUT = ROOT / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "single": "#A9CBE3",
    "bench": "#98D4BC",
    "oracle": "#F1D49B",
    "tiny": "#9CC7E4",
    "qwen": "#C5B9E8",
    "bert": "#F0B8C8",
    "negative": "#E6AFC0",
    "positive": "#90CFB4",
    "grid": "#DCE6EE",
    "text": "#26384B",
    "muted": "#60758A",
}


def configure() -> None:
    candidates = [
        "/usr/share/fonts/MyFonts/MSYH.TTC",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.edgecolor": "#B7C8D6",
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_note(fig: plt.Figure, text: str, *, y: float = 0.015) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        color=COLORS["text"],
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F2F7FB", "edgecolor": "#C8D8E5"},
    )


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}%") -> None:
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            fontsize=10,
            fontweight="bold",
        )


def load_mapping(path: str, key: str) -> Counter[str]:
    data = json.loads((ROOT / path).read_text())
    return Counter(item["selected_model"] for item in data[key].values())


def chart_overview() -> None:
    labels = [
        "语言 Subject\nMMLU-Pro ID",
        "语言 Query\nMMLU ID",
        "多模态 Subject\nMMMU ID",
        "多模态 Query\nMMMU ID",
        "统一路由\nBabyVision 原池",
        "统一路由\nBabyVision 替换池",
    ]
    single = np.array([57.6130, 70.1754, 51.1111, 51.1111, 17.0103, 14.6907])
    bench = np.array([50.6649, 70.1754, 48.3333, 51.1111, 17.2680, 12.6289])
    gains = bench - single
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 8.2))
    height = 0.30
    b1 = ax.barh(y - height / 2, single, height, color=COLORS["single"], label="最佳单模型", edgecolor="white")
    bench_colors = [COLORS["bench"] if value >= 0 else COLORS["negative"] for value in gains]
    b2 = ax.barh(y + height / 2, bench, height, color=bench_colors, label="Bench-CoE", edgecolor="white")
    for index, (single_value, bench_value, gain) in enumerate(zip(single, bench, gains)):
        ax.text(single_value + 0.45, index - height / 2, f"{single_value:.2f}%", va="center", fontsize=10)
        ax.text(bench_value + 0.45, index + height / 2, f"{bench_value:.2f}%", va="center", fontsize=10, fontweight="bold")
        color = COLORS["positive"] if gain >= 0 else COLORS["negative"]
        ax.text(max(single_value, bench_value) + 5.0, index, f"{gain:+.2f} pp", va="center", fontsize=12, fontweight="bold", color=color)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 78)
    ax.set_xlabel("准确率（%）")
    ax.set_title("核心结果：Bench-CoE 相对最佳单模型的变化", fontsize=20, fontweight="bold", pad=18)
    ax.legend(ncol=2, loc="lower right")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    add_note(fig, "读图：语言 Subject 低于最佳单模型 6.95 pp；MMMU Subject 低 2.78 pp；MMMU Query 与最佳单模型持平。")
    fig.subplots_adjust(left=0.20, right=0.95, top=0.88, bottom=0.13)
    save(fig, "01_核心结果总览.png")


def best_validation(path: str) -> float | None:
    data = json.loads((ROOT / path).read_text())
    best = data.get("best_epoch")
    row = next((item for item in data.get("epochs", []) if item.get("epoch") == best), None)
    return None if row is None else row.get("validation", {}).get("accuracy")


def chart_training() -> None:
    specs = [
        ("语言 Query", "BERT", "router/bert-base-mmlu-test-query-7b-9b/train_metrics.json"),
        ("MMMU Subject", "TinyLLaVA", "router/tinyllava-mmmu-30subject-1p6b-2p2b/train_metrics.json"),
        ("MMMU Subject", "Qwen3-VL", "router/qwen3vl-2b-mmmu-30subject-1p6b-2p2b/train_metrics.json"),
        ("GAOKAO Subject", "TinyLLaVA", "router/tinyllava-gaokao-mm-8subject-1p6b-2p2b/train_metrics.json"),
        ("GAOKAO Subject", "Qwen3-VL", "router/qwen3vl-2b-gaokao-mm-8subject-1p6b-2p2b/train_metrics.json"),
        ("MMMU Query", "TinyLLaVA", "router/tinyllava-mmmu-query-1p6b-2p2b/train_metrics.json"),
        ("MMMU Query", "Qwen3-VL", "router/qwen3vl-2b-mmmu-query-1p6b-2p2b/train_metrics.json"),
        ("GAOKAO strict Query", "TinyLLaVA", "router/tinyllava-gaokao-mm-query-strict/train_metrics.json"),
        ("GAOKAO strict Query", "Qwen3-VL", "router/qwen3vl-gaokao-mm-query-strict/train_metrics.json"),
        ("统一 Query", "BERT", "router/bert-base-unified-mmlu-test-mmmu-train-query/train_metrics.json"),
        ("统一 Query", "TinyLLaVA", "router/tinyllava-unified-mmlu-test-mmmu-train-query/train_metrics.json"),
        ("统一 Query", "Qwen3-VL", "router/qwen3vl-unified-mmlu-test-mmmu-train-query/train_metrics.json"),
    ]
    names = [f"{task} / {router}" for task, router, _ in specs]
    values = [100 * float(best_validation(path)) for _, _, path in specs]
    colors = [COLORS["bert"] if "BERT" in name else COLORS["tiny"] if "Tiny" in name else COLORS["qwen"] for name in names]
    fig, ax = plt.subplots(figsize=(13, 8.2))
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colors, edgecolor="white", height=0.70)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("最佳 checkpoint 的内部验证准确率（%）")
    ax.set_title("路由模型训练：学科分类通常高于直接专家分类", fontsize=20, fontweight="bold", pad=18)
    label_bars(ax, bars)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    add_note(fig, "说明：Subject 路由识别学科，验证准确率达 76.11%–96.12%；Query 路由直接区分专家，类别不平衡使准确率多为 60%–73%。")
    fig.subplots_adjust(left=0.29, right=0.95, top=0.90, bottom=0.10)
    save(fig, "02_路由训练验证准确率.png")


def chart_mapping() -> None:
    mappings = [
        ("MMLU-Pro\n14类", load_mapping("mmlu_validation_7b_9b/validation_7b_9b_expert_category_mapping.json", "category_winners")),
        ("GAOKAO\n9学科", load_mapping("gaokao_no_qwen35_deepseek_qwen3/local_expert_subject_mapping.json", "subject_winners")),
        ("MMMU\n30学科", load_mapping("mmmu_train720_1p6b_2p2b/mmmu_train720_1p6b_2p2b_expert_subject_mapping.json", "subject_winners")),
        ("GAOKAO-MM\n8学科", load_mapping("gaokao_mm_train517_1p6b_2p2b/gaokao_mm_train517_1p6b_2p2b_expert_subject_mapping.json", "subject_winners")),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.0))
    palette = ["#9CC7E4", "#98D4BC", "#E8B8C8", "#F1D49B", "#C5B9E8", "#B8D9D7"]
    for ax, (title, counts) in zip(axes.flat, mappings):
        labels, values = zip(*counts.most_common())
        bars = ax.barh(np.arange(len(labels)), values, color=palette[: len(labels)], edgecolor="white")
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=15, fontweight="bold", pad=2)
        ax.set_xlabel("负责的学科数")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for bar in bars:
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va="center", fontsize=10, fontweight="bold")
    fig.suptitle("学科评估榜最终选择了哪些专家？", fontsize=21, fontweight="bold", y=0.985)
    add_note(fig, "读图：MMMU 的 30 个学科中 InternVL3.5-2B 负责 16 个，专家分配明显不均衡；GAOKAO-MM 主要依赖 LFM2.5-VL。")
    fig.subplots_adjust(left=0.18, right=0.96, top=0.84, bottom=0.11, hspace=0.44, wspace=0.46)
    save(fig, "03_学科专家映射分布.png")


def chart_id(df: pd.DataFrame) -> None:
    selected = [
        ("语言 Subject\nMMLU-Pro", "MMLU validation", "MMLU-Pro test", "BERT"),
        ("语言 Query\nMMLU", "MMLU test-source", "Internal holdout", "BERT"),
        ("MMMU Subject\nTiny", "MMMU train720", "MMMU holdout", "TinyLLaVA"),
        ("MMMU Subject\nQwen", "MMMU train720", "MMMU holdout", "Qwen3-VL-2B"),
        ("MMMU Query\nTiny", "MMMU train720", "MMMU holdout", "TinyLLaVA"),
        ("GAOKAO Subject\nTiny", "GAOKAO-MM train517", "GAOKAO-MM holdout", "TinyLLaVA"),
        ("GAOKAO Subject\nQwen", "GAOKAO-MM train517", "GAOKAO-MM holdout", "Qwen3-VL-2B"),
        ("GAOKAO Strict Query\nTiny", "GAOKAO-MM train517 strict", "GAOKAO-MM holdout", "TinyLLaVA"),
    ]
    rows = []
    for label, source, dataset, router in selected:
        match = df[(df.source_leaderboard == source) & (df.evaluation_dataset == dataset) & (df.router == router)]
        if "Subject" in label:
            match = match[match.routing_level == "Subject"]
        elif "Query" in label:
            match = match[match.routing_level == "Query"]
        row = match.iloc[0]
        rows.append((label, 100 * row.best_single_accuracy, 100 * row.routed_accuracy, 100 * row.oracle_any_accuracy if pd.notna(row.oracle_any_accuracy) else np.nan))
    labels = [row[0] for row in rows]
    single = np.array([row[1] for row in rows])
    bench = np.array([row[2] for row in rows])
    oracle = np.array([row[3] for row in rows])
    x = np.arange(len(labels)); width = 0.25
    fig, ax = plt.subplots(figsize=(14, 7.4))
    b1 = ax.bar(x - width, single, width, label="最佳单模型", color=COLORS["single"])
    b2 = ax.bar(x, bench, width, label="Bench-CoE", color=COLORS["bench"])
    b3 = ax.bar(x + width, oracle, width, label="Oracle", color=COLORS["oracle"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("准确率（%）")
    ax.set_title("分布内实验：最终准确率、最佳单模型与理论上限", fontsize=20, fontweight="bold", pad=18)
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in [b1, b2]:
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    add_note(fig, "结论：GAOKAO-MM Subject 是最明确的正增益；MMMU Subject 虽有高学科识别率，但最终准确率仍低于最佳单模型。")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20)
    save(fig, "04_分布内结果对比.png")


def chart_route_vs_gain(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    points = [
        (14.89, -6.95, "MMLU-val Subject / BERT", "Subject", (8, 8)),
        (83.89, -3.89, "MMMU Subject / Tiny", "Subject", (8, 8)),
        (76.11, -2.78, "MMMU Subject / Qwen", "Subject", (8, 8)),
        (95.00, 1.55, "GAOKAO-MM Subject\nTiny + Qwen", "Subject", (-115, 14)),
        (62.67, 0.00, "MMMU Query\nTiny + Qwen", "Query", (-75, 24)),
        (64.71, 0.00, "GAOKAO-MM Strict Query\nTiny + Qwen", "Query", (-15, -48)),
        (70.11, 0.00, "语言 Query / BERT", "Language", (28, -28)),
    ]
    styles = {
        "Subject": ("o", COLORS["tiny"]),
        "Query": ("s", COLORS["bench"]),
        "Language": ("D", COLORS["bert"]),
    }
    for x_value, y_value, label, group, offset in points:
        marker, color = styles[group]
        ax.scatter(x_value, y_value, s=170, marker=marker, color=color, edgecolor="white", linewidth=1.3, zorder=3)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#94A6B6", "linewidth": 0.8},
        )
    ax.axhline(0, color="#8799AA", linewidth=1.2)
    ax.set_xlabel("路由类别准确率（%）")
    ax.set_ylabel("Bench-CoE 相对最佳单模型增益（百分点）")
    ax.set_title("路由预测得准，为什么最终准确率仍可能不提升？", fontsize=20, fontweight="bold", pad=18)
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker=styles[group][0], color="none", markerfacecolor=styles[group][1], markeredgecolor="white", markersize=10, label=group)
        for group in ["Subject", "Query", "Language"]
    ]
    ax.legend(handles=handles, loc="lower right")
    add_note(fig, "关键现象：右下角点表示“路由类别准确率很高，但最终性能下降”。原因是学科内最佳专家并不一定能迁移到目标样本。")
    fig.subplots_adjust(left=0.11, right=0.95, top=0.88, bottom=0.14)
    save(fig, "05_路由准确率与最终增益.png")


def chart_language_ood(df: pd.DataFrame) -> None:
    data = df[(df.section == "Language") & (df.distribution == "OOD")].copy()
    data["label"] = data.source_leaderboard + " → " + data.evaluation_dataset + " / " + data.routing_level
    y = np.arange(len(data)); height = 0.34
    fig, ax = plt.subplots(figsize=(13, 7.2))
    b1 = ax.barh(y - height / 2, 100 * data.best_single_accuracy, height, label="最佳单模型", color=COLORS["single"])
    b2 = ax.barh(y + height / 2, 100 * data.routed_accuracy, height, label="Bench-CoE", color=COLORS["bench"])
    ax.set_yticks(y, data.label)
    ax.invert_yaxis()
    ax.set_xlabel("准确率（%）")
    ax.set_title("语言任务 OOD：路由总体接近但未超过最佳单模型", fontsize=20, fontweight="bold", pad=18)
    ax.legend(loc="lower right")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    add_note(fig, "BBH 上路由与最佳单模型差距较小；GPQA 上差距扩大，说明源域学科/专家标签难以覆盖高难推理分布。")
    fig.subplots_adjust(left=0.32, right=0.96, top=0.88, bottom=0.12)
    save(fig, "06_语言OOD结果.png")


def chart_multimodal_ood(df: pd.DataFrame) -> None:
    data = df[(df.section == "Multimodal") & (df.distribution == "OOD") & (df.evaluation_dataset != "BabyVision")].copy()
    data = data[~data.source_leaderboard.str.contains("all646", case=False, na=False)]
    subject = data[data.routing_level == "Subject"]
    query = data[data.routing_level == "Query"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2))
    for ax, subset, title in [(axes[0], subject, "Subject 路由"), (axes[1], query, "Query 路由")]:
        grouped = subset.groupby(["source_leaderboard", "evaluation_dataset"])["bench_coe_delta"].max().reset_index()
        source_names = {
            "MMMU train720": "MMMU",
            "GAOKAO-MM train517": "G-MM",
            "GAOKAO-MM strict": "G-MM strict",
        }
        dataset_names = {
            "CMMMU dev": "CMMMU",
            "MMMU-Pro test": "MMMU-Pro",
            "MathVista testmini": "MathVista",
            "MMMU validation": "MMMU-val",
        }
        grouped["label"] = grouped.source_leaderboard.map(source_names).fillna(grouped.source_leaderboard) + " → " + grouped.evaluation_dataset.map(dataset_names).fillna(grouped.evaluation_dataset)
        grouped = grouped.sort_values("bench_coe_delta")
        colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in grouped.bench_coe_delta]
        bars = ax.barh(np.arange(len(grouped)), 100 * grouped.bench_coe_delta, color=colors, edgecolor="white")
        ax.set_yticks(np.arange(len(grouped)), grouped.label, fontsize=9)
        ax.axvline(0, color="#7D91A2", linewidth=1)
        ax.set_xlabel("相对最佳单模型增益（百分点）")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for bar, value in zip(bars, grouped.bench_coe_delta):
            ax.text(-0.15, bar.get_y() + bar.get_height() / 2, f"{100*value:+.2f}", va="center", ha="right", fontsize=8, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    fig.suptitle("多模态 OOD：多数路由未超过目标数据集最佳单模型", fontsize=21, fontweight="bold", y=0.98)
    add_note(fig, "每个源域→目标域取 TinyLLaVA/Qwen3-VL 中较好的结果。MathVista 差距最大，表明跨任务视觉推理能力难由源域路由标签迁移。")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.12, wspace=0.58)
    save(fig, "07_多模态OOD增益.png")


def chart_unified(df: pd.DataFrame) -> None:
    internal = df[(df.section == "Unified") & (df.evaluation_dataset == "Internal validation")]
    holdout = df[(df.section == "Unified") & (df.evaluation_dataset == "MMMU clean holdout")]
    routers = ["BERT", "TinyLLaVA", "Qwen3-VL-2B"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    for ax, subset, title in [(axes[0], internal, "统一内部验证"), (axes[1], holdout, "MMMU 干净留出")]:
        values = [100 * subset[subset.router == router].routed_accuracy.iloc[0] for router in routers]
        bars = ax.bar(routers, values, color=[COLORS["bert"], COLORS["tiny"], COLORS["qwen"]], edgecolor="white")
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_ylabel("Bench-CoE 准确率（%）")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10, fontweight="bold")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("统一语言+多模态路由：模态识别稳定，三种路由差距较小", fontsize=21, fontweight="bold", y=0.98)
    add_note(fig, "内部验证整体约 69.6%；MMMU 留出为 48.89%–51.11%。跨模态错误仅 0、0、1，但专家选择仍接近最强单模型。")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14, wspace=0.28)
    save(fig, "08_统一路由结果.png")


def chart_babyvision(df: pd.DataFrame) -> None:
    original = df[(df.evaluation_dataset == "BabyVision") & ~df.source_leaderboard.str.contains("swap", case=False) & (df.source_leaderboard == "MMLU+MMMU")]
    swap = df[(df.evaluation_dataset == "BabyVision") & (df.source_leaderboard == "MMLU+MMMU swap")]
    routers = ["BERT", "TinyLLaVA", "Qwen3-VL-2B"]
    x = np.arange(len(routers)); width = 0.34
    fig, ax = plt.subplots(figsize=(12, 7.2))
    original_values = [100 * original[original.router == router].routed_accuracy.iloc[0] for router in routers]
    swap_values = [100 * swap[swap.router == router].routed_accuracy.iloc[0] for router in routers]
    b1 = ax.bar(x - width / 2, original_values, width, label="原池：含 Qwen3.5-2B", color=COLORS["bench"])
    b2 = ax.bar(x + width / 2, swap_values, width, label="替换池：Qwen3-VL-2B", color=COLORS["negative"])
    ax.axhline(17.01, color=COLORS["single"], linestyle="--", linewidth=2, label="原池最佳单模型 17.01%")
    ax.axhline(14.69, color="#8CAEC8", linestyle=":", linewidth=2, label="替换池最佳单模型 14.69%")
    ax.set_xticks(x, routers)
    ax.set_ylabel("BabyVision 准确率（%）")
    ax.set_ylim(0, 21)
    ax.set_title("BabyVision：替换 Qwen3.5-2B 后，统一路由整体下降", fontsize=20, fontweight="bold", pad=18)
    ax.bar_label(b1, fmt="%.2f%%", padding=3, fontsize=10, fontweight="bold")
    ax.bar_label(b2, fmt="%.2f%%", padding=3, fontsize=10, fontweight="bold")
    ax.legend(ncol=2, loc="upper left")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    add_note(fig, "最重要结果：原池统一 Qwen3-VL 达 17.27%，略高于 Qwen3.5-2B；替换后降至 12.63%，说明 Qwen3.5-2B 提供关键互补性。")
    fig.subplots_adjust(left=0.09, right=0.96, top=0.87, bottom=0.14)
    save(fig, "09_BabyVision替换实验.png")


def build_visual_report() -> None:
    content = """# Bench-CoE 实验结果可视化汇报版

生成日期：2026-07-28。本报告用于组会/答辩口头汇报；完整数字、专家池和逐学科映射仍保留在 [`ALL_EXPERIMENTS_SUMMARY.md`](ALL_EXPERIMENTS_SUMMARY.md)。

## 1. 先讲总体结论

![核心结果总览](visualizations/01_核心结果总览.png)

**汇报说明：**

- MMMU Subject 最好结果低于最佳单模型 **2.78 pp**；MMMU Query 与最佳单模型持平。
- 语言 Subject 在 MMLU-Pro 上低于最佳单模型 **6.95 pp**，说明当前路由收益具有明显的数据集依赖性。
- BabyVision 原专家池中，统一 Qwen3-VL 路由达到 **17.27%**，比最佳单模型高 **0.26 pp**。
- 将 Qwen3.5-2B 替换为 Qwen3-VL-2B-Instruct 后，统一路由降至 **12.63%**，说明专家互补性比单纯模型更新更重要。

## 2. 路由模型本身训练得怎么样？

![路由训练验证准确率](visualizations/02_路由训练验证准确率.png)

**汇报说明：** Subject 路由只需要识别学科，验证准确率较高；Query 路由需要直接区分专家，受到标签不平衡和专家能力重叠影响，准确率相对较低。

## 3. 学科评估榜最终选出了哪些专家？

![学科专家映射分布](visualizations/03_学科专家映射分布.png)

**汇报说明：** MMMU 中 InternVL3.5-2B 负责超过一半学科；GAOKAO-MM 主要由 LFM2.5-VL 负责。这种不均衡会使 Query 路由容易塌缩到少数强专家。

## 4. 分布内实验

![分布内结果](visualizations/04_分布内结果对比.png)

**汇报说明：**

- GAOKAO-MM Subject 是最明确的有效路由结果。
- MMMU Subject 的学科识别率很高，但最终 Bench-CoE 低于最佳单模型，说明“识别正确学科”不等于“该学科专家能迁移到每道题”。
- Oracle 与实际结果差距较大，专家之间存在互补性，但当前路由尚未充分利用。

![路由准确率与最终增益](visualizations/05_路由准确率与最终增益.png)

**汇报说明：** 图中右下角表示路由类别预测很准、最终性能却下降。该图用于强调本文后续需要优化的不只是分类准确率，还包括专家映射与决策目标。

## 5. 分布外实验

### 5.1 语言任务

![语言OOD](visualizations/06_语言OOD结果.png)

**汇报说明：** BBH 上路由接近最佳单模型；GPQA 差距明显，说明跨高难推理任务时源域专家标签迁移有限。

### 5.2 多模态任务

![多模态OOD](visualizations/07_多模态OOD增益.png)

**汇报说明：** 多数 OOD 组合未超过目标数据集的最佳单模型，尤其 MathVista 差距较大。Subject 路由总体比 Query 路由更稳定，但仍受源域学科专家映射限制。

## 6. 统一语言+多模态路由

![统一路由](visualizations/08_统一路由结果.png)

**汇报说明：** 三种路由器的整体差距不大，跨模态错误极少，说明模态识别不是主要瓶颈；真正瓶颈仍是多模态专家的精细选择。

## 7. BabyVision 与专家替换

![BabyVision替换](visualizations/09_BabyVision替换实验.png)

**汇报说明：**

- 原池中 Qwen3-VL 统一路由取得全部 BabyVision 实验最佳结果 **17.27%**。
- 替换 Qwen3.5-2B 后，最佳单模型、Oracle 和三种统一路由均下降。
- 因此专家池更新不能只看参数规模或模型发布时间，需要重新评估其与其它专家的互补性。

## 8. 建议汇报结论

1. Subject 路由具有更低标签成本和更好的跨域稳定性，但学科映射的迁移能力决定最终上限。
2. Query 路由在 ID 数据上可能有效，但容易因类别不平衡而退化为选择最强单模型。
3. GAOKAO-MM 是当前最明确的正向结果；BabyVision 原池统一 Qwen3-VL 是最有代表性的 OOD 正增益结果。
4. 后续重点应从单纯提升路由分类准确率转向类别重加权、专家负载均衡、代价敏感训练和专家互补性建模。
"""
    (ROOT / "ALL_EXPERIMENTS_VISUAL_REPORT.md").write_text(content, encoding="utf-8")


def main() -> None:
    configure()
    df = pd.read_csv(ROOT / "ALL_EXPERIMENTS_RESULTS.csv")
    chart_overview()
    chart_training()
    chart_mapping()
    chart_id(df)
    chart_route_vs_gain(df)
    chart_language_ood(df)
    chart_multimodal_ood(df)
    chart_unified(df)
    chart_babyvision(df)
    build_visual_report()


if __name__ == "__main__":
    main()
