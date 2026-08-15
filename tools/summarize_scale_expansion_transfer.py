from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fcs-scale-summary")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/bench_coe/scale_expansion_and_transfer_summary_20260801"

BASELINE_RESULTS = {
    "language": {
        "improve5": ROOT
        / "outputs/bench_coe/mmlu_val_source_language_all_experiments_exclude_qwen35_deepseek_qwen3/improve5/summary.json",
        "improve6": ROOT
        / "outputs/bench_coe/mmlu_val_source_language_all_experiments_exclude_qwen35_deepseek_qwen3/improve6/summary.json",
    },
    "vision": {
        "improve5": ROOT / "outputs/bench_coe/mmmu_pro_val_source_improve5/summary.json",
        "improve6": ROOT / "outputs/bench_coe/mmmu_pro_val_source_improve6/summary.json",
    },
}

COHORTS = [
    {
        "modality": "语言",
        "scale": "2B–4B",
        "role": "扩展",
        "models": [
            ("Qwen3-1.7B", "已完成"),
            ("Qwen2.5-3B-Instruct", "已完成"),
            ("granite-3.3-2b-instruct", "测试中"),
            ("internlm2_5-1_8b-chat", "已完成"),
            ("gemma-2-2b-it", "已完成"),
            ("Ministral-3-3B-Instruct-2512", "兼容阻塞"),
            ("Llama-3.2-3B-Instruct", "已完成"),
            ("DeepSeek-R1-Distill-Qwen-1.5B", "下载中"),
        ],
    },
    {
        "modality": "语言",
        "scale": "7B–9B",
        "role": "已完成基线",
        "models": [("已有14模型专家池", "已完成") for _ in range(14)],
    },
    {
        "modality": "语言",
        "scale": "约14B",
        "role": "扩展",
        "models": [
            ("Qwen3-14B", "已完成"),
            ("Qwen2.5-14B-Instruct", "测试中"),
            ("Baichuan2-13B-Chat", "测试中"),
            ("Mistral-Nemo-Instruct-2407", "下载中"),
            ("DeepSeek-R1-Distill-Qwen-14B", "下载中"),
        ],
    },
    {
        "modality": "视觉语言",
        "scale": "2B–4B",
        "role": "已完成基线",
        "models": [("已有11模型专家池", "已完成") for _ in range(11)],
    },
    {
        "modality": "视觉语言",
        "scale": "7B–9B",
        "role": "扩展",
        "models": [
            ("Qwen3-VL-8B-Instruct", "已完成"),
            ("GLM-4.1V-9B-Thinking", "已完成"),
            ("InternVL3_5-8B", "已完成"),
            ("Qwen2.5-VL-7B-Instruct", "下载中"),
            ("Qwen3-VL-8B-Thinking", "下载中"),
            ("Llama-3.1-Nemotron-Nano-VL-8B-V1", "兼容阻塞"),
        ],
    },
    {
        "modality": "视觉语言",
        "scale": "约14B",
        "role": "扩展",
        "models": [
            ("Phi-4-reasoning-vision-15B", "已完成"),
            ("InternVL3_5-14B", "下载中"),
            ("gemma-3-12b-it", "下载中"),
        ],
    },
]

CASE_LABELS = {
    "mmlu_val_to_bbh": "BBH",
    "mmlu_val_to_gpqa": "GPQA",
    "mmlu_val_to_mmstar": "MMStar",
    "mmlu_val_to_mmlu_test": "MMLU-Pro",
    "mmlu_val_to_gaokao2010_2022": "GAOKAO",
    "mmmu_pro_val_to_cmmmu": "CMMMU",
    "mmmu_pro_val_to_mathvista": "MathVista",
    "mmmu_pro_val_to_mmmu_pro_test": "MMMU-Pro",
}


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_best_rows(path: Path) -> dict[str, dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, object]] = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        best = max(case_rows, key=lambda row: float(row["target_accuracy"]))
        output[case_id] = {
            "method": best["method"],
            "gain_pct": float(best["gain_vs_best_single_target"]) * 100,
            "accuracy_pct": float(best["target_accuracy"]) * 100,
            "best_single_pct": float(best["best_single_target"]) * 100,
            "models_used": int(best["models_used"]),
        }
    return output


def cohort_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cohort in COHORTS:
        for index, (model, status) in enumerate(cohort["models"], start=1):
            display_model = model
            if cohort["role"] == "已完成基线":
                display_model = f"{model}-{index:02d}"
            rows.append(
                {
                    "modality": cohort["modality"],
                    "scale": cohort["scale"],
                    "role": cohort["role"],
                    "model": display_model,
                    "status": status,
                }
            )
    return rows


def expansion_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["role"] == "扩展"]


def draw_model_counts(rows: list[dict[str, str]], path: Path) -> None:
    labels = [f"{cohort['modality']}\n{cohort['scale']}" for cohort in COHORTS]
    values = [len(cohort["models"]) for cohort in COHORTS]
    colors = ["#4C78A8" if cohort["role"] == "已完成基线" else "#F58518" for cohort in COHORTS]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    ax.set_ylabel("模型/专家数量")
    ax.set_title("跨参数规模模型池：已完成基线与新增扩展")
    ax.set_ylim(0, max(values) + 3)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.96,
        "蓝色：已有改进5/6结果   橙色：新增规模扩展",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_expansion_status(rows: list[dict[str, str]], path: Path) -> None:
    expanded = expansion_rows(rows)
    groups = [("语言", "2B–4B"), ("语言", "约14B"), ("视觉语言", "7B–9B"), ("视觉语言", "约14B")]
    statuses = ["已完成", "测试中", "下载中", "兼容阻塞"]
    colors = ["#54A24B", "#4C78A8", "#F2CF5B", "#E45756"]
    data = {
        group: Counter(row["status"] for row in expanded if (row["modality"], row["scale"]) == group)
        for group in groups
    }
    fig, ax = plt.subplots(figsize=(10.5, 6))
    bottoms = np.zeros(len(groups))
    x = np.arange(len(groups))
    for status, color in zip(statuses, colors):
        values = np.asarray([data[group][status] for group in groups])
        bars = ax.bar(x, values, bottom=bottoms, label=status, color=color)
        for bar, value, bottom in zip(bars, values, bottoms):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
        bottoms += values
    ax.set_xticks(x, [f"{modality}\n{scale}" for modality, scale in groups])
    ax.set_ylabel("新增模型数量")
    ax.set_title("新增22个跨规模模型的基础测试准备状态")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, max(bottoms) + 1.5)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_gains(results: dict[str, dict[str, dict[str, object]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1.5, 1]})
    panels = [
        ("language", ["mmlu_val_to_bbh", "mmlu_val_to_gpqa", "mmlu_val_to_mmstar", "mmlu_val_to_mmlu_test", "mmlu_val_to_gaokao2010_2022"], "语言7B–9B：相对最佳单模型增益"),
        ("vision", ["mmmu_pro_val_to_cmmmu", "mmmu_pro_val_to_mathvista", "mmmu_pro_val_to_mmmu_pro_test"], "视觉语言2B–4B - 相对最佳单模型增益"),
    ]
    for ax, (modality, cases, title) in zip(axes, panels):
        x = np.arange(len(cases))
        width = 0.36
        improve5 = [results[modality]["improve5"][case]["gain_pct"] for case in cases]
        improve6 = [results[modality]["improve6"][case]["gain_pct"] for case in cases]
        bars5 = ax.bar(x - width / 2, improve5, width, label="改进5", color="#4C78A8")
        bars6 = ax.bar(x + width / 2, improve6, width, label="改进6", color="#F58518")
        ax.axhline(0, color="#444444", linewidth=0.9)
        ax.set_xticks(x, [CASE_LABELS[case] for case in cases], rotation=20)
        ax.set_ylabel("增益（百分点）")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.legend()
        for bars in (bars5, bars6):
            for bar in bars:
                value = bar.get_height()
                offset = 0.12 if value >= 0 else -0.18
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    f"{value:+.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8.5,
                )
    fig.suptitle("已有参数规模上的改进5/6迁移效果", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_roadmap(path: Path) -> None:
    matrix = np.asarray([[1, 2, 1], [2, 1, 0]])
    cmap = ListedColormap(["#F2CF5B", "#4C78A8", "#54A24B"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(3), ["2B–4B", "7B–9B", "约14B"])
    ax.set_yticks(range(2), ["语言任务", "视觉语言任务"])
    ax.set_title("改进5/6跨参数规模迁移验证路线图")
    labels = {
        (0, 0): "基础测试进行中\n改进5/6待运行",
        (0, 1): "改进5/6已完成\n14模型专家池",
        (0, 2): "基础测试进行中\n改进5/6待运行",
        (1, 0): "改进5/6已完成\n11模型专家池",
        (1, 1): "测试/下载进行中\n改进5/6待运行",
        (1, 2): "下载与测试准备中\n改进5/6待运行",
    }
    for (row, col), text in labels.items():
        ax.text(col, row, text, ha="center", va="center", fontsize=10, fontweight="bold")
    for edge in np.arange(-0.5, 3, 1):
        ax.axvline(edge, color="white", linewidth=2)
    for edge in np.arange(-0.5, 2, 1):
        ax.axhline(edge, color="white", linewidth=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_dashboard(image_paths: dict[str, Path], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(image_paths[key]).convert("RGB") for key in ("counts", "status", "gains", "roadmap")]
    tile_width = 2500
    tile_height = 1250
    title_height = 100
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 2 + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        font = ImageFont.truetype(font_path, 34)
    except OSError:
        font = ImageFont.load_default()
    draw.text((30, 20), "扩展模型数量与改进5/6跨参数规模迁移实验总览", fill="#222222", font=font)
    for index, image in enumerate(images):
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        column = index % 2
        row = index // 2
        x = column * tile_width + (tile_width - image.width) // 2
        y = title_height + row * tile_height + (tile_height - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(path, quality=95)
    for image in images:
        image.close()


def write_csv_files(rows: list[dict[str, str]], results: dict[str, dict[str, dict[str, object]]]) -> None:
    with (OUTPUT_DIR / "model_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modality", "scale", "role", "model", "status"])
        writer.writeheader()
        writer.writerows(rows)
    gain_rows: list[dict[str, object]] = []
    for modality, improvements in results.items():
        for improvement, cases in improvements.items():
            for case_id, values in cases.items():
                gain_rows.append(
                    {
                        "modality": modality,
                        "improvement": improvement,
                        "case_id": case_id,
                        "case_label": CASE_LABELS[case_id],
                        **values,
                    }
                )
    with (OUTPUT_DIR / "baseline_improve56_best_gains.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gain_rows[0]))
        writer.writeheader()
        writer.writerows(gain_rows)


def pct_summary(values: list[float]) -> tuple[float, int, int]:
    return sum(values) / len(values), sum(value > 0 for value in values), sum(value >= 0 for value in values)


def write_report(rows: list[dict[str, str]], results: dict[str, dict[str, dict[str, object]]], generated_at: str) -> None:
    expanded = expansion_rows(rows)
    status_counts = Counter(row["status"] for row in expanded)
    total_baseline = sum(len(cohort["models"]) for cohort in COHORTS if cohort["role"] == "已完成基线")
    total_expanded = len(expanded)
    language5 = [value["gain_pct"] for value in results["language"]["improve5"].values()]
    language6 = [value["gain_pct"] for value in results["language"]["improve6"].values()]
    vision5 = [value["gain_pct"] for value in results["vision"]["improve5"].values()]
    vision6 = [value["gain_pct"] for value in results["vision"]["improve6"].values()]
    summaries = {
        "language5": pct_summary(language5),
        "language6": pct_summary(language6),
        "vision5": pct_summary(vision5),
        "vision6": pct_summary(vision6),
    }
    lines = [
        "# 扩展模型数量与跨参数规模迁移实验汇总",
        "",
        f"- 汇总时间：`{generated_at}`",
        f"- 已完成基线专家池：**{total_baseline}** 个模型（语言7B–9B为14个；视觉语言2B–4B为11个）",
        f"- 新增跨规模模型：**{total_expanded}** 个",
        f"- 计划总模型/专家规模：**{total_baseline + total_expanded}** 个",
        "- 重要说明：新增规模目前仍处于下载和单模型基础测试阶段，新的改进5/6跨规模结果尚未生成，因此不能提前宣称策略在新规模上仍然有效。",
        "",
        "## 一、模型数量扩展",
        "",
        "| 模态 | 参数规模 | 角色 | 模型数 | 当前说明 |",
        "|---|---:|---|---:|---|",
    ]
    for cohort in COHORTS:
        counts = Counter(status for _, status in cohort["models"])
        status_text = "、".join(f"{key}{value}" for key, value in counts.items())
        lines.append(
            f"| {cohort['modality']} | {cohort['scale']} | {cohort['role']} | {len(cohort['models'])} | {status_text} |"
        )
    lines.extend(
        [
            "",
            f"新增22个模型中：已完成 **{status_counts['已完成']}**，测试中 **{status_counts['测试中']}**，下载中 **{status_counts['下载中']}**，兼容阻塞 **{status_counts['兼容阻塞']}**。",
            "",
            "![模型池扩展](model_pool_expansion.png)",
            "",
            "![新增模型状态](expanded_model_status.png)",
            "",
            "## 二、已有规模上的改进5/6结果",
            "",
            "### 语言任务：7B–9B、14模型专家池",
            "",
            f"- 改进5平均增益：**{summaries['language5'][0]:+.2f}个百分点**；5个迁移任务中{summaries['language5'][1]}个严格提升、{summaries['language5'][2]}个不下降。",
            f"- 改进6平均增益：**{summaries['language6'][0]:+.2f}个百分点**；5个迁移任务中{summaries['language6'][1]}个严格提升、{summaries['language6'][2]}个不下降。",
            "- 最大增益出现在BBH：改进5和改进6均为 **+8.45个百分点**。",
            "- GPQA仍是主要短板：改进5为 **-3.02个百分点**，改进6改善到 **-0.59个百分点**，但仍未超过最佳单模型。",
            "",
            "### 视觉语言任务：2B–4B、11模型专家池",
            "",
            f"- 改进5平均增益：**{summaries['vision5'][0]:+.2f}个百分点**，3/3任务均提升。",
            f"- 改进6平均增益：**{summaries['vision6'][0]:+.2f}个百分点**，3/3任务均提升。",
            "- 最大增益出现在MathVista：改进5和改进6均为 **+1.90个百分点**。",
            "",
            "![已有改进增益](baseline_improve56_gains.png)",
            "",
            "## 三、跨参数规模迁移验证状态",
            "",
            "| 模态 | 2B–4B | 7B–9B | 约14B |",
            "|---|---|---|---|",
            "| 语言 | 基础测试进行中，改进5/6待运行 | 改进5/6已完成 | 基础测试进行中，改进5/6待运行 |",
            "| 视觉语言 | 改进5/6已完成 | 下载/基础测试进行中，改进5/6待运行 | 下载/基础测试准备中，改进5/6待运行 |",
            "",
            "![迁移路线图](scale_transfer_roadmap.png)",
            "",
            "## 四、简要结论",
            "",
            "1. 已有结果表明，改进5和改进6在语言7B–9B与视觉语言2B–4B模型池上总体有效，但并非所有数据集都稳定提升，GPQA存在负迁移。",
            "2. 本轮将验证范围从25个已有专家扩展到计划47个跨尺度专家，重点补齐语言2B–4B/约14B和视觉语言7B–9B/约14B。",
            "3. 当前只能确认模型池和基础评测正在扩展；是否具有参数规模鲁棒性，需要等待各新规模模型在相同数据集完成单模型预测后，再分别运行改进5和改进6并比较增益。",
            "4. 最终判断应同时看平均增益、正迁移任务比例、最差任务退化和不同规模趋势，不能只看单个数据集的最高值。",
            "",
            "## 五、总览图",
            "",
            "![总览图](scale_expansion_transfer_dashboard.png)",
        ]
    )
    (OUTPUT_DIR / "scale_expansion_and_transfer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = cohort_rows()
    results = {
        modality: {
            improvement: load_best_rows(path)
            for improvement, path in improvements.items()
        }
        for modality, improvements in BASELINE_RESULTS.items()
    }
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    image_paths = {
        "counts": OUTPUT_DIR / "model_pool_expansion.png",
        "status": OUTPUT_DIR / "expanded_model_status.png",
        "gains": OUTPUT_DIR / "baseline_improve56_gains.png",
        "roadmap": OUTPUT_DIR / "scale_transfer_roadmap.png",
    }
    draw_model_counts(rows, image_paths["counts"])
    draw_expansion_status(rows, image_paths["status"])
    draw_gains(results, image_paths["gains"])
    draw_roadmap(image_paths["roadmap"])
    draw_dashboard(image_paths, OUTPUT_DIR / "scale_expansion_transfer_dashboard.png")
    write_csv_files(rows, results)
    write_report(rows, results, generated_at)
    payload = {
        "generated_at": generated_at,
        "cohorts": COHORTS,
        "baseline_best_results": results,
        "notes": {
            "baseline_language": "7B–9B language Improve5/6 completed with 14 experts",
            "baseline_vision": "2B–4B vision-language Improve5/6 completed with 11 experts",
            "new_scale_results": "pending base evaluation completion",
        },
    }
    (OUTPUT_DIR / "scale_expansion_and_transfer_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
