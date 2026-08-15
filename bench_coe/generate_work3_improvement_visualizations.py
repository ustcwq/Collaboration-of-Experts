from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bench_coe.generate_experiment_visualizations import COLORS, add_note, configure


OUT = Path("outputs/bench_coe/visualizations")
OUT.mkdir(parents=True, exist_ok=True)


CMMMU = [
    {"method": "FATE Failure Ecology", "short": "FATE", "accuracy": 39.22, "gain": 0.67, "low": -2.44, "high": 3.11, "repairs": 87, "harms": 81, "switches": 794},
    {"method": "Multiview Signature", "short": "Multiview", "accuracy": 39.22, "gain": 0.67, "low": -1.67, "high": 3.22, "repairs": 67, "harms": 61, "switches": 583},
    {"method": "Error-Awareness Ridge", "short": "Error-Aware", "accuracy": 38.67, "gain": 0.11, "low": -2.33, "high": 2.33, "repairs": 61, "harms": 60, "switches": 737},
    {"method": "Ontology Translation", "short": "Ontology", "accuracy": 38.56, "gain": 0.00, "low": -0.44, "high": 0.44, "repairs": 2, "harms": 2, "switches": 31},
    {"method": "LEAF Posterior Vote", "short": "LEAF", "accuracy": 37.89, "gain": -0.67, "low": -4.11, "high": 2.56, "repairs": 101, "harms": 107, "switches": 765},
    {"method": "Shadow Relative Advantage", "short": "Shadow", "accuracy": 37.67, "gain": -0.89, "low": -4.22, "high": 1.78, "repairs": 100, "harms": 108, "switches": 857},
    {"method": "Source Global Best", "short": "Global Best", "accuracy": 37.00, "gain": -1.56, "low": -4.78, "high": 1.22, "repairs": 103, "harms": 117, "switches": 900},
]

MATHVISTA = [
    {"method": "RepairChain", "short": "RepairChain", "accuracy": 62.90, "gain": 2.10, "low": 0.10, "high": 4.00, "repairs": 64, "harms": 43, "switches": 330},
    {"method": "DARE Reliability", "short": "DARE", "accuracy": 62.80, "gain": 2.00, "low": -0.10, "high": 4.00, "repairs": 70, "harms": 50, "switches": 405},
    {"method": "ECC Code Decoder", "short": "ECC", "accuracy": 61.80, "gain": 1.00, "low": -1.00, "high": 3.10, "repairs": 64, "harms": 54, "switches": 547},
    {"method": "LEAF Posterior Vote", "short": "LEAF", "accuracy": 61.50, "gain": 0.70, "low": -1.90, "high": 3.20, "repairs": 84, "harms": 77, "switches": 287},
    {"method": "Source Global Best", "short": "Global Best", "accuracy": 60.80, "gain": 0.00, "low": 0.00, "high": 0.00, "repairs": 0, "harms": 0, "switches": 0},
    {"method": "Trace Stage Capability", "short": "Trace", "accuracy": 60.80, "gain": 0.00, "low": 0.00, "high": 0.00, "repairs": 0, "harms": 0, "switches": 0},
]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, lines: list[str], color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#BDD1E1",
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + height - 0.075, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color="#28599B", va="top")
    ax.text(x + 0.025, y + height - 0.17, "\n".join(lines), transform=ax.transAxes, fontsize=11.5, color=COLORS["text"], va="top", linespacing=1.55)


def method_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.axis("off")
    ax.set_title("工作三方法：从源域失败知识到目标修复路由", fontsize=22, fontweight="bold", pad=18)

    rounded_box(ax, (0.03, 0.40), 0.19, 0.39, "① 源域校准", ["MMMU-Pro 专家正确性", "输出模式与失败状态", "建立强基座与互补关系"], "#DDEAF7")
    rounded_box(ax, (0.28, 0.40), 0.19, 0.39, "② 目标无标签信号", ["问题与图像特征", "专家答案分歧", "多视角 signature"], "#DDF3F2")
    rounded_box(ax, (0.53, 0.40), 0.21, 0.39, "③ 决策模块", ["FATE：失败生态", "DARE：可靠性估计", "ECC：错误修复", "LEAF：后验投票"], "#E1F1E5")
    rounded_box(ax, (0.80, 0.40), 0.17, 0.39, "④ 路由动作", ["保留强基座", "可靠时切换", "RepairChain 逐级修复"], "#FFF1CC")

    for start, end in [((0.225, 0.595), (0.275, 0.595)), ((0.475, 0.595), (0.525, 0.595)), ((0.745, 0.595), (0.795, 0.595))]:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": "#4B79C5", "lw": 2.8})

    ax.text(0.5, 0.27, "关键变化", transform=ax.transAxes, ha="center", fontsize=15, fontweight="bold", color="#28599B")
    ax.text(
        0.5,
        0.19,
        "不再只预测“属于哪个学科”，而是估计“强专家是否会失败、替代专家是否真的能够修复”。",
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
        color=COLORS["text"],
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#EEF5FB", "edgecolor": "#C8D8E5"},
    )
    add_note(fig, "无污染协议：目标标签不参与路由、阈值或方法选择，只用于最终评分与配对置信区间。")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.87, bottom=0.13)
    save(fig, "10_工作三_改进方法流程.png")


def gain_panel(ax: plt.Axes, rows: list[dict[str, float]], title: str, best_single: str) -> None:
    ordered = list(reversed(rows))
    y = np.arange(len(ordered))
    gains = np.array([row["gain"] for row in ordered])
    colors = [COLORS["positive"] if value > 0 else "#B9C8D4" if value == 0 else COLORS["negative"] for value in gains]
    bars = ax.barh(y, gains, color=colors, edgecolor="white", height=0.66)
    ax.set_yticks(y, [row["method"] for row in ordered], fontsize=10)
    ax.axvline(0, color="#7F93A4", linewidth=1.2)
    ax.set_xlim(-2.05, 2.65)
    ax.set_xlabel("相对最佳单模型增益（百分点）")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, row in zip(bars, ordered):
        value = row["gain"]
        x = value + 0.08
        ha = "left"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}  ({row['accuracy']:.2f}%)", va="center", ha=ha, fontsize=9, fontweight="bold")
    ax.text(0.02, 0.98, f"最佳单模型：{best_single}", transform=ax.transAxes, va="top", fontsize=10.5, color=COLORS["muted"])


def method_results() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2))
    gain_panel(axes[0], CMMMU, "CMMMU：多个模块仅形成小幅趋势", "Qwen3-VL-2B  38.56%")
    gain_panel(axes[1], MATHVISTA, "MathVista：修复模块形成稳定正增益", "InternVL3.5-2B  60.80%")
    fig.suptitle("工作三结果：具体方法相对最佳单模型的变化", fontsize=22, fontweight="bold", y=0.98)
    add_note(fig, "括号内为最终准确率。CMMMU 的最好点估计为 +0.67 pp；MathVista 的 RepairChain 为 +2.10 pp。")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.87, bottom=0.13, wspace=0.52)
    save(fig, "11_工作三_具体方法结果.png")


def forest_panel(ax: plt.Axes, rows: list[dict[str, float]], title: str) -> None:
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-5, 0, color="#F8E8EE", alpha=0.62)
    ax.axvspan(0, 5, color="#E8F5EE", alpha=0.72)
    ax.axvline(0, color="#647B8D", linewidth=1.3)
    for pos, row in zip(y, rows):
        color = "#3E9D78" if row["low"] > 0 else "#C37C95" if row["gain"] < 0 else "#4E83B6"
        ax.plot([row["low"], row["high"]], [pos, pos], color=color, linewidth=4, solid_capstyle="round")
        ax.scatter(row["gain"], pos, s=95, color=color, edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(row["high"] + 0.13, pos, f"{row['gain']:+.2f}", va="center", fontsize=9, fontweight="bold", color=color)
    ax.set_yticks(y, [row["method"] for row in rows], fontsize=10)
    ax.set_xlim(-5.2, 5.2)
    ax.set_xlabel("相对最佳单模型增益及配对 95% CI（百分点）")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)


def confidence_intervals() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    forest_panel(axes[0], CMMMU[:4], "CMMMU：所有领先方法的区间均跨 0")
    forest_panel(axes[1], MATHVISTA[:4], "MathVista：仅 RepairChain 下界为正")
    fig.suptitle("工作三统计解释：点估计提升是否可靠？", fontsize=22, fontweight="bold", y=0.98)
    add_note(fig, "绿色区间完全位于 0 右侧才表示当前配对 bootstrap 下的可靠正增益；点估计为正并不等于已确认提升。")
    fig.subplots_adjust(left=0.18, right=0.96, top=0.86, bottom=0.14, wspace=0.54)
    save(fig, "12_工作三_置信区间.png")


def efficiency_panel(ax: plt.Axes, rows: list[dict[str, float]], title: str, xlim: tuple[float, float], ylim: tuple[float, float], offsets: dict[str, tuple[int, int]]) -> None:
    ax.axhline(0, color="#7F93A4", linewidth=1.2)
    for row in rows:
        net = row["repairs"] - row["harms"]
        color = COLORS["positive"] if net > 0 else COLORS["negative"] if net < 0 else "#AABBC8"
        size = 130 + 85 * abs(row["gain"])
        ax.scatter(row["switches"], net, s=size, color=color, edgecolor="white", linewidth=1.2, alpha=0.95)
        offset = offsets.get(row["short"], (7, 7))
        ax.annotate(row["short"], (row["switches"], net), xytext=offset, textcoords="offset points", fontsize=9, arrowprops={"arrowstyle": "-", "color": "#9AAAB8", "lw": 0.7})
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("切换次数")
    ax.set_ylabel("净修复 = Repairs − Harms")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def repair_efficiency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    efficiency_panel(
        axes[0],
        CMMMU,
        "CMMMU：大量切换只带来少量净修复",
        (-30, 960),
        (-16, 10),
        {"FATE": (-42, 10), "Multiview": (-60, -18), "Error-Aware": (-52, 10), "LEAF": (-12, -20), "Shadow": (-45, -18), "Global Best": (-76, 10)},
    )
    efficiency_panel(
        axes[1],
        MATHVISTA,
        "MathVista：RepairChain 以较少切换获得最多净修复",
        (-25, 610),
        (-2, 24),
        {"RepairChain": (-62, 10), "DARE": (8, 4), "ECC": (-30, 10), "LEAF": (-42, -18), "Global Best": (8, -16), "Trace": (8, 8)},
    )
    fig.suptitle("工作三路由诊断：切换得多不等于修复得好", fontsize=22, fontweight="bold", y=0.98)
    add_note(fig, "理想方法位于左上区域：用较少切换获得较多净修复。RepairChain 的净修复为 +21，明显优于 CMMMU 上的高频切换策略。")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14, wspace=0.28)
    save(fig, "13_工作三_修复效率.png")


def main() -> None:
    configure()
    method_pipeline()
    method_results()
    confidence_intervals()
    repair_efficiency()


if __name__ == "__main__":
    main()
