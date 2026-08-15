from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "outputs/bench_coe/scale_transfer_improve56_20260802"
REPORT_DIR = ROOT / "outputs/bench_coe/scale_expansion_and_transfer_summary_20260802"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    output = []
    for case_id, case_rows in sorted(grouped.items()):
        candidates = [row for row in case_rows if row.get("method") != "source_global_best"] or case_rows
        best = max(candidates, key=lambda row: float(row.get("target_accuracy", 0.0)))
        output.append({"case_id": case_id, **best})
    return output


def collect() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cohort_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir() and path.name != "logs"):
        for improve in ("improve5", "improve6"):
            rows = load_rows(cohort_dir / improve / "summary.json")
            for row in best_rows(rows):
                records.append({"cohort": cohort_dir.name, "strategy": improve, **row})
    return records


def render_plot(records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['cohort']}\n{row['strategy']}\n{row['case_id']}" for row in records]
    gains = [float(row.get("gain_vs_best_single_target", 0.0)) * 100 for row in records]
    colors = ["#2878B5" if row["strategy"] == "improve5" else "#C82423" for row in records]
    width = max(12, len(records) * 0.62)
    fig, ax = plt.subplots(figsize=(width, 7))
    ax.bar(range(len(records)), gains, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Best gain over best single model (percentage points)")
    ax.set_title("Improve5/6 transfer across model scales")
    ax.set_xticks(range(len(records)), labels, rotation=65, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "improve5_improve6_scale_transfer_gains.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect()
    (REPORT_DIR / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 扩展模型规模后的改进5和改进6实验汇总",
        "",
        "本报告分别汇总 2B–4B 与约 14B 语言模型、7B–9B 与约 14B 视觉语言模型上的迁移实验。",
        "正增益表示路由策略优于该组中表现最好的单模型，负增益表示尚未稳定超过最强单模型。",
        "",
        "| 模型组 | 策略 | 迁移任务 | 最佳方法 | 准确率 | 最强单模型 | 增益 | 模型数 |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| `{row['cohort']}` | `{row['strategy']}` | `{row['case_id']}` | "
            f"`{row.get('method', '')}` | {float(row.get('target_accuracy', 0)):.2%} | "
            f"{row.get('best_single_model_target', '')} ({float(row.get('best_single_target', 0)):.2%}) | "
            f"{float(row.get('gain_vs_best_single_target', 0)) * 100:+.2f} pp | {row.get('models_used', '')} |"
        )
    lines.extend([
        "",
        "## 简要说明",
        "",
        "- 改进5侧重从模型失败模式与输出分歧中识别局部优势专家。",
        "- 改进6在改进5基础上增加自适应失败生态建模，比较其跨任务迁移稳定性。",
        "- 各组独立比较，避免不同参数规模和语言/视觉任务之间直接混合造成偏差。",
        "- 图中展示每个任务上最佳改进方法相对该组最强单模型的百分点增益。",
        "",
    ])
    (REPORT_DIR / "improve5_improve6_scale_transfer_summary.md").write_text("\n".join(lines), encoding="utf-8")
    if records:
        render_plot(records)


if __name__ == "__main__":
    main()
