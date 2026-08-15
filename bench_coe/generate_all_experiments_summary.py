from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("outputs/bench_coe")


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def delta(value: Any, baseline: Any) -> str:
    if value is None or baseline is None:
        return "N/A"
    return f"{100 * (float(value) - float(baseline)):+.2f} pp"


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ]


def winner_rows(path: str, key: str) -> list[list[Any]]:
    mapping = load(path)
    winners = mapping[key]
    return [
        [subject, item["selected_model"], pct(item.get("best_accuracy"))]
        for subject, item in winners.items()
    ]


def winner_counts(path: str, key: str) -> str:
    mapping = load(path)
    counts = Counter(item["selected_model"] for item in mapping[key].values())
    return "; ".join(f"{model}: {count}" for model, count in counts.most_common())


def comparison_rows(paths: list[tuple[str, str]], key: str) -> tuple[list[str], list[list[str]]]:
    mappings = [(name, load(path)[key]) for name, path in paths]
    subjects = list(mappings[0][1])
    return [name for name, _ in mappings], [
        [subject, *[mapping[subject]["selected_model"] for _, mapping in mappings]]
        for subject in subjects
    ]


def training_row(name: str, router: str, objective: str, path: str) -> list[Any]:
    metrics = load(path)
    best_epoch = metrics.get("best_epoch")
    best_record = next(
        (row for row in metrics.get("epochs", []) if row.get("epoch") == best_epoch),
        {},
    )
    validation = best_record.get("validation", {})
    train_size = metrics.get("train_size")
    validation_size = metrics.get("validation_size")
    labeled_count = metrics.get("source_labeled_count")
    if labeled_count is None and train_size is not None:
        labeled_count = train_size + (validation_size or 0)
    return [
        name,
        router,
        objective,
        labeled_count if labeled_count is not None else "N/A",
        train_size if train_size is not None else "N/A",
        validation_size if validation_size is not None else "N/A",
        best_epoch if best_epoch is not None else "N/A",
        metrics.get("stopped_epoch", len(metrics.get("epochs", []))),
        pct(validation.get("accuracy")),
    ]


def evaluation_row(
    section: str,
    level: str,
    source: str,
    dataset: str,
    distribution: str,
    router: str,
    path: str,
    route_accuracy_key: str | None = None,
) -> dict[str, Any]:
    summary = load(path)
    route_accuracy = summary.get(route_accuracy_key) if route_accuracy_key else None
    routed_accuracy = summary.get("routed_accuracy")
    best_single_accuracy = summary.get("best_single_accuracy")
    return {
        "section": section,
        "routing_level": level,
        "source_leaderboard": source,
        "evaluation_dataset": dataset,
        "distribution": distribution,
        "router": router,
        "count": summary.get("count", summary.get("examples")),
        "route_label_accuracy": route_accuracy,
        "route_label_count": summary.get(
            "query_label_count", summary.get("subject_accuracy_count")
        ),
        "routed_accuracy": routed_accuracy,
        "best_single_model": summary.get("best_single_model"),
        "best_single_accuracy": best_single_accuracy,
        "bench_coe_delta": (
            routed_accuracy - best_single_accuracy
            if routed_accuracy is not None and best_single_accuracy is not None
            else None
        ),
        "oracle_any_accuracy": summary.get(
            "oracle_any_expert_accuracy", summary.get("oracle_subject_accuracy")
        ),
        "cross_modality_errors": summary.get("cross_modality_route_count"),
        "unavailable_routes": summary.get("unavailable_expert_route_count"),
        "path": str(ROOT / path),
    }


def main() -> None:
    pools = json.loads(Path("bench_coe/configs/expert_pools.json").read_text())["pools"]
    lines = [
        "# Bench-CoE 全实验汇总",
        "",
        "生成日期：2026-07-28。所有训练和推理任务仅使用 GPU 0–3；未在 GPU 4–7 上运行实验。",
        "",
        "> **汇报建议：先看本节 9 张图，再将后面的完整表格作为数字附录。** 独立图文版见 `outputs/bench_coe/ALL_EXPERIMENTS_VISUAL_REPORT.md`。",
        "",
        "## 0. 图表化汇报",
        "",
        "### 0.1 核心结果",
        "",
        "![核心结果总览](visualizations/01_核心结果总览.png)",
        "",
        "**读图结论：** 语言 Subject 低于最佳单模型 6.95 pp；MMMU Subject 最好结果低 2.78 pp；MMMU Query 与最佳单模型持平；Qwen3.5-2B 被替换后统一路由下降 2.06 pp。",
        "",
        "### 0.2 路由训练与专家分配",
        "",
        "![路由训练验证准确率](visualizations/02_路由训练验证准确率.png)",
        "",
        "**说明：** Subject 路由只识别学科，验证准确率为 76.11%–96.12%；Query 路由直接区分专家，受类别不平衡影响，多为 60%–73%。",
        "",
        "![学科专家映射分布](visualizations/03_学科专家映射分布.png)",
        "",
        "**说明：** MMMU 中 InternVL3.5-2B 负责 16/30 个学科；GAOKAO-MM 中 LFM2.5-VL 负责 5/8 个学科，专家分配不均衡是 Query 路由塌缩的重要原因。",
        "",
        "### 0.3 分布内实验",
        "",
        "![分布内结果](visualizations/04_分布内结果对比.png)",
        "",
        "**说明：** GAOKAO-MM Subject 明确超过最佳单模型；MMMU Subject 虽然学科识别准确率高，但最终 Bench-CoE 仍低于最佳单模型。",
        "",
        "![路由准确率与最终增益](visualizations/05_路由准确率与最终增益.png)",
        "",
        "**说明：** 右下角点表示“路由类别预测准确，但最终性能下降”，说明优化目标不能只关注路由分类准确率。",
        "",
        "### 0.4 分布外实验",
        "",
        "![语言OOD](visualizations/06_语言OOD结果.png)",
        "",
        "**说明：** BBH 上路由接近最佳单模型；GPQA 上差距扩大，体现高难推理任务的分布迁移问题。",
        "",
        "![多模态OOD](visualizations/07_多模态OOD增益.png)",
        "",
        "**说明：** 多数多模态 OOD 组合未超过目标数据集最佳单模型，MathVista 的差距最明显。",
        "",
        "### 0.5 统一路由和 BabyVision",
        "",
        "![统一路由](visualizations/08_统一路由结果.png)",
        "",
        "**说明：** 三种统一路由差距较小且跨模态错误极少，主要瓶颈是多模态专家的精细选择，而不是模态识别。",
        "",
        "![BabyVision替换](visualizations/09_BabyVision替换实验.png)",
        "",
        "**说明：** 原池统一 Qwen3-VL 达到 17.27%；将 Qwen3.5-2B 替换后降至 12.63%，证明专家互补性需要单独评估。",
        "",
        "---",
        "",
        "以下为完整实验设置与逐项数字附录。",
        "",
        "## 1. 实验目标与统一口径",
        "",
        "- Subject 路由输出学科类别，类别数等于学科数；学科类别再通过学科专家映射选择专家。因此专家池更新时只更新映射，不必重训 subject 分类器。",
        "- Query 路由直接输出专家类别；若多个专家答对同一问题，标签分配给源域整体准确率最高的正确专家。专家类别发生变化时重新构建标签并训练分类头。",
        "- Bench-CoE 结果指路由后实际选中专家答案的准确率（Routed Accuracy），并与同一可用专家池的最佳单模型和 oracle-any-expert 比较。",
        "- 所有可早停训练均保留验证损失最优 checkpoint。MMLU-Pro validation 仅 70 条，因此同时报告 test-source query 方案；GAOKAO-MM query 同时报告严格 train517 和全646来源方案。",
        "",
        "### 核心结果总览",
        "",
        "| Setting | Best Bench-CoE Result | Best Single | Gain |",
        "| --- | --- | --- | --- |",
        "| 语言 ID：MMLU validation subject → MMLU-Pro test | BERT 50.66% | Qwen2.5-7B-Instruct 57.61% | -6.95 pp |",
        "| 语言 ID：MMLU test-source query holdout | BERT 70.18% | Qwen2.5-7B-Instruct 70.18% | +0.00 pp |",
        "| 多模态 ID：MMMU subject holdout | Qwen3-VL 48.33% | InternVL3_5-2B 51.11% | -2.78 pp |",
        "| 多模态 ID：MMMU query holdout | TinyLLaVA / Qwen3-VL 均为 51.11% | InternVL3_5-2B 51.11% | +0.00 pp |",
        "| BabyVision OOD：原专家池 | 统一 Qwen3-VL 17.27% | Qwen3.5-2B 17.01% | +0.26 pp |",
        "| BabyVision OOD：Qwen3-VL 替换池 | 统一 Qwen3-VL 12.63% | LFM2.5-VL-1.6B 14.69% | -2.06 pp |",
        "",
        "## 2. 专家模型池",
        "",
    ]
    pool_rows = []
    for name, pool in pools.items():
        eligible = [item["name"] for item in pool["models"] if not item.get("exclude_reason")]
        excluded = [
            f'{item["name"]}：{item["exclude_reason"]}'
            for item in pool["models"]
            if item.get("exclude_reason")
        ]
        pool_rows.append(
            [
                name,
                pool["modality"],
                pool["scale_group"],
                "、".join(eligible),
                "；".join(excluded) or "无",
            ]
        )
    lines += table(["Pool", "Modality", "Scale", "Eligible Experts", "Excluded"], pool_rows)
    lines += [
        "",
        "### 路由模型",
        "",
        "| Router | 用途 | 训练方式 |",
        "| --- | --- | --- |",
        "| BERT-base-uncased | 语言 subject/query；统一语言+多模态 query | 全参数分类微调 |",
        "| TinyLLaVA-Phi-2-SigLIP-3.1B | 多模态 subject/query；统一路由 | 冻结特征提取器，仅训练 LayerNorm+Linear 分类头 |",
        "| Qwen3-VL-2B-Instruct | 多模态 subject/query；统一路由 | 冻结特征提取器，仅训练 LayerNorm+Linear 分类头 |",
        "",
        "## 3. 学科性能评估榜与专家映射",
        "",
        "### 3.1 MMLU-Pro validation 语言榜（14 类）",
        "",
        f"专家分配统计：{winner_counts('mmlu_validation_7b_9b/validation_7b_9b_expert_category_mapping.json', 'category_winners')}。",
        "",
    ]
    lines += table(
        ["Category", "Selected Expert", "Source Accuracy"],
        winner_rows(
            "mmlu_validation_7b_9b/validation_7b_9b_expert_category_mapping.json",
            "category_winners",
        ),
    )
    lines += [
        "",
        "### 3.2 GAOKAO-Bench 语言榜（9 学科）",
        "",
        f"专家分配统计：{winner_counts('gaokao_no_qwen35_deepseek_qwen3/local_expert_subject_mapping.json', 'subject_winners')}。",
        "",
    ]
    lines += table(
        ["Subject", "Selected Expert", "Source Accuracy"],
        winner_rows(
            "gaokao_no_qwen35_deepseek_qwen3/local_expert_subject_mapping.json",
            "subject_winners",
        ),
    )
    lines += [
        "",
        "### 3.3 MMMU train720 多模态榜（30 学科）",
        "",
        f"专家分配统计：{winner_counts('mmmu_train720_1p6b_2p2b/mmmu_train720_1p6b_2p2b_expert_subject_mapping.json', 'subject_winners')}。",
        "",
    ]
    lines += table(
        ["Subject", "Selected Expert", "Source Accuracy"],
        winner_rows(
            "mmmu_train720_1p6b_2p2b/mmmu_train720_1p6b_2p2b_expert_subject_mapping.json",
            "subject_winners",
        ),
    )
    lines += [
        "",
        "### 3.4 GAOKAO-MM train517 多模态榜（8 学科）",
        "",
        f"专家分配统计：{winner_counts('gaokao_mm_train517_1p6b_2p2b/gaokao_mm_train517_1p6b_2p2b_expert_subject_mapping.json', 'subject_winners')}。",
        "",
    ]
    lines += table(
        ["Subject", "Selected Expert", "Source Accuracy"],
        winner_rows(
            "gaokao_mm_train517_1p6b_2p2b/gaokao_mm_train517_1p6b_2p2b_expert_subject_mapping.json",
            "subject_winners",
        ),
    )
    lines += [
        "",
        "### 3.5 BabyVision 可判分池映射变化",
        "",
        "| Source Leaderboard | Expert Assignment Counts |",
        "| --- | --- |",
        f"| MMMU / 原6专家池 | {winner_counts('mmmu_train720_1p6b_2p2b/mmmu_train720_1p6b_2p2b_expert_subject_mapping.json', 'subject_winners')} |",
        f"| MMMU / BabyVision judged4 | {winner_counts('mmmu_train720_babyvision_judged4/mmmu_train720_babyvision_judged4_expert_subject_mapping.json', 'subject_winners')} |",
        f"| MMMU / Qwen3-VL替换池 | {winner_counts('mmmu_train720_babyvision_qwen3vl_swap/mmmu_train720_babyvision_qwen3vl_swap_expert_subject_mapping.json', 'subject_winners')} |",
        f"| GAOKAO-MM / 原6专家池 | {winner_counts('gaokao_mm_train517_1p6b_2p2b/gaokao_mm_train517_1p6b_2p2b_expert_subject_mapping.json', 'subject_winners')} |",
        f"| GAOKAO-MM / judged4 | {winner_counts('gaokao_mm_train517_babyvision_judged4/gaokao_mm_train517_babyvision_judged4_expert_subject_mapping.json', 'subject_winners')} |",
        f"| GAOKAO-MM / Qwen3-VL替换池 | {winner_counts('gaokao_mm_train517_babyvision_qwen3vl_swap/gaokao_mm_train517_babyvision_qwen3vl_swap_expert_subject_mapping.json', 'subject_winners')} |",
        "",
        "#### MMMU 逐学科映射对比",
        "",
    ]
    comparison_headers, comparison = comparison_rows(
        [
            ("原6专家池", "mmmu_train720_1p6b_2p2b/mmmu_train720_1p6b_2p2b_expert_subject_mapping.json"),
            ("judged4", "mmmu_train720_babyvision_judged4/mmmu_train720_babyvision_judged4_expert_subject_mapping.json"),
            ("Qwen3-VL替换", "mmmu_train720_babyvision_qwen3vl_swap/mmmu_train720_babyvision_qwen3vl_swap_expert_subject_mapping.json"),
        ],
        "subject_winners",
    )
    lines += table(["Subject", *comparison_headers], comparison)
    lines += ["", "#### GAOKAO-MM 逐学科映射对比", ""]
    comparison_headers, comparison = comparison_rows(
        [
            ("原6专家池", "gaokao_mm_train517_1p6b_2p2b/gaokao_mm_train517_1p6b_2p2b_expert_subject_mapping.json"),
            ("judged4", "gaokao_mm_train517_babyvision_judged4/gaokao_mm_train517_babyvision_judged4_expert_subject_mapping.json"),
            ("Qwen3-VL替换", "gaokao_mm_train517_babyvision_qwen3vl_swap/gaokao_mm_train517_babyvision_qwen3vl_swap_expert_subject_mapping.json"),
        ],
        "subject_winners",
    )
    lines += table(["Subject", *comparison_headers], comparison)
    lines += [
        "",
        "## 4. 路由模型训练/验证准确率",
        "",
        "下表的 Validation Accuracy 是最佳 checkpoint 在内部验证集上的路由类别准确率；subject 路由对应学科识别准确率，query 路由对应专家标签准确率。",
        "",
    ]
    training_specs = [
        ("MMLU test-source query", "BERT", "语言 query", "router/bert-base-mmlu-test-query-7b-9b/train_metrics.json"),
        ("MMMU subject", "TinyLLaVA", "30学科", "router/tinyllava-mmmu-30subject-1p6b-2p2b/train_metrics.json"),
        ("MMMU subject", "Qwen3-VL-2B", "30学科", "router/qwen3vl-2b-mmmu-30subject-1p6b-2p2b/train_metrics.json"),
        ("GAOKAO-MM subject", "TinyLLaVA", "8学科", "router/tinyllava-gaokao-mm-8subject-1p6b-2p2b/train_metrics.json"),
        ("GAOKAO-MM subject", "Qwen3-VL-2B", "8学科", "router/qwen3vl-2b-gaokao-mm-8subject-1p6b-2p2b/train_metrics.json"),
        ("MMMU query", "TinyLLaVA", "专家标签", "router/tinyllava-mmmu-query-1p6b-2p2b/train_metrics.json"),
        ("MMMU query", "Qwen3-VL-2B", "专家标签", "router/qwen3vl-2b-mmmu-query-1p6b-2p2b/train_metrics.json"),
        ("GAOKAO-MM strict query", "TinyLLaVA", "专家标签", "router/tinyllava-gaokao-mm-query-strict/train_metrics.json"),
        ("GAOKAO-MM all646 query", "TinyLLaVA", "专家标签", "router/tinyllava-gaokao-mm-query-all646/train_metrics.json"),
        ("GAOKAO-MM strict query", "Qwen3-VL-2B", "专家标签", "router/qwen3vl-gaokao-mm-query-strict/train_metrics.json"),
        ("GAOKAO-MM all646 query", "Qwen3-VL-2B", "专家标签", "router/qwen3vl-gaokao-mm-query-all646/train_metrics.json"),
        ("Unified query", "BERT", "模态+专家", "router/bert-base-unified-mmlu-test-mmmu-train-query/train_metrics.json"),
        ("Unified query", "TinyLLaVA", "模态+专家", "router/tinyllava-unified-mmlu-test-mmmu-train-query/train_metrics.json"),
        ("Unified query", "Qwen3-VL-2B", "模态+专家", "router/qwen3vl-unified-mmlu-test-mmmu-train-query/train_metrics.json"),
        ("Swap MMMU query", "TinyLLaVA", "4专家标签", "router/qwen3vl_swap/tinyllava_mmmu/train_metrics.json"),
        ("Swap MMMU query", "Qwen3-VL-2B", "4专家标签", "router/qwen3vl_swap/qwen3vl_mmmu/train_metrics.json"),
        ("Swap GAOKAO strict query", "TinyLLaVA", "2专家标签", "router/qwen3vl_swap/tinyllava_gaokao_strict/train_metrics.json"),
        ("Swap GAOKAO all646 query", "TinyLLaVA", "2专家标签", "router/qwen3vl_swap/tinyllava_gaokao_all646/train_metrics.json"),
        ("Swap GAOKAO strict query", "Qwen3-VL-2B", "2专家标签", "router/qwen3vl_swap/qwen3vl_gaokao_strict/train_metrics.json"),
        ("Swap GAOKAO all646 query", "Qwen3-VL-2B", "2专家标签", "router/qwen3vl_swap/qwen3vl_gaokao_all646/train_metrics.json"),
        ("Swap unified query", "BERT", "模态+专家", "router/qwen3vl_swap/bert_unified/train_metrics.json"),
        ("Swap unified query", "TinyLLaVA", "模态+专家", "router/qwen3vl_swap/tinyllava_unified/train_metrics.json"),
        ("Swap unified query", "Qwen3-VL-2B", "模态+专家", "router/qwen3vl_swap/qwen3vl_unified/train_metrics.json"),
    ]
    training_rows = [training_row(*spec) for spec in training_specs]
    lines += table(
        ["Experiment", "Router", "Objective", "Labeled", "Train", "Val", "Best Epoch", "Stopped", "Val Acc"],
        training_rows,
    )
    lines += [
        "",
        "补充：MMLU validation subject 仅 70 条、GAOKAO language subject 训练未设置内部验证集，均固定训练 10 轮；其独立评估路由准确率在下一节报告。",
        "",
        "## 5. 分布内（ID）实验",
        "",
    ]

    evaluations: list[dict[str, Any]] = []
    language_id_specs = [
        ("Language", "Subject", "MMLU validation", "MMLU-Pro test", "ID/test", "BERT", "mmlu_subject_bert_validation_7b_9b_offline/test_summary.json", "subject_accuracy"),
        ("Language", "Query", "MMLU test-source", "Internal holdout", "ID holdout", "BERT", "mmlu_test_query_router_holdout/test_summary.json", "query_route_accuracy"),
    ]
    multimodal_id_specs = [
        ("Multimodal", "Subject", "MMMU train720", "MMMU holdout", "ID holdout", "TinyLLaVA", "tinyllava_mmmu_train720_router_on_mmmu_holdout/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "MMMU train720", "MMMU holdout", "ID holdout", "Qwen3-VL-2B", "qwen3vl_mmmu_train720_router_on_mmmu_holdout/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "GAOKAO-MM train517", "GAOKAO-MM holdout", "ID holdout", "TinyLLaVA", "tinyllava-gaokao-mm-router-holdout/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "GAOKAO-MM train517", "GAOKAO-MM holdout", "ID holdout", "Qwen3-VL-2B", "qwen3vl-gaokao-mm-router-holdout/summary.json", "subject_accuracy"),
        ("Multimodal", "Query", "MMMU train720", "MMMU holdout", "ID holdout", "TinyLLaVA", "tiny_mmmu_query_on_mmmu_holdout/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "MMMU train720", "MMMU holdout", "ID holdout", "Qwen3-VL-2B", "qwen_mmmu_query_on_mmmu_holdout/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM train517 strict", "GAOKAO-MM holdout", "ID holdout", "TinyLLaVA", "tinyllava_gaokao_query_strict_holdout/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM all646", "GAOKAO-MM holdout", "ID holdout", "TinyLLaVA", "tinyllava_gaokao_query_all646_holdout/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM train517 strict", "GAOKAO-MM holdout", "ID holdout", "Qwen3-VL-2B", "qwen3vl_gaokao_query_strict_holdout/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM all646", "GAOKAO-MM holdout", "ID holdout", "Qwen3-VL-2B", "qwen3vl_gaokao_query_all646_holdout/summary.json", "query_label_accuracy"),
    ]
    for spec in language_id_specs + multimodal_id_specs:
        evaluations.append(evaluation_row(*spec))
    id_rows = []
    for row in evaluations:
        id_rows.append(
            [
                row["routing_level"], row["source_leaderboard"], row["evaluation_dataset"], row["router"],
                row["count"], pct(row["route_label_accuracy"]), pct(row["routed_accuracy"]),
                row["best_single_model"], pct(row["best_single_accuracy"]),
                delta(row["routed_accuracy"], row["best_single_accuracy"]), pct(row["oracle_any_accuracy"]),
            ]
        )
    lines += table(
        ["Level", "Source", "Evaluation", "Router", "N", "Route Acc", "Bench-CoE", "Best Single", "Single Acc", "Delta", "Oracle"],
        id_rows,
    )

    lines += ["", "### 5.1 统一语言+多模态 ID", ""]
    unified_rows = []
    for router, path in [
        ("BERT", "bert_unified_query_internal_validation/summary.json"),
        ("TinyLLaVA", "tinyllava_unified_query_internal_validation/summary.json"),
        ("Qwen3-VL-2B", "qwen3vl_unified_query_internal_validation/summary.json"),
    ]:
        summary = load(path)
        unified_rows.append([
            router, summary["count"], pct(summary["target_label_accuracy"]), pct(summary["routed_accuracy"]),
            pct(summary["by_modality"]["language"]["routed_accuracy"]),
            pct(summary["by_modality"]["multimodal"]["routed_accuracy"]), summary["cross_modality_route_count"],
        ])
        evaluations.append(evaluation_row("Unified", "Query", "MMLU test-source + MMMU train720", "Internal validation", "ID validation", router, path, "target_label_accuracy"))
    lines += table(["Router", "N", "Route Acc", "Overall Bench-CoE", "Language", "Multimodal", "Cross-Modality"], unified_rows)
    lines += ["", "### 5.2 统一路由 MMMU 干净留出", ""]
    unified_holdout_rows = []
    for router, path in [
        ("BERT", "bert_unified_query_on_mmmu_holdout/summary.json"),
        ("TinyLLaVA", "tinyllava_unified_query_on_mmmu_holdout/summary.json"),
        ("Qwen3-VL-2B", "qwen3vl_unified_query_on_mmmu_holdout/summary.json"),
    ]:
        summary = load(path)
        unified_holdout_rows.append([
            router, summary["count"], pct(summary.get("target_label_accuracy", summary.get("query_label_accuracy"))),
            pct(summary["routed_accuracy"]), summary.get("cross_modality_route_count", 0),
        ])
        evaluations.append(evaluation_row("Unified", "Query", "MMLU test-source + MMMU train720", "MMMU clean holdout", "ID holdout", router, path, "target_label_accuracy"))
    lines += table(["Router", "N", "Route Acc", "Bench-CoE", "Cross-Modality"], unified_holdout_rows)

    lines += ["", "## 6. 分布外（OOD）实验", "", "### 6.1 语言 OOD", ""]
    language_ood_specs = [
        ("Language", "Subject", "MMLU validation", "BBH", "OOD", "BERT", "mmlu_validation_7b_9b_subject_router_on_bbh/summary.json", None),
        ("Language", "Subject", "MMLU validation", "GPQA", "OOD", "BERT", "mmlu_validation_7b_9b_subject_router_on_gpqa/summary.json", "subject_accuracy"),
        ("Language", "Subject", "GAOKAO", "MMLU-Pro test", "OOD", "BERT", "gaokao_9subject_filtered_router_on_mmlu_pro/test_summary.json", "subject_accuracy"),
        ("Language", "Subject", "GAOKAO", "BBH", "OOD", "BERT", "gaokao_9subject_filtered_router_on_bbh/summary.json", None),
        ("Language", "Subject", "GAOKAO", "GPQA", "OOD", "BERT", "gaokao_9subject_filtered_router_on_gpqa/summary.json", "subject_accuracy"),
        ("Language", "Query", "MMLU test-source", "BBH", "OOD", "BERT", "mmlu_test_query_router_on_bbh/summary.json", None),
        ("Language", "Query", "MMLU test-source", "GPQA", "OOD", "BERT", "mmlu_test_query_router_on_gpqa/summary.json", None),
    ]
    language_ood = [evaluation_row(*spec) for spec in language_ood_specs]
    evaluations += language_ood
    lines += table(
        ["Level", "Source", "Dataset", "Router", "N", "Route Acc", "Bench-CoE", "Best Single", "Single Acc", "Delta"],
        [[r["routing_level"], r["source_leaderboard"], r["evaluation_dataset"], r["router"], r["count"], pct(r["route_label_accuracy"]), pct(r["routed_accuracy"]), r["best_single_model"], pct(r["best_single_accuracy"]), delta(r["routed_accuracy"], r["best_single_accuracy"])] for r in language_ood],
    )

    lines += ["", "### 6.2 多模态 Subject OOD", ""]
    subject_ood_specs = []
    for router, prefix in [("TinyLLaVA", "tinyllava_mmmu_train720_router_on_"), ("Qwen3-VL-2B", "qwen3vl_mmmu_train720_router_on_")]:
        for dataset, suffix in [("CMMMU dev", "cmmmu_dev"), ("MMMU-Pro test", "mmmu_pro_test"), ("MathVista testmini", "mathvista_testmini")]:
            subject_ood_specs.append(("Multimodal", "Subject", "MMMU train720", dataset, "OOD", router, f"{prefix}{suffix}/summary.json", "subject_accuracy"))
    for router, prefix in [("TinyLLaVA", "tinyllava_gaokao_mm_router_on_"), ("Qwen3-VL-2B", "qwen3vl_gaokao_mm_router_on_")]:
        for dataset, suffix in [("MMMU validation", "mmmu_validation"), ("CMMMU dev", "cmmmu_dev"), ("MMMU-Pro test", "mmmu_pro_test"), ("MathVista testmini", "mathvista_testmini")]:
            subject_ood_specs.append(("Multimodal", "Subject", "GAOKAO-MM train517", dataset, "OOD", router, f"{prefix}{suffix}/summary.json", "subject_accuracy"))
    subject_ood = [evaluation_row(*spec) for spec in subject_ood_specs]
    evaluations += subject_ood
    lines += table(
        ["Source", "Dataset", "Router", "N", "Subject Acc", "Bench-CoE", "Best Single", "Single Acc", "Delta", "Oracle"],
        [[r["source_leaderboard"], r["evaluation_dataset"], r["router"], r["count"], pct(r["route_label_accuracy"]), pct(r["routed_accuracy"]), r["best_single_model"], pct(r["best_single_accuracy"]), delta(r["routed_accuracy"], r["best_single_accuracy"]), pct(r["oracle_any_accuracy"])] for r in subject_ood],
    )

    lines += ["", "### 6.3 多模态 Query OOD", ""]
    query_ood_specs = []
    for router, prefix in [("TinyLLaVA", "tiny_mmmu_query_on_"), ("Qwen3-VL-2B", "qwen_mmmu_query_on_")]:
        for dataset, suffix in [("CMMMU dev", "cmmmu"), ("MMMU-Pro test", "mmmu_pro"), ("MathVista testmini", "mathvista")]:
            query_ood_specs.append(("Multimodal", "Query", "MMMU train720", dataset, "OOD", router, f"{prefix}{suffix}/summary.json", "query_label_accuracy"))
    for router, prefix in [("TinyLLaVA", "tinyllava_gaokao_query_"), ("Qwen3-VL-2B", "qwen3vl_gaokao_query_")]:
        for variant in ["strict", "all646"]:
            for dataset, suffix in [("MMMU validation", "mmmu"), ("CMMMU dev", "cmmmu"), ("MMMU-Pro test", "mmmu_pro"), ("MathVista testmini", "mathvista")]:
                query_ood_specs.append(("Multimodal", "Query", f"GAOKAO-MM {variant}", dataset, "OOD", router, f"{prefix}{variant}_on_{suffix}/summary.json", "query_label_accuracy"))
    query_ood = [evaluation_row(*spec) for spec in query_ood_specs]
    evaluations += query_ood
    lines += table(
        ["Source", "Dataset", "Router", "N", "Bench-CoE", "Best Single", "Single Acc", "Delta", "Oracle"],
        [[r["source_leaderboard"], r["evaluation_dataset"], r["router"], r["count"], pct(r["routed_accuracy"]), r["best_single_model"], pct(r["best_single_accuracy"]), delta(r["routed_accuracy"], r["best_single_accuracy"]), pct(r["oracle_any_accuracy"])] for r in query_ood],
    )

    lines += ["", "## 7. BabyVision OOD 与专家替换实验", "", "### 7.1 原 judged4 专家池", ""]
    baby_expert_rows = []
    for model in ["InternVL3_5-2B", "LFM2.5-VL-1.6B", "Qwen3.5-2B", "gemma-4-E2B-it"]:
        summary = load(f"babyvision_standardized_1p6b_2p2b/{model}/summary.json")
        baby_expert_rows.append([model, summary["count"], pct(summary["accuracy"]), summary["judge"]])
    lines += table(["Expert", "N", "Accuracy", "Judge"], baby_expert_rows)

    baby_specs = [
        ("Multimodal", "Subject", "MMMU judged4", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_mmmu_subject_router_on_babyvision/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "MMMU judged4", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_mmmu_subject_router_on_babyvision/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "GAOKAO-MM judged4", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_gaokao_mm_subject_router_on_babyvision/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "GAOKAO-MM judged4", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_gaokao_mm_subject_router_on_babyvision/summary.json", "subject_accuracy"),
        ("Multimodal", "Query", "MMMU", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_mmmu_query_router_on_babyvision/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "MMMU", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_mmmu_query_router_on_babyvision/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM strict", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_gaokao_query_strict_on_babyvision/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM strict", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_gaokao_query_strict_on_babyvision/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM all646", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_gaokao_query_all646_on_babyvision/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM all646", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_gaokao_query_all646_on_babyvision/summary.json", "query_label_accuracy"),
        ("Unified", "Query", "MMLU+MMMU", "BabyVision", "OOD", "BERT", "bert_unified_query_on_babyvision/summary.json", None),
        ("Unified", "Query", "MMLU+MMMU", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_unified_query_on_babyvision/summary.json", None),
        ("Unified", "Query", "MMLU+MMMU", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_unified_query_on_babyvision/summary.json", None),
    ]
    baby_rows = [evaluation_row(*spec) for spec in baby_specs]
    evaluations += baby_rows
    lines += [""] + table(
        ["Level", "Source", "Router", "Bench-CoE", "Best Single", "Single Acc", "Delta", "Oracle", "Unavailable"],
        [[r["routing_level"], r["source_leaderboard"], r["router"], pct(r["routed_accuracy"]), r["best_single_model"], pct(r["best_single_accuracy"]), delta(r["routed_accuracy"], r["best_single_accuracy"]), pct(r["oracle_any_accuracy"]), r["unavailable_routes"] or 0] for r in baby_rows],
    )

    lines += ["", "### 7.2 Qwen3-VL-2B-Instruct 替换 Qwen3.5-2B", "", "Qwen3-VL 在 388 题中有 374 个有效本地判分、56 题正确；14 个缺失判分按保守错误计，全量准确率 14.43%，有效判分内准确率 14.97%。", ""]
    swap_expert_rows = []
    for model in ["InternVL3_5-2B", "LFM2.5-VL-1.6B", "Qwen3-VL-2B-Instruct", "gemma-4-E2B-it"]:
        summary = load(f"babyvision_standardized_qwen3vl_swap/{model}/summary.json")
        swap_expert_rows.append([model, summary["count"], pct(summary["accuracy"]), summary["judge"]])
    lines += table(["Expert", "N", "Accuracy", "Judge"], swap_expert_rows)
    swap_specs = [
        ("Multimodal", "Subject", "MMMU Qwen3-VL swap", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_mmmu_subject_router_on_babyvision_qwen3vl_swap/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "MMMU Qwen3-VL swap", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_mmmu_subject_router_on_babyvision_qwen3vl_swap/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "GAOKAO-MM Qwen3-VL swap", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_gaokao_mm_subject_router_on_babyvision_qwen3vl_swap/summary.json", "subject_accuracy"),
        ("Multimodal", "Subject", "GAOKAO-MM Qwen3-VL swap", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_gaokao_mm_subject_router_on_babyvision_qwen3vl_swap/summary.json", "subject_accuracy"),
        ("Multimodal", "Query", "MMMU Qwen3-VL swap", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_mmmu_query_on_babyvision_qwen3vl_swap/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "MMMU Qwen3-VL swap", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_mmmu_query_on_babyvision_qwen3vl_swap/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM strict swap", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_gaokao_strict_query_on_babyvision_qwen3vl_swap/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM strict swap", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_gaokao_strict_query_on_babyvision_qwen3vl_swap/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM all646 swap", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_gaokao_all646_query_on_babyvision_qwen3vl_swap/summary.json", "query_label_accuracy"),
        ("Multimodal", "Query", "GAOKAO-MM all646 swap", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_gaokao_all646_query_on_babyvision_qwen3vl_swap/summary.json", "query_label_accuracy"),
        ("Unified", "Query", "MMLU+MMMU swap", "BabyVision", "OOD", "BERT", "bert_unified_query_on_babyvision_qwen3vl_swap/summary.json", None),
        ("Unified", "Query", "MMLU+MMMU swap", "BabyVision", "OOD", "TinyLLaVA", "tinyllava_unified_query_on_babyvision_qwen3vl_swap/summary.json", None),
        ("Unified", "Query", "MMLU+MMMU swap", "BabyVision", "OOD", "Qwen3-VL-2B", "qwen3vl_unified_query_on_babyvision_qwen3vl_swap/summary.json", None),
    ]
    swap_rows = [evaluation_row(*spec) for spec in swap_specs]
    evaluations += swap_rows
    lines += [""] + table(
        ["Level", "Source", "Router", "Bench-CoE", "Best Single", "Single Acc", "Delta", "Oracle"],
        [[r["routing_level"], r["source_leaderboard"], r["router"], pct(r["routed_accuracy"]), r["best_single_model"], pct(r["best_single_accuracy"]), delta(r["routed_accuracy"], r["best_single_accuracy"]), pct(r["oracle_any_accuracy"])] for r in swap_rows],
    )

    lines += [
        "",
        "## 8. 主要结论",
        "",
        "1. **路由类别识别不等于最终性能提升。** MMMU subject 路由学科准确率达到 76.11%–83.89%，但 Bench-CoE 仍低于 InternVL3.5-2B；专家互补性和映射质量是主要瓶颈。",
        "2. **GAOKAO-MM subject 路由存在明确增益。** 两种路由在 129 条留出集上均达到 10.08%，比最佳单专家 8.53% 提高 1.55 个百分点。",
        "3. **MMMU query 路由发生专家塌缩。** ID 留出 Bench-CoE 等于最佳单模型 51.11%，OOD 上多数也接近或低于最佳单模型，需要类别重加权、代价敏感损失或负载均衡。",
        "4. **小验证集下全量来源标签有时有效。** TinyLLaVA GAOKAO-MM all646 在留出集达到 9.30%，比严格版本 8.53% 和最佳单专家 8.53% 高 0.78 个百分点；Qwen3-VL 未获得同样收益。",
        "5. **统一路由能够稳定识别模态。** ID 验证中跨模态错误为 0、0、1；但多模态部分仍主要选择强单模型，尚未充分逼近 oracle。",
        "6. **BabyVision 的最佳原始结果来自统一 Qwen3-VL 路由。** 17.27% 比 Qwen3.5-2B 单模型 17.01% 高 0.26 个百分点，但距离四专家 oracle 31.19% 仍有 13.92 个百分点。",
        "7. **Qwen3.5-2B 不能被 Qwen3-VL-2B-Instruct 直接替代。** 替换后最佳单模型降至 14.69%、oracle 降至 29.12%、统一 Qwen3-VL Bench-CoE 降至 12.63%。",
        "",
        "## 9. 关键产物",
        "",
        "- 专家池：`bench_coe/configs/expert_pools.json`",
        "- 原增量报告：`outputs/bench_coe/EXPERIMENT_REPORT.md`",
        "- 本总报告：`outputs/bench_coe/ALL_EXPERIMENTS_SUMMARY.md`",
        "- 全结果 CSV：`outputs/bench_coe/ALL_EXPERIMENTS_RESULTS.csv`",
        "- 学科榜：`outputs/bench_coe/mmlu_validation_7b_9b/`、`outputs/bench_coe/gaokao_no_qwen35_deepseek_qwen3/`、`outputs/bench_coe/mmmu_train720_1p6b_2p2b/`、`outputs/bench_coe/gaokao_mm_train517_1p6b_2p2b/`",
        "- 路由模型：`outputs/bench_coe/router/`",
        "- BabyVision 原池/替换池：`outputs/bench_coe/babyvision_standardized_1p6b_2p2b/`、`outputs/bench_coe/babyvision_standardized_qwen3vl_swap/`",
    ]

    (ROOT / "ALL_EXPERIMENTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "section", "routing_level", "source_leaderboard", "evaluation_dataset", "distribution",
        "router", "count", "route_label_accuracy", "route_label_count", "routed_accuracy",
        "best_single_model", "best_single_accuracy", "bench_coe_delta", "oracle_any_accuracy",
        "cross_modality_errors", "unavailable_routes", "path",
    ]
    with (ROOT / "ALL_EXPERIMENTS_RESULTS.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluations)


if __name__ == "__main__":
    main()
