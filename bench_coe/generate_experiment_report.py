from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/bench_coe")


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ]


def main() -> None:
    lines = [
        "# Bench-CoE 增量实验报告",
        "",
        "生成日期：2026-07-28。所有 GPU 任务均通过 `CUDA_VISIBLE_DEVICES` 绑定 GPU 0–3；未在 GPU 4–7 上启动实验。",
        "",
        "## 训练策略",
        "",
        "- 路由分类头均使用内部验证集、验证损失最佳 checkpoint 和 early stopping。",
        "- MMLU-Pro validation 仅 70 条，因此另做 MMLU-Pro test-source 标签实验；该版本明确标记为 test-source，并使用内部留出及 OOD 结果，不将整个 test-source 训练结果表述为无泄漏测试。",
        "- GAOKAO-MM query 严格训练子集仅 73 条有效标签，因此按要求额外训练全 646 条来源版本，并保留 18 条内部验证标签。",
        "- 学科专家映射仅由训练划分构建；更新专家映射不需要重训 subject 路由器。",
        "",
        "## 语言 Subject / Query 路由",
        "",
    ]
    rows = []
    for experiment, dataset, path in [
        ("MMLU validation subject", "MMLU-Pro test", "mmlu_subject_bert_validation_7b_9b_offline/test_summary.json"),
        ("MMLU validation subject", "BBH", "mmlu_validation_7b_9b_subject_router_on_bbh/summary.json"),
        ("MMLU validation subject", "GPQA", "mmlu_validation_7b_9b_subject_router_on_gpqa/summary.json"),
        ("GAOKAO subject", "MMLU-Pro test", "gaokao_9subject_filtered_router_on_mmlu_pro/test_summary.json"),
        ("GAOKAO subject", "BBH", "gaokao_9subject_filtered_router_on_bbh/summary.json"),
        ("GAOKAO subject", "GPQA", "gaokao_9subject_filtered_router_on_gpqa/summary.json"),
        ("MMLU test-source query", "Internal holdout", "mmlu_test_query_router_holdout/test_summary.json"),
        ("MMLU test-source query", "BBH", "mmlu_test_query_router_on_bbh/summary.json"),
        ("MMLU test-source query", "GPQA", "mmlu_test_query_router_on_gpqa/summary.json"),
    ]:
        summary = load(path)
        route_label_accuracy = summary.get("query_route_accuracy", summary.get("subject_accuracy"))
        rows.append([experiment, dataset, pct(route_label_accuracy), pct(summary["routed_accuracy"]), summary["best_single_model"], pct(summary["best_single_accuracy"])])
    lines += table(["Experiment", "Dataset", "Route Label Acc", "Routed Acc", "Best Single", "Best Single Acc"], rows)
    lines += ["", 
        "## MMMU Subject 路由",
        "",
    ]
    rows = []
    for router, prefix in [("TinyLLaVA", "tinyllava_mmmu_train720_router_on_"), ("Qwen3-VL-2B", "qwen3vl_mmmu_train720_router_on_")]:
        for dataset, suffix in [("MMMU holdout", "mmmu_holdout"), ("CMMMU dev", "cmmmu_dev"), ("MMMU-Pro test", "mmmu_pro_test"), ("MathVista testmini", "mathvista_testmini")]:
            summary = load(f"{prefix}{suffix}/summary.json")
            rows.append([router, dataset, summary["count"], pct(summary["subject_accuracy"]), pct(summary["routed_accuracy"]), pct(summary["best_single_accuracy"])])
    lines += table(["Router", "Dataset", "N", "Subject Acc", "Routed Acc", "Best Single"], rows)
    lines += ["", "## GAOKAO-MM Subject 路由", ""]
    rows = []
    for router, path in [("TinyLLaVA", "tinyllava-gaokao-mm-router-holdout/summary.json"), ("Qwen3-VL-2B", "qwen3vl-gaokao-mm-router-holdout/summary.json")]:
        summary = load(path)
        rows.append([router, summary["count"], pct(summary["subject_accuracy"]), pct(summary["routed_accuracy"]), pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Router", "N", "Subject Acc", "Routed Acc", "Best Single", "Oracle Any"], rows)
    lines += ["", "## 多模态 Query 路由留出", ""]
    rows = []
    for router, path in [
        ("TinyLLaVA / MMMU", "tiny_mmmu_query_on_mmmu_holdout/summary.json"),
        ("Qwen3-VL / MMMU", "qwen_mmmu_query_on_mmmu_holdout/summary.json"),
        ("TinyLLaVA / GAOKAO strict", "tinyllava_gaokao_query_strict_holdout/summary.json"),
        ("TinyLLaVA / GAOKAO all646", "tinyllava_gaokao_query_all646_holdout/summary.json"),
        ("Qwen3-VL / GAOKAO strict", "qwen3vl_gaokao_query_strict_holdout/summary.json"),
        ("Qwen3-VL / GAOKAO all646", "qwen3vl_gaokao_query_all646_holdout/summary.json"),
    ]:
        summary = load(path)
        rows.append([router, summary["count"], pct(summary["query_label_accuracy"]), summary["query_label_count"], pct(summary["routed_accuracy"]), pct(summary["best_single_accuracy"])])
    lines += table(["Experiment", "N", "Query Label Acc", "Labeled N", "Routed Acc", "Best Single"], rows)
    lines += ["", "## 统一语言+多模态 Query 路由", ""]
    rows = []
    for router, path in [
        ("BERT", "bert_unified_query_internal_validation/summary.json"),
        ("TinyLLaVA", "tinyllava_unified_query_internal_validation/summary.json"),
        ("Qwen3-VL-2B", "qwen3vl_unified_query_internal_validation/summary.json"),
    ]:
        summary = load(path)
        rows.append([router, summary["count"], pct(summary["routed_accuracy"]), pct(summary["by_modality"]["language"]["routed_accuracy"]), pct(summary["by_modality"]["multimodal"]["routed_accuracy"]), summary["cross_modality_route_count"]])
    lines += table(["Router", "Validation N", "Overall Routed", "Language Routed", "Multimodal Routed", "Cross-Modality Errors"], rows)
    lines += ["", "### MMMU 干净留出对齐", ""]
    rows = []
    for router, path in [
        ("BERT", "bert_unified_query_on_mmmu_holdout/summary.json"),
        ("TinyLLaVA", "tinyllava_unified_query_on_mmmu_holdout/summary.json"),
        ("Qwen3-VL-2B", "qwen3vl_unified_query_on_mmmu_holdout/summary.json"),
    ]:
        summary = load(path)
        rows.append([router, summary["count"], pct(summary["routed_accuracy"]), pct(summary["target_label_accuracy"] if "target_label_accuracy" in summary else summary["query_label_accuracy"]), summary.get("cross_modality_route_count", 0)])
    lines += table(["Router", "N", "Routed Acc", "Query Label Acc", "Cross-Modality Errors"], rows)
    lines += [
        "",
        "## BabyVision OOD 测试",
        "",
        "BabyVision 使用 388 条对齐样本。下表仅采用具有完整 Qwen3.5-9B local-text-judge 判分的 4 个同规模专家：InternVL3.5-2B、LFM2.5-VL-1.6B、Qwen3.5-2B、Gemma-4-E2B-it。SmolVLM2-Base 仅完成部分判分，SmolVLM2-Instruct 未判分，因此不纳入单专家与 oracle 基线；query 路由若选择缺失专家则按保守错误计并单独统计。",
        "",
        "### 单专家基线",
        "",
    ]
    rows = []
    for model in ["InternVL3_5-2B", "LFM2.5-VL-1.6B", "Qwen3.5-2B", "gemma-4-E2B-it"]:
        summary = load(f"babyvision_standardized_1p6b_2p2b/{model}/summary.json")
        rows.append([model, summary["count"], pct(summary["accuracy"]), summary["judge"]])
    lines += table(["Expert", "N", "Accuracy", "Judge"], rows)
    lines += ["", "### Subject 路由", ""]
    rows = []
    for experiment, path in [
        ("TinyLLaVA / MMMU mapping", "tinyllava_mmmu_subject_router_on_babyvision/summary.json"),
        ("Qwen3-VL / MMMU mapping", "qwen3vl_mmmu_subject_router_on_babyvision/summary.json"),
        ("TinyLLaVA / GAOKAO-MM mapping", "tinyllava_gaokao_mm_subject_router_on_babyvision/summary.json"),
        ("Qwen3-VL / GAOKAO-MM mapping", "qwen3vl_gaokao_mm_subject_router_on_babyvision/summary.json"),
    ]:
        summary = load(path)
        rows.append([experiment, pct(summary["routed_accuracy"]), pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Experiment", "Routed Acc", "Best Single", "Oracle Any"], rows)
    lines += ["", "### Query 路由", ""]
    rows = []
    for experiment, path in [
        ("TinyLLaVA / MMMU", "tinyllava_mmmu_query_router_on_babyvision/summary.json"),
        ("Qwen3-VL / MMMU", "qwen3vl_mmmu_query_router_on_babyvision/summary.json"),
        ("TinyLLaVA / GAOKAO strict", "tinyllava_gaokao_query_strict_on_babyvision/summary.json"),
        ("Qwen3-VL / GAOKAO strict", "qwen3vl_gaokao_query_strict_on_babyvision/summary.json"),
        ("TinyLLaVA / GAOKAO all646", "tinyllava_gaokao_query_all646_on_babyvision/summary.json"),
        ("Qwen3-VL / GAOKAO all646", "qwen3vl_gaokao_query_all646_on_babyvision/summary.json"),
    ]:
        summary = load(path)
        rows.append([experiment, pct(summary["routed_accuracy"]), summary["unavailable_expert_route_count"], pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Experiment", "Routed Acc", "Unavailable Routes", "Best Available Single", "Available Oracle"], rows)
    lines += ["", "### 统一语言+多模态路由", ""]
    rows = []
    for router, path in [
        ("BERT", "bert_unified_query_on_babyvision/summary.json"),
        ("TinyLLaVA", "tinyllava_unified_query_on_babyvision/summary.json"),
        ("Qwen3-VL-2B", "qwen3vl_unified_query_on_babyvision/summary.json"),
    ]:
        summary = load(path)
        rows.append([router, pct(summary["routed_accuracy"]), summary.get("cross_modality_route_count", 0), summary.get("unavailable_expert_route_count", 0), pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Router", "Routed Acc", "Cross-Modality", "Unavailable Routes", "Best Single", "Oracle Any"], rows)
    lines += [
        "",
        "## BabyVision：Qwen3-VL-2B-Instruct 替换 Qwen3.5-2B",
        "",
        "替换实验保持 1.6B–2.2B 同规模约束，专家池为 InternVL3.5-2B、LFM2.5-VL-1.6B、Qwen3-VL-2B-Instruct、Gemma-4-E2B-it。Qwen3-VL 在 388 题中有 374 个有效 Qwen3.5-9B local-text-judge 判分，剩余 14 题按保守错误计；因此其全量准确率为 14.43%（有效判分内为 14.97%）。",
        "",
        "### 替换后单专家基线",
        "",
    ]
    rows = []
    for model in ["InternVL3_5-2B", "LFM2.5-VL-1.6B", "Qwen3-VL-2B-Instruct", "gemma-4-E2B-it"]:
        summary = load(f"babyvision_standardized_qwen3vl_swap/{model}/summary.json")
        rows.append([model, summary["count"], pct(summary["accuracy"]), summary["judge"]])
    lines += table(["Expert", "N", "Accuracy", "Judge"], rows)
    lines += ["", "### 替换后 Subject 路由", ""]
    rows = []
    for experiment, path in [
        ("TinyLLaVA / MMMU mapping", "tinyllava_mmmu_subject_router_on_babyvision_qwen3vl_swap/summary.json"),
        ("Qwen3-VL / MMMU mapping", "qwen3vl_mmmu_subject_router_on_babyvision_qwen3vl_swap/summary.json"),
        ("TinyLLaVA / GAOKAO-MM mapping", "tinyllava_gaokao_mm_subject_router_on_babyvision_qwen3vl_swap/summary.json"),
        ("Qwen3-VL / GAOKAO-MM mapping", "qwen3vl_gaokao_mm_subject_router_on_babyvision_qwen3vl_swap/summary.json"),
    ]:
        summary = load(path)
        rows.append([experiment, pct(summary["routed_accuracy"]), pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Experiment", "Routed Acc", "Best Single", "Oracle Any"], rows)
    lines += ["", "### 替换后 Query 路由", ""]
    rows = []
    for experiment, path in [
        ("TinyLLaVA / MMMU", "tinyllava_mmmu_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("Qwen3-VL / MMMU", "qwen3vl_mmmu_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("TinyLLaVA / GAOKAO strict", "tinyllava_gaokao_strict_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("Qwen3-VL / GAOKAO strict", "qwen3vl_gaokao_strict_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("TinyLLaVA / GAOKAO all646", "tinyllava_gaokao_all646_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("Qwen3-VL / GAOKAO all646", "qwen3vl_gaokao_all646_query_on_babyvision_qwen3vl_swap/summary.json"),
    ]:
        summary = load(path)
        rows.append([experiment, pct(summary["routed_accuracy"]), summary["unavailable_expert_route_count"], pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Experiment", "Routed Acc", "Unavailable Routes", "Best Single", "Oracle Any"], rows)
    lines += ["", "### 替换后统一语言+多模态路由", ""]
    rows = []
    for router, path in [
        ("BERT", "bert_unified_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("TinyLLaVA", "tinyllava_unified_query_on_babyvision_qwen3vl_swap/summary.json"),
        ("Qwen3-VL-2B", "qwen3vl_unified_query_on_babyvision_qwen3vl_swap/summary.json"),
    ]:
        summary = load(path)
        rows.append([router, pct(summary["routed_accuracy"]), summary.get("cross_modality_route_count", 0), summary.get("unavailable_expert_route_count", 0), pct(summary["best_single_accuracy"]), pct(summary["oracle_any_expert_accuracy"])])
    lines += table(["Router", "Routed Acc", "Cross-Modality", "Unavailable Routes", "Best Single", "Oracle Any"], rows)
    lines += [
        "",
        "### 替换前后摘要",
        "",
        "- 最佳单专家由 Qwen3.5-2B 的 17.01% 降至 LFM2.5-VL-1.6B 的 14.69%；四专家 oracle 从 31.19% 降至 29.12%。",
        "- 原统一 Qwen3-VL 路由为 17.27%；替换并重训后为 12.63%，不再超过最佳单专家。",
        "- 新 MMMU query 标签共 524 条，统一训练集共 10,403 条；统一 BERT 在第 6 轮早停、最佳第 4 轮。",
        "- GAOKAO-MM 严格/全量 query 标签分别为 59/71 条，均保留 20% 内部验证并采用早停。",
    ]
    lines += [
        "",
        "## 主要结论",
        "",
        "- MMMU subject 分类可达到较高准确率，但当前同规模专家池的学科映射通常未超过 InternVL3.5-2B 单模型，说明瓶颈主要在专家互补性而非学科识别。",
        "- GAOKAO-MM 的专家整体准确率较低，但 subject 路由在留出集上将 routed accuracy 从最佳单专家 8.53% 提升到 10.08%。",
        "- MMMU query 路由明显塌缩到 InternVL3.5-2B，留出 routed accuracy 与最佳单模型相同；需要后续使用类别重加权或代价敏感损失改善专家利用率。",
        "- 统一路由能可靠区分模态，跨模态误路由极少；但在 MMMU 干净留出上仍接近选择最佳单模型，尚未充分利用 oracle 专家互补空间。",
        "- BabyVision 上统一 Qwen3-VL 路由达到 17.27%，略高于最佳可用单专家 Qwen3.5-2B 的 17.01%；可用 4 专家 oracle 为 31.19%，仍存在较大可挖掘空间。",
        "- 将 Qwen3.5-2B 替换为 Qwen3-VL-2B-Instruct 后，最佳单专家、oracle 与统一路由均下降，说明 Qwen3.5-2B 在 BabyVision 上提供了关键且不可直接替代的专家互补性。",
        "",
        "## 关键产物",
        "",
        "- 专家池配置：`bench_coe/configs/expert_pools.json`",
        "- MMMU 训练子集学科榜：`outputs/bench_coe/mmmu_train720_1p6b_2p2b/`",
        "- GAOKAO-MM 训练子集学科榜：`outputs/bench_coe/gaokao_mm_train517_1p6b_2p2b/`",
        "- 单数据集路由器：`outputs/bench_coe/router/`",
        "- 统一三路由器：`outputs/bench_coe/router/*unified-mmlu-test-mmmu-train-query/`",
        "- BabyVision 标准化专家缓存：`outputs/bench_coe/babyvision_standardized_1p6b_2p2b/`",
        "- Qwen3-VL 替换实验缓存与路由：`outputs/bench_coe/babyvision_standardized_qwen3vl_swap/`、`outputs/bench_coe/router/qwen3vl_swap/`",
        "- BabyVision 各路由预测：`outputs/bench_coe/*babyvision*/`",
        "- 全部逐题预测和 OOD 汇总：`outputs/bench_coe/`",
    ]
    (ROOT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
