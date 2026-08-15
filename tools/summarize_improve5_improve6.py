#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
REPORT_ROOT = OUTPUTS_ROOT / "bench_coe"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_root(path: Path, improvement: str) -> Path:
    candidates = [parent for parent in path.parents if improvement in parent.name.lower()]
    if not candidates:
        raise ValueError(f"Cannot identify experiment root for {path}")
    return candidates[0]


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for improvement in ("improve5", "improve6"):
        pattern = f"{improvement}_results.json"
        for path in sorted(OUTPUTS_ROOT.rglob(pattern)):
            if improvement not in str(path).lower():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                files.append({"path": str(path.relative_to(REPO_ROOT)), "status": "invalid_non_list", "rows": 0})
                continue
            root = experiment_root(path, improvement)
            case_dir = path.parent
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.update({
                    "improvement": improvement,
                    "experiment": str(root.relative_to(REPORT_ROOT)),
                    "case_directory": case_dir.name,
                    "result_file": str(path.relative_to(REPO_ROOT)),
                    "result_index": index,
                })
                rows.append(row)
            files.append({"path": str(path.relative_to(REPO_ROOT)), "status": "loaded", "rows": len(payload)})
    return rows, files


def json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def percentage(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{100 * value:.3f}%"


def compact_json(value: Any, limit: int = 180) -> str:
    text = json_text(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "improvement", "experiment", "case_id", "case_directory", "method",
        "target_accuracy", "gain_vs_best_single_target", "transfer_ratio",
        "best_single_target", "best_single_model_target", "instance_oracle_target",
        "oracle_gain", "target_samples", "source_samples", "models_used",
        "source_global_best", "source_global_best_accuracy", "routed_models", "metadata",
        "source_root", "target_root", "result_file", "result_index",
    ]
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    fields = [field for field in preferred if field in all_fields] + sorted(all_fields.difference(preferred))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_text(row.get(field, "")) if isinstance(row.get(field), (dict, list)) else row.get(field, "") for field in fields})


def experiment_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots: set[Path] = set()
    for path in REPORT_ROOT.iterdir():
        if path.is_dir() and ("improve5" in path.name.lower() or "improve6" in path.name.lower()):
            roots.add(path)
    combined = REPORT_ROOT / "mmlu_val_source_language_all_experiments_exclude_qwen35_deepseek_qwen3"
    for improvement in ("improve5", "improve6"):
        path = combined / improvement
        if path.is_dir():
            roots.add(path)
    audit = []
    for root in sorted(roots):
        relative = str(root.relative_to(REPORT_ROOT))
        group = [row for row in rows if row["experiment"] == relative]
        summary_path = root / "summary.json"
        summary_rows = None
        summary_status = "missing"
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_rows = len(payload) if isinstance(payload, list) else None
                summary_status = "loaded" if isinstance(payload, list) else "invalid_non_list"
            except Exception as error:
                summary_status = f"invalid:{type(error).__name__}"
        audit.append({
            "experiment": relative,
            "structured_result_rows": len(group),
            "structured_cases": len({row.get("case_id", row["case_directory"]) for row in group}),
            "top_summary_status": summary_status,
            "top_summary_rows": summary_rows,
            "summary_matches_details": summary_rows == len(group) if summary_rows is not None else None,
        })
    return audit


def build_report(rows: list[dict[str, Any]], files: list[dict[str, Any]], audits: list[dict[str, Any]]) -> str:
    by_improvement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_experiment: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_improvement[row["improvement"]].append(row)
        by_experiment[(row["improvement"], row["experiment"])].append(row)

    lines = [
        "# Improve5 与 Improve6 全部实验结果汇总",
        "",
        f"- 生成时间（UTC）：`{utc_now()}`",
        f"- 扫描目录：`{OUTPUTS_ROOT}`",
        f"- 结果文件数：**{len(files)}**",
        f"- 方法结果行数：**{len(rows)}**",
        "- 完整原始字段另存于同目录的 JSON 和 CSV；本报告中的 routed-model/metadata 为可读性进行截断，数值未截断。",
        "",
        "## 总览",
        "",
    ]

    overview = []
    for improvement in ("improve5", "improve6"):
        group = by_improvement.get(improvement, [])
        experiments = {row["experiment"] for row in group}
        cases = {(row["experiment"], row.get("case_id", row["case_directory"])) for row in group}
        methods = {row.get("method", "") for row in group}
        best_rows = []
        for case in cases:
            candidates = [row for row in group if (row["experiment"], row.get("case_id", row["case_directory"])) == case]
            numeric = [row for row in candidates if isinstance(row.get("target_accuracy"), (int, float))]
            if numeric:
                best_rows.append(max(numeric, key=lambda row: row["target_accuracy"]))
        positive = sum(float(row.get("gain_vs_best_single_target", 0)) > 0 for row in best_rows)
        overview.append([improvement, str(len(experiments)), str(len(cases)), str(len(methods)), str(len(group)), f"{positive}/{len(best_rows)}"])
    lines.extend(markdown_table(["改进", "实验套件", "Case", "不同方法", "全部结果行", "Case 最优超过单模型"], overview))

    lines.extend(["", "## 跨实验最佳结果", ""])
    best_body = []
    for improvement in ("improve5", "improve6"):
        group = by_improvement.get(improvement, [])
        cases = sorted({(row["experiment"], row.get("case_id", row["case_directory"])) for row in group})
        for experiment, case_id in cases:
            candidates = [row for row in group if row["experiment"] == experiment and row.get("case_id", row["case_directory"]) == case_id and isinstance(row.get("target_accuracy"), (int, float))]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: row["target_accuracy"])
            best_body.append([
                improvement, experiment, str(case_id), str(best.get("method", "")),
                percentage(best.get("target_accuracy")), percentage(best.get("best_single_target")),
                percentage(best.get("gain_vs_best_single_target")), percentage(best.get("instance_oracle_target")),
            ])
    lines.extend(markdown_table(["改进", "实验套件", "Case", "最优方法", "准确率", "最佳单模型", "增益", "实例 Oracle"], best_body))

    lines.extend(["", "## 方法胜出频次", ""])
    for improvement in ("improve5", "improve6"):
        wins: Counter[str] = Counter()
        group = by_improvement.get(improvement, [])
        cases = {(row["experiment"], row.get("case_id", row["case_directory"])) for row in group}
        for experiment, case_id in cases:
            candidates = [row for row in group if row["experiment"] == experiment and row.get("case_id", row["case_directory"]) == case_id and isinstance(row.get("target_accuracy"), (int, float))]
            if candidates:
                wins[str(max(candidates, key=lambda row: row["target_accuracy"]).get("method", "unknown"))] += 1
        lines.append(f"### {improvement}")
        lines.append("")
        lines.extend(markdown_table(["方法", "Case 胜出次数"], [[method, str(count)] for method, count in wins.most_common()]))
        lines.append("")

    lines.extend(["## 逐实验完整结果", ""])
    for improvement in ("improve5", "improve6"):
        lines.extend([f"# {improvement}", ""])
        for (kind, experiment), experiment_rows in sorted(by_experiment.items()):
            if kind != improvement:
                continue
            lines.extend([f"## `{experiment}`", ""])
            case_ids = sorted({str(row.get("case_id", row["case_directory"])) for row in experiment_rows})
            for case_id in case_ids:
                case_rows = [row for row in experiment_rows if str(row.get("case_id", row["case_directory"])) == case_id]
                first = case_rows[0]
                numeric = [row for row in case_rows if isinstance(row.get("target_accuracy"), (int, float))]
                best = max(numeric, key=lambda row: row["target_accuracy"]) if numeric else None
                lines.extend([
                    f"### `{case_id}`",
                    "",
                    f"- 来源：`{first.get('source_root', 'unknown')}`（{first.get('source_samples', 'unknown')} samples）",
                    f"- 目标：`{first.get('target_root', 'unknown')}`（{first.get('target_samples', 'unknown')} samples）",
                    f"- 最佳单模型：`{first.get('best_single_model_target', 'unknown')}`，{percentage(first.get('best_single_target'))}",
                    f"- 实例 Oracle：{percentage(first.get('instance_oracle_target'))}",
                    f"- 本组最优：`{best.get('method') if best else 'unknown'}`，{percentage(best.get('target_accuracy') if best else None)}，相对最佳单模型 {percentage(best.get('gain_vs_best_single_target') if best else None)}",
                    "",
                ])
                body = []
                for row in case_rows:
                    body.append([
                        str(row.get("method", "")), percentage(row.get("target_accuracy")),
                        percentage(row.get("gain_vs_best_single_target")), percentage(row.get("transfer_ratio")),
                        str(row.get("models_used", "")), compact_json(row.get("routed_models", "")),
                        compact_json(row.get("metadata", "")),
                    ])
                lines.extend(markdown_table(["方法", "目标准确率", "对最佳单模型增益", "迁移比", "模型数", "路由分布", "元数据"], body))
                lines.append("")

    loaded_paths = {item["path"] for item in files if item["status"] == "loaded"}
    lines.extend(["## 文件审计", "", f"- 成功解析：**{len(loaded_paths)}** 个方法结果文件。"])
    invalid = [item for item in files if item["status"] != "loaded"]
    lines.append(f"- 无效或无法解析：**{len(invalid)}** 个。")
    for item in invalid:
        lines.append(f"  - `{item['path']}`：{item['status']}")
    lines.extend(["", "### 已解析结果文件", ""])
    lines.extend(f"- `{item['path']}`（{item['rows']} 行）" for item in files if item["status"] == "loaded")
    lines.extend(["", "### 实验目录与顶层 Summary 一致性", ""])
    lines.extend(markdown_table(
        ["实验目录", "结构化结果行", "Case", "顶层 summary", "summary 行", "与明细一致"],
        [[
            item["experiment"], str(item["structured_result_rows"]), str(item["structured_cases"]),
            item["top_summary_status"], str(item["top_summary_rows"] if item["top_summary_rows"] is not None else "—"),
            str(item["summary_matches_details"] if item["summary_matches_details"] is not None else "—"),
        ] for item in audits],
    ))
    empty = [item for item in audits if item["structured_result_rows"] == 0]
    if empty:
        lines.extend(["", "### 无结构化结果的目录", ""])
        lines.extend(f"- `{item['experiment']}`" for item in empty)
    supplementary = REPORT_ROOT / "bench_coe_improve6_methods_examples_results_detailed.md"
    if supplementary.is_file():
        lines.extend(["", "### 补充说明文件", "", f"- `{supplementary.relative_to(REPO_ROOT)}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    rows, files = load_rows()
    audits = experiment_audit(rows)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = REPORT_ROOT / "improve5_improve6_all_experiment_results"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    report_path = stem.with_suffix(".md")
    json_path.write_text(json.dumps({"schema_version": "benchcoe_improve56_summary_v1", "generated_at": utc_now(), "rows": rows, "files": files, "experiment_audit": audits}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    report_path.write_text(build_report(rows, files, audits), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "files": len(files), "experiments_audited": len(audits), "json": str(json_path), "csv": str(csv_path), "markdown": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
