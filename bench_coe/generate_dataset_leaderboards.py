from __future__ import annotations

import csv
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("outputs/bench_coe/leaderboards")
GAOKAO_ROOT = Path("GAOKAO-Bench-2010-2022/Data")
MM_ROOT = Path("outputs/multimodal_babyvision_models")
GAOKAO_MM_ROOT = Path("outputs/gaokao_mm_babyvision_models")


GAOKAO_SUBJECT_TO_TYPES: dict[str, tuple[str, ...]] = {
    "English": (
        "2010-2013_English_MCQs",
        "2010-2022_English_Fill_in_Blanks",
        "2012-2022_English_Cloze_Test",
        "2010-2022_English_Reading_Comp",
    ),
    "Math": ("2010-2022_Math_I_MCQs", "2010-2022_Math_II_MCQs"),
    "Chinese": ("2010-2022_Chinese_Modern_Lit", "2010-2022_Chinese_Lang_and_Usage_MCQs"),
    "Physics": ("2010-2022_Physics_MCQs",),
    "Chemistry": ("2010-2022_Chemistry_MCQs",),
    "Biology": ("2010-2022_Biology_MCQs",),
    "History": ("2010-2022_History_MCQs",),
    "Geography": ("2010-2022_Geography_MCQs",),
    "Politics": ("2010-2022_Political_Science_MCQs",),
}
TYPE_TO_GAOKAO_SUBJECT = {
    task: subject for subject, tasks in GAOKAO_SUBJECT_TO_TYPES.items() for task in tasks
}

GAOKAO_MM_SUBJECTS = (
    "Math",
    "Chinese",
    "Physics",
    "Chemistry",
    "Biology",
    "History",
    "Geography",
    "Politics",
)


@dataclass(frozen=True)
class LeaderboardSpec:
    dataset_id: str
    title: str
    root: Path
    group_key: str
    output_stem: str


MM_SPECS = (
    LeaderboardSpec(
        dataset_id="mmmu_pro",
        title="MMMU-Pro standard_10_options test",
        root=MM_ROOT / "mmmu_pro" / "standard_10_options" / "test",
        group_key="by_subject",
        output_stem="mmmu_pro_standard_10_options_test_leaderboard",
    ),
    LeaderboardSpec(
        dataset_id="cmmmu",
        title="CMMMU val",
        root=MM_ROOT / "cmmmu" / "val",
        group_key="by_subcategory",
        output_stem="cmmmu_val_leaderboard",
    ),
    LeaderboardSpec(
        dataset_id="mathvista",
        title="MathVista testmini",
        root=MM_ROOT / "mathvista" / "testmini",
        group_key="by_task",
        output_stem="mathvista_testmini_leaderboard",
    ),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "-")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    header = "".join(
        f"<th data-col='{idx}'>{html.escape(col)} <span class='sort'>▲</span></th>"
        for idx, col in enumerate(columns)
    )
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '-')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #262626; }}
    .wrap {{ padding: 16px 24px; }}
    h1 {{ font-size: 20px; margin: 0 0 12px; }}
    .meta {{ color: #666; margin: 0 0 14px; font-size: 13px; }}
    .table-wrap {{ overflow: auto; border: 1px solid #ddd; max-height: calc(100vh - 96px); }}
    table {{ border-collapse: collapse; min-width: 1100px; width: max-content; background: white; }}
    th, td {{ border-right: 1px solid #e2e2e2; padding: 10px 9px; white-space: nowrap; font-size: 14px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #fafafa; z-index: 2; cursor: pointer; }}
    tr:nth-child(even) td {{ background: #f7f7f7; }}
    th:first-child, td:first-child {{ position: sticky; left: 0; z-index: 3; background: inherit; min-width: 280px; }}
    th:first-child {{ background: #fafafa; z-index: 4; }}
    .sort {{ color: #aaa; font-size: 12px; float: right; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <p class="meta">Click headers to sort. Values are decimal accuracies/scoring rates.</p>
  <div class="table-wrap">
    <table id="leaderboard">
      <thead><tr>{header}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>
</div>
<script>
const table = document.getElementById('leaderboard');
let lastCol = -1;
let asc = true;
for (const th of table.querySelectorAll('th')) {{
  th.addEventListener('click', () => {{
    const col = Number(th.dataset.col);
    asc = lastCol === col ? !asc : true;
    lastCol = col;
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((a, b) => {{
      const av = a.cells[col].textContent.trim();
      const bv = b.cells[col].textContent.trim();
      const an = Number(av);
      const bn = Number(bv);
      let cmp;
      if (!Number.isNaN(an) && !Number.isNaN(bn)) cmp = an - bn;
      else cmp = av.localeCompare(bv);
      return asc ? cmp : -cmp;
    }});
    table.tBodies[0].append(...rows);
  }});
}}
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def format_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def model_size_b(model: str) -> str:
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])", model)
    if matches:
        return matches[-1]
    matches = re.findall(r"(\d+(?:\.\d+)?)B", model)
    if matches:
        return matches[-1]
    return "unknown"


def data_source(model: str) -> str:
    if model in {"Bench-Harness", "GAOKAO-Bert-Bench-CoE", "GAOKAO-MM-Qwen3VL-Bench-CoE"}:
        return "Bench-CoE"
    return "Local Cache"


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, str]:
        try:
            overall = float(row.get("Overall", "-1"))
        except Exception:
            overall = -1.0
        return (-overall, str(row.get("Models", "")))

    return sorted(rows, key=key)


def materialize_leaderboard(output_stem: str, title: str, rows: list[dict[str, Any]], metric_columns: list[str]) -> None:
    rows = sorted_rows(rows)
    columns = ["Models", "Model Size(B)", "Data Source", "Overall"] + metric_columns
    write_csv(OUTPUT_DIR / f"{output_stem}.csv", rows, columns)
    write_markdown(OUTPUT_DIR / f"{output_stem}.md", rows, columns)
    write_html(OUTPUT_DIR / f"{output_stem}.html", title, rows, columns)


def load_mm_leaderboard(spec: LeaderboardSpec) -> tuple[list[dict[str, Any]], list[str]]:
    group_names: set[str] = set()
    raw_rows: list[tuple[str, dict[str, Any]]] = []
    for summary_path in sorted(spec.root.glob("*/summary.json")):
        summary = read_json(summary_path)
        if summary.get("status") != "completed":
            continue
        model = str(summary.get("model") or summary_path.parent.name)
        groups = summary.get(spec.group_key, {})
        if not isinstance(groups, dict):
            groups = {}
        group_names.update(str(group) for group in groups)
        raw_rows.append((model, summary))
    columns = sorted(group_names)
    rows: list[dict[str, Any]] = []
    for model, summary in raw_rows:
        groups = summary.get(spec.group_key, {})
        row = {
            "Models": model,
            "Model Size(B)": model_size_b(model),
            "Data Source": data_source(model),
            "Overall": format_score(float(summary["accuracy"])) if summary.get("accuracy") is not None else "-",
        }
        for group in columns:
            value = groups.get(group, {}).get("accuracy") if isinstance(groups, dict) else None
            row[group] = format_score(float(value)) if value is not None else "-"
        rows.append(row)
    return rows, columns


def normalize_answer_list(value: Any, answer_len: int) -> list[str]:
    if isinstance(value, list):
        answer = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        answer = []
    else:
        answer = [str(value).strip()]
    if len(answer) != answer_len:
        return ["Z"] * answer_len
    return answer


def score_gaokao_item(keyword: str, item: dict[str, Any]) -> tuple[float, float, float]:
    standard = [str(answer).strip() for answer in item.get("standard_answer", [])]
    if not standard:
        return 0.0, 0.0, 0.0
    score = float(item.get("score", 0.0))
    model_answer = normalize_answer_list(item.get("model_answer"), len(standard))
    total_score = len(standard) * score
    correct_score = 0.0
    if keyword == "2010-2022_Physics_MCQs":
        for pred, gold in zip(model_answer, standard):
            if pred == gold:
                correct_score += 6.0
            else:
                is_error = any(char not in gold for char in pred)
                correct_score += 0.0 if is_error else 3.0
    else:
        for pred, gold in zip(model_answer, standard):
            if pred == gold:
                correct_score += score
    return total_score, correct_score, float(len(standard))


def load_gaokao_leaderboard() -> tuple[list[dict[str, Any]], list[str]]:
    subject_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    overall_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in sorted(GAOKAO_ROOT.glob("*.json")):
        if path.name == "correction_score.json":
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict) or "example" not in payload:
            continue
        model = str(payload.get("model_name") or "").strip()
        keyword = str(payload.get("keyword") or payload.get("keywords") or "").strip()
        subject = TYPE_TO_GAOKAO_SUBJECT.get(keyword)
        if not model or subject is None:
            continue
        for item in payload.get("example", []):
            total_score, correct_score, question_num = score_gaokao_item(keyword, item)
            subject_totals[model][f"{subject}:total"] += total_score
            subject_totals[model][f"{subject}:correct"] += correct_score
            subject_totals[model][f"{subject}:questions"] += question_num
            overall_totals[model]["total"] += total_score
            overall_totals[model]["correct"] += correct_score
            overall_totals[model]["questions"] += question_num

    columns = list(GAOKAO_SUBJECT_TO_TYPES)
    rows: list[dict[str, Any]] = []
    for model in sorted(overall_totals):
        total = overall_totals[model]["total"]
        correct = overall_totals[model]["correct"]
        row = {
            "Models": model,
            "Model Size(B)": model_size_b(model),
            "Data Source": data_source(model),
            "Overall": format_score(correct / total if total else None),
        }
        for subject in columns:
            subject_total = subject_totals[model][f"{subject}:total"]
            subject_correct = subject_totals[model][f"{subject}:correct"]
            row[subject] = format_score(subject_correct / subject_total if subject_total else None)
        rows.append(row)
    return rows, columns


def load_gaokao_mm_leaderboard(exclude_models: set[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    exclude_models = exclude_models or set()
    rows: list[dict[str, Any]] = []
    for score_path in sorted(GAOKAO_MM_ROOT.glob("*/correction_score.json")):
        model = score_path.parent.name
        if model in exclude_models:
            continue
        try:
            score = read_json(score_path)
        except Exception:
            continue
        row = {
            "Models": model,
            "Model Size(B)": model_size_b(model),
            "Data Source": data_source(model),
            "Overall": format_score(float(score["accuracy"])) if score.get("accuracy") is not None else "-",
        }
        subjects = score.get("subject", {})
        for subject in GAOKAO_MM_SUBJECTS:
            value = subjects.get(subject, {}).get("accuracy") if isinstance(subjects, dict) else None
            row[subject] = format_score(float(value)) if value is not None else "-"
        rows.append(row)
    return rows, list(GAOKAO_MM_SUBJECTS)


def write_index(generated: list[tuple[str, str]]) -> None:
    links = "\n".join(
        f"<li><a href='{html.escape(stem)}.html'>{html.escape(title)}</a> "
        f"(<a href='{html.escape(stem)}.csv'>csv</a>, <a href='{html.escape(stem)}.md'>md</a>)</li>"
        for stem, title in generated
    )
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Dataset Leaderboards</title></head>
<body>
<h1>Dataset Leaderboards</h1>
<ul>
{links}
</ul>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, str]] = []

    for spec in MM_SPECS:
        rows, columns = load_mm_leaderboard(spec)
        materialize_leaderboard(spec.output_stem, spec.title, rows, columns)
        generated.append((spec.output_stem, spec.title))

    gaokao_rows, gaokao_columns = load_gaokao_leaderboard()
    gaokao_stem = "gaokao_bench_2010_2022_leaderboard"
    materialize_leaderboard(gaokao_stem, "GAOKAO-Bench-2010-2022", gaokao_rows, gaokao_columns)
    generated.append((gaokao_stem, "GAOKAO-Bench-2010-2022"))

    gaokao_mm_rows, gaokao_mm_columns = load_gaokao_mm_leaderboard()
    gaokao_mm_stem = "gaokao_mm_leaderboard"
    materialize_leaderboard(gaokao_mm_stem, "GAOKAO-MM", gaokao_mm_rows, gaokao_mm_columns)
    generated.append((gaokao_mm_stem, "GAOKAO-MM"))

    gaokao_mm_no_4b_rows, gaokao_mm_no_4b_columns = load_gaokao_mm_leaderboard(
        exclude_models={"Qwen3-VL-4B-Instruct"}
    )
    gaokao_mm_no_4b_stem = "gaokao_mm_exclude_qwen3vl4b_leaderboard"
    materialize_leaderboard(
        gaokao_mm_no_4b_stem,
        "GAOKAO-MM (exclude Qwen3-VL-4B-Instruct)",
        gaokao_mm_no_4b_rows,
        gaokao_mm_no_4b_columns,
    )
    generated.append((gaokao_mm_no_4b_stem, "GAOKAO-MM (exclude Qwen3-VL-4B-Instruct)"))

    write_index(generated)
    manifest = {
        "output_dir": str(OUTPUT_DIR),
        "leaderboards": [
            {"title": title, "html": f"{stem}.html", "csv": f"{stem}.csv", "markdown": f"{stem}.md"}
            for stem, title in generated
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
