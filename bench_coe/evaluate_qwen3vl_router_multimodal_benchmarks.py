from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench_coe.evaluate_qwen3vl_router_benchmarks import (
    format_percent,
    route_rows_parallel,
    row_line,
    total_count,
)
from bench_coe.gaokao_utils import read_json, write_json
from bench_coe.run_multimodal_babyvision_models import (
    load_cmmmu_samples,
    load_mathvista_samples,
    load_mmmu_pro_samples,
    mmmu_pro_setting_slug,
)


SUBJECTS = ["Math", "Chinese", "Physics", "Chemistry", "Biology", "History", "Geography", "Politics"]
BENCHMARKS = ("cmmmu", "mathvista", "mmmu_pro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the GAOKAO-MM-trained Qwen3-VL router on multimodal benchmarks using cached VLM experts."
    )
    parser.add_argument("--benchmarks", default="all", help="Comma list: cmmmu,mathvista,mmmu_pro, or all.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bench_coe"))
    parser.add_argument("--single-root", type=Path, default=Path("outputs/multimodal_babyvision_models"))
    parser.add_argument("--gaokao-mm-leaderboard", type=Path, default=Path("outputs/gaokao_mm_babyvision_models/leaderboard.csv"))
    parser.add_argument(
        "--route-label-manifest",
        type=Path,
        default=Path("outputs/bench_coe/router/qwen3vl-2b-gaokao-mm-subject-lora/route_label_manifest.json"),
    )
    parser.add_argument("--router-model-path", type=Path, default=Path("models_v/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/bench_coe/router/qwen3vl-2b-gaokao-mm-subject-lora/adapter"))
    parser.add_argument("--transformers-src", type=Path, default=Path("transformers/src"))
    parser.add_argument("--local-deps", type=Path, default=Path(".codex_deps/qwen3vl"))
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--num-router-workers", type=int, default=4)
    parser.add_argument("--router-max-new-tokens", type=int, default=8)
    parser.add_argument("--router-temperature", type=float, default=0.0)
    parser.add_argument("--default-subject", default="Math")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--coe-name", default="GAOKAO-MM-Qwen3VL-Bench-CoE")

    parser.add_argument("--cmmmu-data-dir", type=Path, default=Path("data/CMMMU"))
    parser.add_argument("--cmmmu-split", default="val", choices=["dev", "val", "test"])
    parser.add_argument("--mathvista-data-dir", type=Path, default=Path("data/MathVista"))
    parser.add_argument("--mathvista-split", default="testmini", choices=["testmini", "test"])
    parser.add_argument("--mmmu-pro-data-dir", type=Path, default=Path("data/MMMU_Pro"))
    parser.add_argument(
        "--mmmu-pro-setting",
        default="standard (10 options)",
        choices=["standard (10 options)", "standard (4 options)", "vision"],
    )
    parser.add_argument("--mmmu-pro-split", default="test", choices=["test"])
    parser.add_argument("--image-layout", choices=["grid", "vertical"], default="grid")
    parser.add_argument("--max-tile-edge", type=int, default=980)
    parser.add_argument("--combined-image-bg", default="white")
    return parser.parse_args()


def parse_benchmarks(value: str) -> list[str]:
    if value == "all":
        return list(BENCHMARKS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(selected).difference(BENCHMARKS))
    if missing:
        raise ValueError(f"Unknown benchmark(s): {missing}; valid values are {list(BENCHMARKS)}")
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def loader_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir / "_sample_cache",
        cmmmu_data_dir=args.cmmmu_data_dir,
        cmmmu_split=args.cmmmu_split,
        mathvista_data_dir=args.mathvista_data_dir,
        mathvista_split=args.mathvista_split,
        mmmu_pro_data_dir=args.mmmu_pro_data_dir,
        mmmu_pro_setting=args.mmmu_pro_setting,
        mmmu_pro_split=args.mmmu_pro_split,
        max_examples_per_benchmark=args.max_examples,
        limit_per_category=None,
        image_layout=args.image_layout,
        max_tile_edge=args.max_tile_edge,
        combined_image_bg=args.combined_image_bg,
    )


def load_samples(args: argparse.Namespace, benchmark: str, output_dir: Path) -> list[dict[str, Any]]:
    local_args = loader_args(args, output_dir)
    if benchmark == "cmmmu":
        samples = load_cmmmu_samples(local_args, "_router")
    elif benchmark == "mathvista":
        samples = load_mathvista_samples(local_args, "_router")
    elif benchmark == "mmmu_pro":
        samples = load_mmmu_pro_samples(local_args, "_router")
    else:
        raise ValueError(benchmark)
    rows = []
    for sample in samples:
        row = dict(sample)
        row["question_id"] = str(sample["id"])
        row["route_text"] = str(sample["prompt"])
        image_path = sample.get("image_path")
        row["route_image_paths"] = [str(image_path)] if image_path else []
        row.pop("raw", None)
        rows.append(row)
    return rows


def prediction_dir(args: argparse.Namespace, benchmark: str, model_name: str) -> Path:
    if benchmark == "cmmmu":
        return args.single_root / "cmmmu" / args.cmmmu_split / model_name
    if benchmark == "mathvista":
        return args.single_root / "mathvista" / args.mathvista_split / model_name
    if benchmark == "mmmu_pro":
        return args.single_root / "mmmu_pro" / mmmu_pro_setting_slug(args.mmmu_pro_setting) / args.mmmu_pro_split / model_name
    raise ValueError(benchmark)


def benchmark_output_dir(args: argparse.Namespace, benchmark: str) -> Path:
    if benchmark == "cmmmu":
        return args.output_root / "cmmmu_qwen3vl_gaokao_mm_router_front4"
    if benchmark == "mathvista":
        return args.output_root / "mathvista_qwen3vl_gaokao_mm_router_front4"
    if benchmark == "mmmu_pro":
        return args.output_root / f"mmmu_pro_{mmmu_pro_setting_slug(args.mmmu_pro_setting)}_qwen3vl_gaokao_mm_router_front4"
    raise ValueError(benchmark)


def completed_single_summaries(args: argparse.Namespace, benchmark: str) -> list[dict[str, Any]]:
    if benchmark == "cmmmu":
        root = args.single_root / "cmmmu" / args.cmmmu_split
    elif benchmark == "mathvista":
        root = args.single_root / "mathvista" / args.mathvista_split
    elif benchmark == "mmmu_pro":
        root = args.single_root / "mmmu_pro" / mmmu_pro_setting_slug(args.mmmu_pro_setting) / args.mmmu_pro_split
    else:
        raise ValueError(benchmark)

    rows = []
    for path in sorted(root.glob("*/summary.json")):
        summary = read_json(path)
        if summary.get("status") == "completed" and summary.get("accuracy") is not None:
            rows.append(summary)
    rows.sort(key=lambda row: (-float(row["accuracy"]), str(row["model"])))
    return rows


def numeric(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def subject_to_expert_from_gaokao(
    args: argparse.Namespace,
    available_models: set[str],
    single_rows: list[dict[str, Any]],
) -> dict[str, str]:
    best_overall = str(single_rows[0]["model"]) if single_rows else next(iter(sorted(available_models)))
    mapping = {subject: best_overall for subject in SUBJECTS}
    if not args.gaokao_mm_leaderboard.exists():
        return mapping

    with args.gaokao_mm_leaderboard.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for subject in SUBJECTS:
        best_model = None
        best_acc = None
        for row in rows:
            model = str(row.get("model", ""))
            if model not in available_models:
                continue
            acc = numeric(row.get(subject))
            if acc is None:
                continue
            if best_acc is None or acc > best_acc:
                best_model = model
                best_acc = acc
        if best_model is not None:
            mapping[subject] = best_model
    return mapping


def effective_manifest(base_manifest: dict[str, Any], subject_to_model: dict[str, str], model_names: list[str]) -> dict[str, Any]:
    manifest = dict(base_manifest)
    manifest["subject_to_model"] = subject_to_model
    manifest["subject_to_model_index"] = {subject: model_names.index(model) for subject, model in subject_to_model.items()}
    manifest["route_label_to_model"] = {
        str(label): subject_to_model[subject]
        for subject, label in manifest["subject_to_route_label"].items()
    }
    manifest["route_label_to_model_index"] = {
        str(label): manifest["subject_to_model_index"][subject]
        for subject, label in manifest["subject_to_route_label"].items()
    }
    manifest["expert_mapping_source"] = str(Path("outputs/gaokao_mm_babyvision_models/leaderboard.csv"))
    return manifest


def load_prediction_cache(args: argparse.Namespace, benchmark: str, model_name: str) -> dict[str, dict[str, Any]]:
    path = prediction_dir(args, benchmark, model_name) / "predictions.json"
    if path.exists():
        rows = read_json(path)
    else:
        path = prediction_dir(args, benchmark, model_name) / "predictions.jsonl"
        rows = read_jsonl(path)
    return {str(row["id"]): row for row in rows}


def combine_rows(args: argparse.Namespace, benchmark: str, routed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    caches: dict[str, dict[str, dict[str, Any]]] = {}
    final_rows = []
    for row in routed_rows:
        model_name = row["routed_model"]
        if model_name not in caches:
            caches[model_name] = load_prediction_cache(args, benchmark, model_name)
        cached = caches[model_name].get(str(row["id"]))
        if cached is None:
            raise KeyError(f"Missing cached prediction: benchmark={benchmark}, model={model_name}, id={row['id']}")
        item = dict(row)
        item["prediction"] = cached.get("prediction")
        item["response"] = cached.get("response", "")
        item["is_correct"] = cached.get("is_correct")
        item["expert_prediction_file"] = str(prediction_dir(args, benchmark, model_name) / "predictions.json")
        final_rows.append(item)
    return final_rows


def stats_dict() -> dict[str, float]:
    return {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}


def finalize(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for item in stats.values():
        denom = item["correct"] + item["wrong"]
        item["accuracy"] = item["correct"] / denom if denom else 0.0
    return dict(sorted(stats.items()))


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[str, Any]:
    correct = 0.0
    wrong = 0.0
    groups = {key: defaultdict(stats_dict) for key in group_keys}
    for row in rows:
        if row.get("is_correct") is None:
            continue
        is_correct = bool(row.get("is_correct"))
        correct += 1 if is_correct else 0
        wrong += 0 if is_correct else 1
        for key in group_keys:
            value = row.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                stats = groups[key][str(item)]
                if is_correct:
                    stats["correct"] += 1
                else:
                    stats["wrong"] += 1
    total = correct + wrong
    summary: dict[str, Any] = {
        "total": {"correct": correct, "wrong": wrong, "accuracy": correct / total if total else 0.0},
        "examples": int(total),
    }
    for key, stats in groups.items():
        summary[key] = finalize(stats)
    return summary


def group_key_for_render(benchmark: str) -> str:
    if benchmark == "cmmmu":
        return "category"
    if benchmark == "mathvista":
        return "task"
    if benchmark == "mmmu_pro":
        return "domain"
    raise ValueError(benchmark)


def benchmark_title(args: argparse.Namespace, benchmark: str) -> str:
    if benchmark == "cmmmu":
        return f"CMMMU ({args.cmmmu_split})"
    if benchmark == "mathvista":
        return f"MathVista ({args.mathvista_split})"
    if benchmark == "mmmu_pro":
        return f"MMMU_Pro ({args.mmmu_pro_setting}, {args.mmmu_pro_split})"
    raise ValueError(benchmark)


def txt_filename(benchmark: str) -> str:
    return f"Bench_Harness_Result_qwen3vl_gaokao_mm_router_{benchmark}.txt"


def render_txt(
    args: argparse.Namespace,
    benchmark: str,
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    single_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    group_key = group_key_for_render(benchmark)
    columns = sorted(summary[group_key]) + ["Average"]
    name_width = 34
    col_width = 18
    best = single_rows[0] if single_rows else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(summary["total"]["accuracy"])
    counts = [str(total_count(summary[group_key][col])) for col in columns[:-1]] + [str(int(summary["examples"]))]
    lines = [
        "=" * 100,
        f"Bench-Harness: GAOKAO-MM Qwen3-VL router -> {benchmark_title(args, benchmark)}",
        "=" * 100,
        "| Routing Mode: qwen3vl_gaokao_mm_subject_multimodal",
        f"| Benchmark: {benchmark_title(args, benchmark)}",
        f"| Samples: {int(summary['examples'])}",
        f"| Single model source: {args.single_root}",
        f"| Expert mapping source: {args.gaokao_mm_leaderboard}",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for item in single_rows:
        model = str(item["model"])
        prefix = "* " if model == best_model else "  "
        group_stats = item.get(f"by_{group_key}", {})
        values = [
            format_percent(group_stats.get(col, {}).get("accuracy")) if col in group_stats else "N/A"
            for col in columns[:-1]
        ]
        values.append(format_percent(float(item["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model, values))
    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(summary[group_key][col]["accuracy"]) for col in columns[:-1]]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]))
    lines.append("")
    lines.append("Subject -> expert mapping:")
    for subject in SUBJECTS:
        lines.append(f"- {subject}: {manifest['subject_to_model'].get(subject)}")
    lines.append("")
    lines.append("Routed models:")
    for model, count in Counter(row["routed_model"] for row in rows).most_common():
        lines.append(f"- {model}: {count}")
    lines.append("Routed GAOKAO-MM subjects:")
    for subject, count in Counter(row["routed_subject"] for row in rows).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {output_dir / 'test_predictions.json'}")
    write_text(output_dir / txt_filename(benchmark), lines)


def evaluate_one(args: argparse.Namespace, benchmark: str, base_manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = benchmark_output_dir(args, benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    single_rows = completed_single_summaries(args, benchmark)
    if not single_rows:
        raise RuntimeError(f"No completed single-model caches found for {benchmark} under {args.single_root}")
    model_names = [str(row["model"]) for row in single_rows]
    subject_to_model = subject_to_expert_from_gaokao(args, set(model_names), single_rows)
    manifest = effective_manifest(base_manifest, subject_to_model, model_names)
    write_json(output_dir / "effective_route_label_manifest.json", manifest)

    if args.resume and (output_dir / "test_predictions.json").exists() and (output_dir / "test_summary.json").exists():
        rows = read_json(output_dir / "test_predictions.json")
        summary = read_json(output_dir / "test_summary.json")
        render_txt(args, benchmark, output_dir, rows, summary, single_rows, manifest)
        return summary

    samples = load_samples(args, benchmark, output_dir)
    routed = route_rows_parallel(args, samples, manifest, output_dir, benchmark)
    final_rows = combine_rows(args, benchmark, routed)
    group_keys = {
        "cmmmu": ["category", "subcategory", "type", "difficulty", "routed_model", "routed_subject"],
        "mathvista": ["category", "task", "context", "grade", "question_type", "answer_type", "skills", "routed_model", "routed_subject"],
        "mmmu_pro": ["domain", "subject", "difficulty", "img_type", "routed_model", "routed_subject"],
    }[benchmark]
    summary = summarize(final_rows, group_keys)
    summary["benchmark"] = benchmark
    summary["prediction_file"] = str(output_dir / "test_predictions.json")
    write_json(output_dir / "test_predictions.json", final_rows)
    write_json(output_dir / "test_summary.json", summary)
    render_txt(args, benchmark, output_dir, final_rows, summary, single_rows, manifest)
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.adapter_path.exists():
        raise FileNotFoundError(args.adapter_path)
    benchmarks = parse_benchmarks(args.benchmarks)
    base_manifest = read_json(args.route_label_manifest)
    start = time.time()
    summaries = {benchmark: evaluate_one(args, benchmark, base_manifest) for benchmark in benchmarks}
    write_json(
        args.output_root / "qwen3vl_gaokao_mm_router_multimodal_front4_run_manifest.json",
        {
            "benchmarks": benchmarks,
            "router_model_path": str(args.router_model_path),
            "adapter_path": str(args.adapter_path),
            "route_label_manifest": str(args.route_label_manifest),
            "single_root": str(args.single_root),
            "gaokao_mm_leaderboard": str(args.gaokao_mm_leaderboard),
            "elapsed_seconds": time.time() - start,
            "summaries": summaries,
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
