from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from bench_coe.gaokao_utils import read_json, write_json
from bench_coe.mmlu_utils import MMLU_CATEGORY_ORDER, discover_mmlu_summaries, read_mmlu_summary
from bench_coe.run_official_model_benchmarks import (
    load_bbh_rows,
    load_gpqa_rows,
    load_mmstar_rows,
    run_mmstar_eval_on_dataframe,
    summarize_mmstar_fallback,
)


CHOICES = "ABCDEFGHIJ"
SUBJECT_ALIASES = {
    "math": "Math",
    "mathematics": "Math",
    "数学": "Math",
    "chinese": "Chinese",
    "language": "Chinese",
    "语文": "Chinese",
    "中文": "Chinese",
    "physics": "Physics",
    "物理": "Physics",
    "chemistry": "Chemistry",
    "化学": "Chemistry",
    "biology": "Biology",
    "生物": "Biology",
    "history": "History",
    "历史": "History",
    "geography": "Geography",
    "地理": "Geography",
    "politics": "Politics",
    "political science": "Politics",
    "政治": "Politics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GAOKAO-MM-trained Qwen3-VL subject router by combining cached expert predictions."
    )
    parser.add_argument(
        "--router-model-path",
        type=Path,
        default=Path("models_v/Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/bench_coe/router/qwen3vl-2b-gaokao-mm-subject-lora/adapter"),
    )
    parser.add_argument(
        "--route-label-manifest",
        type=Path,
        default=Path("outputs/bench_coe/router/qwen3vl-2b-gaokao-mm-subject-lora/route_label_manifest.json"),
    )
    parser.add_argument("--transformers-src", type=Path, default=Path("transformers/src"))
    parser.add_argument("--local-deps", type=Path, default=Path(".codex_deps/qwen3vl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bench_coe"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu", "bbh", "gpqa", "mmstar"],
        choices=["mmlu", "bbh", "gpqa", "mmstar"],
    )
    parser.add_argument("--mmlu-data-dir", type=Path, default=Path("MMLU-Pro/data"))
    parser.add_argument("--mmlu-results-dir", type=Path, default=Path("MMLU-Pro/results"))
    parser.add_argument("--mmlu-split", default="test", choices=["test"])
    parser.add_argument("--single-root", type=Path, default=Path("outputs/model_benchmarks/official_code_local_models"))
    parser.add_argument("--bbh-data-dir", type=Path, default=Path("BIG-Bench-Hard/bbh"))
    parser.add_argument("--bbh-tasks", default="all")
    parser.add_argument("--gpqa-data-dir", type=Path, default=Path("data/gpqa"))
    parser.add_argument("--gpqa-configs", default="diamond")
    parser.add_argument("--gpqa-epochs", type=int, default=4)
    parser.add_argument("--shuffle-gpqa-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mmstar-tsv", type=Path, default=Path("data/MMStar/MMStar.tsv"))
    parser.add_argument("--mmstar-eval-dir", type=Path, default=Path("MMStar/eval"))
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--num-router-workers", type=int, default=4)
    parser.add_argument("--router-max-new-tokens", type=int, default=8)
    parser.add_argument("--router-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coe-name", default="GAOKAO-MM-Qwen3VL-Bench-CoE")
    parser.add_argument("--default-subject", default="Math")
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def row_line(name_width: int, col_width: int, name: str, values: list[str]) -> str:
    cells = [name.ljust(name_width)]
    cells.extend(value.ljust(col_width) for value in values)
    return "| " + " | ".join(cells) + " |"


def total_count(stats: dict[str, Any]) -> int:
    return int(float(stats.get("correct", 0.0)) + float(stats.get("wrong", 0.0)))


def stats_dict() -> dict[str, float]:
    return {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}


def finalize_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for item in stats.values():
        denom = item["correct"] + item["wrong"]
        item["accuracy"] = item["correct"] / denom if denom else 0.0
    return dict(sorted(stats.items()))


def normalize_subject_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text.lower()).strip()


def parse_subject(response: str, allowed_subjects: set[str], default_subject: str) -> tuple[str, bool]:
    normalized = normalize_subject_text(response)
    compact = normalized.replace(" ", "")
    for subject in sorted(allowed_subjects, key=len, reverse=True):
        if normalize_subject_text(subject) in normalized.split():
            return subject, True
    for alias, subject in SUBJECT_ALIASES.items():
        if subject in allowed_subjects and (alias in normalized or alias.replace(" ", "") in compact):
            return subject, True
    return default_subject, False


def resolve_subject(label_manifest: dict[str, Any], subject: str) -> tuple[int, str, int]:
    subject_to_route_label = label_manifest["subject_to_route_label"]
    if subject not in subject_to_route_label:
        raise KeyError(f"Subject {subject!r} is not in route manifest.")
    route_label = int(subject_to_route_label[subject])
    model_name = label_manifest["subject_to_model"][subject]
    model_index = int(label_manifest.get("subject_to_model_index", {}).get(subject, -1))
    return route_label, model_name, model_index


def route_prompt(route_text: str, label_manifest: dict[str, Any]) -> str:
    subjects = [
        label_manifest["route_label_to_subject"][str(idx)]
        for idx in range(int(label_manifest["num_route_labels"]))
    ]
    return (
        "Classify this problem into exactly one GAOKAO subject.\n"
        f"Allowed subjects: {', '.join(subjects)}.\n"
        "Return only the English subject label.\n\n"
        f"Problem:\n{route_text.strip()}"
    )


def prepend_existing_path(path: Path) -> None:
    if path.exists():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def ensure_router_pythonpath(args: argparse.Namespace) -> None:
    prepend_existing_path(args.local_deps)
    prepend_existing_path(args.transformers_src)


class Qwen3VLRouter:
    def __init__(self, args: argparse.Namespace):
        ensure_router_pythonpath(args)
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(args.router_model_path.resolve()),
            trust_remote_code=True,
        )
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.router_model_path.resolve()),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        )
        self.model = PeftModel.from_pretrained(base_model, str(args.adapter_path.resolve()))
        self.model.eval()
        self.max_new_tokens = args.router_max_new_tokens
        self.temperature = args.router_temperature

    def generate(self, prompt: str, image_paths: list[str] | None = None) -> str:
        content: list[dict[str, Any]] = []
        for image_path in image_paths or []:
            path = Path(str(image_path))
            if path.exists():
                image_value = str(path.resolve())
            else:
                image_value = str(image_path)
            content.append({"type": "image", "image": image_value})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def route_rows_local(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    label_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_subjects = set(label_manifest["subject_to_route_label"])
    router = Qwen3VLRouter(args)
    routed: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="routing"):
        response = router.generate(
            route_prompt(row["route_text"], label_manifest),
            row.get("route_image_paths") or None,
        )
        subject, parsed = parse_subject(response, allowed_subjects, args.default_subject)
        route_label, model_name, model_index = resolve_subject(label_manifest, subject)
        item = dict(row)
        item["route_label"] = route_label
        item["route_response"] = response
        item["route_parse_ok"] = parsed
        item["routed_subject"] = subject
        item["routed_model"] = model_name
        item["routed_model_index"] = model_index
        routed.append(item)
    return routed


def split_evenly(rows: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    chunks = [[] for _ in range(max(1, parts))]
    for idx, row in enumerate(rows):
        item = dict(row)
        item["_route_order"] = idx
        chunks[idx % len(chunks)].append(item)
    return [chunk for chunk in chunks if chunk]


def worker_base_cmd(args: argparse.Namespace, input_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bench_coe.evaluate_qwen3vl_router_benchmarks",
        "--worker-input",
        str(input_path),
        "--worker-output",
        str(output_path),
        "--router-model-path",
        str(args.router_model_path),
        "--adapter-path",
        str(args.adapter_path),
        "--route-label-manifest",
        str(args.route_label_manifest),
        "--transformers-src",
        str(args.transformers_src),
        "--local-deps",
        str(args.local_deps),
        "--router-max-new-tokens",
        str(args.router_max_new_tokens),
        "--router-temperature",
        str(args.router_temperature),
        "--default-subject",
        str(args.default_subject),
    ]


def route_rows_parallel(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    label_manifest: dict[str, Any],
    output_dir: Path,
    benchmark: str,
) -> list[dict[str, Any]]:
    route_cache = output_dir / f"{benchmark}_routed_rows.json"
    if args.resume and route_cache.exists():
        return read_json(route_cache)

    gpu_list = [gpu.strip() for gpu in args.gpu_devices.split(",") if gpu.strip()]
    worker_count = min(max(1, args.num_router_workers), len(gpu_list), max(1, len(rows)))
    if worker_count <= 1:
        routed = route_rows_local(args, rows, label_manifest)
        write_json(route_cache, routed)
        return routed

    worker_dir = output_dir / "router_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[Any], Path, Path]] = []
    for shard_id, (gpu, chunk) in enumerate(zip(gpu_list, split_evenly(rows, worker_count))):
        input_path = worker_dir / f"{benchmark}_shard{shard_id}.input.json"
        output_path = worker_dir / f"{benchmark}_shard{shard_id}.output.json"
        log_path = worker_dir / f"{benchmark}_shard{shard_id}.log"
        write_json(input_path, {"rows": chunk, "label_manifest": label_manifest})
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        python_paths = [
            str(args.local_deps.resolve()),
            str(args.transformers_src.resolve()),
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
        log_file = log_path.open("w", encoding="utf-8")
        print(f"Launching Qwen3-VL router shard {shard_id} on GPU {gpu}: {len(chunk)} rows")
        proc = subprocess.Popen(
            worker_base_cmd(args, input_path, output_path),
            cwd=Path.cwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((proc, output_path, log_path))

    routed_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for proc, output_path, log_path in processes:
        return_code = proc.wait()
        if return_code != 0:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            failures.append(f"{log_path} exited {return_code}\n{tail}")
            continue
        routed_rows.extend(read_json(output_path))
    if failures:
        raise RuntimeError("Qwen3-VL router worker failed:\n\n" + "\n\n".join(failures))

    routed_rows.sort(key=lambda row: int(row.get("_route_order", 0)))
    for row in routed_rows:
        row.pop("_route_order", None)
    write_json(route_cache, routed_rows)
    return routed_rows


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise ValueError("--worker-input and --worker-output are required in worker mode.")
    payload = read_json(args.worker_input)
    rows = route_rows_local(args, payload["rows"], payload["label_manifest"])
    write_json(args.worker_output, rows)


def limit_rows(rows: list[dict[str, Any]], max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    import random

    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return sorted(copied[:max_examples], key=lambda row: int(row["question_id"]))


def format_mmlu_route_text(row: dict[str, Any]) -> str:
    options = [str(option) for option in row["options"] if str(option) != "N/A"]
    option_text = "\n".join(f"{CHOICES[idx]}. {option}" for idx, option in enumerate(options))
    return f"Question:\n{row['question']}\nOptions:\n{option_text}"


def load_mmlu_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.mmlu_data_dir / f"{args.mmlu_split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        options = [str(option) for option in raw["options"] if str(option) != "N/A"]
        row = {
            "question_id": int(raw["question_id"]),
            "question": str(raw["question"]).strip(),
            "options": options,
            "answer": str(raw["answer"]).strip(),
            "answer_index": int(raw["answer_index"]),
            "cot_content": str(raw.get("cot_content", "")),
            "category": str(raw["category"]).replace("_", " ").strip(),
            "src": str(raw.get("src", "")),
        }
        row["route_text"] = format_mmlu_route_text(row)
        rows.append(row)
    return limit_rows(rows, args.max_examples, args.seed)


def load_mmlu_prediction_cache(results_dir: Path, model_name: str, category: str) -> dict[int, dict[str, Any]]:
    path = results_dir / model_name / "CoT" / "all" / f"{category}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached MMLU-Pro predictions: {path}")
    return {int(row["question_id"]): row for row in read_json(path)}


def combine_mmlu(
    args: argparse.Namespace,
    routed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    caches: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    final: list[dict[str, Any]] = []
    for row in routed_rows:
        key = (row["routed_model"], row["category"])
        if key not in caches:
            caches[key] = load_mmlu_prediction_cache(args.mmlu_results_dir, key[0], key[1])
        cached = caches[key].get(int(row["question_id"]))
        if cached is None:
            raise KeyError(f"Missing MMLU cached row: model={key[0]}, category={key[1]}, qid={row['question_id']}")
        item = dict(row)
        item["pred"] = cached.get("pred")
        item["is_correct"] = item["pred"] == item["answer"]
        item["model_outputs"] = cached.get("model_outputs", "")
        item["expert_prediction_file"] = str(Path(key[0]) / "CoT" / "all" / f"{key[1]}.json")
        final.append(item)
    return final


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[str, Any]:
    correct = 0.0
    wrong = 0.0
    groups: dict[str, dict[str, dict[str, float]]] = {
        key: defaultdict(stats_dict) for key in group_keys
    }
    for row in rows:
        is_correct = bool(row.get("is_correct"))
        if is_correct:
            correct += 1
        else:
            wrong += 1
        for key in group_keys:
            stats = groups[key][str(row.get(key, "Unknown"))]
            if is_correct:
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
    total = correct + wrong
    summary: dict[str, Any] = {
        "total": {
            "correct": correct,
            "wrong": wrong,
            "accuracy": correct / total if total else 0.0,
        },
        "examples": int(total),
    }
    for key, stats in groups.items():
        summary[key] = finalize_stats(stats)
    return summary


def render_mmlu_txt(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    single_summaries: list[dict[str, Any]] = []
    for path in discover_mmlu_summaries(args.mmlu_results_dir / "summary").values():
        item = read_mmlu_summary(path)
        if item["overall_accuracy"] is not None:
            single_summaries.append(item)
    single_summaries.sort(key=lambda item: float(item["overall_accuracy"]), reverse=True)

    categories = [category for category in MMLU_CATEGORY_ORDER if category in summary["category"]]
    columns = categories + ["Average"]
    name_width = 36
    col_width = 17
    best = single_summaries[0] if single_summaries else None
    best_model = best.get("model") if best else None
    best_acc = float(best["overall_accuracy"]) if best else 0.0
    coe_acc = float(summary["total"]["accuracy"])
    counts = [str(total_count(summary["category"][category])) for category in categories]
    counts.append(str(int(summary["examples"])))

    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO-MM prior -> MMLU-Pro",
        "=" * 100,
        "| Routing Mode: qwen3vl_gaokao_mm_subject",
        "| Benchmark: MMLU-Pro",
        f"| Split: {args.mmlu_split}",
        f"| Samples: {int(summary['examples'])}",
        f"| Single model source: {args.mmlu_results_dir / 'summary'}",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 30, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 30, ["-" * 12 for _ in columns]),
    ]
    for item in single_summaries:
        model_name = str(item["model"])
        prefix = "* " if model_name == best_model else "  "
        values = [
            format_percent(item["category_accuracy"].get(category))
            if category in item["category_accuracy"]
            else "N/A"
            for category in categories
        ]
        values.append(format_percent(float(item["overall_accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))
    lines.append(row_line(name_width, col_width, "-" * 30, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(summary["category"][category]["accuracy"]) for category in categories]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]))
    lines.append("")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in rows).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO-MM subjects:")
    for subject, count in Counter(row["routed_subject"] for row in rows).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {output_dir / 'test_predictions.json'}")
    write_text(output_dir / "Bench_Harness_Result_qwen3vl_gaokao_mm_router_mmlupro.txt", lines)


def load_cached_jsonl_predictions(single_dir: Path, model_names: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    caches: dict[str, dict[int, dict[str, Any]]] = {}
    for model_name in sorted(set(model_names)):
        path = single_dir / model_name / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached predictions for {model_name}: {path}")
        caches[model_name] = {int(row["question_id"]): row for row in read_jsonl(path)}
    return caches


def combine_cached_jsonl(
    routed_rows: list[dict[str, Any]],
    caches: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    final: list[dict[str, Any]] = []
    for row in routed_rows:
        cached = caches[row["routed_model"]].get(int(row["question_id"]))
        if cached is None:
            raise KeyError(f"Missing cached row: model={row['routed_model']}, qid={row['question_id']}")
        item = dict(row)
        item["pred"] = cached.get("pred")
        item["is_correct"] = bool(cached.get("is_correct"))
        item["model_outputs"] = cached.get("model_outputs", "")
        item["expert_prediction_file"] = str(Path(row["routed_model"]) / "predictions.jsonl")
        item["expert_prompt_was_truncated"] = cached.get("prompt_was_truncated")
        item["expert_prompt_token_count"] = cached.get("prompt_token_count")
        final.append(item)
    return final


def single_summaries_from_dir(single_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(single_dir.glob("*/summary.json")):
        item = read_json(path)
        if item.get("status") == "completed" and item.get("accuracy") is not None:
            summaries.append(item)
    summaries.sort(key=lambda item: (-float(item["accuracy"]), str(item["model"])))
    return summaries


def render_bbh_txt(args: argparse.Namespace, output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    single_dir = args.single_root / "bbh"
    single_rows = single_summaries_from_dir(single_dir)
    tasks = sorted(summary["task"])
    columns = tasks + ["Average"]
    name_width = 34
    col_width = 15
    best = single_rows[0] if single_rows else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(summary["total"]["accuracy"])
    counts = [str(total_count(summary["task"][task])) for task in tasks] + [str(int(summary["examples"]))]

    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO-MM prior -> BBH",
        "=" * 100,
        "| Routing Mode: qwen3vl_gaokao_mm_subject",
        "| Benchmark: BBH",
        "| Split: test",
        f"| Samples: {int(summary['examples'])}",
        f"| Single model source: {single_dir}",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for item in single_rows:
        model_name = str(item["model"])
        prefix = "* " if model_name == best_model else "  "
        values = [
            format_percent(item.get("by_task", {}).get(task, {}).get("accuracy"))
            if task in item.get("by_task", {})
            else "N/A"
            for task in tasks
        ]
        values.append(format_percent(float(item["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))
    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(summary["task"][task]["accuracy"]) for task in tasks]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]))
    lines.append("")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in rows).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO-MM subjects:")
    for subject, count in Counter(row["routed_subject"] for row in rows).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {output_dir / 'test_predictions.json'}")
    write_text(output_dir / "Bench_Harness_Result_qwen3vl_gaokao_mm_router_bbh.txt", lines)


def aggregate_gpqa_diamond(path: Path, config: str) -> dict[str, Any] | None:
    rows = [row for row in read_jsonl(path) if row.get("config") == config]
    if not rows:
        return None
    by_domain: dict[str, dict[str, float]] = defaultdict(stats_dict)
    correct = 0.0
    wrong = 0.0
    unique_questions = set()
    for row in rows:
        unique_questions.add(row.get("base_question_id", row.get("record_id", row.get("question_id"))))
        stats = by_domain[str(row.get("domain", "Unknown"))]
        if bool(row.get("is_correct")):
            correct += 1
            stats["correct"] += 1
        else:
            wrong += 1
            stats["wrong"] += 1
    return {
        "model": path.parent.name,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / (correct + wrong) if correct + wrong else None,
        "num_examples": int(correct + wrong),
        "unique_questions": len(unique_questions),
        "by_domain": finalize_stats(by_domain),
    }


def render_gpqa_txt(args: argparse.Namespace, output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    single_dir = args.single_root / "gpqa"
    single_rows: list[dict[str, Any]] = []
    for path in sorted(single_dir.glob("*/predictions.jsonl")):
        item = aggregate_gpqa_diamond(path, args.gpqa_configs)
        if item is not None and item.get("accuracy") is not None:
            single_rows.append(item)
    single_rows.sort(key=lambda item: (-float(item["accuracy"]), str(item["model"])))
    domains = sorted(summary["domain"])
    columns = domains + ["Average"]
    name_width = 34
    col_width = 15
    best = single_rows[0] if single_rows else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(summary["total"]["accuracy"])
    counts = [str(total_count(summary["domain"][domain])) for domain in domains] + [str(int(summary["examples"]))]

    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO-MM prior -> GPQA",
        "=" * 100,
        "| Routing Mode: qwen3vl_gaokao_mm_subject",
        "| Benchmark: GPQA",
        f"| Configs: {', '.join(sorted(summary['config']))}",
        f"| Samples: {int(summary['examples'])}",
        f"| Unique Questions: {int(summary['unique_questions'])}",
        f"| Single model source: {single_dir} (filtered to config={args.gpqa_configs})",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for item in single_rows:
        model_name = str(item["model"])
        prefix = "* " if model_name == best_model else "  "
        values = [
            format_percent(item["by_domain"].get(domain, {}).get("accuracy"))
            if domain in item["by_domain"]
            else "N/A"
            for domain in domains
        ]
        values.append(format_percent(float(item["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))
    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(summary["domain"][domain]["accuracy"]) for domain in domains]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]))
    lines.append("")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in rows).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO-MM subjects:")
    for subject, count in Counter(row["routed_subject"] for row in rows).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {output_dir / 'test_predictions.json'}")
    write_text(output_dir / "Bench_Harness_Result_qwen3vl_gaokao_mm_router_gpqa.txt", lines)


def add_routing_stats(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    subject_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    for row in rows:
        for stats in [model_stats[row["routed_model"]], subject_stats[row["routed_subject"]]]:
            if row.get("is_correct"):
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
    summary["routed_model"] = finalize_stats(model_stats)
    summary["routed_subject"] = finalize_stats(subject_stats)
    return summary


def render_mmstar_txt(args: argparse.Namespace, output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    single_dir = args.single_root / "mmstar_text_only"
    single_rows = single_summaries_from_dir(single_dir)
    category_order = [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "math",
        "science & technology",
    ]
    available = set(summary["by_category"])
    categories = [category for category in category_order if category in available]
    categories.extend(sorted(available.difference(categories)))
    columns = categories + ["Average"]
    name_width = 34
    col_width = 20
    best = single_rows[0] if single_rows else None
    best_model = best.get("model") if best else None
    best_acc = float(best["accuracy"]) if best else 0.0
    coe_acc = float(summary["accuracy"])
    counts = [str(total_count(summary["by_category"][category])) for category in categories] + [str(int(summary["num_examples"]))]
    lines = [
        "=" * 100,
        "Bench-Harness: GAOKAO-MM prior -> MMStar",
        "=" * 100,
        "| Routing Mode: qwen3vl_gaokao_mm_subject",
        "| Benchmark: MMStar",
        "| Split: test",
        "| Mode: text-only",
        f"| Samples: {int(summary['num_examples'])}",
        f"| Single model source: {single_dir}",
        "",
        row_line(name_width, col_width, "Model / Metric", columns),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
        row_line(name_width, col_width, "Qs (Count)", counts),
        row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]),
    ]
    for item in single_rows:
        model_name = str(item["model"])
        prefix = "* " if model_name == best_model else "  "
        values = [
            format_percent(item.get("by_category", {}).get(category, {}).get("accuracy"))
            if category in item.get("by_category", {})
            else "N/A"
            for category in categories
        ]
        values.append(format_percent(float(item["accuracy"])))
        lines.append(row_line(name_width, col_width, prefix + model_name, values))
    lines.append(row_line(name_width, col_width, "-" * 28, ["-" * 12 for _ in columns]))
    coe_values = [format_percent(summary["by_category"][category]["accuracy"]) for category in categories]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line(name_width, col_width, args.coe_name, coe_values))
    lines.append(row_line(name_width, col_width, "Gain (vs Best Exp)", [""] * (len(columns) - 1) + [f"{(coe_acc - best_acc) * 100:+.2f}%"]))
    lines.append("")
    lines.append("Routed models:")
    for model_name, count in Counter(row["routed_model"] for row in rows).most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed GAOKAO-MM subjects:")
    for subject, count in Counter(row["routed_subject"] for row in rows).most_common():
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {output_dir / 'test_predictions.json'}")
    write_text(output_dir / "Bench_Harness_Result_qwen3vl_gaokao_mm_router_mmstar.txt", lines)


def write_leaderboard(path: Path, coe_name: str, benchmark: str, summary: dict[str, Any], single_rows: list[dict[str, Any]]) -> None:
    rows = [
        {
            "model": item["model"],
            "benchmark": benchmark,
            "accuracy": item["accuracy"],
            "correct": item.get("correct"),
            "total": item.get("num_examples"),
        }
        for item in single_rows
    ]
    if "total" in summary:
        rows.append(
            {
                "model": coe_name,
                "benchmark": benchmark,
                "accuracy": summary["total"]["accuracy"],
                "correct": summary["total"]["correct"],
                "total": summary["examples"],
            }
        )
    else:
        rows.append(
            {
                "model": coe_name,
                "benchmark": benchmark,
                "accuracy": summary["accuracy"],
                "correct": summary["correct"],
                "total": summary["num_examples"],
            }
        )
    rows.sort(key=lambda row: (-float(row["accuracy"]), str(row["model"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "benchmark", "accuracy", "correct", "total"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_mmlu(args: argparse.Namespace, label_manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_root / "mmlu_pro_qwen3vl_gaokao_mm_router_front4"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "test_predictions.json").exists() and (output_dir / "test_summary.json").exists():
        rows = read_json(output_dir / "test_predictions.json")
        summary = read_json(output_dir / "test_summary.json")
        render_mmlu_txt(args, output_dir, rows, summary)
        return summary
    rows = load_mmlu_rows(args)
    routed = route_rows_parallel(args, rows, label_manifest, output_dir, "mmlu")
    final_rows = combine_mmlu(args, routed)
    summary = summarize_rows(final_rows, ["category", "routed_model", "routed_subject"])
    summary["split"] = args.mmlu_split
    summary["prediction_file"] = str(output_dir / "test_predictions.json")
    write_json(output_dir / "test_predictions.json", final_rows)
    write_json(output_dir / "test_summary.json", summary)
    render_mmlu_txt(args, output_dir, final_rows, summary)
    return summary


def evaluate_bbh(args: argparse.Namespace, label_manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_root / "bbh_qwen3vl_gaokao_mm_router_front4"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "test_predictions.json").exists() and (output_dir / "test_summary.json").exists():
        rows = read_json(output_dir / "test_predictions.json")
        summary = read_json(output_dir / "test_summary.json")
        render_bbh_txt(args, output_dir, rows, summary)
        return summary
    rows = load_bbh_rows(args)
    for row in rows:
        row["route_text"] = str(row["input"])
    routed = route_rows_parallel(args, rows, label_manifest, output_dir, "bbh")
    caches = load_cached_jsonl_predictions(args.single_root / "bbh", [row["routed_model"] for row in routed])
    final_rows = combine_cached_jsonl(routed, caches)
    summary = summarize_rows(final_rows, ["task", "routed_model", "routed_subject"])
    summary["split"] = "test"
    summary["prediction_file"] = str(output_dir / "test_predictions.json")
    write_json(output_dir / "test_predictions.json", final_rows)
    write_json(output_dir / "test_summary.json", summary)
    render_bbh_txt(args, output_dir, final_rows, summary)
    write_leaderboard(output_dir / "bbh_leaderboard.csv", args.coe_name, "BBH", summary, single_summaries_from_dir(args.single_root / "bbh"))
    return summary


def evaluate_gpqa(args: argparse.Namespace, label_manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_root / "gpqa_diamond_qwen3vl_gaokao_mm_router_front4"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "test_predictions.json").exists() and (output_dir / "test_summary.json").exists():
        rows = read_json(output_dir / "test_predictions.json")
        summary = read_json(output_dir / "test_summary.json")
        render_gpqa_txt(args, output_dir, rows, summary)
        return summary
    rows = load_gpqa_rows(args)
    for row in rows:
        option_text = "\n".join(f"{CHOICES[idx]}. {option}" for idx, option in enumerate(row["options"]))
        row["route_text"] = f"Question:\n{row['question']}\nOptions:\n{option_text}"
    routed = route_rows_parallel(args, rows, label_manifest, output_dir, "gpqa")
    caches = load_cached_jsonl_predictions(args.single_root / "gpqa", [row["routed_model"] for row in routed])
    final_rows = combine_cached_jsonl(routed, caches)
    summary = summarize_rows(final_rows, ["config", "domain", "subdomain", "routed_model", "routed_subject"])
    summary["unique_questions"] = len({row.get("base_question_id") for row in final_rows})
    summary["prediction_file"] = str(output_dir / "test_predictions.json")
    write_json(output_dir / "test_predictions.json", final_rows)
    write_json(output_dir / "test_summary.json", summary)
    render_gpqa_txt(args, output_dir, final_rows, summary)
    return summary


def evaluate_mmstar(args: argparse.Namespace, label_manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_root / "mmstar_text_only_qwen3vl_gaokao_mm_router_front4"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "test_predictions.json").exists() and (output_dir / "test_summary.json").exists():
        rows = read_json(output_dir / "test_predictions.json")
        summary = read_json(output_dir / "test_summary.json")
        render_mmstar_txt(args, output_dir, rows, summary)
        return summary
    rows = load_mmstar_rows(args)
    for row in rows:
        row["route_text"] = str(row["question"])
    routed = route_rows_parallel(args, rows, label_manifest, output_dir, "mmstar")
    caches = load_cached_jsonl_predictions(args.single_root / "mmstar_text_only", [row["routed_model"] for row in routed])
    final_rows = combine_cached_jsonl(routed, caches)
    write_json(output_dir / "test_predictions.json", final_rows)
    summary = summarize_mmstar_fallback(args.coe_name, final_rows, args)
    summary = add_routing_stats(summary, final_rows)
    summary["split"] = "test"
    summary["mode"] = "text_only_routed"
    summary["examples"] = len(final_rows)
    summary["prediction_file"] = str(output_dir / "test_predictions.json")

    result_df = pd.DataFrame(
        [
            {
                "index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "l2_category": row["l2_category"],
                "bench": row["bench"],
                "prediction": row.get("pred") or "",
                "model_outputs": row.get("model_outputs", ""),
                "routed_model": row.get("routed_model", ""),
                "routed_subject": row.get("routed_subject", ""),
            }
            for row in final_rows
        ]
    )
    csv_path = output_dir / f"{args.coe_name}_MMStar.csv"
    xlsx_path = output_dir / f"{args.coe_name}_MMStar.xlsx"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        result_df.to_excel(xlsx_path, index=False)
    except Exception:
        # The wrapped MMStar evaluator is patched to read the in-memory DataFrame,
        # so a missing Excel writer should not prevent official scoring.
        pass
    try:
        score_payload, score_file = run_mmstar_eval_on_dataframe(args, result_df, xlsx_path)
        summary["official_mmstar_result_file"] = str(csv_path)
        if xlsx_path.exists():
            summary["official_mmstar_xlsx_file"] = str(xlsx_path)
        summary["official_mmstar_score_file"] = str(score_file)
        summary["official_mmstar_scores"] = score_payload
        summary["accuracy"] = float(score_payload.get("final score", summary["accuracy"]))
        summary["evaluation"] = "MMStar_eval"
    except Exception as exc:
        summary["evaluation_warning"] = f"MMStar_eval failed; fallback scores kept: {exc}"
    write_json(output_dir / "test_summary.json", summary)
    render_mmstar_txt(args, output_dir, final_rows, summary)
    write_leaderboard(output_dir / "mmstar_text_only_leaderboard.csv", args.coe_name, "MMStar", summary, single_summaries_from_dir(args.single_root / "mmstar_text_only"))
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.worker_input is not None:
        run_worker(args)
        return
    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Missing trained Qwen3-VL router adapter: {args.adapter_path}")
    label_manifest = read_json(args.route_label_manifest)
    start = time.time()
    summaries: dict[str, Any] = {}
    if "mmlu" in args.benchmarks:
        summaries["mmlu"] = evaluate_mmlu(args, label_manifest)
    if "bbh" in args.benchmarks:
        summaries["bbh"] = evaluate_bbh(args, label_manifest)
    if "gpqa" in args.benchmarks:
        summaries["gpqa"] = evaluate_gpqa(args, label_manifest)
    if "mmstar" in args.benchmarks:
        summaries["mmstar"] = evaluate_mmstar(args, label_manifest)
    write_json(
        args.output_root / "qwen3vl_gaokao_mm_router_front4_run_manifest.json",
        {
            "router_model_path": str(args.router_model_path),
            "adapter_path": str(args.adapter_path),
            "route_label_manifest": str(args.route_label_manifest),
            "benchmarks": args.benchmarks,
            "gpu_devices": args.gpu_devices,
            "num_router_workers": args.num_router_workers,
            "elapsed_seconds": time.time() - start,
            "summaries": summaries,
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
