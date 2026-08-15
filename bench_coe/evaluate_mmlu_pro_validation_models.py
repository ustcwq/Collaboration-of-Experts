from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


CHOICES = list("ABCDEFGHIJKLMNOP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean MMLU-Pro validation predictions for local expert models."
    )
    parser.add_argument("--models", default="all", help="Comma-separated model names, or all.")
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("MMLU-Pro/data/validation-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/bench_coe/mmlu_pro_validation_single_models"),
    )
    parser.add_argument("--gpu-id", default="0", help="Single GPU id visible to vLLM for this process.")
    parser.add_argument("--gpu-util", type=float, default=0.8)
    parser.add_argument("--ntrain", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--categories", default="", help="Optional comma-separated category subset.")
    parser.add_argument("--summary-suffix", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        item["options"] = [str(opt) for opt in item["options"] if str(opt) != "N/A"]
        item["question_id"] = int(item["question_id"])
        item["answer_index"] = int(item["answer_index"])
        rows.append(item)
    return rows


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.models != "all":
        return [item.strip() for item in args.models.split(",") if item.strip()]
    wanted = []
    for path in sorted(args.model_root.resolve().iterdir()):
        if not path.is_dir():
            continue
        if (path / "config.json").exists() or (path / "params.json").exists():
            wanted.append(path.name)
    return wanted


def initial_prompt() -> str:
    path = Path("MMLU-Pro/cot_prompt_lib/initial_prompt.txt")
    return path.read_text(encoding="utf-8")


def format_cot_example(example: dict[str, Any], including_answer: bool) -> str:
    prompt = "Question:\n"
    prompt += str(example["question"]) + "\n"
    prompt += "Options:\n"
    for idx, opt in enumerate(example["options"]):
        prompt += f"{CHOICES[idx]}. {opt}\n"
    if including_answer:
        cot = str(example["cot_content"]).replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def validation_prompt(
    all_rows: list[dict[str, Any]],
    current: dict[str, Any],
    ntrain: int,
) -> str:
    prompt = initial_prompt().replace("{$}", str(current["category"])) + "\n"
    shots = [
        row
        for row in all_rows
        if row["category"] == current["category"] and row["question_id"] != current["question_id"]
    ][:ntrain]
    for shot in shots:
        prompt += format_cot_example(shot, including_answer=True)
    prompt += format_cot_example(current, including_answer=False)
    return prompt


def extract_answer(text: str) -> str | None:
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def load_vllm(model_path: Path, args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    import transformers
    from bench_coe.run_official_model_benchmarks import import_vllm_objects

    LLM, SamplingParams = import_vllm_objects("LLM", "SamplingParams")

    llm = LLM(
        model=str(model_path),
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, stop=["Question:"])
    return llm, tokenizer, sampling


def load_model(model_path: Path, args: argparse.Namespace):
    if args.backend == "vllm":
        return load_vllm(model_path, args)
    from bench_coe.transformers_text_backend import TransformersSamplingParams, TransformersTextLLM

    llm = TransformersTextLLM(
        str(model_path),
        str(args.gpu_id),
        args.max_model_len,
        args.attn_implementation,
    )
    return llm, llm.get_tokenizer(), TransformersSamplingParams(
        max_tokens=args.max_new_tokens,
        stop=("Question:",),
    )


def output_done(model_out: Path, rows: list[dict[str, Any]]) -> bool:
    if not model_out.is_dir():
        return False
    expected = {str(row["category"]) for row in rows}
    found = {path.stem for path in model_out.glob("*.json")}
    if expected.difference(found):
        return False
    total = 0
    for path in model_out.glob("*.json"):
        try:
            total += len(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return False
    return total == len(rows)


def save_category(path: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    correct = sum(1 for row in rows if row.get("pred") == row.get("answer"))
    return correct, len(rows) - correct


def evaluate_model(model_name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model_root.resolve() / model_name
    model_out = args.output_root / model_name / "CoT" / "validation"
    if output_done(model_out, rows) and not args.overwrite:
        return {"model": model_name, "status": "skipped_existing", "output_dir": str(model_out)}
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    llm, tokenizer, sampling = load_model(model_path, args)
    prompts: list[str] = []
    ordered_rows: list[dict[str, Any]] = []
    for row in rows:
        k = min(args.ntrain, max(0, sum(1 for item in rows if item["category"] == row["category"]) - 1))
        while True:
            prompt = validation_prompt(rows, row, k)
            length = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            if length < args.max_model_len - args.max_new_tokens or k <= 0:
                break
            k -= 1
        prompts.append(prompt)
        ordered_rows.append(row)

    outputs = llm.generate(prompts, sampling)
    by_category: dict[str, list[dict[str, Any]]] = {}
    random.seed(12345)
    for source, output in zip(ordered_rows, outputs):
        response = output.outputs[0].text
        pred = extract_answer(response)
        item = dict(source)
        item["pred"] = pred
        item["model_outputs"] = response
        by_category.setdefault(str(item["category"]), []).append(item)

    total_correct = 0
    total_wrong = 0
    category_stats: dict[str, dict[str, float]] = {}
    for category, cat_rows in sorted(by_category.items()):
        correct, wrong = save_category(model_out / f"{category}.json", cat_rows)
        total_correct += correct
        total_wrong += wrong
        category_stats[category] = {
            "correct": float(correct),
            "wrong": float(wrong),
            "accuracy": correct / (correct + wrong) if correct + wrong else 0.0,
        }

    summary = {
        "model": model_name,
        "split": "validation",
        "examples": total_correct + total_wrong,
        "correct": total_correct,
        "wrong": total_wrong,
        "accuracy": total_correct / (total_correct + total_wrong) if total_correct + total_wrong else 0.0,
        "category": category_stats,
        "output_dir": str(model_out),
        "timestamp": time.time(),
    }
    summary_path = args.output_root / model_name / f"summary_validation{args.summary_suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    rows = read_rows(args.validation_file)
    if args.categories:
        categories = {item.strip() for item in args.categories.split(",") if item.strip()}
        rows = [row for row in rows if str(row["category"]) in categories]
        if not rows:
            raise SystemExit(f"No rows matched categories: {sorted(categories)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_name in selected_models(args):
        print(f"[validation] {model_name}")
        summaries.append(evaluate_model(model_name, rows, args))
    summary_name = f"run_summary{args.summary_suffix}.json"
    (args.output_root / summary_name).write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
