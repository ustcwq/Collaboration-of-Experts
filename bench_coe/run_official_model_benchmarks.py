from __future__ import annotations

import argparse
import importlib.util
import gc
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import traceback
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from bench_coe.gaokao_utils import discover_result_models, filter_local_models, write_json


CHOICES = "ABCD"
BENCHMARKS = ("bbh", "gpqa", "mmstar_text_only")
GPQA_CONFIG_FILES = {
    "diamond": "gpqa_diamond.csv",
    "main": "gpqa_main.csv",
    "extended": "gpqa_extended.csv",
}

MC_RE = re.compile(r"^\(([A-Z])\)$")
INTEGER_RE = re.compile(r"^-?\d+$")
FINAL_ANSWER_MARKER_RE = re.compile(r"(?:final answer)\s*(?:is|:)?\s*", re.IGNORECASE)
ANSWER_MARKER_RE = re.compile(r"(?:answer)\s*(?:is|:)\s*", re.IGNORECASE)
GPQA_ANSWER_RE = re.compile(r"(?:answer|final answer|correct answer)\s*(?:is|:)?\s*\(?([A-D])\)?", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local GAOKAO-result models on BBH, GPQA, and MMStar text-only "
            "using the local benchmark repositories/data."
        )
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--gaokao-data-dir",
        type=Path,
        default=Path("GAOKAO-Bench-2010-2022/Data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_benchmarks/official_code_local_models"),
    )
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated list from bbh,gpqa,mmstar_text_only, or all.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model-name allowlist. Defaults to GAOKAO result models present in --models-dir.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help="Optional model-name denylist.",
    )
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="0 means one worker per listed GPU. Each worker loads one model on one GPU.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--row-shard-count", type=int, default=1)
    parser.add_argument("--row-shard-index", type=int, default=0)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed benchmark/model outputs.",
    )

    parser.add_argument("--bbh-data-dir", type=Path, default=Path("BIG-Bench-Hard/bbh"))
    parser.add_argument(
        "--bbh-cot-dir",
        type=Path,
        default=Path("BIG-Bench-Hard/cot-prompts"),
    )
    parser.add_argument(
        "--bbh-tasks",
        default="all",
        help="Comma-separated BBH task names, or all.",
    )
    parser.add_argument("--bbh-max-new-tokens", type=int, default=512)

    parser.add_argument("--gpqa-data-dir", type=Path, default=Path("data/gpqa"))
    parser.add_argument(
        "--gpqa-configs",
        default="all",
        help="Comma-separated GPQA configs: diamond,main,extended, or all.",
    )
    parser.add_argument("--gpqa-epochs", type=int, default=4)
    parser.add_argument(
        "--shuffle-gpqa-choices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gpqa-prompt-type",
        choices=["zero_shot", "cot"],
        default="zero_shot",
    )
    parser.add_argument("--gpqa-max-new-tokens", type=int, default=256)

    parser.add_argument(
        "--mmstar-tsv",
        type=Path,
        default=Path("data/MMStar/MMStar.tsv"),
    )
    parser.add_argument(
        "--mmstar-eval-dir",
        type=Path,
        default=Path("MMStar/eval"),
    )
    parser.add_argument("--mmstar-max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--mmstar-store-extracted-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store the parsed answer letter in the official prediction column and keep raw text "
            "in model_outputs. This makes MMStar_eval score answer letters instead of verbose text."
        ),
    )

    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_csv_list(value: str, valid_items: tuple[str, ...] | list[str]) -> list[str]:
    valid = list(valid_items)
    if value == "all":
        return valid
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(selected).difference(valid))
    if missing:
        raise ValueError(f"Unknown item(s): {missing}; valid values are {valid}")
    return selected


def completed_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return payload.get("status") == "completed"


def benchmark_output_dir(args: argparse.Namespace, benchmark: str, model_name: str) -> Path:
    return args.output_dir / benchmark / model_name


def benchmark_summary_path(args: argparse.Namespace, benchmark: str, model_name: str) -> Path:
    return benchmark_output_dir(args, benchmark, model_name) / "summary.json"


def is_benchmark_complete(args: argparse.Namespace, benchmark: str, model_name: str) -> bool:
    if not args.resume:
        return False
    summary_path = benchmark_summary_path(args, benchmark, model_name)
    if not completed_summary(summary_path):
        return False
    out_dir = benchmark_output_dir(args, benchmark, model_name)
    if benchmark in {"bbh", "gpqa"}:
        return (out_dir / "predictions.jsonl").exists()
    if benchmark == "mmstar_text_only":
        return (
            (out_dir / f"{model_name}_MMStar.csv").exists()
            and (out_dir / f"{model_name}_MMStar_score.json").exists()
        ) or (out_dir / f"{model_name}_MMStar.xlsx").exists()
    return False


def shard_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.row_shard_count < 1:
        raise ValueError("--row-shard-count must be at least 1")
    if not 0 <= args.row_shard_index < args.row_shard_count:
        raise ValueError("--row-shard-index must be in [0, row_shard_count)")
    if args.row_shard_count == 1:
        return rows
    return rows[args.row_shard_index :: args.row_shard_count]


def clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def limit_rows(rows: list[dict[str, Any]], max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return sorted(copied[:max_examples], key=lambda row: str(row["question_id"]))


def selected_bbh_tasks(data_dir: Path, selected_tasks: str) -> list[str]:
    tasks = sorted(path.stem for path in data_dir.glob("*.json"))
    if selected_tasks == "all":
        return tasks
    wanted = [task.strip() for task in selected_tasks.split(",") if task.strip()]
    missing = sorted(set(wanted).difference(tasks))
    if missing:
        raise FileNotFoundError(f"Unknown BBH tasks: {missing}")
    return [task for task in tasks if task in set(wanted)]


def load_bbh_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in selected_bbh_tasks(args.bbh_data_dir, args.bbh_tasks):
        path = args.bbh_data_dir / f"{task}.json"
        payload = read_json(path)
        for task_index, example in enumerate(payload.get("examples", [])):
            rows.append(
                {
                    "question_id": len(rows),
                    "id": f"{task}:{task_index}",
                    "task": task,
                    "task_index": task_index,
                    "input": str(example["input"]).strip(),
                    "target": str(example["target"]).strip(),
                }
            )
    return limit_rows(rows, args.max_examples, args.seed)


def load_bbh_cot_prompt(args: argparse.Namespace, task: str) -> str:
    prompt_path = args.bbh_cot_dir / f"{task}.txt"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


def build_bbh_prompt(args: argparse.Namespace, row: dict[str, Any]) -> str:
    cot_prompt = load_bbh_cot_prompt(args, row["task"])
    if cot_prompt:
        return f"{cot_prompt}\n\nQ: {row['input']}\nA: Let's think step by step."
    task = row["task"].replace("_", " ")
    return (
        f"The following is a BIG-Bench Hard task named {task}.\n"
        "Think step by step, then give the final answer exactly in the format "
        "\"Final answer: <answer>\". For multiple-choice questions, use the option "
        "letter in parentheses, such as \"Final answer: (A)\".\n\n"
        f"Problem:\n{row['input']}\n\n"
        "Answer: Let's think step by step."
    )


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def strip_answer_noise(text: str) -> str:
    text = text.strip()
    text = re.split(r"\n\s*(?:Problem|Question|Q):", text, maxsplit=1)[0].strip()
    text = text.strip("` ")
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1].strip()
    return text.strip().rstrip(".。")


def segment_after_marker(text: str, matches: list[re.Match[str]]) -> str | None:
    for idx in range(len(matches) - 1, -1, -1):
        start = matches[idx].end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        candidate = text[start:end].strip()
        if not candidate:
            continue
        first_line = next((line.strip() for line in candidate.splitlines() if line.strip()), "")
        if first_line:
            return strip_answer_noise(first_line)
    return None


def final_answer_segment(text: str) -> str:
    final_matches = list(FINAL_ANSWER_MARKER_RE.finditer(text))
    segment = segment_after_marker(text, final_matches)
    if segment:
        return segment
    answer_matches = list(ANSWER_MARKER_RE.finditer(text))
    segment = segment_after_marker(text, answer_matches)
    if segment:
        return segment
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return strip_answer_noise(lines[-1] if lines else text)


def extract_bbh_prediction(gold: str, generated_text: str) -> str | None:
    segment = final_answer_segment(generated_text)
    gold = gold.strip()

    mc_match = MC_RE.match(gold)
    if mc_match:
        letters = re.findall(r"\(([A-Z])\)", segment)
        if letters:
            return f"({letters[-1]})"
        letter_match = re.search(r"\b([A-Z])\b", segment)
        if letter_match:
            return f"({letter_match.group(1)})"
        fallback = re.findall(r"\(([A-Z])\)", generated_text)
        if fallback:
            return f"({fallback[-1]})"
        return None

    gold_lower = gold.lower()
    if gold_lower in {"yes", "no", "true", "false", "valid", "invalid"}:
        match = re.search(r"\b(yes|no|true|false|valid|invalid)\b", segment, re.I)
        if match:
            return match.group(1)
        plausible_match = re.search(r"\b(implausible|plausible)\b", segment, re.I)
        if plausible_match:
            return "no" if plausible_match.group(1).lower() == "implausible" else "yes"
        fallback = re.findall(r"\b(yes|no|true|false|valid|invalid)\b", generated_text, re.I)
        return fallback[-1] if fallback else None

    if INTEGER_RE.match(gold):
        match = re.search(r"-?\d[\d,]*", segment)
        if match:
            return match.group(0).replace(",", "")
        fallback = re.findall(r"-?\d[\d,]*", generated_text)
        return fallback[-1].replace(",", "") if fallback else None

    if not re.search(r"[A-Za-z0-9]", gold):
        symbol_match = re.match(r"[\[\]\(\)\{\}<>\s]+", segment)
        if symbol_match:
            return normalize_spaces(symbol_match.group(0))
    segment = re.sub(r"^\([A-Z]\)\s*", "", segment)
    return normalize_spaces(segment) if segment else None


def normalize_free_text(text: str) -> str:
    text = re.sub(r"^\([A-Z]\)\s*", "", text.strip())
    text = re.sub(r"[,.;:]+", " ", text)
    return normalize_spaces(text).lower()


def is_bbh_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    pred = pred.strip()
    gold = gold.strip()
    if MC_RE.match(gold):
        return pred.upper() == gold.upper()
    if INTEGER_RE.match(gold):
        return pred.replace(",", "") == gold
    if gold.lower() in {"yes", "no", "true", "false", "valid", "invalid"}:
        return pred.lower() == gold.lower()
    return normalize_free_text(pred) == normalize_free_text(gold)


def selected_gpqa_configs(configs: str) -> list[str]:
    if configs == "all":
        return list(GPQA_CONFIG_FILES)
    selected = [item.strip() for item in configs.split(",") if item.strip()]
    missing = sorted(set(selected).difference(GPQA_CONFIG_FILES))
    if missing:
        raise ValueError(f"Unknown GPQA configs: {missing}")
    return selected


def build_gpqa_choices(row: dict[str, Any], base_qid: int, epoch: int, seed: int, shuffle: bool) -> tuple[list[str], str]:
    raw_choices = [
        ("D", clean_cell(row["Correct Answer"])),
        ("A", clean_cell(row["Incorrect Answer 1"])),
        ("B", clean_cell(row["Incorrect Answer 2"])),
        ("C", clean_cell(row["Incorrect Answer 3"])),
    ]
    if shuffle:
        rng = random.Random(seed + base_qid * 1009 + epoch * 9176)
        rng.shuffle(raw_choices)
    label_map: dict[str, str] = {}
    options: list[str] = []
    for idx, (original_label, answer_text) in enumerate(raw_choices):
        new_label = CHOICES[idx]
        label_map[original_label] = new_label
        options.append(answer_text)
    return options, label_map["D"]


def load_gpqa_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_qid = 0
    for config in selected_gpqa_configs(args.gpqa_configs):
        path = args.gpqa_data_dir / GPQA_CONFIG_FILES[config]
        df = pd.read_csv(path)
        for base_index, raw_row in enumerate(df.to_dict(orient="records")):
            question = clean_cell(raw_row["Question"])
            domain = clean_cell(raw_row.get("High-level domain", "Unknown")) or "Unknown"
            subdomain = clean_cell(raw_row.get("Subdomain", "Unknown")) or "Unknown"
            record_id = clean_cell(raw_row.get("Record ID", f"{config}:{base_index}"))
            for epoch in range(args.gpqa_epochs):
                options, answer = build_gpqa_choices(
                    raw_row,
                    base_qid,
                    epoch,
                    args.seed,
                    args.shuffle_gpqa_choices,
                )
                rows.append(
                    {
                        "question_id": len(rows),
                        "base_question_id": base_qid,
                        "id": f"{config}:{base_index}:epoch{epoch}",
                        "config": config,
                        "epoch": epoch,
                        "record_id": record_id,
                        "domain": domain,
                        "subdomain": subdomain,
                        "question": question,
                        "options": options,
                        "answer": answer,
                    }
                )
            base_qid += 1
    return limit_rows(rows, args.max_examples, args.seed)


def format_gpqa_question(row: dict[str, Any]) -> str:
    options = "\n".join(
        f"({CHOICES[idx]}) {option}" for idx, option in enumerate(row["options"])
    )
    return f"{row['question']}\n\nChoices:\n{options}"


def build_gpqa_prompt(args: argparse.Namespace, row: dict[str, Any]) -> str:
    question = format_gpqa_question(row)
    if args.gpqa_prompt_type == "cot":
        return (
            "The following is a GPQA multiple-choice question. Think step by step, "
            "then output the final answer exactly in the format \"The correct answer is (X)\".\n\n"
            f"Question:\n{question}\n\n"
            "Answer: Let's think step by step."
        )
    return (
        f"What is the correct answer to this question: {row['question']}\n\n"
        f"Choices:\n(A) {row['options'][0]}\n(B) {row['options'][1]}"
        f"\n(C) {row['options'][2]}\n(D) {row['options'][3]}"
        "\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""
    )


def extract_gpqa_answer(text: str) -> str | None:
    matches = list(GPQA_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    matches = re.findall(r"\(([A-D])\)", text)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
    return None


def load_mmstar_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    df = pd.read_csv(args.mmstar_tsv, sep="\t")
    rows: list[dict[str, Any]] = []
    for idx, raw_row in enumerate(df.to_dict(orient="records")):
        rows.append(
            {
                "question_id": len(rows),
                "id": str(raw_row.get("index", idx)),
                "index": raw_row.get("index", idx),
                "question": clean_cell(raw_row["question"]),
                "answer": clean_cell(raw_row["answer"]).upper(),
                "category": clean_cell(raw_row["category"]),
                "l2_category": clean_cell(raw_row["l2_category"]),
                "bench": clean_cell(raw_row.get("bench", "")),
            }
        )
    return limit_rows(rows, args.max_examples, args.seed)


def build_mmstar_prompt(row: dict[str, Any]) -> str:
    return (
        "The following is an MMStar multiple-choice question in text-only mode. "
        "Answer with one option letter from A, B, C, or D. If image information is "
        "missing, make the best choice from the text and options.\n\n"
        f"{row['question']}\n\n"
        "Answer with the option letter only:"
    )


def extract_choice_letter(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    match = re.match(r"^\(?([A-Da-d])\)?(?:[.\):\s]|$)", stripped)
    if match:
        return match.group(1).upper()
    matches = list(GPQA_ANSWER_RE.finditer(stripped))
    if matches:
        return matches[-1].group(1).upper()
    paren_matches = re.findall(r"\(([A-Da-d])\)", stripped)
    if paren_matches:
        return paren_matches[-1].upper()
    word_matches = re.findall(r"\b([A-Da-d])\b", stripped)
    if word_matches:
        return word_matches[-1].upper()
    return None


def cleanup_vllm() -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def import_vllm_objects(*names: str) -> tuple[Any, ...]:
    try:
        import vllm

        if all(hasattr(vllm, name) for name in names):
            return tuple(getattr(vllm, name) for name in names)
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    candidates = [
        item
        for item in os.environ.get("BENCH_COE_VLLM_SOURCE", "").split(os.pathsep)
        if item
    ]
    candidates.extend(
        [
            "/home/sm5/ys/Project/vllm",
            str(root / "vllm"),
        ]
    )
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not (candidate_path / "vllm" / "__init__.py").exists():
            continue
        sys.path.insert(0, str(candidate_path))
        for module_name in list(sys.modules):
            if module_name == "vllm" or module_name.startswith("vllm."):
                del sys.modules[module_name]
        try:
            import vllm

            if all(hasattr(vllm, name) for name in names):
                return tuple(getattr(vllm, name) for name in names)
        except Exception:
            continue

    raise ImportError(
        "Could not import required vLLM object(s): "
        + ", ".join(names)
        + ". Set BENCH_COE_VLLM_SOURCE to a vLLM source/install path if needed."
    )


def load_llm(args: argparse.Namespace, model_name: str):
    model_path = args.models_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local model directory: {model_path}")
    if args.backend == "transformers":
        from bench_coe.transformers_text_backend import TransformersTextLLM

        return TransformersTextLLM(
            str(model_path),
            None,
            args.max_model_len,
            args.attn_implementation,
        )

    (LLM,) = import_vllm_objects("LLM")
    return LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )


def make_sampling_params(args: argparse.Namespace, benchmark: str):
    stop_by_benchmark = {
        "bbh": ["\nQ:"],
        "gpqa": ["\nQuestion:"],
        "mmstar_text_only": ["\nQuestion:"],
    }
    if args.backend == "transformers":
        from bench_coe.transformers_text_backend import TransformersSamplingParams

        return TransformersSamplingParams(
            temperature=args.temperature,
            max_tokens=max_new_tokens_for_benchmark(args, benchmark),
            stop=tuple(stop_by_benchmark.get(benchmark, ())),
        )

    (SamplingParams,) = import_vllm_objects("SamplingParams")
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_new_tokens_for_benchmark(args, benchmark),
        stop=stop_by_benchmark.get(benchmark),
    )


def max_new_tokens_for_benchmark(args: argparse.Namespace, benchmark: str) -> int:
    return {
        "bbh": args.bbh_max_new_tokens,
        "gpqa": args.gpqa_max_new_tokens,
        "mmstar_text_only": args.mmstar_max_new_tokens,
    }[benchmark]


def apply_chat_template(llm: Any, prompt: str) -> str:
    try:
        tokenizer = llm.get_tokenizer()
        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return prompt
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def truncate_prompt_if_needed(
    llm: Any,
    prompt: str,
    max_input_tokens: int,
) -> tuple[str, bool, int | None]:
    if max_input_tokens <= 0:
        return prompt, False, None
    try:
        tokenizer = llm.get_tokenizer()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception:
        return prompt, False, None
    token_count = len(token_ids)
    if token_count <= max_input_tokens:
        return prompt, False, token_count
    head_tokens = max(1, max_input_tokens // 2)
    tail_tokens = max_input_tokens - head_tokens
    kept_ids = token_ids[:head_tokens] + token_ids[-tail_tokens:]
    try:
        return tokenizer.decode(kept_ids, skip_special_tokens=False), True, token_count
    except Exception:
        return prompt, False, token_count


def generate_rows(
    args: argparse.Namespace,
    llm: Any,
    rows: list[dict[str, Any]],
    benchmark: str,
    prompt_builder: Any,
    output_parser: Any,
    correct_fn: Any,
) -> list[dict[str, Any]]:
    sampling_params = make_sampling_params(args, benchmark)
    max_input_tokens = args.max_model_len - max_new_tokens_for_benchmark(args, benchmark) - 8
    results: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=benchmark):
        batch_rows = rows[start : start + args.batch_size]
        raw_prompts = [prompt_builder(row) for row in batch_rows]
        prompts: list[str] = []
        truncation_info: list[tuple[bool, int | None]] = []
        for prompt in raw_prompts:
            chat_prompt = apply_chat_template(llm, prompt)
            truncated_prompt, was_truncated, original_token_count = truncate_prompt_if_needed(
                llm,
                chat_prompt,
                max_input_tokens,
            )
            prompts.append(truncated_prompt)
            truncation_info.append((was_truncated, original_token_count))
        outputs = llm.generate(prompts, sampling_params)
        for row, prompt, output, (was_truncated, original_token_count) in zip(
            batch_rows,
            raw_prompts,
            outputs,
            truncation_info,
        ):
            generated_text = output.outputs[0].text
            pred = output_parser(row, generated_text)
            result = dict(row)
            if args.save_prompts:
                result["prompt"] = prompt
            result["prompt_was_truncated"] = was_truncated
            if original_token_count is not None:
                result["prompt_token_count"] = original_token_count
            result["pred"] = pred
            result["is_correct"] = bool(correct_fn(row, pred))
            result["model_outputs"] = generated_text
            results.append(result)
    return results


def stats_dict() -> dict[str, float]:
    return {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}


def finalize_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for item in stats.values():
        denom = item["correct"] + item["wrong"]
        item["accuracy"] = item["correct"] / denom if denom else 0.0
    return dict(sorted(stats.items()))


def summarize_bbh(model_name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    task_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    correct = 0.0
    wrong = 0.0
    for row in rows:
        stats = task_stats[row["task"]]
        if row.get("is_correct"):
            correct += 1
            stats["correct"] += 1
        else:
            wrong += 1
            stats["wrong"] += 1
    total = correct + wrong
    return {
        "status": "completed",
        "benchmark": "bbh",
        "model": model_name,
        "num_examples": int(total),
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "by_task": finalize_stats(task_stats),
        "data_source": str(args.bbh_data_dir),
        "prompt_source": str(args.bbh_cot_dir),
    }


def summarize_gpqa(model_name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    config_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    domain_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    subdomain_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    correct = 0.0
    wrong = 0.0
    for row in rows:
        targets = [config_stats[row["config"]], domain_stats[row["domain"]], subdomain_stats[row["subdomain"]]]
        if row.get("is_correct"):
            correct += 1
            for stats in targets:
                stats["correct"] += 1
        else:
            wrong += 1
            for stats in targets:
                stats["wrong"] += 1
    total = correct + wrong
    return {
        "status": "completed",
        "benchmark": "gpqa",
        "model": model_name,
        "num_examples": int(total),
        "unique_questions": len({row["record_id"] for row in rows}),
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "by_config": finalize_stats(config_stats),
        "by_domain": finalize_stats(domain_stats),
        "by_subdomain": finalize_stats(subdomain_stats),
        "data_source": str(args.gpqa_data_dir),
        "configs": selected_gpqa_configs(args.gpqa_configs),
        "epochs": args.gpqa_epochs,
        "shuffle_choices": args.shuffle_gpqa_choices,
        "prompt_type": args.gpqa_prompt_type,
    }


def run_bbh(args: argparse.Namespace, llm: Any, model_name: str) -> dict[str, Any]:
    out_dir = benchmark_output_dir(args, "bbh", model_name)
    if is_benchmark_complete(args, "bbh", model_name):
        return read_json(out_dir / "summary.json")
    rows = shard_rows(load_bbh_rows(args), args)
    results = generate_rows(
        args=args,
        llm=llm,
        rows=rows,
        benchmark="bbh",
        prompt_builder=lambda row: build_bbh_prompt(args, row),
        output_parser=lambda row, text: extract_bbh_prediction(str(row["target"]), text),
        correct_fn=lambda row, pred: is_bbh_correct(pred, str(row["target"])),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", results)
    summary = summarize_bbh(model_name, results, args)
    write_json(out_dir / "summary.json", summary)
    pd.DataFrame(summary["by_task"]).T.to_csv(out_dir / "task_summary.csv")
    return summary


def run_gpqa(args: argparse.Namespace, llm: Any, model_name: str) -> dict[str, Any]:
    out_dir = benchmark_output_dir(args, "gpqa", model_name)
    if is_benchmark_complete(args, "gpqa", model_name):
        return read_json(out_dir / "summary.json")
    rows = shard_rows(load_gpqa_rows(args), args)
    results = generate_rows(
        args=args,
        llm=llm,
        rows=rows,
        benchmark="gpqa",
        prompt_builder=lambda row: build_gpqa_prompt(args, row),
        output_parser=lambda row, text: extract_gpqa_answer(text),
        correct_fn=lambda row, pred: pred == row["answer"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", results)
    summary = summarize_gpqa(model_name, results, args)
    write_json(out_dir / "summary.json", summary)
    pd.DataFrame(summary["by_config"]).T.to_csv(out_dir / "config_summary.csv")
    pd.DataFrame(summary["by_domain"]).T.to_csv(out_dir / "domain_summary.csv")
    return summary


def load_mmstar_eval(args: argparse.Namespace):
    mmstar_path = args.mmstar_eval_dir.resolve() / "vlmeval" / "evaluate" / "mmstar.py"
    if not mmstar_path.exists():
        raise FileNotFoundError(f"Missing MMStar evaluator: {mmstar_path}")

    smp_module = types.ModuleType("vlmeval.smp")
    smp_module.load = lambda path: pd.read_excel(path)
    smp_module.dump = lambda obj, path: write_json(Path(path), obj)
    smp_module.tqdm = tqdm
    smp_module.get_logger = lambda name: logging.getLogger(name)

    vlmeval_module = types.ModuleType("vlmeval")
    vlmeval_module.__path__ = []
    sys.modules["vlmeval"] = vlmeval_module
    sys.modules["vlmeval.smp"] = smp_module

    module_name = "_bench_coe_mmstar_eval"
    spec = importlib.util.spec_from_file_location(module_name, mmstar_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MMStar evaluator from {mmstar_path}")
    mmstar_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mmstar_module
    spec.loader.exec_module(mmstar_module)
    return mmstar_module


def run_mmstar_eval_on_dataframe(
    args: argparse.Namespace,
    result_df: pd.DataFrame,
    eval_file: Path,
) -> tuple[dict[str, Any], Path]:
    mmstar_module = load_mmstar_eval(args)
    score_file = eval_file.with_name(eval_file.stem + "_score.json")

    original_load = mmstar_module.load
    original_dump = mmstar_module.dump

    def patched_load(_: str) -> pd.DataFrame:
        return result_df.copy()

    def patched_dump(obj: Any, path: str) -> None:
        write_json(Path(path), obj)

    mmstar_module.load = patched_load
    mmstar_module.dump = patched_dump
    try:
        mmstar_module.MMStar_eval(str(eval_file))
    finally:
        mmstar_module.load = original_load
        mmstar_module.dump = original_dump

    return read_json(score_file), score_file


def summarize_mmstar_fallback(model_name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    category_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    l2_stats: dict[str, dict[str, float]] = defaultdict(stats_dict)
    correct = 0.0
    wrong = 0.0
    for row in rows:
        targets = [category_stats[row["category"]], l2_stats[f"{row['category']}({row['l2_category']})"]]
        if row.get("is_correct"):
            correct += 1
            for stats in targets:
                stats["correct"] += 1
        else:
            wrong += 1
            for stats in targets:
                stats["wrong"] += 1
    total = correct + wrong
    return {
        "status": "completed",
        "benchmark": "mmstar_text_only",
        "model": model_name,
        "num_examples": int(total),
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "by_category": finalize_stats(category_stats),
        "by_l2_category": finalize_stats(l2_stats),
        "data_source": str(args.mmstar_tsv),
        "evaluation": "fallback_letter_match",
    }


def run_mmstar(args: argparse.Namespace, llm: Any, model_name: str) -> dict[str, Any]:
    benchmark = "mmstar_text_only"
    out_dir = benchmark_output_dir(args, benchmark, model_name)
    if is_benchmark_complete(args, benchmark, model_name):
        return read_json(out_dir / "summary.json")
    rows = shard_rows(load_mmstar_rows(args), args)
    results = generate_rows(
        args=args,
        llm=llm,
        rows=rows,
        benchmark=benchmark,
        prompt_builder=build_mmstar_prompt,
        output_parser=lambda row, text: extract_choice_letter(text),
        correct_fn=lambda row, pred: pred == row["answer"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", results)

    xlsx_rows: list[dict[str, Any]] = []
    for row in results:
        prediction_value = row.get("pred") if args.mmstar_store_extracted_prediction else row.get("model_outputs", "")
        if not prediction_value:
            prediction_value = row.get("model_outputs", "")
        xlsx_rows.append(
            {
                "index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "l2_category": row["l2_category"],
                "bench": row["bench"],
                "prediction": prediction_value,
                "model_outputs": row.get("model_outputs", ""),
            }
        )
    result_df = pd.DataFrame(xlsx_rows)
    csv_file = out_dir / f"{model_name}_MMStar.csv"
    result_df.to_csv(csv_file, index=False, encoding="utf-8")
    result_file = out_dir / f"{model_name}_MMStar.xlsx"
    try:
        result_df.to_excel(result_file, index=False)
    except Exception:
        result_file = out_dir / f"{model_name}_MMStar.xlsx"

    summary = summarize_mmstar_fallback(model_name, results, args)
    try:
        score_payload, score_file = run_mmstar_eval_on_dataframe(args, result_df, result_file)
        summary["official_mmstar_result_file"] = str(csv_file)
        if result_file.exists():
            summary["official_mmstar_xlsx_file"] = str(result_file)
        summary["official_mmstar_score_file"] = str(score_file)
        summary["official_mmstar_scores"] = score_payload
        summary["accuracy"] = float(score_payload.get("final score", summary["accuracy"]))
        summary["evaluation"] = "MMStar_eval"
    except Exception as exc:
        summary["evaluation_warning"] = f"MMStar_eval failed; fallback scores kept: {exc}"
    write_json(out_dir / "summary.json", summary)
    pd.DataFrame(summary.get("by_category", {})).T.to_csv(out_dir / "category_summary.csv")
    return summary


def run_benchmark_for_model(args: argparse.Namespace, llm: Any, benchmark: str, model_name: str) -> dict[str, Any]:
    if benchmark == "bbh":
        return run_bbh(args, llm, model_name)
    if benchmark == "gpqa":
        return run_gpqa(args, llm, model_name)
    if benchmark == "mmstar_text_only":
        return run_mmstar(args, llm, model_name)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def path_fields() -> set[str]:
    return {
        "models_dir",
        "gaokao_data_dir",
        "output_dir",
        "bbh_data_dir",
        "bbh_cot_dir",
        "gpqa_data_dir",
        "mmstar_tsv",
        "mmstar_eval_dir",
        "worker_input",
        "worker_output",
    }


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def args_from_json(payload: dict[str, Any]) -> argparse.Namespace:
    converted = dict(payload)
    for key in path_fields():
        if key in converted and converted[key] is not None:
            converted[key] = Path(converted[key])
    return argparse.Namespace(**converted)


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise ValueError("--worker-input and --worker-output are required in worker mode.")
    payload = read_json(args.worker_input)
    worker_args = args_from_json(payload["args"])
    model_name = payload["model_name"]
    benchmarks = payload["benchmarks"]
    worker_result: dict[str, Any] = {
        "model": model_name,
        "benchmarks": {},
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        print(f"[worker] loading {model_name}", flush=True)
        llm = load_llm(worker_args, model_name)
        try:
            for benchmark in benchmarks:
                try:
                    print(f"[worker] {model_name}: running {benchmark}", flush=True)
                    summary = run_benchmark_for_model(worker_args, llm, benchmark, model_name)
                    worker_result["benchmarks"][benchmark] = {
                        "status": summary.get("status", "completed"),
                        "accuracy": summary.get("accuracy"),
                        "num_examples": summary.get("num_examples"),
                    }
                    print(
                        f"[worker] {model_name}: {benchmark} accuracy={summary.get('accuracy')} "
                        f"n={summary.get('num_examples')}",
                        flush=True,
                    )
                except Exception as exc:
                    traceback_text = traceback.format_exc()
                    worker_result["benchmarks"][benchmark] = {
                        "status": "failed",
                        "error": str(exc),
                    }
                    fail_dir = benchmark_output_dir(worker_args, benchmark, model_name)
                    fail_dir.mkdir(parents=True, exist_ok=True)
                    write_json(
                        fail_dir / "summary.json",
                        {
                            "status": "failed",
                            "benchmark": benchmark,
                            "model": model_name,
                            "error": str(exc),
                            "traceback": traceback_text,
                        },
                    )
                    print(traceback_text, flush=True)
        finally:
            del llm
            cleanup_vllm()
        failed = [
            benchmark
            for benchmark, info in worker_result["benchmarks"].items()
            if info.get("status") == "failed"
        ]
        worker_result["status"] = "failed" if failed else "completed"
    except Exception as exc:
        worker_result["status"] = "failed"
        worker_result["error"] = str(exc)
        worker_result["traceback"] = traceback.format_exc()
        print(worker_result["traceback"], flush=True)
    finally:
        worker_result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(args.worker_output, worker_result)
        if worker_result["status"] == "failed":
            raise SystemExit(1)


def discover_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        candidate_models = list(args.models)
    else:
        gaokao_result_models = discover_result_models(args.gaokao_data_dir)
        candidate_models = filter_local_models(gaokao_result_models, args.models_dir)
    if args.exclude_models:
        excluded = set(args.exclude_models)
        candidate_models = [model_name for model_name in candidate_models if model_name not in excluded]
    missing = [model_name for model_name in candidate_models if not (args.models_dir / model_name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Requested models are missing under {args.models_dir}: {missing}")
    return sorted(dict.fromkeys(candidate_models))


def pending_benchmarks_for_model(args: argparse.Namespace, model_name: str, benchmarks: list[str]) -> list[str]:
    return [
        benchmark
        for benchmark in benchmarks
        if not is_benchmark_complete(args, benchmark, model_name)
    ]


def make_worker_payload(args: argparse.Namespace, model_name: str, benchmarks: list[str]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "benchmarks": benchmarks,
        "args": args_to_json(args),
    }


def run_parent(args: argparse.Namespace) -> None:
    benchmarks = parse_csv_list(args.benchmarks, list(BENCHMARKS))
    model_names = discover_models(args)
    if not model_names:
        raise RuntimeError("No local GAOKAO-result models were found to evaluate.")

    gpu_list = [item.strip() for item in args.gpu_devices.split(",") if item.strip()]
    if not gpu_list:
        raise ValueError("--gpu-devices must contain at least one GPU id.")
    parallel_workers = args.parallel_workers or len(gpu_list)
    parallel_workers = min(parallel_workers, len(gpu_list))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmarks": benchmarks,
        "models": model_names,
        "gpu_devices": gpu_list,
        "parallel_workers": parallel_workers,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": args_to_json(args),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(args.output_dir / "model_list.json", model_names)

    jobs: list[tuple[str, list[str]]] = []
    for model_name in model_names:
        pending = pending_benchmarks_for_model(args, model_name, benchmarks)
        if pending:
            jobs.append((model_name, pending))
    if not jobs:
        print("All requested model/benchmark outputs are already completed.")
        return

    worker_root = args.output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    waves = [jobs[start : start + parallel_workers] for start in range(0, len(jobs), parallel_workers)]

    all_worker_results: dict[str, Any] = {}
    for wave_id, wave in enumerate(waves, start=1):
        print(f"Starting wave {wave_id}/{len(waves)} with {len(wave)} worker(s)", flush=True)
        processes: list[tuple[subprocess.Popen[Any], str, Path, Path, Any]] = []
        for (model_name, pending), gpu in zip(wave, gpu_list):
            safe_model = sanitize_name(model_name)
            input_path = worker_root / f"{safe_model}.input.json"
            output_path = worker_root / f"{safe_model}.output.json"
            log_path = worker_root / f"{safe_model}.log"
            write_json(input_path, make_worker_payload(args, model_name, pending))
            cmd = [
                sys.executable,
                "-m",
                "bench_coe.run_official_model_benchmarks",
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(output_path),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            vllm_source = Path(os.environ.get("BENCH_COE_VLLM_SOURCE", "/home/sm5/ys/Project/vllm"))
            if (vllm_source / "vllm" / "__init__.py").exists():
                env.setdefault("BENCH_COE_VLLM_SOURCE", str(vllm_source))
                old_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    str(vllm_source)
                    if not old_pythonpath
                    else str(vllm_source) + os.pathsep + old_pythonpath
                )
            cache_dir = worker_root / "cache" / safe_model
            cache_dir.mkdir(parents=True, exist_ok=True)
            env.setdefault("VLLM_CACHE_ROOT", str(cache_dir / "vllm"))
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "torchinductor"))
            log_file = log_path.open("w", encoding="utf-8")
            print(
                f"  GPU {gpu}: {model_name} -> {','.join(pending)}",
                flush=True,
            )
            process = subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, model_name, output_path, log_path, log_file))

        for process, model_name, output_path, log_path, log_file in processes:
            return_code = process.wait()
            log_file.close()
            if output_path.exists():
                all_worker_results[model_name] = read_json(output_path)
            else:
                all_worker_results[model_name] = {
                    "model": model_name,
                    "status": "failed",
                    "error": f"Worker exited with {return_code} before writing output.",
                }
            if return_code != 0:
                print(f"Worker failed for {model_name}; see {log_path}", flush=True)
            else:
                print(f"Worker completed for {model_name}; log {log_path}", flush=True)
        write_json(args.output_dir / "worker_results.json", all_worker_results)

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(args.output_dir / "worker_results.json", all_worker_results)
    print(f"Finished. Outputs are under {args.output_dir}", flush=True)


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args()
    if args.worker_input is not None:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
