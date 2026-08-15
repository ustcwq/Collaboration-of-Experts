from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import read_json, write_json


CHOICES = "ABCDEFGHIJ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Subject-Bert-Bench-CoE on local MMLU-Pro parquet files."
    )
    parser.add_argument(
        "--router-dir",
        type=Path,
        default=Path("outputs/bench_coe/router/bert-base-gaokao/model"),
    )
    parser.add_argument(
        "--route-label-manifest",
        type=Path,
        default=Path("outputs/bench_coe/router/bert-base-gaokao/route_label_manifest.json"),
    )
    parser.add_argument(
        "--subject-expert-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON override for subject-label routers. Accepts either "
            "{subject: model} or {'subject_to_model': {...}, 'subject_to_model_index': {...}}."
        ),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--mmlu-data-dir", type=Path, default=Path("MMLU-Pro/data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/mmlu_pro_subject_bert_bench_coe"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        choices=["validation", "test"],
    )
    parser.add_argument(
        "--selected-subjects",
        default="all",
        help="Comma-separated MMLU-Pro categories, or all.",
    )
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument(
        "--validation-ntrain",
        type=int,
        default=0,
        help="Few-shot examples for validation split. Default 0 avoids validation leakage.",
    )
    parser.add_argument("--router-max-length", type=int, default=256)
    parser.add_argument("--router-batch-size", type=int, default=128)
    parser.add_argument(
        "--router-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument(
        "--expert-gpus",
        default=None,
        help="Physical GPUs for expert-parallel workers. Defaults to --gpu-devices.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="0 means use the number of visible GPUs from --gpu-devices.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit per split.",
    )
    parser.add_argument(
        "--experts",
        nargs="*",
        default=None,
        help="Optional expert model names to run. Routed examples for other experts are left pending.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing split prediction JSON files and only run missing examples.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Store full prompts in prediction files. Disabled by default to keep outputs small.",
    )
    parser.add_argument(
        "--parallel-experts",
        action="store_true",
        help="Run routed expert shards concurrently across --expert-gpus.",
    )
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        options = [opt for opt in row["options"] if opt != "N/A"]
        row["options"] = options
        rows.append(row)
    return rows


def filter_subjects(rows: list[dict[str, Any]], selected_subjects: str) -> list[dict[str, Any]]:
    if selected_subjects == "all":
        return rows
    selected = {item.strip().replace("_", " ") for item in selected_subjects.split(",")}
    return [row for row in rows if row["category"].replace("_", " ") in selected]


def limit_rows(
    rows: list[dict[str, Any]],
    max_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return sorted(copied[:max_examples], key=lambda row: int(row["question_id"]))


def format_router_text(row: dict[str, Any]) -> str:
    option_text = "\n".join(
        f"{CHOICES[idx]}. {option}" for idx, option in enumerate(row["options"])
    )
    return f"Question:\n{row['question']}\nOptions:\n{option_text}"


def choose_router_device(router_device: str) -> torch.device:
    if router_device == "cpu":
        return torch.device("cpu")
    if router_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("router-device=cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_route_label(
    label_manifest: dict[str, Any],
    label: int,
) -> tuple[str | None, str, int]:
    label_key = str(label)
    if label_manifest.get("label_mode") == "subject":
        subject = label_manifest["route_label_to_subject"][label_key]
        subject_to_model = label_manifest.get("subject_to_model", {})
        model_name = subject_to_model.get(subject)
        if not model_name:
            raise KeyError(f"Missing subject_to_model entry for routed subject: {subject}")
        model_index_raw = label_manifest.get("subject_to_model_index", {}).get(subject)
        if model_index_raw is None:
            model_index_raw = label_manifest.get("route_label_to_model_index", {}).get(
                label_key, -1
            )
        return subject, model_name, int(model_index_raw)

    model_name = label_manifest["route_label_to_model"][label_key]
    model_index = int(label_manifest["route_label_to_model_index"][label_key])
    return None, model_name, model_index


def apply_subject_expert_map(
    label_manifest: dict[str, Any],
    override_path: Path | None,
) -> dict[str, Any]:
    if override_path is None:
        return label_manifest
    if label_manifest.get("label_mode") != "subject":
        raise ValueError("--subject-expert-map can only be used with subject-label routers.")

    override = read_json(override_path)
    subject_to_model = override.get("subject_to_model", override)
    subject_to_model_index = override.get("subject_to_model_index", {})
    if not isinstance(subject_to_model, dict):
        raise ValueError("--subject-expert-map must contain a subject->model mapping.")

    merged = dict(label_manifest)
    merged["subject_to_model"] = {
        **label_manifest.get("subject_to_model", {}),
        **subject_to_model,
    }
    merged_model_index = {
        **label_manifest.get("subject_to_model_index", {}),
        **subject_to_model_index,
    }
    original_subject_to_model = label_manifest.get("subject_to_model", {})
    for subject, model_name in subject_to_model.items():
        if (
            subject not in subject_to_model_index
            and original_subject_to_model.get(subject) != model_name
        ):
            merged_model_index[subject] = -1
    merged["subject_to_model_index"] = merged_model_index
    return merged


@torch.no_grad()
def route_rows(
    rows: list[dict[str, Any]],
    router_dir: Path,
    label_manifest: dict[str, Any],
    max_length: int,
    batch_size: int,
    router_device: str,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(router_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(router_dir))
    device = choose_router_device(router_device)
    model.to(device)
    model.eval()

    routed_rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="routing"):
        batch_rows = rows[start : start + batch_size]
        texts = [format_router_text(row) for row in batch_rows]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = logits.softmax(dim=-1)
        route_labels = probs.argmax(dim=-1).detach().cpu().tolist()
        route_probs = probs.max(dim=-1).values.detach().cpu().tolist()
        for row, label, confidence in zip(batch_rows, route_labels, route_probs):
            routed_subject, routed_model, routed_model_index = resolve_route_label(
                label_manifest, int(label)
            )
            routed = dict(row)
            routed["route_label"] = int(label)
            routed["route_confidence"] = float(confidence)
            if routed_subject is not None:
                routed["routed_subject"] = routed_subject
            routed["routed_model"] = routed_model
            routed["routed_model_index"] = routed_model_index
            routed_rows.append(routed)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return routed_rows


def make_val_bank(validation_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        bank[row["category"]].append(row)
    return bank


def format_mmlu_example(row: dict[str, Any], including_answer: bool) -> str:
    prompt = f"Question:\n{row['question']}\nOptions:\n"
    for idx, option in enumerate(row["options"]):
        prompt += f"{CHOICES[idx]}. {option}\n"
    if including_answer:
        cot_content = str(row.get("cot_content", "") or "")
        if cot_content.startswith("A: "):
            cot_content = cot_content[3:]
        if not cot_content:
            cot_content = f"The answer is ({row['answer']})."
        prompt += f"Answer: {cot_content}\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_prompt(
    row: dict[str, Any],
    val_bank: dict[str, list[dict[str, Any]]],
    ntrain: int,
) -> str:
    category = row["category"]
    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{category}. Think step by step and then output the answer in the "
        'format of "The answer is (X)" at the end.\n\n'
    )
    few_shot_examples = []
    for candidate in val_bank.get(category, []):
        if candidate["question_id"] == row["question_id"]:
            continue
        few_shot_examples.append(candidate)
        if len(few_shot_examples) >= ntrain:
            break
    for example in few_shot_examples:
        prompt += format_mmlu_example(example, including_answer=True)
    prompt += format_mmlu_example(row, including_answer=False)
    return prompt


def extract_answer(text: str) -> str | None:
    patterns = [
        r"answer is \(?([A-J])\)?",
        r"[aA]nswer:\s*\(?([A-J])\)?",
        r"答案(?:是|为)?\s*[（(]?([A-J])[）)]?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_json(path)
    return {int(row["question_id"]): row for row in rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}
    )
    model_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"correct": 0.0, "wrong": 0.0, "accuracy": 0.0}
    )
    total_correct = 0.0
    total_wrong = 0.0
    for row in rows:
        pred = row.get("pred")
        is_correct = pred == row["answer"]
        target_category = category_stats[row["category"]]
        target_model = model_stats[row["routed_model"]]
        if is_correct:
            total_correct += 1
            target_category["correct"] += 1
            target_model["correct"] += 1
        else:
            total_wrong += 1
            target_category["wrong"] += 1
            target_model["wrong"] += 1
    for stats in list(category_stats.values()) + list(model_stats.values()):
        denom = stats["correct"] + stats["wrong"]
        stats["accuracy"] = stats["correct"] / denom if denom else 0.0
    return {
        "total": {
            "correct": total_correct,
            "wrong": total_wrong,
            "accuracy": total_correct / (total_correct + total_wrong)
            if total_correct + total_wrong
            else 0.0,
        },
        "category": dict(sorted(category_stats.items())),
        "routed_model": dict(sorted(model_stats.items())),
    }


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


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def split_evenly(rows: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    if parts <= 1:
        return [rows]
    chunks = [[] for _ in range(parts)]
    for idx, row in enumerate(rows):
        chunks[idx % parts].append(row)
    return [chunk for chunk in chunks if chunk]


def load_llm(args: argparse.Namespace, model_path: Path):
    from vllm import LLM, SamplingParams

    visible_devices = [item for item in args.gpu_devices.split(",") if item.strip()]
    tp_size = args.tensor_parallel_size or max(1, len(visible_devices))
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["Question:"],
    )
    return llm, sampling_params


def generate_for_expert(
    args: argparse.Namespace,
    model_name: str,
    rows: list[dict[str, Any]],
    val_bank: dict[str, list[dict[str, Any]]],
    ntrain: int,
) -> list[dict[str, Any]]:
    model_path = args.models_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing expert model path: {model_path}")

    print(f"Loading expert {model_name} from {model_path}")
    llm, sampling_params = load_llm(args, model_path)
    completed: list[dict[str, Any]] = []
    start_time = time.time()
    for start in tqdm(range(0, len(rows), args.batch_size), desc=model_name):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [build_prompt(row, val_bank, ntrain) for row in batch_rows]
        outputs = llm.generate(prompts, sampling_params)
        for row, prompt, output in zip(batch_rows, prompts, outputs):
            generated_text = output.outputs[0].text
            result = dict(row)
            if args.save_prompts:
                result["prompt"] = prompt
            result["pred"] = extract_answer(generated_text)
            result["model_outputs"] = generated_text
            completed.append(result)
    print(f"Finished {model_name}: {len(completed)} examples in {time.time() - start_time:.1f}s")
    del llm
    cleanup_vllm()
    return completed


def plan_parallel_waves(
    pending_by_model: dict[str, list[dict[str, Any]]],
    expert_gpus: list[str],
) -> list[list[tuple[str, str, int, list[dict[str, Any]]]]]:
    models_by_size = sorted(
        pending_by_model,
        key=lambda model_name: len(pending_by_model[model_name]),
        reverse=True,
    )
    if not models_by_size:
        return []
    if not expert_gpus:
        raise ValueError("No expert GPUs were provided.")

    if len(models_by_size) <= len(expert_gpus):
        assignments: dict[str, list[str]] = {model_name: [] for model_name in models_by_size}
        gpu_iter = iter(expert_gpus)
        for model_name in models_by_size:
            assignments[model_name].append(next(gpu_iter))
        remaining_gpus = list(gpu_iter)
        for gpu in remaining_gpus:
            model_name = max(
                models_by_size,
                key=lambda name: len(pending_by_model[name]) / (len(assignments[name]) + 1),
            )
            assignments[model_name].append(gpu)

        jobs: list[tuple[str, str, int, list[dict[str, Any]]]] = []
        for model_name in models_by_size:
            gpu_list = assignments[model_name]
            for shard_id, (gpu, chunk) in enumerate(
                zip(gpu_list, split_evenly(pending_by_model[model_name], len(gpu_list)))
            ):
                jobs.append((model_name, gpu, shard_id, chunk))
        return [jobs]

    waves: list[list[tuple[str, str, int, list[dict[str, Any]]]]] = []
    for offset in range(0, len(models_by_size), len(expert_gpus)):
        wave: list[tuple[str, str, int, list[dict[str, Any]]]] = []
        for gpu, model_name in zip(expert_gpus, models_by_size[offset : offset + len(expert_gpus)]):
            wave.append((model_name, gpu, 0, pending_by_model[model_name]))
        waves.append(wave)
    return waves


def worker_payload_path(
    output_dir: Path,
    split: str,
    model_name: str,
    shard_id: int,
    suffix: str,
) -> Path:
    return (
        output_dir
        / "workers"
        / f"{split}_{sanitize_name(model_name)}_shard{shard_id}.{suffix}"
    )


def run_parallel_experts(
    args: argparse.Namespace,
    split: str,
    pending_by_model: dict[str, list[dict[str, Any]]],
    validation_rows: list[dict[str, Any]],
    ntrain: int,
) -> list[dict[str, Any]]:
    expert_gpus = args.expert_gpus or args.gpu_devices
    gpu_list = [gpu.strip() for gpu in expert_gpus.split(",") if gpu.strip()]
    waves = plan_parallel_waves(pending_by_model, gpu_list)
    if not waves:
        return []

    (args.output_dir / "workers").mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict[str, Any]] = []
    for wave_id, jobs in enumerate(waves):
        print(f"Starting expert wave {wave_id + 1}/{len(waves)} with {len(jobs)} worker(s)")
        processes: list[tuple[subprocess.Popen[Any], Path, Path]] = []
        for model_name, gpu, shard_id, rows in jobs:
            input_path = worker_payload_path(
                args.output_dir, split, model_name, shard_id, "input.json"
            )
            output_path = worker_payload_path(
                args.output_dir, split, model_name, shard_id, "output.json"
            )
            log_path = worker_payload_path(args.output_dir, split, model_name, shard_id, "log")
            write_json(
                input_path,
                {
                    "model_name": model_name,
                    "rows": rows,
                    "validation_rows": validation_rows,
                    "ntrain": ntrain,
                },
            )
            cmd = [
                sys.executable,
                "-m",
                "bench_coe.evaluate_subject_bert_bench_coe",
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(output_path),
                "--models-dir",
                str(args.models_dir),
                "--gpu-devices",
                gpu,
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-model-len",
                str(args.max_model_len),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--dtype",
                str(args.dtype),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.save_prompts:
                cmd.append("--save-prompts")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            cache_dir = (
                args.output_dir
                / "workers"
                / "cache"
                / f"{split}_{sanitize_name(model_name)}_shard{shard_id}"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            env.setdefault("VLLM_CACHE_ROOT", str(cache_dir / "vllm"))
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "torchinductor"))
            log_file = log_path.open("w", encoding="utf-8")
            print(
                f"Launching {model_name} shard {shard_id} on GPU {gpu}: "
                f"{len(rows)} examples"
            )
            proc = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((proc, output_path, log_path))

        failures: list[tuple[int, Path]] = []
        for proc, output_path, log_path in processes:
            return_code = proc.wait()
            if return_code != 0:
                failures.append((return_code, log_path))
                continue
            generated_rows.extend(read_json(output_path))

        if failures:
            messages = []
            for return_code, log_path in failures:
                tail = ""
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = "\n".join(lines[-80:])
                messages.append(f"{log_path} exited {return_code}\n{tail}")
            raise RuntimeError("Parallel expert worker failed:\n" + "\n\n".join(messages))
    return generated_rows


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise ValueError("--worker-input and --worker-output are required in worker mode.")
    payload = read_json(args.worker_input)
    val_bank = make_val_bank(payload["validation_rows"])
    rows = generate_for_expert(
        args,
        payload["model_name"],
        payload["rows"],
        val_bank,
        int(payload["ntrain"]),
    )
    write_json(args.worker_output, rows)


def evaluate_split(
    args: argparse.Namespace,
    split: str,
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    label_manifest: dict[str, Any],
) -> dict[str, Any]:
    output_path = args.output_dir / f"{split}_predictions.json"
    existing = load_existing(output_path) if args.resume else {}
    routed_rows = route_rows(
        rows,
        args.router_dir,
        label_manifest,
        args.router_max_length,
        args.router_batch_size,
        args.router_device,
    )
    val_bank = make_val_bank(validation_rows)
    split_ntrain = args.validation_ntrain if split == "validation" else args.ntrain

    done_rows = dict(existing)
    pending_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expert_filter = set(args.experts) if args.experts else None
    for row in routed_rows:
        qid = int(row["question_id"])
        if qid in done_rows and done_rows[qid].get("model_outputs"):
            continue
        if expert_filter is not None and row["routed_model"] not in expert_filter:
            row = dict(row)
            row["pred"] = None
            row["model_outputs"] = None
            row["pending_reason"] = "expert not selected in --experts"
            done_rows[qid] = row
            continue
        pending_by_model[row["routed_model"]].append(row)

    if args.parallel_experts:
        generated_rows = run_parallel_experts(
            args, split, pending_by_model, validation_rows, split_ntrain
        )
        for row in generated_rows:
            done_rows[int(row["question_id"])] = row
        write_json(output_path, [done_rows[key] for key in sorted(done_rows)])
    else:
        for model_name in sorted(pending_by_model):
            generated_rows = generate_for_expert(
                args, model_name, pending_by_model[model_name], val_bank, split_ntrain
            )
            for row in generated_rows:
                done_rows[int(row["question_id"])] = row
            write_json(output_path, [done_rows[key] for key in sorted(done_rows)])

    final_rows = [done_rows[int(row["question_id"])] for row in routed_rows]
    summary = summarize(final_rows)
    summary["split"] = split
    summary["examples"] = len(final_rows)
    summary["ntrain"] = split_ntrain
    summary["prediction_file"] = str(output_path)
    write_json(output_path, final_rows)
    write_json(args.output_dir / f"{split}_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_input is not None:
        run_worker(args)
        return

    label_manifest = apply_subject_expert_map(
        read_json(args.route_label_manifest), args.subject_expert_map
    )
    validation_rows_all = load_split(args.mmlu_data_dir, "validation")
    validation_rows_all = filter_subjects(validation_rows_all, args.selected_subjects)

    run_summaries = {}
    for split in args.splits:
        rows = load_split(args.mmlu_data_dir, split)
        rows = filter_subjects(rows, args.selected_subjects)
        rows = limit_rows(rows, args.max_examples_per_split, args.seed)
        print(f"Evaluating split={split}, examples={len(rows)}")
        run_summaries[split] = evaluate_split(
            args, split, rows, validation_rows_all, label_manifest
        )

    run_manifest = {
        "router_dir": str(args.router_dir),
        "route_label_manifest": str(args.route_label_manifest),
        "subject_expert_map": str(args.subject_expert_map)
        if args.subject_expert_map
        else None,
        "models_dir": str(args.models_dir),
        "mmlu_data_dir": str(args.mmlu_data_dir),
        "splits": args.splits,
        "selected_subjects": args.selected_subjects,
        "gpu_devices": args.gpu_devices,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_examples_per_split": args.max_examples_per_split,
        "summaries": run_summaries,
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
