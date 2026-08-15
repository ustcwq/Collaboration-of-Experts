from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = {
    "gaokao_2010_2022": (Path("GAOKAO-Bench-2010-2022"), ("Obj_Prompt.json",)),
    "gaokao_2023_2024": (Path("GAOKAO-Bench-2023-2024"), ("2023_Obj_Prompt.json", "2024_Obj_Prompt.json")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched local vLLM smoke evaluation on GAOKAO text benchmarks")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples-per-task", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def prompt_configs(root: Path, names: tuple[str, ...]) -> dict[str, dict[str, str]]:
    result = {}
    for name in names:
        payload = json.loads((root / "Bench" / name).read_text(encoding="utf-8"))
        for item in payload["examples"]:
            result[item["keyword"]] = item
    return result


def stable_rows(root: Path, configs: dict[str, dict[str, str]], limit: int) -> list[dict[str, Any]]:
    rows = []
    for keyword, config in sorted(configs.items()):
        source = root / "Data" / f"Bench-Harness_{keyword}.json"
        if not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        examples = sorted(
            payload.get("example", []),
            key=lambda item: (str(item.get("year", "")), int(item.get("index", 0))),
        )
        if limit > 0:
            examples = examples[:limit]
        for item in examples:
            rows.append({
                "keyword": keyword,
                "question_type": config["type"],
                "prefix_prompt": config["prefix_prompt"],
                "source_file": str(source),
                **{key: value for key, value in item.items() if key not in {"model_answer", "model_output"}},
            })
    return rows


def extract_answer(text: str, question_type: str, answer_length: int) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    marker = compact.rfind("【答案】")
    tail = compact[marker:] if marker >= 0 else compact[-80:]
    letters = re.findall(r"[A-G]", tail)
    if question_type == "single_choice":
        return letters[-1:] if letters else []
    if question_type == "multi_choice":
        return ["".join(dict.fromkeys(letter for letter in letters if letter <= "D"))] if letters else []
    return letters[:answer_length]


def main() -> int:
    args = parse_args()
    root, prompt_names = DATASETS[args.dataset]
    model_path = args.model_path.resolve()
    model_name = args.model_name or model_path.name
    rows = stable_rows(root, prompt_configs(root, prompt_names), args.max_examples_per_task)
    if not rows:
        raise SystemExit(f"No GAOKAO rows found for {args.dataset}")

    output_dir = args.output_dir / args.dataset / model_name
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"
    if args.resume and summary_path.is_file() and predictions_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            prediction_count = sum(1 for line in predictions_path.open(encoding="utf-8") if line.strip())
            if int(summary.get("total", -1)) == len(rows) and prediction_count == len(rows):
                print(json.dumps({"status": "skipped_existing", **summary}, ensure_ascii=False, indent=2))
                return 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    if args.backend == "vllm":
        from bench_coe.run_official_model_benchmarks import import_vllm_objects

        LLM, SamplingParams = import_vllm_objects("LLM", "SamplingParams")
        llm = LLM(
            model=str(model_path),
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
        )
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    else:
        from bench_coe.transformers_text_backend import TransformersSamplingParams, TransformersTextLLM

        llm = TransformersTextLLM(
            str(model_path),
            args.gpu_id,
            args.max_model_len,
            args.attn_implementation,
        )
        tokenizer = llm.get_tokenizer()
        sampling = TransformersSamplingParams(max_tokens=args.max_new_tokens)
    predictions = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = []
        for row in batch:
            text = row["prefix_prompt"] + "\n" + str(row["question"]).strip()
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        outputs = llm.generate(prompts, sampling)
        for row, output in zip(batch, outputs):
            generated = output.outputs[0].text.strip()
            gold = [str(item) for item in row.get("standard_answer", row.get("answer", []))]
            prediction = extract_answer(generated, row["question_type"], len(gold))
            predictions.append({**row, "model_name": model_name, "model_answer": prediction, "model_output": generated, "is_correct": prediction == gold})

    task_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in predictions:
        task_stats[row["keyword"]]["total"] += 1
        task_stats[row["keyword"]]["correct"] += int(row["is_correct"])
    correct = sum(item["correct"] for item in task_stats.values())
    total = sum(item["total"] for item in task_stats.values())
    summary = {
        "dataset": args.dataset,
        "model": model_name,
        "sample_policy": (
            f"first {args.max_examples_per_task} rows per official objective task, ordered by year/index"
            if args.max_examples_per_task > 0
            else "all rows from every official objective task, ordered by year/index"
        ),
        "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "tasks": {key: {**value, "accuracy": value["correct"] / value["total"] if value["total"] else 0.0} for key, value in sorted(task_stats.items())},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
