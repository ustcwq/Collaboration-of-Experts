from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    manifest_sha256,
    sha256_file,
    validate_test_receipt,
    write_json,
    write_jsonl,
)
from .data import FORBIDDEN_OBSERVABLE_ROW_KEYS
from .gpqa_long_reasoning import (
    extract_explicit_answer,
    read_source_summaries,
    source_best_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-free GPQA long reasoning on one shard")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--max-questions", type=int)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt(row: dict[str, Any], token_budget: int) -> str:
    options = list(row["options"])
    rendered = "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options))
    return (
        "Solve this graduate-level multiple-choice problem carefully. Reason independently, "
        f"but keep the reasoning within {token_budget} tokens. End with exactly one line in the "
        "form `Answer: (A)`, `Answer: (B)`, `Answer: (C)`, or `Answer: (D)`.\n\n"
        f"Question:\n{row['question']}\n\nOptions:\n{rendered}\n"
    )


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _finalizer_prompt(tokenizer: Any, prompt: str, reasoning: str) -> str:
    instruction = (
        "Use the reasoning above to choose the single best option. Do not add new reasoning. "
        "Reply with exactly one of the allowed answer strings."
    )
    if not getattr(tokenizer, "chat_template", None):
        return f"{prompt}\n\nReasoning:\n{reasoning}\n\n{instruction}"
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": instruction},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    if args.shard_count != int(config["generation"]["shard_count"]):
        raise ValueError("CLI shard count differs from frozen config")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    source = config["source_selection"]
    source_root = Path(source["summary_root"])
    source_summary_paths = sorted(source_root.glob("*/summary_validation.json"))
    selected_model, source_accuracy = source_best_model(read_source_summaries(source_root))
    if selected_model != str(source["selected_model"]):
        raise RuntimeError(
            f"Frozen model {source['selected_model']} is not source-best ({selected_model})"
        )
    selected_summary = source_root / selected_model / "summary_validation.json"
    if sha256_file(selected_summary) != str(source["selected_summary_sha256"]):
        raise PermissionError("Selected source summary hash mismatch")
    if abs(source_accuracy - float(source["selected_source_accuracy"])) > 1e-12:
        raise RuntimeError("Selected source accuracy differs from frozen config")

    target = config["target_observables"]
    observable_root = Path(target["cache_path"])
    observable_manifest = observable_root / "observable_manifest.json"
    if sha256_file(observable_manifest) != str(target["observable_manifest_sha256"]):
        raise PermissionError("GPQA observable manifest hash mismatch")
    observable_path = observable_root / str(target["metadata_expert_id"]) / "observables.jsonl"
    rows = _read_jsonl(observable_path)
    for row in rows:
        leaked = FORBIDDEN_OBSERVABLE_ROW_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"GPQA observable row contains forbidden fields: {sorted(leaked)}")
    epoch_zero = sorted(
        (row for row in rows if int(row.get("epoch", -1)) == 0),
        key=lambda row: str(row["id"]),
    )
    if len(epoch_zero) != int(target["expected_epoch0_questions"]):
        raise RuntimeError("Unexpected GPQA epoch-0 question count")
    if args.max_questions is not None:
        epoch_zero = epoch_zero[: args.max_questions]
    shard_rows = [
        row for index, row in enumerate(epoch_zero) if index % args.shard_count == args.shard_index
    ]

    generation = config["generation"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from bench_coe.run_official_model_benchmarks import import_vllm_objects

    LLM, SamplingParams = import_vllm_objects("LLM", "SamplingParams")
    model_path = Path(config["model_path"])
    weight_hashes: dict[str, str] = {}
    for name, expected_digest in config.get("model_weight_sha256", {}).items():
        weight_path = model_path / str(name)
        actual_digest = sha256_file(weight_path)
        if actual_digest != str(expected_digest):
            raise PermissionError(f"Model weight hash mismatch: {weight_path}")
        weight_hashes[str(weight_path)] = actual_digest
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
        max_model_len=int(generation["max_model_len"]),
        trust_remote_code=True,
        dtype=str(generation["dtype"]),
        enforce_eager=bool(generation["enforce_eager"]),
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(config["protocol_seed"]),
    )
    finalizer_choices = [str(value) for value in generation.get("finalizer_choices", [])]
    finalizer_sampling = None
    if finalizer_choices:
        from vllm.sampling_params import GuidedDecodingParams

        finalizer_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(generation["finalizer_max_tokens"]),
            seed=int(config["protocol_seed"]),
            guided_decoding=GuidedDecodingParams(choice=finalizer_choices),
        )
    batch_size = int(generation["batch_size"])
    result: list[dict[str, Any]] = []
    for start in range(0, len(shard_rows), batch_size):
        batch = shard_rows[start : start + batch_size]
        prompts = [
            _apply_chat_template(
                tokenizer,
                _prompt(row, int(generation["reasoning_token_budget"])),
            )
            for row in batch
        ]
        outputs = llm.generate(prompts, sampling)
        reasonings = [output.outputs[0].text for output in outputs]
        if finalizer_sampling is not None:
            finalizer_prompts = [
                _finalizer_prompt(
                    tokenizer,
                    _prompt(row, int(generation["reasoning_token_budget"])),
                    reasoning,
                )
                for row, reasoning in zip(batch, reasonings)
            ]
            finalized = llm.generate(finalizer_prompts, finalizer_sampling)
            final_responses = [output.outputs[0].text for output in finalized]
        else:
            final_responses = reasonings
        for row, reasoning, final_response in zip(batch, reasonings, final_responses):
            response = reasoning if final_response == reasoning else (
                f"{reasoning}\n\n[FINALIZER]\n{final_response}"
            )
            result.append(
                {
                    "id": str(row["id"]),
                    "base_question_id": row["base_question_id"],
                    "config": str(row["config"]),
                    "epoch": int(row["epoch"]),
                    "record_id": str(row["record_id"]),
                    "domain": str(row.get("domain", "")),
                    "subdomain": str(row.get("subdomain", "")),
                    "question": str(row["question"]),
                    "options": list(row["options"]),
                    "prediction": extract_explicit_answer(final_response),
                    "reasoning_response": reasoning,
                    "finalizer_response": final_response,
                    "response": response,
                }
            )

    prediction_path = args.output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, result)
    model_inputs = [
        model_path / name
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if (model_path / name).exists()
    ]
    receipt_path = Path(config["test_receipt"])
    input_paths = [
        args.config,
        receipt_path,
        observable_manifest,
        observable_path,
        *source_summary_paths,
        *model_inputs,
    ]
    input_hashes = files_manifest(
        [
            *input_paths,
        ]
    )
    input_hashes.update(weight_hashes)
    input_hashes = dict(sorted(input_hashes.items()))
    environment = environment_manifest(
        sys.argv, int(config["protocol_seed"]), input_paths
    )
    environment["input_hashes"] = input_hashes
    environment["input_manifest_sha256"] = manifest_sha256(input_hashes)
    manifest = {
        "protocol": config["protocol_name"],
        "model": selected_model,
        "source_accuracy": source_accuracy,
        "source_candidates": len(source_summary_paths),
        "physical_gpu": args.gpu_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "questions": len(result),
        "valid_explicit_predictions": sum(row["prediction"] is not None for row in result),
        "guided_finalizer_enabled": bool(finalizer_choices),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "input_hashes": input_hashes,
        "input_manifest_sha256": manifest_sha256(input_hashes),
        "environment": environment,
        "target_labels_opened": False,
        "runtime_seconds": time.time() - started,
        "command": sys.argv,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
