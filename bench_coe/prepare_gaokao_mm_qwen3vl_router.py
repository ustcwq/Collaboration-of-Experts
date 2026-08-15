from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench_coe.gaokao_utils import dump_jsonl, read_json, write_json


SUBJECT_ORDER = [
    "Math",
    "Chinese",
    "Physics",
    "Chemistry",
    "Biology",
    "History",
    "Geography",
    "Politics",
]

FILE_SUBJECTS = {
    "2010-2023_Math_MCQs": "Math",
    "2010-2023_Chinese_Pratical_Lit": "Chinese",
    "2010-2023_Physics_MCQs": "Physics",
    "2010-2023_Chemistry_MCQs": "Chemistry",
    "2010-2023_Biology_MCQs": "Biology",
    "2010-2023_History_MCQs": "History",
    "2010-2023_Geography_MCQs": "Geography",
    "2010-2023_Political_Science_MCQs": "Politics",
}

DEFAULT_SUBJECT_TO_MODEL = {
    "Math": "Qwen2.5-7B-Instruct",
    "Chinese": "glm-4-9b-chat",
    "Physics": "glm-4-9b-chat",
    "Chemistry": "glm-4-9b-chat",
    "Biology": "General-Reasoner-7B-preview",
    "History": "glm-4-9b-chat",
    "Geography": "Qwen2.5-7B-Instruct",
    "Politics": "glm-4-9b-chat",
}

DEFAULT_SUBJECT_TO_MODEL_INDEX = {
    "Math": 7,
    "Chinese": 12,
    "Physics": 12,
    "Chemistry": 12,
    "Biology": 2,
    "History": 12,
    "Geography": 7,
    "Politics": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GAOKAO-MM subject-router data and LlamaFactory configs for Qwen3-VL."
    )
    parser.add_argument("--gaokao-mm-data-dir", type=Path, default=Path("GAOKAO-MM/Data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models_v/Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/router/qwen3vl-2b-gaokao-mm-subject-lora"),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    return parser.parse_args()


def route_prompt(question: str, image_count: int) -> str:
    image_tokens = "<image>" * image_count
    if image_tokens:
        image_tokens += "\n"
    labels = ", ".join(SUBJECT_ORDER)
    return (
        f"{image_tokens}"
        "Classify this GAOKAO-MM problem into exactly one subject.\n"
        f"Allowed subjects: {labels}.\n"
        "Return only the English subject label.\n\n"
        f"Problem:\n{question.strip()}"
    )


def resolve_image_paths(data_dir: Path, raw_paths: list[Any]) -> list[str]:
    resolved: list[str] = []
    for raw in raw_paths:
        if raw is None:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing GAOKAO-MM image: {path}")
        resolved.append(str(path))
    return resolved


def load_samples(data_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for json_path in sorted(data_dir.glob("*.json")):
        payload = read_json(json_path)
        keyword = str(payload.get("keywords") or json_path.stem)
        subject = FILE_SUBJECTS.get(keyword)
        if subject is None:
            raise KeyError(f"Unknown GAOKAO-MM keyword: {keyword}")
        for item in payload.get("example", []):
            images = resolve_image_paths(data_dir, list(item.get("picture") or []))
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            sample_id = f"{keyword}:{item.get('index', len(samples))}"
            samples.append(
                {
                    "id": sample_id,
                    "keyword": keyword,
                    "subject": subject,
                    "question": question,
                    "images": images,
                    "year": item.get("year"),
                    "index": item.get("index"),
                }
            )
    samples.sort(key=lambda row: (SUBJECT_ORDER.index(row["subject"]), str(row["id"])))
    return samples


def stratified_split(
    samples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_subject[sample["subject"]].append(sample)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for subject in SUBJECT_ORDER:
        rows = list(by_subject.get(subject, []))
        rng.shuffle(rows)
        if not rows:
            continue
        val_count = max(1, round(len(rows) * val_ratio))
        val.extend(rows[:val_count])
        train.extend(rows[val_count:])

    train.sort(key=lambda row: (SUBJECT_ORDER.index(row["subject"]), str(row["id"])))
    val.sort(key=lambda row: (SUBJECT_ORDER.index(row["subject"]), str(row["id"])))
    return train, val


def to_llamafactory_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": route_prompt(sample["question"], len(sample["images"])),
            },
            {
                "role": "assistant",
                "content": sample["subject"],
            },
        ],
        "images": sample["images"],
    }


def to_qwenvl_row(sample: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "conversations": [
            {
                "from": "human",
                "value": route_prompt(sample["question"], len(sample["images"])),
            },
            {
                "from": "gpt",
                "value": sample["subject"],
            },
        ],
    }
    if sample["images"]:
        row["image"] = sample["images"]
    return row


def write_llamafactory_data(output_dir: Path, train: list[dict[str, Any]], val: list[dict[str, Any]]) -> Path:
    dataset_dir = output_dir / "llamafactory_data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(dataset_dir / "gaokao_mm_router_train.json", [to_llamafactory_row(row) for row in train])
    write_json(dataset_dir / "gaokao_mm_router_val.json", [to_llamafactory_row(row) for row in val])
    write_json(
        dataset_dir / "dataset_info.json",
        {
            "gaokao_mm_router_train": {
                "file_name": "gaokao_mm_router_train.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            },
            "gaokao_mm_router_val": {
                "file_name": "gaokao_mm_router_val.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            },
        },
    )
    return dataset_dir


def write_qwenvl_data(output_dir: Path, train: list[dict[str, Any]], val: list[dict[str, Any]]) -> Path:
    dataset_dir = output_dir / "qwenvl_data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(dataset_dir / "gaokao_mm_router_train_qwenvl.json", [to_qwenvl_row(row) for row in train])
    write_json(dataset_dir / "gaokao_mm_router_val_qwenvl.json", [to_qwenvl_row(row) for row in val])
    return dataset_dir


def build_route_manifest() -> dict[str, Any]:
    subject_to_route_label = {subject: idx for idx, subject in enumerate(SUBJECT_ORDER)}
    route_label_to_subject = {str(idx): subject for subject, idx in subject_to_route_label.items()}
    route_label_to_model = {
        str(idx): DEFAULT_SUBJECT_TO_MODEL[subject]
        for subject, idx in subject_to_route_label.items()
    }
    route_label_to_model_index = {
        str(idx): DEFAULT_SUBJECT_TO_MODEL_INDEX[subject]
        for subject, idx in subject_to_route_label.items()
    }
    return {
        "label_mode": "gaokao_mm_subject",
        "router_type": "qwen3vl_sft_lora",
        "num_route_labels": len(SUBJECT_ORDER),
        "subject_to_route_label": subject_to_route_label,
        "route_label_to_subject": route_label_to_subject,
        "subject_to_model": DEFAULT_SUBJECT_TO_MODEL,
        "subject_to_model_index": DEFAULT_SUBJECT_TO_MODEL_INDEX,
        "route_label_to_model": route_label_to_model,
        "route_label_to_model_index": route_label_to_model_index,
        "note": "GAOKAO-MM has no English subject; downstream routing is restricted to these 8 subjects.",
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in payload.items():
        if value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        else:
            text = str(value)
            if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ","]) or text.strip() != text:
                lines.append(f"{key}: {json.dumps(text, ensure_ascii=False)}")
            else:
                lines.append(f"{key}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_configs(args: argparse.Namespace, dataset_dir: Path) -> dict[str, str]:
    model_path = args.model_path.resolve()
    adapter_dir = (args.output_dir / "adapter").resolve()
    train_config = args.output_dir / "train_qwen3vl_router_lora.yaml"
    infer_config = args.output_dir / "infer_qwen3vl_router_lora.yaml"
    merge_config = args.output_dir / "merge_qwen3vl_router_lora.yaml"

    write_yaml(
        train_config,
        {
            "model_name_or_path": model_path,
            "image_max_pixels": args.image_max_pixels,
            "video_max_pixels": 16384,
            "trust_remote_code": True,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target": "all",
            "dataset_dir": dataset_dir.resolve(),
            "dataset": "gaokao_mm_router_train",
            "eval_dataset": "gaokao_mm_router_val",
            "template": "qwen3_vl_nothink",
            "cutoff_len": args.cutoff_len,
            "max_samples": 100000,
            "preprocessing_num_workers": 8,
            "dataloader_num_workers": 2,
            "output_dir": adapter_dir,
            "logging_steps": 5,
            "save_steps": 200,
            "save_total_limit": 2,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": "none",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": True,
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": None,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "epoch",
        },
    )
    write_yaml(
        infer_config,
        {
            "model_name_or_path": model_path,
            "adapter_name_or_path": adapter_dir,
            "template": "qwen3_vl_nothink",
            "infer_backend": "huggingface",
            "trust_remote_code": True,
            "finetuning_type": "lora",
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 8,
        },
    )
    write_yaml(
        merge_config,
        {
            "model_name_or_path": model_path,
            "adapter_name_or_path": adapter_dir,
            "template": "qwen3_vl_nothink",
            "trust_remote_code": True,
            "export_dir": (args.output_dir / "merged").resolve(),
            "export_size": 5,
            "export_device": "cpu",
            "export_legacy_format": False,
        },
    )
    return {
        "train_config": str(train_config),
        "infer_config": str(infer_config),
        "merge_config": str(merge_config),
        "adapter_dir": str(adapter_dir),
    }


def main() -> None:
    args = parse_args()
    if not args.gaokao_mm_data_dir.exists():
        raise FileNotFoundError(args.gaokao_mm_data_dir)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.gaokao_mm_data_dir)
    train, val = stratified_split(samples, args.val_ratio, args.seed)
    dataset_dir = write_llamafactory_data(args.output_dir, train, val)
    qwenvl_dataset_dir = write_qwenvl_data(args.output_dir, train, val)
    configs = write_configs(args, dataset_dir)
    manifest = build_route_manifest()
    write_json(args.output_dir / "route_label_manifest.json", manifest)
    dump_jsonl(args.output_dir / "gaokao_mm_router_samples.jsonl", samples)

    train_counts = Counter(row["subject"] for row in train)
    val_counts = Counter(row["subject"] for row in val)
    sample_manifest = {
        "gaokao_mm_data_dir": str(args.gaokao_mm_data_dir),
        "model_path": str(args.model_path.resolve()),
        "output_dir": str(args.output_dir),
        "subjects": SUBJECT_ORDER,
        "total_samples": len(samples),
        "train_samples": len(train),
        "validation_samples": len(val),
        "train_counts": dict(train_counts),
        "validation_counts": dict(val_counts),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "qwenvl_train_file": str(qwenvl_dataset_dir / "gaokao_mm_router_train_qwenvl.json"),
        "qwenvl_validation_file": str(qwenvl_dataset_dir / "gaokao_mm_router_val_qwenvl.json"),
        **configs,
    }
    write_json(args.output_dir / "training_manifest.json", sample_manifest)
    print(json.dumps(sample_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
