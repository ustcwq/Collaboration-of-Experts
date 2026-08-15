from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(items, **_: Any):
        return items

from bench_coe.run_gaokao_mm_babyvision_models import (
    discover_babyvision_models,
    ensure_combined_image,
    read_json,
    sanitize_name,
    write_json,
)


BENCHMARKS = ("cmmmu", "mmmu", "mathvista", "mmmu_pro")
CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_FIELD_RE = re.compile(r"^image_(\d+)$")
ANSWER_MARKER_RE = re.compile(
    r"(?:final answer|answer|答案|正确答案|最终答案)\s*(?:is|:|：|为)?\s*",
    re.IGNORECASE,
)

MMMU_DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BabyVision-tested local VLMs on local CMMMU/MMMU/MathVista/MMMU_Pro "
            "data using the BabyVision Transformers backend."
        )
    )
    parser.add_argument("--benchmarks", default="all", help="Comma list: cmmmu,mmmu,mathvista,mmmu_pro, or all.")
    parser.add_argument("--babyvision-dir", type=Path, default=Path("BabyVision"))
    parser.add_argument(
        "--babyvision-output-dir",
        type=Path,
        default=Path("BabyVision/outputs/rerun_local_skip_judge_fast"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("BabyVision/models"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--exclude-models", nargs="*", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/multimodal_babyvision_models"),
    )
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--parallel-workers", type=int, default=0, help="0 means one worker per listed GPU.")
    parser.add_argument("--max-examples-per-benchmark", type=int, default=None)
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-skip-errors", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--cmmmu-data-dir", type=Path, default=Path("data/CMMMU"))
    parser.add_argument("--cmmmu-split", default="val", choices=["dev", "val", "test"])
    parser.add_argument("--mmmu-data-dir", type=Path, default=Path("data/MMMU"))
    parser.add_argument("--mmmu-split", default="validation", choices=["dev", "validation", "test"])
    parser.add_argument("--mmmu-pro-data-dir", type=Path, default=Path("data/MMMU_Pro"))
    parser.add_argument(
        "--mmmu-pro-setting",
        default="standard (10 options)",
        choices=["standard (10 options)", "standard (4 options)", "vision"],
    )
    parser.add_argument("--mmmu-pro-split", default="test", choices=["test"])
    parser.add_argument("--mathvista-data-dir", type=Path, default=Path("data/MathVista"))
    parser.add_argument("--mathvista-split", default="testmini", choices=["testmini", "test"])

    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--image-layout", choices=["grid", "vertical"], default="grid")
    parser.add_argument("--max-tile-edge", type=int, default=980)
    parser.add_argument("--combined-image-bg", default="white")

    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_benchmarks(value: str) -> list[str]:
    if value == "all":
        return list(BENCHMARKS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(selected).difference(BENCHMARKS))
    if missing:
        raise ValueError(f"Unknown benchmark(s): {missing}; valid values are {list(BENCHMARKS)}")
    return selected


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def args_from_json(payload: dict[str, Any]) -> argparse.Namespace:
    path_keys = {
        "babyvision_dir",
        "babyvision_output_dir",
        "models_dir",
        "output_dir",
        "cmmmu_data_dir",
        "mmmu_data_dir",
        "mmmu_pro_data_dir",
        "mathvista_data_dir",
        "worker_input",
        "worker_output",
    }
    converted = {key: Path(value) if key in path_keys and value is not None else value for key, value in payload.items()}
    return argparse.Namespace(**converted)


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if set(value) == {"bytes"}:
            return None
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return str(value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl_last(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("id", row.get("pid", "")))
            if sid:
                rows[sid] = row
    return rows


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def image_cache_dir(args: argparse.Namespace, benchmark: str, model_name: str) -> Path:
    return args.output_dir / "_image_cache" / benchmark / model_name


def save_image_value(value: Any, path: Path) -> Path | None:
    if is_missing(value):
        return None
    if path.exists():
        return path
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = None
    if isinstance(value, dict) and value.get("bytes"):
        image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    elif hasattr(value, "save"):
        image = value.convert("RGB") if hasattr(value, "convert") else value
    elif isinstance(value, (str, Path)) and Path(value).exists():
        image = Image.open(value).convert("RGB")
    if image is None:
        return None
    tmp = path.with_suffix(path.suffix + ".tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)
    return path


def blank_image(cache_root: Path) -> Path:
    path = cache_root / "_blank.png"
    if path.exists():
        return path
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), "white")
    image.save(path)
    return path


def collect_numbered_images(
    row: dict[str, Any],
    sample_id: str,
    cache_root: Path,
    max_idx: int = 7,
) -> tuple[list[Path], dict[str, str]]:
    paths: list[Path] = []
    filename_to_label: dict[str, str] = {}
    for idx in range(1, max_idx + 1):
        key = f"image_{idx}"
        if key not in row or is_missing(row.get(key)):
            continue
        filename = row.get(f"image_{idx}_filename") or f"{sample_id}_{idx}.png"
        image_path = save_image_value(row[key], cache_root / f"{sample_id}_{idx}.png")
        if image_path is None:
            continue
        label = f"Image {len(paths) + 1}"
        paths.append(image_path)
        filename_to_label[str(filename)] = label
        filename_to_label[f"image {idx}"] = label
        filename_to_label[f"<image {idx}>"] = label
    return paths, filename_to_label


def combine_or_blank(paths: list[Path], cache_root: Path, sample_id: str, args: argparse.Namespace) -> Path:
    if not paths:
        return blank_image(cache_root)
    if len(paths) == 1:
        return paths[0]
    return ensure_combined_image(
        paths,
        cache_root / "_combined" / f"{sample_id}.png",
        args.image_layout,
        args.max_tile_edge,
        args.combined_image_bg,
    )


def clean_cmmmu_text(text: str, filename_to_label: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        filename = match.group(1)
        return f"[{filename_to_label.get(filename, 'Image')}]"

    return re.sub(r"<img=[\"']([^\"']+)[\"']>", repl, str(text))


def clean_mmmu_text(text: str) -> str:
    return re.sub(r"<image\s+(\d+)>", lambda m: f"[Image {m.group(1)}]", str(text))


def parse_options_string(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def option_block(options: list[str]) -> str:
    return "\n".join(f"{CHOICES[idx]}. {option}" for idx, option in enumerate(options))


def build_cmmmu_prompt(row: dict[str, Any], filename_to_label: dict[str, str]) -> str:
    qtype = str(row.get("type", ""))
    question = clean_cmmmu_text(str(row.get("question", "")), filename_to_label)
    if qtype == "选择":
        options = [
            clean_cmmmu_text(str(row.get(f"option{idx}", "")), filename_to_label)
            for idx in range(1, 5)
        ]
        return (
            "请根据题目和图像作答。只输出最终答案，不要解释。\n"
            "选择题请只输出选项字母；如果是多选，请按字母顺序连续输出，例如 AC。\n\n"
            f"问题：{question}\n选项：\n{option_block(options)}\n最终答案："
        )
    if qtype == "判断":
        return (
            "请根据题目和图像判断陈述对错。只输出“对”或“错”，不要解释。\n\n"
            f"问题：{question}\n最终答案："
        )
    return (
        "请根据题目和图像作答。只输出填空答案，不要解释。\n\n"
        f"问题：{question}\n最终答案："
    )


def build_mmmu_prompt(row: dict[str, Any]) -> str:
    question_type = str(row.get("question_type", "multiple-choice"))
    question = clean_mmmu_text(str(row.get("question", "")))
    options = [clean_mmmu_text(option) for option in parse_options_string(row.get("options"))]
    if question_type == "multiple-choice":
        letters = ", ".join(CHOICES[: len(options)])
        return (
            "Answer the multimodal multiple-choice question. Return only the final option letter; "
            f"valid options are {letters}. Do not explain.\n\n"
            f"Question: {question}\nOptions:\n{option_block(options)}\nAnswer:"
        )
    return (
        "Answer the multimodal question. Return only the final short answer. Do not explain.\n\n"
        f"Question: {question}\nAnswer:"
    )


def build_mmmu_pro_prompt(row: dict[str, Any], setting: str) -> str:
    question = clean_mmmu_text(str(row.get("question", "")))
    options = [clean_mmmu_text(option) for option in parse_options_string(row.get("options"))]
    letters = ", ".join(CHOICES[: len(options)])
    setting_note = "MMMU-Pro vision setting" if setting == "vision" else f"MMMU-Pro {setting}"
    return (
        f"Answer the {setting_note} multiple-choice question. Return only the final option letter; "
        f"valid options are {letters}. Do not explain.\n\n"
        f"Question: {question}\nOptions:\n{option_block(options)}\nAnswer:"
    )


def build_mathvista_prompt(row: dict[str, Any]) -> str:
    query = str(row.get("query") or row.get("question") or "")
    question_type = str(row.get("question_type", ""))
    choices = row.get("choices")
    if isinstance(choices, list) and choices and "Choices:" not in query and "Options:" not in query:
        query += "\nChoices:\n" + option_block([str(item) for item in choices])
    if question_type == "multi_choice":
        suffix = "Return only the option letter or exact option text. Do not explain."
    else:
        suffix = "Return only the final value or phrase. Do not explain."
    return f"{query}\n\n{suffix}\nAnswer:"


def select_tail_after_marker(text: str) -> str:
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if matches:
        return text[matches[-1].end() :].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def parse_choice_answer(response: str, option_count: int, expected: str | None = None) -> str:
    if not response:
        return ""
    text = select_tail_after_marker(response)
    valid = CHOICES[:option_count]
    bracketed = re.findall(r"[\(（]([A-Z])[\)）]", text.upper())
    bare = re.findall(r"\b([A-Z])\b", text.upper())
    letters = [letter for letter in bracketed + bare if letter in valid]
    if not letters:
        compact = re.sub(r"[^A-Z]", "", text.upper())
        letters = [letter for letter in compact if letter in valid]
    if not letters:
        return ""
    if expected and len(expected) > 1:
        return "".join(letter for letter in valid if letter in set(letters))
    return letters[-1]


def parse_tf_answer(response: str) -> str:
    text = select_tail_after_marker(response)
    lowered = text.lower()
    if any(token in text for token in ("不对", "错误", "不正确", "不准确", "错")) or "false" in lowered:
        return "错"
    if any(token in text for token in ("正确", "对", "准确")) or "true" in lowered:
        return "对"
    return ""


def parse_short_answer(response: str) -> str:
    text = select_tail_after_marker(response)
    text = re.sub(r"^[：:\s]+", "", text).strip()
    return text.splitlines()[0].strip() if text else ""


def normalize_number_text(text: Any) -> float | None:
    try:
        cleaned = str(text).replace(",", "").replace("，", "").strip()
        return float(cleaned)
    except Exception:
        return None


def is_open_correct(gold: Any, pred: str) -> bool:
    if gold is None or pred is None:
        return False
    gold_text = str(gold).strip()
    pred_text = str(pred).strip()
    if not gold_text or not pred_text:
        return False
    gold_num = normalize_number_text(gold_text)
    pred_numbers = extract_numbers(pred_text)
    if gold_num is not None and pred_numbers:
        return any(abs(float(num) - gold_num) < 1e-6 for num in pred_numbers)
    return gold_text.lower() in pred_text.lower() or pred_text.lower() in gold_text.lower()


def extract_numbers(text: str) -> list[float]:
    pattern = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?"
    numbers: list[float] = []
    for match in re.findall(pattern, str(text)):
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            pass
    return numbers


def normalize_mathvista_prediction(
    extraction: str,
    choices: Any,
    question_type: str,
    answer_type: str,
    precision: Any,
) -> str | None:
    extraction = str(extraction or "").strip()
    if not extraction:
        return None
    if question_type == "multi_choice":
        choice_list = [str(item) for item in choices] if isinstance(choices, list) else []
        letter = parse_choice_answer(extraction, len(choice_list) or 26)
        if letter and choice_list:
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choice_list):
                return choice_list[idx]
        if choice_list:
            import difflib

            return max(choice_list, key=lambda choice: difflib.SequenceMatcher(None, extraction, choice).ratio())
        return letter or extraction
    if answer_type == "integer":
        nums = extract_numbers(extraction)
        return str(int(round(nums[-1]))) if nums else None
    if answer_type == "float":
        nums = extract_numbers(extraction)
        if not nums:
            return None
        try:
            digits = int(precision) if precision is not None and not math.isnan(float(precision)) else 2
        except Exception:
            digits = 2
        return str(round(nums[-1], digits))
    return extraction


def row_without_images(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: to_jsonable(value)
        for key, value in row.items()
        if not key.startswith("image_") and key not in {"decoded_image", "image"}
    }


def load_cmmmu_samples(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cache_root = image_cache_dir(args, "cmmmu", model_name)
    counts_by_category: dict[str, int] = defaultdict(int)
    for path in sorted(args.cmmmu_data_dir.glob(f"*/{args.cmmmu_split}-*.parquet")):
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            category = str(row.get("category", path.parent.name))
            if args.limit_per_category is not None and counts_by_category[category] >= args.limit_per_category:
                continue
            sample_id = str(row["id"])
            image_paths, filename_to_label = collect_numbered_images(row, sample_id, cache_root, max_idx=5)
            combined = combine_or_blank(image_paths, cache_root, sample_id, args)
            prompt = build_cmmmu_prompt(row, filename_to_label)
            samples.append(
                {
                    "id": sample_id,
                    "benchmark": "cmmmu",
                    "split": args.cmmmu_split,
                    "category": category,
                    "subcategory": to_jsonable(row.get("subcategory")),
                    "difficulty": to_jsonable(row.get("difficulty_level")),
                    "type": str(row.get("type", "")),
                    "answer": to_jsonable(row.get("answer")),
                    "prompt": prompt,
                    "image_path": str(combined),
                    "raw": row_without_images(row),
                }
            )
            counts_by_category[category] += 1
            if args.max_examples_per_benchmark is not None and len(samples) >= args.max_examples_per_benchmark:
                return samples
    return samples


def load_mmmu_samples(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cache_root = image_cache_dir(args, "mmmu", model_name)
    counts_by_subject: dict[str, int] = defaultdict(int)
    for path in sorted(args.mmmu_data_dir.glob(f"*/{args.mmmu_split}-*.parquet")):
        subject = path.parent.name
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            if args.limit_per_category is not None and counts_by_subject[subject] >= args.limit_per_category:
                continue
            sample_id = str(row["id"])
            image_paths, _ = collect_numbered_images(row, sample_id, cache_root, max_idx=7)
            combined = combine_or_blank(image_paths, cache_root, sample_id, args)
            options = parse_options_string(row.get("options"))
            prompt = build_mmmu_prompt(row)
            samples.append(
                {
                    "id": sample_id,
                    "benchmark": "mmmu",
                    "split": args.mmmu_split,
                    "subject": subject,
                    "domain": mmmu_subject_to_domain(subject),
                    "subfield": to_jsonable(row.get("subfield")),
                    "difficulty": to_jsonable(row.get("topic_difficulty")),
                    "question_type": str(row.get("question_type", "")),
                    "options": options,
                    "answer": to_jsonable(row.get("answer")),
                    "prompt": prompt,
                    "image_path": str(combined),
                    "raw": row_without_images(row),
                }
            )
            counts_by_subject[subject] += 1
            if args.max_examples_per_benchmark is not None and len(samples) >= args.max_examples_per_benchmark:
                return samples
    return samples


def mmmu_pro_setting_slug(setting: str) -> str:
    return (
        setting.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_mmmu_pro_samples(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cache_root = image_cache_dir(args, "mmmu_pro", model_name)
    setting_dir = args.mmmu_pro_data_dir / args.mmmu_pro_setting
    counts_by_subject: dict[str, int] = defaultdict(int)
    for path in sorted(setting_dir.glob(f"{args.mmmu_pro_split}-*.parquet")):
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            subject = str(row.get("subject", "Unknown"))
            if args.limit_per_category is not None and counts_by_subject[subject] >= args.limit_per_category:
                continue
            sample_id = str(row["id"])
            image_paths, _ = collect_numbered_images(row, sample_id, cache_root, max_idx=7)
            combined = combine_or_blank(image_paths, cache_root, sample_id, args)
            options = [clean_mmmu_text(option) for option in parse_options_string(row.get("options"))]
            samples.append(
                {
                    "id": sample_id,
                    "benchmark": "mmmu_pro",
                    "split": args.mmmu_pro_split,
                    "setting": args.mmmu_pro_setting,
                    "subject": subject,
                    "domain": mmmu_subject_to_domain(subject),
                    "difficulty": to_jsonable(row.get("topic_difficulty")),
                    "img_type": to_jsonable(row.get("img_type")),
                    "question_type": "multiple-choice",
                    "options": options,
                    "answer": to_jsonable(row.get("answer")),
                    "prompt": build_mmmu_pro_prompt(row, args.mmmu_pro_setting),
                    "image_path": str(combined),
                    "raw": row_without_images(row),
                }
            )
            counts_by_subject[subject] += 1
            if args.max_examples_per_benchmark is not None and len(samples) >= args.max_examples_per_benchmark:
                return samples
    return samples


def mmmu_subject_to_domain(subject: str) -> str:
    for domain, subjects in MMMU_DOMAIN_CAT2SUB_CAT.items():
        if subject in subjects:
            return domain
    return "Unknown"


def load_mathvista_samples(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cache_root = image_cache_dir(args, "mathvista", model_name)
    counts_by_category: dict[str, int] = defaultdict(int)
    for path in sorted((args.mathvista_data_dir / "data").glob(f"{args.mathvista_split}-*.parquet")):
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            metadata = to_jsonable(row.get("metadata") or {})
            category = str(metadata.get("category", "Unknown")) if isinstance(metadata, dict) else "Unknown"
            if args.limit_per_category is not None and counts_by_category[category] >= args.limit_per_category:
                continue
            sample_id = str(row["pid"])
            image_value = row.get("decoded_image")
            image_path = save_image_value(image_value, cache_root / f"{sample_id}.png")
            if image_path is None:
                image_path = blank_image(cache_root)
            samples.append(
                {
                    "id": sample_id,
                    "pid": sample_id,
                    "benchmark": "mathvista",
                    "split": args.mathvista_split,
                    "category": category,
                    "task": metadata.get("task") if isinstance(metadata, dict) else None,
                    "context": metadata.get("context") if isinstance(metadata, dict) else None,
                    "grade": metadata.get("grade") if isinstance(metadata, dict) else None,
                    "skills": metadata.get("skills") if isinstance(metadata, dict) else None,
                    "question_type": str(row.get("question_type", "")),
                    "answer_type": str(row.get("answer_type", "")),
                    "choices": to_jsonable(row.get("choices")),
                    "precision": to_jsonable(row.get("precision")),
                    "answer": to_jsonable(row.get("answer")),
                    "prompt": build_mathvista_prompt(row),
                    "image_path": str(image_path),
                    "raw": row_without_images(row),
                    "metadata": metadata,
                }
            )
            counts_by_category[category] += 1
            if args.max_examples_per_benchmark is not None and len(samples) >= args.max_examples_per_benchmark:
                return samples
    return samples


def load_samples(args: argparse.Namespace, benchmark: str, model_name: str) -> list[dict[str, Any]]:
    if benchmark == "cmmmu":
        return load_cmmmu_samples(args, model_name)
    if benchmark == "mmmu":
        return load_mmmu_samples(args, model_name)
    if benchmark == "mmmu_pro":
        return load_mmmu_pro_samples(args, model_name)
    if benchmark == "mathvista":
        return load_mathvista_samples(args, model_name)
    raise ValueError(benchmark)


def parse_prediction(sample: dict[str, Any], response: str) -> Any:
    benchmark = sample["benchmark"]
    if benchmark == "cmmmu":
        qtype = sample.get("type")
        if qtype == "选择":
            return parse_choice_answer(response, 4, str(sample.get("answer") or ""))
        if qtype == "判断":
            return parse_tf_answer(response)
        return parse_short_answer(response)
    if benchmark == "mmmu":
        if sample.get("question_type") == "multiple-choice":
            return parse_choice_answer(response, max(1, len(sample.get("options") or [])), str(sample.get("answer") or ""))
        return parse_short_answer(response)
    if benchmark == "mmmu_pro":
        return parse_choice_answer(response, max(1, len(sample.get("options") or [])), str(sample.get("answer") or ""))
    if benchmark == "mathvista":
        extraction = parse_short_answer(response)
        return normalize_mathvista_prediction(
            extraction,
            sample.get("choices"),
            str(sample.get("question_type", "")),
            str(sample.get("answer_type", "")),
            sample.get("precision"),
        )
    raise ValueError(benchmark)


def is_correct(sample: dict[str, Any], prediction: Any) -> bool | None:
    answer = sample.get("answer")
    if answer is None:
        return None
    if sample["benchmark"] == "cmmmu":
        if sample.get("type") == "填空":
            return is_open_correct(answer, str(prediction or ""))
        return str(prediction or "").strip() == str(answer).strip()
    if sample["benchmark"] == "mmmu":
        if sample.get("question_type") == "multiple-choice":
            return str(prediction or "").strip() == str(answer).strip()
        return is_open_correct(answer, str(prediction or ""))
    if sample["benchmark"] == "mmmu_pro":
        return str(prediction or "").strip() == str(answer).strip()
    if sample["benchmark"] == "mathvista":
        return str(prediction) == str(answer)
    return None


def completed_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return read_json(path).get("status") == "completed"
    except Exception:
        return False


def benchmark_output_dir(args: argparse.Namespace, benchmark: str, model_name: str) -> Path:
    split = {
        "cmmmu": args.cmmmu_split,
        "mmmu": args.mmmu_split,
        "mathvista": args.mathvista_split,
        "mmmu_pro": args.mmmu_pro_split,
    }[benchmark]
    if benchmark == "mmmu_pro":
        return args.output_dir / benchmark / mmmu_pro_setting_slug(args.mmmu_pro_setting) / split / model_name
    return args.output_dir / benchmark / split / model_name


def summarize_rows(benchmark: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("is_correct") is not None]
    correct = sum(1 for row in scored if row.get("is_correct"))
    total = len(scored)
    summary: dict[str, Any] = {
        "status": "completed",
        "benchmark": benchmark,
        "num_examples": len(rows),
        "scored_examples": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": correct / total if total else None,
    }
    group_keys = {
        "cmmmu": ["category", "subcategory", "type", "difficulty"],
        "mmmu": ["domain", "subject", "question_type", "difficulty"],
        "mathvista": ["category", "task", "context", "grade", "question_type", "answer_type", "skills"],
        "mmmu_pro": ["domain", "subject", "difficulty", "img_type"],
    }[benchmark]
    for key in group_keys:
        stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
        for row in scored:
            value = row.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                label = str(item)
                stats[label]["total"] += 1
                if row.get("is_correct"):
                    stats[label]["correct"] += 1
        summary[f"by_{key}"] = {
            label: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for label, value in sorted(stats.items())
        }
    return summary


def run_benchmark(args: argparse.Namespace, vlm: Any, benchmark: str, model_name: str) -> dict[str, Any]:
    out_dir = benchmark_output_dir(args, benchmark, model_name)
    summary_path = out_dir / "summary.json"
    predictions_path = out_dir / "predictions.jsonl"
    answers_path = out_dir / "answers.jsonl"
    if args.resume and completed_summary(summary_path):
        return read_json(summary_path)

    samples = load_samples(args, benchmark, model_name)
    existing = read_jsonl_last(predictions_path) if args.resume else {}
    skip_ids = set()
    for sid, row in existing.items():
        if args.resume_skip_errors or not row.get("model_error"):
            skip_ids.add(sid)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        summary_path,
        {
            "status": "running",
            "benchmark": benchmark,
            "model": model_name,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_examples": len(samples),
            "already_done": len(skip_ids),
        },
    )

    for sample in tqdm(samples, desc=f"{model_name}:{benchmark}"):
        sid = str(sample["id"])
        if sid in skip_ids:
            continue
        gen = vlm.generate(sample["image_path"], sample["prompt"], "answer_only")
        response = str(gen.get("stdout") or "")
        prediction = parse_prediction(sample, response)
        correct = is_correct(sample, prediction)
        record = {
            **{key: sample.get(key) for key in sample if key not in {"prompt", "raw"}},
            "prompt": sample["prompt"],
            "raw": sample.get("raw"),
            "response": response,
            "prediction": prediction,
            "is_correct": correct,
            "model_error": gen.get("error"),
            "model_returncode": gen.get("returncode"),
            "model_latency_seconds": gen.get("latency_seconds"),
            "model_input_mode": gen.get("input_mode"),
            "model_input_tokens": gen.get("input_tokens"),
            "model_input_truncated": gen.get("input_truncated"),
            "model_generation": gen.get("generation"),
        }
        append_jsonl(predictions_path, record)
        if benchmark == "cmmmu":
            append_jsonl(
                answers_path,
                {"id": int(sample["id"]), "type": sample.get("type"), "answer": prediction or ""},
            )

    rows_by_id = read_jsonl_last(predictions_path)
    rows = [rows_by_id[str(sample["id"])] for sample in samples if str(sample["id"]) in rows_by_id]
    summary = summarize_rows(benchmark, rows)
    summary.update(
        {
            "model": model_name,
            "split": samples[0]["split"] if samples else None,
            "setting": samples[0].get("setting") if samples else None,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predictions_path": str(predictions_path),
        }
    )
    write_json(summary_path, summary)
    write_json(out_dir / "predictions.json", rows)
    return summary


def run_model(args: argparse.Namespace, model_name: str, benchmarks: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(args.babyvision_dir.resolve()))
    from babyvision_eval.backends.transformers_vlm import LocalVlm, LocalVlmConfig

    result: dict[str, Any] = {
        "model": model_name,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }
    model_path = args.models_dir / model_name
    def make_config(attn_implementation: str) -> LocalVlmConfig:
        return LocalVlmConfig(
            model_path=str(model_path),
            model_name=model_name,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            attn_implementation=attn_implementation,
            device="cuda:0",
            device_map=None,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            local_files_only=True,
        )

    attn_attempts = [args.attn_implementation]
    for fallback in ("auto", "eager"):
        if fallback not in attn_attempts:
            attn_attempts.append(fallback)
    try:
        last_exc: Exception | None = None
        load_errors = []
        vlm = None
        for attn_implementation in attn_attempts:
            try:
                vlm = LocalVlm(make_config(attn_implementation))
                result["load_debug"] = getattr(vlm, "load_debug", {})
                result["attn_implementation_used"] = attn_implementation
                break
            except Exception as exc:
                last_exc = exc
                load_errors.append(
                    {
                        "attn_implementation": attn_implementation,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if "Invalid attention implementation" not in str(exc):
                    break
        if vlm is None:
            assert last_exc is not None
            raise last_exc
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"model load failed: {type(exc).__name__}: {exc}",
                "load_errors": load_errors if "load_errors" in locals() else [],
                "traceback": traceback.format_exc(),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return result

    failed = False
    for benchmark in benchmarks:
        try:
            summary = run_benchmark(args, vlm, benchmark, model_name)
            result["benchmarks"][benchmark] = {
                "status": summary.get("status"),
                "accuracy": summary.get("accuracy"),
                "correct": summary.get("correct"),
                "total": summary.get("scored_examples"),
                "summary_path": str(benchmark_output_dir(args, benchmark, model_name) / "summary.json"),
            }
        except Exception as exc:
            failed = True
            result["benchmarks"][benchmark] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    result["status"] = "failed" if failed else "completed"
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise ValueError("--worker-input and --worker-output are required in worker mode.")
    payload = read_json(args.worker_input)
    worker_args = args_from_json(payload["args"])
    model_name = payload["model_name"]
    benchmarks = payload["benchmarks"]
    try:
        result = run_model(worker_args, model_name, benchmarks)
    except Exception as exc:
        result = {
            "model": model_name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    write_json(args.worker_output, result)
    if result.get("status") == "failed":
        raise SystemExit(1)


def pending_benchmarks(args: argparse.Namespace, model_name: str, benchmarks: list[str]) -> list[str]:
    pending = []
    for benchmark in benchmarks:
        if not args.resume or not completed_summary(benchmark_output_dir(args, benchmark, model_name) / "summary.json"):
            pending.append(benchmark)
    return pending


def write_leaderboards(args: argparse.Namespace, model_names: list[str], benchmarks: list[str]) -> None:
    for benchmark in benchmarks:
        rows = []
        for model_name in model_names:
            summary_path = benchmark_output_dir(args, benchmark, model_name) / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            if summary.get("status") != "completed":
                continue
            rows.append(
                {
                    "model": model_name,
                    "benchmark": benchmark,
                    "split": summary.get("split"),
                    "setting": summary.get("setting"),
                    "accuracy": summary.get("accuracy"),
                    "correct": summary.get("correct"),
                    "total": summary.get("scored_examples"),
                    "num_examples": summary.get("num_examples"),
                }
            )
        rows.sort(key=lambda row: float(row.get("accuracy") or 0.0), reverse=True)
        if not rows:
            continue
        out_dir = args.output_dir / "leaderboards"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{benchmark}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        md_lines = [
            "| " + " | ".join(rows[0].keys()) + " |",
            "| " + " | ".join(["---"] * len(rows[0])) + " |",
        ]
        for row in rows:
            md_lines.append(
                "| "
                + " | ".join(
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                    for value in row.values()
                )
                + " |"
            )
        (out_dir / f"{benchmark}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def run_parent(args: argparse.Namespace) -> None:
    benchmarks = parse_benchmarks(args.benchmarks)
    model_names = discover_babyvision_models(args)
    if not model_names:
        raise RuntimeError("No BabyVision-tested local models were found.")
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
        pending = pending_benchmarks(args, model_name, benchmarks)
        if pending:
            jobs.append((model_name, pending))
    if not jobs:
        write_leaderboards(args, model_names, benchmarks)
        print(f"All requested outputs are already completed. Outputs are under {args.output_dir}", flush=True)
        return

    worker_root = args.output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    free_gpus = gpu_list[:parallel_workers]
    queued_jobs = list(jobs)
    active: list[dict[str, Any]] = []
    total_jobs = len(queued_jobs)
    launched_jobs = 0
    completed_jobs = 0

    def launch_worker(model_name: str, pending: list[str], gpu: str) -> dict[str, Any]:
        nonlocal launched_jobs
        launched_jobs += 1
        safe_model = sanitize_name(model_name)
        input_path = worker_root / f"{safe_model}.input.json"
        output_path = worker_root / f"{safe_model}.output.json"
        log_path = worker_root / f"{safe_model}.log"
        if output_path.exists():
            output_path.unlink()
        write_json(input_path, {"model_name": model_name, "benchmarks": pending, "args": args_to_json(args)})
        cmd = [
            sys.executable,
            "-m",
            "bench_coe.run_multimodal_babyvision_models",
            "--worker-input",
            str(input_path),
            "--worker-output",
            str(output_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        old_pythonpath = env.get("PYTHONPATH")
        paths = [
            str((Path.cwd() / ".codex_deps/qwen3vl").resolve()),
            str((Path.cwd() / "transformers/src").resolve()),
            str(args.babyvision_dir.resolve()),
            str(Path.cwd().resolve()),
        ]
        env["PYTHONPATH"] = os.pathsep.join(paths + ([old_pythonpath] if old_pythonpath else []))
        cache_dir = worker_root / "cache" / safe_model
        cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "torchinductor"))
        env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
        log_file = log_path.open("w", encoding="utf-8")
        print(
            f"Starting job {launched_jobs}/{total_jobs} on GPU {gpu}: "
            f"{model_name} -> {','.join(pending)}",
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
        return {
            "process": process,
            "model_name": model_name,
            "gpu": gpu,
            "output_path": output_path,
            "log_path": log_path,
            "log_file": log_file,
        }

    while queued_jobs or active:
        while queued_jobs and free_gpus and len(active) < parallel_workers:
            model_name, pending = queued_jobs.pop(0)
            gpu = free_gpus.pop(0)
            active.append(launch_worker(model_name, pending, gpu))

        made_progress = False
        for job in list(active):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            active.remove(job)
            completed_jobs += 1
            job["log_file"].close()
            free_gpus.append(job["gpu"])
            if job["output_path"].exists():
                all_results[job["model_name"]] = read_json(job["output_path"])
            else:
                all_results[job["model_name"]] = {
                    "model": job["model_name"],
                    "status": "failed",
                    "error": f"Worker exited with {return_code} before writing output.",
                }
            if return_code != 0:
                print(
                    f"Worker failed for {job['model_name']} on GPU {job['gpu']} "
                    f"({completed_jobs}/{total_jobs}); see {job['log_path']}",
                    flush=True,
                )
            else:
                print(
                    f"Worker completed for {job['model_name']} on GPU {job['gpu']} "
                    f"({completed_jobs}/{total_jobs}); log {job['log_path']}",
                    flush=True,
                )
            write_json(args.output_dir / "worker_results.json", all_results)
            write_leaderboards(args, model_names, benchmarks)
            made_progress = True

        if active and not made_progress:
            time.sleep(5)

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(args.output_dir / "worker_results.json", all_results)
    write_leaderboards(args, model_names, benchmarks)
    print(f"Finished. Outputs are under {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker_input is not None:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
