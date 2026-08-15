from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(items, **_: Any):
        return items


SCORE_TEMPLATE: dict[str, Any] = {
    "model_name": None,
    "correct_question_num": 0.0,
    "question_num": 646,
    "accuracy": 0.0,
    "subject": {
        "Math": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 80,
            "type": {"2010-2023_Math_MCQs": {"correct_question_num": 0.0, "question_num": 80, "accuracy": 0.0}},
        },
        "Chinese": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 16,
            "type": {"2010-2023_Chinese_Pratical_Lit": {"correct_question_num": 0.0, "question_num": 16, "accuracy": 0.0}},
        },
        "Physics": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 174,
            "type": {"2010-2023_Physics_MCQs": {"correct_question_num": 0.0, "question_num": 174, "accuracy": 0.0}},
        },
        "Chemistry": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 67,
            "type": {"2010-2023_Chemistry_MCQs": {"correct_question_num": 0.0, "question_num": 67, "accuracy": 0.0}},
        },
        "Biology": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 21,
            "type": {"2010-2023_Biology_MCQs": {"correct_question_num": 0.0, "question_num": 21, "accuracy": 0.0}},
        },
        "History": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 34,
            "type": {"2010-2023_History_MCQs": {"correct_question_num": 0.0, "question_num": 34, "accuracy": 0.0}},
        },
        "Geography": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 221,
            "type": {"2010-2023_Geography_MCQs": {"correct_question_num": 0.0, "question_num": 221, "accuracy": 0.0}},
        },
        "Politics": {
            "correct_question_num": 0.0,
            "accuracy": 0.0,
            "question_num": 33,
            "type": {"2010-2023_Political_Science_MCQs": {"correct_question_num": 0.0, "question_num": 33, "accuracy": 0.0}},
        },
    },
}

KNOWN_LOCAL_NAME_MAP = {
    "kimi_vl_a3b_instruct": "Kimi-VL-A3B-Instruct",
}

SKIP_OUTPUT_PREFIXES = (
    "run_",
    "judge_",
    "gemini_",
    "claude_",
    "codex_",
)

ANSWER_RE = re.compile(r"[A-D]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the local models already tested by BabyVision on GAOKAO-MM "
            "with the BabyVision Transformers VLM backend."
        )
    )
    parser.add_argument("--gaokao-mm-dir", type=Path, default=Path("GAOKAO-MM"))
    parser.add_argument("--babyvision-dir", type=Path, default=Path("BabyVision"))
    parser.add_argument(
        "--babyvision-output-dir",
        type=Path,
        default=Path("BabyVision/outputs/rerun_local_skip_judge_fast"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("BabyVision/models"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gaokao_mm_babyvision_models"),
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--exclude-models", nargs="*", default=None)
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="0 means one worker per listed GPU.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume-skip-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a sample already has a record, skip it even when the previous generation errored.",
    )
    parser.add_argument("--image-layout", choices=["grid", "vertical"], default="grid")
    parser.add_argument("--max-tile-edge", type=int, default=980)
    parser.add_argument("--combined-image-bg", default="white")
    parser.add_argument(
        "--use-judge-extractor",
        action="store_true",
        help="Use a local text LLM only to extract A/B/C/D when the official regex finds no answer.",
    )
    parser.add_argument("--judge-model-path", type=Path, default=Path("BabyVision/models/Qwen3.5-9B"))
    parser.add_argument("--judge-model-name", default="Qwen3.5-9B")
    parser.add_argument("--judge-max-new-tokens", type=int, default=32)
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def args_from_json(payload: dict[str, Any]) -> argparse.Namespace:
    path_keys = {
        "gaokao_mm_dir",
        "babyvision_dir",
        "babyvision_output_dir",
        "models_dir",
        "output_dir",
        "judge_model_path",
        "worker_input",
        "worker_output",
    }
    converted = {key: Path(value) if key in path_keys and value is not None else value for key, value in payload.items()}
    return argparse.Namespace(**converted)


def strip_output_suffix(name: str) -> str:
    for marker in (
        "__judge_skipped",
        "__codex_judge",
        "__gemini_judge__judged_by_gemini",
        "__judged_by_gemini",
        "_judged_by_gemini",
    ):
        if name.endswith(marker):
            return name[: -len(marker)]
    return name


def discover_babyvision_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        candidates = list(args.models)
    else:
        candidates = []
        for path in sorted(args.babyvision_output_dir.iterdir()):
            if not path.is_dir():
                continue
            if path.name.startswith(SKIP_OUTPUT_PREFIXES):
                continue
            name = strip_output_suffix(path.name)
            name = KNOWN_LOCAL_NAME_MAP.get(name, name)
            candidates.append(name)

    if args.exclude_models:
        excluded = set(args.exclude_models)
        candidates = [name for name in candidates if name not in excluded]

    model_names = sorted(dict.fromkeys(candidates))
    missing = [name for name in model_names if not (args.models_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Requested/discovered model directories are missing under {args.models_dir}: {missing}")
    return model_names


def load_prompt_specs(gaokao_mm_dir: Path) -> list[dict[str, str]]:
    payload = read_json(gaokao_mm_dir / "Bench" / "MCQ_prompt.json")
    return list(payload["examples"])


def resolve_picture_path(data_file: Path, picture: str) -> Path:
    return (data_file.parent / picture).resolve()


def ensure_combined_image(
    picture_paths: list[Path],
    cache_path: Path,
    layout: str,
    max_tile_edge: int,
    bg: str,
) -> Path:
    if len(picture_paths) == 1:
        return picture_paths[0]
    if cache_path.exists():
        return cache_path

    from PIL import Image, ImageDraw

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for idx, path in enumerate(picture_paths, start=1):
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = min(1.0, float(max_tile_edge) / max(width, height))
            if scale < 1.0:
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            label_h = 28
            label = Image.new("RGB", (image.width, label_h), bg)
            draw = ImageDraw.Draw(label)
            draw.text((8, 7), f"Image {idx}", fill="black")
            tile = Image.new("RGB", (image.width, image.height + label_h), bg)
            tile.paste(label, (0, 0))
            tile.paste(image, (0, label_h))
            tiles.append(tile)

    margin = 14
    if layout == "vertical" or len(tiles) <= 2:
        total_w = max(tile.width for tile in tiles)
        total_h = sum(tile.height for tile in tiles) + margin * (len(tiles) + 1)
        canvas = Image.new("RGB", (total_w + 2 * margin, total_h), bg)
        y = margin
        for tile in tiles:
            canvas.paste(tile, (margin, y))
            y += tile.height + margin
    else:
        columns = 2
        rows = (len(tiles) + columns - 1) // columns
        col_w = max(tile.width for tile in tiles)
        row_h = max(tile.height for tile in tiles)
        canvas = Image.new(
            "RGB",
            (columns * col_w + (columns + 1) * margin, rows * row_h + (rows + 1) * margin),
            bg,
        )
        for idx, tile in enumerate(tiles):
            row = idx // columns
            col = idx % columns
            canvas.paste(tile, (margin + col * (col_w + margin), margin + row * (row_h + margin)))

    tmp = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    canvas.save(tmp, format="PNG")
    try:
        os.replace(tmp, cache_path)
    except FileNotFoundError:
        if not cache_path.exists():
            raise
    return cache_path


def load_tasks(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    prompt_specs = load_prompt_specs(args.gaokao_mm_dir)
    image_cache_root = args.output_dir / "_combined_images"
    tasks: list[dict[str, Any]] = []
    for spec in prompt_specs:
        keyword = spec["keyword"]
        data_file = args.gaokao_mm_dir / "Data" / f"{keyword}.json"
        data = read_json(data_file)
        examples = data["example"]
        if args.limit_per_task is not None:
            examples = examples[: args.limit_per_task]
        prepared_examples = []
        for item in examples:
            pictures = [resolve_picture_path(data_file, pic) for pic in item.get("picture", [])]
            if not pictures:
                combined_image = None
            elif len(pictures) == 1:
                combined_image = pictures[0]
            else:
                combined_image = ensure_combined_image(
                    pictures,
                    image_cache_root / keyword / f"{int(item['index']):04d}.png",
                    args.image_layout,
                    args.max_tile_edge,
                    args.combined_image_bg,
                )
            prepared = dict(item)
            prepared["question"] = str(item["question"]).strip() + "\n"
            prepared["resolved_picture"] = [str(path) for path in pictures]
            prepared["combined_image"] = str(combined_image) if combined_image else None
            prepared_examples.append(prepared)
        tasks.append(
            {
                "keyword": keyword,
                "question_type": spec["type"],
                "prompt": spec["prefix_prompt"],
                "model_name": model_name,
                "example": prepared_examples,
            }
        )
    return tasks


def extract_choice_answer(model_output: str, question_type: str) -> list[str]:
    if not model_output:
        return []
    if question_type == "single_choice":
        matches = re.findall(r"[A-D]", model_output[::-1])
        return [matches[0]] if matches else []

    model_answer: list[str] = []
    answer = ""
    content = re.sub(r"\s+", "", model_output)
    answer_index = content.find("【答案】")
    search_space = content[answer_index:] if answer_index > 0 else content[-10:]
    for match in re.findall(r"[A-D]", search_space):
        answer += match
    if answer:
        model_answer.append(answer)
    return model_answer


def is_task_complete(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return len(payload.get("example", [])) >= expected_count


def load_existing_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = read_json(path)
    except Exception:
        return []
    return list(payload.get("example", []))


def task_save_path(model_dir: Path, model_name: str, keyword: str) -> Path:
    return model_dir / f"{model_name}_{keyword}.json"


def make_error_result(item: dict[str, Any], error: str, stdout: str = "") -> dict[str, Any]:
    return {
        "stdout": stdout,
        "error": error,
        "returncode": None,
        "latency_seconds": None,
        "input_mode": None,
        "input_tokens": None,
        "input_truncated": None,
        "generation": None,
    }


def run_model_tasks(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    sys.path.insert(0, str((args.babyvision_dir).resolve()))
    from babyvision_eval.backends.transformers_vlm import LocalVlm, LocalVlmConfig

    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args, model_name)

    cfg = LocalVlmConfig(
        model_path=str(args.models_dir / model_name),
        model_name=model_name,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
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

    status: dict[str, Any] = {
        "model": model_name,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {},
    }
    write_json(model_dir / "summary.json", status)

    try:
        vlm = LocalVlm(cfg)
        status["load_debug"] = getattr(vlm, "load_debug", {})
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": f"model load failed: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        write_json(model_dir / "summary.json", status)
        return status

    for task in tasks:
        keyword = task["keyword"]
        save_path = task_save_path(model_dir, model_name, keyword)
        examples = task["example"]
        if args.resume and is_task_complete(save_path, len(examples)):
            status["tasks"][keyword] = {"status": "skipped", "examples": len(examples)}
            write_json(model_dir / "summary.json", status)
            continue

        existing = load_existing_examples(save_path) if args.resume else []
        done_by_index = {int(item.get("index", -1)): item for item in existing}
        output_examples = [done_by_index[int(item["index"])] for item in examples if int(item["index"]) in done_by_index]
        output_examples.sort(key=lambda item: int(item.get("index", -1)))

        for item in tqdm(examples, desc=f"{model_name}:{keyword}"):
            index = int(item["index"])
            if index in done_by_index and args.resume_skip_errors:
                continue
            if index in done_by_index and done_by_index[index].get("model_error") in (None, ""):
                continue

            prompt_text = task["prompt"] + "\n" + item["question"]
            image_path = item.get("combined_image")
            if image_path:
                gen_result = vlm.generate(image_path, prompt_text, "answer_only")
            else:
                gen_result = make_error_result(item, "no image_path on sample")
            model_output = str(gen_result.get("stdout") or "")
            model_answer = extract_choice_answer(model_output, task["question_type"])
            record = {
                "index": index,
                "year": item.get("year"),
                "category": item.get("category"),
                "score": item.get("score"),
                "question": item["question"],
                "standard_answer": item.get("answer", []),
                "analysis": item.get("analysis"),
                "picture": item.get("picture", []),
                "resolved_picture": item.get("resolved_picture", []),
                "combined_image": image_path,
                "model_answer": model_answer,
                "model_output": model_output,
                "model_error": gen_result.get("error"),
                "model_returncode": gen_result.get("returncode"),
                "model_latency_seconds": gen_result.get("latency_seconds"),
                "model_input_mode": gen_result.get("input_mode"),
                "model_input_tokens": gen_result.get("input_tokens"),
                "model_input_truncated": gen_result.get("input_truncated"),
                "model_generation": gen_result.get("generation"),
            }
            done_by_index[index] = record
            output_examples = [done_by_index[int(row["index"])] for row in examples if int(row["index"]) in done_by_index]
            output_examples.sort(key=lambda row: int(row["index"]))
            write_json(
                save_path,
                {
                    "keyword": keyword,
                    "model_name": model_name,
                    "prompt": task["prompt"],
                    "question_type": task["question_type"],
                    "example": output_examples,
                },
            )

        status["tasks"][keyword] = {"status": "completed", "examples": len(output_examples)}
        write_json(model_dir / "summary.json", status)

    if args.use_judge_extractor:
        fill_missing_answers_with_judge(args, model_name, model_dir)

    correction = score_model_dir(model_name, model_dir)
    write_json(model_dir / "correction_score.json", correction)
    status.update(
        {
            "status": "completed",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": correction["accuracy"],
            "correct_question_num": correction["correct_question_num"],
            "question_num": correction["question_num"],
        }
    )
    write_json(model_dir / "summary.json", status)
    return status


class TextAnswerExtractor:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.judge_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.judge_model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        self.model.eval()
        self.max_new_tokens = args.judge_max_new_tokens

    def extract(self, question_type: str, question: str, model_output: str) -> list[str]:
        import torch

        if question_type == "single_choice":
            instruction = "从模型回答中抽取最终选项，只能输出 A、B、C、D 中的一个字母。"
        else:
            instruction = "从模型回答中抽取最终选项，只能输出 A、B、C、D 组成的字符串，例如 AB 或 ACD。"
        prompt = (
            f"{instruction}\n\n题目：\n{question}\n\n模型回答：\n{model_output}\n\n最终选项："
        )
        try:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            chat = prompt
        inputs = self.tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        letters = ANSWER_RE.findall(text)
        if not letters:
            return []
        if question_type == "single_choice":
            return [letters[-1]]
        return ["".join(dict.fromkeys(letters))]


def fill_missing_answers_with_judge(args: argparse.Namespace, model_name: str, model_dir: Path) -> None:
    extractor: TextAnswerExtractor | None = None
    for result_path in sorted(model_dir.glob(f"{model_name}_2010-2023_*.json")):
        payload = read_json(result_path)
        question_type = payload.get("question_type", "single_choice")
        changed = False
        for item in payload.get("example", []):
            if item.get("model_answer"):
                continue
            if not str(item.get("model_output", "")).strip():
                continue
            if extractor is None:
                extractor = TextAnswerExtractor(args)
            extracted = extractor.extract(question_type, str(item.get("question", "")), str(item.get("model_output", "")))
            if extracted:
                item["model_answer"] = extracted
                item["answer_extractor"] = args.judge_model_name
                changed = True
        if changed:
            write_json(result_path, payload)


def normalize_model_answer(value: Any, answer_len: int) -> list[str]:
    if isinstance(value, list):
        answer = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        answer = []
    else:
        answer = [str(value).strip()]
    if len(answer) != answer_len:
        return ["Z"] * answer_len
    return answer


def score_model_dir(model_name: str, model_dir: Path) -> dict[str, Any]:
    score_dict = copy.deepcopy(SCORE_TEMPLATE)
    score_dict["model_name"] = model_name

    for result_path in sorted(model_dir.glob(f"{model_name}_*.json")):
        if result_path.name in {"correction_score.json", "summary.json"}:
            continue
        payload = read_json(result_path)
        keyword = payload.get("keyword", payload.get("keywords"))
        if not keyword:
            continue
        subject_key = None
        for key, value in score_dict["subject"].items():
            if keyword in value["type"]:
                subject_key = key
                break
        if subject_key is None:
            continue

        correct = 0.0
        for item in payload.get("example", []):
            standard_answer = [str(x).strip() for x in item.get("standard_answer", [])]
            if len(standard_answer) != 1:
                continue
            model_answer = normalize_model_answer(item.get("model_answer", []), len(standard_answer))
            if keyword in {"2010-2023_Physics_MCQs", "2010-2023_Chinese_Pratical_Lit"}:
                if model_answer[0].lower() == standard_answer[0].lower():
                    correct += 1
                elif model_answer[0].lower() in standard_answer[0].lower():
                    correct += 0.5
            elif model_answer[0].lower() == standard_answer[0].lower():
                correct += 1

        type_stats = score_dict["subject"][subject_key]["type"][keyword]
        type_stats["correct_question_num"] = correct
        type_stats["accuracy"] = round(correct / type_stats["question_num"], 3)
        score_dict["subject"][subject_key]["correct_question_num"] += correct

    total_correct = 0.0
    for value in score_dict["subject"].values():
        value["accuracy"] = round(value["correct_question_num"] / value["question_num"], 3)
        total_correct += value["correct_question_num"]
    score_dict["correct_question_num"] = total_correct
    score_dict["accuracy"] = round(total_correct / score_dict["question_num"], 3)
    return score_dict


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise ValueError("--worker-input and --worker-output are required in worker mode.")
    payload = read_json(args.worker_input)
    worker_args = args_from_json(payload["args"])
    model_name = payload["model_name"]
    result: dict[str, Any] = {
        "model": model_name,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        result = run_model_tasks(worker_args, model_name)
    except Exception as exc:
        result = {
            "model": model_name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(worker_args.output_dir / model_name / "summary.json", result)
    write_json(args.worker_output, result)
    if result.get("status") == "failed":
        raise SystemExit(1)


def model_complete(args: argparse.Namespace, model_name: str) -> bool:
    if not args.resume:
        return False
    summary_path = args.output_dir / model_name / "summary.json"
    if not summary_path.exists():
        return False
    try:
        payload = read_json(summary_path)
    except Exception:
        return False
    return payload.get("status") == "completed"


def write_leaderboard(output_dir: Path, model_names: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    subjects = list(SCORE_TEMPLATE["subject"])
    for model_name in model_names:
        score_path = output_dir / model_name / "correction_score.json"
        if not score_path.exists():
            continue
        score = read_json(score_path)
        row: dict[str, Any] = {
            "model": model_name,
            "correct_question_num": score.get("correct_question_num"),
            "question_num": score.get("question_num"),
            "accuracy": score.get("accuracy"),
        }
        for subject in subjects:
            row[subject] = score.get("subject", {}).get(subject, {}).get("accuracy")
        rows.append(row)
    rows.sort(key=lambda row: float(row.get("accuracy") or 0.0), reverse=True)

    if not rows:
        return
    csv_path = output_dir / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "leaderboard.md"
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parent(args: argparse.Namespace) -> None:
    model_names = discover_babyvision_models(args)
    if not model_names:
        raise RuntimeError("No BabyVision-tested local model directories were found.")

    gpu_list = [item.strip() for item in args.gpu_devices.split(",") if item.strip()]
    if not gpu_list:
        raise ValueError("--gpu-devices must contain at least one GPU id.")
    parallel_workers = args.parallel_workers or len(gpu_list)
    parallel_workers = min(parallel_workers, len(gpu_list))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": model_names,
        "gpu_devices": gpu_list,
        "parallel_workers": parallel_workers,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": args_to_json(args),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(args.output_dir / "model_list.json", model_names)

    jobs = [name for name in model_names if not model_complete(args, name)]
    if not jobs:
        write_leaderboard(args.output_dir, model_names)
        print(f"All requested models are already completed. Outputs are under {args.output_dir}", flush=True)
        return

    worker_root = args.output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    waves = [jobs[start : start + parallel_workers] for start in range(0, len(jobs), parallel_workers)]
    all_results: dict[str, Any] = {}

    for wave_id, wave in enumerate(waves, start=1):
        print(f"Starting wave {wave_id}/{len(waves)} with {len(wave)} worker(s)", flush=True)
        processes: list[tuple[subprocess.Popen[Any], str, Path, Path, Any]] = []
        for model_name, gpu in zip(wave, gpu_list):
            safe_model = sanitize_name(model_name)
            input_path = worker_root / f"{safe_model}.input.json"
            output_path = worker_root / f"{safe_model}.output.json"
            log_path = worker_root / f"{safe_model}.log"
            write_json(input_path, {"model_name": model_name, "args": args_to_json(args)})
            cmd = [
                sys.executable,
                "-m",
                "bench_coe.run_gaokao_mm_babyvision_models",
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
            print(f"  GPU {gpu}: {model_name}", flush=True)
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
                all_results[model_name] = read_json(output_path)
            else:
                all_results[model_name] = {
                    "model": model_name,
                    "status": "failed",
                    "error": f"Worker exited with {return_code} before writing output.",
                }
            if return_code != 0:
                print(f"Worker failed for {model_name}; see {log_path}", flush=True)
            else:
                print(f"Worker completed for {model_name}; log {log_path}", flush=True)
        write_json(args.output_dir / "worker_results.json", all_results)
        write_leaderboard(args.output_dir, model_names)

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(args.output_dir / "worker_results.json", all_results)
    write_leaderboard(args.output_dir, model_names)
    print(f"Finished. Outputs are under {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker_input is not None:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
