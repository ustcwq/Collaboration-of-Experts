from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import (
    SUBJECT_ORDER,
    TASK_TO_SUBJECT,
    format_router_text,
    get_model_answer,
    get_standard_answer,
    load_result_payload,
    read_json,
    write_json,
)
from bench_coe.mmlu_utils import MMLU_CATEGORY_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route GAOKAO questions with an MMLU-Pro category BERT router."
    )
    parser.add_argument(
        "--router-dir",
        type=Path,
        default=Path("outputs/bench_coe/router/bert-base-mmlu-category/model"),
    )
    parser.add_argument(
        "--route-label-manifest",
        type=Path,
        default=Path(
            "outputs/bench_coe/router/bert-base-mmlu-category/route_label_manifest.json"
        ),
    )
    parser.add_argument(
        "--category-expert-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON override. Accepts either {category: model} or "
            "{'category_to_model': {...}, 'category_to_model_index': {...}}."
        ),
    )
    parser.add_argument(
        "--gaokao-data-dir",
        type=Path,
        default=Path("GAOKAO-Bench-2010-2022/Data"),
    )
    parser.add_argument(
        "--benchmark",
        choices=["gaokao2010", "gaokao2023", "gaokao2024"],
        default="gaokao2010",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_coe/mmlu_router_on_gaokao2010"),
    )
    parser.add_argument("--router-max-length", type=int, default=256)
    parser.add_argument("--router-batch-size", type=int, default=256)
    parser.add_argument("--router-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--reextract-empty",
        action="store_true",
        help="Try to extract choices from model_output when model_answer is empty.",
    )
    parser.add_argument(
        "--save-model-output",
        action="store_true",
        help="Store selected expert raw outputs in predictions.json.",
    )
    return parser.parse_args()


def infer_gaokao_subject(keyword: str) -> str:
    if keyword in TASK_TO_SUBJECT:
        return TASK_TO_SUBJECT[keyword]
    if "English_" in keyword:
        return "English"
    if "Math" in keyword:
        return "Math"
    if "Chinese_" in keyword:
        return "Chinese"
    if "Physics_" in keyword:
        return "Physics"
    if "Chemistry_" in keyword:
        return "Chemistry"
    if "Biology_" in keyword:
        return "Biology"
    if "History_" in keyword:
        return "History"
    if "Geography_" in keyword:
        return "Geography"
    if "Political_Science_" in keyword:
        return "Politics"
    return "Unknown"


def load_gaokao_rows(data_dir: Path, benchmark: str) -> list[dict[str, Any]]:
    if benchmark == "gaokao2010":
        paths = sorted((data_dir / "Objective_Questions").glob("*.json"))
    else:
        year = "2023" if benchmark == "gaokao2023" else "2024"
        paths = sorted(data_dir.glob(f"Bench-Harness_{year}_*.json"))

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        keyword = payload.get("keyword") or payload.get("keywords") or path.stem
        subject = infer_gaokao_subject(keyword)
        for example in payload.get("example", []):
            standard_answer = get_standard_answer(example)
            if not standard_answer:
                continue
            rows.append(
                {
                    "id": f"{keyword}:{example.get('index')}",
                    "benchmark": benchmark,
                    "keyword": keyword,
                    "subject": subject,
                    "index": example.get("index"),
                    "year": example.get("year"),
                    "category": example.get("category"),
                    "question": str(example.get("question", "")).strip(),
                    "standard_answer": standard_answer,
                    "score": float(example.get("score", 0.0)),
                }
            )
    return rows


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
    return sorted(copied[:max_examples], key=lambda row: row["id"])


def choose_router_device(router_device: str) -> torch.device:
    if router_device == "cpu":
        return torch.device("cpu")
    if router_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("router-device=cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_category_expert_map(
    label_manifest: dict[str, Any],
    override_path: Path | None,
) -> dict[str, Any]:
    if override_path is None:
        return label_manifest
    override = read_json(override_path)
    category_to_model = override.get("category_to_model", override)
    category_to_model_index = override.get("category_to_model_index", {})
    if not isinstance(category_to_model, dict):
        raise ValueError("--category-expert-map must contain a category->model mapping.")

    merged = dict(label_manifest)
    merged["category_to_model"] = {
        **label_manifest.get("category_to_model", {}),
        **category_to_model,
    }
    merged_model_index = {
        **label_manifest.get("category_to_model_index", {}),
        **category_to_model_index,
    }
    original_category_to_model = label_manifest.get("category_to_model", {})
    for category, model_name in category_to_model.items():
        if (
            category not in category_to_model_index
            and original_category_to_model.get(category) != model_name
        ):
            merged_model_index[category] = -1
    merged["category_to_model_index"] = merged_model_index
    return merged


def resolve_route_label(
    label_manifest: dict[str, Any],
    label: int,
) -> tuple[str, str, int]:
    label_key = str(label)
    if "route_label_to_category" not in label_manifest:
        model_name = label_manifest["route_label_to_model"][label_key]
        model_index_raw = label_manifest.get("route_label_to_model_index", {}).get(
            label_key, -1
        )
        return model_name, model_name, int(model_index_raw)
    category = label_manifest["route_label_to_category"][label_key]
    model_name = label_manifest.get("category_to_model", {}).get(category)
    if not model_name:
        model_name = label_manifest.get("route_label_to_model", {}).get(label_key)
    if not model_name:
        raise KeyError(f"Missing category_to_model entry for routed category: {category}")
    model_index_raw = label_manifest.get("category_to_model_index", {}).get(category)
    if model_index_raw is None:
        model_index_raw = label_manifest.get("route_label_to_model_index", {}).get(
            label_key, -1
        )
    return category, model_name, int(model_index_raw)


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
        texts = [format_router_text(row["question"]) for row in batch_rows]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        probs = logits.softmax(dim=-1)
        route_labels = probs.argmax(dim=-1).detach().cpu().tolist()
        route_probs = probs.max(dim=-1).values.detach().cpu().tolist()
        for row, label, confidence in zip(batch_rows, route_labels, route_probs):
            routed_category, routed_model, routed_model_index = resolve_route_label(
                label_manifest, int(label)
            )
            routed = dict(row)
            routed["route_label"] = int(label)
            routed["route_confidence"] = float(confidence)
            routed["routed_category"] = routed_category
            routed["routed_model"] = routed_model
            routed["routed_model_index"] = routed_model_index
            routed_rows.append(routed)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return routed_rows


def score_answer(
    keyword: str,
    standard_answer: list[str],
    model_answer: list[str],
    score_per_choice: float,
) -> tuple[float, float, float]:
    total_score = len(standard_answer) * score_per_choice
    correct_score = 0.0

    if keyword.endswith("Physics_MCQs"):
        for idx, expected in enumerate(standard_answer):
            predicted = model_answer[idx]
            if predicted == expected:
                correct_score += score_per_choice
            else:
                has_wrong_choice = any(choice not in expected for choice in predicted)
                if not has_wrong_choice:
                    correct_score += score_per_choice / 2.0
    else:
        for idx, expected in enumerate(standard_answer):
            if model_answer[idx] == expected:
                correct_score += score_per_choice

    return correct_score, total_score, float(len(standard_answer))


class ResultCache:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.cache: dict[tuple[str, str], dict[str, dict[str, Any]] | None] = {}

    def get(self, model_name: str, keyword: str, index: Any) -> dict[str, Any] | None:
        cache_key = (model_name, keyword)
        if cache_key not in self.cache:
            payload = load_result_payload(self.data_dir, model_name, keyword)
            if payload is None:
                self.cache[cache_key] = None
            else:
                self.cache[cache_key] = {
                    str(example.get("index")): example
                    for example in payload.get("example", [])
                }
        index_map = self.cache[cache_key]
        if index_map is None:
            return None
        return index_map.get(str(index))


def update_stats(
    stats: dict[str, float],
    correct_score: float,
    total_score: float,
    question_num: float,
) -> None:
    stats["correct_score"] += correct_score
    stats["total_score"] += total_score
    stats["question_num"] += question_num
    stats["example_num"] += 1.0


def finalize_stats(raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for key, stats in raw.items():
        total_score = stats["total_score"]
        finalized[key] = {
            "correct_score": stats["correct_score"],
            "total_score": total_score,
            "question_num": stats["question_num"],
            "example_num": stats["example_num"],
            "accuracy": stats["correct_score"] / total_score if total_score else 0.0,
        }
    return dict(sorted(finalized.items()))


def score_rows(
    rows: list[dict[str, Any]],
    data_dir: Path,
    reextract_empty: bool,
    save_model_output: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = ResultCache(data_dir)
    total_raw = defaultdict(float)
    subject_raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    category_raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    model_raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    missing_results: Counter[str] = Counter()
    scored_rows: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="scoring CoE"):
        standard_answer = row["standard_answer"]
        selected = cache.get(row["routed_model"], row["keyword"], row["index"])
        if selected is None:
            missing_results[f"{row['routed_model']}::{row['keyword']}"] += 1
            selected = {}
        model_answer = get_model_answer(
            selected,
            len(standard_answer),
            reextract_empty=reextract_empty,
        )
        correct_score, total_score, question_num = score_answer(
            row["keyword"],
            standard_answer,
            model_answer,
            float(row["score"]),
        )
        update_stats(total_raw, correct_score, total_score, question_num)
        update_stats(subject_raw[row["subject"]], correct_score, total_score, question_num)
        update_stats(
            category_raw[row["routed_category"]],
            correct_score,
            total_score,
            question_num,
        )
        update_stats(
            model_raw[row["routed_model"]],
            correct_score,
            total_score,
            question_num,
        )
        scored = dict(row)
        scored["selected_model_answer"] = model_answer
        scored["correct_score"] = correct_score
        scored["total_score"] = total_score
        if save_model_output:
            scored["selected_model_output"] = selected.get("model_output", "")
        scored_rows.append(scored)

    total = {
        "correct_score": total_raw["correct_score"],
        "total_score": total_raw["total_score"],
        "question_num": total_raw["question_num"],
        "example_num": total_raw["example_num"],
        "accuracy": total_raw["correct_score"] / total_raw["total_score"]
        if total_raw["total_score"]
        else 0.0,
    }
    summary = {
        "total": total,
        "subject": finalize_stats(subject_raw),
        "routed_category": finalize_stats(category_raw),
        "routed_model": finalize_stats(model_raw),
        "missing_results": dict(sorted(missing_results.items())),
    }
    return summary, scored_rows


def score_single_model(
    model_name: str,
    rows: list[dict[str, Any]],
    data_dir: Path,
    reextract_empty: bool,
) -> dict[str, Any]:
    cache = ResultCache(data_dir)
    total_raw = defaultdict(float)
    subject_raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    missing_results = 0
    for row in rows:
        standard_answer = row["standard_answer"]
        selected = cache.get(model_name, row["keyword"], row["index"])
        if selected is None:
            missing_results += 1
            selected = {}
        model_answer = get_model_answer(
            selected,
            len(standard_answer),
            reextract_empty=reextract_empty,
        )
        correct_score, total_score, question_num = score_answer(
            row["keyword"],
            standard_answer,
            model_answer,
            float(row["score"]),
        )
        update_stats(total_raw, correct_score, total_score, question_num)
        update_stats(subject_raw[row["subject"]], correct_score, total_score, question_num)
    total = {
        "correct_score": total_raw["correct_score"],
        "total_score": total_raw["total_score"],
        "question_num": total_raw["question_num"],
        "example_num": total_raw["example_num"],
        "accuracy": total_raw["correct_score"] / total_raw["total_score"]
        if total_raw["total_score"]
        else 0.0,
    }
    return {
        "total": total,
        "subject": finalize_stats(subject_raw),
        "missing_result_examples": missing_results,
    }


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def subject_accuracy(summary: dict[str, Any], subject: str) -> float | None:
    subject_stats = summary.get("subject", {}).get(subject)
    if subject_stats is None:
        return None
    return float(subject_stats["accuracy"])


def render_bench_harness_txt(
    path: Path,
    benchmark: str,
    rows: list[dict[str, Any]],
    coe_summary: dict[str, Any],
    single_model_summary: dict[str, Any],
) -> None:
    subjects = [subject for subject in SUBJECT_ORDER if subject in coe_summary["subject"]]
    subjects.append("Average")
    col_width = 15
    name_width = 34

    def row_line(name: str, values: list[str]) -> str:
        cells = [name.ljust(name_width)]
        cells.extend(value.ljust(col_width) for value in values)
        return "| " + " | ".join(cells) + " |"

    question_counts = []
    for subject in subjects[:-1]:
        count = int(coe_summary["subject"][subject]["example_num"])
        question_counts.append(str(count))
    question_counts.append(str(len(rows)))

    single_totals = {
        model: float(summary["total"]["accuracy"])
        for model, summary in single_model_summary.items()
    }
    best_model = max(single_totals, key=single_totals.get) if single_totals else None
    best_acc = single_totals[best_model] if best_model else 0.0
    coe_acc = float(coe_summary["total"]["accuracy"])

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Bench-Harness: MMLU-Pro prior -> GAOKAO")
    lines.append("=" * 100)
    lines.append("| Routing Mode: bert_mmlu_category")
    lines.append(f"| Benchmark: {benchmark}")
    lines.append(f"| Samples: {len(rows)}")
    lines.append("")
    lines.append(row_line("Model / Metric", subjects))
    lines.append(row_line("-" * 28, ["-" * 12 for _ in subjects]))
    lines.append(row_line("Qs (Count)", question_counts))
    lines.append(row_line("-" * 28, ["-" * 12 for _ in subjects]))

    for model_name, summary in sorted(
        single_model_summary.items(),
        key=lambda item: float(item[1]["total"]["accuracy"]),
        reverse=True,
    ):
        prefix = "* " if model_name == best_model else "  "
        values = [
            format_percent(subject_accuracy(summary, subject))
            for subject in subjects[:-1]
        ]
        values.append(format_percent(float(summary["total"]["accuracy"])))
        lines.append(row_line(prefix + model_name, values))

    lines.append(row_line("-" * 28, ["-" * 12 for _ in subjects]))
    coe_values = [
        format_percent(subject_accuracy(coe_summary, subject)) for subject in subjects[:-1]
    ]
    coe_values.append(format_percent(coe_acc))
    lines.append(row_line("MMLU-Bert-Bench-CoE", coe_values))
    gain = coe_acc - best_acc
    gain_values = [""] * (len(subjects) - 1) + [f"{gain * 100:+.2f}%"]
    lines.append(row_line("Gain (vs Best Exp)", gain_values))
    lines.append("")

    routed_model_counts = Counter(row["routed_model"] for row in rows)
    routed_category_counts = Counter(row["routed_category"] for row in rows)
    lines.append("Routed models:")
    for model_name, count in routed_model_counts.most_common():
        lines.append(f"- {model_name}: {count}")
    lines.append("Routed MMLU categories:")
    for category in MMLU_CATEGORY_ORDER:
        count = routed_category_counts.get(category, 0)
        if count:
            lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append(f"Saved predictions to: {path.parent / 'predictions.json'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_manifest = apply_category_expert_map(
        read_json(args.route_label_manifest), args.category_expert_map
    )
    rows = limit_rows(
        load_gaokao_rows(args.gaokao_data_dir, args.benchmark),
        args.max_examples,
        args.seed,
    )
    if not rows:
        raise SystemExit("No GAOKAO rows were loaded.")

    routed_rows = route_rows(
        rows,
        args.router_dir,
        label_manifest,
        args.router_max_length,
        args.router_batch_size,
        args.router_device,
    )
    coe_summary, scored_rows = score_rows(
        routed_rows,
        args.gaokao_data_dir,
        args.reextract_empty,
        args.save_model_output,
    )

    model_names = label_manifest.get("model_names") or sorted(
        set(label_manifest.get("category_to_model", {}).values())
    )
    single_model_summary: dict[str, Any] = {}
    for model_name in tqdm(model_names, desc="single models"):
        single_model_summary[model_name] = score_single_model(
            model_name,
            rows,
            args.gaokao_data_dir,
            args.reextract_empty,
        )

    route_distribution = {
        "routed_category": dict(sorted(Counter(row["routed_category"] for row in routed_rows).items())),
        "routed_model": dict(sorted(Counter(row["routed_model"] for row in routed_rows).items())),
    }
    summary = {
        "benchmark": args.benchmark,
        "gaokao_data_dir": str(args.gaokao_data_dir),
        "router_dir": str(args.router_dir),
        "route_label_manifest": str(args.route_label_manifest),
        "category_expert_map": str(args.category_expert_map)
        if args.category_expert_map
        else None,
        "route_distribution": route_distribution,
        "coe": coe_summary,
        "single_model": single_model_summary,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "predictions.json", scored_rows)
    render_bench_harness_txt(
        args.output_dir / f"Bench_Harness_Result_mmlu_prior_{args.benchmark}.txt",
        args.benchmark,
        routed_rows,
        coe_summary,
        single_model_summary,
    )

    print(f"Rows: {len(rows)}")
    print(f"CoE accuracy: {coe_summary['total']['accuracy']:.4f}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(
        "Wrote "
        f"{args.output_dir / f'Bench_Harness_Result_mmlu_prior_{args.benchmark}.txt'}"
    )


if __name__ == "__main__":
    main()
