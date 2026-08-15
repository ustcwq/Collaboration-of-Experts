from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL = "Ministral-3-3B-Instruct-2512"


def json_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def json_row_count(directory: Path) -> int:
    total = 0
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return -1
        if not isinstance(payload, list):
            continue
        total += len(payload)
    return total


def jsonl_row_count(path: Path) -> int:
    try:
        return sum(1 for line in path.open(encoding="utf-8") if line.strip())
    except OSError:
        return -1


def gaokao_complete(root: Path, dataset: str, expected: int) -> bool:
    directory = root / dataset / MODEL
    summary = json_payload(directory / "summary.json")
    return bool(
        summary
        and int(summary.get("total", -1)) == expected
        and jsonl_row_count(directory / "predictions.jsonl") == expected
    )


def mmlu_complete(root: Path, expected: int) -> bool:
    directory = root / MODEL / "CoT/validation"
    summary = json_payload(root / MODEL / "summary_validation.json")
    return bool(
        summary
        and int(summary.get("examples", -1)) == expected
        and json_row_count(directory) == expected
    )


def official_complete(root: Path, benchmark: str, expected: int) -> bool:
    directory = root / benchmark / MODEL
    summary = json_payload(directory / "summary.json")
    return bool(
        summary
        and summary.get("status") == "completed"
        and int(summary.get("num_examples", -1)) == expected
        and jsonl_row_count(directory / "predictions.jsonl") == expected
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()

    benchmark_root = Path("outputs/model_benchmarks/family_scale_expansion_full_20260731/text")
    validation_root = Path("outputs/model_benchmarks/improve56_scale_sources_20260801/text/mmlu_validation")
    validation_marker = Path(
        "outputs/bench_coe/autonomous_remaining_supervisor_20260802/ministral_validation_stop_refresh_complete"
    )
    official_root = benchmark_root / "official"
    checks = {
        "gaokao_2010_2022": gaokao_complete(benchmark_root / "gaokao", "gaokao_2010_2022", 1676),
        "gaokao_2023_2024": gaokao_complete(benchmark_root / "gaokao", "gaokao_2023_2024", 183),
        "mmlu_test": mmlu_complete(benchmark_root / "mmlu_pro_test", 12032),
        "mmlu_validation": mmlu_complete(validation_root, 70) and validation_marker.is_file(),
        "bbh": official_complete(official_root, "bbh", 6511),
        "gpqa": official_complete(official_root, "gpqa", 4768),
        "mmstar_text_only": official_complete(official_root, "mmstar_text_only", 1500),
    }
    payload = {
        "model": MODEL,
        "complete": all(checks.values()),
        "checks": checks,
        "missing": [name for name, complete in checks.items() if not complete],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.status_file:
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
        args.status_file.write_text(text, encoding="utf-8")
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
