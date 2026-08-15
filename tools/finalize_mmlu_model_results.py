from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    result_dir = args.output_root / args.model / "CoT/validation"
    rows = []
    category_stats = {}
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        rows.extend(payload)
        correct = sum(1 for row in payload if row.get("pred") == row.get("answer"))
        category_stats[path.stem] = {
            "correct": float(correct),
            "wrong": float(len(payload) - correct),
            "accuracy": correct / len(payload) if payload else 0.0,
        }
    if len(rows) != args.expected:
        raise SystemExit(f"Expected {args.expected} rows, found {len(rows)}")
    correct = sum(int(row.get("pred") == row.get("answer")) for row in rows)
    summary = {
        "model": args.model,
        "split": "validation",
        "examples": len(rows),
        "correct": correct,
        "wrong": len(rows) - correct,
        "accuracy": correct / len(rows),
        "category": category_stats,
        "output_dir": str(result_dir),
        "timestamp": time.time(),
        "sharded": True,
    }
    destination = args.output_root / args.model / "summary_validation.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
