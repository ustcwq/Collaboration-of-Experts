from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bench_coe.run_official_model_benchmarks import (
    parse_args as parse_benchmark_args,
    summarize_bbh,
    summarize_gpqa,
    write_jsonl,
)
from bench_coe.gaokao_utils import write_json


EXPECTED = {"bbh": 6511, "gpqa": 4768}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for index in range(args.shard_count):
        path = args.shard_root / f"shard{index}" / args.benchmark / args.model / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: int(row["question_id"]))
    ids = [str(row["id"]) for row in rows]
    expected = EXPECTED[args.benchmark]
    if len(rows) != expected or len(set(ids)) != expected:
        raise RuntimeError(
            f"{args.benchmark} merge mismatch: rows={len(rows)} unique={len(set(ids))} expected={expected}"
        )

    saved_argv = sys.argv
    try:
        sys.argv = ["merge_official_text_shards"]
        benchmark_args = parse_benchmark_args()
    finally:
        sys.argv = saved_argv
    benchmark_args.output_dir = args.output_root

    output_dir = args.output_root / args.benchmark / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    if args.benchmark == "bbh":
        summary = summarize_bbh(args.model, rows, benchmark_args)
        pd.DataFrame(summary["by_task"]).T.to_csv(output_dir / "task_summary.csv")
    else:
        summary = summarize_gpqa(args.model, rows, benchmark_args)
        pd.DataFrame(summary["by_config"]).T.to_csv(output_dir / "config_summary.csv")
        pd.DataFrame(summary["by_domain"]).T.to_csv(output_dir / "domain_summary.csv")
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
