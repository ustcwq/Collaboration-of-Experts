from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import write_json
from .locked_protocol import audit_gpqa_units


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GPQA overlap and repeated statistical units")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/model_benchmarks/official_code_local_models/gpqa"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/gpqa"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bench_coe/innovation/audits/gpqa_statistical_units_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_gpqa_units(args.cache_root, args.raw_root)
    write_json(args.output, result)
    print(f"GPQA unit audit: {args.output}")


if __name__ == "__main__":
    main()
