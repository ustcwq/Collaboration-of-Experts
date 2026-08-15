from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import sha256_file, write_json
from .expanded_expert_bridge import filter_rows_by_id_prefix, row_id


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a row list: {path}")
    return payload


def _prediction_path(root: Path, expert: str) -> Path:
    for name in ("predictions.json", "predictions.jsonl"):
        path = root / expert / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing raw prediction cache for {expert}: {root / expert}")


def _id_set_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project one ID-defined split from a mixed expanded MMMU-Pro cache"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    projection = config["target_projection"]
    input_root = Path(projection["input_cache_path"])
    prefix = str(projection["id_prefix"])
    expected = int(projection["expected_questions"])
    experts = tuple(sorted(str(value) for value in config["experts"]))

    input_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    canonical_ids: list[str] | None = None
    for expert in experts:
        input_path = _prediction_path(input_root, expert)
        rows = filter_rows_by_id_prefix(_read_rows(input_path), prefix)
        ids = [row_id(row) for row in rows]
        if len(ids) != expected:
            raise RuntimeError(f"{expert} projected {len(ids)} rows; expected {expected}")
        if canonical_ids is None:
            canonical_ids = ids
        elif ids != canonical_ids:
            raise RuntimeError(f"{expert} projected IDs differ from the common ID set")

        output_path = args.output_dir / expert / "predictions.json"
        write_json(output_path, rows)
        input_hashes[str(input_path)] = sha256_file(input_path)
        output_hashes[str(output_path)] = sha256_file(output_path)

    assert canonical_ids is not None
    manifest = {
        "role": "target_label_projection",
        "dataset": str(projection["dataset"]),
        "split": str(projection["split"]),
        "modality": str(projection["modality"]),
        "selection_rule": {"field": "id", "prefix": prefix},
        "expert_ids": list(experts),
        "questions": len(canonical_ids),
        "question_ids_sha256": _id_set_sha256(canonical_ids),
        "input_raw_cache_hashes": input_hashes,
        "output_target_cache_hashes": output_hashes,
        "contains_target_labels": True,
        "contains_source_rows": False,
    }
    manifest_path = args.output_dir / "target_projection_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest_sha256": sha256_file(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
