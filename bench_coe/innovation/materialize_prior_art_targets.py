from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .artifacts import sha256_file, write_json
from .data import export_label_free_observables, load_family_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize all physically label-free targets for a prior-art OOD config"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    family_map = load_family_map(Path(config["family_map"]))
    experts = [str(value) for value in config["experts"]]
    rows: list[dict[str, object]] = []
    for target in config["targets"]:
        output = Path(target["observable_cache_path"])
        manifest = export_label_free_observables(
            Path(target["label_cache_path"]),
            output,
            str(target["dataset"]),
            str(target["split"]),
            str(target["modality"]),
            family_map,
            experts,
            Path(config["dataset_registry"]),
            str(config["dataset_registry_sha256"]),
        )
        questions = int(manifest["questions"])
        if questions != int(target["expected_questions"]):
            raise RuntimeError(
                f"Target {target['name']} has {questions} questions; "
                f"expected {target['expected_questions']}"
            )
        manifest_path = output / "observable_manifest.json"
        rows.append(
            {
                "name": str(target["name"]),
                "dataset": str(target["dataset"]),
                "split": str(target["split"]),
                "questions": questions,
                "observable_cache_path": str(output),
                "observable_manifest_sha256": sha256_file(manifest_path),
            }
        )
    receipt = {
        "config": str(args.config),
        "config_sha256_before_manifest_pinning": sha256_file(args.config),
        "targets": rows,
    }
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
