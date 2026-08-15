from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .artifacts import environment_manifest, read_selections, sha256_file, validate_test_receipt, write_csv, write_json, write_jsonl, write_selections
from .data import CacheAdapter, EvaluationLabelAdapter, assert_disjoint, load_family_map
from .evaluation import evaluate, holm_adjust
from .selectors import baseline_selectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-resistant Bench-CoE cached innovation experiments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("predict", "evaluate"), required=True)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--limit-source", type=int)
    parser.add_argument("--limit-target", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def make_source_adapter(config: dict[str, Any], family_map: dict[str, str], experts: list[str]) -> CacheAdapter:
    spec = config["source"]
    return CacheAdapter.from_source_registry(
        Path(spec["cache_path"]),
        str(spec["dataset"]),
        str(spec["split"]),
        str(spec["modality"]),
        family_map,
        experts,
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )


def make_target_adapter(config: dict[str, Any], family_map: dict[str, str], experts: list[str]) -> CacheAdapter:
    spec = config["target"]
    return CacheAdapter.from_target_observables(
        Path(spec["observable_cache_path"]),
        str(spec["dataset"]),
        str(spec["split"]),
        str(spec["modality"]),
        family_map,
        experts,
        str(spec["observable_manifest_sha256"]),
    )


def markdown_summary(rows: list[dict[str, Any]], reproduction: dict[str, Any]) -> str:
    lines = [
        "# Baseline reproduction",
        "",
        "Target labels were loaded only after every prediction JSONL had been written and hashed.",
        "",
        "| Method | Accuracy | Delta vs source best | Rescue | Harm | Switch precision | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method']}` | {row['accuracy']:.4f} | {row['delta_vs_source_best_single']:+.4f} | "
            f"{row['rescue_count']} | {row['harm_count']} | {row['switch_precision']:.4f} | {row['exact_mcnemar_p']:.4g} |"
        )
    lines.extend(["", "## Historical-choice comparison", "", "```json", json.dumps(reproduction, indent=2), "```", ""])
    return "\n".join(lines)


def compare_historical(
    prediction_by_method: dict[str, list],
    configured_paths: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for method, path_string in configured_paths.items():
        path = Path(path_string)
        if not path.exists() or method not in prediction_by_method:
            report[method] = {"status": "unavailable", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        historical = payload.get(method, {})
        current = {
            selection.question_id.rsplit("::", 1)[-1]: selection.selected_expert_id
            for selection in prediction_by_method[method]
        }
        common = sorted(set(current).intersection(historical))
        mismatches = [question_id for question_id in common if current[question_id] != historical[question_id]]
        first = mismatches[0] if mismatches else None
        report[method] = {
            "status": "match" if not mismatches and len(common) == len(current) else "mismatch",
            "path": str(path),
            "compared": len(common),
            "current_predictions": len(current),
            "mismatch_count": len(mismatches),
            "first_mismatch": None
            if first is None
            else {"question_id": first, "current": current[first], "historical": historical[first]},
        }
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    receipt_path = Path(config["test_receipt"])
    validate_test_receipt(receipt_path, args.config)
    seed = int(config.get("seed", 20260808))
    experts = [str(item) for item in config["experts"]]
    family_map = load_family_map(Path(config.get("family_map", "configs/innovation/expert_families.yaml")))
    target_adapter = make_target_adapter(config, family_map, experts)
    target_batch = target_adapter.load_observables(args.limit_target)
    output_dir = args.output_dir or Path(config["output_dir"])
    if args.phase == "predict":
        source_adapter = make_source_adapter(config, family_map, experts)
        source_batch = source_adapter.load_observables(args.limit_source)
        source_labels = source_adapter.load_source_labels(args.limit_source)
        assert_disjoint(source_batch, target_batch)
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "config.json", config)
        manifest = environment_manifest(
            sys.argv,
            seed,
            [
                args.config,
                Path(config.get("family_map", "configs/innovation/expert_families.yaml")),
                Path(config["dataset_registry"]),
                Path(config["source"]["cache_path"]),
                Path(config["target"]["observable_cache_path"]),
                receipt_path,
            ],
        )
        manifest.update(
            {
                "phase": "prediction_only_process",
                "source_questions": len(source_batch.question_ids),
                "target_questions": len(target_batch.question_ids),
                "experts": experts,
                "label_firewall": "target adapter can open only physically sanitized observables.jsonl",
            }
        )
        selected_names = set(config.get("methods", []))
        methods = [method for method in baseline_selectors(seed, int(config.get("knn_k", 32))) if not selected_names or method.name in selected_names]
        prediction_hashes: dict[str, str] = {}
        for method in methods:
            method.fit(source_batch, source_labels)
            selections = method.predict(target_batch)
            prediction_hashes[method.name] = write_selections(output_dir / "predictions" / f"{method.name}.jsonl", selections)
        if "source_best_single" not in prediction_hashes:
            raise ValueError("source_best_single must be included as the evaluation reference")
        manifest["prediction_hashes"] = prediction_hashes
        write_json(output_dir / "prediction_manifest.json", manifest)
        print(json.dumps({"output_dir": str(output_dir), "phase": "predict", "methods": sorted(prediction_hashes)}, indent=2))
        return

    if args.evaluation_config is None:
        raise ValueError("--evaluation-config is required for the evaluate phase")
    evaluation_config = load_config(args.evaluation_config)
    validate_test_receipt(receipt_path, args.evaluation_config)
    if not output_dir.exists():
        raise FileNotFoundError(f"Prediction output does not exist: {output_dir}")
    prediction_manifest = json.loads((output_dir / "prediction_manifest.json").read_text(encoding="utf-8"))
    prediction_by_method = {
        method: read_selections(output_dir / "predictions" / f"{method}.jsonl")
        for method in prediction_manifest["prediction_hashes"]
    }
    for method, expected_hash in prediction_manifest["prediction_hashes"].items():
        actual_hash = sha256_file(output_dir / "predictions" / f"{method}.jsonl")
        if actual_hash != expected_hash:
            raise RuntimeError(f"Prediction hash mismatch for {method}")
    label_spec = evaluation_config["target_labels"]
    target_labels = EvaluationLabelAdapter.from_registry(
        Path(label_spec["cache_path"]),
        str(label_spec["dataset"]),
        str(label_spec["split"]),
        str(label_spec["modality"]),
        experts,
        Path(evaluation_config["dataset_registry"]),
        str(evaluation_config["dataset_registry_sha256"]),
    ).load(args.limit_target)
    baseline = prediction_by_method["source_best_single"]
    summaries: list[dict[str, Any]] = []
    for method_name, selections in prediction_by_method.items():
        summary, per_query = evaluate(
            method_name,
            selections,
            baseline,
            target_batch,
            target_labels,
            bootstrap_samples=int(config.get("bootstrap_samples", 1000)),
            seed=seed,
        )
        summaries.append(summary)
        write_jsonl(output_dir / "per_query" / f"{method_name}.jsonl", per_query)
    corrections = holm_adjust({row["method"]: float(row["exact_mcnemar_p"]) for row in summaries if row["method"] != "source_best_single"})
    for row in summaries:
        if row["method"] in corrections:
            row["holm"] = corrections[row["method"]]
    reproduction = compare_historical(prediction_by_method, config.get("historical_choices", {}))
    write_json(output_dir / "summary.json", summaries)
    write_csv(
        output_dir / "summary.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in summaries
        ],
    )
    write_json(output_dir / "historical_comparison.json", reproduction)
    (output_dir / "REPORT.md").write_text(markdown_summary(summaries, reproduction), encoding="utf-8")
    evaluator_manifest = environment_manifest(
        sys.argv,
        seed,
        [
            args.config,
            args.evaluation_config,
            Path(evaluation_config["dataset_registry"]),
            output_dir / "prediction_manifest.json",
            Path(label_spec["cache_path"]),
            receipt_path,
        ],
    )
    evaluator_manifest["phase"] = "evaluation_only_process"
    evaluator_manifest["verified_prediction_hashes"] = prediction_manifest["prediction_hashes"]
    evaluator_manifest["summary_sha256"] = sha256_file(output_dir / "summary.json")
    write_json(output_dir / "run_manifest.json", evaluator_manifest)
    print(json.dumps({"output_dir": str(output_dir), "methods": [row["method"] for row in summaries]}, indent=2))


if __name__ == "__main__":
    main()
