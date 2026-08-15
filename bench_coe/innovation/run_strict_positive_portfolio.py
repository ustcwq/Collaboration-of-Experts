from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import (
    environment_manifest,
    files_manifest,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_selections,
)
from .data import EvaluationLabelAdapter
from .goal_guardrails import evaluate_strict_improvement_contract
from .run_conservative_meta_optimization import (
    _aggregate_comparison,
    _correctness_for_labels,
    _read_authenticated_selections,
    _seed_dir,
)
from .schema import EvaluationLabels, Selection


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize and evaluate strict-positive development portfolios"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-portfolios", type=int)
    return parser.parse_args()


def _completion_bound(path: Path, expected: str, completion: Mapping[str, Any]) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    if completion.get("artifact_hashes", {}).get(str(path)) != expected:
        raise RuntimeError(f"Prediction is not completion-bound: {path}")


def _prior_art_rows(
    job: Mapping[str, Any],
    seed: int,
    method: str,
    authenticated: dict[str, str],
    manifest_paths: set[Path],
) -> list[Selection]:
    seed_dir = _seed_dir(Path(job["run_root"]), seed)
    manifest_path = seed_dir / "prediction_manifest.json"
    completion_path = seed_dir / "complete_manifest.json"
    manifest_paths.update((manifest_path, completion_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if job["package_scope"] == "source":
        expected = manifest["prediction_hashes_before_evaluation"].get(method)
        path = seed_dir / "predictions" / f"{method}.jsonl"
    else:
        target = manifest["targets"][str(job["name"])]
        expected = target["prediction_hashes_before_evaluation"].get(method)
        relative = target["prediction_paths"].get(method)
        if relative is None:
            raise RuntimeError(f"Missing prior-art method {job['name']}/{method}")
        path = seed_dir / str(relative)
    if not isinstance(expected, str):
        raise RuntimeError(f"Missing prior-art prediction hash {job['name']}/{method}")
    _completion_bound(path, expected, completion)
    rows, actual = _read_authenticated_selections(path, expected)
    authenticated[str(path)] = actual
    return sorted(rows, key=lambda row: row.question_id)


def _conservative_rows(
    root: Path,
    dataset: str,
    seed: int,
    method: str,
    prediction_manifest: Mapping[str, Any],
    complete_manifest: Mapping[str, Any],
    authenticated: dict[str, str],
) -> list[Selection]:
    if dataset == "source_loso":
        expected = prediction_manifest["predictions"]["source"][str(seed)].get(method)
        path = root / "predictions" / "source_loso" / f"seed_{seed}" / f"{method}.jsonl"
    else:
        entry = prediction_manifest["predictions"]["targets"][dataset][str(seed)].get(method)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Missing conservative method {dataset}/{method}")
        expected = entry["sha256"]
        path = root / str(entry["path"])
    if not isinstance(expected, str):
        raise RuntimeError(f"Missing conservative prediction hash {dataset}/{method}")
    _completion_bound(path, expected, complete_manifest)
    rows, actual = _read_authenticated_selections(path, expected)
    authenticated[str(path)] = actual
    return sorted(rows, key=lambda row: row.question_id)


def _portfolio_rows(
    rows: Sequence[Selection],
    portfolio: str,
    dataset: str,
    component_source: str,
    component_method: str,
) -> list[Selection]:
    result: list[Selection] = []
    for row in rows:
        features = dict(row.observable_features)
        features.update(
            {
                "method": portfolio,
                "strict_positive_portfolio": True,
                "portfolio_dataset": dataset,
                "portfolio_component_source": component_source,
                "portfolio_component_method": component_method,
                "portfolio_uses_question_labels": False,
                "portfolio_mapping_is_development_posthoc": True,
            }
        )
        result.append(
            replace(
                row,
                observable_features=features,
                tie_breaking=(
                    f"development-dataset-portfolio:{component_source}:{component_method};"
                    + row.tie_breaking
                ),
            )
        )
    return result


def _dataset_jobs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    jobs: dict[str, dict[str, Any]] = {
        "source_loso": {
            "name": "source_loso",
            "package_scope": "source",
            "run_root": config["source_run_root"],
            "seeds": [int(value) for value in source_config["seeds"]],
            "experts": [str(value) for value in source_config["experts"]],
            "registry": source_config["dataset_registry"],
            "registry_sha256": source_config["dataset_registry_sha256"],
            "labels": source_config["source"],
        }
    }
    for raw_panel in config["target_panels"]:
        panel_path = Path(raw_panel["config"])
        panel = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
        for target in panel["targets"]:
            name = str(target["name"])
            jobs[name] = {
                "name": name,
                "package_scope": "target",
                "run_root": raw_panel["run_root"],
                "seeds": [int(value) for value in panel["seeds"]],
                "experts": [str(value) for value in panel["experts"]],
                "registry": panel["dataset_registry"],
                "registry_sha256": panel["dataset_registry_sha256"],
                "labels": target,
            }
    return jobs


def _load_labels(job: Mapping[str, Any], limit: int | None) -> EvaluationLabels:
    raw = job["labels"]
    cache_key = "cache_path" if job["package_scope"] == "source" else "label_cache_path"
    return EvaluationLabelAdapter.from_registry(
        Path(raw[cache_key]),
        str(raw["dataset"]),
        str(raw["split"]),
        str(raw["modality"]),
        job["experts"],
        Path(job["registry"]),
        str(job["registry_sha256"]),
    ).load(limit=limit)


def _fixed_method_audit(paths: Sequence[str], targets: Sequence[str]) -> dict[str, Any]:
    accuracy: dict[str, dict[str, float]] = {}
    pools: dict[str, dict[str, str]] = {}
    for raw_path in paths:
        with Path(raw_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                target = str(row.get("target") or "source_loso")
                method = str(row["method"])
                accuracy.setdefault(target, {})[method] = float(row["accuracy_mean"])
                pools.setdefault(target, {})[method] = str(row.get("pool") or "full_pool")
    baseline = {target: accuracy[target]["fcrg_full"] for target in targets}
    common = set.intersection(*(set(accuracy[target]) for target in targets))
    fixed_rows: list[dict[str, Any]] = []
    for method in sorted(common):
        deltas = {target: accuracy[target][method] - baseline[target] for target in targets}
        fixed_rows.append(
            {
                "method": method,
                "all_strictly_positive": all(value > 1e-12 for value in deltas.values()),
                "worst_delta": min(deltas.values()),
                "sum_delta": sum(deltas.values()),
                "deltas": deltas,
            }
        )
    positive_counts = {
        target: sum(
            value > baseline[target] + 1e-12
            for method, value in accuracy[target].items()
            if pools[target].get(method) == "full_pool"
        )
        for target in targets
    }
    combinations = 1
    for count in positive_counts.values():
        combinations *= count
    return {
        "fixed_method_count": len(fixed_rows),
        "fixed_methods_meeting_all_strict_improvements": sum(
            row["all_strictly_positive"] for row in fixed_rows
        ),
        "best_fixed_methods_by_worst_delta": sorted(
            fixed_rows,
            key=lambda row: (row["worst_delta"], row["sum_delta"], row["method"]),
            reverse=True,
        )[:10],
        "strictly_improving_full_pool_candidate_count_by_target": positive_counts,
        "independent_full_pool_portfolio_combination_count": combinations,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_test_receipt(Path(config["test_receipt"]), args.config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)

    jobs = _dataset_jobs(config)
    required_targets = [str(value) for value in config["acceptance"]["strict_improvement_targets"]]
    if set(jobs) != set(required_targets):
        raise RuntimeError("Configured dataset jobs do not exactly match strict targets")
    portfolios = list(config["portfolios"])
    if args.max_portfolios is not None:
        portfolios = portfolios[: args.max_portfolios]
    if not portfolios:
        raise ValueError("No portfolios are configured")
    names = [str(value["name"]) for value in portfolios]
    if len(names) != len(set(names)):
        raise ValueError("Portfolio names must be unique")
    for portfolio in portfolios:
        if set(portfolio["components"]) != set(required_targets):
            raise ValueError(f"Portfolio components are incomplete: {portfolio['name']}")

    conservative_root = Path(config["conservative_run_root"])
    conservative_prediction_path = conservative_root / "prediction_manifest.json"
    conservative_complete_path = conservative_root / "complete_manifest.json"
    conservative_prediction = json.loads(conservative_prediction_path.read_text(encoding="utf-8"))
    conservative_complete = json.loads(conservative_complete_path.read_text(encoding="utf-8"))
    conservative_prediction_hash = sha256_file(conservative_prediction_path)
    if conservative_prediction_hash != conservative_complete["prediction_manifest_sha256_before_labels"]:
        raise RuntimeError("Conservative prediction manifest is not completion-bound")

    authenticated: dict[str, str] = {}
    manifest_paths: set[Path] = {conservative_prediction_path, conservative_complete_path}
    references: dict[str, dict[int, list[Selection]]] = {}
    candidates: dict[str, dict[str, dict[int, list[Selection]]]] = {
        name: {} for name in names
    }
    output_hashes: dict[str, dict[str, dict[str, dict[str, str]]]] = {
        name: {} for name in names
    }
    for dataset in required_targets:
        job = jobs[dataset]
        references[dataset] = {}
        for name in names:
            candidates[name][dataset] = {}
            output_hashes[name][dataset] = {}
        for seed in job["seeds"]:
            reference = _prior_art_rows(
                job,
                seed,
                "fcrg_full",
                authenticated,
                manifest_paths,
            )
            if args.max_questions is not None:
                reference = reference[: args.max_questions]
            references[dataset][seed] = reference
            reference_ids = {row.question_id for row in reference}
            for portfolio in portfolios:
                name = str(portfolio["name"])
                component = portfolio["components"][dataset]
                source = str(component["source"])
                method = str(component["method"])
                if source == "prior_art":
                    rows = _prior_art_rows(
                        job,
                        seed,
                        method,
                        authenticated,
                        manifest_paths,
                    )
                elif source == "conservative_meta":
                    rows = _conservative_rows(
                        conservative_root,
                        dataset,
                        seed,
                        method,
                        conservative_prediction,
                        conservative_complete,
                        authenticated,
                    )
                else:
                    raise ValueError(f"Unknown component source: {source}")
                if args.max_questions is not None:
                    rows = rows[: args.max_questions]
                if {row.question_id for row in rows} != reference_ids:
                    raise RuntimeError(f"Portfolio/reference IDs differ: {name}/{dataset}/{seed}")
                rows = _portfolio_rows(rows, name, dataset, source, method)
                candidates[name][dataset][seed] = rows
                relative = Path("predictions") / name / dataset / f"seed_{seed}.jsonl"
                digest = write_selections(args.output_dir / relative, rows)
                output_hashes[name][dataset][str(seed)] = {
                    "path": str(relative),
                    "sha256": digest,
                }

    environment = environment_manifest(
        sys.argv,
        int(config["protocol_seed"]),
        [
            args.config,
            Path(config["source_config"]),
            conservative_prediction_path,
            conservative_complete_path,
            *[Path(panel["config"]) for panel in config["target_panels"]],
            *sorted(manifest_paths),
        ],
    )
    environment["authenticated_input_predictions"] = authenticated
    write_json(args.output_dir / "environment.json", environment)
    prediction_manifest = {
        "protocol": config["protocol_name"],
        "scope": config["scope"],
        "portfolios": {str(row["name"]): row["components"] for row in portfolios},
        "predictions": output_hashes,
        "conservative_prediction_manifest_sha256": conservative_prediction_hash,
        "selection_provenance": config["selection_provenance"],
        "labels_opened": False,
        "written_before_label_adapters": True,
        "innovation_code_manifest_sha256": environment["innovation_code_manifest_sha256"],
    }
    write_json(args.output_dir / "prediction_manifest.json", prediction_manifest)
    prediction_manifest_hash = sha256_file(args.output_dir / "prediction_manifest.json")

    labels_by_dataset = {
        dataset: _load_labels(jobs[dataset], args.max_questions) for dataset in required_targets
    }
    aggregate_rows: list[dict[str, Any]] = []
    for name in names:
        for dataset in required_targets:
            job = jobs[dataset]
            labels = labels_by_dataset[dataset]
            reference_matrix = np.stack(
                [
                    _correctness_for_labels(references[dataset][seed], labels)
                    for seed in job["seeds"]
                ]
            )
            candidate_matrix = np.stack(
                [
                    _correctness_for_labels(candidates[name][dataset][seed], labels)
                    for seed in job["seeds"]
                ]
            )
            aggregate_rows.append(
                _aggregate_comparison(name, dataset, candidate_matrix, reference_matrix)
            )
    write_json(args.output_dir / "aggregate_summary.json", aggregate_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)

    goal_rows: list[dict[str, Any]] = []
    portfolio_by_name = {str(row["name"]): row for row in portfolios}
    for name in names:
        by_target = {
            str(row["target"]): row for row in aggregate_rows if row["method"] == name
        }
        goal_rows.append(
            {
                "method": name,
                "components": portfolio_by_name[name]["components"],
                "selection_provenance": config["selection_provenance"],
                **evaluate_strict_improvement_contract(by_target, config["acceptance"]),
            }
        )
    write_json(args.output_dir / "strict_improvement_matrix.json", goal_rows)
    write_csv(
        args.output_dir / "strict_improvement_matrix.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in goal_rows
        ],
    )
    audit = _fixed_method_audit(config["aggregate_audit_inputs"], required_targets)
    write_json(args.output_dir / "fixed_method_and_search_audit.json", audit)
    strict = [row for row in goal_rows if row["strict_user_goal_met"]]
    decision = {
        "scope": config["scope"],
        "prediction_manifest_sha256_before_labels": prediction_manifest_hash,
        "strict_improvement_target_count": len(required_targets),
        "portfolio_count": len(names),
        "strict_goal_met_count": len(strict),
        "strict_goal_met_methods": [row["method"] for row in strict],
        "all_non_test_datasets_strictly_improve": bool(strict),
        "target_labels_used_for_question_predictions": False,
        "dataset_method_mapping_selected_from_known_development_results": True,
        "can_authorize_locked_test": False,
    }
    write_json(args.output_dir / "decision.json", decision)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "prediction_manifest_sha256": prediction_manifest_hash,
            "labels_opened_after_prediction_manifest": True,
            "datasets": required_targets,
        },
    )
    completion_paths = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "complete_manifest.json"
    ]
    write_json(
        args.output_dir / "complete_manifest.json",
        {
            "runtime_seconds": time.time() - started,
            "prediction_manifest_sha256_before_labels": prediction_manifest_hash,
            "artifact_hashes": files_manifest(completion_paths),
            "decision": decision,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), **decision}, indent=2))


if __name__ == "__main__":
    main()
