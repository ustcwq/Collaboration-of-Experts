from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import environment_manifest, validate_test_receipt, write_csv, write_json, write_jsonl, write_selections
from .data import CacheAdapter, load_family_map
from .dcrg import DCRGSelector, estimate_rescue_graphs
from .evaluation import evaluate, holm_adjust, paired_selection_comparison
from .schema import EvaluationLabels, Selection
from .selectors import LegacyRepairChainSelector, SourceBestSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-registered source LOSO validation for DCRG")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-environments", type=int)
    return parser.parse_args()


def accuracy_for(selections: list[Selection], correctness: dict[tuple[str, str], bool]) -> float:
    values = [bool(correctness.get((item.question_id, item.selected_expert_id or ""), False)) for item in selections]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    receipt_path = Path(config["test_receipt"])
    validate_test_receipt(receipt_path, args.config)
    seed = int(config.get("seed", 20260808))
    family_map_path = Path(config.get("family_map", "configs/innovation/expert_families.yaml"))
    family_map = load_family_map(family_map_path)
    source_spec = config["source"]
    adapter = CacheAdapter.from_source_registry(
        Path(source_spec["cache_path"]),
        source_spec["dataset"],
        source_spec["split"],
        source_spec["modality"],
        family_map,
        config["experts"],
        Path(config["dataset_registry"]),
        str(config["dataset_registry_sha256"]),
    )
    batch = adapter.load_observables()
    labels = adapter.load_source_labels()
    by_environment: dict[str, list[str]] = defaultdict(list)
    for question_id, environment in labels.environment_by_question.items():
        by_environment[environment].append(question_id)
    environments = sorted(by_environment)
    if args.max_environments is not None:
        environments = environments[: args.max_environments]

    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)
    started = time.perf_counter()
    combined: dict[str, list[Selection]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    fold_by_question: dict[str, tuple[str, int]] = {}
    prediction_hashes: dict[str, str] = {}

    modes = ("raw_c", "c_minus_g", "residual", "stable")
    for fold_index, environment in enumerate(environments):
        test_ids = sorted(by_environment[environment])
        train_ids = sorted(set(batch.question_ids).difference(test_ids))
        train_batch = batch.subset(train_ids)
        test_batch = batch.subset(test_ids)
        train_labels = labels.subset(train_ids)

        source_best = SourceBestSelector().fit(train_batch, train_labels)
        repair_chain = LegacyRepairChainSelector(neighbors=int(config.get("knn_k", 32))).fit(train_batch, train_labels)
        dcrg = DCRGSelector(
            seed=seed + fold_index,
            folds=int(config.get("oof_folds", 5)),
            min_support=int(config.get("min_support", 8)),
            min_environments=int(config.get("min_environments", 3)),
        ).fit(train_batch, train_labels)
        no_difficulty = DCRGSelector(
            seed=seed + fold_index,
            folds=int(config.get("oof_folds", 5)),
            min_support=int(config.get("min_support", 8)),
            min_environments=int(config.get("min_environments", 3)),
            adjust_difficulty=False,
        ).fit(train_batch, train_labels)

        fold_predictions: dict[str, list[Selection]] = {
            "source_best_single": source_best.predict(test_batch),
            "repair_chain": repair_chain.predict(test_batch),
        }
        for mode in modes:
            fold_predictions[f"dcrg_{mode}"] = dcrg.predict_with_mode(test_batch, mode)
        dcrg.randomized_graph = True
        fold_predictions["dcrg_stable_randomized"] = dcrg.predict_with_mode(test_batch, "stable")
        dcrg.randomized_graph = False
        dcrg.two_hop = True
        fold_predictions["dcrg_stable_two_hop"] = dcrg.predict_with_mode(test_batch, "stable")
        dcrg.two_hop = False
        fold_predictions["dcrg_stable_no_difficulty"] = no_difficulty.predict_with_mode(test_batch, "stable")

        environment_vector = np.asarray([train_labels.environment_by_question[qid] for qid in train_batch.question_ids], dtype=str)
        self_graphs, _ = estimate_rescue_graphs(
            dcrg.correctness_,
            dcrg.expected_,
            environment_vector,
            dcrg.experts_,
            min_support=dcrg.min_support,
            min_environments=dcrg.min_environments,
            self_loops=True,
        )
        if not np.allclose(self_graphs["stable"], dcrg.graphs_["stable"]):
            raise AssertionError("Stable self-loop ablation unexpectedly changed the graph")
        fold_predictions["dcrg_stable_self_loop"] = dcrg.predict_with_mode(test_batch, "stable")

        for method, predictions in fold_predictions.items():
            combined[method].extend(predictions)
        for question_id in test_ids:
            fold_by_question[question_id] = (environment, fold_index)
        for edge in dcrg.edge_rows():
            edge_rows.append({"heldout_environment": environment, **edge, "environment_signs": json.dumps(edge["environment_signs"], sort_keys=True)})

    # Persist all source-LOSO predictions before constructing evaluator labels.
    for method, predictions in combined.items():
        combined[method] = sorted(predictions, key=lambda item: item.question_id)
        prediction_hashes[method] = write_selections(output_dir / "predictions" / f"{method}.jsonl", combined[method])
    prediction_manifest = environment_manifest(
        sys.argv,
        seed,
            [
                args.config,
                family_map_path,
                Path(config["dataset_registry"]),
                Path(source_spec["cache_path"]),
                receipt_path,
            ],
    )
    prediction_manifest.update(
        {
            "heldout_environments": environments,
            "prediction_hashes_before_evaluation": prediction_hashes,
            "protocol": "source leave-one-subject-out; each heldout label excluded from fit",
        }
    )
    write_json(output_dir / "prediction_manifest.json", prediction_manifest)

    evaluation_labels = EvaluationLabels(labels.dataset, labels.split, dict(labels.correctness))
    baseline = combined["source_best_single"]
    summaries: list[dict[str, Any]] = []
    for method, predictions in sorted(combined.items()):
        summary, per_query = evaluate(
            method,
            predictions,
            baseline,
            batch.subset(item.question_id for item in predictions),
            evaluation_labels,
            bootstrap_samples=int(config.get("bootstrap_samples", 1000)),
            seed=seed,
        )
        summaries.append(summary)
        write_jsonl(output_dir / "per_query" / f"{method}.jsonl", per_query)
    for method, predictions in sorted(combined.items()):
        by_environment_predictions: dict[str, list[Selection]] = defaultdict(list)
        for selection in predictions:
            by_environment_predictions[fold_by_question[selection.question_id][0]].append(selection)
        for environment, items in sorted(by_environment_predictions.items()):
            environment_rows.append(
                {
                    "environment": environment,
                    "fold": fold_by_question[items[0].question_id][1],
                    "samples": len(items),
                    "method": method,
                    "accuracy": accuracy_for(items, dict(evaluation_labels.correctness)),
                }
            )
    paired = [
        paired_selection_comparison(
            f"{method}_vs_repair_chain",
            predictions,
            combined["repair_chain"],
            evaluation_labels,
            seed=seed,
            bootstrap_samples=int(config.get("bootstrap_samples", 1000)),
        )
        for method, predictions in sorted(combined.items())
        if method not in {"repair_chain", "source_best_single"}
    ]
    corrections = holm_adjust({row["comparison"]: float(row["exact_mcnemar_p"]) for row in paired})
    for row in paired:
        row["holm"] = corrections[row["comparison"]]
    write_json(output_dir / "summary.json", summaries)
    write_csv(output_dir / "summary.csv", [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in summaries])
    write_csv(output_dir / "per_environment.csv", environment_rows)
    write_csv(output_dir / "edges.csv", edge_rows)
    write_json(output_dir / "paired_comparisons.json", paired)
    write_csv(
        output_dir / "paired_comparisons.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in paired],
    )

    env_lookup = {(row["environment"], row["method"]): float(row["accuracy"]) for row in environment_rows}
    deltas = [env_lookup[(environment, "dcrg_stable")] - env_lookup[(environment, "repair_chain")] for environment in environments]
    stable_summary = next(row for row in summaries if row["method"] == "dcrg_stable")
    repair_summary = next(row for row in summaries if row["method"] == "repair_chain")
    gate = {
        "comparison": "dcrg_stable_vs_repair_chain",
        "macro_delta": float(np.mean(deltas)) if deltas else 0.0,
        "micro_delta": float(stable_summary["accuracy"] - repair_summary["accuracy"]),
        "worst_environment_delta": float(min(deltas)) if deltas else 0.0,
        "nonnegative_environment_fraction": float(np.mean(np.asarray(deltas) >= 0.0)) if deltas else 0.0,
        "required_macro_delta": 0.0025,
        "required_worst_delta": -0.005,
        "required_nonnegative_fraction": 2.0 / 3.0,
    }
    stable_paired = next(row for row in paired if row["comparison"] == "dcrg_stable_vs_repair_chain")
    gate["paired_bootstrap_delta_ci95"] = stable_paired["paired_bootstrap_delta_ci95"]
    gate["exact_mcnemar_p"] = stable_paired["exact_mcnemar_p"]
    gate["holm"] = stable_paired["holm"]
    gate["decision"] = (
        "GO"
        if gate["macro_delta"] >= gate["required_macro_delta"]
        and gate["worst_environment_delta"] >= gate["required_worst_delta"]
        and gate["nonnegative_environment_fraction"] >= gate["required_nonnegative_fraction"]
        else "NO-GO"
    )
    gate["runtime_seconds"] = time.perf_counter() - started
    write_json(output_dir / "gate.json", gate)
    print(json.dumps({"output_dir": str(output_dir), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
