from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    environment_manifest,
    files_manifest,
    read_selections,
    sha256_file,
    validate_test_receipt,
    write_csv,
    write_json,
    write_jsonl,
)
from .data import CacheAdapter, load_family_map
from .evaluation import (
    evaluate,
    holm_adjust,
    paired_selection_comparison,
    selection_correctness,
)
from .locked_protocol import (
    create_label_access_marker,
    load_protocol,
    stratified_paired_bootstrap_delta,
    validate_preregistration,
    validate_protocol,
)
from .locked_musr_labels import load_musr_evaluation_answers
from .schema import EvaluationLabels, ObservableQueryBatch, Selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open MuSR labels exactly once after prediction sealing")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load_and_authenticate_predictions(
    config: dict[str, Any], run_root: Path
) -> tuple[dict[str, list[Selection]], dict[str, Any]]:
    seal_path = run_root / "prediction_seal.json"
    manifest_path = run_root / "selection_manifest.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if seal.get("status") != "prediction_boundary_sealed":
        raise PermissionError("Prediction package has no completed boundary seal")
    if seal.get("selection_manifest_sha256") != sha256_file(manifest_path):
        raise PermissionError("Selection manifest changed after sealing")
    if seal.get("target_observable_manifest_sha256") != sha256_file(
        run_root / "target_observables" / "observable_manifest.json"
    ):
        raise PermissionError("Target observables changed after sealing")
    if list(seal.get("methods", [])) != list(config["methods"]):
        raise PermissionError("Sealed methods differ from preregistration")
    predictions: dict[str, list[Selection]] = {}
    for method in config["methods"]:
        relative = manifest.get("prediction_paths", {}).get(method)
        expected_hash = seal.get("prediction_hashes", {}).get(method)
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise PermissionError(f"Sealed prediction metadata is missing for {method}")
        path = run_root / relative
        if sha256_file(path) != expected_hash:
            raise PermissionError(f"Prediction changed after sealing: {method}")
        predictions[method] = read_selections(path)
    return predictions, seal


def _ensure_single_use(config: dict[str, Any], run_root: Path) -> None:
    if (run_root / "label_access_started.json").exists():
        raise RuntimeError("This locked test has already been opened and cannot be rerun")
    if (run_root / "evaluation").exists():
        raise RuntimeError("An evaluation directory already exists")
    ledger_path = Path(str(config["single_use_ledger"]))
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        consumed = {str(item["experiment_id"]) for item in ledger.get("consumed", [])}
        if str(config["experiment_id"]) in consumed:
            raise RuntimeError("Experiment ID is already present in the single-use ledger")


def _target_batch(config: dict[str, Any], run_root: Path) -> ObservableQueryBatch:
    target_cache = run_root / "target_observables"
    manifest = target_cache / "observable_manifest.json"
    experts = [str(value) for value in config["experts"]]
    batch = CacheAdapter.from_target_observables(
        target_cache,
        str(config["target"]["dataset"]),
        str(config["target"]["split"]),
        str(config["target"]["modality"]),
        load_family_map(Path(str(config["family_map"]))),
        experts,
        sha256_file(manifest),
    ).load_observables()
    if len(batch.question_ids) != int(config["target"]["expected_questions"]):
        raise RuntimeError("Target batch count differs from frozen protocol")
    return batch


def _evaluation_labels(
    config: dict[str, Any], batch: ObservableQueryBatch
) -> tuple[EvaluationLabels, dict[str, str]]:
    answers = load_musr_evaluation_answers(config["target"]["raw_files"])
    if set(answers) != set(batch.question_ids):
        raise RuntimeError("Evaluator labels and sealed prediction IDs are not aligned")
    correctness = {
        (record.question_id, record.expert_id): bool(
            record.valid_output and record.normalized_answer == answers[record.question_id]
        )
        for record in batch.records
    }
    return EvaluationLabels("musr", "test", correctness), answers


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _task_by_question(batch: ObservableQueryBatch) -> dict[str, str]:
    return {question_id: batch.for_question(question_id)[0].subject for question_id in batch.question_ids}


def _method_task_accuracy(
    selections: list[Selection], labels: EvaluationLabels, task_by_question: dict[str, str]
) -> dict[str, float]:
    correctness = selection_correctness(selections, labels)
    by_task: dict[str, list[bool]] = {}
    for question_id, value in correctness.items():
        by_task.setdefault(task_by_question[question_id], []).append(value)
    return {task: sum(values) / len(values) for task, values in sorted(by_task.items())}


def _comparison(
    name: str,
    candidate_name: str,
    reference_name: str,
    predictions: dict[str, list[Selection]],
    labels: EvaluationLabels,
    task_by_question: dict[str, str],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    result = paired_selection_comparison(
        name,
        predictions[candidate_name],
        predictions[reference_name],
        labels,
        seed=seed,
        bootstrap_samples=samples,
    )
    candidate = selection_correctness(predictions[candidate_name], labels)
    reference = selection_correctness(predictions[reference_name], labels)
    result.update(
        {
            "candidate": candidate_name,
            "reference": reference_name,
            "stratified_paired_bootstrap_delta_ci95": list(
                stratified_paired_bootstrap_delta(
                    candidate,
                    reference,
                    task_by_question,
                    seed=seed,
                    samples=samples,
                )
            ),
        }
    )
    return result


def _plot_results(
    output_dir: Path,
    method_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    primary_candidate: str,
    primary_reference: str,
) -> None:
    matplotlib_cache = Path("/tmp/benchcoe_locked_musr_matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["method"]) for row in method_rows]
    values = [100.0 * float(row["accuracy"]) for row in method_rows]
    lower = [100.0 * (float(row["accuracy"]) - float(row["wilson_ci95"][0])) for row in method_rows]
    upper = [100.0 * (float(row["wilson_ci95"][1]) - float(row["accuracy"])) for row in method_rows]
    colors = [
        "#187A6B" if name == primary_candidate else "#D97706" if name == primary_reference else "#5B6470"
        for name in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    positions = list(range(len(names)))
    axes[0].barh(positions, values, xerr=[lower, upper], color=colors, alpha=0.92, capsize=3)
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy (%) with Wilson 95% CI")
    axes[0].set_title("Frozen MuSR test accuracy")
    axes[0].grid(axis="x", alpha=0.22)
    for position, value in zip(positions, values, strict=True):
        axes[0].text(value + 0.15, position, f"{value:.2f}", va="center", fontsize=8)

    comparison_names = [f"{row['candidate']} vs {row['reference']}" for row in comparisons]
    deltas = [100.0 * float(row["delta"]) for row in comparisons]
    lows = [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][0]) for row in comparisons]
    highs = [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][1]) for row in comparisons]
    errors = [[delta - low for delta, low in zip(deltas, lows, strict=True)], [high - delta for delta, high in zip(deltas, highs, strict=True)]]
    comparison_colors = ["#187A6B" if index == 0 else "#7A8694" for index in range(len(deltas))]
    positions = list(range(len(comparison_names)))
    for index, (delta, position) in enumerate(zip(deltas, positions, strict=True)):
        axes[1].errorbar(
            [delta],
            [position],
            xerr=[[errors[0][index]], [errors[1][index]]],
            fmt="o",
            color="#25313C",
            ecolor=comparison_colors[index],
            capsize=4,
        )
    axes[1].axvline(0.0, color="#B91C1C", linewidth=1.2)
    axes[1].set_yticks(positions, comparison_names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Paired accuracy delta (percentage points)")
    axes[1].set_title("Preregistered paired comparisons")
    axes[1].grid(axis="x", alpha=0.22)
    fig.suptitle("Bench-CoE locked MuSR evaluation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "locked_musr_results.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "locked_musr_results.pdf", bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    config: dict[str, Any],
    method_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Locked MuSR confirmatory result",
        "",
        f"Status: **{decision['decision']}**",
        "",
        "This is a single experimenter-blind evaluation. MuSR labels were opened only after all "
        "expert and selector predictions were hashed. It does not establish that the benchmark "
        "was absent from model pretraining.",
        "",
        "## Accuracy",
        "",
        "| Method | Calls | Accuracy | 95% Wilson CI | Task macro | Delta vs 14-call source vote |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        ci = row["wilson_ci95"]
        lines.append(
            f"| {row['method']} | {row['nominal_model_calls']} | {100*row['accuracy']:.2f}% | "
            f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]% | {100*row['task_macro_accuracy']:.2f}% | "
            f"{100*row['delta_vs_primary_reference']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Comparison | Delta | Stratified 95% CI | Rescue/Harm | McNemar p | Holm p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        ci = row["stratified_paired_bootstrap_delta_ci95"]
        holm = row.get("holm_adjusted_p")
        lines.append(
            f"| {row['candidate']} vs {row['reference']} | {100*row['delta']:+.2f} pp | "
            f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp | "
            f"{row['rescue_count']}/{row['harm_count']} | {row['exact_mcnemar_p']:.4g} | "
            f"{'N/A' if holm is None else f'{holm:.4g}'} |"
        )
    lines.extend(
        [
            "",
            "## Confirmatory decision",
            "",
            f"The frozen primary method was `{config['hypotheses']['primary']['candidate']}` and the "
            f"equal-budget reference was `{config['hypotheses']['primary']['reference']}`. "
            f"The preregistered superiority rule is **{'met' if decision['primary_success'] else 'not met'}**.",
            "",
            "All earlier V1-V4 benchmark portfolios remain development-only and are not combined "
            "with this result as confirmatory evidence. All methods and negative comparisons above "
            "are retained without post-hoc method replacement.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_ledger(config: dict[str, Any], run_root: Path, evaluation_seal: Path) -> None:
    ledger_path = Path(str(config["single_use_ledger"]))
    ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else {"schema_version": 1, "consumed": []}
    )
    ledger["consumed"].append(
        {
            "experiment_id": config["experiment_id"],
            "dataset": "musr",
            "split": "test",
            "consumed_unix": time.time(),
            "run_root": str(run_root),
            "evaluation_seal_sha256": sha256_file(evaluation_seal),
        }
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
    write_json(temporary, ledger)
    os.replace(temporary, ledger_path)


def main() -> None:
    args = parse_args()
    config = load_protocol(args.config)
    validate_protocol(config)
    validate_test_receipt(Path(str(config["test_receipt"])), args.config)
    run_root = Path(str(config["output_root"]))
    prereg = validate_preregistration(args.config, run_root)
    _ensure_single_use(config, run_root)
    predictions, prediction_seal = _load_and_authenticate_predictions(config, run_root)
    batch = _target_batch(config, run_root)
    expected_ids = set(batch.question_ids)
    for method, selections in predictions.items():
        if {selection.question_id for selection in selections} != expected_ids:
            raise RuntimeError(f"Sealed method has incomplete IDs: {method}")

    prediction_seal_path = run_root / "prediction_seal.json"
    marker = create_label_access_marker(run_root, sha256_file(prediction_seal_path))
    labels, _ = _evaluation_labels(config, batch)
    output_dir = run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=False)
    task_by_question = _task_by_question(batch)
    baseline_name = "source_best_single"
    primary_reference = str(config["hypotheses"]["primary"]["reference"])
    reference_correct = selection_correctness(predictions[primary_reference], labels)
    method_rows: list[dict[str, Any]] = []
    per_query_hashes: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    samples = int(config["hypotheses"]["bootstrap_samples"])
    seed = int(config["hypotheses"]["bootstrap_seed"])
    for index, (method, method_spec) in enumerate(config["methods"].items()):
        summary, per_query = evaluate(
            method,
            predictions[method],
            predictions[baseline_name],
            batch,
            labels,
            bootstrap_samples=samples,
            seed=seed + index,
        )
        correctness = selection_correctness(predictions[method], labels)
        successes = sum(correctness.values())
        task_accuracy = _method_task_accuracy(predictions[method], labels, task_by_question)
        row = {
            "method": method,
            "role": method_spec["role"],
            "nominal_model_calls": int(method_spec["nominal_model_calls"]),
            "samples": len(correctness),
            "correct": successes,
            "accuracy": successes / len(correctness),
            "wilson_ci95": _wilson_interval(successes, len(correctness)),
            "task_macro_accuracy": sum(task_accuracy.values()) / len(task_accuracy),
            "per_task_accuracy": task_accuracy,
            "delta_vs_primary_reference": (
                sum(float(correctness[qid]) - float(reference_correct[qid]) for qid in correctness)
                / len(correctness)
            ),
            "delta_vs_source_best_single": summary["delta_vs_source_best_single"],
            "oracle_accuracy": summary["oracle_accuracy"],
            "switch_count_vs_source_best": summary["switch_count"],
        }
        method_rows.append(row)
        summaries[method] = summary
        per_query_path = output_dir / "per_query" / f"{method}.jsonl"
        write_jsonl(per_query_path, per_query)
        per_query_hashes[method] = sha256_file(per_query_path)

    primary = config["hypotheses"]["primary"]
    comparison_specs = [
        (str(primary["candidate"]), str(primary["reference"]), "primary")
    ] + [
        (str(pair[0]), str(pair[1]), "secondary")
        for pair in config["hypotheses"]["secondary_comparisons"]
    ]
    comparisons: list[dict[str, Any]] = []
    for index, (candidate, reference, family) in enumerate(comparison_specs):
        row = _comparison(
            f"{candidate}_vs_{reference}",
            candidate,
            reference,
            predictions,
            labels,
            task_by_question,
            seed + 100 + index,
            samples,
        )
        row["family"] = family
        comparisons.append(row)
    secondary = [row for row in comparisons if row["family"] == "secondary"]
    adjusted = holm_adjust({str(row["comparison"]): float(row["exact_mcnemar_p"]) for row in secondary})
    for row in secondary:
        row.update(adjusted[str(row["comparison"])])
    primary_result = comparisons[0]
    stratified_ci = primary_result["stratified_paired_bootstrap_delta_ci95"]
    primary_success = bool(
        primary_result["delta"] > 0.0
        and primary_result["exact_mcnemar_p"] < 0.05
        and stratified_ci[0] > 0.0
    )
    decision = {
        "primary_success": primary_success,
        "decision": "PRIMARY SUPERIORITY CONFIRMED" if primary_success else "PRIMARY SUPERIORITY NOT CONFIRMED",
        "frozen_success_rule": primary["success_rule"],
        "negative_results_preserved": True,
        "no_posthoc_replacement": True,
    }

    write_json(output_dir / "method_results.json", method_rows)
    write_csv(output_dir / "method_results.csv", method_rows)
    write_json(output_dir / "comparisons.json", comparisons)
    write_csv(output_dir / "comparisons.csv", comparisons)
    write_json(output_dir / "full_method_diagnostics.json", summaries)
    write_json(output_dir / "confirmatory_decision.json", decision)
    _plot_results(
        output_dir,
        method_rows,
        comparisons,
        str(primary["candidate"]),
        str(primary["reference"]),
    )
    report_path = output_dir / "LOCKED_MUSR_RESULTS.md"
    report_path.write_text(
        _markdown_report(config, method_rows, comparisons, decision), encoding="utf-8"
    )
    run_environment = environment_manifest(
        sys.argv,
        seed,
        [args.config, prediction_seal_path, run_root / "selection_manifest.json"],
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "locked_test_consumed_once",
        "questions": len(batch.question_ids),
        "statistical_unit": config["target"]["statistical_unit"],
        "task_counts": {
            task: list(task_by_question.values()).count(task)
            for task in sorted(set(task_by_question.values()))
        },
        "protocol_sha256": prereg["protocol_sha256"],
        "prediction_seal_sha256": sha256_file(prediction_seal_path),
        "label_access_marker_sha256": sha256_file(marker),
        "per_query_hashes": per_query_hashes,
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
        "environment": run_environment,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    write_json(manifest_path, manifest)
    artifact_hashes = files_manifest([output_dir])
    evaluation_seal = output_dir / "evaluation_seal.json"
    write_json(
        evaluation_seal,
        {
            "experiment_id": config["experiment_id"],
            "status": "immutable_evaluation_seal",
            "sealed_unix": time.time(),
            "artifact_hashes_before_seal": artifact_hashes,
            "prediction_seal_sha256": sha256_file(prediction_seal_path),
            "rerun_allowed": False,
        },
    )
    _append_ledger(config, run_root, evaluation_seal)
    print(report_path.read_text(encoding="utf-8"))
    print(f"Evaluation sealed: {evaluation_seal}")


if __name__ == "__main__":
    main()
