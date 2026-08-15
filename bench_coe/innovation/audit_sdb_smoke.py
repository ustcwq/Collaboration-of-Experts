from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .artifacts import sha256_file, write_json
from .blind_falsification_jury import FORBIDDEN_AUDIT_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen label-free SDB smoke")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("SDB configuration must be a mapping")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON objects in {path}")
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise PermissionError(
                        f"SDB smoke artifact contains labels: {sorted(leaked)}"
                    )
                rows.append(row)
    return rows


def _single_manifest_by_identity(
    paths: list[Path], identity_key: str, expected: tuple[str, ...]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = str(manifest[identity_key])
        if identity in result:
            raise RuntimeError(f"Duplicate SDB smoke manifest for {identity}")
        result[identity] = (path, manifest)
    if set(result) != set(expected):
        raise RuntimeError(
            f"SDB smoke identities differ: expected {sorted(expected)}, got {sorted(result)}"
        )
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / max(1, denominator)


def _pairwise_agreement(sides: list[str]) -> tuple[int, int]:
    agreeing = 0
    pairs = 0
    for left_index in range(len(sides)):
        for right_index in range(left_index + 1, len(sides)):
            pairs += 1
            agreeing += int(sides[left_index] == sides[right_index])
    return agreeing, pairs


def audit_smoke(
    config_path: Path, config: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    authors = tuple(str(value) for value in config["author_models"])
    checkers = tuple(str(value) for value in config["checker_models"])
    gates = config["smoke_acceptance"]
    probe_manifests = _single_manifest_by_identity(
        sorted((run_root / "smoke" / "probes").glob("*/probe_manifest.json")),
        "author",
        authors,
    )
    check_manifests = _single_manifest_by_identity(
        sorted((run_root / "smoke" / "checks").glob("*/check_manifest.json")),
        "checker",
        checkers,
    )

    probe_rows: dict[str, dict[str, Any]] = {}
    author_results: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, str] = {str(config_path): sha256_file(config_path)}
    author_gate_pass = True
    for author, (manifest_path, manifest) in sorted(probe_manifests.items()):
        path = manifest_path.parent / "probes.jsonl"
        if manifest.get("probe_sha256") != sha256_file(path):
            raise PermissionError("SDB smoke probe hash changed")
        rows = _read_jsonl(path)
        for row in rows:
            probe_id = str(row["probe_id"])
            if probe_id in probe_rows:
                raise RuntimeError("Duplicate SDB smoke probe ID")
            probe_rows[probe_id] = row
        questions = int(manifest["questions"])
        parsed = int(manifest["parsed_probes"])
        nonabstaining = int(manifest["nonabstaining_probes"])
        bijections = int(manifest["mapping_bijections"])
        side_counts = {
            str(key): int(value)
            for key, value in manifest["presented_left_authored_outcome_counts"].items()
        }
        dataset_count = len({str(row["dataset"]) for row in rows})
        result = {
            "questions": questions,
            "probe_parse_rate": _safe_rate(parsed, questions),
            "nonabstaining_probe_rate": _safe_rate(nonabstaining, questions),
            "mapping_bijection_rate": _safe_rate(bijections, nonabstaining),
            "minimum_presented_side_fraction": min(
                _safe_rate(side_counts.get(str(side), 0), nonabstaining)
                for side in (1, 2)
            ),
            "dataset_count": dataset_count,
            "truncated_model_calls": int(manifest["truncated_model_calls"]),
            "labels_read": manifest.get("labels_read"),
            "gate_pass": False,
        }
        result["gate_pass"] = bool(
            questions == int(gates["questions_per_author"])
            and result["probe_parse_rate"] >= float(gates["minimum_probe_parse_rate"])
            and result["nonabstaining_probe_rate"]
            >= float(gates["minimum_nonabstaining_probe_rate"])
            and result["mapping_bijection_rate"]
            >= float(gates["minimum_mapping_bijection_rate"])
            and result["minimum_presented_side_fraction"]
            >= float(gates["minimum_presented_side_fraction"])
            and result["truncated_model_calls"]
            <= int(gates["maximum_prompt_truncations"])
            and result["labels_read"] is False
            and (
                not bool(gates["require_two_development_datasets_per_author"])
                or dataset_count >= 2
            )
            and manifest.get("mapping_was_sealed_from_checkers") is True
            and manifest.get("original_task_was_sealed_from_checkers") is True
            and manifest.get("post_commit_permutation") is True
        )
        author_gate_pass = author_gate_pass and bool(result["gate_pass"])
        author_results[author] = result
        artifact_hashes[str(path)] = sha256_file(path)
        artifact_hashes[str(manifest_path)] = sha256_file(manifest_path)

    checks_by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checker_results: dict[str, dict[str, Any]] = {}
    checker_gate_pass = True
    for checker, (manifest_path, manifest) in sorted(check_manifests.items()):
        raw_path = manifest_path.parent / "raw_checks.jsonl"
        path = manifest_path.parent / "checks.jsonl"
        if manifest.get("raw_checks_sha256") != sha256_file(raw_path):
            raise PermissionError("SDB smoke raw-check hash changed")
        if manifest.get("check_sha256") != sha256_file(path):
            raise PermissionError("SDB smoke revealed-check hash changed")
        rows = _read_jsonl(path)
        for row in rows:
            if row.get("frozen_raw_checks_sha256") != manifest["raw_checks_sha256"]:
                raise PermissionError("SDB reveal does not bind the frozen raw-check hash")
            if row.get("mapping_revealed_after_raw_outputs_frozen") is not True:
                raise PermissionError("SDB mapping was revealed before raw-output freeze")
            selected = row.get("selected_candidate")
            rejected = row.get("rejected_candidate")
            if selected is not None and selected == rejected:
                raise PermissionError("SDB reveal is not a candidate bijection")
            checks_by_probe[str(row["probe_id"])].append(row)
        total = int(manifest["model_calls"])
        parsed = int(manifest["parsed_checks"])
        decided = int(manifest["decided_checks"])
        side_counts = Counter(
            str(row["outcome_side"])
            for row in rows
            if row.get("parse_error") is None and not bool(row.get("uncertain"))
        )
        largest_side_rate = max(
            (_safe_rate(side_counts[side], decided) for side in ("LEFT", "RIGHT")),
            default=0.0,
        )
        result = {
            "model_calls": total,
            "check_parse_rate": _safe_rate(parsed, total),
            "decided_check_rate": _safe_rate(decided, total),
            "largest_presented_side_selection_rate": largest_side_rate,
            "truncated_model_calls": int(manifest["truncated_model_calls"]),
            "labels_read": manifest.get("labels_read"),
            "gate_pass": False,
        }
        result["gate_pass"] = bool(
            result["check_parse_rate"] >= float(gates["minimum_check_parse_rate"])
            and result["decided_check_rate"] >= float(gates["minimum_decided_check_rate"])
            and result["largest_presented_side_selection_rate"]
            <= float(gates["maximum_single_presented_side_selection_rate"])
            and result["truncated_model_calls"]
            <= int(gates["maximum_prompt_truncations"])
            and result["labels_read"] is False
            and manifest.get("original_task_was_hidden") is True
            and manifest.get("candidate_pair_was_hidden") is True
            and manifest.get("outcome_mapping_was_hidden") is True
            and manifest.get("mapping_revealed_after_raw_outputs_frozen") is True
        )
        checker_gate_pass = checker_gate_pass and bool(result["gate_pass"])
        checker_results[checker] = result
        artifact_hashes[str(raw_path)] = sha256_file(raw_path)
        artifact_hashes[str(path)] = sha256_file(path)
        artifact_hashes[str(manifest_path)] = sha256_file(manifest_path)

    nonabstaining_probe_ids = {
        probe_id
        for probe_id, row in probe_rows.items()
        if row.get("parse_error") is None and not bool(row.get("abstained"))
    }
    complete = 0
    agreement_eligible = 0
    agreeing_pairs = 0
    checker_pairs = 0
    for probe_id in sorted(nonabstaining_probe_ids):
        author = str(probe_rows[probe_id]["author_id"])
        expected_checkers = {checker for checker in checkers if checker != author}
        rows = checks_by_probe.get(probe_id, [])
        actual_checkers = {str(row["checker_id"]) for row in rows}
        complete += int(actual_checkers == expected_checkers)
        sides = [
            str(row["outcome_side"])
            for row in rows
            if row.get("parse_error") is None and not bool(row.get("uncertain"))
        ]
        if len(sides) >= 2:
            agreement_eligible += 1
            agreeing, pairs = _pairwise_agreement(sides)
            agreeing_pairs += agreeing
            checker_pairs += pairs
    complete_rate = _safe_rate(complete, len(nonabstaining_probe_ids))
    eligible_rate = _safe_rate(agreement_eligible, len(nonabstaining_probe_ids))
    agreement_rate = _safe_rate(agreeing_pairs, checker_pairs)
    joint_result = {
        "nonabstaining_probes": len(nonabstaining_probe_ids),
        "complete_cross_checked_probes": complete,
        "complete_cross_check_rate": complete_rate,
        "agreement_eligible_probes": agreement_eligible,
        "agreement_eligible_probe_rate": eligible_rate,
        "agreeing_checker_pairs": agreeing_pairs,
        "checker_pairs": checker_pairs,
        "cross_checker_pairwise_agreement_rate": agreement_rate,
        "gate_pass": bool(
            complete_rate >= float(gates["minimum_complete_cross_check_rate"])
            and eligible_rate >= float(gates["minimum_agreement_eligible_probe_rate"])
            and checker_pairs > 0
            and agreement_rate
            >= float(gates["minimum_cross_checker_pairwise_agreement_rate"])
        ),
    }
    full_run_authorized = bool(
        author_gate_pass and checker_gate_pass and joint_result["gate_pass"]
    )
    return {
        "protocol": str(config["protocol_name"]),
        "status": (
            "passed_label_free_sdb_smoke_gates"
            if full_run_authorized
            else "retained_negative_sdb_smoke_result"
        ),
        "decision_time_unix": time.time(),
        "labels_read_for_generation_or_decision": False,
        "authors": author_results,
        "checkers": checker_results,
        "joint_cross_check": joint_result,
        "policy": {
            "parser_weakened_post_hoc": False,
            "smoke_gates_weakened_post_hoc": False,
            "full_run_authorized": full_run_authorized,
            "blind_test_authorized": False,
            "failed_smoke_must_use_new_output_paths": not full_run_authorized,
        },
        "sha256": artifact_hashes,
    }


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    run_root = args.run_root or Path(str(config["output_root"]))
    output_path = run_root / "smoke" / "smoke_gate.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    result = audit_smoke(args.config, config, run_root)
    if not all(math.isfinite(float(value)) for value in (
        result["joint_cross_check"]["complete_cross_check_rate"],
        result["joint_cross_check"]["agreement_eligible_probe_rate"],
        result["joint_cross_check"]["cross_checker_pairwise_agreement_rate"],
    )):
        raise RuntimeError("SDB smoke audit produced a non-finite rate")
    write_json(output_path, result)
    print(f"Completed SDB smoke audit: {output_path}")
    if not result["policy"]["full_run_authorized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
