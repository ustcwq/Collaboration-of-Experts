from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _target_names(raw: Any, field: str) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"acceptance.{field} must be a sequence of target names")
    names = [str(value) for value in raw]
    if not names or any(not value for value in names) or len(names) != len(set(names)):
        raise ValueError(f"acceptance.{field} must contain unique non-empty target names")
    return names


def validate_acceptance_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    primary_target = str(contract.get("primary_target", ""))
    if not primary_target:
        raise ValueError("acceptance.primary_target must be non-empty")
    relative_targets = _target_names(
        contract.get("relative_non_regression_targets"),
        "relative_non_regression_targets",
    )
    if primary_target in relative_targets:
        raise ValueError("The primary target must not be a relative non-regression target")

    raw_floors = contract.get("absolute_accuracy_floors")
    if not isinstance(raw_floors, Mapping) or not raw_floors:
        raise ValueError("acceptance.absolute_accuracy_floors must be a non-empty mapping")
    floors: dict[str, float] = {}
    for target, raw_floor in raw_floors.items():
        name = str(target)
        floor = float(raw_floor)
        if not name or not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"Invalid absolute accuracy floor for {target!r}: {raw_floor!r}")
        floors[name] = floor
    if not set(floors).issubset(relative_targets):
        raise ValueError("Every absolute-floor target must also require relative non-regression")

    minimum_primary_delta = float(contract.get("minimum_primary_delta", 0.0))
    tolerance = float(contract.get("tolerance", 1e-12))
    if not math.isfinite(minimum_primary_delta):
        raise ValueError("acceptance.minimum_primary_delta must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("acceptance.tolerance must be finite and non-negative")
    return {
        "primary_target": primary_target,
        "minimum_primary_delta": minimum_primary_delta,
        "relative_non_regression_targets": relative_targets,
        "absolute_accuracy_floors": floors,
        "tolerance": tolerance,
    }


def evaluate_acceptance_contract(
    by_target: Mapping[str, Mapping[str, Any]],
    raw_contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract = validate_acceptance_contract(raw_contract)
    primary_target = contract["primary_target"]
    relative_targets = contract["relative_non_regression_targets"]
    floors = contract["absolute_accuracy_floors"]
    tolerance = float(contract["tolerance"])
    required_targets = [primary_target, *relative_targets]
    missing_targets = [target for target in required_targets if target not in by_target]

    accuracies: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for target, row in by_target.items():
        accuracy = float(row["accuracy_mean"])
        delta = float(row["delta_vs_fcrg_full"])
        if not math.isfinite(accuracy) or not math.isfinite(delta):
            raise ValueError(f"Non-finite aggregate result for {target}")
        accuracies[str(target)] = accuracy
        deltas[str(target)] = delta

    primary_delta = deltas.get(primary_target)
    primary_improves = bool(
        primary_delta is not None
        and primary_delta > float(contract["minimum_primary_delta"]) + tolerance
    )
    relative_checks = {
        target: {
            "delta_vs_fcrg_full": deltas.get(target),
            "minimum_delta": 0.0,
            "met": bool(target in deltas and deltas[target] >= -tolerance),
        }
        for target in relative_targets
    }
    floor_checks = {
        target: {
            "accuracy": accuracies.get(target),
            "floor": floor,
            "margin": (
                accuracies[target] - floor if target in accuracies else None
            ),
            "met": bool(target in accuracies and accuracies[target] >= floor - tolerance),
        }
        for target, floor in floors.items()
    }
    relative_met = bool(relative_checks) and all(
        bool(check["met"]) for check in relative_checks.values()
    )
    floors_met = bool(floor_checks) and all(
        bool(check["met"]) for check in floor_checks.values()
    )
    all_present = not missing_targets
    return {
        "primary_target": primary_target,
        "minimum_primary_delta": float(contract["minimum_primary_delta"]),
        "primary_target_delta_vs_fcrg_full": primary_delta,
        "primary_target_improves": primary_improves,
        "required_targets": required_targets,
        "missing_required_targets": missing_targets,
        "all_required_targets_present": all_present,
        "all_dataset_accuracy": accuracies,
        "all_dataset_delta_vs_fcrg_full": deltas,
        "relative_non_regression_checks": relative_checks,
        "relative_non_regression_met": relative_met,
        "absolute_accuracy_floor_checks": floor_checks,
        "absolute_accuracy_floors_met": floors_met,
        "strict_user_goal_met": bool(
            all_present and primary_improves and relative_met and floors_met
        ),
        "worst_required_delta_vs_fcrg_full": (
            min(deltas[target] for target in required_targets if target in deltas)
            if any(target in deltas for target in required_targets)
            else None
        ),
    }


def evaluate_strict_improvement_contract(
    by_target: Mapping[str, Mapping[str, Any]],
    raw_contract: Mapping[str, Any],
) -> dict[str, Any]:
    required_targets = _target_names(
        raw_contract.get("strict_improvement_targets"),
        "strict_improvement_targets",
    )
    minimum_delta = float(raw_contract.get("minimum_delta", 0.0))
    tolerance = float(raw_contract.get("tolerance", 1e-12))
    if not math.isfinite(minimum_delta):
        raise ValueError("acceptance.minimum_delta must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("acceptance.tolerance must be finite and non-negative")
    raw_floors = raw_contract.get("absolute_accuracy_floors", {})
    if not isinstance(raw_floors, Mapping):
        raise ValueError("acceptance.absolute_accuracy_floors must be a mapping")
    floors = {str(target): float(value) for target, value in raw_floors.items()}
    for target, floor in floors.items():
        if target not in required_targets:
            raise ValueError("Every absolute-floor target must require strict improvement")
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"Invalid absolute accuracy floor for {target}: {floor}")

    missing = [target for target in required_targets if target not in by_target]
    accuracies: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for target, row in by_target.items():
        accuracy = float(row["accuracy_mean"])
        delta = float(row["delta_vs_fcrg_full"])
        if not math.isfinite(accuracy) or not math.isfinite(delta):
            raise ValueError(f"Non-finite aggregate result for {target}")
        accuracies[str(target)] = accuracy
        deltas[str(target)] = delta
    strict_checks = {
        target: {
            "delta_vs_fcrg_full": deltas.get(target),
            "minimum_delta_exclusive": minimum_delta,
            "met": bool(
                target in deltas and deltas[target] > minimum_delta + tolerance
            ),
        }
        for target in required_targets
    }
    floor_checks = {
        target: {
            "accuracy": accuracies.get(target),
            "floor": floor,
            "margin": accuracies[target] - floor if target in accuracies else None,
            "met": bool(target in accuracies and accuracies[target] >= floor - tolerance),
        }
        for target, floor in floors.items()
    }
    all_present = not missing
    all_strict = bool(strict_checks) and all(check["met"] for check in strict_checks.values())
    all_floors = all(check["met"] for check in floor_checks.values())
    return {
        "required_strict_improvement_targets": required_targets,
        "minimum_delta_exclusive": minimum_delta,
        "missing_required_targets": missing,
        "all_required_targets_present": all_present,
        "all_dataset_accuracy": accuracies,
        "all_dataset_delta_vs_fcrg_full": deltas,
        "strict_improvement_checks": strict_checks,
        "all_datasets_strictly_improve": all_strict,
        "absolute_accuracy_floor_checks": floor_checks,
        "absolute_accuracy_floors_met": all_floors,
        "strict_user_goal_met": bool(all_present and all_strict and all_floors),
        "worst_required_delta_vs_fcrg_full": (
            min(deltas[target] for target in required_targets if target in deltas)
            if any(target in deltas for target in required_targets)
            else None
        ),
    }
