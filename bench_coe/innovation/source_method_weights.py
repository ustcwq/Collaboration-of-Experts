from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import Selection, SourceTrainingLabels


def source_method_profiles(
    rows_by_seed: Mapping[int, Mapping[str, Sequence[Selection]]],
    labels: SourceTrainingLabels,
    *,
    reference_method: str = "fcrg_full",
) -> dict[str, dict[str, float]]:
    if not isinstance(labels, SourceTrainingLabels) or labels.role != "source":
        raise TypeError("Source method profiling requires SourceTrainingLabels")
    if not rows_by_seed:
        raise ValueError("Source method profiling requires at least one seed")
    method_sets = [set(rows) for rows in rows_by_seed.values()]
    methods = sorted(set.intersection(*method_sets))
    if reference_method not in methods:
        raise ValueError(f"Missing source reference method: {reference_method}")
    correctness: dict[str, list[bool]] = defaultdict(list)
    environment_accuracy: dict[str, list[float]] = defaultdict(list)
    rescue: dict[str, int] = defaultdict(int)
    harm: dict[str, int] = defaultdict(int)
    for _, rows_by_method in sorted(rows_by_seed.items()):
        indexed = {
            method: {row.question_id: row for row in rows_by_method[method]}
            for method in methods
        }
        ids = set(indexed[reference_method])
        if any(set(indexed[method]) != ids for method in methods):
            raise ValueError("Source method predictions are not aligned")
        by_environment: dict[str, list[str]] = defaultdict(list)
        for question_id in ids:
            by_environment[str(labels.environment_by_question[question_id])].append(question_id)
        reference_values = {
            question_id: bool(
                indexed[reference_method][question_id].selected_expert_id is not None
                and labels.get(
                    question_id,
                    indexed[reference_method][question_id].selected_expert_id or "",
                )
            )
            for question_id in ids
        }
        for method in methods:
            values: dict[str, bool] = {}
            for question_id, row in indexed[method].items():
                value = bool(
                    row.selected_expert_id is not None
                    and labels.get(question_id, row.selected_expert_id or "")
                )
                values[question_id] = value
                correctness[method].append(value)
                rescue[method] += int(value and not reference_values[question_id])
                harm[method] += int(reference_values[question_id] and not value)
            for environment, environment_ids in by_environment.items():
                environment_accuracy[method].append(
                    float(np.mean([values[question_id] for question_id in environment_ids]))
                )
    profiles: dict[str, dict[str, float]] = {}
    for method in methods:
        values = np.asarray(correctness[method], dtype=float)
        environments = np.asarray(environment_accuracy[method], dtype=float)
        accuracy = float(values.mean())
        standard_error = float(np.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / len(values)))
        profiles[method] = {
            "accuracy": accuracy,
            "accuracy_lcb95": max(0.0, accuracy - 1.96 * standard_error),
            "environment_mean": float(environments.mean()),
            "environment_std": float(environments.std()),
            "environment_min": float(environments.min()),
            "rescue_count": float(rescue[method]),
            "harm_count": float(harm[method]),
            "net_delta": float((rescue[method] - harm[method]) / len(values)),
            "switch_precision": float(rescue[method] / max(1, rescue[method] + harm[method])),
            "samples": float(len(values)),
        }
    return profiles


def source_weight_candidates(
    profiles: Mapping[str, Mapping[str, float]],
    *,
    reference_method: str = "fcrg_full",
) -> dict[str, dict[str, float]]:
    methods = tuple(sorted(profiles))
    if reference_method not in profiles:
        raise ValueError(f"Missing source reference profile: {reference_method}")
    accuracy = {method: float(profiles[method]["accuracy"]) for method in methods}
    lcb = {method: float(profiles[method]["accuracy_lcb95"]) for method in methods}
    candidates: dict[str, dict[str, float]] = {}
    for power in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        name = str(power).replace(".", "p")
        candidates[f"accuracy_p{name}"] = {
            method: max(accuracy[method], 1e-6) ** power for method in methods
        }
    for temperature in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
        name = str(temperature).replace(".", "p")
        best = max(accuracy.values())
        candidates[f"accuracy_softmax_t{name}"] = {
            method: float(np.exp(np.clip((accuracy[method] - best) / temperature, -60.0, 0.0)))
            for method in methods
        }
    ranked_accuracy = sorted(methods, key=lambda method: (-accuracy[method], method))
    ranked_lcb = sorted(methods, key=lambda method: (-lcb[method], method))
    for count in (1, 2, 3, 5, 8, 10, 15, 20, 30):
        keep_accuracy = set(ranked_accuracy[:count])
        keep_lcb = set(ranked_lcb[:count])
        candidates[f"accuracy_top{count}"] = {
            method: float(method in keep_accuracy) for method in methods
        }
        candidates[f"lcb_top{count}"] = {
            method: float(method in keep_lcb) for method in methods
        }
    for strength in (0.25, 0.5, 1.0, 2.0):
        name = str(strength).replace(".", "p")
        candidates[f"environment_lcb_s{name}"] = {
            method: max(
                float(profiles[method]["environment_mean"])
                - strength * float(profiles[method]["environment_std"]),
                1e-6,
            )
            for method in methods
        }
    reference_accuracy = accuracy[reference_method]
    for power in (1.0, 2.0, 4.0):
        name = str(power).replace(".", "p")
        candidates[f"positive_delta_p{name}"] = {
            method: (max(accuracy[method] - reference_accuracy, 0.0) + 1e-4) ** power
            for method in methods
        }
    return candidates
