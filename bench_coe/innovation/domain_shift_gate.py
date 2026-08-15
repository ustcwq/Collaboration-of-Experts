from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import Selection


DOMAIN_FEATURES = (
    "valid_fraction",
    "cluster_fraction",
    "partition_entropy",
    "top1_share",
    "top2_share",
    "cluster_margin",
    "mean_uncertainty",
    "std_uncertainty",
    "missing_fraction",
    "normalized_answer_clusters",
    "normalized_top_family_breadth",
    "normalized_fcrg_margin",
)


@dataclass(frozen=True)
class DomainPolicy:
    kind: str
    metric: str = "none"
    quantile: float = 1.0

    @property
    def policy_id(self) -> str:
        quantile = str(self.quantile).replace(".", "p")
        return f"{self.kind}__{self.metric}__q{quantile}"


@dataclass(frozen=True)
class RobustReference:
    median: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if self.median.ndim != 1 or self.scale.shape != self.median.shape:
            raise ValueError("Robust reference vectors are not aligned")
        if not np.isfinite(self.median).all() or not np.isfinite(self.scale).all():
            raise ValueError("Robust reference contains non-finite values")
        if np.any(self.scale <= 0.0):
            raise ValueError("Robust reference scales must be positive")


def _normalized_margin(scores: Mapping[str, float]) -> float:
    values = sorted((float(value) for value in scores.values() if np.isfinite(value)), reverse=True)
    if not values:
        return 0.0
    gap = values[0] - (values[1] if len(values) > 1 else 0.0)
    return float(gap / max(abs(values[0]), 1e-8))


def domain_feature_matrix(selections: Sequence[Selection]) -> np.ndarray:
    rows: list[list[float]] = []
    for selection in selections:
        features = selection.observable_features
        valid_experts = max(float(features.get("valid_experts", 0.0)), 1.0)
        rows.append(
            [
                float(features.get("valid_fraction", 0.0)),
                float(features.get("cluster_fraction", 0.0)),
                float(features.get("partition_entropy", 0.0)),
                float(features.get("top1_share", 0.0)),
                float(features.get("top2_share", 0.0)),
                float(features.get("cluster_margin", 0.0)),
                float(features.get("mean_uncertainty", 0.0)) / 4.0,
                float(features.get("std_uncertainty", 0.0)) / 4.0,
                float(features.get("missing_fraction", 0.0)),
                float(features.get("answer_clusters", 0.0)) / valid_experts,
                float(features.get("top_cluster_family_breadth", 0.0)) / valid_experts,
                _normalized_margin(selection.cluster_scores),
            ]
        )
    matrix = np.nan_to_num(np.asarray(rows, dtype=float))
    if matrix.ndim != 2 or matrix.shape[1] != len(DOMAIN_FEATURES):
        raise ValueError("Domain feature matrix has an invalid shape")
    return matrix


def fit_robust_reference(source: np.ndarray) -> RobustReference:
    if source.ndim != 2 or len(source) < 2:
        raise ValueError("Robust domain fitting requires at least two rows")
    median = np.median(source, axis=0)
    q25, q75 = np.quantile(source, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = np.std(source, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return RobustReference(np.asarray(median, dtype=float), np.asarray(scale, dtype=float))


def robust_transform(values: np.ndarray, reference: RobustReference) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != len(reference.median):
        raise ValueError("Domain rows do not match the robust reference")
    return np.clip((values - reference.median) / reference.scale, -8.0, 8.0)


def distribution_distances(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target domain features are not aligned")
    if len(source) < 2 or len(target) < 2:
        raise ValueError("Distribution distance requires at least two source and target rows")
    reference = fit_robust_reference(source)
    source_z = robust_transform(source, reference)
    target_z = robust_transform(target, reference)
    mean_shift = float(np.sqrt(np.mean((target_z.mean(axis=0) - source_z.mean(axis=0)) ** 2)))
    scale_shift = float(np.sqrt(np.mean((target_z.std(axis=0) - source_z.std(axis=0)) ** 2)))
    source_quantiles = np.quantile(source_z, [0.1, 0.5, 0.9], axis=0)
    target_quantiles = np.quantile(target_z, [0.1, 0.5, 0.9], axis=0)
    quantile_shift = float(np.sqrt(np.mean((target_quantiles - source_quantiles) ** 2)))
    combined = float(np.sqrt(mean_shift**2 + 0.25 * scale_shift**2 + 0.5 * quantile_shift**2))
    return {
        "mean": mean_shift,
        "scale": scale_shift,
        "quantile": quantile_shift,
        "combined": combined,
    }


def environment_distribution_distances(
    source: np.ndarray,
    environment_index: np.ndarray,
) -> dict[str, np.ndarray]:
    if environment_index.shape != (len(source),):
        raise ValueError("Environment labels do not align with source domain features")
    values: dict[str, list[float]] = {key: [] for key in ("mean", "scale", "quantile", "combined")}
    for environment in sorted(set(int(value) for value in environment_index)):
        heldout = environment_index == environment
        if heldout.sum() < 2 or (~heldout).sum() < 2:
            continue
        distances = distribution_distances(source[~heldout], source[heldout])
        for metric, value in distances.items():
            values[metric].append(value)
    if not all(values.values()):
        raise ValueError("No source environments support domain-distance calibration")
    return {metric: np.asarray(rows, dtype=float) for metric, rows in values.items()}


def nearest_source_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_environment: np.ndarray | None = None,
) -> np.ndarray:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Nearest-source features are not aligned")
    reference = fit_robust_reference(source)
    source_z = robust_transform(source, reference)
    target_z = robust_transform(target, reference)
    squared = (
        np.sum(target_z**2, axis=1, keepdims=True)
        + np.sum(source_z**2, axis=1)[None, :]
        - 2.0 * target_z @ source_z.T
    )
    squared = np.maximum(squared / max(1, source.shape[1]), 0.0)
    if source_environment is not None:
        if len(target) != len(source) or source_environment.shape != (len(source),):
            raise ValueError("Environment-excluded support distance requires aligned source rows")
        same_environment = source_environment[:, None] == source_environment[None, :]
        squared[same_environment] = np.inf
    distance = np.sqrt(np.min(squared, axis=1))
    if not np.isfinite(distance).all():
        raise ValueError("Some rows have no disjoint source support")
    return distance


def guarded_method_name(base_method: str, policy: DomainPolicy) -> str:
    raw = f"dguard__{policy.policy_id}__{base_method}"
    if len(raw) <= 190:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"dguard__{policy.kind}__{policy.metric}__{digest}"


def apply_domain_policy(
    base: Sequence[Selection],
    reference: Sequence[Selection],
    active: np.ndarray,
    policy: DomainPolicy,
    *,
    base_method: str | None = None,
    domain_diagnostics: Mapping[str, Any],
) -> list[Selection]:
    if len(base) != len(reference) or active.shape != (len(base),):
        raise ValueError("Domain policy inputs are not aligned")
    base_by_id = {selection.question_id: selection for selection in base}
    reference_by_id = {selection.question_id: selection for selection in reference}
    if set(base_by_id) != set(reference_by_id):
        raise ValueError("Base and reference prediction IDs differ")
    ids = sorted(base_by_id)
    resolved_base_method = base_method or str(
        base[0].observable_features.get("method", "base")
    )
    method = guarded_method_name(resolved_base_method, policy)
    result: list[Selection] = []
    for index, question_id in enumerate(ids):
        chosen = base_by_id[question_id] if bool(active[index]) else reference_by_id[question_id]
        features = dict(chosen.observable_features)
        features.update(
            {
                "method": method,
                "domain_guard_policy": policy.policy_id,
                "domain_guard_active": bool(active[index]),
                "domain_guard_fallback": not bool(active[index]),
                "domain_guard_uses_target_labels": False,
                "domain_diagnostics": dict(domain_diagnostics),
            }
        )
        result.append(
            Selection(
                question_id=chosen.question_id,
                selected_cluster_id=chosen.selected_cluster_id,
                selected_expert_id=chosen.selected_expert_id,
                normalized_answer=chosen.normalized_answer,
                cluster_scores=dict(chosen.cluster_scores),
                expert_scores=dict(chosen.expert_scores),
                fallback_reason=chosen.fallback_reason,
                observable_features=features,
                tie_breaking=(
                    "source-calibrated-label-free-domain-guard; "
                    + chosen.tie_breaking
                ),
            )
        )
    return result
