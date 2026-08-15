from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import Selection


@dataclass(frozen=True)
class ReliabilityStats:
    accuracy: np.ndarray
    beta_lcb: np.ndarray
    rescue_rate: np.ndarray
    harm_rate: np.ndarray
    worst_environment_delta: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            self.accuracy.shape,
            self.beta_lcb.shape,
            self.rescue_rate.shape,
            self.harm_rate.shape,
            self.worst_environment_delta.shape,
        }
        if len(shapes) != 1 or len(self.accuracy.shape) != 1:
            raise ValueError("Reliability statistics must be aligned one-dimensional arrays")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.accuracy,
                self.beta_lcb,
                self.rescue_rate,
                self.harm_rate,
                self.worst_environment_delta,
            )
        ):
            raise ValueError("Reliability statistics contain non-finite values")


@dataclass(frozen=True)
class VoteScheme:
    pool: str
    reference: str
    weighting: str
    aggregation: str
    top_k: int
    family_balanced: bool
    temperature: float = 0.02
    risk_penalty: float = 1.0

    @property
    def scheme_id(self) -> str:
        balance = "family" if self.family_balanced else "raw"
        return (
            f"{self.pool}__{self.reference}__{self.weighting}"
            f"_t{self.temperature:g}_r{self.risk_penalty:g}__{self.aggregation}"
            f"__k{self.top_k}__{balance}"
        ).replace(".", "p")


@dataclass(frozen=True)
class VoteRecipe:
    scheme: VoteScheme
    min_share: float
    min_margin: float
    min_families: int

    @property
    def method(self) -> str:
        raw = (
            f"cmeta__{self.scheme.scheme_id}__share{self.min_share:g}"
            f"__margin{self.min_margin:g}__families{self.min_families}"
        ).replace(".", "p")
        if len(raw) <= 180:
            return raw
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"cmeta__{self.scheme.pool}__{self.scheme.weighting}__{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scheme": asdict(self.scheme),
            "min_share": self.min_share,
            "min_margin": self.min_margin,
            "min_families": self.min_families,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VoteRecipe":
        return cls(
            scheme=VoteScheme(**dict(value["scheme"])),
            min_share=float(value["min_share"]),
            min_margin=float(value["min_margin"]),
            min_families=int(value["min_families"]),
        )


@dataclass(frozen=True)
class PredictionTable:
    question_ids: tuple[str, ...]
    methods: tuple[str, ...]
    selections: Mapping[str, tuple[Selection, ...]]

    def __post_init__(self) -> None:
        if not self.question_ids or not self.methods:
            raise ValueError("Prediction tables must contain questions and methods")
        if set(self.selections) != set(self.methods):
            raise ValueError("Prediction table methods and selection keys differ")
        expected = set(self.question_ids)
        for method in self.methods:
            values = self.selections[method]
            if len(values) != len(self.question_ids):
                raise ValueError(f"Prediction count differs for {method}")
            if {value.question_id for value in values} != expected:
                raise ValueError(f"Prediction question IDs differ for {method}")

    @classmethod
    def from_selections(
        cls,
        selections: Mapping[str, Sequence[Selection]],
        methods: Sequence[str],
        *,
        limit: int | None = None,
    ) -> "PredictionTable":
        method_tuple = tuple(str(value) for value in methods)
        if len(method_tuple) != len(set(method_tuple)):
            raise ValueError("Prediction methods must be unique")
        missing = set(method_tuple).difference(selections)
        if missing:
            raise ValueError(f"Missing base predictions: {sorted(missing)}")
        first = sorted(selections[method_tuple[0]], key=lambda row: row.question_id)
        question_ids = tuple(row.question_id for row in first)
        if limit is not None:
            question_ids = question_ids[:limit]
        keep = set(question_ids)
        aligned: dict[str, tuple[Selection, ...]] = {}
        for method in method_tuple:
            values = tuple(
                row
                for row in sorted(selections[method], key=lambda item: item.question_id)
                if row.question_id in keep
            )
            aligned[method] = values
        return cls(question_ids, method_tuple, aligned)

    def selection(self, method_index: int, question_index: int) -> Selection:
        return self.selections[self.methods[method_index]][question_index]

    def subset_indices(self, indices: Sequence[int]) -> "PredictionTable":
        selected = tuple(int(value) for value in indices)
        return PredictionTable(
            question_ids=tuple(self.question_ids[index] for index in selected),
            methods=self.methods,
            selections={
                method: tuple(self.selections[method][index] for index in selected)
                for method in self.methods
            },
        )

    def cluster_matrix(self) -> np.ndarray:
        result = np.full((len(self.methods), len(self.question_ids)), -1, dtype=np.int32)
        for method_index, method in enumerate(self.methods):
            for question_index, selection in enumerate(self.selections[method]):
                if selection.selected_cluster_id is not None:
                    result[method_index, question_index] = int(selection.selected_cluster_id)
        return result


@dataclass(frozen=True)
class VoteDiagnostics:
    reference_method_index: int
    winning_cluster: np.ndarray
    chosen_method_index: np.ndarray
    winning_share: np.ndarray
    winning_margin: np.ndarray
    supporting_families: np.ndarray
    cluster_votes: tuple[Mapping[str, float], ...]

    def __post_init__(self) -> None:
        rows = len(self.winning_cluster)
        for value in (
            self.chosen_method_index,
            self.winning_share,
            self.winning_margin,
            self.supporting_families,
        ):
            if len(value) != rows:
                raise ValueError("Vote diagnostics are not aligned")
        if len(self.cluster_votes) != rows:
            raise ValueError("Cluster-vote diagnostics are not aligned")


def correctness_matrix_from_selections(
    table: PredictionTable,
    correctness: Mapping[tuple[str, str], bool],
) -> np.ndarray:
    result = np.zeros((len(table.methods), len(table.question_ids)), dtype=bool)
    for method_index, method in enumerate(table.methods):
        for question_index, selection in enumerate(table.selections[method]):
            expert = selection.selected_expert_id
            result[method_index, question_index] = bool(
                expert is not None and correctness.get((selection.question_id, expert), False)
            )
    return result


def reliability_statistics(
    correctness_by_seed: Sequence[np.ndarray],
    environment_index: np.ndarray,
    training_mask: np.ndarray,
    reference_index: int,
) -> ReliabilityStats:
    if not correctness_by_seed:
        raise ValueError("At least one source seed is required")
    method_count, question_count = correctness_by_seed[0].shape
    if environment_index.shape != (question_count,) or training_mask.shape != (question_count,):
        raise ValueError("Source environment and training masks are not aligned")
    if not training_mask.any():
        raise ValueError("Reliability fitting received an empty source partition")
    for values in correctness_by_seed:
        if values.shape != (method_count, question_count):
            raise ValueError("Source correctness matrices are not aligned")

    stacked = np.concatenate([values[:, training_mask] for values in correctness_by_seed], axis=1)
    reference = stacked[reference_index]
    accuracy = stacked.mean(axis=1)
    rescue = np.mean(stacked & ~reference[None, :], axis=1)
    harm = np.mean(~stacked & reference[None, :], axis=1)
    successes = stacked.sum(axis=1).astype(float)
    samples = float(stacked.shape[1])
    posterior_mean = (successes + 1.0) / (samples + 2.0)
    posterior_variance = (
        (successes + 1.0) * (samples - successes + 1.0)
        / ((samples + 2.0) ** 2 * (samples + 3.0))
    )
    beta_lcb = posterior_mean - 1.96 * np.sqrt(np.maximum(posterior_variance, 0.0))

    environment_deltas: list[np.ndarray] = []
    for environment in sorted(set(int(value) for value in environment_index[training_mask])):
        mask = training_mask & (environment_index == environment)
        if not mask.any():
            continue
        by_seed = np.concatenate([values[:, mask] for values in correctness_by_seed], axis=1)
        environment_deltas.append(by_seed.mean(axis=1) - by_seed[reference_index].mean())
    worst = (
        np.min(np.stack(environment_deltas, axis=0), axis=0)
        if environment_deltas
        else np.zeros(method_count, dtype=float)
    )
    return ReliabilityStats(
        accuracy=np.asarray(accuracy, dtype=float),
        beta_lcb=np.asarray(beta_lcb, dtype=float),
        rescue_rate=np.asarray(rescue, dtype=float),
        harm_rate=np.asarray(harm, dtype=float),
        worst_environment_delta=np.asarray(worst, dtype=float),
    )


def _weight_score(stats: ReliabilityStats, scheme: VoteScheme) -> np.ndarray:
    if scheme.weighting == "equal":
        return np.zeros_like(stats.accuracy)
    if scheme.weighting == "rank":
        order = np.argsort(-stats.accuracy, kind="mergesort")
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(len(order), dtype=float)
        return -np.log1p(ranks)
    if scheme.weighting == "softmax_accuracy":
        return stats.accuracy / max(scheme.temperature, 1e-9)
    if scheme.weighting == "beta_lcb":
        return stats.beta_lcb / max(scheme.temperature, 1e-9)
    if scheme.weighting == "safe_utility":
        utility = stats.rescue_rate - scheme.risk_penalty * stats.harm_rate
        return utility / max(scheme.temperature, 1e-9)
    if scheme.weighting == "minimax":
        return stats.worst_environment_delta / max(scheme.temperature, 1e-9)
    raise ValueError(f"Unknown reliability weighting: {scheme.weighting}")


def method_weights(
    stats: ReliabilityStats,
    scheme: VoteScheme,
    methods: Sequence[str],
    family_by_method: Mapping[str, str],
    pool_methods: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    method_tuple = tuple(methods)
    if scheme.reference not in method_tuple:
        raise ValueError(f"Reference method is absent: {scheme.reference}")
    requested = set(str(value) for value in pool_methods)
    active = np.asarray([index for index, method in enumerate(method_tuple) if method in requested], dtype=int)
    if len(active) == 0:
        raise ValueError(f"Method pool {scheme.pool} is empty")
    missing_families = set(method_tuple[index] for index in active).difference(family_by_method)
    if missing_families:
        raise ValueError(f"Missing method-family assignments: {sorted(missing_families)}")

    active = np.asarray(
        sorted(active, key=lambda index: (-stats.accuracy[index], method_tuple[index])),
        dtype=int,
    )
    if scheme.top_k > 0:
        active = active[: min(scheme.top_k, len(active))]
    scores = _weight_score(stats, scheme)[active]
    scores = scores - float(np.max(scores))
    weights = np.exp(np.clip(scores, -60.0, 0.0))
    if scheme.family_balanced:
        for family in sorted({family_by_method[method_tuple[index]] for index in active}):
            members = np.asarray(
                [offset for offset, index in enumerate(active) if family_by_method[method_tuple[index]] == family],
                dtype=int,
            )
            total = float(weights[members].sum())
            if total > 0.0:
                weights[members] /= total
    weights /= max(float(weights.sum()), 1e-12)
    return active, weights


def _method_cluster_contributions(selection: Selection, aggregation: str) -> dict[int, float]:
    selected = selection.selected_cluster_id
    if selected is None:
        return {}
    if aggregation == "hard":
        return {int(selected): 1.0}
    if aggregation != "cluster_rank":
        raise ValueError(f"Unknown cluster aggregation: {aggregation}")
    values = {
        int(cluster): float(score)
        for cluster, score in selection.cluster_scores.items()
        if math.isfinite(float(score))
    }
    if not values:
        return {int(selected): 1.0}
    ordered = sorted(values, key=lambda cluster: (-values[cluster], cluster))
    width = len(ordered)
    raw = {cluster: float(width - rank) for rank, cluster in enumerate(ordered)}
    denominator = sum(raw.values())
    return {cluster: value / denominator for cluster, value in raw.items()}


def vote_diagnostics(
    table: PredictionTable,
    scheme: VoteScheme,
    active: np.ndarray,
    weights: np.ndarray,
    family_by_method: Mapping[str, str],
) -> VoteDiagnostics:
    if len(active) != len(weights) or not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("Active methods and normalized vote weights are not aligned")
    reference_index = table.methods.index(scheme.reference)
    rows = len(table.question_ids)
    winner = np.full(rows, -1, dtype=np.int32)
    chosen = np.full(rows, reference_index, dtype=np.int32)
    share = np.zeros(rows, dtype=float)
    margin = np.zeros(rows, dtype=float)
    families = np.zeros(rows, dtype=np.int32)
    vote_rows: list[Mapping[str, float]] = []

    for question_index in range(rows):
        votes: dict[int, float] = {}
        contributions_by_method: dict[int, dict[int, float]] = {}
        for offset, method_index in enumerate(active):
            selection = table.selection(int(method_index), question_index)
            contributions = _method_cluster_contributions(selection, scheme.aggregation)
            contributions_by_method[int(method_index)] = contributions
            for cluster, contribution in contributions.items():
                votes[cluster] = votes.get(cluster, 0.0) + float(weights[offset]) * contribution
        reference_cluster = table.selection(reference_index, question_index).selected_cluster_id
        if not votes:
            vote_rows.append({})
            continue
        ordered = sorted(
            votes,
            key=lambda cluster: (
                -votes[cluster],
                0 if reference_cluster is not None and cluster == int(reference_cluster) else 1,
                cluster,
            ),
        )
        selected_cluster = int(ordered[0])
        winner[question_index] = selected_cluster
        total = max(sum(votes.values()), 1e-12)
        share[question_index] = votes[selected_cluster] / total
        runner_up = votes[ordered[1]] if len(ordered) > 1 else 0.0
        margin[question_index] = (votes[selected_cluster] - runner_up) / total
        supporters = [
            (offset, int(method_index))
            for offset, method_index in enumerate(active)
            if table.selection(int(method_index), question_index).selected_cluster_id == selected_cluster
        ]
        families[question_index] = len(
            {family_by_method[table.methods[method_index]] for _, method_index in supporters}
        )
        if supporters:
            chosen[question_index] = sorted(
                supporters,
                key=lambda item: (-weights[item[0]], table.methods[item[1]]),
            )[0][1]
        else:
            chosen[question_index] = reference_index
            winner[question_index] = (
                int(reference_cluster) if reference_cluster is not None else -1
            )
        vote_rows.append(
            {str(cluster): float(value / total) for cluster, value in sorted(votes.items())}
        )
    return VoteDiagnostics(
        reference_method_index=reference_index,
        winning_cluster=winner,
        chosen_method_index=chosen,
        winning_share=share,
        winning_margin=margin,
        supporting_families=families,
        cluster_votes=tuple(vote_rows),
    )


def recipe_choices(table: PredictionTable, recipe: VoteRecipe, diagnostics: VoteDiagnostics) -> np.ndarray:
    reference_clusters = table.cluster_matrix()[diagnostics.reference_method_index]
    switch = (
        (diagnostics.winning_cluster >= 0)
        & (diagnostics.winning_cluster != reference_clusters)
        & (diagnostics.winning_share + 1e-12 >= recipe.min_share)
        & (diagnostics.winning_margin + 1e-12 >= recipe.min_margin)
        & (diagnostics.supporting_families >= recipe.min_families)
    )
    return np.where(
        switch,
        diagnostics.chosen_method_index,
        diagnostics.reference_method_index,
    ).astype(np.int32)


def materialize_recipe_selections(
    table: PredictionTable,
    recipe: VoteRecipe,
    diagnostics: VoteDiagnostics,
) -> list[Selection]:
    choices = recipe_choices(table, recipe, diagnostics)
    result: list[Selection] = []
    for question_index, method_index in enumerate(choices):
        chosen = table.selection(int(method_index), question_index)
        reference = table.selection(diagnostics.reference_method_index, question_index)
        switched = int(method_index) != diagnostics.reference_method_index
        features = dict(chosen.observable_features)
        features.update(
            {
                "method": recipe.method,
                "meta_selector": "source_oof_reliability_conservative_cluster_consensus",
                "chosen_base_method": table.methods[int(method_index)],
                "reference_method": recipe.scheme.reference,
                "switched_from_reference": switched,
                "winning_vote_share": float(diagnostics.winning_share[question_index]),
                "winning_vote_margin": float(diagnostics.winning_margin[question_index]),
                "supporting_method_families": int(
                    diagnostics.supporting_families[question_index]
                ),
                "vote_weighting": recipe.scheme.weighting,
                "vote_aggregation": recipe.scheme.aggregation,
                "family_balanced": recipe.scheme.family_balanced,
                "top_k_methods": recipe.scheme.top_k,
                "source_only_frozen_gate": {
                    "min_share": recipe.min_share,
                    "min_margin": recipe.min_margin,
                    "min_families": recipe.min_families,
                },
            }
        )
        result.append(
            Selection(
                question_id=chosen.question_id,
                selected_cluster_id=chosen.selected_cluster_id,
                selected_expert_id=chosen.selected_expert_id,
                normalized_answer=chosen.normalized_answer,
                cluster_scores=diagnostics.cluster_votes[question_index],
                expert_scores=dict(chosen.expert_scores),
                fallback_reason=chosen.fallback_reason,
                observable_features=features,
                tie_breaking=(
                    "source_oof_weighted_cluster_vote_then_reference_cluster_then_cluster_id; "
                    "conservative_source_frozen_gate; chosen_base_weight_then_method_id"
                ),
            )
        )
        if not switched and (
            result[-1].selected_cluster_id != reference.selected_cluster_id
            or result[-1].normalized_answer != reference.normalized_answer
        ):
            raise RuntimeError("Conservative fallback differs from its reference prediction")
    return result


def generate_recipes(config: Mapping[str, Any]) -> list[VoteRecipe]:
    recipes: list[VoteRecipe] = []
    references = [str(value) for value in config.get("references", ["fcrg_full"])]
    for pool in sorted(config["method_pools"]):
        for reference in references:
            for raw_weight in config["weightings"]:
                if isinstance(raw_weight, str):
                    weighting = raw_weight
                    temperature = 0.02
                    risk_penalty = 1.0
                else:
                    weighting = str(raw_weight["name"])
                    temperature = float(raw_weight.get("temperature", 0.02))
                    risk_penalty = float(raw_weight.get("risk_penalty", 1.0))
                for aggregation in config["aggregations"]:
                    for top_k in config["top_k"]:
                        for family_balanced in config["family_balanced"]:
                            scheme = VoteScheme(
                                pool=str(pool),
                                reference=reference,
                                weighting=weighting,
                                aggregation=str(aggregation),
                                top_k=int(top_k),
                                family_balanced=bool(family_balanced),
                                temperature=temperature,
                                risk_penalty=risk_penalty,
                            )
                            for min_share in config["min_share"]:
                                for min_margin in config["min_margin"]:
                                    for min_families in config["min_families"]:
                                        recipes.append(
                                            VoteRecipe(
                                                scheme=scheme,
                                                min_share=float(min_share),
                                                min_margin=float(min_margin),
                                                min_families=int(min_families),
                                            )
                                        )
    methods = [recipe.method for recipe in recipes]
    if len(methods) != len(set(methods)):
        raise RuntimeError("Generated conservative meta-selector method IDs are not unique")
    return recipes
