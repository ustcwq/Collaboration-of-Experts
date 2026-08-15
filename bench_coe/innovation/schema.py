from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


LabelKey = tuple[str, str]
_SOURCE_LABEL_CAPABILITY = object()


@dataclass(frozen=True)
class CanonicalPredictionRecord:
    dataset: str
    split: str
    question_id: str
    raw_question_id: str
    subject: str
    modality: str
    expert_id: str
    expert_family: str
    raw_answer: str | None
    raw_output: str
    normalized_answer: str | None
    per_query_cluster_id: int | None
    uncertainty: float
    valid_output: bool
    missing_reason: str | None
    source_global_accuracy: float | None = None
    source_fingerprint: tuple[float, ...] = ()
    inference_cost: float | None = None
    observable_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid_output and self.normalized_answer is None:
            raise ValueError("A valid output must have a normalized answer")
        if not self.valid_output and self.per_query_cluster_id is not None:
            raise ValueError("A missing/invalid output cannot belong to an answer cluster")
        forbidden = {"answer", "gold", "target", "correct", "correctness", "is_correct"}
        leaked = forbidden.intersection(self.observable_metadata)
        if leaked:
            raise ValueError(f"Label-like metadata is forbidden in observable records: {sorted(leaked)}")


@dataclass(frozen=True)
class AnswerCluster:
    question_id: str
    cluster_id: int
    normalized_answer: str
    expert_ids: tuple[str, ...]
    family_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExpertPool:
    expert_ids: tuple[str, ...]
    family_by_expert: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.expert_ids) != len(set(self.expert_ids)):
            raise ValueError("Duplicate expert IDs")
        missing = set(self.expert_ids).difference(self.family_by_expert)
        if missing:
            raise ValueError(f"Missing expert-family assignments: {sorted(missing)}")


@dataclass(frozen=True)
class ObservableQueryBatch:
    dataset: str
    split: str
    modality: str
    pool: ExpertPool
    records: tuple[CanonicalPredictionRecord, ...]

    def __post_init__(self) -> None:
        seen: set[LabelKey] = set()
        for record in self.records:
            if (record.dataset, record.split, record.modality) != (self.dataset, self.split, self.modality):
                raise ValueError("Record metadata does not match its batch")
            key = (record.question_id, record.expert_id)
            if key in seen:
                raise ValueError(f"Duplicate question/expert record: {key}")
            seen.add(key)
        expected = set(self.pool.expert_ids)
        for question_id in self.question_ids:
            actual = {record.expert_id for record in self.for_question(question_id)}
            if actual != expected:
                raise ValueError(f"Question {question_id} has an inconsistent expert pool")

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.question_id for record in self.records}))

    def for_question(self, question_id: str) -> tuple[CanonicalPredictionRecord, ...]:
        return tuple(sorted((r for r in self.records if r.question_id == question_id), key=lambda r: r.expert_id))

    def clusters(self, question_id: str) -> tuple[AnswerCluster, ...]:
        by_cluster: dict[int, list[CanonicalPredictionRecord]] = {}
        for record in self.for_question(question_id):
            if record.valid_output and record.per_query_cluster_id is not None:
                by_cluster.setdefault(record.per_query_cluster_id, []).append(record)
        result: list[AnswerCluster] = []
        for cluster_id, records in sorted(by_cluster.items()):
            result.append(
                AnswerCluster(
                    question_id=question_id,
                    cluster_id=cluster_id,
                    normalized_answer=records[0].normalized_answer or "",
                    expert_ids=tuple(sorted(record.expert_id for record in records)),
                    family_ids=tuple(sorted({record.expert_family for record in records})),
                )
            )
        return tuple(result)

    def subset(self, question_ids: Iterable[str]) -> "ObservableQueryBatch":
        keep = set(question_ids)
        return ObservableQueryBatch(
            dataset=self.dataset,
            split=self.split,
            modality=self.modality,
            pool=self.pool,
            records=tuple(record for record in self.records if record.question_id in keep),
        )


@dataclass(frozen=True)
class SourceTrainingLabels:
    dataset: str
    split: str
    correctness: Mapping[LabelKey, bool]
    environment_by_question: Mapping[str, str]
    role: str = "source"
    _capability: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.role != "source" or self._capability is not _SOURCE_LABEL_CAPABILITY:
            raise ValueError("SourceTrainingLabels must be created by a source-role cache adapter")

    @classmethod
    def _from_source_adapter(
        cls,
        dataset: str,
        split: str,
        correctness: Mapping[LabelKey, bool],
        environment_by_question: Mapping[str, str],
    ) -> "SourceTrainingLabels":
        return cls(dataset, split, correctness, environment_by_question, _capability=_SOURCE_LABEL_CAPABILITY)

    def get(self, question_id: str, expert_id: str) -> bool | None:
        return self.correctness.get((question_id, expert_id))

    def subset(self, question_ids: Iterable[str]) -> "SourceTrainingLabels":
        keep = set(question_ids)
        return SourceTrainingLabels._from_source_adapter(
            self.dataset,
            self.split,
            {key: value for key, value in self.correctness.items() if key[0] in keep},
            {key: value for key, value in self.environment_by_question.items() if key in keep},
        )


@dataclass(frozen=True)
class EvaluationLabels:
    dataset: str
    split: str
    correctness: Mapping[LabelKey, bool]

    def get(self, question_id: str, expert_id: str) -> bool | None:
        return self.correctness.get((question_id, expert_id))


@dataclass(frozen=True)
class Selection:
    question_id: str
    selected_cluster_id: int | None
    selected_expert_id: str | None
    normalized_answer: str | None
    cluster_scores: Mapping[str, float]
    expert_scores: Mapping[str, float]
    fallback_reason: str | None
    observable_features: Mapping[str, Any]
    tie_breaking: str = "highest_score_then_lexicographic_id"


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    split: str
    modality: str
    source_or_target: str
    development_or_locked_test: str
    cache_path: str
    sample_count: int
    expert_ids: tuple[str, ...]
