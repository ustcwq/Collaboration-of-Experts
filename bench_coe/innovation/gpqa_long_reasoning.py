from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import Selection


_FINAL_PATTERNS = (
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?([A-D])\)?", re.I),
    re.compile(r"\\boxed\s*\{?\(?([A-D])\)?\}?", re.I),
)


def extract_explicit_answer(text: str) -> str | None:
    matches: list[tuple[int, str]] = []
    for pattern in _FINAL_PATTERNS:
        matches.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(text))
    return max(matches, default=(0, ""), key=lambda item: item[0])[1] or None


def normalize_option(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:")


def source_best_model(summary_rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    candidates = [
        (str(row["model"]), float(row["accuracy"]))
        for row in summary_rows
        if row.get("model") is not None and row.get("accuracy") is not None
    ]
    if not candidates:
        raise ValueError("No source model summaries with accuracy")
    model, accuracy = sorted(candidates, key=lambda item: (-item[1], item[0]))[0]
    return model, accuracy


def _query_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    config = row.get("config")
    base_question_id = row.get("base_question_id")
    if config is None or base_question_id is None:
        return None
    normalized_config = str(config).strip()
    normalized_base_question_id = str(base_question_id).strip()
    if not normalized_config or not normalized_base_question_id:
        return None
    return normalized_config, normalized_base_question_id


def semantic_answer_by_query(
    inference_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    result: dict[tuple[str, str], str] = {}
    seen_queries: set[tuple[str, str]] = set()
    valid_rows = 0
    for row in inference_rows:
        query_key = _query_key(row)
        if query_key is None:
            continue
        if query_key in seen_queries:
            raise ValueError(
                "Long-reasoning inference contains duplicate query identity "
                f"{query_key!r}"
            )
        seen_queries.add(query_key)
        prediction = str(row.get("prediction") or "").upper()
        options = list(row.get("options") or [])
        if prediction not in "ABCD" or len(options) != 4:
            continue
        answer = str(options[ord(prediction) - ord("A")])
        if not normalize_option(answer):
            continue
        result[query_key] = answer
        valid_rows += 1
    audit = {
        "inference_rows": len(inference_rows),
        "valid_rows": valid_rows,
        "invalid_rows": len(inference_rows) - valid_rows,
        "unique_query_identities": len(seen_queries),
        "semantic_queries": len(result),
        "duplicate_query_rows": 0,
    }
    return result, audit


def apply_semantic_reasoning_overrides(
    base: Sequence[Selection],
    metadata_by_question: Mapping[str, Mapping[str, Any]],
    semantic_by_query: Mapping[tuple[str, str], str],
    *,
    model_id: str,
    inference_manifest_sha256: str,
) -> tuple[list[Selection], dict[str, int]]:
    result: list[Selection] = []
    counts = {
        "overridden": 0,
        "fallback_missing_query_identity": 0,
        "fallback_missing_prediction": 0,
        "fallback_mapping_failure": 0,
    }
    for row in base:
        metadata = metadata_by_question.get(row.question_id, {})
        query_key = _query_key(metadata)
        if query_key is None:
            counts["fallback_missing_query_identity"] += 1
            result.append(row)
            continue
        config, base_question_id = query_key
        record_id = str(metadata.get("record_id") or "")
        semantic = semantic_by_query.get(query_key)
        if semantic is None:
            counts["fallback_missing_prediction"] += 1
            result.append(row)
            continue
        normalized_semantic = normalize_option(semantic)
        options = list(metadata.get("options") or [])
        matches = [
            index
            for index, option in enumerate(options)
            if normalize_option(option) == normalized_semantic
        ]
        if len(matches) != 1 or matches[0] >= 4:
            counts["fallback_mapping_failure"] += 1
            result.append(row)
            continue
        answer = "abcd"[matches[0]]
        features = dict(row.observable_features)
        features.update(
            {
                "gpqa_long_reasoning_override": True,
                "gpqa_long_reasoning_model": model_id,
                "gpqa_long_reasoning_record_id": record_id,
                "gpqa_long_reasoning_config": config,
                "gpqa_long_reasoning_base_question_id": base_question_id,
                "gpqa_long_reasoning_inference_manifest_sha256": inference_manifest_sha256,
                "gpqa_long_reasoning_uses_target_labels": False,
                "gpqa_semantic_permutation_mapping": True,
                "gpqa_query_local_answer_identity": True,
            }
        )
        result.append(
            replace(
                row,
                selected_cluster_id=-2,
                selected_expert_id=model_id,
                normalized_answer=answer,
                cluster_scores={"long_reasoning": 1.0},
                expert_scores={model_id: 1.0},
                fallback_reason=None,
                observable_features=features,
                tie_breaking="query_local_source_best_long_reasoning_then_v3_fallback",
            )
        )
        counts["overridden"] += 1
    return result, counts


def read_source_summaries(root: Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary_validation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
