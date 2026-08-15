from __future__ import annotations

import hashlib
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "benchcoe_unified_v1"
VALID_ROLES = {"source_calibration", "source_validation", "target_locked_test", "secondary_test"}
VALID_MODALITIES = {"text", "vision_language"}
VALID_TASK_TYPES = {"multiple_choice", "exact_match", "open_ended", "code", "mixed"}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def content_digest(question: str, context: str, choices: list[dict[str, str]], images: list[dict[str, Any]]) -> str:
    payload = {
        "question": normalize_text(question).casefold(),
        "context": normalize_text(context).casefold(),
        "choices": [{"label": item["label"], "text": normalize_text(item["text"]).casefold()} for item in choices],
        "images": [item.get("sha256") for item in images],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_sample_id(dataset: str, split: str, native_id: Any, content_hash: str) -> str:
    suffix = normalize_text(native_id) or content_hash[:20]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", suffix).strip("-") or content_hash[:20]
    return f"{dataset}:{split}:{safe}"


def canonicalize_choices(raw: Any) -> list[dict[str, str]]:
    raw = to_builtin(raw)
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                raise ValueError("String choices must encode a list or mapping")
        return canonicalize_choices(parsed)
    if isinstance(raw, dict):
        if isinstance(raw.get("label"), list) and isinstance(raw.get("text"), list):
            return [{"label": str(label), "text": str(text)} for label, text in zip(raw["label"], raw["text"])]
        return [{"label": str(label), "text": str(text)} for label, text in raw.items()]
    if isinstance(raw, list):
        rows = []
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                label = item.get("label", item.get("key", chr(65 + index)))
                text = item.get("text", item.get("value", item.get("content", "")))
            else:
                label, text = chr(65 + index), item
            rows.append({"label": str(label), "text": str(text)})
        return rows
    raise ValueError(f"Unsupported choices value: {type(raw).__name__}")


def canonicalize_answer(raw: Any, choices: list[dict[str, str]]) -> str:
    raw = to_builtin(raw)
    if raw is None:
        return ""
    if isinstance(raw, list):
        return ",".join(canonicalize_answer(item, choices) for item in raw)
    text = normalize_text(raw)
    labels = [item["label"] for item in choices]
    if text in labels:
        return text
    if text.isdigit() and choices:
        index = int(text)
        if 0 <= index < len(choices):
            return choices[index]["label"]
        if 1 <= index <= len(choices):
            return choices[index - 1]["label"]
    for item in choices:
        if normalize_text(item["text"]).casefold() == text.casefold():
            return item["label"]
    match = re.fullmatch(r"[\[(]?\s*([A-Za-z])\s*[\])]?[.]?", text)
    if match and match.group(1).upper() in {label.upper() for label in labels}:
        target = match.group(1).upper()
        return next(label for label in labels if label.upper() == target)
    return text


@dataclass
class UnifiedSample:
    sample_id: str
    dataset: str
    dataset_revision: str
    split: str
    role: str
    modality: str
    task_type: str
    question: str
    context: str = ""
    choices: list[dict[str, str]] = field(default_factory=list)
    answer_canonical: str = ""
    answer_raw: Any = None
    images: list[dict[str, Any]] = field(default_factory=list)
    category: str = ""
    language: str = ""
    native_metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    license: str = ""
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version {self.schema_version}")
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid role {self.role}")
        if self.modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality {self.modality}")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Invalid task_type {self.task_type}")
        if not normalize_text(self.question):
            raise ValueError("Question is required")
        labels = [item["label"] for item in self.choices]
        if len(labels) != len(set(labels)):
            raise ValueError("Choice labels must be unique")
        if self.task_type == "multiple_choice" and self.answer_canonical and self.answer_canonical not in labels:
            raise ValueError(f"Answer {self.answer_canonical!r} is not a choice label")
        for expected, image in enumerate(self.images):
            if image.get("index") != expected:
                raise ValueError("Image order/index is not contiguous")
            if Path(str(image.get("relative_path", ""))).is_absolute():
                raise ValueError("Image paths must be relative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
