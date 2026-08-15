from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from .locked_protocol import (
    CHOICE_LABELS,
    REQUIRED_MUSR_COLUMNS,
    canonical_question_id,
    parse_choices,
)


def load_musr_evaluation_answers(raw_files: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Evaluator-only target label reader, isolated from every prediction CLI."""

    answers: dict[str, str] = {}
    for spec in raw_files:
        task = str(spec["task"])
        path = Path(str(spec["path"]))
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = set(REQUIRED_MUSR_COLUMNS).difference(fields)
            if missing:
                raise ValueError(f"MuSR file is missing columns {sorted(missing)}: {path}")
            for index, raw in enumerate(reader):
                choices = parse_choices(raw["choices"])
                answer_index = int(str(raw["answer_index"]).strip())
                if not 0 <= answer_index < len(choices):
                    raise ValueError(f"Out-of-range MuSR answer index at {task}:{index}")
                answer_choice = str(raw["answer_choice"] or "").strip()
                if answer_choice and answer_choice != choices[answer_index]:
                    raise ValueError(f"MuSR answer fields disagree at {task}:{index}")
                raw_id = f"{task}:{index:04d}"
                answers[canonical_question_id(raw_id)] = CHOICE_LABELS[answer_index].lower()
    return answers
