from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from bench_coe.assets.locking import sha256_file, write_json
from bench_coe.assets.validation import inspect_image
from .schema import UnifiedSample, canonicalize_answer, canonicalize_choices, content_digest, stable_sample_id, to_builtin


QUESTION_KEYS = ("question", "prompt", "input", "query", "problem")
CONTEXT_KEYS = ("context", "passage", "article", "hint", "description")
CHOICE_KEYS = ("choices", "options", "candidates")
ANSWER_KEYS = ("answer", "label", "target", "gold", "correct_answer", "standard_answer")
ID_KEYS = ("id", "sample_id", "question_id", "index", "uid")
IMAGE_KEYS = ("images", "image", "image_path", "image_paths", "img_list")
CATEGORY_KEYS = ("category", "subject", "topic", "domain")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_value(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    folded = {str(key).casefold(): value for key, value in row.items()}
    for key in keys:
        if key.casefold() in folded and folded[key.casefold()] is not None:
            return folded[key.casefold()]
    return default


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "samples", "examples", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if isinstance(payload, dict):
            mapped = []
            for native_id, item in payload.items():
                if isinstance(item, dict):
                    mapped.append({"id": native_id, **item})
            payload = mapped
        if not isinstance(payload, list):
            raise ValueError(f"JSON input must contain rows: {path}")
        yield from (item for item in payload if isinstance(item, dict))
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("Parquet preprocessing requires pandas and a parquet engine") from error
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                inner_suffix = Path(name).suffix.lower()
                try:
                    raw = archive.read(name)
                except RuntimeError as error:
                    raise RuntimeError(f"Encrypted archive member requires authorization: {name}") from error
                if inner_suffix == ".csv":
                    text = raw.decode("utf-8-sig")
                    yield from csv.DictReader(text.splitlines())
                elif inner_suffix == ".jsonl":
                    for line in raw.decode("utf-8").splitlines():
                        if line.strip():
                            item = json.loads(line)
                            if isinstance(item, dict):
                                yield item
                elif inner_suffix == ".json":
                    payload = json.loads(raw.decode("utf-8"))
                    if isinstance(payload, list):
                        yield from (item for item in payload if isinstance(item, dict))
        return
    raise ValueError(f"Unsupported raw data file: {path}")


def discover_data_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet", ".pq", ".zip"} and not path.name.startswith(".benchcoe_"))


def _image_values(row: dict[str, Any]) -> list[Any]:
    numbered = []
    for key in sorted(row, key=lambda value: str(value)):
        if str(key).casefold().startswith("image_") and row[key] not in (None, ""):
            numbered.append(row[key])
    if numbered:
        return numbered
    raw = first_value(row, IMAGE_KEYS, [])
    if raw in (None, ""):
        return []
    return raw if isinstance(raw, list) else [raw]


def materialize_images(values: list[Any], raw_root: Path, image_root: Path, dataset: str) -> list[dict[str, Any]]:
    output = []
    destination_root = image_root / dataset
    destination_root.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(values):
        image_bytes = None
        source_value = value.get("path") if isinstance(value, dict) else value
        if isinstance(value, dict) and value.get("bytes") is not None:
            image_bytes = bytes(value["bytes"])
        elif isinstance(value, Image.Image):
            buffer = io.BytesIO()
            value.save(buffer, format=value.format or "PNG")
            image_bytes = buffer.getvalue()
        if image_bytes is not None:
            digest = hashlib.sha256(image_bytes).hexdigest()
            try:
                with Image.open(io.BytesIO(image_bytes)) as embedded:
                    suffix = f".{(embedded.format or 'png').lower()}"
            except Exception as error:
                raise ValueError("Embedded image bytes cannot be decoded") from error
            destination = destination_root / f"{digest}{suffix}"
            if not destination.exists():
                destination.write_bytes(image_bytes)
        else:
            if not isinstance(source_value, str):
                raise ValueError(f"Unsupported image reference {value!r}")
            source = (raw_root / source_value).resolve() if not Path(source_value).is_absolute() else Path(source_value).resolve()
            source.relative_to(raw_root.resolve())
            digest = sha256_file(source)
            suffix = source.suffix.lower() or ".bin"
            destination = destination_root / f"{digest}{suffix}"
            if not destination.exists():
                shutil.copy2(source, destination)
        metadata = inspect_image(destination, image_root)
        metadata["index"] = index
        output.append(metadata)
    return output


def convert_row(
    row: dict[str, Any], dataset: str, revision: str, split: str, role: str,
    modality: str, task_type: str, license_name: str, raw_root: Path, image_root: Path,
) -> UnifiedSample:
    choices = canonicalize_choices(first_value(row, CHOICE_KEYS, []))
    answer_raw = first_value(row, ANSWER_KEYS, None)
    images = materialize_images(_image_values(row), raw_root, image_root, dataset)
    question = str(first_value(row, QUESTION_KEYS, ""))
    context = str(first_value(row, CONTEXT_KEYS, ""))
    digest = content_digest(question, context, choices, images)
    native_id = first_value(row, ID_KEYS, "")
    resolved_task_type = task_type
    if task_type == "mixed":
        resolved_task_type = "multiple_choice" if choices else "open_ended"
    sample = UnifiedSample(
        sample_id=stable_sample_id(dataset, split, native_id, digest),
        dataset=dataset,
        dataset_revision=revision,
        split=split,
        role=role,
        modality="vision_language" if modality in {"vision_language", "mixed"} else "text",
        task_type=resolved_task_type,
        question=question,
        context=context,
        choices=choices,
        answer_canonical=canonicalize_answer(answer_raw, choices),
        answer_raw=to_builtin(answer_raw),
        images=images,
        category=str(first_value(row, CATEGORY_KEYS, "")),
        language=str(row.get("language", row.get("lang", ""))),
        native_metadata={key: to_builtin(value) for key, value in row.items() if key not in set(QUESTION_KEYS + CONTEXT_KEYS + CHOICE_KEYS + ANSWER_KEYS + ID_KEYS + IMAGE_KEYS + CATEGORY_KEYS + ("language", "lang"))},
        content_hash=digest,
        license=license_name,
    )
    sample.validate()
    return sample


def stable_smoke_subset(samples: list[UnifiedSample], limit: int = 8) -> list[UnifiedSample]:
    return sorted(samples, key=lambda item: hashlib.sha256(item.sample_id.encode()).hexdigest())[:limit]


def qa_samples(samples: list[UnifiedSample]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_ids = Counter(item.sample_id for item in samples)
    hashes = Counter(item.content_hash for item in samples)
    invalid_answers = []
    duplicate_choices = 0
    missing_question = 0
    missing_answer = 0
    image_errors = 0
    for item in samples:
        if not item.question.strip():
            missing_question += 1
        if item.answer_raw in (None, ""):
            missing_answer += 1
        labels = [choice["label"] for choice in item.choices]
        texts = [choice["text"].strip().casefold() for choice in item.choices]
        duplicate_choices += int(len(texts) != len(set(texts)))
        if item.task_type == "multiple_choice" and item.answer_canonical and item.answer_canonical not in labels:
            invalid_answers.append(item.sample_id)
        image_errors += sum(1 for image in item.images if image.get("decode_status") != "ok")
    overlaps = []
    for sample_id, count in sample_ids.items():
        if count > 1:
            overlaps.append({"type": "duplicate_sample_id", "sample_id": sample_id, "count": count})
    for content_hash, count in hashes.items():
        if count > 1:
            overlaps.append({"type": "duplicate_content_hash", "content_hash": content_hash, "count": count})
    choices_distribution = Counter(len(item.choices) for item in samples)
    qa = {
        "generated_at": utc_now(),
        "sample_count": len(samples),
        "split_counts": dict(Counter(item.split for item in samples)),
        "role_counts": dict(Counter(item.role for item in samples)),
        "category_counts": dict(Counter(item.category for item in samples)),
        "language_counts": dict(Counter(item.language for item in samples)),
        "choice_count_distribution": {str(key): value for key, value in sorted(choices_distribution.items())},
        "random_guess_baselines": {str(key): 1.0 / key for key in choices_distribution if key > 0},
        "missing_question": missing_question,
        "missing_answer": missing_answer,
        "duplicate_choice_rows": duplicate_choices,
        "invalid_answer_ids": invalid_answers,
        "duplicate_sample_ids": sum(1 for count in sample_ids.values() if count > 1),
        "duplicate_content_hashes": sum(1 for count in hashes.values() if count > 1),
        "open_answer_count": sum(item.task_type in {"open_ended", "code"} for item in samples),
        "image_count": sum(len(item.images) for item in samples),
        "image_errors": image_errors,
        "status": "pass" if not (missing_question or invalid_answers or image_errors or overlaps) else "review_required",
    }
    return qa, overlaps


def write_processed_dataset(
    samples: list[UnifiedSample],
    output_dir: Path,
    raw_files: list[Path],
    smoke: bool = False,
    preselected: bool = False,
    selection: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = stable_smoke_subset(samples) if smoke and not preselected else samples
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "id_map.jsonl").open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps({"sample_id": sample.sample_id, "content_hash": sample.content_hash}, ensure_ascii=False) + "\n")
    qa, overlaps = qa_samples(selected)
    write_json(output_dir / "qa_report.json", qa)
    with (output_dir / "overlap_report.jsonl").open("w", encoding="utf-8") as handle:
        for item in overlaps:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "benchcoe_dataset_manifest_v1",
        "generated_at": utc_now(),
        "dataset": selected[0].dataset if selected else output_dir.name,
        "dataset_revision": selected[0].dataset_revision if selected else "unknown",
        "sample_schema": "benchcoe_unified_v1",
        "sample_count": len(selected),
        "smoke_subset": smoke,
        "selection": selection or ("sha256(sample_id) ascending" if smoke else "all converted rows"),
        "raw_files": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in raw_files],
        "samples_sha256": sha256_file(samples_path),
        "qa_status": qa["status"],
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    (output_dir / "README.generated.md").write_text(
        f"# {manifest['dataset']} processed data\n\n"
        f"- Schema: `benchcoe_unified_v1`\n- Revision: `{manifest['dataset_revision']}`\n"
        f"- Samples: {manifest['sample_count']}\n- Smoke subset: {str(smoke).lower()}\n- QA: `{qa['status']}`\n",
        encoding="utf-8",
    )
    return manifest


def detect_cross_dataset_overlaps(processed_root: Path) -> list[dict[str, Any]]:
    seen_exact: dict[str, tuple[str, str, str]] = {}
    seen_text: dict[str, tuple[str, str, str]] = {}
    overlaps = []
    for samples_path in sorted(processed_root.glob("*/samples.jsonl")):
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            current = (row["dataset"], row["split"], row["role"])
            exact = row["content_hash"]
            normalized = hashlib.sha256((" ".join((row.get("question", "") + " " + row.get("context", "")).casefold().split())).encode()).hexdigest()
            for kind, key, index in (("exact_content_hash", exact, seen_exact), ("normalized_text_hash", normalized, seen_text)):
                previous = index.get(key)
                if previous and previous != current:
                    overlaps.append({"type": kind, "hash": key, "left": previous, "right": current, "role_conflict": previous[2] != current[2]})
                else:
                    index[key] = current
    return overlaps
