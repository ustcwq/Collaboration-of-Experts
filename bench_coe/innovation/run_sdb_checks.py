from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from bench_coe.run_official_model_benchmarks import (
    apply_chat_template,
    cleanup_vllm,
    import_vllm_objects,
    load_llm,
    truncate_prompt_if_needed,
)

from .artifacts import environment_manifest, sha256_file, write_json, write_jsonl
from .blind_falsification_jury import FORBIDDEN_AUDIT_KEYS
from .sealed_diagnostic_bijection import (
    ParsedProbeCheck,
    PresentedDiagnosticProbe,
    build_blind_probe_check_prompt,
    parse_blind_probe_check_output,
    reveal_probe_candidate,
)


@dataclass(frozen=True)
class BlindProbeView:
    probe_id: str
    question_id: str
    dataset: str
    environment: str
    author_id: str
    probe: str
    left_text: str
    right_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve sealed diagnostic probes without candidate context"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checker", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--probe-path", type=Path, action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
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
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def _resolve_probe_paths(
    config: dict[str, Any],
    run_root: Path,
    checker: str,
    explicit_paths: list[Path],
    smoke: bool,
) -> list[Path]:
    if smoke != bool(explicit_paths):
        raise ValueError("Explicit SDB probe paths are required exactly for smoke checks")
    if explicit_paths:
        paths = list(explicit_paths)
    else:
        paths = [
            run_root / "probes" / str(author) / "probes.jsonl"
            for author in config["author_models"]
            if str(author) != checker
        ]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate SDB probe input path")
    return sorted(paths)


def _authenticate_probe_path(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_read") is not False:
        raise PermissionError("SDB probe manifest crossed the label boundary")
    if manifest.get("probe_sha256") != sha256_file(path):
        raise PermissionError("SDB probe artifact changed after generation")
    for key in (
        "mapping_was_sealed_from_checkers",
        "original_task_was_sealed_from_checkers",
        "post_commit_permutation",
    ):
        if manifest.get(key) is not True:
            raise PermissionError(f"SDB probe manifest lacks boundary: {key}")
    return manifest


def _load_blind_probe_views(
    paths: list[Path], checker: str
) -> tuple[list[BlindProbeView], dict[str, str]]:
    views: list[BlindProbeView] = []
    input_hashes: dict[str, str] = {}
    seen: set[str] = set()
    for path in paths:
        manifest = _authenticate_probe_path(path)
        input_hashes[str(path)] = str(manifest["probe_sha256"])
        for row in _read_jsonl(path):
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(f"SDB probe contains labels: {sorted(leaked)}")
            if str(row["author_id"]) == checker:
                raise ValueError("An SDB author may not solve its own probe")
            if row.get("parse_error") is not None or bool(row.get("abstained")):
                continue
            probe_id = str(row["probe_id"])
            if probe_id in seen:
                raise ValueError("Duplicate SDB probe ID")
            seen.add(probe_id)
            if row.get("mapping_was_sealed_from_checkers") is not True:
                raise PermissionError("SDB row exposes an unsealed mapping protocol")
            if row.get("original_task_was_sealed_from_checkers") is not True:
                raise PermissionError("SDB row exposes an unsealed original-task protocol")
            visible_values = (
                row.get("probe"),
                row.get("presented_left_text"),
                row.get("presented_right_text"),
            )
            if any(not isinstance(value, str) or not value for value in visible_values):
                raise ValueError("Parsed SDB probe lacks a visible checker field")
            views.append(
                BlindProbeView(
                    probe_id=probe_id,
                    question_id=str(row["question_id"]),
                    dataset=str(row["dataset"]),
                    environment=str(row["environment"]),
                    author_id=str(row["author_id"]),
                    probe=str(row["probe"]),
                    left_text=str(row["presented_left_text"]),
                    right_text=str(row["presented_right_text"]),
                )
            )
    return sorted(views, key=lambda row: row.probe_id), input_hashes


def _model_args(config: dict[str, Any]) -> SimpleNamespace:
    generation = config["check_generation"]
    return SimpleNamespace(
        models_dir=Path(str(config["models_dir"])),
        backend=str(generation["backend"]),
        max_model_len=int(generation["max_model_len"]),
        attn_implementation="eager",
        gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
        trust_remote_code=bool(generation["trust_remote_code"]),
        dtype=str(generation["dtype"]),
    )


def _sampling_params(config: dict[str, Any]) -> Any:
    generation = config["check_generation"]
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    guided_decoding = None
    if bool(generation.get("guided_regex", False)):
        from vllm.sampling_params import GuidedDecodingParams

        decided = (
            r"OUTCOME: (?:LEFT|RIGHT)\n"
            r"DERIVATION: [^\n]+\n"
            r"CONFIDENCE: (?:100|[0-9]{1,2})"
        )
        uncertain = (
            r"OUTCOME: UNCERTAIN\n"
            r"DERIVATION: NONE\n"
            r"CONFIDENCE: (?:50|[0-4]?[0-9])"
        )
        guided_decoding = GuidedDecodingParams(
            regex=rf"(?:{decided}|{uncertain})",
            disable_fallback=True,
        )
    return SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        guided_decoding=guided_decoding,
    )


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"SDB check worker must see physical GPU {physical_gpu}; got {visible!r}"
        )


def _generate_raw_checks(
    config: dict[str, Any],
    checker: str,
    views: list[BlindProbeView],
) -> list[dict[str, Any]]:
    """Generate from prompt-visible views; this function cannot access sealed mappings."""
    llm = load_llm(_model_args(config), checker)
    try:
        generation = config["check_generation"]
        max_input_tokens = (
            int(generation["max_model_len"])
            - int(generation["max_new_tokens"])
            - 8
        )
        tasks: list[tuple[BlindProbeView, str, str, bool, int]] = []
        for view in views:
            raw_prompt = build_blind_probe_check_prompt(
                view.probe, view.left_text, view.right_text
            )
            prompt = apply_chat_template(llm, raw_prompt)
            prompt, truncated, token_count = truncate_prompt_if_needed(
                llm, prompt, max_input_tokens
            )
            tasks.append((view, raw_prompt, prompt, truncated, token_count))

        sampling = _sampling_params(config)
        batch_size = int(generation["batch_size"])
        rows: list[dict[str, Any]] = []
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            started = time.perf_counter()
            generated = llm.generate([task[2] for task in batch], sampling)
            latency = (time.perf_counter() - started) / max(1, len(batch))
            for task, output in zip(batch, generated, strict=True):
                view, raw_prompt, prompt, truncated, token_count = task
                raw_output = str(output.outputs[0].text)
                parsed = parse_blind_probe_check_output(raw_output)
                row = {
                    "check_id": f"{view.probe_id}::{checker}",
                    "probe_id": view.probe_id,
                    "question_id": view.question_id,
                    "dataset": view.dataset,
                    "environment": view.environment,
                    "author_id": view.author_id,
                    "checker_id": checker,
                    "outcome_side": parsed.outcome_side,
                    "derivation": parsed.derivation,
                    "confidence": parsed.confidence,
                    "parse_error": parsed.parse_error,
                    "uncertain": parsed.uncertain,
                    "original_task_was_hidden": True,
                    "candidate_pair_was_hidden": True,
                    "outcome_mapping_was_hidden": True,
                    "raw_output": raw_output,
                    "raw_prompt_sha256": hashlib.sha256(
                        raw_prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_was_truncated": bool(truncated),
                    "prompt_token_count": token_count,
                    "model_latency_seconds": latency,
                }
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise AssertionError(f"SDB raw check emitted labels: {sorted(leaked)}")
                rows.append(row)
        return rows
    finally:
        del llm
        cleanup_vllm()


def _sealed_presentations(paths: list[Path]) -> dict[str, PresentedDiagnosticProbe]:
    result: dict[str, PresentedDiagnosticProbe] = {}
    for path in paths:
        _authenticate_probe_path(path)
        for row in _read_jsonl(path):
            if row.get("parse_error") is not None or bool(row.get("abstained")):
                continue
            probe_id = str(row["probe_id"])
            if probe_id in result:
                raise ValueError("Duplicate sealed SDB probe ID")
            result[probe_id] = PresentedDiagnosticProbe(
                probe=str(row["probe"]),
                left_text=str(row["presented_left_text"]),
                right_text=str(row["presented_right_text"]),
                left_candidate=str(row["sealed_left_candidate"]),
                right_candidate=str(row["sealed_right_candidate"]),
                left_authored_outcome=int(row["presented_left_authored_outcome"]),
                post_commit_permutation_applied=bool(
                    row["post_commit_permutation_applied"]
                ),
            )
    return result


def _parsed_check_from_raw_row(row: dict[str, Any]) -> ParsedProbeCheck:
    parsed = parse_blind_probe_check_output(str(row["raw_output"]))
    expected = (
        row.get("outcome_side"),
        row.get("derivation"),
        int(row.get("confidence", 0)),
        row.get("parse_error"),
    )
    actual = (
        parsed.outcome_side,
        parsed.derivation,
        parsed.confidence,
        parsed.parse_error,
    )
    if actual != expected:
        raise PermissionError("SDB raw check fields do not replay")
    return parsed


def _reveal_checks_after_raw_hash(
    raw_rows: list[dict[str, Any]],
    presentations: dict[str, PresentedDiagnosticProbe],
    raw_checks_sha256: str,
) -> list[dict[str, Any]]:
    if len(raw_checks_sha256) != 64:
        raise ValueError("SDB raw-check hash must be finalized before reveal")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        probe_id = str(raw["probe_id"])
        presentation = presentations.get(probe_id)
        if presentation is None:
            raise RuntimeError("SDB reveal lacks a sealed presentation")
        parsed = _parsed_check_from_raw_row(raw)
        selected, rejected = reveal_probe_candidate(parsed, presentation)
        rows.append(
            {
                **raw,
                "selected_candidate": selected,
                "rejected_candidate": rejected,
                "mapping_revealed_after_raw_outputs_frozen": True,
                "frozen_raw_checks_sha256": raw_checks_sha256,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    checkers = tuple(str(value) for value in config["checker_models"])
    if args.checker not in checkers:
        raise ValueError(f"Unregistered SDB checker: {args.checker}")
    _validate_gpu(args.physical_gpu)
    run_root = args.run_root or Path(str(config["output_root"]))
    probe_paths = _resolve_probe_paths(
        config, run_root, args.checker, args.probe_path, args.smoke
    )
    views, probe_hashes = _load_blind_probe_views(probe_paths, args.checker)
    if not views:
        raise RuntimeError("SDB checker received no parsed non-abstaining probes")
    if args.smoke:
        output_dir = (
            run_root
            / "smoke"
            / "checks"
            / f"{args.checker}_n{len(views)}_gpu{args.physical_gpu}"
        )
        status = "bounded_label_free_sdb_check_smoke"
    else:
        output_dir = run_root / "checks" / args.checker
        status = "completed_label_free_sdb_checks"
    if output_dir.exists():
        raise FileExistsError(output_dir)

    partial = run_root / "check_attempts" / f"{args.checker}.{os.getpid()}"
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    raw_rows = _generate_raw_checks(config, args.checker, views)
    if len(raw_rows) != len(views):
        raise RuntimeError("SDB checker did not cover every blind probe view")
    raw_path = partial / "raw_checks.jsonl"
    write_jsonl(raw_path, raw_rows)
    raw_hash = sha256_file(raw_path)

    presentations = _sealed_presentations(probe_paths)
    rows = _reveal_checks_after_raw_hash(raw_rows, presentations, raw_hash)
    check_path = partial / "checks.jsonl"
    write_jsonl(check_path, rows)
    parsed_rows = [row for row in rows if row["parse_error"] is None]
    decided_rows = [row for row in parsed_rows if not row["uncertain"]]
    side_counts = {
        side: sum(row["outcome_side"] == side for row in decided_rows)
        for side in ("LEFT", "RIGHT")
    }
    generation = config["check_generation"]
    input_manifest_paths = [path.parent / "probe_manifest.json" for path in probe_paths]
    write_json(
        partial / "check_manifest.json",
        {
            "status": status,
            "checker": args.checker,
            "physical_gpu": args.physical_gpu,
            "input_probes": len(views),
            "model_calls": len(raw_rows),
            "parsed_checks": len(parsed_rows),
            "decided_checks": len(decided_rows),
            "uncertain_checks": len(parsed_rows) - len(decided_rows),
            "truncated_model_calls": sum(
                bool(row["prompt_was_truncated"]) for row in raw_rows
            ),
            "presented_side_counts": side_counts,
            "original_task_was_hidden": True,
            "candidate_pair_was_hidden": True,
            "outcome_mapping_was_hidden": True,
            "mapping_revealed_after_raw_outputs_frozen": True,
            "raw_checks_sha256": raw_hash,
            "check_sha256": sha256_file(check_path),
            "input_probe_hashes": probe_hashes,
            "prompt_version": str(generation["prompt_version"]),
            "parser_version": str(generation["parser_version"]),
            "prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(build_blind_probe_check_prompt).encode("utf-8")
            ).hexdigest(),
            "parser_sha256": hashlib.sha256(
                inspect.getsource(parse_blind_probe_check_output).encode("utf-8")
            ).hexdigest(),
            "reveal_sha256": hashlib.sha256(
                inspect.getsource(reveal_probe_candidate).encode("utf-8")
            ).hexdigest(),
            "labels_read": False,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(generation["seed"]),
                [args.config, *probe_paths, *input_manifest_paths],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed SDB checks: {output_dir}")


if __name__ == "__main__":
    main()
