from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from collections import defaultdict
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
from .blind_falsification_jury import FORBIDDEN_AUDIT_KEYS, FalsificationQuestion
from .cross_examined_certificates import (
    CounterexampleCertificate,
    build_certificate_prompt,
    build_certificate_prompt_v2,
    build_sealed_effect_witness_prompt_v3,
    parse_certificate_output,
    parse_certificate_output_v2,
    parse_sealed_effect_witness_output_v3,
    sealed_witness_candidate_fields,
)
from .sealed_counterfactual_parity import (
    build_committed_counterfactual_challenge_prompt_v6,
    build_hardened_counterfactual_challenge_prompt_v5,
    build_sealed_counterfactual_challenge_prompt_v4,
    counterfactual_trace_slot,
    effect_option_sets,
    parse_committed_counterfactual_challenge_output_v6,
    parse_hardened_counterfactual_challenge_output_v5,
    parse_sealed_counterfactual_challenge_output_v4,
    permute_committed_counterfactual_challenge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate label-free C3 option certificates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke-questions", type=int, default=0)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("C3 configuration must be a mapping")
    return value


def _load_questions(run_root: Path) -> list[FalsificationQuestion]:
    path = run_root / "development_observables" / "questions.jsonl"
    rows: list[FalsificationQuestion] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(value)
            if leaked:
                raise PermissionError(f"C3 generation input contains labels: {sorted(leaked)}")
            rows.append(
                FalsificationQuestion(
                    question_id=str(value["question_id"]),
                    dataset=str(value["dataset"]),
                    environment=str(value["environment"]),
                    question=str(value["question"]),
                    options=tuple(str(item) for item in value["options"]),
                    option_labels=tuple(str(item) for item in value["option_labels"]),
                )
            )
    if len({row.question_id for row in rows}) != len(rows):
        raise ValueError("C3 observable questions contain duplicate IDs")
    return sorted(rows, key=lambda row: row.question_id)


def _load_stage0_responses(
    run_root: Path,
    generator: str,
    questions: list[FalsificationQuestion],
) -> dict[str, str]:
    path = run_root / "development_observables" / "base_predictions.jsonl"
    expected = {question.question_id for question in questions}
    responses: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
            if leaked:
                raise PermissionError(
                    f"C3 private Stage-0 input contains labels: {sorted(leaked)}"
                )
            if str(row["expert_id"]) != generator:
                continue
            question_id = str(row["question_id"])
            if question_id not in expected:
                continue
            if question_id in responses:
                raise ValueError("Duplicate private Stage-0 response for C3 generator")
            responses[question_id] = str(row.get("response", ""))
    if set(responses) != expected:
        raise RuntimeError(f"C3 lacks private Stage-0 responses for {generator}")
    return responses


def _stratified_smoke_questions(
    questions: list[FalsificationQuestion], count: int
) -> list[FalsificationQuestion]:
    by_dataset: dict[str, list[FalsificationQuestion]] = defaultdict(list)
    for question in questions:
        by_dataset[question.dataset].append(question)
    selected: list[FalsificationQuestion] = []
    position = 0
    while len(selected) < count:
        added = False
        for dataset in sorted(by_dataset):
            rows = by_dataset[dataset]
            if position < len(rows):
                selected.append(rows[position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    if len(selected) != count:
        raise RuntimeError("C3 smoke selection exceeds the available questions")
    return selected


def _stratified_parity_smoke_questions(
    questions: list[FalsificationQuestion],
    count: int,
    seed: int,
    generator: str,
) -> list[FalsificationQuestion]:
    buckets: dict[tuple[str, int], list[FalsificationQuestion]] = defaultdict(list)
    for question in questions:
        slot = counterfactual_trace_slot(seed, question.question_id, generator)
        buckets[(question.dataset, slot)].append(question)
    selected: list[FalsificationQuestion] = []
    position = 0
    while len(selected) < count:
        added = False
        for key in sorted(buckets):
            rows = buckets[key]
            if position < len(rows):
                selected.append(rows[position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    if len(selected) != count:
        raise RuntimeError("C3 parity smoke selection exceeds the available strata")
    return selected


def _model_args(config: dict[str, Any]) -> SimpleNamespace:
    generation = config["certificate_generation"]
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
    generation = config["certificate_generation"]
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    guided_decoding = None
    if bool(generation.get("guided_regex", False)):
        from vllm.sampling_params import GuidedDecodingParams

        prompt_version = str(generation["prompt_version"])
        if prompt_version == "sealed_effect_witness_v3":
            option_set = r"(?:(?:[A-Z](?:, ?[A-Z])*)|NONE)"
            regex = (
                r"INVARIANT: [^\n]+\n"
                r"DERIVATION: [^\n]+\n"
                r"BOUNDARY: [^\n]+\n"
                rf"ELIMINATED_OPTIONS: {option_set}\n"
                rf"SUPPORTED_OPTIONS: {option_set}\n"
                r"CONFIDENCE: (?:100|[0-9]{1,2})"
            )
        elif prompt_version == "committed_counterfactual_permutation_v6":
            regex = (
                r"RULE: [^\n]+\n"
                r"TRACE_1: [^\n]+\n"
                r"TRACE_2: [^\n]+\n"
                r"FIRST_DIFFERING_STEP: [^\n]+\n"
                r"(?:"
                r"SEALED_VALID_TRACE: (?:1|2)\n"
                r"SEALED_EFFECT: (?:ELIMINATES|SUPPORTS)\n"
                r"SEALED_OPTION: [A-Z]\n"
                r"CONFIDENCE: (?:100|[0-9]{1,2})"
                r"|"
                r"SEALED_VALID_TRACE: NONE\n"
                r"SEALED_EFFECT: NONE\n"
                r"SEALED_OPTION: NONE\n"
                r"CONFIDENCE: (?:50|[0-4]?[0-9])"
                r")"
            )
        elif prompt_version in {
            "sealed_counterfactual_parity_v4",
            "hardened_sealed_counterfactual_parity_v5",
        }:
            regex = (
                r"RULE: [^\n]+\n"
                r"TRACE_1: [^\n]+\n"
                r"TRACE_2: [^\n]+\n"
                r"FIRST_DIFFERING_STEP: [^\n]+\n"
                r"SEALED_VALID_TRACE: (?:1|2|NONE)\n"
                r"SEALED_EFFECT: (?:ELIMINATES|SUPPORTS|NONE)\n"
                r"SEALED_OPTION: (?:[A-Z]|NONE)\n"
                r"CONFIDENCE: (?:100|[0-9]{1,2})"
            )
        else:
            regex = (
                r"VERDICT: (?:FALSIFIED|INCONCLUSIVE|SURVIVES)\n"
                r"CONFIDENCE: (?:100|[0-9]{1,2})\n"
                r"ALTERNATIVE: (?:[A-Z]|NONE)\n"
                r"PREMISE: [^\n]+\n"
                r"CHECK: [^\n]+\n"
                r"FAILURE: [^\n]+"
            )
        guided_decoding = GuidedDecodingParams(
            regex=regex,
            disable_fallback=True,
        )
    return SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        guided_decoding=guided_decoding,
    )


def _protocol_functions(config: dict[str, Any]) -> tuple[Any, Any, bool, bool]:
    generation = config["certificate_generation"]
    protocol = (
        str(generation["prompt_version"]),
        str(generation["parser_version"]),
    )
    if protocol == (
        "sealed_counterexample_certificate_v1",
        "anchored_certificate_fields_v1",
    ):
        return build_certificate_prompt, parse_certificate_output, False, False
    if protocol == (
        "two_sided_sealed_certificate_v2",
        "anchored_certificate_fields_v2",
    ):
        return build_certificate_prompt_v2, parse_certificate_output_v2, False, False
    if protocol == (
        "sealed_effect_witness_v3",
        "sealed_effect_set_fields_v3",
    ):
        return (
            build_sealed_effect_witness_prompt_v3,
            parse_sealed_effect_witness_output_v3,
            True,
            False,
        )
    if protocol == (
        "sealed_counterfactual_parity_v4",
        "sealed_counterfactual_challenge_fields_v4",
    ):
        return (
            build_sealed_counterfactual_challenge_prompt_v4,
            parse_sealed_counterfactual_challenge_output_v4,
            True,
            True,
        )
    if protocol == (
        "hardened_sealed_counterfactual_parity_v5",
        "hardened_counterfactual_challenge_fields_v5",
    ):
        return (
            build_hardened_counterfactual_challenge_prompt_v5,
            parse_hardened_counterfactual_challenge_output_v5,
            True,
            True,
        )
    if protocol == (
        "committed_counterfactual_permutation_v6",
        "committed_counterfactual_challenge_fields_v6",
    ):
        return (
            build_committed_counterfactual_challenge_prompt_v6,
            parse_committed_counterfactual_challenge_output_v6,
            True,
            True,
        )
    raise ValueError(f"Unknown C3 certificate protocol: {protocol}")


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"C3 certificate worker must see physical GPU {physical_gpu}; got {visible!r}"
        )


def _generate(
    config: dict[str, Any],
    generator: str,
    questions: list[FalsificationQuestion],
    private_responses: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    prompt_builder, output_parser, set_valued, counterfactual_pair = (
        _protocol_functions(config)
    )
    post_commit_permutation = (
        str(config["certificate_generation"]["prompt_version"])
        == "committed_counterfactual_permutation_v6"
    )
    if counterfactual_pair and private_responses is None:
        raise RuntimeError("Counterfactual challenge generation needs private Stage-0 traces")
    llm = load_llm(_model_args(config), generator)
    try:
        generation = config["certificate_generation"]
        max_input_tokens = (
            int(generation["max_model_len"])
            - int(generation["max_new_tokens"])
            - 8
        )
        tasks: list[
            tuple[
                FalsificationQuestion,
                str | None,
                str,
                bool,
                int | None,
                int | None,
            ]
        ] = []
        for question in questions:
            candidates: tuple[str | None, ...] = (
                (None,) if set_valued else tuple(question.option_labels)
            )
            for candidate in candidates:
                expected_valid_trace = None
                if counterfactual_pair:
                    expected_valid_trace = counterfactual_trace_slot(
                        int(config["certificate_generation"]["seed"]),
                        question.question_id,
                        generator,
                    )
                    if private_responses is None:
                        raise AssertionError("Private responses disappeared")
                    raw_prompt = (
                        prompt_builder(
                            question,
                            private_responses[question.question_id],
                        )
                        if post_commit_permutation
                        else prompt_builder(
                            question,
                            private_responses[question.question_id],
                            expected_valid_trace,
                        )
                    )
                elif set_valued:
                    raw_prompt = prompt_builder(question)
                else:
                    raw_prompt = prompt_builder(question, candidate)
                prompt = apply_chat_template(llm, raw_prompt)
                prompt, truncated, token_count = truncate_prompt_if_needed(
                    llm, prompt, max_input_tokens
                )
                tasks.append(
                    (
                        question,
                        candidate,
                        prompt,
                        truncated,
                        token_count,
                        expected_valid_trace,
                    )
                )
        sampling = _sampling_params(config)
        batch_size = int(generation["batch_size"])
        rows: list[dict[str, Any]] = []
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            started = time.perf_counter()
            generated = llm.generate([task[2] for task in batch], sampling)
            latency = (time.perf_counter() - started) / max(1, len(batch))
            for task, output in zip(batch, generated, strict=True):
                (
                    question,
                    candidate,
                    prompt,
                    truncated,
                    token_count,
                    expected_valid_trace,
                ) = task
                raw_output = str(output.outputs[0].text)
                if counterfactual_pair:
                    if expected_valid_trace is None:
                        raise AssertionError("Counterfactual trace slot disappeared")
                    if post_commit_permutation:
                        committed_challenge = output_parser(
                            raw_output, question.option_labels
                        )
                        author_valid_trace = committed_challenge.valid_trace
                        parsed_challenge, permutation_applied = (
                            permute_committed_counterfactual_challenge(
                                committed_challenge, expected_valid_trace
                            )
                        )
                    else:
                        parsed_challenge = output_parser(
                            raw_output,
                            question.option_labels,
                            expected_valid_trace,
                        )
                        author_valid_trace = parsed_challenge.valid_trace
                        permutation_applied = False
                    confidence = parsed_challenge.confidence
                    premise = parsed_challenge.rule
                    check = parsed_challenge.trace_1
                    failure = parsed_challenge.first_differing_step
                    eliminated, supported = effect_option_sets(
                        parsed_challenge.effect, parsed_challenge.option
                    )
                    parse_error = parsed_challenge.parse_error
                    challenge_rule = parsed_challenge.rule
                    trace_1 = parsed_challenge.trace_1
                    trace_2 = parsed_challenge.trace_2
                    first_differing_step = parsed_challenge.first_differing_step
                    sealed_valid_trace = parsed_challenge.valid_trace
                    sealed_effect = parsed_challenge.effect
                    witness_id = f"{question.question_id}::{generator}"
                    output_candidates = question.option_labels
                elif set_valued:
                    (
                        confidence,
                        premise,
                        check,
                        failure,
                        eliminated,
                        supported,
                        parse_error,
                    ) = output_parser(raw_output, question.option_labels)
                    witness_id = f"{question.question_id}::{generator}"
                    output_candidates = question.option_labels
                    challenge_rule = None
                    trace_1 = None
                    trace_2 = None
                    first_differing_step = None
                    sealed_valid_trace = None
                    sealed_effect = None
                    author_valid_trace = None
                    permutation_applied = False
                else:
                    if candidate is None:
                        raise AssertionError("Candidate-wise C3 task lacks a candidate")
                    (
                        verdict,
                        confidence,
                        alternative,
                        premise,
                        check,
                        failure,
                        parse_error,
                    ) = output_parser(raw_output, question.option_labels)
                    eliminated = ()
                    supported = ()
                    witness_id = None
                    output_candidates = (candidate,)
                    challenge_rule = None
                    trace_1 = None
                    trace_2 = None
                    first_differing_step = None
                    sealed_valid_trace = None
                    sealed_effect = None
                    author_valid_trace = None
                    permutation_applied = False
                for output_candidate in output_candidates:
                    if set_valued:
                        verdict, alternative = sealed_witness_candidate_fields(
                            output_candidate,
                            question.option_labels,
                            eliminated,
                            supported,
                        )
                    certificate = CounterexampleCertificate(
                        question_id=question.question_id,
                        generator_id=generator,
                        candidate=output_candidate,
                        verdict=verdict,
                        confidence=confidence,
                        alternative=alternative,
                        premise=premise,
                        check=check,
                        failure=failure,
                        parse_error=parse_error,
                        witness_id=witness_id,
                        claimed_eliminated_options=tuple(eliminated),
                        claimed_supported_options=tuple(supported),
                        claim_was_sealed=set_valued,
                        counterfactual_pair=counterfactual_pair,
                        challenge_rule=challenge_rule,
                        trace_1=trace_1,
                        trace_2=trace_2,
                        first_differing_step=first_differing_step,
                        sealed_valid_trace=sealed_valid_trace,
                        sealed_effect=sealed_effect,
                    )
                    row = {
                        "certificate_id": certificate.certificate_id,
                        "witness_id": witness_id,
                        "question_id": question.question_id,
                        "dataset": question.dataset,
                        "environment": question.environment,
                        "generator_id": generator,
                        "candidate": output_candidate,
                        "verdict": verdict,
                        "confidence": confidence,
                        "alternative": alternative,
                        "premise": premise,
                        "check": check,
                        "failure": failure,
                        "claimed_eliminated_options": list(eliminated),
                        "claimed_supported_options": list(supported),
                        "claim_was_sealed": set_valued,
                        "counterfactual_pair": counterfactual_pair,
                        "challenge_rule": challenge_rule,
                        "trace_1": trace_1,
                        "trace_2": trace_2,
                        "first_differing_step": first_differing_step,
                        "sealed_valid_trace": sealed_valid_trace,
                        "sealed_effect": sealed_effect,
                        "author_valid_trace": author_valid_trace,
                        "post_commit_permutation_applied": permutation_applied,
                        "required_valid_trace": expected_valid_trace,
                        "parse_error": parse_error,
                        "raw_output": raw_output,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "prompt_was_truncated": bool(truncated),
                        "prompt_token_count": token_count,
                        "model_latency_seconds": latency,
                    }
                    leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                    if leaked:
                        raise AssertionError(
                            f"C3 certificate emitted labels: {sorted(leaked)}"
                        )
                    rows.append(row)
        return rows
    finally:
        del llm
        cleanup_vllm()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    generators = {str(value) for value in config["certificate_models"]}
    if args.generator not in generators:
        raise ValueError(f"Unregistered C3 certificate generator: {args.generator}")
    _validate_gpu(args.physical_gpu)
    run_root = args.run_root or Path(str(config["output_root"]))
    questions = _load_questions(run_root)
    if args.smoke_questions:
        if not 1 <= args.smoke_questions <= 8:
            raise ValueError("C3 smoke tests must use between one and eight questions")
        if (
            str(config["certificate_generation"]["prompt_version"])
            in {
                "hardened_sealed_counterfactual_parity_v5",
                "committed_counterfactual_permutation_v6",
            }
        ):
            questions = _stratified_parity_smoke_questions(
                questions,
                args.smoke_questions,
                int(config["certificate_generation"]["seed"]),
                args.generator,
            )
        else:
            questions = _stratified_smoke_questions(
                questions, args.smoke_questions
            )
        output_dir = (
            run_root
            / "smoke"
            / "certificates"
            / f"{args.generator}_n{args.smoke_questions}_gpu{args.physical_gpu}"
        )
        status = "bounded_label_free_c3_certificate_smoke"
    else:
        output_dir = run_root / "certificates" / args.generator
        status = "completed_label_free_c3_certificates"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = run_root / "certificate_attempts" / f"{args.generator}.{os.getpid()}"
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    _, _, _, counterfactual_pair = _protocol_functions(config)
    private_responses = (
        _load_stage0_responses(run_root, args.generator, questions)
        if counterfactual_pair
        else None
    )
    rows = _generate(
        config, args.generator, questions, private_responses=private_responses
    )
    expected = sum(len(question.option_labels) for question in questions)
    if len(rows) != expected:
        raise RuntimeError("C3 certificate generation did not cover every question/option")
    output_path = partial / "certificates.jsonl"
    write_jsonl(output_path, rows)
    question_path = run_root / "development_observables" / "questions.jsonl"
    prompt_builder, output_parser, set_valued, counterfactual_pair = (
        _protocol_functions(config)
    )
    post_commit_permutation = (
        str(config["certificate_generation"]["prompt_version"])
        == "committed_counterfactual_permutation_v6"
    )
    witness_rows = {
        str(row["witness_id"]): row
        for row in rows
        if row.get("witness_id") is not None
    }
    base_path = run_root / "development_observables" / "base_predictions.jsonl"
    witness_values = list(witness_rows.values())
    all_option_effect_witnesses = sum(
        len(
            set(row.get("claimed_eliminated_options", ())).union(
                row.get("claimed_supported_options", ())
            )
        )
        == len(
            next(
                question.option_labels
                for question in questions
                if question.question_id == row["question_id"]
            )
        )
        for row in witness_values
        if row["parse_error"] is None
    )
    manifest_inputs = [args.config, question_path]
    if counterfactual_pair:
        manifest_inputs.append(base_path)
    write_json(
        partial / "certificate_manifest.json",
        {
            "status": status,
            "generator": args.generator,
            "physical_gpu": args.physical_gpu,
            "questions": len(questions),
            "certificates": len(rows),
            "parsed_certificates": sum(row["parse_error"] is None for row in rows),
            "truncated_prompts": sum(bool(row["prompt_was_truncated"]) for row in rows),
            "model_calls": len(witness_rows) if set_valued else len(rows),
            "witnesses": len(witness_rows) if set_valued else None,
            "parsed_witnesses": (
                sum(row["parse_error"] is None for row in witness_rows.values())
                if set_valued
                else None
            ),
            "truncated_model_calls": (
                sum(bool(row["prompt_was_truncated"]) for row in witness_rows.values())
                if set_valued
                else sum(bool(row["prompt_was_truncated"]) for row in rows)
            ),
            "claims_are_sealed_from_checkers": set_valued,
            "counterfactual_pairs": counterfactual_pair,
            "private_stage0_responses_read": counterfactual_pair,
            "post_commit_permutation": (
                post_commit_permutation if counterfactual_pair else None
            ),
            "permuted_witnesses": (
                sum(
                    bool(row.get("post_commit_permutation_applied"))
                    for row in witness_values
                )
                if post_commit_permutation
                else None
            ),
            "author_valid_trace_counts": (
                {
                    str(slot): sum(
                        row.get("author_valid_trace") == slot
                        for row in witness_values
                    )
                    for slot in (1, 2)
                }
                if post_commit_permutation
                else None
            ),
            "base_prediction_sha256": (
                sha256_file(base_path) if counterfactual_pair else None
            ),
            "abstaining_witnesses": (
                sum(
                    row["parse_error"] is None
                    and row.get("sealed_valid_trace") is None
                    for row in witness_values
                )
                if counterfactual_pair
                else None
            ),
            "nonabstaining_witnesses": (
                sum(
                    row["parse_error"] is None
                    and row.get("sealed_valid_trace") in (1, 2)
                    for row in witness_values
                )
                if counterfactual_pair
                else None
            ),
            "all_option_effect_witnesses": all_option_effect_witnesses,
            "all_option_effect_rate": (
                all_option_effect_witnesses / max(1, len(witness_values))
                if set_valued
                else None
            ),
            "required_valid_trace_counts": (
                {
                    str(slot): sum(
                        row.get("required_valid_trace") == slot
                        for row in witness_values
                    )
                    for slot in (1, 2)
                }
                if counterfactual_pair
                else None
            ),
            "certificate_sha256": sha256_file(output_path),
            "question_sha256": sha256_file(question_path),
            "prompt_version": str(config["certificate_generation"]["prompt_version"]),
            "parser_version": str(config["certificate_generation"]["parser_version"]),
            "prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(prompt_builder).encode("utf-8")
            ).hexdigest(),
            "parser_sha256": hashlib.sha256(
                inspect.getsource(output_parser).encode("utf-8")
            ).hexdigest(),
            "labels_read": False,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(config["certificate_generation"]["seed"]),
                manifest_inputs,
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed C3 certificates: {output_dir}")


if __name__ == "__main__":
    main()
