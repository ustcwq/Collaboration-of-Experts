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
from .c3_prior_art_controls import (
    CANDIDATE_VISIBLE_PARSER_VERSION,
    CANDIDATE_VISIBLE_PROMPT_VERSION,
    ISOLATED_PRIOR_ART_CONTROL_PROMPTS,
    UNSEALED_PARSER_VERSION,
    UNSEALED_PROMPT_VERSION,
    build_candidate_visible_commit_first_prompt_v8_control,
    build_unsealed_set_aware_prompt_v8_control,
    parse_candidate_visible_proof_output_v8_control,
    parse_unsealed_proof_output_v8_control,
)
from .cross_examined_certificates import (
    CertificateCheck,
    CounterexampleCertificate,
    build_certificate_check_prompt,
    build_sealed_effect_reconstruction_prompt_v3,
    build_target_blind_check_prompt_v2,
    parse_certificate_check_output,
    parse_target_blind_check_output_v2,
    reconstructed_check_status,
)
from .sealed_counterfactual_parity import (
    ISOLATED_TRACE_VIEWS,
    PARITY_ORIENTATIONS,
    build_blind_counterfactual_parity_prompt_v4,
    build_blind_isolated_trace_audit_prompt_v7,
    build_commitment_conditioned_pair_audit_prompt_v8_ablation,
    build_commitment_conditioned_proof_audit_prompt_v8,
    build_hardened_blind_counterfactual_parity_prompt_v5,
    canonical_trace_index,
    combine_isolated_trace_audits,
    effect_option_sets,
    parse_blind_counterfactual_parity_output_v4,
    parse_blind_isolated_trace_audit_output_v7,
    parse_commitment_conditioned_pair_audit_output_v8_ablation,
    parse_commitment_conditioned_proof_audit_output_v8,
    sealed_triple_matches,
    validate_c3_v8_mechanism_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blindly cross-examine C3 certificates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checker", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--certificate-path", type=Path)
    parser.add_argument("--smoke-certificates", type=int, default=0)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("C3 configuration must be a mapping")
    return value


def _resolve_input_run_root(
    config_path: Path, config: dict[str, Any], output_run_root: Path
) -> tuple[Path, dict[str, str] | None]:
    ablation = config.get("mechanism_ablation")
    if ablation is None:
        return output_run_root, None
    if not isinstance(ablation, dict):
        raise TypeError("C3 mechanism_ablation must be a mapping")
    base_config_path = Path(str(ablation["base_config_path"]))
    base_config = _load_config(base_config_path)
    expected_hash = str(ablation["base_config_sha256"])
    actual_hash = sha256_file(base_config_path)
    if actual_hash != expected_hash:
        raise PermissionError("C3 mechanism ablation base config changed")
    name = validate_c3_v8_mechanism_ablation(base_config, config)
    input_run_root = Path(str(base_config["output_root"]))
    if input_run_root == output_run_root:
        raise ValueError("C3 mechanism ablation must use a separate output root")
    expected_output_root = input_run_root / "mechanism_ablations" / name
    if output_run_root.resolve() != expected_output_root.resolve():
        raise PermissionError("C3 mechanism ablation output root differs")
    return input_run_root, {
        "name": name,
        "base_config_path": str(base_config_path),
        "base_config_sha256": actual_hash,
    }


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


def _load_questions(run_root: Path) -> dict[str, FalsificationQuestion]:
    path = run_root / "development_observables" / "questions.jsonl"
    result: dict[str, FalsificationQuestion] = {}
    for value in _read_jsonl(path):
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(value)
        if leaked:
            raise PermissionError(f"C3 check input contains labels: {sorted(leaked)}")
        question = FalsificationQuestion(
            question_id=str(value["question_id"]),
            dataset=str(value["dataset"]),
            environment=str(value["environment"]),
            question=str(value["question"]),
            options=tuple(str(item) for item in value["options"]),
            option_labels=tuple(str(item) for item in value["option_labels"]),
        )
        if question.question_id in result:
            raise ValueError("C3 check questions contain duplicate IDs")
        result[question.question_id] = question
    return result


def _load_stage0_commitments(
    run_root: Path,
    checker: str,
    questions: dict[str, FalsificationQuestion],
) -> dict[str, str | None]:
    path = run_root / "development_observables" / "base_predictions.jsonl"
    commitments: dict[str, str | None] = {}
    for row in _read_jsonl(path):
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(f"C3 commitment input contains labels: {sorted(leaked)}")
        if str(row["expert_id"]) != checker:
            continue
        question_id = str(row["question_id"])
        question = questions.get(question_id)
        if question is None or question_id in commitments:
            raise ValueError("C3 Stage-0 commitments are not one-to-one with questions")
        answer = None if row.get("prediction") is None else str(row["prediction"])
        commitments[question_id] = (
            answer if answer in question.option_labels else None
        )
    if set(commitments) != set(questions):
        raise RuntimeError(f"C3 lacks Stage-0 commitments for checker {checker}")
    return commitments


def _load_private_stage0_responses(
    run_root: Path,
    checker: str,
    questions: dict[str, FalsificationQuestion],
) -> dict[str, str]:
    path = run_root / "development_observables" / "base_predictions.jsonl"
    responses: dict[str, str] = {}
    for row in _read_jsonl(path):
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(
                f"C3 private checker input contains labels: {sorted(leaked)}"
            )
        if str(row["expert_id"]) != checker:
            continue
        question_id = str(row["question_id"])
        if question_id not in questions or question_id in responses:
            raise ValueError("C3 private checker responses are not one-to-one")
        responses[question_id] = str(row.get("response", ""))
    if set(responses) != set(questions):
        raise RuntimeError(f"C3 lacks private Stage-0 responses for checker {checker}")
    return responses


def _certificate_from_row(row: dict[str, Any]) -> CounterexampleCertificate:
    leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
    if leaked:
        raise PermissionError(f"C3 certificate contains labels: {sorted(leaked)}")
    certificate = CounterexampleCertificate(
        question_id=str(row["question_id"]),
        generator_id=str(row["generator_id"]),
        candidate=str(row["candidate"]),
        verdict=str(row["verdict"]),
        confidence=int(row["confidence"]),
        alternative=None if row.get("alternative") is None else str(row["alternative"]),
        premise=None if row.get("premise") is None else str(row["premise"]),
        check=None if row.get("check") is None else str(row["check"]),
        failure=None if row.get("failure") is None else str(row["failure"]),
        parse_error=None if row.get("parse_error") is None else str(row["parse_error"]),
        witness_id=None if row.get("witness_id") is None else str(row["witness_id"]),
        claimed_eliminated_options=tuple(
            str(value) for value in row.get("claimed_eliminated_options", [])
        ),
        claimed_supported_options=tuple(
            str(value) for value in row.get("claimed_supported_options", [])
        ),
        claim_was_sealed=bool(row.get("claim_was_sealed", False)),
        counterfactual_pair=bool(row.get("counterfactual_pair", False)),
        challenge_rule=(
            None if row.get("challenge_rule") is None else str(row["challenge_rule"])
        ),
        trace_1=None if row.get("trace_1") is None else str(row["trace_1"]),
        trace_2=None if row.get("trace_2") is None else str(row["trace_2"]),
        first_differing_step=(
            None
            if row.get("first_differing_step") is None
            else str(row["first_differing_step"])
        ),
        sealed_valid_trace=(
            None
            if row.get("sealed_valid_trace") is None
            else int(row["sealed_valid_trace"])
        ),
        sealed_effect=(
            None if row.get("sealed_effect") is None else str(row["sealed_effect"])
        ),
    )
    if str(row["certificate_id"]) != certificate.certificate_id:
        raise PermissionError("C3 certificate ID is not canonical")
    return certificate


def _load_certificates(
    config: dict[str, Any],
    run_root: Path,
    checker: str,
    explicit_path: Path | None,
) -> tuple[list[CounterexampleCertificate], list[Path]]:
    paths = (
        [explicit_path]
        if explicit_path is not None
        else [
            run_root / "certificates" / str(generator) / "certificates.jsonl"
            for generator in config["certificate_models"]
            if str(generator) != checker
        ]
    )
    result: list[CounterexampleCertificate] = []
    for path in paths:
        if path is None:
            raise AssertionError("C3 certificate path resolution failed")
        manifest_path = path.parent / "certificate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("labels_read") is not False:
            raise PermissionError("C3 source certificate manifest crossed the label boundary")
        if manifest.get("certificate_sha256") != sha256_file(path):
            raise PermissionError("C3 source certificate file changed after generation")
        for row in _read_jsonl(path):
            certificate = _certificate_from_row(row)
            if certificate.generator_id == checker:
                if explicit_path is not None:
                    raise ValueError("C3 checker cannot inspect its own certificates")
                continue
            if certificate.parse_error is None and (
                not certificate.counterfactual_pair
                or certificate.sealed_valid_trace is not None
            ):
                result.append(certificate)
    identities = [row.certificate_id for row in result]
    if len(identities) != len(set(identities)):
        raise ValueError("C3 check input contains duplicate certificates")
    return sorted(result, key=lambda row: row.certificate_id), [Path(path) for path in paths if path is not None]


def _stratified_smoke_certificates(
    certificates: list[CounterexampleCertificate],
    questions: dict[str, FalsificationQuestion],
    count: int,
) -> list[CounterexampleCertificate]:
    by_dataset: dict[str, list[CounterexampleCertificate]] = defaultdict(list)
    for certificate in certificates:
        by_dataset[questions[certificate.question_id].dataset].append(certificate)
    selected: list[CounterexampleCertificate] = []
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
        raise RuntimeError("C3 check smoke selection exceeds parsed certificates")
    return selected


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


def _guided_regex_for_prompt_version(prompt_version: str) -> str:
    option_set = r"(?:(?:[A-Z](?:, ?[A-Z])*)|NONE)"
    if prompt_version == "commitment_conditioned_proof_audit_v8":
        return (
            r"TRACE_STATUS: (?:VALID|INVALID|INCONCLUSIVE)\n"
            r"COUNTERTEST: [^\n]+\n"
            r"COUNTERTEST_RESULT: (?:SURVIVES|BREAKS|UNCERTAIN)\n"
            r"RECOMPUTATION: [^\n]+\n"
            r"COMMITMENT_RELATION: (?:CONSISTENT|CONFLICTS|UNRELATED|UNCERTAIN)\n"
            r"EFFECT: (?:ELIMINATES|SUPPORTS|NONE)\n"
            r"OPTION: (?:[A-Z]|NONE)\n"
            r"CONFIDENCE: (?:100|[0-9]{1,2})\n"
            r"FLAW_CODE: (?:RULE_MISAPPLIED|ARITHMETIC|UNSUPPORTED_PREMISE|"
            r"QUESTION_MISMATCH|OTHER|UNCERTAIN|NONE)\n"
            r"FLAW_DETAIL: [^\n]+"
        )
    if prompt_version in ISOLATED_PRIOR_ART_CONTROL_PROMPTS:
        return _guided_regex_for_prompt_version(
            "commitment_conditioned_proof_audit_v8"
        )
    if prompt_version == "commitment_conditioned_pair_audit_v8_ablation":
        return (
            r"PAIR_STATUS: (?:ONE_VALID|BOTH_INVALID|INCONCLUSIVE)\n"
            r"COUNTERTEST: [^\n]+\n"
            r"COUNTERTEST_RESULT: (?:ONE_SURVIVES_ONE_BREAKS|BOTH_BREAK|UNCERTAIN)\n"
            r"RECOMPUTATION: [^\n]+\n"
            r"COMMITMENT_RELATION: (?:CONSISTENT|CONFLICTS|UNRELATED|UNCERTAIN)\n"
            r"VALID_TRACE: (?:1|2|NONE)\n"
            r"EFFECT: (?:ELIMINATES|SUPPORTS|NONE)\n"
            r"OPTION: (?:[A-Z]|NONE)\n"
            r"CONFIDENCE: (?:100|[0-9]{1,2})\n"
            r"FIRST_FLAW: [^\n]+"
        )
    if prompt_version == "blind_isolated_trace_audit_v7":
        confidence = r"(?:100|[0-9]{1,2})"
        low_confidence = r"(?:[0-9]|[1-4][0-9]|50)"
        flaw_code = (
            r"(?:RULE_MISAPPLIED|ARITHMETIC|UNSUPPORTED_PREMISE|"
            r"QUESTION_MISMATCH|OTHER)"
        )
        return (
            rf"(?:TRACE_STATUS: VALID\nEFFECT: (?:ELIMINATES|SUPPORTS)\n"
            rf"OPTION: [A-Z]\nCONFIDENCE: {confidence}\nFLAW_CODE: NONE\n"
            r"FLAW_DETAIL: NONE|"
            rf"TRACE_STATUS: INVALID\nEFFECT: NONE\nOPTION: NONE\n"
            rf"CONFIDENCE: {confidence}\nFLAW_CODE: {flaw_code}\n"
            r"FLAW_DETAIL: [^\n]+|"
            rf"TRACE_STATUS: INCONCLUSIVE\nEFFECT: NONE\nOPTION: NONE\n"
            rf"CONFIDENCE: {low_confidence}\nFLAW_CODE: UNCERTAIN\n"
            r"FLAW_DETAIL: [^\n]+)"
        )
    if prompt_version in {
        "blind_counterfactual_parity_v4",
        "hardened_blind_counterfactual_parity_v5",
    }:
        return (
            r"PAIR_STATUS: (?:ONE_VALID|BOTH_INVALID|INCONCLUSIVE)\n"
            r"VALID_TRACE: (?:1|2|NONE)\n"
            r"EFFECT: (?:ELIMINATES|SUPPORTS|NONE)\n"
            r"OPTION: (?:[A-Z]|NONE)\n"
            r"CONFIDENCE: (?:100|[0-9]{1,2})\n"
            r"FIRST_FLAW: [^\n]+"
        )
    return (
        r"LOGIC_STATUS: (?:VALID|INVALID|INCONCLUSIVE)\n"
        r"CONFIDENCE: (?:100|[0-9]{1,2})\n"
        rf"ELIMINATED_OPTIONS: {option_set}\n"
        rf"SUPPORTED_OPTIONS: {option_set}\n"
        r"FIRST_INVALID_STEP: [^\n]+"
    )


def _sampling_params(config: dict[str, Any]) -> Any:
    generation = config["check_generation"]
    (SamplingParams,) = import_vllm_objects("SamplingParams")
    guided_decoding = None
    if bool(generation.get("guided_regex", False)):
        from vllm.sampling_params import GuidedDecodingParams

        regex = _guided_regex_for_prompt_version(
            str(generation["prompt_version"])
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


def _protocol_functions(
    config: dict[str, Any]
) -> tuple[Any, Any, bool, bool, bool]:
    generation = config["check_generation"]
    protocol = (
        str(generation["prompt_version"]),
        str(generation["parser_version"]),
    )
    if protocol == (
        "blind_certificate_cross_examination_v1",
        "anchored_certificate_check_fields_v1",
    ):
        return (
            build_certificate_check_prompt,
            parse_certificate_check_output,
            False,
            False,
            False,
        )
    if protocol == (
        "target_blind_effect_reconstruction_v2",
        "target_blind_effect_fields_v2",
    ):
        return (
            build_target_blind_check_prompt_v2,
            parse_target_blind_check_output_v2,
            True,
            False,
            False,
        )
    if protocol == (
        "sealed_effect_reconstruction_v3",
        "target_blind_effect_fields_v2",
    ):
        return (
            build_sealed_effect_reconstruction_prompt_v3,
            parse_target_blind_check_output_v2,
            True,
            True,
            False,
        )
    if protocol == (
        "blind_counterfactual_parity_v4",
        "blind_counterfactual_parity_fields_v4",
    ):
        return (
            build_blind_counterfactual_parity_prompt_v4,
            parse_blind_counterfactual_parity_output_v4,
            True,
            True,
            True,
        )
    if protocol == (
        "hardened_blind_counterfactual_parity_v5",
        "blind_counterfactual_parity_fields_v4",
    ):
        return (
            build_hardened_blind_counterfactual_parity_prompt_v5,
            parse_blind_counterfactual_parity_output_v4,
            True,
            True,
            True,
        )
    if protocol == (
        "blind_isolated_trace_audit_v7",
        "isolated_trace_audit_fields_v7",
    ):
        return (
            build_blind_isolated_trace_audit_prompt_v7,
            parse_blind_isolated_trace_audit_output_v7,
            True,
            True,
            True,
        )
    if protocol == (
        "commitment_conditioned_proof_audit_v8",
        "proof_obligation_audit_fields_v8",
    ):
        return (
            build_commitment_conditioned_proof_audit_prompt_v8,
            parse_commitment_conditioned_proof_audit_output_v8,
            True,
            True,
            True,
        )
    if protocol == (
        "commitment_conditioned_pair_audit_v8_ablation",
        "pair_proof_obligation_audit_fields_v8_ablation",
    ):
        return (
            build_commitment_conditioned_pair_audit_prompt_v8_ablation,
            parse_commitment_conditioned_pair_audit_output_v8_ablation,
            True,
            True,
            True,
        )
    if protocol == (
        CANDIDATE_VISIBLE_PROMPT_VERSION,
        CANDIDATE_VISIBLE_PARSER_VERSION,
    ):
        return (
            build_candidate_visible_commit_first_prompt_v8_control,
            parse_candidate_visible_proof_output_v8_control,
            False,
            True,
            True,
        )
    if protocol == (UNSEALED_PROMPT_VERSION, UNSEALED_PARSER_VERSION):
        return (
            build_unsealed_set_aware_prompt_v8_control,
            parse_unsealed_proof_output_v8_control,
            False,
            True,
            True,
        )
    raise ValueError(f"Unknown C3 check protocol: {protocol}")


def _counterfactual_audit_views(config: dict[str, Any]) -> tuple[str, ...]:
    prompt_version = str(config["check_generation"]["prompt_version"])
    if prompt_version in {
        "blind_isolated_trace_audit_v7",
        "commitment_conditioned_proof_audit_v8",
        *ISOLATED_PRIOR_ART_CONTROL_PROMPTS,
    }:
        return ISOLATED_TRACE_VIEWS
    return PARITY_ORIENTATIONS


def _audit_protocol_name(
    config: dict[str, Any], counterfactual_pair: bool
) -> str:
    prompt_version = str(config["check_generation"]["prompt_version"])
    if prompt_version == "commitment_conditioned_proof_audit_v8":
        return "commitment_conditioned_proof_audit_v8"
    if prompt_version == "blind_isolated_trace_audit_v7":
        return "isolated_trace_pointwise_v7"
    if prompt_version == "commitment_conditioned_pair_audit_v8_ablation":
        return "commitment_conditioned_pair_audit_v8_ablation"
    if prompt_version == CANDIDATE_VISIBLE_PROMPT_VERSION:
        return "candidate_visible_commit_first_v8_control"
    if prompt_version == UNSEALED_PROMPT_VERSION:
        return "unsealed_set_aware_v8_control"
    return "dual_orientation_pairwise" if counterfactual_pair else "single"


def _is_proof_obligation_audit(config: dict[str, Any]) -> bool:
    return str(config["check_generation"]["prompt_version"]) in {
        "commitment_conditioned_proof_audit_v8",
        "commitment_conditioned_pair_audit_v8_ablation",
        *ISOLATED_PRIOR_ART_CONTROL_PROMPTS,
    }


def _sealed_claim_is_hidden(config: dict[str, Any], set_valued: bool) -> bool:
    return set_valued and str(
        config["check_generation"]["prompt_version"]
    ) != UNSEALED_PROMPT_VERSION


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"C3 check worker must see physical GPU {physical_gpu}; got {visible!r}"
        )


def _generate(
    config: dict[str, Any],
    checker: str,
    questions: dict[str, FalsificationQuestion],
    certificates: list[CounterexampleCertificate],
    commitments: dict[str, str | None],
    private_stage0_responses: dict[str, str],
) -> list[dict[str, Any]]:
    (
        prompt_builder,
        output_parser,
        target_was_hidden,
        set_valued,
        counterfactual_pair,
    ) = _protocol_functions(config)
    proof_obligation_audit = _is_proof_obligation_audit(config)
    sealed_claim_was_hidden = _sealed_claim_is_hidden(config, set_valued)
    prompt_version = str(config["check_generation"]["prompt_version"])
    audit_protocol = _audit_protocol_name(config, counterfactual_pair)
    if proof_obligation_audit and set(private_stage0_responses) != set(questions):
        raise RuntimeError("v8 proof audits require every private Stage-0 response")
    llm = load_llm(_model_args(config), checker)
    try:
        generation = config["check_generation"]
        max_input_tokens = (
            int(generation["max_model_len"])
            - int(generation["max_new_tokens"])
            - 8
        )
        grouped: dict[str, list[CounterexampleCertificate]] = defaultdict(list)
        if set_valued:
            for certificate in certificates:
                if not certificate.claim_was_sealed or certificate.witness_id is None:
                    raise PermissionError("v3 check input lacks a sealed witness identity")
                grouped[certificate.witness_id].append(certificate)
        else:
            for certificate in certificates:
                grouped[certificate.certificate_id].append(certificate)
        tasks: list[
            tuple[
                tuple[CounterexampleCertificate, ...],
                str,
                bool,
                int | None,
                str,
                str,
            ]
        ] = []
        for witness_id in sorted(grouped):
            certificate_group = tuple(
                sorted(grouped[witness_id], key=lambda row: row.candidate)
            )
            certificate = certificate_group[0]
            if set_valued:
                signature = (
                    certificate.question_id,
                    certificate.generator_id,
                    certificate.premise,
                    certificate.check,
                    certificate.failure,
                    certificate.confidence,
                    certificate.claimed_eliminated_options,
                    certificate.claimed_supported_options,
                    certificate.counterfactual_pair,
                    certificate.challenge_rule,
                    certificate.trace_1,
                    certificate.trace_2,
                    certificate.first_differing_step,
                    certificate.sealed_valid_trace,
                    certificate.sealed_effect,
                )
                if any(
                    (
                        row.question_id,
                        row.generator_id,
                        row.premise,
                        row.check,
                        row.failure,
                        row.confidence,
                        row.claimed_eliminated_options,
                        row.claimed_supported_options,
                        row.counterfactual_pair,
                        row.challenge_rule,
                        row.trace_1,
                        row.trace_2,
                        row.first_differing_step,
                        row.sealed_valid_trace,
                        row.sealed_effect,
                    )
                    != signature
                    for row in certificate_group
                ):
                    raise PermissionError("Expanded v3 certificates disagree within a witness")
                expected_candidates = set(
                    questions[certificate.question_id].option_labels
                )
                if {row.candidate for row in certificate_group} != expected_candidates:
                    raise PermissionError("v3 witness lacks exact candidate expansion")
            question = questions[certificate.question_id]
            orientations = (
                _counterfactual_audit_views(config)
                if counterfactual_pair
                else ("single",)
            )
            for orientation in orientations:
                if counterfactual_pair:
                    if any(
                        value is None
                        for value in (
                            certificate.challenge_rule,
                            certificate.trace_1,
                            certificate.trace_2,
                            certificate.first_differing_step,
                        )
                    ):
                        raise PermissionError("Parsed v4 challenge lacks visible content")
                    prompt_arguments = (
                        question,
                        str(certificate.challenge_rule),
                        str(certificate.trace_1),
                        str(certificate.trace_2),
                        str(certificate.first_differing_step),
                        orientation,
                    )
                    raw_prompt = (
                        prompt_builder(
                            *prompt_arguments,
                            private_stage0_responses[certificate.question_id],
                            certificate,
                        )
                        if prompt_version in ISOLATED_PRIOR_ART_CONTROL_PROMPTS
                        else prompt_builder(
                            *prompt_arguments,
                            private_stage0_responses[certificate.question_id],
                        )
                        if proof_obligation_audit
                        else prompt_builder(*prompt_arguments)
                    )
                else:
                    raw_prompt = prompt_builder(question, certificate)
                prompt = apply_chat_template(llm, raw_prompt)
                if certificate.generator_id in prompt:
                    raise PermissionError("C3 check prompt leaked the certificate generator")
                prompt, truncated, token_count = truncate_prompt_if_needed(
                    llm, prompt, max_input_tokens
                )
                tasks.append(
                    (
                        certificate_group,
                        prompt,
                        truncated,
                        token_count,
                        orientation,
                        hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
                    )
                )
        sampling = _sampling_params(config)
        batch_size = int(generation["batch_size"])
        rows: list[dict[str, Any]] = []
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            started = time.perf_counter()
            generated = llm.generate([task[1] for task in batch], sampling)
            latency = (time.perf_counter() - started) / max(1, len(batch))
            for task, output in zip(batch, generated, strict=True):
                (
                    certificate_group,
                    prompt,
                    truncated,
                    token_count,
                    orientation,
                    raw_prompt_sha256,
                ) = task
                certificate = certificate_group[0]
                question = questions[certificate.question_id]
                raw_output = str(output.outputs[0].text)
                isolated_trace_audit = orientation in ISOLATED_TRACE_VIEWS
                if isolated_trace_audit:
                    parsed_audit = output_parser(raw_output, question.option_labels)
                    logic_status = parsed_audit.trace_status
                    confidence = parsed_audit.confidence
                    eliminated_options, supported_options = effect_option_sets(
                        parsed_audit.effect, parsed_audit.option
                    )
                    first_flaw = parsed_audit.flaw_detail
                    parse_error = parsed_audit.parse_error
                    presented_valid_trace = None
                    canonical_valid_trace = (
                        1 + ISOLATED_TRACE_VIEWS.index(orientation)
                        if logic_status == "VALID" and parse_error is None
                        else None
                    )
                    reconstructed_effect = parsed_audit.effect
                    pair_status = None
                    flaw_code = parsed_audit.flaw_code
                    countertest = getattr(parsed_audit, "countertest", None)
                    countertest_result = getattr(
                        parsed_audit, "countertest_result", None
                    )
                    recomputation = getattr(parsed_audit, "recomputation", None)
                    commitment_relation = getattr(
                        parsed_audit, "commitment_relation", None
                    )
                    independent = commitments[certificate.question_id]
                elif counterfactual_pair:
                    parsed_audit = output_parser(raw_output, question.option_labels)
                    logic_status = {
                        "ONE_VALID": "VALID",
                        "BOTH_INVALID": "INVALID",
                        "INCONCLUSIVE": "INCONCLUSIVE",
                    }[parsed_audit.pair_status]
                    confidence = parsed_audit.confidence
                    eliminated_options, supported_options = effect_option_sets(
                        parsed_audit.effect, parsed_audit.option
                    )
                    first_flaw = parsed_audit.first_flaw
                    parse_error = parsed_audit.parse_error
                    presented_valid_trace = parsed_audit.presented_valid_trace
                    canonical_valid_trace = canonical_trace_index(
                        presented_valid_trace, orientation
                    )
                    reconstructed_effect = parsed_audit.effect
                    pair_status = parsed_audit.pair_status
                    flaw_code = None
                    countertest = getattr(parsed_audit, "countertest", None)
                    countertest_result = getattr(
                        parsed_audit, "countertest_result", None
                    )
                    recomputation = getattr(parsed_audit, "recomputation", None)
                    commitment_relation = getattr(
                        parsed_audit, "commitment_relation", None
                    )
                    independent = commitments[certificate.question_id]
                elif target_was_hidden:
                    (
                        logic_status,
                        confidence,
                        eliminated_options,
                        supported_options,
                        first_flaw,
                        parse_error,
                    ) = output_parser(raw_output, question.option_labels)
                    independent = commitments[certificate.question_id]
                    presented_valid_trace = None
                    canonical_valid_trace = None
                    reconstructed_effect = None
                    pair_status = None
                    flaw_code = None
                    countertest = None
                    countertest_result = None
                    recomputation = None
                    commitment_relation = None
                else:
                    status, confidence, independent, first_flaw, parse_error = (
                        output_parser(raw_output, question.option_labels)
                    )
                    logic_status = None
                    eliminated_options = ()
                    supported_options = ()
                    presented_valid_trace = None
                    canonical_valid_trace = None
                    reconstructed_effect = None
                    pair_status = None
                    flaw_code = None
                    countertest = None
                    countertest_result = None
                    recomputation = None
                    commitment_relation = None
                for expanded_certificate in certificate_group:
                    if target_was_hidden or counterfactual_pair:
                        status = reconstructed_check_status(
                            expanded_certificate,
                            logic_status,
                            eliminated_options,
                            supported_options,
                        )
                    check = CertificateCheck(
                        certificate_id=expanded_certificate.certificate_id,
                        question_id=expanded_certificate.question_id,
                        generator_id=expanded_certificate.generator_id,
                        checker_id=checker,
                        candidate=expanded_certificate.candidate,
                        status=status,
                        confidence=confidence,
                        independent_answer=independent,
                        first_flaw=first_flaw,
                        parse_error=parse_error,
                        logic_status=logic_status,
                        eliminated_options=tuple(eliminated_options),
                        supported_options=tuple(supported_options),
                        target_was_hidden=target_was_hidden,
                        counterfactual_pair=counterfactual_pair,
                        orientation=orientation,
                        presented_valid_trace=presented_valid_trace,
                        canonical_valid_trace=canonical_valid_trace,
                        reconstructed_effect=reconstructed_effect,
                    )
                    row = {
                        "certificate_id": check.certificate_id,
                        "witness_id": expanded_certificate.witness_id,
                        "question_id": check.question_id,
                        "dataset": question.dataset,
                        "environment": question.environment,
                        "generator_id": check.generator_id,
                        "checker_id": check.checker_id,
                        "candidate": check.candidate,
                        "status": check.status,
                        "confidence": check.confidence,
                        "independent_answer": check.independent_answer,
                        "first_flaw": check.first_flaw,
                        "parse_error": check.parse_error,
                        "logic_status": check.logic_status,
                        "eliminated_options": list(check.eliminated_options),
                        "supported_options": list(check.supported_options),
                        "target_was_hidden": check.target_was_hidden,
                        "sealed_claim_was_hidden": sealed_claim_was_hidden,
                        "counterfactual_pair": counterfactual_pair,
                        "orientation": orientation,
                        "pair_status": pair_status,
                        "audit_protocol": audit_protocol,
                        "trace_under_audit": orientation if isolated_trace_audit else None,
                        "flaw_code": flaw_code,
                        "countertest": countertest,
                        "countertest_result": countertest_result,
                        "recomputation": recomputation,
                        "commitment_relation": commitment_relation,
                        "presented_valid_trace": presented_valid_trace,
                        "canonical_valid_trace": canonical_valid_trace,
                        "reconstructed_effect": reconstructed_effect,
                        "commitment_source": (
                            "stage0_base_prediction"
                            if proof_obligation_audit
                            else "checker_output"
                        ),
                        "raw_output": raw_output,
                        "raw_prompt_sha256": raw_prompt_sha256,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "prompt_was_truncated": bool(truncated),
                        "prompt_token_count": token_count,
                        "model_latency_seconds": latency,
                    }
                    leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                    if leaked:
                        raise AssertionError(f"C3 check emitted labels: {sorted(leaked)}")
                    rows.append(row)
        return rows
    finally:
        del llm
        cleanup_vllm()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    checkers = {str(value) for value in config["checker_models"]}
    if args.checker not in checkers:
        raise ValueError(f"Unregistered C3 checker: {args.checker}")
    if bool(args.certificate_path) != bool(args.smoke_certificates):
        raise ValueError("Explicit C3 certificate paths are allowed only for bounded smoke tests")
    if args.smoke_certificates and not 1 <= args.smoke_certificates <= 64:
        raise ValueError("C3 check smoke tests must contain between 1 and 64 certificates")
    _validate_gpu(args.physical_gpu)
    run_root = args.run_root or Path(str(config["output_root"]))
    input_run_root, ablation_metadata = _resolve_input_run_root(
        args.config, config, run_root
    )
    questions = _load_questions(input_run_root)
    commitments = _load_stage0_commitments(
        input_run_root, args.checker, questions
    )
    certificates, certificate_paths = _load_certificates(
        config, input_run_root, args.checker, args.certificate_path
    )
    _, _, _, set_valued, counterfactual_pair = _protocol_functions(config)
    proof_obligation_audit = _is_proof_obligation_audit(config)
    sealed_claim_was_hidden = _sealed_claim_is_hidden(config, set_valued)
    private_stage0_responses = (
        _load_private_stage0_responses(input_run_root, args.checker, questions)
        if proof_obligation_audit
        else {}
    )
    audit_protocol = _audit_protocol_name(config, counterfactual_pair)
    if args.smoke_certificates:
        if set_valued:
            representatives = list(
                {
                    str(row.witness_id): row
                    for row in certificates
                    if row.witness_id is not None
                }.values()
            )
            selected = _stratified_smoke_certificates(
                representatives, questions, args.smoke_certificates
            )
            selected_witnesses = {str(row.witness_id) for row in selected}
            certificates = [
                row
                for row in certificates
                if str(row.witness_id) in selected_witnesses
            ]
        else:
            certificates = _stratified_smoke_certificates(
                certificates, questions, args.smoke_certificates
            )
        output_dir = (
            run_root
            / "smoke"
            / "checks"
            / f"{args.checker}_n{args.smoke_certificates}_gpu{args.physical_gpu}"
        )
        status = "bounded_label_free_c3_check_smoke"
    else:
        output_dir = run_root / "checks" / args.checker
        status = "completed_label_free_c3_checks"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = run_root / "check_attempts" / f"{args.checker}.{os.getpid()}"
    partial.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows = _generate(
        config,
        args.checker,
        questions,
        certificates,
        commitments,
        private_stage0_responses,
    )
    expected_check_rows = len(certificates) * (2 if counterfactual_pair else 1)
    if len(rows) != expected_check_rows:
        raise RuntimeError("C3 cross-examination did not cover every input certificate")
    output_path = partial / "checks.jsonl"
    write_jsonl(output_path, rows)
    question_path = (
        input_run_root / "development_observables" / "questions.jsonl"
    )
    base_path = (
        input_run_root / "development_observables" / "base_predictions.jsonl"
    )
    (
        prompt_builder,
        output_parser,
        target_was_hidden,
        set_valued,
        counterfactual_pair,
    ) = _protocol_functions(config)
    reconstruction_rows = {
        f"{row.get('witness_id') or row['certificate_id']}::{row.get('orientation', 'single')}": row
        for row in rows
    }
    audit_views = (
        _counterfactual_audit_views(config)
        if counterfactual_pair
        else ("single",)
    )
    isolated_trace_audit = audit_views == ISOLATED_TRACE_VIEWS
    parity_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    certificate_by_witness = {
        str(certificate.witness_id): certificate
        for certificate in certificates
        if certificate.witness_id is not None
    }
    if counterfactual_pair:
        for row in reconstruction_rows.values():
            parity_groups[(str(row["witness_id"]), str(row["generator_id"]))][
                str(row["orientation"])
            ] = row
    position_invariant_pairs = 0
    sealed_triple_audits = 0
    paired_sealed_triple_matches = 0
    complete_isolated_trace_pairs = 0
    one_valid_one_invalid_pairs = 0
    isolated_sealed_triple_matches = 0
    if isolated_trace_audit:
        for by_view in parity_groups.values():
            if set(by_view) != set(ISOLATED_TRACE_VIEWS):
                continue
            trace_1_row = by_view["trace_1"]
            trace_2_row = by_view["trace_2"]
            complete_isolated_trace_pairs += int(
                trace_1_row["parse_error"] is None
                and trace_2_row["parse_error"] is None
            )
            combined = combine_isolated_trace_audits(
                output_parser(
                    str(trace_1_row["raw_output"]),
                    questions[str(trace_1_row["question_id"])].option_labels,
                ),
                output_parser(
                    str(trace_2_row["raw_output"]),
                    questions[str(trace_2_row["question_id"])].option_labels,
                ),
            )
            one_valid_one_invalid_pairs += int(combined.pair_status == "ONE_VALID")
            eliminated, supported = effect_option_sets(
                combined.effect, combined.option
            )
            certificate = certificate_by_witness[str(trace_1_row["witness_id"])]
            isolated_sealed_triple_matches += int(
                combined.parse_error is None
                and combined.pair_status == "ONE_VALID"
                and sealed_triple_matches(
                    certificate.sealed_valid_trace,
                    certificate.sealed_effect,
                    certificate.claimed_eliminated_options,
                    certificate.claimed_supported_options,
                    combined.presented_valid_trace,
                    combined.effect,
                    eliminated,
                    supported,
                )
            )
    else:
        for by_orientation in parity_groups.values():
            if set(by_orientation) != set(PARITY_ORIENTATIONS):
                continue
            canonical = by_orientation["canonical"]
            mirrored = by_orientation["mirrored"]
            same_underlying = (
                canonical["parse_error"] is None
                and mirrored["parse_error"] is None
                and canonical["logic_status"] == mirrored["logic_status"]
                and canonical["canonical_valid_trace"]
                == mirrored["canonical_valid_trace"]
                and canonical["reconstructed_effect"]
                == mirrored["reconstructed_effect"]
                and canonical["eliminated_options"] == mirrored["eliminated_options"]
                and canonical["supported_options"] == mirrored["supported_options"]
            )
            valid_flip = (
                canonical["logic_status"] != "VALID"
                or (
                    canonical["presented_valid_trace"] is not None
                    and mirrored["presented_valid_trace"] is not None
                    and canonical["presented_valid_trace"]
                    == 3 - mirrored["presented_valid_trace"]
                )
            )
            position_invariant_pairs += int(same_underlying and valid_flip)
            orientation_matches: list[bool] = []
            for row in (canonical, mirrored):
                certificate = certificate_by_witness[str(row["witness_id"])]
                matches = row["parse_error"] is None and sealed_triple_matches(
                    certificate.sealed_valid_trace,
                    certificate.sealed_effect,
                    certificate.claimed_eliminated_options,
                    certificate.claimed_supported_options,
                    row["canonical_valid_trace"],
                    row["reconstructed_effect"],
                    row["eliminated_options"],
                    row["supported_options"],
                )
                sealed_triple_audits += int(matches)
                orientation_matches.append(matches)
            paired_sealed_triple_matches += int(all(orientation_matches))
    write_json(
        partial / "check_manifest.json",
        {
            "status": status,
            "checker": args.checker,
            "physical_gpu": args.physical_gpu,
            "input_certificates": len(certificates),
            "certificates_checked": len(rows),
            "parsed_checks": sum(row["parse_error"] is None for row in rows),
            "truncated_prompts": sum(bool(row["prompt_was_truncated"]) for row in rows),
            "model_calls": len(reconstruction_rows),
            "reconstructions": len(reconstruction_rows) if set_valued else None,
            "parsed_reconstructions": (
                sum(
                    row["parse_error"] is None
                    for row in reconstruction_rows.values()
                )
                if set_valued
                else None
            ),
            "truncated_model_calls": sum(
                bool(row["prompt_was_truncated"])
                for row in reconstruction_rows.values()
            ),
            "sealed_claim_was_hidden": sealed_claim_was_hidden,
            "counterfactual_pairs": counterfactual_pair,
            "parity_orientations": (
                list(PARITY_ORIENTATIONS)
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "isolated_trace_views": (
                list(ISOLATED_TRACE_VIEWS) if isolated_trace_audit else None
            ),
            "audit_protocol": audit_protocol,
            "position_invariant_pairs": (
                position_invariant_pairs
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "position_invariant_pair_rate": (
                position_invariant_pairs / max(1, len(parity_groups))
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "parity_pairs": (
                len(parity_groups)
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "sealed_triple_audits": (
                sealed_triple_audits
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "sealed_triple_audit_rate": (
                sealed_triple_audits / max(1, 2 * len(parity_groups))
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "paired_sealed_triple_matches": (
                paired_sealed_triple_matches
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "paired_sealed_triple_match_rate": (
                paired_sealed_triple_matches / max(1, len(parity_groups))
                if counterfactual_pair and not isolated_trace_audit
                else None
            ),
            "isolated_trace_pairs": (
                len(parity_groups) if isolated_trace_audit else None
            ),
            "complete_isolated_trace_pairs": (
                complete_isolated_trace_pairs if isolated_trace_audit else None
            ),
            "complete_isolated_trace_pair_rate": (
                complete_isolated_trace_pairs / max(1, len(parity_groups))
                if isolated_trace_audit
                else None
            ),
            "one_valid_one_invalid_pairs": (
                one_valid_one_invalid_pairs if isolated_trace_audit else None
            ),
            "one_valid_one_invalid_pair_rate": (
                one_valid_one_invalid_pairs / max(1, len(parity_groups))
                if isolated_trace_audit
                else None
            ),
            "isolated_sealed_triple_matches": (
                isolated_sealed_triple_matches if isolated_trace_audit else None
            ),
            "isolated_sealed_triple_match_rate": (
                isolated_sealed_triple_matches / max(1, len(parity_groups))
                if isolated_trace_audit
                else None
            ),
            "check_sha256": sha256_file(output_path),
            "question_sha256": sha256_file(question_path),
            "base_prediction_sha256": sha256_file(base_path),
            "input_certificate_hashes": {
                str(path): sha256_file(path) for path in certificate_paths
            },
            "prompt_version": str(config["check_generation"]["prompt_version"]),
            "parser_version": str(config["check_generation"]["parser_version"]),
            "target_was_hidden": target_was_hidden,
            "commitments_from_stage0": proof_obligation_audit,
            "private_stage0_responses_read": proof_obligation_audit,
            "proof_obligations_required": proof_obligation_audit,
            "mechanism_ablation": ablation_metadata,
            "input_run_root": str(input_run_root),
            "prompt_builder_sha256": hashlib.sha256(
                inspect.getsource(prompt_builder).encode("utf-8")
            ).hexdigest(),
            "parser_sha256": hashlib.sha256(
                inspect.getsource(output_parser).encode("utf-8")
            ).hexdigest(),
            "pair_combiner_sha256": (
                hashlib.sha256(
                    inspect.getsource(combine_isolated_trace_audits).encode("utf-8")
                ).hexdigest()
                if isolated_trace_audit
                else None
            ),
            "labels_read": False,
            "started_unix": started,
            "finished_unix": time.time(),
            "environment": environment_manifest(
                sys.argv,
                int(config["check_generation"]["seed"]),
                [
                    args.config,
                    *(
                        [Path(ablation_metadata["base_config_path"])]
                        if ablation_metadata is not None
                        else []
                    ),
                    question_path,
                    base_path,
                    *certificate_paths,
                ],
            ),
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    print(f"Completed C3 certificate checks: {output_dir}")


if __name__ == "__main__":
    main()
