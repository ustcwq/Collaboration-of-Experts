from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import environment_manifest, sha256_file, write_csv, write_json, write_jsonl
from .aggregate_cfmad_style import authenticate_completed_cfmad
from .aggregate_pre_pair_style import (
    BUDGET_MATCHED_METHOD as PREPAIR_BUDGET_MATCHED_METHOD,
    TOP2_METHOD as PREPAIR_TOP2_METHOD,
    authenticate_completed_pre_pair,
)
from .cfmad_style import CFMAD_STYLE_METHOD
from .blind_falsification_jury import (
    FORBIDDEN_AUDIT_KEYS,
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
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
from .certificate_baselines import (
    MinorityVetoCourt,
    MinorityVetoVariant,
    MinoritySentinelStyleCourt,
    MinoritySentinelStyleVariant,
    StaticCalibrationVariant,
    StaticCheckerCalibrationCourt,
)
from .cross_examined_certificates import (
    C3Decision,
    C3Variant,
    CertificateCheck,
    CounterexampleCertificate,
    CrossExaminedCertificateCourt,
    build_certificate_check_prompt,
    build_certificate_prompt,
    build_certificate_prompt_v2,
    build_sealed_effect_reconstruction_prompt_v3,
    build_sealed_effect_witness_prompt_v3,
    build_target_blind_check_prompt_v2,
    parse_certificate_check_output,
    parse_certificate_output,
    parse_certificate_output_v2,
    parse_sealed_effect_witness_output_v3,
    parse_target_blind_check_output_v2,
    reconstructed_check_status,
    sealed_witness_candidate_fields,
)
from .equal_call_single_model import (
    aggregate_equal_call_answers,
    build_independent_solution_prompt,
    build_self_revision_prompt,
    parse_equal_call_answer,
)
from .evaluate_bfj_development import (
    _comparison,
    _expert_accuracies,
    _majority_answer,
    _method_summary,
    _source_labels,
    _subset_rows,
    _weighted_answer,
    leave_one_environment_out,
)
from .schema import SourceTrainingLabels
from .sealed_counterfactual_parity import (
    ISOLATED_TRACE_VIEWS,
    PARITY_ORIENTATIONS,
    build_blind_counterfactual_parity_prompt_v4,
    build_blind_isolated_trace_audit_prompt_v7,
    build_commitment_conditioned_pair_audit_prompt_v8_ablation,
    build_commitment_conditioned_proof_audit_prompt_v8,
    build_committed_counterfactual_challenge_prompt_v6,
    build_hardened_blind_counterfactual_parity_prompt_v5,
    build_hardened_counterfactual_challenge_prompt_v5,
    build_sealed_counterfactual_challenge_prompt_v4,
    canonical_trace_index,
    combine_isolated_trace_audits,
    counterfactual_trace_slot,
    effect_option_sets,
    parse_committed_counterfactual_challenge_output_v6,
    parse_hardened_counterfactual_challenge_output_v5,
    parse_blind_counterfactual_parity_output_v4,
    parse_blind_isolated_trace_audit_output_v7,
    parse_commitment_conditioned_pair_audit_output_v8_ablation,
    parse_commitment_conditioned_proof_audit_output_v8,
    parse_sealed_counterfactual_challenge_output_v4,
    permute_committed_counterfactual_challenge,
    sealed_triple_matches,
    validate_c3_v8_mechanism_ablation,
)


REQUIRED_PROMPT_MECHANISM_ABLATIONS = (
    "no_checker_private_precommitment",
    "pair_visible_with_precommitment",
    "candidate_visible_commit_first",
    "unsealed_set_aware",
)


# This registry is deliberately stricter than the current v8 execution queue.  It keeps
# a strong development result from silently authorizing a blind test while a comparison
# promised by the protocol is still absent.  One prediction may satisfy two requirements
# only when the relationship is stated explicitly below.
_STATIC_REQUIRED_COMPARISONS = (
    (
        "full_development_best_single_envelope",
        "exact",
        ("full_development_best_single_descriptive",),
    ),
    ("deployable_nested_oof_best_single", "exact", ("best_single_nested_oof",)),
    ("majority_vote", "exact", ("majority_vote",)),
    ("source_accuracy_weighted_vote", "exact", ("source_weighted_vote",)),
    ("source_fitted_expert_confusion_bayes", "exact", ("source_confusion_bayes",)),
    (
        "cross_model_consensus",
        "exact_equivalent_answer_plurality",
        ("majority_vote",),
    ),
    (
        "uncalibrated_all_option_falsification",
        "exact",
        ("uncalibrated_certificates",),
    ),
    (
        "bfj_without_certificate_cross_examination",
        "exact_mechanism_ablation",
        ("c3_certificates_only",),
    ),
    (
        "direct_anonymous_multi_verifier_scoring",
        "exact",
        ("direct_anonymous_answer_vote",),
    ),
    (
        "agent_auditor_style_localized_divergence",
        "style_adaptation_pair_visible_bidirectional_localization",
        ("agent_auditor_style_localized_divergence",),
    ),
    (
        "beyond_consensus_static_validator_calibration",
        "style_adaptation",
        ("beyond_consensus_static_calibration_nested",),
    ),
    (
        "beyond_consensus_minority_veto",
        "style_adaptation",
        ("beyond_consensus_minority_veto_nested",),
    ),
    (
        "minority_sentinel_supervised_flip_gate",
        "style_adaptation",
        ("minority_sentinel_style_nested",),
    ),
    (
        "commit_first_candidate_aware_verification",
        "exact_equal_call_prompt_control",
        ("c3_prompt_ablation::candidate_visible_commit_first",),
    ),
    (
        "target_blind_reconstruction_without_conditional_reliability",
        "exact_uncalibrated_readout",
        ("uncalibrated_cross_examined_certificates",),
    ),
    (
        "unsealed_set_aware_verification",
        "exact_equal_call_prompt_control",
        ("c3_prompt_ablation::unsealed_set_aware",),
    ),
    (
        "sealed_effect_without_reveal_agreement_features",
        "exact_feature_ablation",
        ("c3_no_sealed_set_agreement",),
    ),
    (
        "target_aware_cross_examination_same_certificate_content",
        "exact_equal_call_prompt_control",
        ("c3_prompt_ablation::candidate_visible_commit_first",),
    ),
    (
        "el_dgr_style_conservative_certificate_admissibility",
        "style_adaptation_noncompensatory_multi_axis_gate",
        ("el_dgr_style_conservative_admissibility",),
    ),
    (
        "cfmad_style_preset_option_critique",
        "style_adaptation_seeded_two_stance_four_backbone_ensemble",
        (CFMAD_STYLE_METHOD,),
    ),
    (
        "without_generator_answer_dependence",
        "exact_feature_ablation",
        ("c3_no_generator_answer_dependence",),
    ),
    (
        "without_checker_answer_dependence",
        "exact_feature_ablation",
        ("c3_no_checker_answer_dependence",),
    ),
    (
        "without_generator_checker_pair_effects",
        "exact_feature_ablation",
        ("c3_no_generator_checker_pair_effects",),
    ),
    (
        "without_open_option_completion",
        "exact_feature_ablation",
        ("c3_closed_option_set",),
    ),
    (
        "without_conservative_intervention_gate",
        "exact_feature_ablation",
        ("c3_no_intervention_gate",),
    ),
    (
        "prepair_original_shape_two_candidate",
        "style_reproduction",
        (PREPAIR_TOP2_METHOD,),
    ),
    (
        "prepair_42_call_three_candidate",
        "style_reproduction",
        (PREPAIR_BUDGET_MATCHED_METHOD,),
    ),
)


@dataclass(frozen=True)
class C3DevelopmentData:
    questions: tuple[FalsificationQuestion, ...]
    base_predictions: tuple[BasePrediction, ...]
    certificates: tuple[CounterexampleCertificate, ...]
    checks: tuple[CertificateCheck, ...]
    answers: Mapping[str, str]
    dataset_by_question: Mapping[str, str]
    environment_by_question: Mapping[str, str]
    generation_quality: Mapping[str, Any]
    equal_call_predictions: Mapping[str, Mapping[str, str | None]] = field(
        default_factory=dict
    )
    equal_call_call_budgets: Mapping[str, float] = field(default_factory=dict)
    pre_pair_predictions: Mapping[str, Mapping[str, str | None]] = field(
        default_factory=dict
    )
    pre_pair_call_budgets: Mapping[str, float] = field(default_factory=dict)
    cfmad_predictions: Mapping[str, Mapping[str, str | None]] = field(
        default_factory=dict
    )
    cfmad_call_budgets: Mapping[str, float] = field(default_factory=dict)
    mechanism_ablation_checks: Mapping[str, tuple[CertificateCheck, ...]] = field(
        default_factory=dict
    )
    mechanism_ablation_quality: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate C3 with nested leave-one-environment-out development folds"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--equal-call-config", type=Path)
    parser.add_argument("--prepair-config", type=Path)
    parser.add_argument("--cfmad-config", type=Path)
    parser.add_argument(
        "--mechanism-ablation-config", type=Path, action="append", default=[]
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("C3 configuration must be a mapping")
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


def _recorded_input_hash(environment: Mapping[str, Any], path: Path) -> str | None:
    inputs = environment.get("input_hashes", {})
    if not isinstance(inputs, dict):
        return None
    direct = inputs.get(str(path))
    if isinstance(direct, str):
        return direct
    resolved = path.resolve()
    for raw_path, value in inputs.items():
        try:
            if Path(str(raw_path)).resolve() == resolved and isinstance(value, str):
                return value
        except OSError:
            continue
    return None


def _load_and_authenticate_equal_call_baselines(
    c3_config_path: Path,
    c3_config: Mapping[str, Any],
    baseline_config_path: Path | None,
    run_root: Path,
    question_by_id: Mapping[str, FalsificationQuestion],
    question_path: Path,
) -> tuple[
    dict[str, dict[str, str | None]],
    dict[str, float],
    list[dict[str, Any]],
]:
    required = bool(
        c3_config.get("acceptance", {}).get(
            "require_equal_call_single_model_baselines", False
        )
    )
    if baseline_config_path is None:
        if required:
            raise PermissionError(
                "Equal-call single-model baselines are required but no sidecar config was supplied"
            )
        return {}, {}, []
    baseline_config = _load_config(baseline_config_path)
    if int(baseline_config.get("protocol_version", -1)) != 1:
        raise ValueError("Unknown equal-call single-model protocol")
    if baseline_config.get("c3_config_sha256") != sha256_file(c3_config_path):
        raise PermissionError("Equal-call sidecar is not bound to this C3 configuration")
    configured_root = Path(str(baseline_config.get("run_root", "")))
    if configured_root.resolve() != run_root.resolve():
        raise PermissionError("Equal-call sidecar points to a different run root")
    policy = baseline_config.get("data_policy", {})
    if (
        policy.get("generation_reads_labels") is not False
        or policy.get("model_pool_equals_prefrozen_c3_generator_checker_pool") is not True
        or policy.get("development_accuracy_used_for_model_selection") is not False
        or policy.get("certificate_or_check_outputs_used_for_model_selection") is not False
        or policy.get("target_labels_control_generation_or_aggregation") is not False
    ):
        raise PermissionError("Equal-call sidecar lacks the frozen label-free boundary")
    frozen_pool = {
        str(value) for value in c3_config["certificate_models"]
    }.union(str(value) for value in c3_config["checker_models"])
    models = tuple(str(value) for value in baseline_config.get("models", ()))
    if set(models) != frozen_pool or len(models) != len(frozen_pool):
        raise PermissionError("Equal-call model pool differs from the prefrozen C3 pool")
    methods = baseline_config.get("methods", {})
    if set(methods) != {"self_consistency", "self_revision"}:
        raise PermissionError("Equal-call protocol lacks a required single-model control")
    parity_multiplier = (
        2
        if str(c3_config.get("check_generation", {}).get("prompt_version", ""))
        in {
            "blind_counterfactual_parity_v4",
            "hardened_blind_counterfactual_parity_v5",
            "blind_isolated_trace_audit_v7",
            "commitment_conditioned_proof_audit_v8",
        }
        else 1
    )
    expected_calls = len(c3_config["experts"]) + len(
        c3_config["certificate_models"]
    ) + parity_multiplier * sum(
        checker != generator
        for generator in c3_config["certificate_models"]
        for checker in c3_config["checker_models"]
    )
    if int(baseline_config.get("calls_per_question", -1)) != expected_calls:
        raise PermissionError("Equal-call budget differs from the C3 call budget")
    if int(methods["self_consistency"].get("samples", -1)) != expected_calls:
        raise PermissionError("Self-consistency call budget is not equal to C3")
    revision_initial = int(methods["self_revision"].get("initial_samples", -1))
    revisions_per_initial = int(
        methods["self_revision"].get("revisions_per_initial", -1)
    )
    if revision_initial * (1 + revisions_per_initial) != expected_calls:
        raise PermissionError("Self-revision call budget is not equal to C3")

    source_hashes = {
        "independent_prompt_builder_sha256": hashlib.sha256(
            inspect.getsource(build_independent_solution_prompt).encode("utf-8")
        ).hexdigest(),
        "revision_prompt_builder_sha256": hashlib.sha256(
            inspect.getsource(build_self_revision_prompt).encode("utf-8")
        ).hexdigest(),
        "parser_sha256": hashlib.sha256(
            inspect.getsource(parse_equal_call_answer).encode("utf-8")
        ).hexdigest(),
        "aggregator_sha256": hashlib.sha256(
            inspect.getsource(aggregate_equal_call_answers).encode("utf-8")
        ).hexdigest(),
    }
    acceptance = baseline_config["acceptance"]
    minimum_valid_fraction = float(
        acceptance["minimum_valid_final_sample_fraction_per_question"]
    )
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("Equal-call valid-sample fraction must be in (0, 1]")
    expected_question_ids = set(question_by_id)
    predictions: dict[str, dict[str, str | None]] = {}
    budgets: dict[str, float] = {}
    quality: list[dict[str, Any]] = []
    for method in sorted(methods):
        expected_initial = (
            expected_calls if method == "self_consistency" else revision_initial
        )
        expected_revision = 0 if method == "self_consistency" else (
            revision_initial * revisions_per_initial
        )
        final_phase = "initial" if method == "self_consistency" else "revision"
        for model in models:
            directory = run_root / "equal_call_single_model" / method / model
            sample_path = directory / "samples.jsonl"
            prediction_path = directory / "predictions.jsonl"
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "completed_label_free_equal_call_single_model":
                raise PermissionError(f"Equal-call control is incomplete: {method}/{model}")
            if (
                manifest.get("model") != model
                or manifest.get("method") != method
                or manifest.get("labels_read") is not False
            ):
                raise PermissionError(f"Equal-call identity or label boundary failed: {method}/{model}")
            if int(manifest.get("protocol_version", -1)) != int(
                baseline_config["protocol_version"]
            ):
                raise PermissionError(f"Equal-call protocol drifted: {method}/{model}")
            if int(manifest.get("questions", -1)) != len(question_by_id):
                raise RuntimeError(f"Equal-call question coverage is incomplete: {method}/{model}")
            if int(manifest.get("calls_per_question", -1)) != expected_calls:
                raise PermissionError(f"Equal-call budget drifted: {method}/{model}")
            if int(manifest.get("actual_model_calls", -1)) != (
                len(question_by_id) * expected_calls
            ):
                raise RuntimeError(f"Equal-call actual call count differs: {method}/{model}")
            expected_file_hashes = {
                "question_sha256": sha256_file(question_path),
                "sample_sha256": sha256_file(sample_path),
                "prediction_sha256": sha256_file(prediction_path),
                "c3_config_sha256": sha256_file(c3_config_path),
                "baseline_config_sha256": sha256_file(baseline_config_path),
            }
            for key, digest in {**expected_file_hashes, **source_hashes}.items():
                if manifest.get(key) != digest:
                    raise PermissionError(f"Equal-call {key} differs: {method}/{model}")
            environment = manifest.get("environment", {})
            if _recorded_input_hash(environment, c3_config_path) != sha256_file(
                c3_config_path
            ) or _recorded_input_hash(environment, baseline_config_path) != sha256_file(
                baseline_config_path
            ):
                raise PermissionError(f"Equal-call config provenance differs: {method}/{model}")

            sample_rows = _read_jsonl(sample_path)
            if len(sample_rows) != len(question_by_id) * expected_calls:
                raise RuntimeError(f"Equal-call samples are incomplete: {method}/{model}")
            by_question_phase: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = defaultdict(list)
            seen_samples: set[tuple[str, str, int]] = set()
            parsed_samples = 0
            truncated = 0
            for row in sample_rows:
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise PermissionError(
                        f"Equal-call sample contains labels: {sorted(leaked)}"
                    )
                question_id = str(row["question_id"])
                question = question_by_id.get(question_id)
                if question is None:
                    raise ValueError("Equal-call sample references an unknown question")
                if (
                    row.get("dataset") != question.dataset
                    or row.get("environment") != question.environment
                ):
                    raise PermissionError("Equal-call sample metadata differs from its question")
                phase = str(row["phase"])
                if phase not in {"initial", "revision"}:
                    raise ValueError("Equal-call sample has an unknown phase")
                identity = (question_id, phase, int(row["sample_index"]))
                if identity in seen_samples:
                    raise ValueError(f"Duplicate equal-call sample: {identity}")
                seen_samples.add(identity)
                reparsed = parse_equal_call_answer(
                    str(row["raw_output"]), question.option_labels
                )
                stored = (
                    None if row.get("prediction") is None else str(row["prediction"]),
                    None if row.get("reason") is None else str(row["reason"]),
                    None if row.get("parse_error") is None else str(row["parse_error"]),
                )
                if reparsed != stored:
                    raise PermissionError("Equal-call sample differs from the frozen parser")
                parsed_samples += int(reparsed[2] is None)
                truncated += int(bool(row["prompt_was_truncated"]))
                by_question_phase[(question_id, phase)].append(row)
            for question_id in expected_question_ids:
                initial = by_question_phase[(question_id, "initial")]
                revision = by_question_phase[(question_id, "revision")]
                if {int(row["sample_index"]) for row in initial} != set(
                    range(expected_initial)
                ) or len(initial) != expected_initial:
                    raise RuntimeError("Equal-call initial sample grid is incomplete")
                if {int(row["sample_index"]) for row in revision} != set(
                    range(expected_revision)
                ) or len(revision) != expected_revision:
                    raise RuntimeError("Equal-call revision sample grid is incomplete")
                if method == "self_consistency":
                    if any(row.get("parent_sample_index") is not None for row in initial):
                        raise PermissionError("Self-consistency unexpectedly has parent samples")
                else:
                    if any(row.get("parent_sample_index") is not None for row in initial):
                        raise PermissionError("Self-revision initial samples have parents")
                    if {
                        int(row["parent_sample_index"]) for row in revision
                    } != set(range(revision_initial)):
                        raise RuntimeError("Self-revision lacks one revision per initial sample")

            prediction_rows = _read_jsonl(prediction_path)
            if len(prediction_rows) != len(question_by_id):
                raise RuntimeError(f"Equal-call predictions are incomplete: {method}/{model}")
            method_name = f"equal_call::{method}::{model}"
            method_predictions: dict[str, str | None] = {}
            final_valid_total = 0
            for row in prediction_rows:
                leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
                if leaked:
                    raise PermissionError(
                        f"Equal-call prediction contains labels: {sorted(leaked)}"
                    )
                question_id = str(row["question_id"])
                question = question_by_id.get(question_id)
                if question is None or question_id in method_predictions:
                    raise ValueError("Equal-call prediction identity is unknown or duplicated")
                final_rows = sorted(
                    by_question_phase[(question_id, final_phase)],
                    key=lambda value: int(value["sample_index"]),
                )
                aggregate, counts = aggregate_equal_call_answers(
                    [
                        None
                        if value.get("prediction") is None
                        else str(value["prediction"])
                        for value in final_rows
                    ],
                    question.option_labels,
                )
                stored_prediction = (
                    None if row.get("prediction") is None else str(row["prediction"])
                )
                stored_counts = {
                    str(key): int(value) for key, value in row.get("vote_counts", {}).items()
                }
                valid_final = sum(value.get("prediction") is not None for value in final_rows)
                minimum_valid = int(
                    np.ceil(
                        minimum_valid_fraction
                        * len(final_rows)
                    )
                )
                if (
                    stored_prediction != aggregate
                    or stored_counts != counts
                    or int(row.get("valid_final_samples", -1)) != valid_final
                    or int(row.get("final_samples", -1)) != len(final_rows)
                    or row.get("tie_breaking")
                    != "first_valid_sample_among_plurality_ties"
                ):
                    raise PermissionError("Equal-call prediction differs from frozen aggregation")
                if valid_final < minimum_valid:
                    raise RuntimeError("Equal-call control has too few valid samples for a question")
                method_predictions[question_id] = stored_prediction
                final_valid_total += valid_final
            if set(method_predictions) != expected_question_ids:
                raise RuntimeError("Equal-call prediction grid is incomplete")
            final_samples = len(question_by_id) * (
                expected_initial if final_phase == "initial" else expected_revision
            )
            final_parse_rate = final_valid_total / max(1, final_samples)
            if final_parse_rate < float(acceptance["minimum_final_parse_rate"]):
                raise RuntimeError("Equal-call final parse rate is below the frozen threshold")
            if truncated > int(acceptance["maximum_prompt_truncations"]):
                raise RuntimeError("Equal-call prompt truncation exceeds the frozen threshold")
            if (
                parsed_samples != int(manifest.get("parsed_samples", -1))
                or truncated != int(manifest.get("truncated_prompts", -1))
                or final_valid_total != int(manifest.get("parsed_final_samples", -1))
                or final_samples != int(manifest.get("final_samples", -1))
            ):
                raise RuntimeError("Equal-call manifest counts differ from authenticated rows")
            predictions[method_name] = method_predictions
            budgets[method_name] = float(expected_calls)
            quality.append(
                {
                    "method": method,
                    "model": model,
                    "calls_per_question": expected_calls,
                    "samples": len(sample_rows),
                    "sample_parse_rate": parsed_samples / max(1, len(sample_rows)),
                    "final_parse_rate": final_parse_rate,
                    "truncated_prompts": truncated,
                }
            )
    return predictions, budgets, quality


def _certificate_from_row(
    row: Mapping[str, Any],
    question: FalsificationQuestion,
    output_parser: Any,
    sealed_effect_set: bool,
    counterfactual_pair: bool,
    seed: int,
    post_commit_permutation: bool = False,
) -> CounterexampleCertificate:
    leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
    if leaked:
        raise PermissionError(f"C3 certificate contains labels: {sorted(leaked)}")
    if counterfactual_pair:
        expected_valid_trace = counterfactual_trace_slot(
            seed, question.question_id, str(row["generator_id"])
        )
        if post_commit_permutation:
            committed_challenge = output_parser(
                str(row["raw_output"]), question.option_labels
            )
            parsed_challenge, permutation_applied = (
                permute_committed_counterfactual_challenge(
                    committed_challenge, expected_valid_trace
                )
            )
            if (
                row.get("author_valid_trace") != committed_challenge.valid_trace
                or bool(row.get("post_commit_permutation_applied", False))
                != permutation_applied
            ):
                raise PermissionError(
                    "C3 v6 post-commit permutation metadata drifted"
                )
        else:
            parsed_challenge = output_parser(
                str(row["raw_output"]),
                question.option_labels,
                expected_valid_trace,
            )
        eliminated, supported = effect_option_sets(
            parsed_challenge.effect, parsed_challenge.option
        )
        verdict, alternative = sealed_witness_candidate_fields(
            str(row["candidate"]), question.option_labels, eliminated, supported
        )
        stored_challenge = (
            None if row.get("challenge_rule") is None else str(row["challenge_rule"]),
            None if row.get("trace_1") is None else str(row["trace_1"]),
            None if row.get("trace_2") is None else str(row["trace_2"]),
            None
            if row.get("first_differing_step") is None
            else str(row["first_differing_step"]),
            None
            if row.get("sealed_valid_trace") is None
            else int(row["sealed_valid_trace"]),
            None if row.get("sealed_effect") is None else str(row["sealed_effect"]),
            (
                None
                if not eliminated and not supported
                else (eliminated or supported)[0]
            ),
            int(row["confidence"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        reparsed_challenge = (
            parsed_challenge.rule,
            parsed_challenge.trace_1,
            parsed_challenge.trace_2,
            parsed_challenge.first_differing_step,
            parsed_challenge.valid_trace,
            parsed_challenge.effect,
            parsed_challenge.option,
            parsed_challenge.confidence,
            parsed_challenge.parse_error,
        )
        if reparsed_challenge != stored_challenge:
            raise PermissionError(
                "C3 counterfactual challenge differs from the frozen parser"
            )
        if (
            row.get("required_valid_trace") != expected_valid_trace
            or row.get("counterfactual_pair") is not True
            or row.get("claim_was_sealed") is not True
            or row.get("witness_id") is None
        ):
            raise PermissionError("C3 v4 counterfactual seal metadata drifted")
        stored = (
            verdict,
            parsed_challenge.confidence,
            alternative,
            parsed_challenge.rule,
            parsed_challenge.trace_1,
            parsed_challenge.first_differing_step,
            parsed_challenge.parse_error,
        )
        stored_candidate = (
            str(row["verdict"]),
            int(row["confidence"]),
            None if row.get("alternative") is None else str(row["alternative"]),
            None if row.get("premise") is None else str(row["premise"]),
            None if row.get("check") is None else str(row["check"]),
            None if row.get("failure") is None else str(row["failure"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        if stored != stored_candidate:
            raise PermissionError(
                "C3 expanded candidate differs from its counterfactual challenge"
            )
        if tuple(row.get("claimed_eliminated_options", ())) != eliminated or tuple(
            row.get("claimed_supported_options", ())
        ) != supported:
            raise PermissionError("C3 v4 stored signed effect differs from its seal")
        challenge_rule = parsed_challenge.rule
        trace_1 = parsed_challenge.trace_1
        trace_2 = parsed_challenge.trace_2
        first_differing_step = parsed_challenge.first_differing_step
        sealed_valid_trace = parsed_challenge.valid_trace
        sealed_effect = parsed_challenge.effect
    else:
        reparsed = output_parser(str(row["raw_output"]), question.option_labels)
    if sealed_effect_set and not counterfactual_pair:
        (
            confidence,
            premise,
            check_text,
            failure,
            eliminated,
            supported,
            parse_error,
        ) = reparsed
        verdict, alternative = sealed_witness_candidate_fields(
            str(row["candidate"]), question.option_labels, eliminated, supported
        )
        stored_witness = (
            int(row["confidence"]),
            None if row.get("premise") is None else str(row["premise"]),
            None if row.get("check") is None else str(row["check"]),
            None if row.get("failure") is None else str(row["failure"]),
            tuple(str(value) for value in row.get("claimed_eliminated_options", [])),
            tuple(str(value) for value in row.get("claimed_supported_options", [])),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        if reparsed != stored_witness:
            raise PermissionError("C3 sealed witness parse differs from the frozen parser")
        stored = (
            verdict,
            confidence,
            alternative,
            premise,
            check_text,
            failure,
            parse_error,
        )
        stored_candidate = (
            str(row["verdict"]),
            int(row["confidence"]),
            None if row.get("alternative") is None else str(row["alternative"]),
            None if row.get("premise") is None else str(row["premise"]),
            None if row.get("check") is None else str(row["check"]),
            None if row.get("failure") is None else str(row["failure"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        if stored != stored_candidate:
            raise PermissionError("C3 expanded candidate differs from its sealed witness")
        if row.get("claim_was_sealed") is not True or row.get("witness_id") is None:
            raise PermissionError("C3 v3 witness claim was not sealed")
        challenge_rule = None
        trace_1 = None
        trace_2 = None
        first_differing_step = None
        sealed_valid_trace = None
        sealed_effect = None
    elif not counterfactual_pair:
        stored = (
            str(row["verdict"]),
            int(row["confidence"]),
            None if row.get("alternative") is None else str(row["alternative"]),
            None if row.get("premise") is None else str(row["premise"]),
            None if row.get("check") is None else str(row["check"]),
            None if row.get("failure") is None else str(row["failure"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        if reparsed != stored:
            raise PermissionError("C3 certificate parse differs from the frozen parser")
        challenge_rule = None
        trace_1 = None
        trace_2 = None
        first_differing_step = None
        sealed_valid_trace = None
        sealed_effect = None
    certificate = CounterexampleCertificate(
        question_id=str(row["question_id"]),
        generator_id=str(row["generator_id"]),
        candidate=str(row["candidate"]),
        verdict=stored[0],
        confidence=stored[1],
        alternative=stored[2],
        premise=stored[3],
        check=stored[4],
        failure=stored[5],
        parse_error=stored[6],
        witness_id=None if row.get("witness_id") is None else str(row["witness_id"]),
        claimed_eliminated_options=tuple(
            str(value) for value in row.get("claimed_eliminated_options", [])
        ),
        claimed_supported_options=tuple(
            str(value) for value in row.get("claimed_supported_options", [])
        ),
        claim_was_sealed=bool(row.get("claim_was_sealed", False)),
        counterfactual_pair=counterfactual_pair,
        challenge_rule=challenge_rule,
        trace_1=trace_1,
        trace_2=trace_2,
        first_differing_step=first_differing_step,
        sealed_valid_trace=sealed_valid_trace,
        sealed_effect=sealed_effect,
    )
    if str(row["certificate_id"]) != certificate.certificate_id:
        raise PermissionError("C3 certificate ID is not canonical")
    if sealed_effect_set and certificate.witness_id != (
        f"{certificate.question_id}::{certificate.generator_id}"
    ):
        raise PermissionError("C3 sealed witness ID is not canonical")
    return certificate


def _check_from_row(
    row: Mapping[str, Any],
    question: FalsificationQuestion,
    certificate: CounterexampleCertificate,
    output_parser: Any,
    target_was_hidden: bool,
    counterfactual_pair: bool,
    sealed_claim_was_hidden: bool = True,
) -> CertificateCheck:
    leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
    if leaked:
        raise PermissionError(f"C3 certificate check contains labels: {sorted(leaked)}")
    raw_output = str(row["raw_output"])
    orientation = str(row.get("orientation", ""))
    isolated_trace_audit = orientation in ISOLATED_TRACE_VIEWS
    proof_obligation_audit = (
        output_parser
        in {
            parse_commitment_conditioned_proof_audit_output_v8,
            parse_commitment_conditioned_pair_audit_output_v8_ablation,
            parse_candidate_visible_proof_output_v8_control,
            parse_unsealed_proof_output_v8_control,
        }
    )
    if isolated_trace_audit:
        if not certificate.counterfactual_pair:
            raise PermissionError("C3 isolated audit crossed its protocol boundary")
        parsed_audit = output_parser(raw_output, question.option_labels)
        logic_status = parsed_audit.trace_status
        confidence = parsed_audit.confidence
        eliminated, supported = effect_option_sets(
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
        status = reconstructed_check_status(
            certificate, logic_status, eliminated, supported
        )
        reparsed = (
            logic_status,
            confidence,
            tuple(eliminated),
            tuple(supported),
            first_flaw,
            parse_error,
            canonical_valid_trace,
            reconstructed_effect,
            status,
            parsed_audit.flaw_code,
        )
        stored = (
            None if row.get("logic_status") is None else str(row["logic_status"]),
            int(row["confidence"]),
            tuple(str(value) for value in row.get("eliminated_options", [])),
            tuple(str(value) for value in row.get("supported_options", [])),
            None if row.get("first_flaw") is None else str(row["first_flaw"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
            (
                None
                if row.get("canonical_valid_trace") is None
                else int(row["canonical_valid_trace"])
            ),
            (
                None
                if row.get("reconstructed_effect") is None
                else str(row["reconstructed_effect"])
            ),
            str(row["status"]),
            None if row.get("flaw_code") is None else str(row["flaw_code"]),
        )
        if reparsed != stored:
            raise PermissionError("C3 isolated audit differs from the frozen parser")
        if proof_obligation_audit:
            proof_reparsed = (
                parsed_audit.countertest,
                parsed_audit.countertest_result,
                parsed_audit.recomputation,
                parsed_audit.commitment_relation,
            )
            proof_stored = tuple(
                None if row.get(field) is None else str(row[field])
                for field in (
                    "countertest",
                    "countertest_result",
                    "recomputation",
                    "commitment_relation",
                )
            )
            if proof_reparsed != proof_stored:
                raise PermissionError(
                    "C3 v8 proof obligations differ from the frozen parser"
                )
        expected_audit_protocol = {
            parse_commitment_conditioned_proof_audit_output_v8: (
                "commitment_conditioned_proof_audit_v8"
            ),
            parse_candidate_visible_proof_output_v8_control: (
                "candidate_visible_commit_first_v8_control"
            ),
            parse_unsealed_proof_output_v8_control: (
                "unsealed_set_aware_v8_control"
            ),
        }.get(output_parser, "isolated_trace_pointwise_v7")
        if (
            row.get("counterfactual_pair") is not True
            or row.get("target_was_hidden") is not target_was_hidden
            or row.get("sealed_claim_was_hidden")
            is not sealed_claim_was_hidden
            or row.get("audit_protocol") != expected_audit_protocol
            or row.get("trace_under_audit") != orientation
            or row.get("pair_status") is not None
            or row.get("presented_valid_trace") is not None
        ):
            raise PermissionError("C3 isolated-audit metadata drifted")
        independent = (
            None
            if row.get("independent_answer") is None
            else str(row["independent_answer"])
        )
    elif counterfactual_pair:
        if not target_was_hidden or not certificate.counterfactual_pair:
            raise PermissionError("C3 v4 parity audit crossed its protocol boundary")
        if orientation not in PARITY_ORIENTATIONS:
            raise PermissionError("C3 v4 parity orientation is invalid")
        parsed_audit = output_parser(raw_output, question.option_labels)
        logic_status = {
            "ONE_VALID": "VALID",
            "BOTH_INVALID": "INVALID",
            "INCONCLUSIVE": "INCONCLUSIVE",
        }[parsed_audit.pair_status]
        confidence = parsed_audit.confidence
        eliminated, supported = effect_option_sets(
            parsed_audit.effect, parsed_audit.option
        )
        first_flaw = parsed_audit.first_flaw
        parse_error = parsed_audit.parse_error
        presented_valid_trace = parsed_audit.presented_valid_trace
        canonical_valid_trace = canonical_trace_index(
            presented_valid_trace, orientation
        )
        reconstructed_effect = parsed_audit.effect
        status = reconstructed_check_status(
            certificate, logic_status, eliminated, supported
        )
        reparsed = (
            parsed_audit.pair_status,
            logic_status,
            confidence,
            tuple(eliminated),
            tuple(supported),
            first_flaw,
            parse_error,
            presented_valid_trace,
            canonical_valid_trace,
            reconstructed_effect,
            status,
        )
        stored = (
            None if row.get("pair_status") is None else str(row["pair_status"]),
            None if row.get("logic_status") is None else str(row["logic_status"]),
            int(row["confidence"]),
            tuple(str(value) for value in row.get("eliminated_options", [])),
            tuple(str(value) for value in row.get("supported_options", [])),
            None if row.get("first_flaw") is None else str(row["first_flaw"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
            (
                None
                if row.get("presented_valid_trace") is None
                else int(row["presented_valid_trace"])
            ),
            (
                None
                if row.get("canonical_valid_trace") is None
                else int(row["canonical_valid_trace"])
            ),
            (
                None
                if row.get("reconstructed_effect") is None
                else str(row["reconstructed_effect"])
            ),
            str(row["status"]),
        )
        if reparsed != stored:
            raise PermissionError("C3 v4 parity audit differs from the frozen parser")
        if proof_obligation_audit:
            proof_reparsed = (
                parsed_audit.countertest,
                parsed_audit.countertest_result,
                parsed_audit.recomputation,
                parsed_audit.commitment_relation,
            )
            proof_stored = tuple(
                None if row.get(field) is None else str(row[field])
                for field in (
                    "countertest",
                    "countertest_result",
                    "recomputation",
                    "commitment_relation",
                )
            )
            if proof_reparsed != proof_stored:
                raise PermissionError(
                    "C3 pairwise proof obligations differ from the frozen parser"
                )
        if (
            row.get("counterfactual_pair") is not True
            or row.get("target_was_hidden") is not True
            or row.get("sealed_claim_was_hidden") is not True
        ):
            raise PermissionError("C3 v4 blind-audit metadata drifted")
        independent = (
            None
            if row.get("independent_answer") is None
            else str(row["independent_answer"])
        )
    elif target_was_hidden:
        reparsed = output_parser(raw_output, question.option_labels)
        logic_status, confidence, eliminated, supported, first_flaw, parse_error = reparsed
        status = reconstructed_check_status(
            certificate, logic_status, eliminated, supported
        )
        stored = (
            None if row.get("logic_status") is None else str(row["logic_status"]),
            int(row["confidence"]),
            tuple(str(value) for value in row.get("eliminated_options", [])),
            tuple(str(value) for value in row.get("supported_options", [])),
            None if row.get("first_flaw") is None else str(row["first_flaw"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        if reparsed != stored or str(row["status"]) != status:
            raise PermissionError("C3 target-blind check differs from the frozen parser")
        if row.get("target_was_hidden") is not True:
            raise PermissionError("C3 target-blind check exposed its target")
        independent = (
            None
            if row.get("independent_answer") is None
            else str(row["independent_answer"])
        )
        orientation = "single"
        presented_valid_trace = None
        canonical_valid_trace = None
        reconstructed_effect = None
    else:
        reparsed = output_parser(raw_output, question.option_labels)
        status, confidence, independent, first_flaw, parse_error = reparsed
        eliminated = ()
        supported = ()
        logic_status = None
        stored = (
            str(row["status"]),
            int(row["confidence"]),
            None
            if row.get("independent_answer") is None
            else str(row["independent_answer"]),
            None if row.get("first_flaw") is None else str(row["first_flaw"]),
            None if row.get("parse_error") is None else str(row["parse_error"]),
        )
        if reparsed != stored:
            raise PermissionError("C3 check parse differs from the frozen parser")
        orientation = "single"
        presented_valid_trace = None
        canonical_valid_trace = None
        reconstructed_effect = None
    return CertificateCheck(
        certificate_id=str(row["certificate_id"]),
        question_id=str(row["question_id"]),
        generator_id=str(row["generator_id"]),
        checker_id=str(row["checker_id"]),
        candidate=str(row["candidate"]),
        status=status,
        confidence=confidence,
        independent_answer=independent,
        first_flaw=first_flaw,
        parse_error=parse_error,
        logic_status=logic_status,
        eliminated_options=tuple(eliminated),
        supported_options=tuple(supported),
        target_was_hidden=target_was_hidden,
        counterfactual_pair=counterfactual_pair,
        orientation=orientation,
        presented_valid_trace=presented_valid_trace,
        canonical_valid_trace=canonical_valid_trace,
        reconstructed_effect=reconstructed_effect,
    )


def _certificate_protocol_functions(
    config: Mapping[str, Any]
) -> tuple[Any, Any, bool, bool]:
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


def _check_protocol_functions(
    config: Mapping[str, Any]
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
            False,
            True,
        )
    raise ValueError(f"Unknown C3 check protocol: {protocol}")


def _counterfactual_audit_views(config: Mapping[str, Any]) -> tuple[str, ...]:
    if str(config["check_generation"]["prompt_version"]) in {
        "blind_isolated_trace_audit_v7",
        "commitment_conditioned_proof_audit_v8",
        *ISOLATED_PRIOR_ART_CONTROL_PROMPTS,
    }:
        return ISOLATED_TRACE_VIEWS
    return PARITY_ORIENTATIONS


def _is_proof_obligation_audit(config: Mapping[str, Any]) -> bool:
    return str(config["check_generation"]["prompt_version"]) in {
        "commitment_conditioned_proof_audit_v8",
        "commitment_conditioned_pair_audit_v8_ablation",
        *ISOLATED_PRIOR_ART_CONTROL_PROMPTS,
    }


def _validate_mechanism_ablation_config(
    base_config_path: Path,
    base_config: Mapping[str, Any],
    base_run_root: Path,
    ablation_config_path: Path,
    ablation_config: Mapping[str, Any],
) -> tuple[str, Path]:
    ablation = ablation_config.get("mechanism_ablation")
    if not isinstance(ablation, dict):
        raise TypeError("C3 mechanism ablation config lacks its boundary block")
    recorded_base_path = Path(str(ablation.get("base_config_path", "")))
    if recorded_base_path.resolve() != base_config_path.resolve():
        raise PermissionError("C3 mechanism ablation references another base config")
    if ablation.get("base_config_sha256") != sha256_file(base_config_path):
        raise PermissionError("C3 mechanism ablation base config hash differs")
    name = validate_c3_v8_mechanism_ablation(base_config, ablation_config)
    output_root = Path(str(ablation_config.get("output_root", "")))
    expected_output_root = base_run_root / "mechanism_ablations" / name
    if output_root.resolve() != expected_output_root.resolve():
        raise PermissionError("C3 mechanism ablation output root differs")
    return name, output_root


def load_and_authenticate_c3_data(
    config_path: Path,
    config: Mapping[str, Any],
    run_root: Path,
    equal_call_config_path: Path | None = None,
    pre_pair_config_path: Path | None = None,
    cfmad_config_path: Path | None = None,
    check_config_path: Path | None = None,
    check_config: Mapping[str, Any] | None = None,
    check_run_root: Path | None = None,
) -> C3DevelopmentData:
    effective_check_config = check_config or config
    effective_check_config_path = check_config_path or config_path
    effective_check_run_root = check_run_root or run_root
    if check_config is not None:
        for key in ("seed", "experts", "certificate_models", "checker_models"):
            if check_config.get(key) != config.get(key):
                raise PermissionError(
                    f"C3 mechanism ablation changed a fixed field: {key}"
                )
        if check_config.get("certificate_generation") != config.get(
            "certificate_generation"
        ):
            raise PermissionError(
                "C3 mechanism ablation changed certificate generation"
            )
        ablation = check_config.get("mechanism_ablation")
        if not isinstance(ablation, dict):
            raise PermissionError("C3 checker override is not a declared ablation")
    question_path = run_root / "development_observables" / "questions.jsonl"
    base_path = run_root / "development_observables" / "base_predictions.jsonl"
    observable_manifest_path = (
        run_root / "development_observables" / "observable_manifest.json"
    )
    label_path = run_root / "development_labels" / "labels.jsonl"
    label_manifest_path = run_root / "development_labels" / "label_manifest.json"
    observable_manifest = json.loads(observable_manifest_path.read_text(encoding="utf-8"))
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    if observable_manifest.get("question_sha256") != sha256_file(question_path):
        raise PermissionError("C3 questions changed after preparation")
    if observable_manifest.get("base_prediction_sha256") != sha256_file(base_path):
        raise PermissionError("C3 base predictions changed after preparation")
    if label_manifest.get("label_sha256") != sha256_file(label_path):
        raise PermissionError("C3 development labels changed after preparation")
    if observable_manifest.get("generation_reads_labels") is not False:
        raise PermissionError("C3 observable manifest lacks a label-free generation boundary")

    question_rows = _read_jsonl(question_path)
    base_rows = _read_jsonl(base_path)
    label_rows = _read_jsonl(label_path)
    for row in question_rows + base_rows:
        leaked = FORBIDDEN_AUDIT_KEYS.intersection(row)
        if leaked:
            raise PermissionError(f"C3 observable contains labels: {sorted(leaked)}")
    questions = tuple(
        FalsificationQuestion(
            question_id=str(row["question_id"]),
            dataset=str(row["dataset"]),
            environment=str(row["environment"]),
            question=str(row["question"]),
            options=tuple(str(value) for value in row["options"]),
            option_labels=tuple(str(value) for value in row["option_labels"]),
        )
        for row in question_rows
    )
    question_by_id = {row.question_id: row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("C3 development questions contain duplicate IDs")
    expected_questions = sum(int(row["expected_questions"]) for row in config["datasets"])
    if len(questions) != expected_questions:
        raise RuntimeError("C3 development question count differs from the protocol")
    experts = tuple(str(value) for value in config["experts"])
    base_predictions = tuple(
        BasePrediction(
            question_id=str(row["question_id"]),
            expert_id=str(row["expert_id"]),
            answer=None if row.get("prediction") is None else str(row["prediction"]),
        )
        for row in base_rows
    )
    expected_base = {
        (question.question_id, expert) for question in questions for expert in experts
    }
    actual_base = {(row.question_id, row.expert_id) for row in base_predictions}
    if actual_base != expected_base or len(base_predictions) != len(expected_base):
        raise RuntimeError("C3 base predictions are not a complete question/expert grid")
    base_answer_by_key = {
        (row.question_id, row.expert_id): row.answer for row in base_predictions
    }
    base_response_by_key = {
        (str(row["question_id"]), str(row["expert_id"])): str(
            row.get("response", "")
        )
        for row in base_rows
    }
    answers = {str(row["question_id"]): str(row["answer"]) for row in label_rows}
    if set(answers) != set(question_by_id) or len(answers) != len(label_rows):
        raise RuntimeError("C3 development answers are not aligned one-to-one")
    for question_id, answer in answers.items():
        if answer not in question_by_id[question_id].option_labels:
            raise ValueError(f"C3 development answer is outside its options: {question_id}")

    (
        certificate_prompt_builder,
        certificate_output_parser,
        sealed_effect_set,
        counterfactual_pair,
    ) = _certificate_protocol_functions(config)
    post_commit_permutation = (
        str(config["certificate_generation"]["prompt_version"])
        == "committed_counterfactual_permutation_v6"
    )
    certificate_prompt_hash = hashlib.sha256(
        inspect.getsource(certificate_prompt_builder).encode("utf-8")
    ).hexdigest()
    certificate_parser_hash = hashlib.sha256(
        inspect.getsource(certificate_output_parser).encode("utf-8")
    ).hexdigest()
    certificates: list[CounterexampleCertificate] = []
    certificate_quality: list[dict[str, Any]] = []
    certificate_hashes: dict[str, str] = {}
    expected_certificate_count = sum(len(row.option_labels) for row in questions)
    for generator in (str(value) for value in config["certificate_models"]):
        directory = run_root / "certificates" / generator
        path = directory / "certificates.jsonl"
        manifest = json.loads(
            (directory / "certificate_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "completed_label_free_c3_certificates":
            raise PermissionError(f"C3 generator is incomplete: {generator}")
        if manifest.get("generator") != generator or manifest.get("labels_read") is not False:
            raise PermissionError(f"C3 generator identity or label boundary failed: {generator}")
        if sealed_effect_set:
            if manifest.get("claims_are_sealed_from_checkers") is not True:
                raise PermissionError(f"C3 v3 claims were not sealed: {generator}")
            if int(manifest.get("witnesses", -1)) != len(questions):
                raise RuntimeError(f"C3 v3 witness coverage is incomplete: {generator}")
        if counterfactual_pair:
            if (
                manifest.get("counterfactual_pairs") is not True
                or manifest.get("private_stage0_responses_read") is not True
            ):
                raise PermissionError(
                    f"C3 v4 private counterfactual boundary failed: {generator}"
                )
            if manifest.get("base_prediction_sha256") != sha256_file(base_path):
                raise PermissionError(
                    f"C3 v4 generator used different private Stage-0 traces: {generator}"
                )
            recorded_post_commit = manifest.get("post_commit_permutation")
            if (
                post_commit_permutation
                and recorded_post_commit is not True
            ) or (
                not post_commit_permutation
                and recorded_post_commit not in {None, False}
            ):
                raise PermissionError(
                    f"C3 post-commit permutation boundary differs: {generator}"
                )
        if int(manifest.get("questions", -1)) != len(questions):
            raise RuntimeError(f"C3 generator question count is incomplete: {generator}")
        if int(manifest.get("certificates", -1)) != expected_certificate_count:
            raise RuntimeError(f"C3 generator option coverage is incomplete: {generator}")
        if manifest.get("question_sha256") != sha256_file(question_path):
            raise PermissionError(f"C3 generator used different questions: {generator}")
        if manifest.get("certificate_sha256") != sha256_file(path):
            raise PermissionError(f"C3 certificate file changed: {generator}")
        if manifest.get("prompt_builder_sha256") != certificate_prompt_hash:
            raise PermissionError(f"C3 certificate prompt drifted: {generator}")
        if manifest.get("parser_sha256") != certificate_parser_hash:
            raise PermissionError(f"C3 certificate parser drifted: {generator}")
        if manifest.get("prompt_version") != str(
            config["certificate_generation"]["prompt_version"]
        ) or manifest.get("parser_version") != str(
            config["certificate_generation"]["parser_version"]
        ):
            raise PermissionError(f"C3 certificate protocol version differs: {generator}")
        if _recorded_input_hash(manifest.get("environment", {}), config_path) != sha256_file(config_path):
            raise PermissionError(f"C3 generator used a different config: {generator}")
        rows = _read_jsonl(path)
        actual_keys: set[tuple[str, str, str]] = set()
        authenticated_rows: list[
            tuple[Mapping[str, Any], CounterexampleCertificate]
        ] = []
        parsed = 0
        truncated = 0
        by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            question = question_by_id.get(str(row["question_id"]))
            if question is None:
                raise ValueError("C3 certificate references an unknown question")
            if (
                row.get("dataset") != question.dataset
                or row.get("environment") != question.environment
            ):
                raise PermissionError("C3 certificate question metadata drifted")
            certificate = _certificate_from_row(
                row,
                question,
                certificate_output_parser,
                sealed_effect_set,
                counterfactual_pair,
                int(config["certificate_generation"]["seed"]),
                post_commit_permutation,
            )
            identity = (
                certificate.question_id,
                certificate.generator_id,
                certificate.candidate,
            )
            if identity in actual_keys:
                raise ValueError(f"Duplicate C3 certificate: {identity}")
            actual_keys.add(identity)
            certificates.append(certificate)
            authenticated_rows.append((row, certificate))
            parsed += int(certificate.parse_error is None)
            truncated += int(bool(row["prompt_was_truncated"]))
            by_dataset[question.dataset]["total"] += 1
            by_dataset[question.dataset]["parsed"] += int(
                certificate.parse_error is None
            )
        expected_keys = {
            (question.question_id, generator, candidate)
            for question in questions
            for candidate in question.option_labels
        }
        if actual_keys != expected_keys or len(rows) != len(expected_keys):
            raise RuntimeError(f"C3 generator lacks exact option coverage: {generator}")
        if parsed != int(manifest["parsed_certificates"]):
            raise RuntimeError(f"C3 parsed certificate count differs: {generator}")
        if truncated != int(manifest["truncated_prompts"]):
            raise RuntimeError(f"C3 truncation count differs: {generator}")
        if counterfactual_pair:
            witness_groups: dict[
                str, list[tuple[Mapping[str, Any], CounterexampleCertificate]]
            ] = defaultdict(list)
            for row, certificate in authenticated_rows:
                if certificate.witness_id is None:
                    raise PermissionError("C3 v4 certificate lacks a witness ID")
                witness_groups[certificate.witness_id].append((row, certificate))
            if len(witness_groups) != len(questions):
                raise RuntimeError(f"C3 v4 witness coverage is incomplete: {generator}")
            parsed_witnesses = 0
            abstaining_witnesses = 0
            nonabstaining_witnesses = 0
            truncated_model_calls = 0
            all_option_effect_witnesses = 0
            slot_counts = {str(slot): 0 for slot in (1, 2)}
            for witness_id, group in witness_groups.items():
                source_row, representative = group[0]
                def expansion_signature(
                    member_row: Mapping[str, Any],
                    member: CounterexampleCertificate,
                ) -> tuple[Any, ...]:
                    return (
                        member.question_id,
                        member.generator_id,
                        member.confidence,
                        member.premise,
                        member.check,
                        member.failure,
                        member.parse_error,
                        member.claimed_eliminated_options,
                        member.claimed_supported_options,
                        member.challenge_rule,
                        member.trace_1,
                        member.trace_2,
                        member.first_differing_step,
                        member.sealed_valid_trace,
                        member.sealed_effect,
                        str(member_row["raw_output"]),
                        member_row.get("required_valid_trace"),
                        member_row.get("author_valid_trace"),
                        bool(
                            member_row.get(
                                "post_commit_permutation_applied", False
                            )
                        ),
                        bool(member_row["prompt_was_truncated"]),
                    )

                signature = expansion_signature(source_row, representative)
                if any(
                    expansion_signature(member_row, member) != signature
                    for member_row, member in group
                ):
                    raise PermissionError(
                        f"C3 v4 candidate expansion drifted within {witness_id}"
                    )
                parsed_witnesses += int(representative.parse_error is None)
                abstaining_witnesses += int(
                    representative.parse_error is None
                    and representative.sealed_valid_trace is None
                )
                nonabstaining_witnesses += int(
                    representative.parse_error is None
                    and representative.sealed_valid_trace in (1, 2)
                )
                truncated_model_calls += int(bool(source_row["prompt_was_truncated"]))
                all_option_effect_witnesses += int(
                    len(
                        set(representative.claimed_eliminated_options).union(
                            representative.claimed_supported_options
                        )
                    )
                    == len(question_by_id[representative.question_id].option_labels)
                )
                slot_counts[str(int(source_row["required_valid_trace"]))] += 1
            replayed_manifest_values = {
                "model_calls": len(witness_groups),
                "parsed_witnesses": parsed_witnesses,
                "truncated_model_calls": truncated_model_calls,
                "abstaining_witnesses": abstaining_witnesses,
                "nonabstaining_witnesses": nonabstaining_witnesses,
                "all_option_effect_witnesses": all_option_effect_witnesses,
                "required_valid_trace_counts": slot_counts,
            }
            if post_commit_permutation:
                replayed_manifest_values.update(
                    {
                        "permuted_witnesses": sum(
                            bool(group[0][0].get("post_commit_permutation_applied"))
                            for group in witness_groups.values()
                        ),
                        "author_valid_trace_counts": {
                            str(slot): sum(
                                group[0][0].get("author_valid_trace") == slot
                                for group in witness_groups.values()
                            )
                            for slot in (1, 2)
                        },
                    }
                )
            for key, value in replayed_manifest_values.items():
                if manifest.get(key) != value:
                    raise PermissionError(
                        f"C3 v4 generator manifest field drifted ({key}): {generator}"
                    )
            all_option_effect_rate = all_option_effect_witnesses / max(
                1, len(witness_groups)
            )
            if not np.isclose(
                float(manifest.get("all_option_effect_rate", -1.0)),
                all_option_effect_rate,
            ):
                raise PermissionError(
                    f"C3 v4 all-option effect rate drifted: {generator}"
                )
            quality_gate = config.get("smoke_acceptance", {})
            witness_parse_rate = parsed_witnesses / max(1, len(witness_groups))
            nonabstaining_rate = nonabstaining_witnesses / max(
                1, len(witness_groups)
            )
            if witness_parse_rate < float(
                quality_gate.get("minimum_witness_parse_rate", 0.0)
            ):
                raise RuntimeError(f"C3 v4 witness parse gate failed: {generator}")
            if nonabstaining_rate < float(
                quality_gate.get("minimum_nonabstaining_witness_rate", 0.0)
            ):
                raise RuntimeError(
                    f"C3 v4 non-abstaining witness gate failed: {generator}"
                )
            if all_option_effect_rate > float(
                quality_gate.get("maximum_all_option_effect_rate", 1.0)
            ):
                raise RuntimeError(
                    f"C3 v4 all-option effect gate failed: {generator}"
                )
            if truncated_model_calls > int(
                quality_gate.get("maximum_prompt_truncations", len(witness_groups))
            ):
                raise RuntimeError(f"C3 v4 truncation gate failed: {generator}")
        certificate_hashes[str(path)] = sha256_file(path)
        certificate_quality_row: dict[str, Any] = {
            "generator": generator,
            "certificates": len(rows),
            "parsed": parsed,
            "parse_rate": parsed / max(1, len(rows)),
            "truncated": truncated,
            "by_dataset": {
                dataset: {
                    "certificates": int(counts["total"]),
                    "parsed": int(counts["parsed"]),
                    "parse_rate": counts["parsed"] / max(1, counts["total"]),
                }
                for dataset, counts in sorted(by_dataset.items())
            },
        }
        if counterfactual_pair:
            certificate_quality_row.update(
                {
                    "witnesses": len(witness_groups),
                    "parsed_witnesses": parsed_witnesses,
                    "witness_parse_rate": parsed_witnesses
                    / max(1, len(witness_groups)),
                    "abstaining_witnesses": abstaining_witnesses,
                    "nonabstaining_witnesses": nonabstaining_witnesses,
                    "nonabstaining_witness_rate": nonabstaining_witnesses
                    / max(1, len(witness_groups)),
                    "truncated_model_calls": truncated_model_calls,
                    "all_option_effect_rate": all_option_effect_rate,
                    "required_valid_trace_counts": slot_counts,
                }
            )
        certificate_quality.append(certificate_quality_row)
    certificate_by_id = {row.certificate_id: row for row in certificates}
    if len(certificate_by_id) != len(certificates):
        raise ValueError("C3 combined certificates contain duplicate IDs")

    (
        check_prompt_builder,
        check_output_parser,
        target_was_hidden,
        sealed_claim_was_hidden,
        check_counterfactual_pair,
    ) = (
        _check_protocol_functions(effective_check_config)
    )
    if check_counterfactual_pair != counterfactual_pair:
        raise PermissionError("C3 certificate/check counterfactual protocols disagree")
    check_audit_views = (
        _counterfactual_audit_views(effective_check_config)
        if check_counterfactual_pair
        else ("single",)
    )
    isolated_trace_audit = check_audit_views == ISOLATED_TRACE_VIEWS
    proof_obligation_audit = _is_proof_obligation_audit(
        effective_check_config
    )
    check_prompt_hash = hashlib.sha256(
        inspect.getsource(check_prompt_builder).encode("utf-8")
    ).hexdigest()
    check_parser_hash = hashlib.sha256(
        inspect.getsource(check_output_parser).encode("utf-8")
    ).hexdigest()
    checks: list[CertificateCheck] = []
    check_quality: list[dict[str, Any]] = []
    for checker in (
        str(value) for value in effective_check_config["checker_models"]
    ):
        directory = effective_check_run_root / "checks" / checker
        path = directory / "checks.jsonl"
        manifest = json.loads((directory / "check_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_label_free_c3_checks":
            raise PermissionError(f"C3 checker is incomplete: {checker}")
        if manifest.get("checker") != checker or manifest.get("labels_read") is not False:
            raise PermissionError(f"C3 checker identity or label boundary failed: {checker}")
        if manifest.get("question_sha256") != sha256_file(question_path):
            raise PermissionError(f"C3 checker used different questions: {checker}")
        if manifest.get("check_sha256") != sha256_file(path):
            raise PermissionError(f"C3 check file changed: {checker}")
        if manifest.get("prompt_builder_sha256") != check_prompt_hash:
            raise PermissionError(f"C3 check prompt drifted: {checker}")
        if manifest.get("parser_sha256") != check_parser_hash:
            raise PermissionError(f"C3 check parser drifted: {checker}")
        if manifest.get("prompt_version") != str(
            effective_check_config["check_generation"]["prompt_version"]
        ) or manifest.get("parser_version") != str(
            effective_check_config["check_generation"]["parser_version"]
        ):
            raise PermissionError(f"C3 check protocol version differs: {checker}")
        if manifest.get("target_was_hidden") is not target_was_hidden:
            raise PermissionError(f"C3 target-blind boundary differs: {checker}")
        if manifest.get("sealed_claim_was_hidden") is not sealed_claim_was_hidden:
            raise PermissionError(f"C3 sealed-claim boundary differs: {checker}")
        if check_config is not None:
            expected_ablation = {
                "name": str(ablation["name"]),
                "base_config_path": str(ablation["base_config_path"]),
                "base_config_sha256": str(ablation["base_config_sha256"]),
            }
            if manifest.get("mechanism_ablation") != expected_ablation:
                raise PermissionError(
                    f"C3 mechanism ablation metadata differs: {checker}"
                )
            if Path(str(manifest.get("input_run_root", ""))).resolve() != run_root.resolve():
                raise PermissionError(
                    f"C3 mechanism ablation used another input run: {checker}"
                )
        if check_counterfactual_pair:
            if manifest.get("counterfactual_pairs") is not True:
                raise PermissionError(f"C3 counterfactual checker differs: {checker}")
            if isolated_trace_audit:
                combiner_hash = hashlib.sha256(
                    inspect.getsource(combine_isolated_trace_audits).encode("utf-8")
                ).hexdigest()
                expected_audit_protocol = {
                    "commitment_conditioned_proof_audit_v8": (
                        "commitment_conditioned_proof_audit_v8"
                    ),
                    CANDIDATE_VISIBLE_PROMPT_VERSION: (
                        "candidate_visible_commit_first_v8_control"
                    ),
                    UNSEALED_PROMPT_VERSION: "unsealed_set_aware_v8_control",
                }.get(
                    str(
                        effective_check_config["check_generation"][
                            "prompt_version"
                        ]
                    ),
                    "isolated_trace_pointwise_v7",
                )
                if (
                    manifest.get("audit_protocol") != expected_audit_protocol
                    or manifest.get("isolated_trace_views")
                    != list(ISOLATED_TRACE_VIEWS)
                    or manifest.get("parity_orientations") is not None
                    or manifest.get("pair_combiner_sha256") != combiner_hash
                ):
                    raise PermissionError(
                        f"C3 isolated checker protocol differs: {checker}"
                    )
                if (
                    manifest.get("private_stage0_responses_read")
                    is not proof_obligation_audit
                    or manifest.get("proof_obligations_required")
                    is not proof_obligation_audit
                ):
                    raise PermissionError(
                        f"C3 v8 private proof boundary differs: {checker}"
                    )
            elif (
                manifest.get("parity_orientations")
                != list(PARITY_ORIENTATIONS)
            ):
                raise PermissionError(
                    f"C3 v4 checker parity protocol differs: {checker}"
                )
            elif proof_obligation_audit and (
                manifest.get("audit_protocol")
                != "commitment_conditioned_pair_audit_v8_ablation"
                or manifest.get("private_stage0_responses_read") is not True
                or manifest.get("proof_obligations_required") is not True
            ):
                raise PermissionError(
                    f"C3 pair-visible proof boundary differs: {checker}"
                )
        if manifest.get("base_prediction_sha256") != sha256_file(base_path):
            raise PermissionError(f"C3 checker used different Stage-0 commitments: {checker}")
        if (
            target_was_hidden or proof_obligation_audit
        ) and manifest.get("commitments_from_stage0") is not True:
            raise PermissionError(f"C3 checker lacks frozen Stage-0 commitments: {checker}")
        if _recorded_input_hash(
            manifest.get("environment", {}), effective_check_config_path
        ) != sha256_file(effective_check_config_path):
            raise PermissionError(f"C3 checker used a different config: {checker}")
        expected_inputs = {
            path: digest
            for path, digest in certificate_hashes.items()
            if Path(path).parent.name != checker
        }
        if manifest.get("input_certificate_hashes") != expected_inputs:
            raise PermissionError(f"C3 checker certificate inputs differ: {checker}")
        expected_ids = {
            certificate.certificate_id
            for certificate in certificates
            if certificate.generator_id != checker
            and certificate.parse_error is None
            and (
                not check_counterfactual_pair
                or certificate.sealed_valid_trace is not None
            )
        }
        rows = _read_jsonl(path)
        expected_row_keys = {
            (certificate_id, orientation)
            for certificate_id in expected_ids
            for orientation in check_audit_views
        }
        actual_row_keys: set[tuple[str, str]] = set()
        authenticated_check_rows: list[
            tuple[Mapping[str, Any], CertificateCheck, CounterexampleCertificate]
        ] = []
        parsed = 0
        truncated = 0
        by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            question = question_by_id.get(str(row["question_id"]))
            if question is None:
                raise ValueError("C3 check references an unknown question")
            if (
                row.get("dataset") != question.dataset
                or row.get("environment") != question.environment
            ):
                raise PermissionError("C3 check question metadata drifted")
            certificate_id = str(row["certificate_id"])
            certificate = certificate_by_id.get(certificate_id)
            if certificate is None:
                raise ValueError("C3 check references an unknown certificate")
            if proof_obligation_audit:
                if any(
                    value is None
                    for value in (
                        certificate.challenge_rule,
                        certificate.trace_1,
                        certificate.trace_2,
                        certificate.first_differing_step,
                    )
                ):
                    raise PermissionError("C3 v8 certificate lacks prompt content")
                prompt_arguments = (
                    question,
                    str(certificate.challenge_rule),
                    str(certificate.trace_1),
                    str(certificate.trace_2),
                    str(certificate.first_differing_step),
                    str(row.get("orientation", "")),
                    base_response_by_key[(question.question_id, checker)],
                )
                expected_raw_prompt = (
                    check_prompt_builder(*prompt_arguments, certificate)
                    if str(
                        effective_check_config["check_generation"][
                            "prompt_version"
                        ]
                    )
                    in ISOLATED_PRIOR_ART_CONTROL_PROMPTS
                    else check_prompt_builder(*prompt_arguments)
                )
                expected_raw_prompt_sha256 = hashlib.sha256(
                    expected_raw_prompt.encode("utf-8")
                ).hexdigest()
                if row.get("raw_prompt_sha256") != expected_raw_prompt_sha256:
                    raise PermissionError(
                        "C3 v8 raw prompt does not replay from the checker's own Stage-0 response"
                    )
            check = _check_from_row(
                row,
                question,
                certificate,
                check_output_parser,
                target_was_hidden,
                check_counterfactual_pair,
                sealed_claim_was_hidden,
            )
            if target_was_hidden or proof_obligation_audit:
                expected_commitment = base_answer_by_key[
                    (check.question_id, check.checker_id)
                ]
                if expected_commitment not in question.option_labels:
                    expected_commitment = None
                if check.independent_answer != expected_commitment:
                    raise PermissionError(
                        "C3 checker commitment differs from its frozen Stage-0 answer"
                    )
                if row.get("commitment_source") != "stage0_base_prediction":
                    raise PermissionError("C3 checker commitment source is not frozen")
            if sealed_claim_was_hidden:
                if row.get("sealed_claim_was_hidden") is not True:
                    raise PermissionError("C3 v3 check exposed its sealed claim")
                if row.get("witness_id") != certificate.witness_id:
                    raise PermissionError("C3 v3 check witness identity drifted")
            elif row.get("sealed_claim_was_hidden") is not False:
                raise PermissionError("C3 unsealed control did not disclose its claim")
            if (
                check.question_id != certificate.question_id
                or check.generator_id != certificate.generator_id
                or check.candidate != certificate.candidate
                or check.checker_id != checker
            ):
                raise PermissionError("C3 check metadata differs from its certificate")
            row_key = (check.certificate_id, check.orientation)
            if row_key in actual_row_keys:
                raise ValueError("Duplicate C3 check for one checker/certificate")
            actual_row_keys.add(row_key)
            checks.append(check)
            authenticated_check_rows.append((row, check, certificate))
            parsed += int(check.parse_error is None)
            truncated += int(bool(row["prompt_was_truncated"]))
            by_dataset[question.dataset]["total"] += 1
            by_dataset[question.dataset]["parsed"] += int(check.parse_error is None)
        if actual_row_keys != expected_row_keys or len(rows) != len(expected_row_keys):
            raise RuntimeError(f"C3 checker lacks exact parsed-certificate coverage: {checker}")
        if len(rows) != int(manifest["certificates_checked"]):
            raise RuntimeError(f"C3 checker row count differs from manifest: {checker}")
        if parsed != int(manifest["parsed_checks"]):
            raise RuntimeError(f"C3 parsed check count differs: {checker}")
        if truncated != int(manifest["truncated_prompts"]):
            raise RuntimeError(f"C3 check truncation count differs: {checker}")
        if check_counterfactual_pair:
            if int(manifest.get("input_certificates", -1)) != len(expected_ids):
                raise RuntimeError(f"C3 v4 checker input count differs: {checker}")
            reconstructions: dict[
                tuple[str, str],
                list[
                    tuple[
                        Mapping[str, Any],
                        CertificateCheck,
                        CounterexampleCertificate,
                    ]
                ],
            ] = defaultdict(list)
            for row, check, certificate in authenticated_check_rows:
                if certificate.witness_id is None:
                    raise PermissionError("C3 v4 checked certificate lacks a witness ID")
                reconstructions[(certificate.witness_id, check.orientation)].append(
                    (row, check, certificate)
                )
            representatives: dict[
                tuple[str, str], tuple[Mapping[str, Any], CertificateCheck, CounterexampleCertificate]
            ] = {}
            for reconstruction_key, group in reconstructions.items():
                representative_row, representative_check, representative_certificate = group[0]

                def reconstruction_signature(
                    member_row: Mapping[str, Any], member: CertificateCheck
                ) -> tuple[Any, ...]:
                    return (
                        str(member_row["raw_output"]),
                        bool(member_row["prompt_was_truncated"]),
                        member.question_id,
                        member.generator_id,
                        member.checker_id,
                        member.confidence,
                        member.independent_answer,
                        member.first_flaw,
                        member.parse_error,
                        member.logic_status,
                        member.eliminated_options,
                        member.supported_options,
                        member.orientation,
                        member.presented_valid_trace,
                        member.canonical_valid_trace,
                        member.reconstructed_effect,
                        member_row.get("pair_status"),
                        member_row.get("countertest"),
                        member_row.get("countertest_result"),
                        member_row.get("recomputation"),
                        member_row.get("commitment_relation"),
                        member_row.get("raw_prompt_sha256"),
                    )

                signature = reconstruction_signature(
                    representative_row, representative_check
                )
                if any(
                    reconstruction_signature(member_row, member_check) != signature
                    for member_row, member_check, _ in group
                ):
                    raise PermissionError(
                        f"C3 v4 candidate-expanded audit drifted: {reconstruction_key}"
                    )
                representatives[reconstruction_key] = (
                    representative_row,
                    representative_check,
                    representative_certificate,
                )
            parity_groups: dict[
                str,
                dict[
                    str,
                    tuple[Mapping[str, Any], CertificateCheck, CounterexampleCertificate],
                ],
            ] = defaultdict(dict)
            for (witness_id, orientation), representative in representatives.items():
                parity_groups[witness_id][orientation] = representative
            position_invariant_pairs = 0
            sealed_triple_audits = 0
            paired_sealed_triple_matches = 0
            complete_isolated_trace_pairs = 0
            one_valid_one_invalid_pairs = 0
            isolated_sealed_triple_matches = 0
            for witness_id, orientations in parity_groups.items():
                if isolated_trace_audit:
                    if set(orientations) != set(ISOLATED_TRACE_VIEWS):
                        raise RuntimeError(
                            f"C3 v7 isolated trace views are incomplete: {witness_id}"
                        )
                    trace_1_row, trace_1_check, trace_1_certificate = orientations[
                        "trace_1"
                    ]
                    trace_2_row, trace_2_check, trace_2_certificate = orientations[
                        "trace_2"
                    ]
                    if trace_1_certificate.witness_id != trace_2_certificate.witness_id:
                        raise PermissionError(
                            "C3 v7 isolated pair crossed witness identities"
                        )
                    complete_isolated_trace_pairs += int(
                        trace_1_check.parse_error is None
                        and trace_2_check.parse_error is None
                    )
                    combined = combine_isolated_trace_audits(
                        check_output_parser(
                            str(trace_1_row["raw_output"]),
                            question_by_id[trace_1_check.question_id].option_labels,
                        ),
                        check_output_parser(
                            str(trace_2_row["raw_output"]),
                            question_by_id[trace_2_check.question_id].option_labels,
                        ),
                    )
                    one_valid_one_invalid_pairs += int(
                        combined.pair_status == "ONE_VALID"
                    )
                    combined_eliminated, combined_supported = effect_option_sets(
                        combined.effect, combined.option
                    )
                    isolated_sealed_triple_matches += int(
                        combined.parse_error is None
                        and combined.pair_status == "ONE_VALID"
                        and sealed_triple_matches(
                            trace_1_certificate.sealed_valid_trace,
                            trace_1_certificate.sealed_effect,
                            trace_1_certificate.claimed_eliminated_options,
                            trace_1_certificate.claimed_supported_options,
                            combined.presented_valid_trace,
                            combined.effect,
                            combined_eliminated,
                            combined_supported,
                        )
                    )
                    if (
                        trace_1_row.get("orientation") != "trace_1"
                        or trace_2_row.get("orientation") != "trace_2"
                    ):
                        raise PermissionError("C3 v7 isolated rows changed trace view")
                    continue
                if set(orientations) != set(PARITY_ORIENTATIONS):
                    raise RuntimeError(
                        f"C3 v4 parity orientations are incomplete: {witness_id}"
                    )
                canonical_row, canonical, canonical_certificate = orientations[
                    "canonical"
                ]
                mirrored_row, mirrored, mirrored_certificate = orientations[
                    "mirrored"
                ]
                if canonical_certificate.witness_id != mirrored_certificate.witness_id:
                    raise PermissionError("C3 v4 parity pair crossed witness identities")
                same_underlying = (
                    canonical.parse_error is None
                    and mirrored.parse_error is None
                    and canonical.logic_status == mirrored.logic_status
                    and canonical.canonical_valid_trace
                    == mirrored.canonical_valid_trace
                    and canonical.reconstructed_effect
                    == mirrored.reconstructed_effect
                    and canonical.eliminated_options == mirrored.eliminated_options
                    and canonical.supported_options == mirrored.supported_options
                )
                valid_flip = canonical.logic_status != "VALID" or (
                    canonical.presented_valid_trace is not None
                    and mirrored.presented_valid_trace is not None
                    and canonical.presented_valid_trace
                    == 3 - mirrored.presented_valid_trace
                )
                position_invariant_pairs += int(same_underlying and valid_flip)
                orientation_matches = []
                for member in (canonical, mirrored):
                    matches = member.parse_error is None and sealed_triple_matches(
                        canonical_certificate.sealed_valid_trace,
                        canonical_certificate.sealed_effect,
                        canonical_certificate.claimed_eliminated_options,
                        canonical_certificate.claimed_supported_options,
                        member.canonical_valid_trace,
                        member.reconstructed_effect,
                        member.eliminated_options,
                        member.supported_options,
                    )
                    sealed_triple_audits += int(matches)
                    orientation_matches.append(matches)
                paired_sealed_triple_matches += int(all(orientation_matches))
                if canonical_row.get("orientation") != "canonical" or mirrored_row.get(
                    "orientation"
                ) != "mirrored":
                    raise PermissionError("C3 v4 parity rows changed orientation")
            parsed_reconstructions = sum(
                check.parse_error is None for _, check, _ in representatives.values()
            )
            truncated_model_calls = sum(
                bool(row["prompt_was_truncated"])
                for row, _, _ in representatives.values()
            )
            replayed_manifest_values = (
                {
                    "model_calls": len(representatives),
                    "reconstructions": len(representatives),
                    "parsed_reconstructions": parsed_reconstructions,
                    "truncated_model_calls": truncated_model_calls,
                    "isolated_trace_pairs": len(parity_groups),
                    "complete_isolated_trace_pairs": complete_isolated_trace_pairs,
                    "one_valid_one_invalid_pairs": one_valid_one_invalid_pairs,
                    "isolated_sealed_triple_matches": isolated_sealed_triple_matches,
                }
                if isolated_trace_audit
                else {
                    "model_calls": len(representatives),
                    "reconstructions": len(representatives),
                    "parsed_reconstructions": parsed_reconstructions,
                    "truncated_model_calls": truncated_model_calls,
                    "parity_pairs": len(parity_groups),
                    "position_invariant_pairs": position_invariant_pairs,
                    "sealed_triple_audits": sealed_triple_audits,
                    "paired_sealed_triple_matches": paired_sealed_triple_matches,
                }
            )
            for key, value in replayed_manifest_values.items():
                if manifest.get(key) != value:
                    raise PermissionError(
                        f"C3 v4 checker manifest field drifted ({key}): {checker}"
                    )
            replayed_rates = (
                {
                    "complete_isolated_trace_pair_rate": complete_isolated_trace_pairs
                    / max(1, len(parity_groups)),
                    "one_valid_one_invalid_pair_rate": one_valid_one_invalid_pairs
                    / max(1, len(parity_groups)),
                    "isolated_sealed_triple_match_rate": isolated_sealed_triple_matches
                    / max(1, len(parity_groups)),
                }
                if isolated_trace_audit
                else {
                    "position_invariant_pair_rate": position_invariant_pairs
                    / max(1, len(parity_groups)),
                    "sealed_triple_audit_rate": sealed_triple_audits
                    / max(1, 2 * len(parity_groups)),
                    "paired_sealed_triple_match_rate": paired_sealed_triple_matches
                    / max(1, len(parity_groups)),
                }
            )
            for key, value in replayed_rates.items():
                if not np.isclose(float(manifest.get(key, -1.0)), value):
                    raise PermissionError(
                        f"C3 v4 checker manifest rate drifted ({key}): {checker}"
                    )
            quality_gate = effective_check_config.get("smoke_acceptance", {})
            reconstruction_parse_rate = parsed_reconstructions / max(
                1, len(representatives)
            )
            if reconstruction_parse_rate < float(
                quality_gate.get("minimum_reconstruction_parse_rate", 0.0)
            ):
                raise RuntimeError(
                    f"C3 v4 parity-audit parse gate failed: {checker}"
                )
            metric_gates = (
                (
                    (
                        "complete_isolated_trace_pair_rate",
                        "minimum_complete_isolated_trace_pair_rate",
                    ),
                    (
                        "one_valid_one_invalid_pair_rate",
                        "minimum_one_valid_one_invalid_pair_rate",
                    ),
                    (
                        "isolated_sealed_triple_match_rate",
                        "minimum_isolated_sealed_triple_match_rate",
                    ),
                )
                if isolated_trace_audit
                else (
                    (
                        "position_invariant_pair_rate",
                        "minimum_position_invariant_pair_rate",
                    ),
                    ("sealed_triple_audit_rate", "minimum_sealed_triple_audit_rate"),
                    (
                        "paired_sealed_triple_match_rate",
                        "minimum_paired_sealed_triple_match_rate",
                    ),
                )
            )
            for metric, gate in metric_gates:
                if replayed_rates[metric] < float(quality_gate.get(gate, 0.0)):
                    raise RuntimeError(
                        f"C3 v4 checker quality gate failed ({metric}): {checker}"
                    )
            if truncated_model_calls > int(
                quality_gate.get("maximum_prompt_truncations", len(representatives))
            ):
                raise RuntimeError(f"C3 v4 check truncation gate failed: {checker}")
        check_quality_row: dict[str, Any] = {
            "checker": checker,
            "checks": len(rows),
            "parsed": parsed,
            "parse_rate": parsed / max(1, len(rows)),
            "truncated": truncated,
            "by_dataset": {
                dataset: {
                    "checks": int(counts["total"]),
                    "parsed": int(counts["parsed"]),
                    "parse_rate": counts["parsed"] / max(1, counts["total"]),
                }
                for dataset, counts in sorted(by_dataset.items())
            },
        }
        if check_counterfactual_pair:
            common_quality = {
                "reconstructions": len(representatives),
                "parsed_reconstructions": parsed_reconstructions,
                "reconstruction_parse_rate": parsed_reconstructions
                / max(1, len(representatives)),
            }
            if isolated_trace_audit:
                common_quality.update(
                    {
                        "isolated_trace_pairs": len(parity_groups),
                        "complete_isolated_trace_pairs": complete_isolated_trace_pairs,
                        "complete_isolated_trace_pair_rate": replayed_rates[
                            "complete_isolated_trace_pair_rate"
                        ],
                        "one_valid_one_invalid_pairs": one_valid_one_invalid_pairs,
                        "one_valid_one_invalid_pair_rate": replayed_rates[
                            "one_valid_one_invalid_pair_rate"
                        ],
                        "isolated_sealed_triple_matches": isolated_sealed_triple_matches,
                        "isolated_sealed_triple_match_rate": replayed_rates[
                            "isolated_sealed_triple_match_rate"
                        ],
                    }
                )
            else:
                common_quality.update(
                    {
                        "position_invariant_pairs": position_invariant_pairs,
                        "position_invariant_pair_rate": replayed_rates[
                            "position_invariant_pair_rate"
                        ],
                        "sealed_triple_audits": sealed_triple_audits,
                        "sealed_triple_audit_rate": replayed_rates[
                            "sealed_triple_audit_rate"
                        ],
                        "paired_sealed_triple_matches": paired_sealed_triple_matches,
                        "paired_sealed_triple_match_rate": replayed_rates[
                            "paired_sealed_triple_match_rate"
                        ],
                    }
                )
            check_quality_row.update(common_quality)
        check_quality.append(check_quality_row)
    expected_check_pairs = {
        (certificate.certificate_id, checker, orientation)
        for certificate in certificates
        if certificate.parse_error is None
        and (
            not check_counterfactual_pair
            or certificate.sealed_valid_trace is not None
        )
        for checker in (
            str(value) for value in effective_check_config["checker_models"]
        )
        if checker != certificate.generator_id
        for orientation in check_audit_views
    }
    actual_check_pairs = {
        (row.certificate_id, row.checker_id, row.orientation) for row in checks
    }
    if actual_check_pairs != expected_check_pairs or len(checks) != len(expected_check_pairs):
        raise RuntimeError("C3 combined check grid is incomplete")
    equal_call_predictions, equal_call_budgets, equal_call_quality = (
        _load_and_authenticate_equal_call_baselines(
            config_path,
            config,
            equal_call_config_path,
            run_root,
            question_by_id,
            question_path,
        )
    )
    pre_pair_required = bool(
        config.get("acceptance", {}).get(
            "require_pre_pair_style_pointwise_pairwise_baseline", False
        )
    )
    if pre_pair_config_path is None:
        if pre_pair_required:
            raise PermissionError(
                "PRePair-style pointwise-pairwise baseline is required but no sidecar config was supplied"
            )
        pre_pair_predictions: dict[str, dict[str, str | None]] = {}
        pre_pair_budgets: dict[str, float] = {}
        pre_pair_quality: dict[str, Any] = {}
    else:
        (
            pre_pair_predictions,
            pre_pair_budgets,
            pre_pair_quality,
        ) = authenticate_completed_pre_pair(
            config_path,
            config,
            pre_pair_config_path,
            run_root,
        )
    cfmad_required = bool(
        config.get("acceptance", {}).get("require_full_ablation_and_cost_report", False)
    )
    if cfmad_config_path is None:
        if cfmad_required:
            raise PermissionError(
                "CFMAD-style staged prior-art control is required but no sidecar config was supplied"
            )
        cfmad_predictions: dict[str, dict[str, str | None]] = {}
        cfmad_budgets: dict[str, float] = {}
        cfmad_quality: dict[str, Any] = {}
    else:
        cfmad_predictions, cfmad_budgets, cfmad_quality = (
            authenticate_completed_cfmad(
                config_path,
                config,
                cfmad_config_path,
                run_root,
            )
        )
    return C3DevelopmentData(
        questions=questions,
        base_predictions=base_predictions,
        certificates=tuple(certificates),
        checks=tuple(checks),
        answers=answers,
        dataset_by_question={row.question_id: row.dataset for row in questions},
        environment_by_question={row.question_id: row.environment for row in questions},
        generation_quality={
            "certificate_generators": certificate_quality,
            "certificate_checkers": check_quality,
            "equal_call_single_model": equal_call_quality,
            "prepair_style": pre_pair_quality,
            "cfmad_style": cfmad_quality,
        },
        equal_call_predictions=equal_call_predictions,
        equal_call_call_budgets=equal_call_budgets,
        pre_pair_predictions=pre_pair_predictions,
        pre_pair_call_budgets=pre_pair_budgets,
        cfmad_predictions=cfmad_predictions,
        cfmad_call_budgets=cfmad_budgets,
    )


def c3_variants_from_config(config: Mapping[str, Any]) -> tuple[C3Variant, ...]:
    grid = config["variant_grid"]
    result: list[C3Variant] = []
    index = 0
    for regularization_c in grid["regularization_c"]:
        for intervention_margin in grid["intervention_margin"]:
            result.append(
                C3Variant(
                    name=f"c3_grid_{index:03d}",
                    regularization_c=float(regularization_c),
                    intervention_margin=float(intervention_margin),
                    open_option_set=bool(grid["open_option_set"]),
                    use_certificates=bool(grid["use_certificates"]),
                    use_checks=bool(grid["use_checks"]),
                    use_generator_answer_dependence=bool(
                        grid["use_generator_answer_dependence"]
                    ),
                    use_checker_answer_dependence=bool(
                        grid["use_checker_answer_dependence"]
                    ),
                    use_generator_checker_pair_effects=bool(
                        grid["use_generator_checker_pair_effects"]
                    ),
                    use_sealed_set_agreement=bool(
                        grid.get("use_sealed_set_agreement", True)
                    ),
                    use_counterfactual_parity=bool(
                        grid.get("use_counterfactual_parity", True)
                    ),
                )
            )
            index += 1
    return tuple(result)


def _answers_from_decisions(decisions: Sequence[C3Decision]) -> dict[str, str]:
    return {row.question_id: row.answer for row in decisions}


def _source_confusion_bayes_predictions(
    train_questions: Sequence[FalsificationQuestion],
    train_base: Sequence[BasePrediction],
    train_answers: Mapping[str, str],
    target_questions: Sequence[FalsificationQuestion],
    target_base: Sequence[BasePrediction],
    reference_expert: str,
) -> dict[str, str]:
    train_question_by_id = {row.question_id: row for row in train_questions}
    if set(train_answers) != set(train_question_by_id):
        raise PermissionError(
            "Source confusion Bayes answer scope differs from outer training IDs"
        )
    target_base_by_question: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in target_base:
        target_base_by_question[row.question_id][row.expert_id] = row.answer

    prior_counts: dict[tuple[str, tuple[str, ...]], Counter[str]] = defaultdict(
        Counter
    )
    report_counts: dict[
        tuple[str, tuple[str, ...]],
        dict[str, dict[str, Counter[str]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for question in train_questions:
        answer = train_answers.get(question.question_id)
        if answer not in question.option_labels:
            raise PermissionError(
                "Source confusion Bayes lacks a valid outer-training label"
            )
        for group in (question.dataset, "*"):
            prior_counts[(group, question.option_labels)][str(answer)] += 1
    for row in train_base:
        question = train_question_by_id.get(row.question_id)
        if question is None:
            raise ValueError("Source confusion Bayes received an unknown training row")
        answer = train_answers[question.question_id]
        if row.answer not in question.option_labels:
            continue
        for group in (question.dataset, "*"):
            report_counts[(group, question.option_labels)][row.expert_id][
                answer
            ][str(row.answer)] += 1

    predictions: dict[str, str] = {}
    for question in target_questions:
        dataset_key = (question.dataset, question.option_labels)
        fallback_key = ("*", question.option_labels)
        key = dataset_key if prior_counts.get(dataset_key) else fallback_key
        counts = prior_counts.get(key)
        if not counts:
            raise RuntimeError(
                "Source confusion Bayes has no matching outer-training label space"
            )
        base = target_base_by_question[question.question_id]
        scores: dict[str, float] = {}
        class_count = len(question.option_labels)
        total_questions = sum(counts.values())
        for candidate in question.option_labels:
            score = float(
                np.log((counts[candidate] + 1.0) / (total_questions + class_count))
            )
            for expert, report in base.items():
                if report not in question.option_labels:
                    continue
                row_counts = report_counts[key][expert][candidate]
                score += float(
                    np.log(
                        (row_counts[str(report)] + 1.0)
                        / (sum(row_counts.values()) + class_count)
                    )
                )
            scores[candidate] = score
        maximum = max(scores.values())
        tied = tuple(
            candidate
            for candidate in question.option_labels
            if abs(scores[candidate] - maximum) <= 1e-12
        )
        reference = base.get(reference_expert)
        predictions[question.question_id] = (
            str(reference)
            if reference in tied
            else sorted(tied)[0]
        )
    return predictions


def select_c3_variant_nested(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    certificates: Sequence[CounterexampleCertificate],
    checks: Sequence[CertificateCheck],
    labels: SourceTrainingLabels,
    answers: Mapping[str, str],
    variants: Sequence[C3Variant],
    seed: int,
) -> tuple[C3Variant, list[dict[str, Any]]]:
    question_ids = tuple(sorted(row.question_id for row in questions))
    if set(answers) != set(question_ids):
        raise PermissionError(
            "Nested C3 answer scope differs from outer training IDs"
        )
    folds = leave_one_environment_out(question_ids, labels.environment_by_question)
    question_by_id = {row.question_id: row for row in questions}
    variant_index = {variant.name: index for index, variant in enumerate(variants)}
    variants_by_c: dict[float, list[C3Variant]] = defaultdict(list)
    for variant in variants:
        variants_by_c[variant.regularization_c].append(variant)
    correct_by_environment: dict[str, dict[str, int]] = {
        variant.name: {} for variant in variants
    }
    count_by_environment: dict[str, int] = {}
    for fold_index, (environment, train_ids, validation_ids) in enumerate(folds):
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        validation_questions = [question_by_id[question_id] for question_id in validation_ids]
        train_base = _subset_rows(base_predictions, train_ids)
        validation_base = _subset_rows(base_predictions, validation_ids)
        train_certificates = _subset_rows(certificates, train_ids)
        validation_certificates = _subset_rows(certificates, validation_ids)
        train_checks = _subset_rows(checks, train_ids)
        validation_checks = _subset_rows(checks, validation_ids)
        train_labels = labels.subset(train_ids)
        count_by_environment[environment] = len(validation_ids)
        for grouped_variants in variants_by_c.values():
            fitted = CrossExaminedCertificateCourt(
                grouped_variants[0], seed=seed + fold_index
            ).fit(
                train_questions,
                train_base,
                train_certificates,
                train_checks,
                train_labels,
            )
            for variant in grouped_variants:
                predictor = fitted.with_variant(variant)
                predicted = _answers_from_decisions(
                    predictor.predict(
                        validation_questions,
                        validation_base,
                        validation_certificates,
                        validation_checks,
                    )
                )
                correct_by_environment[variant.name][environment] = sum(
                    predicted[question_id] == answers[question_id]
                    for question_id in validation_ids
                )
    rows: list[dict[str, Any]] = []
    for variant in variants:
        environment_accuracies = [
            correct_by_environment[variant.name][environment]
            / count_by_environment[environment]
            for environment in sorted(count_by_environment)
        ]
        total_correct = sum(correct_by_environment[variant.name].values())
        total_count = sum(count_by_environment.values())
        rows.append(
            {
                **asdict(variant),
                "inner_environment_count": len(environment_accuracies),
                "macro_environment_accuracy": float(np.mean(environment_accuracies)),
                "micro_accuracy": total_correct / max(1, total_count),
                "correct": total_correct,
                "samples": total_count,
                "variant_order": variant_index[variant.name],
                "selected": False,
            }
        )
    selected_row = sorted(
        rows,
        key=lambda row: (
            -float(row["macro_environment_accuracy"]),
            -float(row["micro_accuracy"]),
            int(row["variant_order"]),
        ),
    )[0]
    selected_row["selected"] = True
    return variants[variant_index[str(selected_row["name"])]], rows


def static_calibration_variants_from_config(
    config: Mapping[str, Any],
) -> tuple[StaticCalibrationVariant, ...]:
    grid = config["near_prior_baseline_grid"]
    variants: list[StaticCalibrationVariant] = []
    for index, (prior_strength, margin) in enumerate(
        (
            (prior_strength, margin)
            for prior_strength in grid["static_base_prior_strength"]
            for margin in grid["static_intervention_margin"]
        )
    ):
        variants.append(
            StaticCalibrationVariant(
                name=f"static_checker_grid_{index:03d}",
                base_prior_strength=float(prior_strength),
                intervention_margin=float(margin),
            )
        )
    return tuple(variants)


def minority_veto_variants_from_config(
    config: Mapping[str, Any],
) -> tuple[MinorityVetoVariant, ...]:
    return tuple(
        MinorityVetoVariant(
            name=f"minority_veto_grid_{index:03d}",
            veto_threshold=int(threshold),
        )
        for index, threshold in enumerate(
            config["near_prior_baseline_grid"]["minority_veto_threshold"]
        )
    )


def minority_sentinel_style_variants_from_config(
    config: Mapping[str, Any],
) -> tuple[MinoritySentinelStyleVariant, ...]:
    grid = config["near_prior_baseline_grid"]
    thresholds = grid.get(
        "minority_sentinel_flip_threshold",
        [0.50, 0.70, 0.80, 0.90, 0.95, 1.00],
    )
    return tuple(
        MinoritySentinelStyleVariant(
            name=f"minority_sentinel_style_grid_{index:03d}",
            flip_threshold=float(threshold),
            n_estimators=int(grid.get("minority_sentinel_n_estimators", 50)),
            learning_rate=float(grid.get("minority_sentinel_learning_rate", 0.05)),
            max_depth=int(grid.get("minority_sentinel_max_depth", 2)),
        )
        for index, threshold in enumerate(thresholds)
    )


def _baseline_selection_rows(
    variants: Sequence[Any],
    correct_by_environment: Mapping[str, Mapping[str, int]],
    count_by_environment: Mapping[str, int],
    selector: str,
) -> tuple[Any, list[dict[str, Any]]]:
    variant_index = {variant.name: index for index, variant in enumerate(variants)}
    rows: list[dict[str, Any]] = []
    for variant in variants:
        environment_accuracies = [
            correct_by_environment[variant.name][environment]
            / count_by_environment[environment]
            for environment in sorted(count_by_environment)
        ]
        total_correct = sum(correct_by_environment[variant.name].values())
        total_count = sum(count_by_environment.values())
        rows.append(
            {
                "selector": selector,
                **asdict(variant),
                "inner_environment_count": len(environment_accuracies),
                "macro_environment_accuracy": float(np.mean(environment_accuracies)),
                "micro_accuracy": total_correct / max(1, total_count),
                "correct": total_correct,
                "samples": total_count,
                "variant_order": variant_index[variant.name],
                "selected": False,
            }
        )
    selected_row = sorted(
        rows,
        key=lambda row: (
            -float(row["macro_environment_accuracy"]),
            -float(row["micro_accuracy"]),
            int(row["variant_order"]),
        ),
    )[0]
    selected_row["selected"] = True
    return variants[variant_index[str(selected_row["name"])]], rows


def select_static_calibration_nested(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    checks: Sequence[CertificateCheck],
    labels: SourceTrainingLabels,
    variants: Sequence[StaticCalibrationVariant],
) -> tuple[StaticCalibrationVariant, list[dict[str, Any]]]:
    question_ids = tuple(sorted(row.question_id for row in questions))
    question_by_id = {row.question_id: row for row in questions}
    folds = leave_one_environment_out(question_ids, labels.environment_by_question)
    correct_by_environment: dict[str, dict[str, int]] = {
        variant.name: {} for variant in variants
    }
    count_by_environment: dict[str, int] = {}
    for environment, train_ids, validation_ids in folds:
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        validation_questions = [question_by_id[question_id] for question_id in validation_ids]
        train_base = _subset_rows(base_predictions, train_ids)
        validation_base = _subset_rows(base_predictions, validation_ids)
        train_checks = _subset_rows(checks, train_ids)
        validation_checks = _subset_rows(checks, validation_ids)
        train_labels = labels.subset(train_ids)
        count_by_environment[environment] = len(validation_ids)
        for variant in variants:
            predicted = StaticCheckerCalibrationCourt(variant).fit(
                train_questions, train_base, train_checks, train_labels
            ).predict(validation_questions, validation_base, validation_checks)
            correct_by_environment[variant.name][environment] = sum(
                bool(labels.get(question_id, candidate_label_key(predicted[question_id])))
                for question_id in validation_ids
            )
    selected, rows = _baseline_selection_rows(
        variants,
        correct_by_environment,
        count_by_environment,
        "beyond_consensus_style_static_checker_calibration",
    )
    return selected, rows


def select_minority_veto_nested(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    checks: Sequence[CertificateCheck],
    labels: SourceTrainingLabels,
    variants: Sequence[MinorityVetoVariant],
) -> tuple[MinorityVetoVariant, list[dict[str, Any]]]:
    question_ids = tuple(sorted(row.question_id for row in questions))
    question_by_id = {row.question_id: row for row in questions}
    folds = leave_one_environment_out(question_ids, labels.environment_by_question)
    correct_by_environment: dict[str, dict[str, int]] = {
        variant.name: {} for variant in variants
    }
    count_by_environment: dict[str, int] = {}
    for environment, train_ids, validation_ids in folds:
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        validation_questions = [question_by_id[question_id] for question_id in validation_ids]
        train_base = _subset_rows(base_predictions, train_ids)
        validation_base = _subset_rows(base_predictions, validation_ids)
        validation_checks = _subset_rows(checks, validation_ids)
        train_labels = labels.subset(train_ids)
        count_by_environment[environment] = len(validation_ids)
        for variant in variants:
            predicted = MinorityVetoCourt(variant).fit(
                train_questions, train_base, train_labels
            ).predict(validation_questions, validation_base, validation_checks)
            correct_by_environment[variant.name][environment] = sum(
                bool(labels.get(question_id, candidate_label_key(predicted[question_id])))
                for question_id in validation_ids
            )
    selected, rows = _baseline_selection_rows(
        variants,
        correct_by_environment,
        count_by_environment,
        "beyond_consensus_minority_veto",
    )
    return selected, rows


def select_minority_sentinel_style_nested(
    questions: Sequence[FalsificationQuestion],
    base_predictions: Sequence[BasePrediction],
    checks: Sequence[CertificateCheck],
    labels: SourceTrainingLabels,
    variants: Sequence[MinoritySentinelStyleVariant],
    seed: int,
    minimum_majority_preservation: float = 0.95,
) -> tuple[MinoritySentinelStyleVariant, list[dict[str, Any]]]:
    question_ids = tuple(sorted(row.question_id for row in questions))
    question_by_id = {row.question_id: row for row in questions}
    folds = leave_one_environment_out(question_ids, labels.environment_by_question)
    correct_by_environment: dict[str, dict[str, int]] = {
        variant.name: {} for variant in variants
    }
    count_by_environment: dict[str, int] = {}
    majority_correct_cases = Counter()
    preserved_majority_correct_cases = Counter()
    for fold_index, (environment, train_ids, validation_ids) in enumerate(folds):
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        validation_questions = [
            question_by_id[question_id] for question_id in validation_ids
        ]
        train_base = _subset_rows(base_predictions, train_ids)
        validation_base = _subset_rows(base_predictions, validation_ids)
        train_checks = _subset_rows(checks, train_ids)
        validation_checks = _subset_rows(checks, validation_ids)
        train_labels = labels.subset(train_ids)
        validation_base_by_question: dict[str, dict[str, str | None]] = defaultdict(dict)
        for row in validation_base:
            validation_base_by_question[row.question_id][row.expert_id] = row.answer
        count_by_environment[environment] = len(validation_ids)
        for variant in variants:
            model = MinoritySentinelStyleCourt(
                variant, seed=seed + fold_index
            ).fit(train_questions, train_base, train_checks, train_labels)
            predicted = model.predict(
                validation_questions, validation_base, validation_checks
            )
            correct_by_environment[variant.name][environment] = sum(
                bool(
                    labels.get(
                        question_id,
                        candidate_label_key(predicted[question_id]),
                    )
                )
                for question_id in validation_ids
            )
            for question in validation_questions:
                majority, runner = model._ranked_answers(
                    question,
                    validation_base_by_question[question.question_id],
                    model.expert_accuracy_,
                )
                if runner is None or labels.get(
                    question.question_id, candidate_label_key(majority)
                ) is not True:
                    continue
                majority_correct_cases[variant.name] += 1
                preserved_majority_correct_cases[variant.name] += int(
                    predicted[question.question_id] == majority
                )
    _, rows = _baseline_selection_rows(
        variants,
        correct_by_environment,
        count_by_environment,
        "minority_sentinel_style_gradient_boosting",
    )
    for row in rows:
        name = str(row["name"])
        total = majority_correct_cases[name]
        preservation = (
            preserved_majority_correct_cases[name] / total if total else 1.0
        )
        row["majority_correct_divergent_cases"] = total
        row["preserved_majority_correct_cases"] = preserved_majority_correct_cases[
            name
        ]
        row["majority_correct_preservation_rate"] = preservation
        row["safety_eligible"] = preservation >= minimum_majority_preservation
        row["selected"] = False
    selected_row = sorted(
        rows,
        key=lambda row: (
            not bool(row["safety_eligible"]),
            -float(row["macro_environment_accuracy"]),
            -float(row["micro_accuracy"]),
            int(row["variant_order"]),
        ),
    )[0]
    if not bool(selected_row["safety_eligible"]):
        raise RuntimeError("Minority-sentinel style grid lacks a safe fallback")
    selected_row["selected"] = True
    by_name = {variant.name: variant for variant in variants}
    return by_name[str(selected_row["name"])], rows


def _fit_predict(
    variant: C3Variant,
    train_questions: Sequence[FalsificationQuestion],
    train_base: Sequence[BasePrediction],
    train_certificates: Sequence[CounterexampleCertificate],
    train_checks: Sequence[CertificateCheck],
    train_labels: SourceTrainingLabels,
    target_questions: Sequence[FalsificationQuestion],
    target_base: Sequence[BasePrediction],
    target_certificates: Sequence[CounterexampleCertificate],
    target_checks: Sequence[CertificateCheck],
    seed: int,
) -> tuple[CrossExaminedCertificateCourt, list[C3Decision]]:
    model = CrossExaminedCertificateCourt(variant, seed=seed).fit(
        train_questions, train_base, train_certificates, train_checks, train_labels
    )
    return model, model.predict(
        target_questions, target_base, target_certificates, target_checks
    )


def _certificate_scores(
    question: FalsificationQuestion,
    certificates: Sequence[CounterexampleCertificate],
    checks: Sequence[CertificateCheck],
    reference: str,
    include_checks: bool,
) -> str:
    scores = {candidate: 0.0 for candidate in question.option_labels}
    certificate_by_id = {
        row.certificate_id: row
        for row in certificates
        if row.question_id == question.question_id and row.parse_error is None
    }
    certificate_value = {"FALSIFIED": -1.0, "INCONCLUSIVE": 0.0, "SURVIVES": 1.0}
    for certificate in certificate_by_id.values():
        scores[certificate.candidate] += (
            certificate_value[certificate.verdict] * certificate.confidence / 100.0
        )
        if certificate.alternative in scores:
            scores[str(certificate.alternative)] += 0.10
    if include_checks:
        check_value = {
            "VALID_REFUTATION": -1.0,
            "INVALID_REFUTATION": 0.0,
            "VALID_SUPPORT": 1.0,
            "INVALID_SUPPORT": 0.0,
            "VALID_IRRELEVANT": 0.0,
            "INCONCLUSIVE": 0.0,
        }
        for check in checks:
            if (
                check.question_id != question.question_id
                or check.certificate_id not in certificate_by_id
                or check.parse_error is not None
            ):
                continue
            scores[check.candidate] += check_value[check.status] * check.confidence / 100.0
            if check.independent_answer in scores:
                scores[str(check.independent_answer)] += 0.10
    return sorted(
        scores,
        key=lambda candidate: (-scores[candidate], candidate != reference, candidate),
    )[0]


def _sealed_exact_agreement_vote(
    question: FalsificationQuestion,
    certificates: Sequence[CounterexampleCertificate],
    checks: Sequence[CertificateCheck],
    reference: str,
) -> str:
    scores = {candidate: 0.0 for candidate in question.option_labels}
    checks_by_certificate: dict[str, list[CertificateCheck]] = defaultdict(list)
    for check in checks:
        if check.question_id == question.question_id:
            checks_by_certificate[check.certificate_id].append(check)
    for certificate in certificates:
        if (
            certificate.question_id != question.question_id
            or certificate.parse_error is not None
            or not certificate.claim_was_sealed
        ):
            continue
        claimed_eliminated = set(certificate.claimed_eliminated_options)
        claimed_supported = set(certificate.claimed_supported_options)
        relation = (
            -1.0
            if certificate.candidate in claimed_eliminated
            else 1.0
            if certificate.candidate in claimed_supported
            else 0.0
        )
        for check in checks_by_certificate.get(certificate.certificate_id, ()):
            exact = (
                check.parse_error is None
                and check.logic_status == "VALID"
                and set(check.eliminated_options) == claimed_eliminated
                and set(check.supported_options) == claimed_supported
            )
            if exact:
                scores[certificate.candidate] += (
                    relation
                    * certificate.confidence
                    * check.confidence
                    / 10000.0
                )
    return sorted(
        scores,
        key=lambda candidate: (-scores[candidate], candidate != reference, candidate),
    )[0]


def _strict_sealed_evidence(
    question: FalsificationQuestion,
    certificates: Sequence[CounterexampleCertificate],
    checks: Sequence[CertificateCheck],
    required_orientations: tuple[str, str],
) -> dict[str, dict[str, set[Any]]]:
    evidence = {
        candidate: {
            "support_pairs": set(),
            "support_generators": set(),
            "support_checkers": set(),
            "eliminate_pairs": set(),
            "eliminate_generators": set(),
            "eliminate_checkers": set(),
        }
        for candidate in question.option_labels
    }
    certificate_by_id = {
        row.certificate_id: row
        for row in certificates
        if row.question_id == question.question_id
    }
    representative_by_witness: dict[str, CounterexampleCertificate] = {}
    for certificate in sorted(
        certificate_by_id.values(), key=lambda row: row.certificate_id
    ):
        if (
            certificate.parse_error is None
            and certificate.claim_was_sealed
            and certificate.counterfactual_pair
            and certificate.witness_id is not None
            and certificate.sealed_valid_trace in (1, 2)
            and certificate.sealed_effect is not None
        ):
            representative_by_witness.setdefault(
                certificate.witness_id, certificate
            )
    checks_by_witness_checker: dict[
        tuple[str, str], dict[str, CertificateCheck]
    ] = defaultdict(dict)
    for check in checks:
        certificate = certificate_by_id.get(check.certificate_id)
        if certificate is None or certificate.witness_id not in representative_by_witness:
            continue
        representative = representative_by_witness[str(certificate.witness_id)]
        if check.certificate_id != representative.certificate_id:
            continue
        key = (str(certificate.witness_id), check.checker_id)
        if check.orientation in checks_by_witness_checker[key]:
            raise ValueError("Duplicate strict-evidence audit orientation")
        checks_by_witness_checker[key][check.orientation] = check

    for (witness_id, checker), by_orientation in checks_by_witness_checker.items():
        if set(by_orientation) != set(required_orientations):
            continue
        certificate = representative_by_witness[witness_id]
        rows = tuple(by_orientation[orientation] for orientation in required_orientations)
        if any(row.parse_error is not None for row in rows):
            continue
        valid_rows = tuple(row for row in rows if row.logic_status == "VALID")
        invalid_rows = tuple(row for row in rows if row.logic_status == "INVALID")
        if required_orientations == ISOLATED_TRACE_VIEWS:
            if len(valid_rows) != 1 or len(invalid_rows) != 1:
                continue
            valid = valid_rows[0]
            invalid = invalid_rows[0]
            exact = (
                valid.canonical_valid_trace == certificate.sealed_valid_trace
                and valid.reconstructed_effect == certificate.sealed_effect
                and set(valid.eliminated_options)
                == set(certificate.claimed_eliminated_options)
                and set(valid.supported_options)
                == set(certificate.claimed_supported_options)
                and not invalid.eliminated_options
                and not invalid.supported_options
                and invalid.reconstructed_effect is None
            )
        else:
            exact = all(
                row.logic_status == "VALID"
                and row.canonical_valid_trace == certificate.sealed_valid_trace
                and row.reconstructed_effect == certificate.sealed_effect
                and set(row.eliminated_options)
                == set(certificate.claimed_eliminated_options)
                and set(row.supported_options)
                == set(certificate.claimed_supported_options)
                for row in rows
            )
        if not exact:
            continue
        signed_options = (
            tuple(certificate.claimed_supported_options)
            if certificate.sealed_effect == "SUPPORTS"
            else tuple(certificate.claimed_eliminated_options)
        )
        if len(signed_options) != 1 or signed_options[0] not in evidence:
            continue
        candidate = signed_options[0]
        prefix = (
            "support" if certificate.sealed_effect == "SUPPORTS" else "eliminate"
        )
        evidence[candidate][f"{prefix}_pairs"].add(
            (certificate.generator_id, checker)
        )
        evidence[candidate][f"{prefix}_generators"].add(certificate.generator_id)
        evidence[candidate][f"{prefix}_checkers"].add(checker)
    return evidence


def _el_dgr_style_conservative_admissibility(
    question: FalsificationQuestion,
    certificates: Sequence[CounterexampleCertificate],
    checks: Sequence[CertificateCheck],
    reference: str,
) -> str:
    evidence = _strict_sealed_evidence(
        question, certificates, checks, ISOLATED_TRACE_VIEWS
    )

    def independently_established(candidate: str, prefix: str) -> bool:
        row = evidence[candidate]
        return (
            len(row[f"{prefix}_generators"]) >= 2
            and len(row[f"{prefix}_checkers"]) >= 2
        )

    if not independently_established(reference, "eliminate"):
        return reference
    admissible = [
        candidate
        for candidate in question.option_labels
        if candidate != reference
        and independently_established(candidate, "support")
        and not evidence[candidate]["eliminate_pairs"]
    ]
    if not admissible:
        return reference
    return sorted(
        admissible,
        key=lambda candidate: (
            -len(evidence[candidate]["support_pairs"]),
            candidate,
        ),
    )[0]


def _agent_auditor_style_localized_divergence_vote(
    question: FalsificationQuestion,
    certificates: Sequence[CounterexampleCertificate],
    pair_visible_checks: Sequence[CertificateCheck],
    reference: str,
) -> str:
    evidence = _strict_sealed_evidence(
        question, certificates, pair_visible_checks, PARITY_ORIENTATIONS
    )
    scores = {
        candidate: (
            len(row["support_pairs"]) - len(row["eliminate_pairs"])
        )
        for candidate, row in evidence.items()
    }
    maximum = max(scores.values())
    tied = tuple(
        candidate
        for candidate in question.option_labels
        if scores[candidate] == maximum
    )
    return reference if reference in tied else sorted(tied)[0]


def _direct_answer_vote(
    question: FalsificationQuestion,
    certificates: Sequence[CounterexampleCertificate],
    checks: Sequence[CertificateCheck],
    reference: str,
) -> str:
    counts = Counter()
    for certificate in certificates:
        if certificate.question_id == question.question_id and certificate.parse_error is None:
            if certificate.alternative in question.option_labels:
                counts[str(certificate.alternative)] += 1
    for check in checks:
        if check.question_id == question.question_id and check.parse_error is None:
            if check.independent_answer in question.option_labels:
                counts[str(check.independent_answer)] += 1
    if not counts:
        return reference
    maximum = max(counts.values())
    tied = sorted(answer for answer, count in counts.items() if count == maximum)
    return reference if reference in tied else tied[0]


def _ablation_variants(selected: C3Variant) -> dict[str, C3Variant]:
    return {
        "c3_certificates_only": replace(
            selected, name="c3_certificates_only", use_checks=False
        ),
        "c3_no_generator_answer_dependence": replace(
            selected,
            name="c3_no_generator_answer_dependence",
            use_generator_answer_dependence=False,
        ),
        "c3_no_checker_answer_dependence": replace(
            selected,
            name="c3_no_checker_answer_dependence",
            use_checker_answer_dependence=False,
        ),
        "c3_no_generator_checker_pair_effects": replace(
            selected,
            name="c3_no_generator_checker_pair_effects",
            use_generator_checker_pair_effects=False,
        ),
        "c3_no_sealed_set_agreement": replace(
            selected,
            name="c3_no_sealed_set_agreement",
            use_sealed_set_agreement=False,
        ),
        "c3_no_counterfactual_parity": replace(
            selected,
            name="c3_no_counterfactual_parity",
            use_counterfactual_parity=False,
        ),
        "c3_closed_option_set": replace(
            selected, name="c3_closed_option_set", open_option_set=False
        ),
        "c3_no_intervention_gate": replace(
            selected, name="c3_no_intervention_gate", intervention_margin=0.0
        ),
    }


def generate_nested_c3_predictions(
    config: Mapping[str, Any], data: C3DevelopmentData
) -> tuple[
    dict[str, dict[str, str | None]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    experts = tuple(str(value) for value in config["experts"])
    generators = tuple(str(value) for value in config["certificate_models"])
    checkers = tuple(str(value) for value in config["checker_models"])
    variants = c3_variants_from_config(config)
    base_only_variants = tuple(
        replace(
            variant,
            name=f"base_outputs_only_grid_{index:03d}",
            use_certificates=False,
            use_checks=False,
        )
        for index, variant in enumerate(variants)
    )
    static_calibration_variants = static_calibration_variants_from_config(config)
    minority_veto_variants = minority_veto_variants_from_config(config)
    minority_sentinel_variants = minority_sentinel_style_variants_from_config(config)
    labels = _source_labels(
        data.questions,
        data.base_predictions,
        data.answers,
        data.environment_by_question,
    )
    question_by_id = {row.question_id: row for row in data.questions}
    base_by_question: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in data.base_predictions:
        base_by_question[row.question_id][row.expert_id] = row.answer
    certificates_by_question: dict[str, list[CounterexampleCertificate]] = defaultdict(list)
    for row in data.certificates:
        certificates_by_question[row.question_id].append(row)
    checks_by_question: dict[str, list[CertificateCheck]] = defaultdict(list)
    for row in data.checks:
        checks_by_question[row.question_id].append(row)

    predictions: dict[str, dict[str, str | None]] = {
        f"single::{expert}": {
            question_id: base_by_question[question_id][expert]
            for question_id in sorted(question_by_id)
        }
        for expert in experts
    }
    core_methods = (
        "best_single_nested_oof",
        "majority_vote",
        "source_weighted_vote",
        "source_confusion_bayes",
        "uncalibrated_certificates",
        "uncalibrated_cross_examined_certificates",
        "direct_anonymous_answer_vote",
        "sealed_exact_agreement_vote",
        "el_dgr_style_conservative_admissibility",
        "beyond_consensus_static_calibration_nested",
        "beyond_consensus_minority_veto_nested",
        "minority_sentinel_style_nested",
        "c3_primary_nested",
        "c3_certificates_only",
        "c3_no_generator_answer_dependence",
        "c3_no_checker_answer_dependence",
        "c3_no_generator_checker_pair_effects",
        "c3_no_sealed_set_agreement",
        "c3_no_counterfactual_parity",
        "c3_closed_option_set",
        "c3_no_intervention_gate",
        "c3_base_outputs_only",
    )
    predictions.update({method: {} for method in core_methods})
    predictions.update({f"c3_single_generator::{model}": {} for model in generators})
    predictions.update({f"c3_single_checker::{model}": {} for model in checkers})
    for method, values in data.equal_call_predictions.items():
        if method in predictions:
            raise ValueError(f"Equal-call method name collides with a C3 method: {method}")
        predictions[method] = dict(values)
    for method, values in data.pre_pair_predictions.items():
        if method in predictions:
            raise ValueError(f"PRePair-style method name collides with a C3 method: {method}")
        predictions[method] = dict(values)
    for method, values in data.cfmad_predictions.items():
        if method in predictions:
            raise ValueError(f"CFMAD-style method name collides with a C3 method: {method}")
        predictions[method] = dict(values)
    nested_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    primary_diagnostics: dict[str, dict[str, Any]] = {}
    outer_folds = leave_one_environment_out(
        question_by_id, data.environment_by_question
    )
    seed = int(config["seed"])
    for outer_index, (outer_environment, train_ids, heldout_ids) in enumerate(outer_folds):
        train_questions = [question_by_id[question_id] for question_id in train_ids]
        heldout_questions = [question_by_id[question_id] for question_id in heldout_ids]
        train_base = _subset_rows(data.base_predictions, train_ids)
        heldout_base = _subset_rows(data.base_predictions, heldout_ids)
        train_certificates = _subset_rows(data.certificates, train_ids)
        heldout_certificates = _subset_rows(data.certificates, heldout_ids)
        train_checks = _subset_rows(data.checks, train_ids)
        heldout_checks = _subset_rows(data.checks, heldout_ids)
        train_labels = labels.subset(train_ids)
        train_answers = {
            question_id: data.answers[question_id] for question_id in train_ids
        }
        selected, search_rows = select_c3_variant_nested(
            train_questions,
            train_base,
            train_certificates,
            train_checks,
            train_labels,
            train_answers,
            variants,
            seed + outer_index * 1000,
        )
        for row in search_rows:
            nested_rows.append(
                {"outer_environment": outer_environment, "selector": "c3", **row}
            )
        selected_base_only, base_only_search_rows = select_c3_variant_nested(
            train_questions,
            train_base,
            (),
            (),
            train_labels,
            train_answers,
            base_only_variants,
            seed + outer_index * 1000,
        )
        for row in base_only_search_rows:
            nested_rows.append(
                {
                    "outer_environment": outer_environment,
                    "selector": "base_outputs_only",
                    **row,
                }
            )
        selected_static, static_search_rows = select_static_calibration_nested(
            train_questions,
            train_base,
            train_checks,
            train_labels,
            static_calibration_variants,
        )
        selected_veto, veto_search_rows = select_minority_veto_nested(
            train_questions,
            train_base,
            train_checks,
            train_labels,
            minority_veto_variants,
        )
        selected_sentinel, sentinel_search_rows = (
            select_minority_sentinel_style_nested(
                train_questions,
                train_base,
                train_checks,
                train_labels,
                minority_sentinel_variants,
                seed + outer_index * 1000,
            )
        )
        for row in static_search_rows + veto_search_rows + sentinel_search_rows:
            nested_rows.append({"outer_environment": outer_environment, **row})
        expert_accuracy = _expert_accuracies(experts, train_ids, train_labels)
        reference_expert = sorted(
            experts, key=lambda expert: (-expert_accuracy[expert], expert)
        )[0]
        confusion_bayes_predictions = _source_confusion_bayes_predictions(
            train_questions,
            train_base,
            train_answers,
            heldout_questions,
            heldout_base,
            reference_expert,
        )
        primary_model, primary_decisions = _fit_predict(
            selected,
            train_questions,
            train_base,
            train_certificates,
            train_checks,
            train_labels,
            heldout_questions,
            heldout_base,
            heldout_certificates,
            heldout_checks,
            seed + outer_index,
        )
        if primary_model.reference_expert_ != reference_expert:
            raise AssertionError("C3 and best-single source references differ")
        primary_by_id = {row.question_id: row for row in primary_decisions}
        _, base_only_decisions = _fit_predict(
            selected_base_only,
            train_questions,
            train_base,
            (),
            (),
            train_labels,
            heldout_questions,
            heldout_base,
            (),
            (),
            seed + outer_index,
        )
        predictions["c3_base_outputs_only"].update(
            _answers_from_decisions(base_only_decisions)
        )
        static_predictions = StaticCheckerCalibrationCourt(selected_static).fit(
            train_questions, train_base, train_checks, train_labels
        ).predict(heldout_questions, heldout_base, heldout_checks)
        veto_predictions = MinorityVetoCourt(selected_veto).fit(
            train_questions, train_base, train_labels
        ).predict(heldout_questions, heldout_base, heldout_checks)
        sentinel_predictions = MinoritySentinelStyleCourt(
            selected_sentinel, seed=seed + outer_index
        ).fit(
            train_questions, train_base, train_checks, train_labels
        ).predict(heldout_questions, heldout_base, heldout_checks)
        for question_id in heldout_ids:
            question = question_by_id[question_id]
            base = base_by_question[question_id]
            reference_answer = base.get(reference_expert)
            if reference_answer not in question.option_labels:
                reference_answer = question.option_labels[0]
            reference = str(reference_answer)
            predictions["best_single_nested_oof"][question_id] = reference
            predictions["majority_vote"][question_id] = _majority_answer(
                question, base, reference
            )
            predictions["source_weighted_vote"][question_id] = _weighted_answer(
                question, base, expert_accuracy, reference
            )
            predictions["source_confusion_bayes"][question_id] = (
                confusion_bayes_predictions[question_id]
            )
            question_certificates = certificates_by_question[question_id]
            question_checks = checks_by_question[question_id]
            predictions["uncalibrated_certificates"][question_id] = _certificate_scores(
                question, question_certificates, question_checks, reference, False
            )
            predictions["uncalibrated_cross_examined_certificates"][question_id] = (
                _certificate_scores(
                    question, question_certificates, question_checks, reference, True
                )
            )
            predictions["direct_anonymous_answer_vote"][question_id] = _direct_answer_vote(
                question, question_certificates, question_checks, reference
            )
            predictions["sealed_exact_agreement_vote"][question_id] = (
                _sealed_exact_agreement_vote(
                    question, question_certificates, question_checks, reference
                )
            )
            predictions["el_dgr_style_conservative_admissibility"][question_id] = (
                _el_dgr_style_conservative_admissibility(
                    question, question_certificates, question_checks, reference
                )
            )
            predictions["beyond_consensus_static_calibration_nested"][question_id] = (
                static_predictions[question_id]
            )
            predictions["beyond_consensus_minority_veto_nested"][question_id] = (
                veto_predictions[question_id]
            )
            predictions["minority_sentinel_style_nested"][question_id] = (
                sentinel_predictions[question_id]
            )
            decision = primary_by_id[question_id]
            predictions["c3_primary_nested"][question_id] = decision.answer
            primary_diagnostics[question_id] = {
                **dict(decision.diagnostics),
                "outer_environment": outer_environment,
                "outer_reference_expert": reference_expert,
                "selected_variant": asdict(selected),
                "candidate_logits": dict(decision.candidate_logits),
                "candidate_probabilities": dict(decision.candidate_probabilities),
                "fallback_reason": decision.fallback_reason,
                "open_set_rescue": decision.open_set_rescue,
            }
        for method, variant in _ablation_variants(selected).items():
            _, decisions = _fit_predict(
                variant,
                train_questions,
                train_base,
                train_certificates,
                train_checks,
                train_labels,
                heldout_questions,
                heldout_base,
                heldout_certificates,
                heldout_checks,
                seed + outer_index,
            )
            predictions[method].update(_answers_from_decisions(decisions))
        for generator in generators:
            train_subset_certificates = [
                row for row in train_certificates if row.generator_id == generator
            ]
            heldout_subset_certificates = [
                row for row in heldout_certificates if row.generator_id == generator
            ]
            certificate_ids = {
                row.certificate_id
                for row in train_subset_certificates + heldout_subset_certificates
            }
            train_subset_checks = [
                row for row in train_checks if row.certificate_id in certificate_ids
            ]
            heldout_subset_checks = [
                row for row in heldout_checks if row.certificate_id in certificate_ids
            ]
            variant = replace(selected, name=f"c3_single_generator::{generator}")
            _, decisions = _fit_predict(
                variant,
                train_questions,
                train_base,
                train_subset_certificates,
                train_subset_checks,
                train_labels,
                heldout_questions,
                heldout_base,
                heldout_subset_certificates,
                heldout_subset_checks,
                seed + outer_index,
            )
            predictions[f"c3_single_generator::{generator}"].update(
                _answers_from_decisions(decisions)
            )
        for checker in checkers:
            train_subset_checks = [row for row in train_checks if row.checker_id == checker]
            heldout_subset_checks = [
                row for row in heldout_checks if row.checker_id == checker
            ]
            variant = replace(selected, name=f"c3_single_checker::{checker}")
            _, decisions = _fit_predict(
                variant,
                train_questions,
                train_base,
                train_certificates,
                train_subset_checks,
                train_labels,
                heldout_questions,
                heldout_base,
                heldout_certificates,
                heldout_subset_checks,
                seed + outer_index,
            )
            predictions[f"c3_single_checker::{checker}"].update(
                _answers_from_decisions(decisions)
            )
        outer_rows.append(
            {
                "outer_fold": outer_index,
                "outer_environment": outer_environment,
                "train_questions": len(train_ids),
                "heldout_questions": len(heldout_ids),
                "reference_expert": reference_expert,
                "reference_train_accuracy": expert_accuracy[reference_expert],
                "selected_variant": asdict(selected),
                "selected_base_outputs_only_variant": asdict(selected_base_only),
                "selected_static_calibration_variant": asdict(selected_static),
                "selected_minority_veto_variant": asdict(selected_veto),
                "selected_minority_sentinel_style_variant": asdict(
                    selected_sentinel
                ),
            }
        )
    expected_ids = set(question_by_id)
    for method, values in predictions.items():
        if set(values) != expected_ids:
            raise RuntimeError(f"Nested C3 method has incomplete predictions: {method}")
    individual_accuracy = {
        expert: sum(
            predictions[f"single::{expert}"][question_id] == data.answers[question_id]
            for question_id in expected_ids
        )
        / len(expected_ids)
        for expert in experts
    }
    descriptive_best = sorted(
        experts, key=lambda expert: (-individual_accuracy[expert], expert)
    )[0]
    predictions["full_development_best_single_descriptive"] = dict(
        predictions[f"single::{descriptive_best}"]
    )
    for row in outer_rows:
        row["full_development_best_single_descriptive"] = descriptive_best
        row["full_development_best_single_accuracy"] = individual_accuracy[descriptive_best]
    return predictions, nested_rows, outer_rows, primary_diagnostics


def _call_budget(method: str, config: Mapping[str, Any], data: C3DevelopmentData) -> float:
    expert_count = len(config["experts"])
    generator_count = len(config["certificate_models"])
    mean_options = float(np.mean([len(question.option_labels) for question in data.questions]))
    if method.startswith("equal_call::"):
        if method not in data.equal_call_call_budgets:
            raise RuntimeError(f"Missing authenticated call budget for {method}")
        return float(data.equal_call_call_budgets[method])
    if method in data.pre_pair_call_budgets:
        return float(data.pre_pair_call_budgets[method])
    if method in data.cfmad_call_budgets:
        return float(data.cfmad_call_budgets[method])
    if method.startswith("single::") or method in {
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
    }:
        return 1.0
    if method in {
        "majority_vote",
        "source_weighted_vote",
        "source_confusion_bayes",
        "c3_base_outputs_only",
    }:
        return float(expert_count)
    certificate_prompt_version = str(
        config["certificate_generation"]["prompt_version"]
    )
    set_valued = certificate_prompt_version in {
        "sealed_effect_witness_v3",
        "sealed_counterfactual_parity_v4",
        "hardened_sealed_counterfactual_parity_v5",
        "committed_counterfactual_permutation_v6",
    }
    parity_multiplier = (
        2
        if str(config["check_generation"]["prompt_version"])
        in {
            "blind_counterfactual_parity_v4",
            "hardened_blind_counterfactual_parity_v5",
            "blind_isolated_trace_audit_v7",
            "commitment_conditioned_proof_audit_v8",
        }
        else 1
    )
    option_multiplier = 1.0 if set_valued else mean_options
    certificate_calls = generator_count * option_multiplier
    full_check_calls = sum(
        checker != generator
        for generator in config["certificate_models"]
        for checker in config["checker_models"]
    ) * option_multiplier * parity_multiplier
    if method in {"uncalibrated_certificates", "c3_certificates_only"}:
        return float(expert_count) + certificate_calls
    if method.startswith("c3_single_generator::"):
        generator = method.split("::", 1)[1]
        return (
            float(expert_count)
            + option_multiplier
            + sum(checker != generator for checker in config["checker_models"])
            * option_multiplier
            * parity_multiplier
        )
    if method.startswith("c3_single_checker::"):
        checker = method.split("::", 1)[1]
        return (
            float(expert_count)
            + certificate_calls
            + sum(generator != checker for generator in config["certificate_models"])
            * option_multiplier
            * parity_multiplier
        )
    return float(expert_count) + certificate_calls + full_check_calls


def _prompt_mechanism_ablation_gate(
    primary_accuracy: float,
    method_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], bool, bool]:
    accuracies = {
        str(row["method"]).split("::", 1)[1]: float(row["accuracy"])
        for row in method_rows
        if str(row["method"]).startswith("c3_prompt_ablation::")
    }
    present = set(accuracies) == set(REQUIRED_PROMPT_MECHANISM_ABLATIONS)
    return (
        accuracies,
        present,
        present and all(primary_accuracy > value for value in accuracies.values()),
    )


def _required_development_comparison_coverage(
    predictions: Mapping[str, Mapping[str, str | None]],
    experts: Sequence[str],
    generators: Sequence[str],
    checkers: Sequence[str],
) -> tuple[list[dict[str, Any]], bool]:
    available = set(predictions)
    equal_call_models = tuple(sorted(set(generators).union(checkers)))
    requirements: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "every_fixed_single_expert",
            "exact",
            tuple(f"single::{expert}" for expert in experts),
        ),
        *_STATIC_REQUIRED_COMPARISONS,
        (
            "one_generator_variants",
            "exact_feature_subset",
            tuple(f"c3_single_generator::{model}" for model in generators),
        ),
        (
            "one_checker_variants",
            "exact_feature_subset",
            tuple(f"c3_single_checker::{model}" for model in checkers),
        ),
        (
            "equal_call_single_model_self_consistency",
            "exact_equal_call_control",
            tuple(
                f"equal_call::self_consistency::{model}"
                for model in equal_call_models
            ),
        ),
        (
            "equal_call_single_model_self_revision",
            "exact_equal_call_control",
            tuple(
                f"equal_call::self_revision::{model}"
                for model in equal_call_models
            ),
        ),
        (
            "private_precommitment_prompt_ablation",
            "exact_equal_call_prompt_ablation",
            ("c3_prompt_ablation::no_checker_private_precommitment",),
        ),
        (
            "sibling_hiding_prompt_ablation",
            "exact_equal_call_prompt_ablation",
            ("c3_prompt_ablation::pair_visible_with_precommitment",),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for requirement, fidelity, methods in requirements:
        missing = tuple(method for method in methods if method not in available)
        rows.append(
            {
                "requirement": requirement,
                "fidelity": fidelity,
                "methods": list(methods),
                "present": not missing,
                "missing_methods": list(missing),
            }
        )
    return rows, all(bool(row["present"]) for row in rows)


def evaluate_c3_predictions(
    config: Mapping[str, Any],
    data: C3DevelopmentData,
    predictions: Mapping[str, Mapping[str, str | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method_rows = [
        _method_summary(method, values, data, _call_budget(method, config, data))
        for method, values in sorted(predictions.items())
    ]
    primary = "c3_primary_nested"
    references = [
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
        "majority_vote",
        "source_weighted_vote",
        "source_confusion_bayes",
        "uncalibrated_certificates",
        "uncalibrated_cross_examined_certificates",
        "direct_anonymous_answer_vote",
        "sealed_exact_agreement_vote",
        "el_dgr_style_conservative_admissibility",
        "agent_auditor_style_localized_divergence",
        "beyond_consensus_static_calibration_nested",
        "beyond_consensus_minority_veto_nested",
        "minority_sentinel_style_nested",
        "c3_certificates_only",
        "c3_no_generator_answer_dependence",
        "c3_no_checker_answer_dependence",
        "c3_no_generator_checker_pair_effects",
        "c3_no_sealed_set_agreement",
        "c3_no_counterfactual_parity",
        "c3_closed_option_set",
        "c3_no_intervention_gate",
        "c3_base_outputs_only",
    ]
    references.extend(
        method
        for method in predictions
        if method.startswith("c3_single_generator::")
        or method.startswith("c3_single_checker::")
        or method.startswith("c3_prompt_ablation::")
        or method.startswith("equal_call::")
        or method.startswith("cfmad_style::")
        or method == CFMAD_STYLE_METHOD
        or method in {PREPAIR_TOP2_METHOD, PREPAIR_BUDGET_MATCHED_METHOD}
    )
    validation = config["outer_validation"]
    comparisons = [
        _comparison(
            primary,
            reference,
            predictions,
            data,
            seed=int(validation["bootstrap_seed"]) + index,
            samples=int(validation["bootstrap_samples"]),
        )
        for index, reference in enumerate(references)
    ]
    by_reference = {str(row["reference"]): row for row in comparisons}
    deployable = by_reference["best_single_nested_oof"]
    descriptive = by_reference["full_development_best_single_descriptive"]
    acceptance = config["acceptance"]
    primary_accuracy = next(
        float(row["accuracy"]) for row in method_rows if row["method"] == primary
    )
    fixed_single_accuracies = [
        float(row["accuracy"])
        for row in method_rows
        if str(row["method"]).startswith("single::")
    ]
    equal_call_accuracies = [
        float(row["accuracy"])
        for row in method_rows
        if str(row["method"]).startswith("equal_call::")
    ]
    pre_pair_accuracies = [
        float(row["accuracy"])
        for row in method_rows
        if str(row["method"])
        in {PREPAIR_TOP2_METHOD, PREPAIR_BUDGET_MATCHED_METHOD}
    ]
    cfmad_accuracies = [
        float(row["accuracy"])
        for row in method_rows
        if str(row["method"]).startswith("cfmad_style::")
        or str(row["method"]) == CFMAD_STYLE_METHOD
    ]
    (
        prompt_ablation_accuracies,
        prompt_ablations_present,
        beats_prompt_ablations,
    ) = _prompt_mechanism_ablation_gate(primary_accuracy, method_rows)
    expected_equal_call_baselines = 2 * len(
        {
            str(value) for value in config["certificate_models"]
        }.union(str(value) for value in config["checker_models"])
    )
    comparison_coverage, comparison_suite_complete = (
        _required_development_comparison_coverage(
            predictions,
            tuple(str(value) for value in config["experts"]),
            tuple(str(value) for value in config["certificate_models"]),
            tuple(str(value) for value in config["checker_models"]),
        )
    )
    checks = {
        "delta_vs_deployable_best_single": float(deployable["delta"])
        >= float(acceptance["minimum_oof_delta_vs_deployable_best_single_pp"]) / 100.0,
        "delta_vs_full_development_best_single": float(descriptive["delta"])
        >= float(acceptance["minimum_oof_delta_vs_full_development_best_single_pp"]) / 100.0,
        "nonnegative_on_every_dataset": all(
            float(value)
            >= float(acceptance["minimum_per_dataset_delta_pp"]) / 100.0
            for value in deployable["per_dataset_delta"].values()
        ),
        "paired_ci_low_above_zero": (
            float(deployable["stratified_paired_bootstrap_delta_ci95"][0]) > 0.0
            if bool(acceptance["require_paired_bootstrap_ci_low_gt_zero"])
            else True
        ),
        "mcnemar_significant": float(deployable["exact_mcnemar_p"])
        < float(acceptance["require_two_sided_mcnemar_p_lt"]),
        "beats_every_fixed_single_model": all(
            primary_accuracy > accuracy for accuracy in fixed_single_accuracies
        ),
        "equal_call_single_model_baselines_present": (
            len(equal_call_accuracies) == expected_equal_call_baselines
            if bool(acceptance["require_equal_call_single_model_baselines"])
            else True
        ),
        "beats_every_equal_call_single_model_baseline": (
            all(primary_accuracy > accuracy for accuracy in equal_call_accuracies)
            if equal_call_accuracies
            else not bool(acceptance["require_equal_call_single_model_baselines"])
        ),
        "all_required_ablations_present": all(
            reference in by_reference
            for reference in (
                "c3_certificates_only",
                "c3_no_generator_answer_dependence",
                "c3_no_checker_answer_dependence",
                "c3_no_generator_checker_pair_effects",
                "c3_no_sealed_set_agreement",
                "c3_no_counterfactual_parity",
                "c3_closed_option_set",
                "c3_no_intervention_gate",
                "c3_base_outputs_only",
            )
        ),
        "prompt_mechanism_ablations_present": prompt_ablations_present,
        "beats_prompt_mechanism_ablations": beats_prompt_ablations,
        "near_prior_baselines_present": all(
            reference in by_reference
            for reference in (
                "beyond_consensus_static_calibration_nested",
                "beyond_consensus_minority_veto_nested",
                "minority_sentinel_style_nested",
            )
        ),
        "pre_pair_style_pointwise_pairwise_baseline_present": (
            PREPAIR_TOP2_METHOD in by_reference
            and PREPAIR_BUDGET_MATCHED_METHOD in by_reference
            if bool(
                acceptance.get(
                    "require_pre_pair_style_pointwise_pairwise_baseline", False
                )
            )
            else True
        ),
        "beats_both_pre_pair_style_baselines": (
            len(pre_pair_accuracies) == 2
            and all(primary_accuracy > accuracy for accuracy in pre_pair_accuracies)
            if bool(
                data.generation_quality.get("prepair_style", {}).get(
                    "require_primary_strictly_beats_both_methods", False
                )
            )
            else True
        ),
        "cfmad_style_staged_controls_present": (
            len(cfmad_accuracies)
            == len(data.generation_quality.get("cfmad_style", {}).get("models", ()))
            + 1
            and CFMAD_STYLE_METHOD in by_reference
            if bool(acceptance.get("require_full_ablation_and_cost_report", False))
            else True
        ),
        "beats_all_cfmad_style_controls": (
            bool(cfmad_accuracies)
            and all(primary_accuracy > accuracy for accuracy in cfmad_accuracies)
            if bool(
                data.generation_quality.get("cfmad_style", {}).get(
                    "require_primary_strictly_beats_all_cfmad_style_methods", False
                )
            )
            else True
        ),
        "full_required_development_comparison_suite_present": (
            comparison_suite_complete
            if bool(acceptance.get("require_full_ablation_and_cost_report", False))
            else True
        ),
    }
    decision = {
        "status": "development_gate_pass" if all(checks.values()) else "development_gate_fail",
        "checks": checks,
        "primary_vs_deployable_best_single": deployable,
        "primary_vs_full_development_best_single_descriptive": descriptive,
        "beats_every_fixed_single_model": checks["beats_every_fixed_single_model"],
        "blind_test_authorized": bool(all(checks.values()))
        and bool(acceptance["blind_test_authorized_only_after_all_development_gates_pass"]),
        "required_development_comparison_coverage": comparison_coverage,
        "claim_boundary": (
            "Development-only nested OOF evidence; no confirmatory or global novelty claim."
        ),
    }
    return method_rows, comparisons, decision


def _plot_results(
    output_dir: Path,
    method_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchcoe_c3_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method = {str(row["method"]): row for row in method_rows}
    shortlist = [
        "c3_primary_nested",
        "best_single_nested_oof",
        "full_development_best_single_descriptive",
        "majority_vote",
        "source_weighted_vote",
        "source_confusion_bayes",
        "uncalibrated_certificates",
        "uncalibrated_cross_examined_certificates",
        "direct_anonymous_answer_vote",
        "sealed_exact_agreement_vote",
        "el_dgr_style_conservative_admissibility",
        "agent_auditor_style_localized_divergence",
        "beyond_consensus_static_calibration_nested",
        "beyond_consensus_minority_veto_nested",
        "minority_sentinel_style_nested",
        CFMAD_STYLE_METHOD,
        PREPAIR_TOP2_METHOD,
        PREPAIR_BUDGET_MATCHED_METHOD,
        "c3_certificates_only",
        "c3_no_generator_answer_dependence",
        "c3_no_checker_answer_dependence",
        "c3_no_generator_checker_pair_effects",
        "c3_no_sealed_set_agreement",
        "c3_no_counterfactual_parity",
        "c3_base_outputs_only",
    ]
    shortlist.extend(
        sorted(
            method
            for method in by_method
            if method.startswith("c3_prompt_ablation::")
        )
    )
    shortlist.extend(
        sorted(method for method in by_method if method.startswith("equal_call::"))
    )
    shortlist.extend(
        sorted(method for method in by_method if method.startswith("cfmad_style::"))
    )
    shortlist = [method for method in shortlist if method in by_method]
    figure_height = max(9.0, 2.0 + 0.38 * len(shortlist))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, figure_height),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    positions = np.arange(len(shortlist))
    values = [100.0 * float(by_method[method]["accuracy"]) for method in shortlist]
    colors = ["#147D64" if method == "c3_primary_nested" else "#66717E" for method in shortlist]
    axes[0].barh(positions, values, color=colors)
    axes[0].set_yticks(positions, shortlist)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Nested OOF accuracy (%)")
    axes[0].set_title("C3 development methods")
    axes[0].grid(axis="x", alpha=0.2)
    for position, value in zip(positions, values, strict=True):
        axes[0].text(value + 0.12, position, f"{value:.2f}", va="center", fontsize=8)

    shown_references = set(shortlist).difference({"c3_primary_nested"})
    shown = [row for row in comparisons if row["reference"] in shown_references]
    positions = np.arange(len(shown))
    deltas = np.asarray([100.0 * float(row["delta"]) for row in shown])
    lows = np.asarray(
        [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][0]) for row in shown]
    )
    highs = np.asarray(
        [100.0 * float(row["stratified_paired_bootstrap_delta_ci95"][1]) for row in shown]
    )
    axes[1].errorbar(
        deltas,
        positions,
        xerr=np.vstack((deltas - lows, highs - deltas)),
        fmt="o",
        color="#263238",
        ecolor="#147D64",
        capsize=4,
    )
    axes[1].axvline(0.0, color="#B3261E", linewidth=1.2)
    axes[1].set_yticks(positions, [str(row["reference"]) for row in shown])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("C3 paired delta (percentage points)")
    axes[1].set_title("Environment-stratified paired comparisons")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Cross-Examined Counterexample Certificates", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "c3_development_results.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "c3_development_results.pdf", bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    method_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    priority = {
        "c3_primary_nested": 0,
        "best_single_nested_oof": 1,
        "full_development_best_single_descriptive": 2,
        "majority_vote": 3,
        "source_weighted_vote": 4,
        "source_confusion_bayes": 5,
        "uncalibrated_certificates": 6,
        "uncalibrated_cross_examined_certificates": 7,
        "direct_anonymous_answer_vote": 8,
        "el_dgr_style_conservative_admissibility": 9,
        "agent_auditor_style_localized_divergence": 10,
        "beyond_consensus_static_calibration_nested": 11,
        "beyond_consensus_minority_veto_nested": 12,
        "minority_sentinel_style_nested": 13,
        PREPAIR_TOP2_METHOD: 14,
        PREPAIR_BUDGET_MATCHED_METHOD: 15,
    }
    shown = sorted(
        method_rows,
        key=lambda row: (priority.get(str(row["method"]), 20), str(row["method"])),
    )
    lines = [
        "# C3 nested development result",
        "",
        f"Decision: **{decision['status']}**",
        "",
        "All C3 predictions are outer leave-one-environment-out predictions. Regularization "
        "and intervention margin were chosen only by a second leave-one-environment-out loop "
        "inside each outer training fold.",
        "",
        "## Accuracy and cost",
        "",
        "| Method | Calls/question | Micro | MMLU-Pro | GPQA | Environment macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in shown:
        per_dataset = row["per_dataset_accuracy"]
        lines.append(
            f"| {row['method']} | {float(row['nominal_model_calls_per_question']):.2f} | "
            f"{100*float(row['accuracy']):.2f}% | "
            f"{100*float(per_dataset.get('mmlu_pro', 0.0)):.2f}% | "
            f"{100*float(per_dataset.get('gpqa', 0.0)):.2f}% | "
            f"{100*float(row['macro_environment_accuracy']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Reference | Delta | Stratified 95% CI | Rescue/Harm | McNemar p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        ci = row["stratified_paired_bootstrap_delta_ci95"]
        lines.append(
            f"| {row['reference']} | {100*float(row['delta']):+.2f} pp | "
            f"[{100*float(ci[0]):+.2f}, {100*float(ci[1]):+.2f}] pp | "
            f"{row['rescue_count']}/{row['harm_count']} | "
            f"{float(row['exact_mcnemar_p']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["checks"].items()],
            "",
            f"Blind test authorized: **{decision['blind_test_authorized']}**.",
            "",
            "## Required-comparison coverage",
            "",
            "| Requirement | Fidelity | Present | Missing method artifacts |",
            "| --- | --- | ---: | --- |",
            *[
                f"| {row['requirement']} | {row['fidelity']} | "
                f"{'yes' if row['present'] else 'no'} | "
                f"{', '.join(row['missing_methods']) if row['missing_methods'] else '-'} |"
                for row in decision["required_development_comparison_coverage"]
            ],
            "",
            "This is development-only evidence. The full-development best single is a "
            "descriptive, post-hoc envelope; the nested OOF best single is the deployable "
            "primary reference. Pretraining contamination is not claimed to be excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    run_root = args.run_root or Path(str(config["output_root"]))
    output_dir = args.output_dir or run_root / "development_evaluation"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started = time.time()
    data = load_and_authenticate_c3_data(
        args.config,
        config,
        run_root,
        args.equal_call_config,
        args.prepair_config,
        args.cfmad_config,
    )
    ablation_checks: dict[str, tuple[CertificateCheck, ...]] = {}
    ablation_quality: dict[str, Mapping[str, Any]] = {}
    seen_ablation_configs: set[Path] = set()
    for ablation_config_path in args.mechanism_ablation_config:
        resolved_path = ablation_config_path.resolve()
        if resolved_path in seen_ablation_configs:
            raise ValueError("Duplicate C3 mechanism ablation config")
        seen_ablation_configs.add(resolved_path)
        ablation_config = _load_config(ablation_config_path)
        ablation_name, ablation_run_root = _validate_mechanism_ablation_config(
            args.config,
            config,
            run_root,
            ablation_config_path,
            ablation_config,
        )
        if ablation_name in ablation_checks:
            raise ValueError(f"Duplicate C3 mechanism ablation: {ablation_name}")
        ablation_data = load_and_authenticate_c3_data(
            args.config,
            config,
            run_root,
            args.equal_call_config,
            args.prepair_config,
            args.cfmad_config,
            check_config_path=ablation_config_path,
            check_config=ablation_config,
            check_run_root=ablation_run_root,
        )
        ablation_checks[ablation_name] = ablation_data.checks
        ablation_quality[ablation_name] = {
            "config_path": str(ablation_config_path),
            "config_sha256": sha256_file(ablation_config_path),
            "check_run_root": str(ablation_run_root),
            "certificate_checkers": ablation_data.generation_quality[
                "certificate_checkers"
            ],
        }
    data = replace(
        data,
        mechanism_ablation_checks=ablation_checks,
        mechanism_ablation_quality=ablation_quality,
        generation_quality={
            **dict(data.generation_quality),
            "prompt_mechanism_ablations": ablation_quality,
        },
    )
    predictions, nested_rows, outer_rows, primary_diagnostics = (
        generate_nested_c3_predictions(config, data)
    )
    for ablation_name, checks in sorted(data.mechanism_ablation_checks.items()):
        ablation_data = replace(
            data,
            checks=checks,
            equal_call_predictions={},
            equal_call_call_budgets={},
            pre_pair_predictions={},
            pre_pair_call_budgets={},
            cfmad_predictions={},
            cfmad_call_budgets={},
            mechanism_ablation_checks={},
            mechanism_ablation_quality={},
        )
        (
            ablation_predictions,
            ablation_nested_rows,
            ablation_outer_rows,
            _,
        ) = generate_nested_c3_predictions(config, ablation_data)
        method_name = f"c3_prompt_ablation::{ablation_name}"
        predictions[method_name] = dict(
            ablation_predictions["c3_primary_nested"]
        )
        if ablation_name == "pair_visible_with_precommitment":
            certificates_by_question: dict[
                str, list[CounterexampleCertificate]
            ] = defaultdict(list)
            checks_by_question: dict[str, list[CertificateCheck]] = defaultdict(
                list
            )
            for certificate in ablation_data.certificates:
                certificates_by_question[certificate.question_id].append(
                    certificate
                )
            for check in ablation_data.checks:
                checks_by_question[check.question_id].append(check)
            localized_predictions: dict[str, str] = {}
            for question in ablation_data.questions:
                reference = predictions["best_single_nested_oof"][
                    question.question_id
                ]
                if reference not in question.option_labels:
                    reference = question.option_labels[0]
                localized_predictions[question.question_id] = (
                    _agent_auditor_style_localized_divergence_vote(
                        question,
                        certificates_by_question[question.question_id],
                        checks_by_question[question.question_id],
                        str(reference),
                    )
                )
            predictions["agent_auditor_style_localized_divergence"] = (
                localized_predictions
            )
        nested_rows.extend(
            {"prompt_mechanism_ablation": ablation_name, **row}
            for row in ablation_nested_rows
        )
        outer_rows.extend(
            {"prompt_mechanism_ablation": ablation_name, **row}
            for row in ablation_outer_rows
        )
    method_rows, comparisons, decision = evaluate_c3_predictions(
        config, data, predictions
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    per_query = [
        {
            "question_id": question_id,
            "dataset": data.dataset_by_question[question_id],
            "environment": data.environment_by_question[question_id],
            "development_answer": data.answers[question_id],
            "predictions": {
                method: values[question_id] for method, values in sorted(predictions.items())
            },
            "correctness": {
                method: values[question_id] == data.answers[question_id]
                for method, values in sorted(predictions.items())
            },
            "c3_primary_diagnostics": primary_diagnostics[question_id],
        }
        for question_id in sorted(data.answers)
    ]
    write_jsonl(output_dir / "per_query_predictions.jsonl", per_query)
    write_jsonl(output_dir / "nested_variant_search.jsonl", nested_rows)
    write_jsonl(output_dir / "outer_fold_selections.jsonl", outer_rows)
    write_json(output_dir / "method_summaries.json", method_rows)
    write_csv(
        output_dir / "method_summaries.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in method_rows
        ],
    )
    write_json(output_dir / "paired_comparisons.json", comparisons)
    write_csv(
        output_dir / "paired_comparisons.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in comparisons
        ],
    )
    write_json(output_dir / "generation_quality.json", data.generation_quality)
    write_json(output_dir / "development_gate.json", decision)
    write_json(
        output_dir / "required_comparison_coverage.json",
        decision["required_development_comparison_coverage"],
    )
    _plot_results(output_dir, method_rows, comparisons)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(
        _markdown_report(method_rows, comparisons, decision), encoding="utf-8"
    )
    artifact_paths = [
        output_dir / "per_query_predictions.jsonl",
        output_dir / "nested_variant_search.jsonl",
        output_dir / "outer_fold_selections.jsonl",
        output_dir / "method_summaries.json",
        output_dir / "paired_comparisons.json",
        output_dir / "generation_quality.json",
        output_dir / "development_gate.json",
        output_dir / "required_comparison_coverage.json",
        report_path,
        output_dir / "c3_development_results.png",
        output_dir / "c3_development_results.pdf",
    ]
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "status": decision["status"],
            "protocol": "outer LOEO with inner LOEO C3 selection",
            "development_only": True,
            "questions": len(data.questions),
            "methods": len(predictions),
            "started_unix": started,
            "finished_unix": time.time(),
            "artifact_hashes": {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in artifact_paths
            },
            "environment": environment_manifest(
                sys.argv,
                int(config["seed"]),
                [
                    args.config,
                    *(
                        [args.equal_call_config]
                        if args.equal_call_config is not None
                        else []
                    ),
                    *(
                        [args.prepair_config]
                        if args.prepair_config is not None
                        else []
                    ),
                    *(
                        [args.cfmad_config]
                        if args.cfmad_config is not None
                        else []
                    ),
                    *args.mechanism_ablation_config,
                    run_root / "development_observables",
                    run_root / "development_labels",
                    run_root / "certificates",
                    run_root / "checks",
                    run_root / "equal_call_single_model",
                    run_root / "prepair_style",
                    run_root / "cfmad_style",
                    run_root / "mechanism_ablations",
                ],
            ),
        },
    )
    print(f"Completed nested C3 development evaluation: {output_dir}")


if __name__ == "__main__":
    main()
