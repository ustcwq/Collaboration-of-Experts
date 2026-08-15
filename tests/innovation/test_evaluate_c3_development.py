from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from bench_coe.innovation.blind_falsification_jury import (
    BasePrediction,
    FalsificationQuestion,
    candidate_label_key,
)
from bench_coe.innovation.cross_examined_certificates import (
    C3Variant,
    CertificateCheck,
    CounterexampleCertificate,
)
from bench_coe.innovation.c3_prior_art_controls import (
    parse_candidate_visible_proof_output_v8_control,
    parse_unsealed_proof_output_v8_control,
)
from bench_coe.innovation.evaluate_c3_development import (
    C3DevelopmentData,
    _certificate_from_row,
    _check_from_row,
    _el_dgr_style_conservative_admissibility,
    _agent_auditor_style_localized_divergence_vote,
    _prompt_mechanism_ablation_gate,
    _required_development_comparison_coverage,
    _source_confusion_bayes_predictions,
    _validate_mechanism_ablation_config,
    c3_variants_from_config,
    generate_nested_c3_predictions,
    minority_veto_variants_from_config,
    minority_sentinel_style_variants_from_config,
    select_c3_variant_nested,
    select_minority_veto_nested,
    select_minority_sentinel_style_nested,
    select_static_calibration_nested,
    static_calibration_variants_from_config,
)
from bench_coe.innovation.artifacts import sha256_file
from bench_coe.innovation.schema import SourceTrainingLabels
from bench_coe.innovation.sealed_counterfactual_parity import (
    counterfactual_trace_slot,
    parse_blind_counterfactual_parity_output_v4,
    parse_committed_counterfactual_challenge_output_v6,
    parse_commitment_conditioned_pair_audit_output_v8_ablation,
    parse_commitment_conditioned_proof_audit_output_v8,
    parse_sealed_counterfactual_challenge_output_v4,
)


class EvaluateC3DevelopmentTests(unittest.TestCase):
    def _data(self):
        questions = [
            FalsificationQuestion(
                f"q{index}",
                "synthetic",
                f"env{index % 4}",
                f"Question {index}",
                ("first", "second"),
                ("A", "B"),
            )
            for index in range(12)
        ]
        answers = {
            question.question_id: ("A" if index % 2 == 0 else "B")
            for index, question in enumerate(questions)
        }
        base = [
            BasePrediction(question.question_id, expert, "A")
            for question in questions
            for expert in ("generator", "checker", "weak")
        ]
        certificates = []
        checks = []
        correctness = {}
        environments = {}
        for question in questions:
            answer = answers[question.question_id]
            environments[question.question_id] = question.environment
            for expert in ("generator", "checker", "weak"):
                correctness[(question.question_id, expert)] = answer == "A"
            for candidate in question.option_labels:
                candidate_is_correct = candidate == answer
                correctness[(question.question_id, candidate_label_key(candidate))] = (
                    candidate_is_correct
                )
                certificate = CounterexampleCertificate(
                    question.question_id,
                    "generator",
                    candidate,
                    "SURVIVES" if candidate_is_correct else "FALSIFIED",
                    95,
                    answer,
                    "premise",
                    "check",
                    "NONE" if candidate_is_correct else "fatal",
                )
                certificates.append(certificate)
                checks.append(
                    CertificateCheck(
                        certificate.certificate_id,
                        question.question_id,
                        "generator",
                        "checker",
                        candidate,
                        "VALID_SUPPORT" if candidate_is_correct else "VALID_REFUTATION",
                        95,
                        "A",
                        "NONE",
                        logic_status="VALID",
                        eliminated_options=() if candidate_is_correct else (candidate,),
                        supported_options=(candidate,) if candidate_is_correct else (),
                        target_was_hidden=True,
                    )
                )
        labels = SourceTrainingLabels._from_source_adapter(
            "synthetic", "development", correctness, environments
        )
        return questions, base, certificates, checks, answers, labels

    def test_variant_grids_are_frozen_and_deterministic(self) -> None:
        config = {
            "variant_grid": {
                "regularization_c": [0.1, 1.0],
                "intervention_margin": [0.0, 0.1],
                "open_option_set": True,
                "use_certificates": True,
                "use_checks": True,
                "use_generator_answer_dependence": True,
                "use_checker_answer_dependence": True,
                "use_generator_checker_pair_effects": True,
            },
            "near_prior_baseline_grid": {
                "static_base_prior_strength": [0.0, 1.0],
                "static_intervention_margin": [0.0, 0.1],
                "minority_veto_threshold": [1, 2],
            },
        }
        self.assertEqual(len(c3_variants_from_config(config)), 4)
        self.assertEqual(len(static_calibration_variants_from_config(config)), 4)
        self.assertEqual(len(minority_veto_variants_from_config(config)), 2)
        self.assertEqual(len(minority_sentinel_style_variants_from_config(config)), 6)
        self.assertEqual(c3_variants_from_config(config), c3_variants_from_config(config))

    def test_mechanism_ablation_config_cannot_change_fixed_inputs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base_path = root / "base.yaml"
            base_path.write_text("seed: 7\n", encoding="utf-8")
            run_root = root / "run"
            base_config = {
                "protocol_version": 8,
                "seed": 7,
                "models_dir": "models",
                "physical_gpus": [0, 1, 2, 3],
                "experts": ["expert"],
                "certificate_models": ["author"],
                "checker_models": ["checker"],
                "datasets": [{"name": "source"}],
                "certificate_generation": {"prompt_version": "frozen"},
                "check_generation": {
                    "prompt_version": "commitment_conditioned_proof_audit_v8",
                    "parser_version": "proof_obligation_audit_fields_v8",
                    "seed": 7,
                    "temperature": 0.0,
                    "max_new_tokens": 256,
                },
            }
            ablation_config = {
                **copy.deepcopy(base_config),
                "output_root": str(
                    run_root
                    / "mechanism_ablations"
                    / "no_checker_private_precommitment"
                ),
                "mechanism_ablation": {
                    "name": "no_checker_private_precommitment",
                    "base_config_path": str(base_path),
                    "base_config_sha256": sha256_file(base_path),
                    "changed_factor": (
                        "checker_private_stage0_response_removed_from_prompt"
                    ),
                    "unchanged_call_budget": True,
                    "target_labels_control_stopping_or_selection": False,
                },
            }
            ablation_config["check_generation"] = {
                **base_config["check_generation"],
                "prompt_version": "blind_isolated_trace_audit_v7",
                "parser_version": "isolated_trace_audit_fields_v7",
            }
            name, output_root = _validate_mechanism_ablation_config(
                base_path,
                base_config,
                run_root,
                root / "ablation.yaml",
                ablation_config,
            )
            self.assertEqual(name, "no_checker_private_precommitment")
            self.assertEqual(output_root, Path(ablation_config["output_root"]))

            tampered = copy.deepcopy(ablation_config)
            tampered["datasets"] = [{"name": "different_source"}]
            with self.assertRaises(PermissionError):
                _validate_mechanism_ablation_config(
                    base_path,
                    base_config,
                    run_root,
                    root / "tampered.yaml",
                    tampered,
                )

            tampered = copy.deepcopy(ablation_config)
            tampered["check_generation"]["temperature"] = 0.2
            with self.assertRaises(PermissionError):
                _validate_mechanism_ablation_config(
                    base_path,
                    base_config,
                    run_root,
                    root / "tampered_decoding.yaml",
                    tampered,
                )

    def test_registered_v8_prior_art_prompt_controls_are_bound_to_frozen_base(self) -> None:
        base_path = Path("configs/innovation/c3_development_v8.yaml")
        base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        run_root = Path(str(base_config["output_root"]))
        expected = {
            "configs/innovation/c3_v8_candidate_visible_commit_first_control.yaml": (
                "candidate_visible_commit_first"
            ),
            "configs/innovation/c3_v8_unsealed_set_aware_control.yaml": (
                "unsealed_set_aware"
            ),
        }
        for raw_path, expected_name in expected.items():
            path = Path(raw_path)
            control = yaml.safe_load(path.read_text(encoding="utf-8"))
            name, output_root = _validate_mechanism_ablation_config(
                base_path, base_config, run_root, path, control
            )
            self.assertEqual(name, expected_name)
            self.assertEqual(
                output_root.resolve(),
                (run_root / "mechanism_ablations" / expected_name).resolve(),
            )

    def test_prompt_mechanism_gate_requires_both_and_strict_improvement(self) -> None:
        complete_rows = [
            {
                "method": "c3_prompt_ablation::no_checker_private_precommitment",
                "accuracy": 0.70,
            },
            {
                "method": "c3_prompt_ablation::pair_visible_with_precommitment",
                "accuracy": 0.71,
            },
            {
                "method": "c3_prompt_ablation::candidate_visible_commit_first",
                "accuracy": 0.69,
            },
            {
                "method": "c3_prompt_ablation::unsealed_set_aware",
                "accuracy": 0.68,
            },
        ]
        _, present, beats = _prompt_mechanism_ablation_gate(0.72, complete_rows)
        self.assertTrue(present)
        self.assertTrue(beats)

        _, present, beats = _prompt_mechanism_ablation_gate(0.71, complete_rows)
        self.assertTrue(present)
        self.assertFalse(beats)

        _, present, beats = _prompt_mechanism_ablation_gate(
            0.72, complete_rows[:-1]
        )
        self.assertFalse(present)
        self.assertFalse(beats)

    def test_required_comparison_coverage_exposes_unimplemented_controls(self) -> None:
        experts = ("expert-a", "expert-b")
        models = ("model-a", "model-b")
        predictions = {
            "single::expert-a": {},
            "single::expert-b": {},
            "full_development_best_single_descriptive": {},
            "best_single_nested_oof": {},
            "majority_vote": {},
        }
        rows, complete = _required_development_comparison_coverage(
            predictions, experts, models, models
        )
        by_requirement = {row["requirement"]: row for row in rows}
        self.assertTrue(by_requirement["every_fixed_single_expert"]["present"])
        self.assertTrue(by_requirement["cross_model_consensus"]["present"])
        self.assertFalse(
            by_requirement["unsealed_set_aware_verification"]["present"]
        )
        self.assertEqual(
            by_requirement["unsealed_set_aware_verification"]["missing_methods"],
            ["c3_prompt_ablation::unsealed_set_aware"],
        )
        self.assertFalse(complete)

    def test_strict_prior_art_style_readouts_require_complete_cross_model_evidence(self) -> None:
        question = FalsificationQuestion(
            "q-strict", "synthetic", "env", "Which?", ("one", "two"), ("A", "B")
        )
        certificates = []
        isolated_checks = []
        pair_visible_checks = []
        claims = (
            ("eliminate-1", "A", "ELIMINATES"),
            ("eliminate-2", "A", "ELIMINATES"),
            ("support-1", "B", "SUPPORTS"),
            ("support-2", "B", "SUPPORTS"),
        )
        for generator, candidate, effect in claims:
            eliminated = (candidate,) if effect == "ELIMINATES" else ()
            supported = (candidate,) if effect == "SUPPORTS" else ()
            certificate = CounterexampleCertificate(
                question_id=question.question_id,
                generator_id=generator,
                candidate=candidate,
                verdict="FALSIFIED" if eliminated else "SURVIVES",
                confidence=90,
                alternative=None,
                premise="rule",
                check="valid trace",
                failure="invalid trace flaw",
                witness_id=f"{question.question_id}::{generator}",
                claimed_eliminated_options=eliminated,
                claimed_supported_options=supported,
                claim_was_sealed=True,
                counterfactual_pair=True,
                challenge_rule="rule",
                trace_1="valid trace",
                trace_2="invalid trace",
                first_differing_step="first flaw",
                sealed_valid_trace=1,
                sealed_effect=effect,
            )
            certificates.append(certificate)
            for checker in ("checker-1", "checker-2"):
                status = "VALID_REFUTATION" if eliminated else "VALID_SUPPORT"
                isolated_checks.extend(
                    (
                        CertificateCheck(
                            certificate.certificate_id,
                            question.question_id,
                            generator,
                            checker,
                            candidate,
                            status,
                            90,
                            None,
                            None,
                            logic_status="VALID",
                            eliminated_options=eliminated,
                            supported_options=supported,
                            target_was_hidden=True,
                            counterfactual_pair=True,
                            orientation="trace_1",
                            canonical_valid_trace=1,
                            reconstructed_effect=effect,
                        ),
                        CertificateCheck(
                            certificate.certificate_id,
                            question.question_id,
                            generator,
                            checker,
                            candidate,
                            "INCONCLUSIVE",
                            90,
                            None,
                            "fatal flaw",
                            logic_status="INVALID",
                            target_was_hidden=True,
                            counterfactual_pair=True,
                            orientation="trace_2",
                        ),
                    )
                )
                for orientation in ("canonical", "mirrored"):
                    pair_visible_checks.append(
                        CertificateCheck(
                            certificate.certificate_id,
                            question.question_id,
                            generator,
                            checker,
                            candidate,
                            status,
                            90,
                            None,
                            "localized flaw",
                            logic_status="VALID",
                            eliminated_options=eliminated,
                            supported_options=supported,
                            target_was_hidden=True,
                            counterfactual_pair=True,
                            orientation=orientation,
                            presented_valid_trace=1,
                            canonical_valid_trace=1,
                            reconstructed_effect=effect,
                        )
                    )
        self.assertEqual(
            _el_dgr_style_conservative_admissibility(
                question, certificates, isolated_checks, "A"
            ),
            "B",
        )
        self.assertEqual(
            _agent_auditor_style_localized_divergence_vote(
                question, certificates, pair_visible_checks, "A"
            ),
            "B",
        )
        self.assertEqual(
            _el_dgr_style_conservative_admissibility(
                question, certificates, isolated_checks[:2], "A"
            ),
            "A",
        )

    def test_v4_authenticator_replays_seal_and_both_orientation_metadata(self) -> None:
        question = FalsificationQuestion(
            "q-auth", "synthetic", "env", "Which follows?", ("one", "two"), ("A", "B")
        )
        seed = 7
        slot = counterfactual_trace_slot(seed, question.question_id, "generator")
        raw_challenge = (
            "RULE: conservation fixes the total\n"
            "TRACE_1: subtracting the known part yields two\n"
            "TRACE_2: adding the known part yields two\n"
            "FIRST_DIFFERING_STEP: subtract versus add\n"
            f"SEALED_VALID_TRACE: {slot}\n"
            "SEALED_EFFECT: SUPPORTS\n"
            "SEALED_OPTION: B\n"
            "CONFIDENCE: 88"
        )
        certificate_row = {
            "certificate_id": "q-auth::generator::B",
            "witness_id": "q-auth::generator",
            "question_id": "q-auth",
            "generator_id": "generator",
            "candidate": "B",
            "verdict": "SURVIVES",
            "confidence": 88,
            "alternative": "B",
            "premise": "conservation fixes the total",
            "check": "subtracting the known part yields two",
            "failure": "subtract versus add",
            "claimed_eliminated_options": [],
            "claimed_supported_options": ["B"],
            "claim_was_sealed": True,
            "counterfactual_pair": True,
            "challenge_rule": "conservation fixes the total",
            "trace_1": "subtracting the known part yields two",
            "trace_2": "adding the known part yields two",
            "first_differing_step": "subtract versus add",
            "sealed_valid_trace": slot,
            "sealed_effect": "SUPPORTS",
            "required_valid_trace": slot,
            "parse_error": None,
            "raw_output": raw_challenge,
        }
        certificate = _certificate_from_row(
            certificate_row,
            question,
            parse_sealed_counterfactual_challenge_output_v4,
            True,
            True,
            seed,
        )
        raw_audit = (
            f"PAIR_STATUS: ONE_VALID\nVALID_TRACE: {slot}\nEFFECT: SUPPORTS\n"
            "OPTION: B\nCONFIDENCE: 91\nFIRST_FLAW: the operation is reversed"
        )
        check_row = {
            "certificate_id": certificate.certificate_id,
            "witness_id": certificate.witness_id,
            "question_id": question.question_id,
            "generator_id": "generator",
            "checker_id": "checker",
            "candidate": "B",
            "status": "VALID_SUPPORT",
            "confidence": 91,
            "independent_answer": "A",
            "first_flaw": "the operation is reversed",
            "parse_error": None,
            "logic_status": "VALID",
            "eliminated_options": [],
            "supported_options": ["B"],
            "target_was_hidden": True,
            "sealed_claim_was_hidden": True,
            "counterfactual_pair": True,
            "orientation": "canonical",
            "pair_status": "ONE_VALID",
            "presented_valid_trace": slot,
            "canonical_valid_trace": slot,
            "reconstructed_effect": "SUPPORTS",
            "raw_output": raw_audit,
        }
        check = _check_from_row(
            check_row,
            question,
            certificate,
            parse_blind_counterfactual_parity_output_v4,
            True,
            True,
        )
        self.assertEqual(check.canonical_valid_trace, slot)

        tampered_certificate = copy.deepcopy(certificate_row)
        tampered_certificate["sealed_effect"] = "ELIMINATES"
        with self.assertRaises(PermissionError):
            _certificate_from_row(
                tampered_certificate,
                question,
                parse_sealed_counterfactual_challenge_output_v4,
                True,
                True,
                seed,
            )
        tampered_check = copy.deepcopy(check_row)
        tampered_check["canonical_valid_trace"] = 3 - slot
        with self.assertRaises(PermissionError):
            _check_from_row(
                tampered_check,
                question,
                certificate,
                parse_blind_counterfactual_parity_output_v4,
                True,
                True,
            )

    def test_all_nested_selectors_cover_each_inner_environment_once(self) -> None:
        questions, base, certificates, checks, answers, labels = self._data()
        c3_variants = (
            C3Variant("c3_a", regularization_c=0.1),
            C3Variant("c3_b", regularization_c=1.0, intervention_margin=0.1),
        )
        with self.assertRaises(PermissionError):
            select_c3_variant_nested(
                questions,
                base,
                certificates,
                checks,
                labels,
                {**answers, "outside_outer_training_fold": "B"},
                c3_variants,
                seed=7,
            )
        selected, c3_rows = select_c3_variant_nested(
            questions,
            base,
            certificates,
            checks,
            labels,
            answers,
            c3_variants,
            seed=7,
        )
        self.assertIn(selected, c3_variants)
        self.assertEqual(sum(bool(row["selected"]) for row in c3_rows), 1)
        self.assertTrue(all(row["samples"] == len(questions) for row in c3_rows))

        static_variants = static_calibration_variants_from_config(
            {
                "near_prior_baseline_grid": {
                    "static_base_prior_strength": [0.0, 1.0],
                    "static_intervention_margin": [0.0],
                    "minority_veto_threshold": [1, 2],
                }
            }
        )
        _, static_rows = select_static_calibration_nested(
            questions, base, checks, labels, static_variants
        )
        veto_variants = (
            minority_veto_variants_from_config(
                {
                    "near_prior_baseline_grid": {
                        "static_base_prior_strength": [0.0],
                        "static_intervention_margin": [0.0],
                        "minority_veto_threshold": [1, 2],
                    }
                }
            )
        )
        _, veto_rows = select_minority_veto_nested(
            questions, base, checks, labels, veto_variants
        )
        sentinel_variants = minority_sentinel_style_variants_from_config(
            {
                "near_prior_baseline_grid": {
                    "static_base_prior_strength": [0.0],
                    "static_intervention_margin": [0.0],
                    "minority_veto_threshold": [1],
                }
            }
        )
        _, sentinel_rows = select_minority_sentinel_style_nested(
            questions,
            base,
            checks,
            labels,
            sentinel_variants,
            seed=7,
        )
        self.assertTrue(all(row["samples"] == len(questions) for row in static_rows))
        self.assertTrue(all(row["samples"] == len(questions) for row in veto_rows))
        self.assertEqual(sum(bool(row["selected"]) for row in static_rows), 1)
        self.assertEqual(sum(bool(row["selected"]) for row in veto_rows), 1)
        self.assertTrue(
            all(row["samples"] == len(questions) for row in sentinel_rows)
        )
        self.assertEqual(sum(bool(row["selected"]) for row in sentinel_rows), 1)
        self.assertTrue(
            all(
                float(row["majority_correct_preservation_rate"]) >= 0.95
                for row in sentinel_rows
                if row["selected"]
            )
        )

    def test_source_confusion_bayes_uses_only_outer_training_labels(self) -> None:
        train_questions = [
            FalsificationQuestion(
                f"train-{index}",
                "synthetic",
                f"env-{index}",
                "Question",
                ("one", "two"),
                ("A", "B"),
            )
            for index in range(6)
        ]
        answers = {
            question.question_id: ("A" if index % 2 == 0 else "B")
            for index, question in enumerate(train_questions)
        }
        train_base = []
        for question in train_questions:
            answer = answers[question.question_id]
            train_base.extend(
                (
                    BasePrediction(question.question_id, "good", answer),
                    BasePrediction(
                        question.question_id,
                        "bad",
                        "B" if answer == "A" else "A",
                    ),
                )
            )
        target = FalsificationQuestion(
            "target", "synthetic", "heldout", "Target", ("one", "two"), ("A", "B")
        )
        predicted = _source_confusion_bayes_predictions(
            train_questions,
            train_base,
            answers,
            (target,),
            (
                BasePrediction("target", "good", "B"),
                BasePrediction("target", "bad", "A"),
            ),
            "good",
        )
        self.assertEqual(predicted, {"target": "B"})
        with self.assertRaises(PermissionError):
            _source_confusion_bayes_predictions(
                train_questions,
                train_base,
                {},
                (target,),
                (),
                "good",
            )
        with self.assertRaises(PermissionError):
            _source_confusion_bayes_predictions(
                train_questions,
                train_base,
                {**answers, "outside_outer_training_fold": "B"},
                (target,),
                (),
                "good",
            )

    def test_v6_authenticator_replays_post_commit_permutation(self) -> None:
        question = FalsificationQuestion(
            "q-v6", "synthetic", "env", "Which follows?", ("one", "two"), ("A", "B")
        )
        seed = 11
        required = counterfactual_trace_slot(seed, question.question_id, "generator")
        author_slot = 3 - required
        raw = (
            "RULE: one checked identity\nTRACE_1: raw first trace\n"
            "TRACE_2: raw second trace\nFIRST_DIFFERING_STEP: first versus second\n"
            f"SEALED_VALID_TRACE: {author_slot}\nSEALED_EFFECT: SUPPORTS\n"
            "SEALED_OPTION: B\nCONFIDENCE: 90"
        )
        row = {
            "certificate_id": "q-v6::generator::B",
            "witness_id": "q-v6::generator",
            "question_id": "q-v6",
            "generator_id": "generator",
            "candidate": "B",
            "verdict": "SURVIVES",
            "confidence": 90,
            "alternative": "B",
            "premise": "one checked identity",
            "check": "raw second trace",
            "failure": "first versus second",
            "claimed_eliminated_options": [],
            "claimed_supported_options": ["B"],
            "claim_was_sealed": True,
            "counterfactual_pair": True,
            "challenge_rule": "one checked identity",
            "trace_1": "raw second trace",
            "trace_2": "raw first trace",
            "first_differing_step": "first versus second",
            "sealed_valid_trace": required,
            "sealed_effect": "SUPPORTS",
            "author_valid_trace": author_slot,
            "post_commit_permutation_applied": True,
            "required_valid_trace": required,
            "parse_error": None,
            "raw_output": raw,
        }
        certificate = _certificate_from_row(
            row,
            question,
            parse_committed_counterfactual_challenge_output_v6,
            True,
            True,
            seed,
            True,
        )
        self.assertEqual(certificate.sealed_valid_trace, required)
        self.assertEqual(certificate.trace_1, "raw second trace")
        tampered = copy.deepcopy(row)
        tampered["post_commit_permutation_applied"] = False
        with self.assertRaises(PermissionError):
            _certificate_from_row(
                tampered,
                question,
                parse_committed_counterfactual_challenge_output_v6,
                True,
                True,
                seed,
                True,
            )

    def test_v8_authenticator_replays_proof_obligations(self) -> None:
        question = FalsificationQuestion(
            "q-v8",
            "synthetic",
            "env",
            "Which follows?",
            ("one", "two"),
            ("A", "B"),
        )
        certificate = CounterexampleCertificate(
            question_id=question.question_id,
            generator_id="generator",
            candidate="B",
            verdict="SURVIVES",
            confidence=90,
            alternative="B",
            premise="conservation fixes the total",
            check="subtracting the known part yields two",
            failure="subtract versus add",
            witness_id="q-v8::generator",
            claimed_supported_options=("B",),
            claim_was_sealed=True,
            counterfactual_pair=True,
            challenge_rule="conservation fixes the total",
            trace_1="subtracting the known part yields two",
            trace_2="adding the known part yields two",
            first_differing_step="subtract versus add",
            sealed_valid_trace=1,
            sealed_effect="SUPPORTS",
        )
        raw_output = (
            "TRACE_STATUS: VALID\n"
            "COUNTERTEST: substitute the stated known part into the total\n"
            "COUNTERTEST_RESULT: SURVIVES\n"
            "RECOMPUTATION: total minus known part equals two\n"
            "COMMITMENT_RELATION: CONSISTENT\n"
            "EFFECT: SUPPORTS\n"
            "OPTION: B\n"
            "CONFIDENCE: 91\n"
            "FLAW_CODE: NONE\n"
            "FLAW_DETAIL: NONE"
        )
        row = {
            "certificate_id": certificate.certificate_id,
            "witness_id": certificate.witness_id,
            "question_id": question.question_id,
            "generator_id": certificate.generator_id,
            "checker_id": "checker",
            "candidate": certificate.candidate,
            "status": "VALID_SUPPORT",
            "confidence": 91,
            "independent_answer": "B",
            "first_flaw": None,
            "parse_error": None,
            "logic_status": "VALID",
            "eliminated_options": [],
            "supported_options": ["B"],
            "target_was_hidden": True,
            "sealed_claim_was_hidden": True,
            "counterfactual_pair": True,
            "orientation": "trace_1",
            "pair_status": None,
            "audit_protocol": "commitment_conditioned_proof_audit_v8",
            "trace_under_audit": "trace_1",
            "flaw_code": "NONE",
            "countertest": "substitute the stated known part into the total",
            "countertest_result": "SURVIVES",
            "recomputation": "total minus known part equals two",
            "commitment_relation": "CONSISTENT",
            "presented_valid_trace": None,
            "canonical_valid_trace": 1,
            "reconstructed_effect": "SUPPORTS",
            "raw_output": raw_output,
        }
        check = _check_from_row(
            row,
            question,
            certificate,
            parse_commitment_conditioned_proof_audit_output_v8,
            True,
            True,
        )
        self.assertEqual(check.status, "VALID_SUPPORT")
        tampered = copy.deepcopy(row)
        tampered["recomputation"] = "different result"
        with self.assertRaises(PermissionError):
            _check_from_row(
                tampered,
                question,
                certificate,
                parse_commitment_conditioned_proof_audit_output_v8,
                True,
                True,
            )

        candidate_visible = {
            **row,
            "target_was_hidden": False,
            "audit_protocol": "candidate_visible_commit_first_v8_control",
        }
        check = _check_from_row(
            candidate_visible,
            question,
            certificate,
            parse_candidate_visible_proof_output_v8_control,
            False,
            True,
            True,
        )
        self.assertFalse(check.target_was_hidden)

        unsealed = {
            **candidate_visible,
            "sealed_claim_was_hidden": False,
            "audit_protocol": "unsealed_set_aware_v8_control",
        }
        check = _check_from_row(
            unsealed,
            question,
            certificate,
            parse_unsealed_proof_output_v8_control,
            False,
            True,
            False,
        )
        self.assertEqual(check.status, "VALID_SUPPORT")

    def test_pair_visible_ablation_authenticator_replays_proof_obligations(self) -> None:
        question = FalsificationQuestion(
            "q-pair-ablation",
            "synthetic",
            "env",
            "Which follows?",
            ("one", "two"),
            ("A", "B"),
        )
        certificate = CounterexampleCertificate(
            question_id=question.question_id,
            generator_id="generator",
            candidate="B",
            verdict="SURVIVES",
            confidence=90,
            alternative="B",
            premise="conservation fixes the total",
            check="subtracting the known part yields two",
            failure="subtract versus add",
            witness_id="q-pair-ablation::generator",
            claimed_supported_options=("B",),
            claim_was_sealed=True,
            counterfactual_pair=True,
            challenge_rule="conservation fixes the total",
            trace_1="subtracting the known part yields two",
            trace_2="adding the known part yields two",
            first_differing_step="subtract versus add",
            sealed_valid_trace=1,
            sealed_effect="SUPPORTS",
        )
        raw_output = (
            "PAIR_STATUS: ONE_VALID\n"
            "COUNTERTEST: substitute the known part into both traces\n"
            "COUNTERTEST_RESULT: ONE_SURVIVES_ONE_BREAKS\n"
            "RECOMPUTATION: subtraction gives two while addition does not\n"
            "COMMITMENT_RELATION: CONSISTENT\n"
            "VALID_TRACE: 1\n"
            "EFFECT: SUPPORTS\n"
            "OPTION: B\n"
            "CONFIDENCE: 91\n"
            "FIRST_FLAW: trace two uses addition"
        )
        row = {
            "certificate_id": certificate.certificate_id,
            "witness_id": certificate.witness_id,
            "question_id": question.question_id,
            "generator_id": certificate.generator_id,
            "checker_id": "checker",
            "candidate": certificate.candidate,
            "status": "VALID_SUPPORT",
            "confidence": 91,
            "independent_answer": "B",
            "first_flaw": "trace two uses addition",
            "parse_error": None,
            "logic_status": "VALID",
            "eliminated_options": [],
            "supported_options": ["B"],
            "target_was_hidden": True,
            "sealed_claim_was_hidden": True,
            "counterfactual_pair": True,
            "orientation": "canonical",
            "pair_status": "ONE_VALID",
            "audit_protocol": "commitment_conditioned_pair_audit_v8_ablation",
            "trace_under_audit": None,
            "flaw_code": None,
            "countertest": "substitute the known part into both traces",
            "countertest_result": "ONE_SURVIVES_ONE_BREAKS",
            "recomputation": "subtraction gives two while addition does not",
            "commitment_relation": "CONSISTENT",
            "presented_valid_trace": 1,
            "canonical_valid_trace": 1,
            "reconstructed_effect": "SUPPORTS",
            "raw_output": raw_output,
        }
        check = _check_from_row(
            row,
            question,
            certificate,
            parse_commitment_conditioned_pair_audit_output_v8_ablation,
            True,
            True,
        )
        self.assertEqual(check.status, "VALID_SUPPORT")
        tampered = copy.deepcopy(row)
        tampered["countertest_result"] = "UNCERTAIN"
        with self.assertRaises(PermissionError):
            _check_from_row(
                tampered,
                question,
                certificate,
                parse_commitment_conditioned_pair_audit_output_v8_ablation,
                True,
                True,
            )

    def test_full_nested_prediction_graph_has_exact_coverage(self) -> None:
        questions, base, certificates, checks, answers, _ = self._data()
        config = {
            "seed": 7,
            "experts": ["generator", "checker", "weak"],
            "certificate_models": ["generator"],
            "checker_models": ["checker"],
            "variant_grid": {
                "regularization_c": [0.1],
                "intervention_margin": [0.0],
                "open_option_set": True,
                "use_certificates": True,
                "use_checks": True,
                "use_generator_answer_dependence": True,
                "use_checker_answer_dependence": True,
                "use_generator_checker_pair_effects": True,
                "use_sealed_set_agreement": True,
            },
            "near_prior_baseline_grid": {
                "static_base_prior_strength": [0.0],
                "static_intervention_margin": [0.0],
                "minority_veto_threshold": [1],
            },
        }
        question_ids = {question.question_id for question in questions}
        equal_call = {question_id: "A" for question_id in question_ids}
        data = C3DevelopmentData(
            questions=tuple(questions),
            base_predictions=tuple(base),
            certificates=tuple(certificates),
            checks=tuple(checks),
            answers=answers,
            dataset_by_question={
                question.question_id: question.dataset for question in questions
            },
            environment_by_question={
                question.question_id: question.environment for question in questions
            },
            generation_quality={},
            equal_call_predictions={"equal_call::synthetic": equal_call},
            equal_call_call_budgets={"equal_call::synthetic": 6.0},
        )
        predictions, nested_rows, outer_rows, diagnostics = (
            generate_nested_c3_predictions(config, data)
        )
        self.assertTrue(nested_rows)
        self.assertEqual(len(outer_rows), 4)
        self.assertEqual(set(diagnostics), question_ids)
        self.assertEqual(predictions["equal_call::synthetic"], equal_call)
        self.assertTrue(
            all(set(method_predictions) == question_ids for method_predictions in predictions.values())
        )


if __name__ == "__main__":
    unittest.main()
