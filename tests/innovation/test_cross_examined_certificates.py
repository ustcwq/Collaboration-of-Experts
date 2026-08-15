from __future__ import annotations

import inspect
import re
import unittest

from bench_coe.innovation import run_c3_certificate_checks, run_c3_certificates
from bench_coe.innovation.run_c3_certificate_checks import (
    _audit_protocol_name,
    _guided_regex_for_prompt_version,
    _stratified_smoke_certificates,
)
from bench_coe.innovation.run_c3_certificates import _stratified_smoke_questions
from bench_coe.innovation.blind_falsification_jury import BasePrediction, FalsificationQuestion, candidate_label_key
from bench_coe.innovation.cross_examined_certificates import (
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
from bench_coe.innovation.c3_prior_art_controls import (
    CANDIDATE_VISIBLE_PROMPT_VERSION,
    UNSEALED_PROMPT_VERSION,
    build_candidate_visible_commit_first_prompt_v8_control,
    build_unsealed_set_aware_prompt_v8_control,
)
from bench_coe.innovation.schema import EvaluationLabels, SourceTrainingLabels
from bench_coe.innovation.sealed_counterfactual_parity import (
    build_commitment_conditioned_pair_audit_prompt_v8_ablation,
    parse_commitment_conditioned_pair_audit_output_v8_ablation,
)


def _question(question_id: str) -> FalsificationQuestion:
    return FalsificationQuestion(
        question_id,
        "synthetic",
        "env",
        f"Which answer is correct for {question_id}?",
        ("first", "second"),
        ("A", "B"),
    )


class CrossExaminedCertificateTests(unittest.TestCase):
    def test_certificate_parser_is_strict_and_semantic(self) -> None:
        parsed = parse_certificate_output(
            "VERDICT: FALSIFIED\nCONFIDENCE: 90\nALTERNATIVE: (B)\n"
            "PREMISE: atomic rule\nCHECK: counterexample\nFAILURE: contradiction",
            ("A", "B"),
        )
        self.assertEqual(parsed, ("FALSIFIED", 90, "B", "atomic rule", "counterexample", "contradiction", None))
        invalid = parse_certificate_output(
            "VERDICT: FALSIFIED\nCONFIDENCE: 90\nALTERNATIVE: B\n"
            "PREMISE: rule\nCHECK: check\nFAILURE: NONE",
            ("A", "B"),
        )
        self.assertEqual(invalid[-1], "falsified_without_failure")

    def test_check_parser_fails_closed(self) -> None:
        parsed = parse_certificate_check_output(
            "CERTIFICATE_STATUS: VALID_REFUTATION\nCONFIDENCE: 82\n"
            "INDEPENDENT_ANSWER: B\nFIRST_FLAW: NONE",
            ("A", "B"),
        )
        self.assertEqual(parsed, ("VALID_REFUTATION", 82, "B", "NONE", None))
        self.assertEqual(
            parse_certificate_check_output("looks valid", ("A", "B"))[-1],
            "missing_required_field",
        )

    def test_v2_certificate_parser_keeps_leading_label_without_gold(self) -> None:
        parsed = parse_certificate_output_v2(
            "VERDICT: FALSIFIED\nCONFIDENCE: 85\nALTERNATIVE: (B) option text\n"
            "PREMISE: rule\nCHECK: calculation\nFAILURE: contradiction",
            ("A", "B"),
        )
        self.assertEqual(parsed[2], "B")
        self.assertIsNone(parsed[-1])

    def test_target_blind_checker_reconstructs_sets(self) -> None:
        parsed = parse_target_blind_check_output_v2(
            "LOGIC_STATUS: VALID\nCONFIDENCE: 91\nELIMINATED_OPTIONS: (A), C\n"
            "SUPPORTED_OPTIONS: B\nFIRST_INVALID_STEP: NONE",
            ("A", "B", "C"),
        )
        self.assertEqual(parsed, ("VALID", 91, ("A", "C"), ("B",), "NONE", None))
        invalid = parse_target_blind_check_output_v2(
            "LOGIC_STATUS: VALID\nCONFIDENCE: 91\nELIMINATED_OPTIONS: A\n"
            "SUPPORTED_OPTIONS: NONE\nFIRST_INVALID_STEP: arithmetic error",
            ("A", "B"),
        )
        self.assertEqual(invalid[-1], "valid_with_invalid_step")

    def test_target_and_claim_are_structurally_hidden_from_checker(self) -> None:
        question = _question("hidden")
        certificate_a = CounterexampleCertificate(
            "hidden", "secret-generator", "A", "FALSIFIED", 90, "B", "p", "c", "f"
        )
        certificate_b = CounterexampleCertificate(
            "hidden", "secret-generator", "B", "SURVIVES", 90, "A", "p", "c", "f"
        )
        prompt_a = build_target_blind_check_prompt_v2(question, certificate_a)
        prompt_b = build_target_blind_check_prompt_v2(question, certificate_b)
        self.assertEqual(prompt_a, prompt_b)
        self.assertNotIn("secret-generator", prompt_a)
        self.assertNotIn("Claimed verdict", prompt_a)
        self.assertEqual(
            reconstructed_check_status(certificate_a, "VALID", ("A",), ()),
            "VALID_REFUTATION",
        )

    def test_v2_generator_prompt_requires_two_sided_audit(self) -> None:
        prompt = build_certificate_prompt_v2(_question("q"), "A")
        self.assertIn("positive derivation", prompt)
        self.assertIn("concrete counterexample", prompt)
        self.assertNotIn("correct answer:", prompt.lower())

    def test_v3_witness_parser_enforces_set_semantics(self) -> None:
        parsed = parse_sealed_effect_witness_output_v3(
            "INVARIANT: an atomic constraint\n"
            "DERIVATION: applying it rules out two values\n"
            "BOUNDARY: applies under the stated definition\n"
            "ELIMINATED_OPTIONS: A, C\n"
            "SUPPORTED_OPTIONS: B\n"
            "CONFIDENCE: 87",
            ("A", "B", "C", "D"),
        )
        self.assertEqual(parsed[0], 87)
        self.assertEqual(parsed[4], ("A", "C"))
        self.assertEqual(parsed[5], ("B",))
        self.assertIsNone(parsed[-1])
        all_eliminated = parse_sealed_effect_witness_output_v3(
            "INVARIANT: rule\nDERIVATION: derivation\nBOUNDARY: boundary\n"
            "ELIMINATED_OPTIONS: A, B\nSUPPORTED_OPTIONS: NONE\nCONFIDENCE: 90",
            ("A", "B"),
        )
        self.assertEqual(all_eliminated[-1], "all_options_eliminated")

    def test_v3_candidate_expansion_is_deterministic(self) -> None:
        self.assertEqual(
            sealed_witness_candidate_fields("A", ("A", "B", "C"), ("A", "C"), ("B",)),
            ("FALSIFIED", "B"),
        )
        self.assertEqual(
            sealed_witness_candidate_fields("B", ("A", "B", "C"), ("A", "C"), ("B",)),
            ("SURVIVES", "B"),
        )

    def test_v3_generator_has_no_named_target_and_checker_hides_claim(self) -> None:
        question = _question("sealed")
        generator_prompt = build_sealed_effect_witness_prompt_v3(question)
        self.assertNotIn("candidate under inspection", generator_prompt.lower())
        self.assertNotIn("prior answer:", generator_prompt.lower())
        certificate_a = CounterexampleCertificate(
            "sealed",
            "secret-generator",
            "A",
            "FALSIFIED",
            90,
            "B",
            "invariant",
            "derivation",
            "boundary",
            witness_id="sealed::secret-generator",
            claimed_eliminated_options=("A",),
            claimed_supported_options=("B",),
            claim_was_sealed=True,
        )
        certificate_b = CounterexampleCertificate(
            "sealed",
            "secret-generator",
            "B",
            "SURVIVES",
            90,
            "B",
            "invariant",
            "derivation",
            "boundary",
            witness_id="sealed::secret-generator",
            claimed_eliminated_options=("A",),
            claimed_supported_options=("B",),
            claim_was_sealed=True,
        )
        prompt_a = build_sealed_effect_reconstruction_prompt_v3(question, certificate_a)
        prompt_b = build_sealed_effect_reconstruction_prompt_v3(question, certificate_b)
        self.assertEqual(prompt_a, prompt_b)
        self.assertNotIn("secret-generator", prompt_a)
        self.assertNotIn("claimed effect: a", prompt_a.lower())
        self.assertNotIn("claimed effect: b", prompt_a.lower())
        self.assertNotIn("confidence: 90", prompt_a.lower())

    def test_prompts_hide_identity_and_popularity(self) -> None:
        question = _question("q")
        certificate = CounterexampleCertificate(
            "q", "secret-generator", "A", "FALSIFIED", 90, "B", "rule", "check", "failure"
        )
        generator_prompt = build_certificate_prompt(question, "A")
        checker_prompt = build_certificate_check_prompt(question, certificate)
        self.assertNotIn("secret-generator", checker_prompt)
        self.assertNotIn("3 votes", generator_prompt + checker_prompt)
        self.assertNotIn("correct answer is", (generator_prompt + checker_prompt).lower())

    def test_open_option_rescue_uses_cross_checked_evidence(self) -> None:
        questions = [_question(f"q{index}") for index in range(12)]
        answers = {question.question_id: ("A" if index % 2 == 0 else "B") for index, question in enumerate(questions)}
        base = [
            BasePrediction(question.question_id, expert, "A")
            for question in questions
            for expert in ("generator", "checker", "weak")
        ]
        certificates = []
        checks = []
        correctness = {}
        environments = {}
        for index, question in enumerate(questions):
            answer = answers[question.question_id]
            environments[question.question_id] = f"env{index % 3}"
            for expert in ("generator", "checker", "weak"):
                correctness[(question.question_id, expert)] = answer == "A"
            for candidate in ("A", "B"):
                correctness[(question.question_id, candidate_label_key(candidate))] = candidate == answer
                verdict = "SURVIVES" if candidate == answer else "FALSIFIED"
                certificate = CounterexampleCertificate(
                    question.question_id,
                    "generator",
                    candidate,
                    verdict,
                    95,
                    answer,
                    "premise",
                    "check",
                    "NONE" if verdict == "SURVIVES" else "fatal",
                )
                certificates.append(certificate)
                checks.append(
                    CertificateCheck(
                        certificate.certificate_id,
                        question.question_id,
                        "generator",
                        "checker",
                        candidate,
                        "VALID_SUPPORT" if candidate == answer else "VALID_REFUTATION",
                        95,
                        answer,
                        "NONE",
                    )
                )
        labels = SourceTrainingLabels._from_source_adapter(
            "synthetic", "development", correctness, environments
        )
        model = CrossExaminedCertificateCourt(
            C3Variant("c3", regularization_c=10.0), seed=3
        ).fit(questions, base, certificates, checks, labels)
        target = _question("heldout")
        target_base = [BasePrediction("heldout", expert, "A") for expert in ("generator", "checker", "weak")]
        target_certificates = [
            CounterexampleCertificate("heldout", "generator", "A", "FALSIFIED", 95, "B", "p", "c", "fatal"),
            CounterexampleCertificate("heldout", "generator", "B", "SURVIVES", 95, "B", "p", "c", "NONE"),
        ]
        target_checks = [
            CertificateCheck(cert.certificate_id, "heldout", "generator", "checker", cert.candidate,
                             "VALID_REFUTATION" if cert.candidate == "A" else "VALID_SUPPORT", 95, "B", "NONE")
            for cert in target_certificates
        ]
        decision = model.predict([target], target_base, target_certificates, target_checks)[0]
        self.assertEqual(decision.answer, "B")
        self.assertTrue(decision.open_set_rescue)

    def test_evaluation_labels_cannot_fit(self) -> None:
        self.assertNotIn("labels", inspect.signature(CrossExaminedCertificateCourt.predict).parameters)
        with self.assertRaises(TypeError):
            CrossExaminedCertificateCourt(C3Variant("reject")).fit(
                [_question("q")],
                [BasePrediction("q", "expert", "A")],
                [],
                [],
                EvaluationLabels("synthetic", "test", {}),  # type: ignore[arg-type]
            )

    def test_generation_modules_do_not_import_label_readers(self) -> None:
        for module in (run_c3_certificates, run_c3_certificate_checks):
            source = inspect.getsource(module)
            self.assertNotIn("development_labels", source)
            self.assertNotIn("SourceTrainingLabels", source)
            self.assertNotIn("EvaluationLabels", source)

    def test_smoke_selection_round_robins_datasets(self) -> None:
        questions = [
            FalsificationQuestion(
                f"{dataset}{index}", dataset, f"{dataset}::env", "q", ("x", "y"), ("A", "B")
            )
            for dataset in ("alpha", "beta")
            for index in range(3)
        ]
        selected = _stratified_smoke_questions(questions, 4)
        self.assertEqual([row.dataset for row in selected], ["alpha", "beta", "alpha", "beta"])
        question_by_id = {row.question_id: row for row in questions}
        certificates = [
            CounterexampleCertificate(
                question.question_id,
                "generator",
                candidate,
                "INCONCLUSIVE",
                50,
                None,
                "p",
                "c",
                "uncertain",
            )
            for question in questions
            for candidate in question.option_labels
        ]
        selected_certificates = _stratified_smoke_certificates(
            certificates, question_by_id, 4
        )
        self.assertEqual(
            [question_by_id[row.question_id].dataset for row in selected_certificates],
            ["alpha", "beta", "alpha", "beta"],
        )

    def test_check_manifest_protocol_names_distinguish_v7_and_v8(self) -> None:
        def config(prompt_version: str) -> dict[str, object]:
            return {"check_generation": {"prompt_version": prompt_version}}

        self.assertEqual(
            _audit_protocol_name(config("blind_isolated_trace_audit_v7"), True),
            "isolated_trace_pointwise_v7",
        )
        self.assertEqual(
            _audit_protocol_name(
                config("commitment_conditioned_proof_audit_v8"), True
            ),
            "commitment_conditioned_proof_audit_v8",
        )
        self.assertEqual(
            _audit_protocol_name(config("blind_counterfactual_parity_v4"), True),
            "dual_orientation_pairwise",
        )
        self.assertEqual(
            _audit_protocol_name(
                config("commitment_conditioned_pair_audit_v8_ablation"), True
            ),
            "commitment_conditioned_pair_audit_v8_ablation",
        )
        self.assertEqual(
            _audit_protocol_name(config(CANDIDATE_VISIBLE_PROMPT_VERSION), True),
            "candidate_visible_commit_first_v8_control",
        )
        self.assertEqual(
            _audit_protocol_name(config(UNSEALED_PROMPT_VERSION), True),
            "unsealed_set_aware_v8_control",
        )

    def test_candidate_visible_and_unsealed_controls_change_only_declared_information(self) -> None:
        question = _question("prior-art-controls")
        certificate = CounterexampleCertificate(
            question_id=question.question_id,
            generator_id="generator",
            candidate="B",
            verdict="SURVIVES",
            confidence=90,
            alternative="B",
            premise="atomic rule",
            check="trace alpha",
            failure="first difference",
            witness_id="prior-art-controls::generator",
            claimed_supported_options=("B",),
            claim_was_sealed=True,
            counterfactual_pair=True,
            challenge_rule="atomic rule",
            trace_1="trace alpha",
            trace_2="trace beta",
            first_differing_step="secret difference",
            sealed_valid_trace=1,
            sealed_effect="SUPPORTS",
        )
        candidate_visible = build_candidate_visible_commit_first_prompt_v8_control(
            question,
            "atomic rule",
            "trace alpha",
            "trace beta",
            "secret difference",
            "trace_1",
            "private answer B",
            certificate,
        )
        self.assertIn("Candidate under verification: (B)", candidate_visible)
        self.assertIn("private answer B", candidate_visible)
        self.assertIn("trace alpha", candidate_visible)
        self.assertNotIn("trace beta", candidate_visible)
        self.assertNotIn("secret difference", candidate_visible)
        self.assertNotIn("Author's unsealed claim", candidate_visible)

        unsealed_valid = build_unsealed_set_aware_prompt_v8_control(
            question,
            "atomic rule",
            "trace alpha",
            "trace beta",
            "secret difference",
            "trace_1",
            "private answer B",
            certificate,
        )
        unsealed_invalid = build_unsealed_set_aware_prompt_v8_control(
            question,
            "atomic rule",
            "trace alpha",
            "trace beta",
            "secret difference",
            "trace_2",
            "private answer B",
            certificate,
        )
        self.assertIn("VALID; SUPPORTS option (B)", unsealed_valid)
        self.assertIn("INVALID; no signed option effect", unsealed_invalid)
        self.assertNotIn("trace beta", unsealed_valid)
        self.assertNotIn("trace alpha", unsealed_invalid)
        self.assertEqual(
            _guided_regex_for_prompt_version(CANDIDATE_VISIBLE_PROMPT_VERSION),
            _guided_regex_for_prompt_version(
                "commitment_conditioned_proof_audit_v8"
            ),
        )

    def test_pair_visible_ablation_has_matching_guided_decoding_schema(self) -> None:
        pattern = _guided_regex_for_prompt_version(
            "commitment_conditioned_pair_audit_v8_ablation"
        )
        output = (
            "PAIR_STATUS: ONE_VALID\n"
            "COUNTERTEST: substitute the boundary value into both traces\n"
            "COUNTERTEST_RESULT: ONE_SURVIVES_ONE_BREAKS\n"
            "RECOMPUTATION: trace one yields the required value\n"
            "COMMITMENT_RELATION: CONSISTENT\n"
            "VALID_TRACE: 1\n"
            "EFFECT: SUPPORTS\n"
            "OPTION: B\n"
            "CONFIDENCE: 91\n"
            "FIRST_FLAW: trace two reverses the operation"
        )
        self.assertIsNotNone(re.fullmatch(pattern, output))
        self.assertIsNone(
            re.fullmatch(
                pattern,
                "LOGIC_STATUS: VALID\nCONFIDENCE: 91\n"
                "ELIMINATED_OPTIONS: NONE\nSUPPORTED_OPTIONS: B\n"
                "FIRST_INVALID_STEP: NONE",
            )
        )

    def test_pair_visible_ablation_is_order_audited_and_seal_hidden(self) -> None:
        question = _question("pair-visible")
        canonical = build_commitment_conditioned_pair_audit_prompt_v8_ablation(
            question,
            "atomic rule",
            "trace alpha",
            "trace beta",
            "secret claimed difference",
            "canonical",
            "private frozen answer B",
        )
        mirrored = build_commitment_conditioned_pair_audit_prompt_v8_ablation(
            question,
            "atomic rule",
            "trace alpha",
            "trace beta",
            "secret claimed difference",
            "mirrored",
            "private frozen answer B",
        )
        self.assertLess(canonical.index("trace alpha"), canonical.index("trace beta"))
        self.assertLess(mirrored.index("trace beta"), mirrored.index("trace alpha"))
        self.assertIn("private frozen answer B", canonical)
        self.assertNotIn("secret claimed difference", canonical)
        self.assertNotIn("SEALED_OPTION", canonical)
        self.assertNotIn("gold answer is", canonical.lower())

    def test_pair_visible_ablation_parser_enforces_proof_branches(self) -> None:
        raw = (
            "PAIR_STATUS: ONE_VALID\n"
            "COUNTERTEST: substitute the boundary value into both traces\n"
            "COUNTERTEST_RESULT: ONE_SURVIVES_ONE_BREAKS\n"
            "RECOMPUTATION: trace one gives two while trace two gives three\n"
            "COMMITMENT_RELATION: CONSISTENT\n"
            "VALID_TRACE: 1\n"
            "EFFECT: SUPPORTS\n"
            "OPTION: B\n"
            "CONFIDENCE: 91\n"
            "FIRST_FLAW: trace two adds instead of subtracting"
        )
        parsed = parse_commitment_conditioned_pair_audit_output_v8_ablation(
            raw, ("A", "B")
        )
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(parsed.presented_valid_trace, 1)
        self.assertEqual(parsed.countertest_result, "ONE_SURVIVES_ONE_BREAKS")
        malformed = parse_commitment_conditioned_pair_audit_output_v8_ablation(
            raw.replace("FIRST_FLAW: trace two adds instead of subtracting", "FIRST_FLAW: NONE"),
            ("A", "B"),
        )
        self.assertEqual(malformed.parse_error, "missing_pair_proof_obligation")

    def test_isolated_trace_pair_features_require_one_valid_and_one_invalid(self) -> None:
        question = _question("isolated-pair")
        certificate = CounterexampleCertificate(
            question_id=question.question_id,
            generator_id="generator",
            candidate="B",
            verdict="SURVIVES",
            confidence=90,
            alternative="B",
            premise="rule",
            check="valid trace",
            failure="first difference",
            witness_id="isolated-pair::generator",
            claimed_supported_options=("B",),
            claim_was_sealed=True,
            counterfactual_pair=True,
            challenge_rule="rule",
            trace_1="valid trace",
            trace_2="invalid trace",
            first_differing_step="first difference",
            sealed_valid_trace=1,
            sealed_effect="SUPPORTS",
        )
        trace_1 = CertificateCheck(
            certificate.certificate_id,
            question.question_id,
            "generator",
            "checker",
            "B",
            "VALID_SUPPORT",
            90,
            "B",
            None,
            logic_status="VALID",
            supported_options=("B",),
            target_was_hidden=True,
            counterfactual_pair=True,
            orientation="trace_1",
            canonical_valid_trace=1,
            reconstructed_effect="SUPPORTS",
        )
        trace_2 = CertificateCheck(
            certificate.certificate_id,
            question.question_id,
            "generator",
            "checker",
            "B",
            "INVALID_SUPPORT",
            85,
            "B",
            "arithmetic error",
            logic_status="INVALID",
            target_was_hidden=True,
            counterfactual_pair=True,
            orientation="trace_2",
        )

        def features(use_parity: bool) -> dict[str, float]:
            court = CrossExaminedCertificateCourt(
                C3Variant("pair", use_counterfactual_parity=use_parity)
            )
            court.expert_ids_ = ("checker", "generator")
            court.expert_accuracy_ = {"checker": 0.5, "generator": 0.5}
            return court._candidate_features(
                question,
                "B",
                {question.question_id: {"checker": "B", "generator": "B"}},
                {(question.question_id, "B"): (certificate,)},
                {certificate.certificate_id: (trace_1, trace_2)},
            )

        paired = features(True)
        self.assertEqual(
            paired["check::isolated_one_valid_one_invalid::true"], 1.0
        )
        self.assertEqual(
            paired["check::isolated_pair_sealed_triple_match::true"], 1.0
        )
        ablated = features(False)
        self.assertNotIn("check::isolated_one_valid_one_invalid::true", ablated)
        self.assertNotIn("check::logic_status::INVALID", ablated)


if __name__ == "__main__":
    unittest.main()
