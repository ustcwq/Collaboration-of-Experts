from __future__ import annotations

import unittest

from bench_coe.innovation.blind_falsification_jury import FalsificationQuestion
from bench_coe.innovation.sealed_counterfactual_parity import (
    build_blind_isolated_trace_audit_prompt_v7,
    build_commitment_conditioned_proof_audit_prompt_v8,
    bounded_private_response,
    build_blind_counterfactual_parity_prompt_v4,
    build_sealed_counterfactual_challenge_prompt_v4,
    canonical_trace_index,
    combine_isolated_trace_audits,
    counterfactual_trace_slot,
    effect_option_sets,
    parse_blind_counterfactual_parity_output_v4,
    parse_blind_isolated_trace_audit_output_v7,
    parse_committed_counterfactual_challenge_output_v6,
    parse_commitment_conditioned_proof_audit_output_v8,
    parse_hardened_counterfactual_challenge_output_v5,
    parse_sealed_counterfactual_challenge_output_v4,
    permute_committed_counterfactual_challenge,
    sealed_triple_matches,
)


def _question() -> FalsificationQuestion:
    return FalsificationQuestion(
        "q", "synthetic", "env", "Which result follows?", ("one", "two"), ("A", "B")
    )


class SealedCounterfactualParityTests(unittest.TestCase):
    def test_trace_slot_is_deterministic_and_binary(self) -> None:
        slot = counterfactual_trace_slot(7, "q", "model")
        self.assertIn(slot, (1, 2))
        self.assertEqual(slot, counterfactual_trace_slot(7, "q", "model"))

    def test_generator_prompt_keeps_private_trace_out_of_checker_prompt(self) -> None:
        generator_prompt = build_sealed_counterfactual_challenge_prompt_v4(
            _question(), "PRIVATE_STAGE0_MARKER answer is A", 2
        )
        checker_prompt = build_blind_counterfactual_parity_prompt_v4(
            _question(), "rule", "valid derivation", "invalid derivation", "step", "canonical"
        )
        self.assertIn("PRIVATE_STAGE0_MARKER", generator_prompt)
        self.assertNotIn("PRIVATE_STAGE0_MARKER", checker_prompt)
        self.assertNotIn("claimed valid trace", checker_prompt.lower().split("are hidden", 1)[1])
        self.assertIn("Required valid-trace slot", generator_prompt)

    def test_challenge_parser_enforces_slot_single_effect_and_no_label_leak(self) -> None:
        text = (
            "RULE: conservation fixes the total\n"
            "TRACE_1: subtracting the known part gives two\n"
            "TRACE_2: adding the known part gives two\n"
            "FIRST_DIFFERING_STEP: subtract versus add\n"
            "SEALED_VALID_TRACE: 1\n"
            "SEALED_EFFECT: SUPPORTS\n"
            "SEALED_OPTION: B\n"
            "CONFIDENCE: 88"
        )
        parsed = parse_sealed_counterfactual_challenge_output_v4(
            text, ("A", "B"), 1
        )
        self.assertIsNone(parsed.parse_error)
        self.assertEqual((parsed.valid_trace, parsed.effect, parsed.option), (1, "SUPPORTS", "B"))
        self.assertEqual(effect_option_sets(parsed.effect, parsed.option), ((), ("B",)))
        self.assertEqual(
            parse_sealed_counterfactual_challenge_output_v4(text, ("A", "B"), 2).parse_error,
            "valid_trace_slot_mismatch",
        )
        leaked = text.replace("RULE: conservation fixes the total", "RULE: option B is correct")
        self.assertEqual(
            parse_sealed_counterfactual_challenge_output_v4(leaked, ("A", "B"), 1).parse_error,
            "visible_option_label_leak",
        )

    def test_challenge_parser_accepts_only_complete_low_confidence_abstention(self) -> None:
        text = (
            "RULE: insufficient evidence\nTRACE_1: no rigorous local trace\n"
            "TRACE_2: no counterfactual pair\nFIRST_DIFFERING_STEP: unavailable\n"
            "SEALED_VALID_TRACE: NONE\nSEALED_EFFECT: NONE\n"
            "SEALED_OPTION: NONE\nCONFIDENCE: 40"
        )
        self.assertTrue(
            parse_sealed_counterfactual_challenge_output_v4(text, ("A", "B"), 1).abstained
        )
        self.assertEqual(
            parse_sealed_counterfactual_challenge_output_v4(
                text.replace("CONFIDENCE: 40", "CONFIDENCE: 80"), ("A", "B"), 1
            ).parse_error,
            "overconfident_abstention",
        )

    def test_parity_parser_and_orientation_mapping_fail_closed(self) -> None:
        output = (
            "PAIR_STATUS: ONE_VALID\nVALID_TRACE: 2\nEFFECT: ELIMINATES\n"
            "OPTION: A\nCONFIDENCE: 91\nFIRST_FLAW: sign was reversed"
        )
        parsed = parse_blind_counterfactual_parity_output_v4(output, ("A", "B"))
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(canonical_trace_index(parsed.presented_valid_trace, "canonical"), 2)
        self.assertEqual(canonical_trace_index(parsed.presented_valid_trace, "mirrored"), 1)
        invalid = output.replace("FIRST_FLAW: sign was reversed", "FIRST_FLAW: NONE")
        self.assertEqual(
            parse_blind_counterfactual_parity_output_v4(invalid, ("A", "B")).parse_error,
            "one_valid_without_rejected_trace_flaw",
        )

    def test_mirrored_prompt_only_swaps_trace_order(self) -> None:
        canonical = build_blind_counterfactual_parity_prompt_v4(
            _question(), "rule", "TRACE_ALPHA", "TRACE_BETA", "step", "canonical"
        )
        mirrored = build_blind_counterfactual_parity_prompt_v4(
            _question(), "rule", "TRACE_ALPHA", "TRACE_BETA", "step", "mirrored"
        )
        self.assertLess(canonical.index("TRACE_ALPHA"), canonical.index("TRACE_BETA"))
        self.assertLess(mirrored.index("TRACE_BETA"), mirrored.index("TRACE_ALPHA"))

    def test_private_response_is_bounded_deterministically(self) -> None:
        response = "x" * 5000
        bounded = bounded_private_response(response)
        self.assertIn("[TRUNCATED_MIDDLE]", bounded)
        self.assertEqual(bounded, bounded_private_response(response))
        self.assertLessEqual(len(bounded), 2400)

    def test_sealed_triple_requires_trace_effect_and_option_identity(self) -> None:
        self.assertTrue(
            sealed_triple_matches(
                2, "SUPPORTS", (), ("B",), 2, "SUPPORTS", (), ("B",)
            )
        )
        self.assertFalse(
            sealed_triple_matches(
                2, "SUPPORTS", (), ("B",), 1, "SUPPORTS", (), ("B",)
            )
        )
        self.assertFalse(
            sealed_triple_matches(
                2, "SUPPORTS", (), ("B",), 2, "ELIMINATES", ("B",), ()
            )
        )

    def test_v5_label_firewall_allows_lowercase_math_but_not_option_ids(self) -> None:
        output = (
            "RULE: applying the operator to p(x) preserves one term\n"
            "TRACE_1: p(x) plus x times p-prime(x) remains\n"
            "TRACE_2: p(x) minus x times p-prime(x) remains\n"
            "FIRST_DIFFERING_STEP: plus versus minus for p(x)\n"
            "SEALED_VALID_TRACE: 1\nSEALED_EFFECT: SUPPORTS\n"
            "SEALED_OPTION: B\nCONFIDENCE: 90"
        )
        parsed = parse_hardened_counterfactual_challenge_output_v5(
            output, ("A", "B"), 1
        )
        self.assertIsNone(parsed.parse_error)
        leaked = output.replace(
            "RULE: applying the operator to p(x) preserves one term",
            "RULE: option B preserves one term",
        )
        self.assertEqual(
            parse_hardened_counterfactual_challenge_output_v5(
                leaked, ("A", "B"), 1
            ).parse_error,
            "visible_option_label_leak",
        )

    def test_v6_commit_then_permute_swaps_only_after_parse(self) -> None:
        output = (
            "RULE: p(x) follows one checked identity\n"
            "TRACE_1: p(x) uses the wrong sign\n"
            "TRACE_2: p(x) uses the correct sign\n"
            "FIRST_DIFFERING_STEP: wrong sign versus correct sign\n"
            "SEALED_VALID_TRACE: 2\nSEALED_EFFECT: SUPPORTS\n"
            "SEALED_OPTION: B\nCONFIDENCE: 90"
        )
        committed = parse_committed_counterfactual_challenge_output_v6(
            output, ("A", "B")
        )
        self.assertEqual(committed.valid_trace, 2)
        presented, swapped = permute_committed_counterfactual_challenge(
            committed, 1
        )
        self.assertTrue(swapped)
        self.assertEqual(presented.valid_trace, 1)
        self.assertEqual(presented.trace_1, committed.trace_2)
        self.assertEqual(presented.trace_2, committed.trace_1)
        unchanged, swapped = permute_committed_counterfactual_challenge(
            committed, 2
        )
        self.assertFalse(swapped)
        self.assertEqual(unchanged, committed)

    def test_v7_isolated_prompt_never_exposes_sibling_trace_or_difference(self) -> None:
        prompt = build_blind_isolated_trace_audit_prompt_v7(
            _question(), "RULE_MARKER", "TRACE_ALPHA", "TRACE_BETA", "DIFF_MARKER", "trace_1"
        )
        self.assertIn("RULE_MARKER", prompt)
        self.assertIn("TRACE_ALPHA", prompt)
        self.assertNotIn("TRACE_BETA", prompt)
        self.assertNotIn("DIFF_MARKER", prompt)
        self.assertIn("whether a pair exists", prompt)

    def test_v7_isolated_parser_enforces_status_dependent_contract(self) -> None:
        valid = parse_blind_isolated_trace_audit_output_v7(
            "TRACE_STATUS: VALID\nEFFECT: SUPPORTS\nOPTION: B\nCONFIDENCE: 91\n"
            "FLAW_CODE: NONE\nFLAW_DETAIL: NONE",
            ("A", "B"),
        )
        invalid = parse_blind_isolated_trace_audit_output_v7(
            "TRACE_STATUS: INVALID\nEFFECT: NONE\nOPTION: NONE\nCONFIDENCE: 88\n"
            "FLAW_CODE: ARITHMETIC\nFLAW_DETAIL: the sign is reversed",
            ("A", "B"),
        )
        self.assertIsNone(valid.parse_error)
        self.assertIsNone(invalid.parse_error)
        combined = combine_isolated_trace_audits(invalid, valid)
        self.assertEqual(combined.pair_status, "ONE_VALID")
        self.assertEqual(combined.presented_valid_trace, 2)
        self.assertEqual((combined.effect, combined.option), ("SUPPORTS", "B"))

        contradictory = parse_blind_isolated_trace_audit_output_v7(
            "TRACE_STATUS: VALID\nEFFECT: SUPPORTS\nOPTION: B\nCONFIDENCE: 91\n"
            "FLAW_CODE: ARITHMETIC\nFLAW_DETAIL: the sign is reversed",
            ("A", "B"),
        )
        self.assertEqual(contradictory.parse_error, "invalid_valid_branch")
        overconfident = parse_blind_isolated_trace_audit_output_v7(
            "TRACE_STATUS: INCONCLUSIVE\nEFFECT: NONE\nOPTION: NONE\nCONFIDENCE: 80\n"
            "FLAW_CODE: UNCERTAIN\nFLAW_DETAIL: premise cannot be checked",
            ("A", "B"),
        )
        self.assertEqual(overconfident.parse_error, "invalid_inconclusive_branch")

    def test_v8_proof_audit_uses_private_commitment_but_hides_sibling(self) -> None:
        prompt = build_commitment_conditioned_proof_audit_prompt_v8(
            _question(), "RULE_MARKER", "TRACE_ALPHA", "TRACE_BETA", "DIFF_MARKER",
            "trace_2", "PRIVATE_COMMITMENT_MARKER"
        )
        self.assertIn("PRIVATE_COMMITMENT_MARKER", prompt)
        self.assertIn("TRACE_BETA", prompt)
        self.assertNotIn("TRACE_ALPHA", prompt)
        self.assertNotIn("DIFF_MARKER", prompt)

    def test_v8_proof_parser_requires_countertest_and_consistent_branch(self) -> None:
        output = (
            "TRACE_STATUS: INVALID\n"
            "COUNTERTEST: substitute the stated value into the rule\n"
            "COUNTERTEST_RESULT: BREAKS\n"
            "RECOMPUTATION: the left side equals three, not four\n"
            "COMMITMENT_RELATION: CONFLICTS\n"
            "EFFECT: NONE\nOPTION: NONE\nCONFIDENCE: 92\n"
            "FLAW_CODE: ARITHMETIC\nFLAW_DETAIL: the final addition is off by one"
        )
        parsed = parse_commitment_conditioned_proof_audit_output_v8(
            output, ("A", "B")
        )
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(parsed.trace_status, "INVALID")
        missing = output.replace(
            "COUNTERTEST: substitute the stated value into the rule",
            "COUNTERTEST: NONE",
        )
        self.assertEqual(
            parse_commitment_conditioned_proof_audit_output_v8(
                missing, ("A", "B")
            ).parse_error,
            "missing_proof_obligation",
        )


if __name__ == "__main__":
    unittest.main()
