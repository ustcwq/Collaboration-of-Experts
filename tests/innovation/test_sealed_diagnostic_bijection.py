from __future__ import annotations

import unittest
import inspect

from bench_coe.innovation.blind_falsification_jury import FalsificationQuestion
from bench_coe.innovation import run_sdb_checks, run_sdb_probes
from bench_coe.innovation.audit_sdb_smoke import _pairwise_agreement
from bench_coe.innovation.sealed_diagnostic_bijection import (
    CandidatePairAssignment,
    assign_candidate_pairs,
    build_blind_probe_check_prompt,
    build_diagnostic_probe_prompt,
    parse_blind_probe_check_output,
    parse_diagnostic_probe_output,
    present_diagnostic_probe,
    presented_left_authored_outcome,
    reveal_probe_candidate,
)


def _question() -> FalsificationQuestion:
    return FalsificationQuestion(
        "q",
        "synthetic",
        "synthetic::env",
        "ORIGINAL_QUESTION_MARKER: which result follows?",
        ("alpha", "beta", "gamma", "delta"),
        ("A", "B", "C", "D"),
    )


def _assignment() -> CandidatePairAssignment:
    return CandidatePairAssignment(
        first="A",
        second="B",
        author_answer="A",
        reason="test",
    )


def _valid_probe_output() -> str:
    return (
        "PROBE: Given x squared equals four and x is positive, what is x?\n"
        "OUTCOME_1: x equals two\n"
        "OUTCOME_2: x equals negative two\n"
        "MAP_OUTCOME_1: A\n"
        "MAP_OUTCOME_2: B\n"
        "BRIDGE_1: the positive root implies the first assigned response\n"
        "BRIDGE_2: the negative root implies the second assigned response\n"
        "CONFIDENCE: 91"
    )


class SealedDiagnosticBijectionTests(unittest.TestCase):
    def test_pair_assignment_is_deterministic_and_spreads_challengers(self) -> None:
        authors = ("m1", "m2", "m3")
        answers = {"m1": "A", "m2": "A", "m3": "A", "other": "B"}
        first = assign_candidate_pairs(
            _question(), authors, answers, ("m1", "m2", "m3", "other")
        )
        second = assign_candidate_pairs(
            _question(), authors, answers, ("m1", "m2", "m3", "other")
        )
        self.assertEqual(first, second)
        self.assertEqual({row.first for row in first.values()}, {"A"})
        self.assertGreaterEqual(len({row.second for row in first.values()}), 2)

    def test_author_prompt_contains_only_supplied_private_trace(self) -> None:
        prompt = build_diagnostic_probe_prompt(
            _question(), "OWN_PRIVATE_MARKER", _assignment()
        )
        self.assertIn("OWN_PRIVATE_MARKER", prompt)
        self.assertNotIn("OTHER_PRIVATE_MARKER", prompt)
        self.assertIn("Assigned candidate X", prompt)
        self.assertIn("mapping are sealed from checkers", prompt)

    def test_probe_parser_requires_exact_bijection_and_no_visible_labels(self) -> None:
        parsed = parse_diagnostic_probe_output(
            _valid_probe_output(), _assignment(), _question().question
        )
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(
            (parsed.map_outcome_1, parsed.map_outcome_2), ("A", "B")
        )
        not_bijective = _valid_probe_output().replace("MAP_OUTCOME_2: B", "MAP_OUTCOME_2: A")
        self.assertEqual(
            parse_diagnostic_probe_output(
                not_bijective, _assignment(), _question().question
            ).parse_error,
            "mapping_is_not_assigned_bijection",
        )
        leaked = _valid_probe_output().replace(
            "PROBE: Given", "PROBE: Candidate A follows when"
        )
        self.assertEqual(
            parse_diagnostic_probe_output(
                leaked, _assignment(), _question().question
            ).parse_error,
            "visible_candidate_label_leak",
        )

    def test_probe_parser_accepts_only_complete_low_confidence_abstention(self) -> None:
        abstention = (
            "PROBE: NONE\nOUTCOME_1: NONE\nOUTCOME_2: NONE\n"
            "MAP_OUTCOME_1: NONE\nMAP_OUTCOME_2: NONE\n"
            "BRIDGE_1: NONE\nBRIDGE_2: NONE\nCONFIDENCE: 40"
        )
        parsed = parse_diagnostic_probe_output(
            abstention, _assignment(), _question().question
        )
        self.assertTrue(parsed.abstained)
        self.assertEqual(
            parse_diagnostic_probe_output(
                abstention.replace("CONFIDENCE: 40", "CONFIDENCE: 80"),
                _assignment(),
                _question().question,
            ).parse_error,
            "overconfident_abstention",
        )

    def test_post_commit_presentation_swaps_outcomes_and_mapping_together(self) -> None:
        parsed = parse_diagnostic_probe_output(
            _valid_probe_output(), _assignment(), _question().question
        )
        first = present_diagnostic_probe(parsed, 1)
        second = present_diagnostic_probe(parsed, 2)
        self.assertEqual((first.left_candidate, first.right_candidate), ("A", "B"))
        self.assertEqual((second.left_candidate, second.right_candidate), ("B", "A"))
        self.assertEqual(first.left_text, second.right_text)
        self.assertEqual(
            presented_left_authored_outcome(7, "q", "m"),
            presented_left_authored_outcome(7, "q", "m"),
        )

    def test_checker_prompt_cannot_depend_on_original_task_or_mapping(self) -> None:
        parsed = parse_diagnostic_probe_output(
            _valid_probe_output(), _assignment(), _question().question
        )
        presentation = present_diagnostic_probe(parsed, 1)
        prompt = build_blind_probe_check_prompt(
            presentation.probe, presentation.left_text, presentation.right_text
        )
        self.assertNotIn("ORIGINAL_QUESTION_MARKER", prompt)
        self.assertNotIn("MAP_OUTCOME", prompt)
        self.assertNotIn("BRIDGE_", prompt)
        self.assertNotIn("Assigned candidate", prompt)

    def test_check_parser_and_reveal_fail_closed(self) -> None:
        parsed_probe = parse_diagnostic_probe_output(
            _valid_probe_output(), _assignment(), _question().question
        )
        presentation = present_diagnostic_probe(parsed_probe, 2)
        check = parse_blind_probe_check_output(
            "OUTCOME: LEFT\nDERIVATION: the positive-root constraint fixes two\n"
            "CONFIDENCE: 88"
        )
        self.assertIsNone(check.parse_error)
        self.assertEqual(reveal_probe_candidate(check, presentation), ("B", "A"))
        uncertain = parse_blind_probe_check_output(
            "OUTCOME: UNCERTAIN\nDERIVATION: NONE\nCONFIDENCE: 40"
        )
        self.assertTrue(uncertain.uncertain)
        self.assertEqual(reveal_probe_candidate(uncertain, presentation), (None, None))
        self.assertEqual(
            parse_blind_probe_check_output(
                "OUTCOME: LEFT\nDERIVATION: NONE\nCONFIDENCE: 90"
            ).parse_error,
            "selected_outcome_without_derivation",
        )

    def test_generation_modules_have_no_label_reader_and_raw_generator_has_no_mapping(self) -> None:
        for module in (run_sdb_probes, run_sdb_checks):
            source = inspect.getsource(module)
            self.assertNotIn("development_labels", source)
            self.assertNotIn("SourceTrainingLabels", source)
            self.assertNotIn("EvaluationLabels", source)
        parameters = inspect.signature(run_sdb_checks._generate_raw_checks).parameters
        self.assertEqual(tuple(parameters), ("config", "checker", "views"))
        source = inspect.getsource(run_sdb_checks._generate_raw_checks)
        self.assertNotIn("sealed_left_candidate", source)
        self.assertNotIn("selected_candidate", source)

    def test_pairwise_agreement_counts_only_distinct_checkers(self) -> None:
        self.assertEqual(_pairwise_agreement(["LEFT", "LEFT", "RIGHT"]), (1, 3))
        self.assertEqual(_pairwise_agreement(["RIGHT", "RIGHT"]), (1, 1))
        self.assertEqual(_pairwise_agreement(["LEFT"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
