from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from bench_coe.innovation.aggregate_pre_pair_style import (
    _source_hashes,
    authenticate_completed_pre_pair,
    authenticate_pre_pair_models,
    expected_aggregate_rows,
)
from bench_coe.innovation.artifacts import sha256_file, write_json, write_jsonl
from bench_coe.innovation.blind_falsification_jury import FalsificationQuestion
from bench_coe.innovation.prepair_style import (
    PAIRWISE_ORIENTATIONS,
    aggregate_order_audited_pre_pair,
    build_pre_pair_pairwise_prompt,
    build_pre_pair_pointwise_prompt,
    parse_pre_pair_pairwise_output,
    parse_pre_pair_pointwise_output,
    rank_candidate_slate,
)
from bench_coe.innovation.run_pre_pair_style import (
    _validate_protocol,
    pre_pair_call_budget,
)


class PrePairStyleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = FalsificationQuestion(
            "q1",
            "synthetic",
            "synthetic::env",
            "Which value follows from the stated constraint?",
            ("one", "two", "three", "four"),
            ("A", "B", "C", "D"),
        )

    def test_candidate_slate_is_label_free_plurality_with_frozen_ties(self) -> None:
        experts = ("e0", "e1", "e2", "e3", "e4", "e5")
        slate = rank_candidate_slate(
            self.question,
            {"e0": "B", "e1": "A", "e2": "B", "e3": "A", "e4": "C", "e5": None},
            experts,
            max_challengers=2,
        )
        self.assertEqual(slate, ("B", "A", "C"))
        with self.assertRaises(ValueError):
            rank_candidate_slate(
                self.question, {"unknown": "A"}, experts, max_challengers=2
            )

    def test_prompts_and_anchored_parsers_preserve_pointwise_then_pairwise_boundary(self) -> None:
        pointwise = build_pre_pair_pointwise_prompt(self.question, "B")
        self.assertIn("Candidate under pointwise analysis: B. two", pointwise)
        self.assertNotIn("vote count is", pointwise.lower())
        self.assertNotIn("answer key:", pointwise.lower())
        self.assertEqual(
            parse_pre_pair_pointwise_output("ANALYSIS: the constraint yields two"),
            ("the constraint yields two", None),
        )
        self.assertEqual(
            parse_pre_pair_pointwise_output("prefix\nANALYSIS: invalid")[1],
            "format_mismatch",
        )
        pairwise = build_pre_pair_pairwise_prompt(
            self.question,
            "A",
            "B",
            "ANALYSIS: one is unsupported",
            "ANALYSIS: two follows",
        )
        self.assertIn("untrusted reasoning aids", pairwise)
        self.assertIn("LEFT pointwise analysis", pairwise)
        self.assertEqual(
            parse_pre_pair_pairwise_output(
                "REASON: the second derivation satisfies the constraint\nWINNER: RIGHT"
            ),
            ("RIGHT", "the second derivation satisfies the constraint", None),
        )
        self.assertEqual(
            parse_pre_pair_pairwise_output("WINNER: LEFT")[2], "format_mismatch"
        )

    @staticmethod
    def _pair_rows(
        question_id: str,
        models: tuple[str, ...],
        challenger: str,
        mapped_winner: str,
        primary: str = "A",
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model in models:
            for orientation in PAIRWISE_ORIENTATIONS:
                left, right = (
                    (primary, challenger)
                    if orientation == "primary_left"
                    else (challenger, primary)
                )
                winner = "LEFT" if left == mapped_winner else "RIGHT"
                rows.append(
                    {
                        "question_id": question_id,
                        "model": model,
                        "challenger": challenger,
                        "orientation": orientation,
                        "left_candidate": left,
                        "right_candidate": right,
                        "winner": winner,
                        "parse_error": None,
                    }
                )
        return rows

    def test_order_audit_uses_only_consistent_judgments_and_can_override_plurality(self) -> None:
        models = ("m0", "m1")
        rows = self._pair_rows("q1", models, "B", "B")
        rows.extend(self._pair_rows("q1", models, "C", "A"))
        result = aggregate_order_audited_pre_pair(
            self.question, ("A", "B", "C"), models, rows
        )
        self.assertEqual(result["prediction"], "B")
        self.assertEqual(
            result["fallback_reason"], "order_consistent_challenger_majority"
        )
        self.assertEqual(result["per_challenger"]["B"]["challenger_wins"], 2)

        inconsistent = self._pair_rows("q1", ("m0",), "B", "B")
        inconsistent[1]["winner"] = "RIGHT"
        fallback = aggregate_order_audited_pre_pair(
            self.question,
            ("A", "B"),
            ("m0",),
            inconsistent,
        )
        self.assertEqual(fallback["prediction"], "A")
        self.assertEqual(fallback["per_challenger"]["B"]["abstentions"], 1)

    def test_frozen_v8_adaptation_has_exact_42_call_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            c3_path = Path(temporary) / "c3.yaml"
            c3 = {
                "experts": [f"expert-{index}" for index in range(14)],
                "certificate_models": [f"judge-{index}" for index in range(4)],
                "checker_models": [f"judge-{index}" for index in range(4)],
            }
            c3_path.write_text(yaml.safe_dump(c3), encoding="utf-8")
            baseline = {
                "protocol_version": 1,
                "c3_config": str(c3_path),
                "c3_config_sha256": sha256_file(c3_path),
                "models": [f"judge-{index}" for index in range(4)],
                "candidate_selection": {
                    "ranking": "plurality_then_first_frozen_expert_then_option_order",
                    "max_challengers": 2,
                },
                "pointwise_prompt_version": "isolated_candidate_analysis_v1",
                "pairwise_prompt_version": "order_audited_transfer_v1",
                "pairwise_orientations": list(PAIRWISE_ORIENTATIONS),
                "calls_per_model_per_question": 7,
                "calls_per_question": 42,
                "data_policy": {
                    "generation_reads_labels": False,
                    "candidate_ranking_reads_labels": False,
                    "model_pool_equals_prefrozen_c3_generator_checker_pool": True,
                    "pointwise_candidates_are_isolated": True,
                    "target_labels_control_generation_or_aggregation": False,
                },
            }
            self.assertEqual(pre_pair_call_budget(c3, baseline), (7, 42))
            _validate_protocol(c3_path, c3, baseline)
            baseline["calls_per_question"] = 41
            with self.assertRaises(ValueError):
                _validate_protocol(c3_path, c3, baseline)

    def test_artifact_authenticator_replays_prompts_parsers_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            observable = run_root / "development_observables"
            observable.mkdir(parents=True)
            question = FalsificationQuestion(
                "q1",
                "synthetic",
                "synthetic::env",
                "Which value follows?",
                ("one", "two", "three"),
                ("A", "B", "C"),
            )
            question_path = observable / "questions.jsonl"
            base_path = observable / "base_predictions.jsonl"
            write_jsonl(
                question_path,
                [
                    {
                        "question_id": question.question_id,
                        "dataset": question.dataset,
                        "environment": question.environment,
                        "question": question.question,
                        "options": list(question.options),
                        "option_labels": list(question.option_labels),
                    }
                ],
            )
            experts = ("e0", "e1", "e2")
            write_jsonl(
                base_path,
                [
                    {
                        "question_id": "q1",
                        "dataset": "synthetic",
                        "expert_id": expert,
                        "prediction": candidate,
                        "response": f"response {candidate}",
                        "model_error": None,
                    }
                    for expert, candidate in zip(experts, ("A", "B", "C"), strict=True)
                ],
            )
            c3 = {
                "experts": list(experts),
                "certificate_models": ["judge"],
                "checker_models": ["judge"],
                "output_root": str(run_root),
            }
            c3_path = root / "c3.yaml"
            c3_path.write_text(yaml.safe_dump(c3), encoding="utf-8")
            baseline = {
                "protocol_version": 1,
                "c3_config": str(c3_path),
                "c3_config_sha256": sha256_file(c3_path),
                "run_root": str(run_root),
                "models": ["judge"],
                "candidate_selection": {
                    "ranking": "plurality_then_first_frozen_expert_then_option_order",
                    "max_challengers": 2,
                },
                "pointwise_prompt_version": "isolated_candidate_analysis_v1",
                "pairwise_prompt_version": "order_audited_transfer_v1",
                "pairwise_orientations": list(PAIRWISE_ORIENTATIONS),
                "calls_per_model_per_question": 7,
                "calls_per_question": 10,
                "acceptance": {
                    "minimum_pointwise_parse_rate": 0.9,
                    "minimum_pairwise_parse_rate": 0.9,
                    "minimum_order_consistent_pair_rate": 0.75,
                    "maximum_prompt_truncations": 0,
                    "require_primary_strictly_beats_both_methods": True,
                },
                "data_policy": {
                    "generation_reads_labels": False,
                    "candidate_ranking_reads_labels": False,
                    "model_pool_equals_prefrozen_c3_generator_checker_pool": True,
                    "pointwise_candidates_are_isolated": True,
                    "target_labels_control_generation_or_aggregation": False,
                },
            }
            baseline_path = root / "prepair.yaml"
            baseline_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
            slate = ("A", "B", "C")
            pointwise_rows = []
            pointwise_outputs = {}
            for candidate in slate:
                raw_output = f"ANALYSIS: evidence for {candidate}"
                pointwise_outputs[candidate] = raw_output
                raw_prompt = build_pre_pair_pointwise_prompt(question, candidate)
                pointwise_rows.append(
                    {
                        "question_id": "q1",
                        "dataset": "synthetic",
                        "environment": "synthetic::env",
                        "model": "judge",
                        "candidate": candidate,
                        "analysis": f"evidence for {candidate}",
                        "parse_error": None,
                        "raw_output": raw_output,
                        "raw_prompt_sha256": hashlib.sha256(
                            raw_prompt.encode("utf-8")
                        ).hexdigest(),
                        "prompt_sha256": "0" * 64,
                        "prompt_was_truncated": False,
                        "prompt_token_count": 10,
                        "model_latency_seconds": 0.1,
                    }
                )
            pairwise_rows = []
            for challenger, mapped_winner in (("B", "B"), ("C", "A")):
                for orientation in PAIRWISE_ORIENTATIONS:
                    left, right = (
                        ("A", challenger)
                        if orientation == "primary_left"
                        else (challenger, "A")
                    )
                    winner = "LEFT" if left == mapped_winner else "RIGHT"
                    raw_output = f"REASON: evidence favors {mapped_winner}\nWINNER: {winner}"
                    raw_prompt = build_pre_pair_pairwise_prompt(
                        question,
                        left,
                        right,
                        pointwise_outputs[left],
                        pointwise_outputs[right],
                    )
                    pairwise_rows.append(
                        {
                            "question_id": "q1",
                            "dataset": "synthetic",
                            "environment": "synthetic::env",
                            "model": "judge",
                            "challenger": challenger,
                            "orientation": orientation,
                            "left_candidate": left,
                            "right_candidate": right,
                            "winner": winner,
                            "reason": f"evidence favors {mapped_winner}",
                            "parse_error": None,
                            "raw_output": raw_output,
                            "raw_prompt_sha256": hashlib.sha256(
                                raw_prompt.encode("utf-8")
                            ).hexdigest(),
                            "prompt_sha256": "1" * 64,
                            "prompt_was_truncated": False,
                            "prompt_token_count": 10,
                            "model_latency_seconds": 0.1,
                        }
                    )
            model_dir = run_root / "prepair_style" / "models" / "judge"
            model_dir.mkdir(parents=True)
            pointwise_path = model_dir / "pointwise.jsonl"
            pairwise_path = model_dir / "pairwise.jsonl"
            prediction_path = model_dir / "predictions.jsonl"
            write_jsonl(pointwise_path, pointwise_rows)
            write_jsonl(pairwise_path, pairwise_rows)
            top2 = aggregate_order_audited_pre_pair(
                question, slate, ("judge",), pairwise_rows, challenger_limit=1
            )
            top3 = aggregate_order_audited_pre_pair(
                question, slate, ("judge",), pairwise_rows
            )
            write_jsonl(
                prediction_path,
                [
                    {
                        "question_id": "q1",
                        "dataset": "synthetic",
                        "environment": "synthetic::env",
                        "model": "judge",
                        "candidate_slate": list(slate),
                        "candidate_vote_counts": {"A": 1, "B": 1, "C": 1},
                        "top2": top2,
                        "budget_matched_top3": top3,
                    }
                ],
            )
            input_hashes = {
                str(c3_path): sha256_file(c3_path),
                str(baseline_path): sha256_file(baseline_path),
                str(question_path): sha256_file(question_path),
                str(base_path): sha256_file(base_path),
            }
            write_json(
                model_dir / "manifest.json",
                {
                    "status": "completed_label_free_pre_pair_style_model",
                    "protocol_version": 1,
                    "model": "judge",
                    "labels_read": False,
                    "questions": 1,
                    "selected_question_ids": ["q1"],
                    "candidate_slate_size": 3,
                    "calls_per_model_per_question": 7,
                    "actual_model_calls": 7,
                    "pointwise_calls": 3,
                    "parsed_pointwise_calls": 3,
                    "pairwise_calls": 4,
                    "parsed_pairwise_calls": 4,
                    "order_audited_pairs": 2,
                    "order_consistent_pairs": 2,
                    "truncated_prompts": 0,
                    "question_sha256": sha256_file(question_path),
                    "base_prediction_sha256": sha256_file(base_path),
                    "pointwise_sha256": sha256_file(pointwise_path),
                    "pairwise_sha256": sha256_file(pairwise_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    "c3_config_sha256": sha256_file(c3_path),
                    "baseline_config_sha256": sha256_file(baseline_path),
                    **_source_hashes(),
                    "environment": {"input_hashes": input_hashes},
                },
            )
            questions, authenticated_rows, quality = authenticate_pre_pair_models(
                c3_path, c3, baseline_path, baseline, run_root
            )
            self.assertEqual(len(authenticated_rows), 4)
            self.assertEqual(quality["nominal_total_calls_per_question"], 10)
            aggregate_rows = expected_aggregate_rows(
                questions, authenticated_rows, c3, baseline, run_root
            )
            aggregate_dir = run_root / "prepair_style" / "aggregate"
            aggregate_dir.mkdir(parents=True)
            aggregate_path = aggregate_dir / "predictions.jsonl"
            write_jsonl(aggregate_path, aggregate_rows)
            model_manifest_hash = sha256_file(model_dir / "manifest.json")
            model_hashes = quality["model_artifact_hashes"]
            self.assertEqual(model_hashes["judge"]["manifest_sha256"], model_manifest_hash)
            write_json(
                aggregate_dir / "manifest.json",
                {
                    "status": "completed_label_free_pre_pair_style_aggregate",
                    "labels_read": False,
                    "questions": 1,
                    "prediction_sha256": sha256_file(aggregate_path),
                    "c3_config_sha256": sha256_file(c3_path),
                    "baseline_config_sha256": sha256_file(baseline_path),
                    "model_artifact_hashes": model_hashes,
                    "aggregator_sha256": _source_hashes()["aggregator_sha256"],
                },
            )
            predictions, budgets, authenticated_quality = authenticate_completed_pre_pair(
                c3_path, c3, baseline_path, run_root
            )
            self.assertEqual(predictions["prepair_style_order_audited_top2"]["q1"], "B")
            self.assertEqual(budgets, {
                "prepair_style_order_audited_top2": 7.0,
                "prepair_style_budget_matched_top3": 10.0,
            })
            self.assertTrue(
                authenticated_quality["require_primary_strictly_beats_both_methods"]
            )

            tampered = json.loads(aggregate_path.read_text(encoding="utf-8").splitlines()[0])
            tampered["predictions"]["prepair_style_budget_matched_top3"] = "C"
            write_jsonl(aggregate_path, [tampered])
            with self.assertRaises(PermissionError):
                authenticate_completed_pre_pair(c3_path, c3, baseline_path, run_root)


if __name__ == "__main__":
    unittest.main()
