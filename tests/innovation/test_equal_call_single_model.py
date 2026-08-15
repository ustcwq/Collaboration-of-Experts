from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from bench_coe.innovation.artifacts import sha256_file, write_json, write_jsonl
from bench_coe.innovation.blind_falsification_jury import FalsificationQuestion
from bench_coe.innovation.equal_call_single_model import (
    aggregate_equal_call_answers,
    build_independent_solution_prompt,
    build_self_revision_prompt,
    parse_equal_call_answer,
)
from bench_coe.innovation.evaluate_c3_development import (
    _load_and_authenticate_equal_call_baselines,
)
from bench_coe.innovation.run_equal_call_single_model import _validate_protocol


class EqualCallSingleModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = FalsificationQuestion(
            "q1",
            "synthetic",
            "synthetic::env",
            "Which choice follows?",
            ("first", "second"),
            ("A", "B"),
        )

    def test_prompts_are_label_free_and_revision_marks_attempt_untrusted(self) -> None:
        initial = build_independent_solution_prompt(self.question)
        revision = build_self_revision_prompt(
            self.question, "REASON: earlier\nFINAL: B"
        )
        self.assertNotIn("correct answer", initial.lower())
        self.assertIn("untrusted evidence", revision)
        self.assertIn("FINAL: B", revision)

    def test_parser_is_anchored_and_checks_option_membership(self) -> None:
        self.assertEqual(
            parse_equal_call_answer("REASON: because\nFINAL: A", ("A", "B")),
            ("A", "because", None),
        )
        self.assertEqual(
            parse_equal_call_answer("REASON: because\nFINAL: C", ("A", "B"))[2],
            "answer_outside_option_set",
        )
        self.assertEqual(
            parse_equal_call_answer("prefix\nREASON: because\nFINAL: A", ("A", "B"))[2],
            "format_mismatch",
        )

    def test_aggregation_uses_first_sample_only_for_plurality_ties(self) -> None:
        self.assertEqual(
            aggregate_equal_call_answers(("B", "A", "A", "B", None), ("A", "B")),
            ("B", {"A": 2, "B": 2}),
        )
        self.assertEqual(
            aggregate_equal_call_answers((None, None), ("A", "B")),
            (None, {"A": 0, "B": 0}),
        )

    def test_v8_equal_call_budget_is_exactly_42(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "c3_v8.yaml"
            experts = [f"expert-{index}" for index in range(14)]
            pool = [f"audit-{index}" for index in range(4)]
            config = {
                "experts": experts,
                "certificate_models": pool,
                "checker_models": pool,
                "check_generation": {
                    "prompt_version": "commitment_conditioned_proof_audit_v8"
                },
            }
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            baseline = {
                "protocol_version": 1,
                "c3_config": str(path),
                "c3_config_sha256": sha256_file(path),
                "models": pool,
                "calls_per_question": 42,
                "methods": {
                    "self_consistency": {"samples": 42},
                    "self_revision": {
                        "initial_samples": 21,
                        "revisions_per_initial": 1,
                    },
                },
                "data_policy": {
                    "generation_reads_labels": False,
                    "model_pool_equals_prefrozen_c3_generator_checker_pool": True,
                    "development_accuracy_used_for_model_selection": False,
                    "certificate_or_check_outputs_used_for_model_selection": False,
                    "target_labels_control_generation_or_aggregation": False,
                },
            }
            _validate_protocol(path, config, baseline)
            baseline["calls_per_question"] = 41
            with self.assertRaises(ValueError):
                _validate_protocol(path, config, baseline)

    def test_authenticator_reparses_and_reaggregates_complete_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            observable = run_root / "development_observables"
            observable.mkdir(parents=True)
            question_path = observable / "questions.jsonl"
            write_jsonl(
                question_path,
                [
                    {
                        "question_id": self.question.question_id,
                        "dataset": self.question.dataset,
                        "environment": self.question.environment,
                        "question": self.question.question,
                        "options": list(self.question.options),
                        "option_labels": list(self.question.option_labels),
                    }
                ],
            )
            c3_config = {
                "experts": ["a", "b"],
                "certificate_models": ["a", "b"],
                "checker_models": ["a", "b"],
                "acceptance": {"require_equal_call_single_model_baselines": True},
            }
            c3_config_path = root / "c3.yaml"
            c3_config_path.write_text(yaml.safe_dump(c3_config), encoding="utf-8")
            baseline_config = {
                "protocol_version": 1,
                "c3_config_sha256": sha256_file(c3_config_path),
                "run_root": str(run_root),
                "models": ["a", "b"],
                "calls_per_question": 6,
                "methods": {
                    "self_consistency": {"samples": 6},
                    "self_revision": {
                        "initial_samples": 3,
                        "revisions_per_initial": 1,
                    },
                },
                "acceptance": {
                    "minimum_final_parse_rate": 0.9,
                    "minimum_valid_final_sample_fraction_per_question": 0.9,
                    "maximum_prompt_truncations": 0,
                },
                "data_policy": {
                    "generation_reads_labels": False,
                    "model_pool_equals_prefrozen_c3_generator_checker_pool": True,
                    "development_accuracy_used_for_model_selection": False,
                    "certificate_or_check_outputs_used_for_model_selection": False,
                    "target_labels_control_generation_or_aggregation": False,
                },
            }
            baseline_config_path = root / "equal.yaml"
            baseline_config_path.write_text(
                yaml.safe_dump(baseline_config), encoding="utf-8"
            )
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
            for method in baseline_config["methods"]:
                for model in baseline_config["models"]:
                    directory = run_root / "equal_call_single_model" / method / model
                    directory.mkdir(parents=True)
                    rows = []
                    initial_count = 6 if method == "self_consistency" else 3
                    for phase, count in (
                        ("initial", initial_count),
                        ("revision", 0 if method == "self_consistency" else 3),
                    ):
                        for index in range(count):
                            rows.append(
                                {
                                    "question_id": "q1",
                                    "dataset": "synthetic",
                                    "environment": "synthetic::env",
                                    "phase": phase,
                                    "sample_index": index,
                                    "parent_sample_index": (
                                        index if phase == "revision" else None
                                    ),
                                    "prediction": "A",
                                    "reason": f"reason {phase} {index}",
                                    "parse_error": None,
                                    "raw_output": (
                                        f"REASON: reason {phase} {index}\nFINAL: A"
                                    ),
                                    "prompt_sha256": "0" * 64,
                                    "prompt_was_truncated": False,
                                    "prompt_token_count": 10,
                                    "model_latency_seconds": 0.1,
                                }
                            )
                    sample_path = directory / "samples.jsonl"
                    prediction_path = directory / "predictions.jsonl"
                    write_jsonl(sample_path, rows)
                    final_count = initial_count if method == "self_consistency" else 3
                    write_jsonl(
                        prediction_path,
                        [
                            {
                                "question_id": "q1",
                                "dataset": "synthetic",
                                "environment": "synthetic::env",
                                "prediction": "A",
                                "vote_counts": {"A": final_count, "B": 0},
                                "valid_final_samples": final_count,
                                "final_samples": final_count,
                                "tie_breaking": "first_valid_sample_among_plurality_ties",
                            }
                        ],
                    )
                    input_hashes = {
                        str(c3_config_path): sha256_file(c3_config_path),
                        str(baseline_config_path): sha256_file(baseline_config_path),
                    }
                    write_json(
                        directory / "manifest.json",
                        {
                            "status": "completed_label_free_equal_call_single_model",
                            "protocol_version": 1,
                            "model": model,
                            "method": method,
                            "questions": 1,
                            "calls_per_question": 6,
                            "actual_model_calls": 6,
                            "samples": 6,
                            "parsed_samples": 6,
                            "final_samples": final_count,
                            "parsed_final_samples": final_count,
                            "truncated_prompts": 0,
                            "labels_read": False,
                            "question_sha256": sha256_file(question_path),
                            "sample_sha256": sha256_file(sample_path),
                            "prediction_sha256": sha256_file(prediction_path),
                            "c3_config_sha256": sha256_file(c3_config_path),
                            "baseline_config_sha256": sha256_file(
                                baseline_config_path
                            ),
                            **source_hashes,
                            "environment": {"input_hashes": input_hashes},
                        },
                    )
            predictions, budgets, quality = (
                _load_and_authenticate_equal_call_baselines(
                    c3_config_path,
                    c3_config,
                    baseline_config_path,
                    run_root,
                    {"q1": self.question},
                    question_path,
                )
            )
            self.assertEqual(len(predictions), 4)
            self.assertTrue(all(value["q1"] == "A" for value in predictions.values()))
            self.assertEqual(set(budgets.values()), {6.0})
            self.assertEqual(len(quality), 4)

            tampered = json.loads(
                (
                    run_root
                    / "equal_call_single_model"
                    / "self_consistency"
                    / "a"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            tampered["actual_model_calls"] = 5
            write_json(
                run_root
                / "equal_call_single_model"
                / "self_consistency"
                / "a"
                / "manifest.json",
                tampered,
            )
            with self.assertRaises(RuntimeError):
                _load_and_authenticate_equal_call_baselines(
                    c3_config_path,
                    c3_config,
                    baseline_config_path,
                    run_root,
                    {"q1": self.question},
                    question_path,
                )


if __name__ == "__main__":
    unittest.main()
