from __future__ import annotations

import copy
import hashlib
import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from bench_coe.innovation.aggregate_cfmad_style import (
    _aggregate_root,
    authenticate_cfmad_models,
    authenticate_completed_cfmad,
    expected_aggregate_rows,
)
from bench_coe.innovation.artifacts import sha256_file, write_json, write_jsonl
from bench_coe.innovation.blind_falsification_jury import FalsificationQuestion
from bench_coe.innovation.cfmad_style import (
    CFMAD_STYLE_METHOD,
    aggregate_cfmad_model_predictions,
    build_cfmad_abduction_prompt,
    build_cfmad_cot_prompt,
    build_cfmad_critic_prompt,
    build_cfmad_defense_prompt,
    build_cfmad_judge_prompt,
    parse_cfmad_abduction_output,
    parse_cfmad_critic_output,
    parse_cfmad_defense_output,
    parse_cfmad_final_output,
    select_primary_candidate,
    select_seeded_counterfactual_candidate,
)
from bench_coe.innovation.run_cfmad_style import (
    _load_yaml,
    _model_root,
    _source_hashes,
    _validate_protocol,
    cfmad_calls_per_model_per_question,
)


class CFMADStyleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = FalsificationQuestion(
            question_id="dataset::17",
            dataset="dataset",
            environment="dataset::subject",
            question="Which statement follows from the premise?",
            options=("alpha", "beta", "gamma", "delta"),
            option_labels=("A", "B", "C", "D"),
        )

    def test_prompts_preserve_preset_stance_pipeline(self) -> None:
        cot = build_cfmad_cot_prompt(self.question)
        abduction = build_cfmad_abduction_prompt(self.question, "B")
        critic = build_cfmad_critic_prompt(
            self.question, "B", "STANCE: B\nARGUMENT: because beta"
        )
        defense = build_cfmad_defense_prompt(
            self.question,
            "B",
            "STANCE: B\nARGUMENT: because beta",
            "CRITIQUE: beta conflicts with the premise",
        )
        self.assertIn("FINAL: <one of A, B, C, D>", cot)
        self.assertIn("PREDETERMINED STANCE: B. beta", abduction)
        self.assertIn("assistant may be wrong", critic.lower())
        self.assertIn("assigned option B", defense)
        self.assertNotIn("gold", " ".join((cot, abduction, critic, defense)).lower())

    def test_judge_sees_exactly_two_trajectories_but_may_choose_any_option(self) -> None:
        trajectories = [
            {
                "candidate": "A",
                "abduction": "STANCE: A\nARGUMENT: a",
                "critic": "CRITIQUE: not a",
                "defense": "DEFENSE: still a",
            },
            {
                "candidate": "C",
                "abduction": "STANCE: C\nARGUMENT: c",
                "critic": "CRITIQUE: not c",
                "defense": "DEFENSE: still c",
            },
        ]
        prompt = build_cfmad_judge_prompt(self.question, trajectories)
        self.assertEqual(prompt.count("OPTION A DEBATE:"), 1)
        self.assertEqual(prompt.count("OPTION C DEBATE:"), 1)
        self.assertIn("one not represented by a debate", prompt)
        self.assertIn("FINAL: <one of A, B, C, D>", prompt)

    def test_judge_rejects_duplicate_stances(self) -> None:
        row = {
            "candidate": "A",
            "abduction": "a",
            "critic": "c",
            "defense": "d",
        }
        with self.assertRaises(ValueError):
            build_cfmad_judge_prompt(self.question, [row, row])

    def test_strict_phase_parsers(self) -> None:
        self.assertEqual(
            parse_cfmad_final_output("REASON: decisive\nFINAL: C", self.question.option_labels),
            ("C", "decisive", None),
        )
        self.assertEqual(
            parse_cfmad_final_output("REASON: decisive\nFINAL: Z", self.question.option_labels),
            (None, None, "answer_outside_option_set"),
        )
        self.assertEqual(
            parse_cfmad_abduction_output("STANCE: B\nARGUMENT: beta", "B"),
            ("beta", None),
        )
        self.assertEqual(
            parse_cfmad_abduction_output("STANCE: A\nARGUMENT: alpha", "B"),
            (None, "stance_mismatch"),
        )
        self.assertEqual(parse_cfmad_critic_output("CRITIQUE: flaw"), ("flaw", None))
        self.assertEqual(parse_cfmad_defense_output("DEFENSE: reply"), ("reply", None))
        self.assertIsNotNone(parse_cfmad_critic_output("extra\nCRITIQUE: flaw")[1])

    def test_cot_primary_selection_is_deterministic_and_query_local(self) -> None:
        winner, counts, tie = select_primary_candidate(
            ["B", "C", "C", "B"], self.question.option_labels
        )
        self.assertEqual(winner, "B")
        self.assertEqual(counts, {"A": 0, "B": 2, "C": 2, "D": 0})
        self.assertEqual(tie, "first_cot_sample_among_plurality_ties")
        fallback, _, reason = select_primary_candidate(
            [None, None], self.question.option_labels
        )
        self.assertEqual(fallback, "A")
        self.assertEqual(reason, "option_order_when_all_cot_parses_fail")

    def test_counterfactual_selection_is_seeded_label_free_and_distinct(self) -> None:
        first = select_seeded_counterfactual_candidate(
            self.question.option_labels,
            "B",
            seed=7,
            question_id=self.question.question_id,
            model_id="model-a",
        )
        second = select_seeded_counterfactual_candidate(
            self.question.option_labels,
            "B",
            seed=7,
            question_id=self.question.question_id,
            model_id="model-a",
        )
        self.assertEqual(first, second)
        self.assertIn(first[0], {"A", "C", "D"})
        self.assertNotEqual(first[0], "B")
        self.assertEqual(len(first[2]), 64)

    def test_four_model_aggregation_has_fixed_tie_break(self) -> None:
        models = ("m1", "m2", "m3", "m4")
        prediction, counts, tie = aggregate_cfmad_model_predictions(
            {"m1": "C", "m2": "B", "m3": "B", "m4": "C"},
            models,
            self.question.option_labels,
        )
        self.assertEqual(prediction, "C")
        self.assertEqual(counts["B"], 2)
        self.assertEqual(counts["C"], 2)
        self.assertEqual(tie, "first_configured_model_among_plurality_ties")

    def test_config_is_bound_to_frozen_v8_and_has_ten_calls(self) -> None:
        c3_path = Path("configs/innovation/c3_development_v8.yaml")
        baseline_path = Path("configs/innovation/c3_cfmad_style_v8.yaml")
        c3 = _load_yaml(c3_path)
        baseline = _load_yaml(baseline_path)
        _validate_protocol(c3_path, c3, baseline)
        self.assertEqual(cfmad_calls_per_model_per_question(baseline), 10)
        self.assertEqual(baseline["ensemble_calls_per_question"], 40)

    def test_protocol_rejects_label_access(self) -> None:
        c3_path = Path("configs/innovation/c3_development_v8.yaml")
        c3 = _load_yaml(c3_path)
        baseline = _load_yaml(Path("configs/innovation/c3_cfmad_style_v8.yaml"))
        contaminated = copy.deepcopy(baseline)
        contaminated["data_policy"]["generation_reads_labels"] = True
        with self.assertRaises(PermissionError):
            _validate_protocol(c3_path, c3, contaminated)

    def test_authenticator_replays_every_stage_before_aggregation(self) -> None:
        c3_path = Path("configs/innovation/c3_development_v8.yaml")
        c3 = _load_yaml(c3_path)
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            question_path = run_root / "development_observables" / "questions.jsonl"
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
            baseline = _load_yaml(Path("configs/innovation/c3_cfmad_style_v8.yaml"))
            baseline["run_root"] = str(run_root)
            baseline_path = run_root / "cfmad.yaml"
            baseline_path.write_text(
                yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8"
            )
            model_predictions = ("A", "B", "B", "A")
            for model, final_prediction in zip(
                baseline["models"], model_predictions, strict=True
            ):
                directory = _model_root(run_root, model, 0)
                directory.mkdir(parents=True)
                cot_prompt = build_cfmad_cot_prompt(self.question)
                cot_rows = []
                for sample_index in range(3):
                    raw_output = "REASON: primary evidence\nFINAL: A"
                    cot_rows.append(
                        {
                            "question_id": self.question.question_id,
                            "dataset": self.question.dataset,
                            "environment": self.question.environment,
                            "model": model,
                            "phase": "cot",
                            "sample_index": sample_index,
                            "prediction": "A",
                            "reason": "primary evidence",
                            "parse_error": None,
                            "raw_output": raw_output,
                            "raw_prompt_sha256": hashlib.sha256(
                                cot_prompt.encode("utf-8")
                            ).hexdigest(),
                            "prompt_sha256": "synthetic",
                            "prompt_was_truncated": False,
                            "prompt_token_count": 1,
                            "model_latency_seconds": 0.0,
                        }
                    )
                primary, counts, tie = select_primary_candidate(
                    ["A", "A", "A"], self.question.option_labels
                )
                counterfactual, index, digest = select_seeded_counterfactual_candidate(
                    self.question.option_labels,
                    primary,
                    seed=int(baseline["seed"]),
                    question_id=self.question.question_id,
                    model_id=model,
                )
                assignment = {
                    "cot_predictions": ["A", "A", "A"],
                    "cot_vote_counts": counts,
                    "primary_candidate": primary,
                    "counterfactual_candidate": counterfactual,
                    "primary_tie_breaking": tie,
                    "counterfactual_index_within_remaining": index,
                    "counterfactual_selection_sha256": digest,
                }
                debate_rows = []
                debates = {}
                for stance_index, candidate in enumerate((primary, counterfactual)):
                    abduction = f"STANCE: {candidate}\nARGUMENT: support {candidate}"
                    critic = f"CRITIQUE: challenge {candidate}"
                    defense = f"DEFENSE: rebut challenge to {candidate}"
                    prompts = {
                        "abduction": build_cfmad_abduction_prompt(
                            self.question, candidate
                        ),
                        "critic": build_cfmad_critic_prompt(
                            self.question, candidate, abduction
                        ),
                        "defense": build_cfmad_defense_prompt(
                            self.question, candidate, abduction, critic
                        ),
                    }
                    outputs = {
                        "abduction": (abduction, f"support {candidate}"),
                        "critic": (critic, f"challenge {candidate}"),
                        "defense": (defense, f"rebut challenge to {candidate}"),
                    }
                    debates[stance_index] = {
                        "candidate": candidate,
                        "abduction": abduction,
                        "critic": critic,
                        "defense": defense,
                    }
                    for phase in ("abduction", "critic", "defense"):
                        raw_output, content = outputs[phase]
                        debate_rows.append(
                            {
                                "question_id": self.question.question_id,
                                "dataset": self.question.dataset,
                                "environment": self.question.environment,
                                "model": model,
                                "phase": phase,
                                "stance_index": stance_index,
                                "candidate": candidate,
                                "content": content,
                                "parse_error": None,
                                "raw_output": raw_output,
                                "raw_prompt_sha256": hashlib.sha256(
                                    prompts[phase].encode("utf-8")
                                ).hexdigest(),
                                "prompt_sha256": "synthetic",
                                "prompt_was_truncated": False,
                                "prompt_token_count": 1,
                                "model_latency_seconds": 0.0,
                            }
                        )
                trajectories = [debates[0], debates[1]]
                judge_prompt = build_cfmad_judge_prompt(
                    self.question, trajectories
                )
                judge_output = f"REASON: compare both debates\nFINAL: {final_prediction}"
                prediction_rows = [
                    {
                        "question_id": self.question.question_id,
                        "dataset": self.question.dataset,
                        "environment": self.question.environment,
                        "model": model,
                        "phase": "judge",
                        "prediction": final_prediction,
                        "reason": "compare both debates",
                        "parse_error": None,
                        "raw_output": judge_output,
                        "trajectories": trajectories,
                        **assignment,
                        "tie_breaking": tie,
                        "raw_prompt_sha256": hashlib.sha256(
                            judge_prompt.encode("utf-8")
                        ).hexdigest(),
                        "prompt_sha256": "synthetic",
                        "prompt_was_truncated": False,
                        "prompt_token_count": 1,
                        "model_latency_seconds": 0.0,
                    }
                ]
                cot_path = directory / "cot.jsonl"
                debate_path = directory / "debates.jsonl"
                prediction_path = directory / "predictions.jsonl"
                write_jsonl(cot_path, cot_rows)
                write_jsonl(debate_path, debate_rows)
                write_jsonl(prediction_path, prediction_rows)
                write_json(
                    directory / "manifest.json",
                    {
                        "status": "completed_label_free_cfmad_style_model",
                        "protocol_version": 1,
                        "model": model,
                        "questions": 1,
                        "selected_question_ids": [self.question.question_id],
                        "calls_per_model_per_question": 10,
                        "actual_model_calls": 10,
                        "phase_calls": {
                            "cot": 3,
                            "abduction": 2,
                            "critic": 2,
                            "defense": 2,
                            "judge": 1,
                        },
                        "parsed_phase_calls": {
                            "cot": 3,
                            "abduction": 2,
                            "critic": 2,
                            "defense": 2,
                            "judge": 1,
                        },
                        "distinct_stance_pairs": 1,
                        "truncated_prompts": 0,
                        "labels_read": False,
                        "base_predictions_read": False,
                        "certificate_or_check_outputs_read": False,
                        "question_sha256": sha256_file(question_path),
                        "cot_sha256": sha256_file(cot_path),
                        "debate_sha256": sha256_file(debate_path),
                        "prediction_sha256": sha256_file(prediction_path),
                        "c3_config_sha256": sha256_file(c3_path),
                        "baseline_config_sha256": sha256_file(baseline_path),
                        **_source_hashes(),
                        "environment": {
                            "input_hashes": {
                                str(c3_path): sha256_file(c3_path),
                                str(baseline_path): sha256_file(baseline_path),
                                str(question_path): sha256_file(question_path),
                            }
                        },
                    },
                )

            questions, model_rows, quality = authenticate_cfmad_models(
                c3_path, c3, baseline_path, baseline, run_root
            )
            aggregate_root = _aggregate_root(run_root, 0)
            aggregate_root.mkdir(parents=True)
            aggregate_path = aggregate_root / "predictions.jsonl"
            aggregate_rows = expected_aggregate_rows(
                questions, model_rows, baseline
            )
            write_jsonl(aggregate_path, aggregate_rows)
            aggregator_hash = hashlib.sha256(
                inspect.getsource(aggregate_cfmad_model_predictions).encode("utf-8")
            ).hexdigest()
            write_json(
                aggregate_root / "manifest.json",
                {
                    "status": "completed_label_free_cfmad_style_aggregate",
                    "labels_read": False,
                    "questions": 1,
                    "prediction_sha256": sha256_file(aggregate_path),
                    "c3_config_sha256": sha256_file(c3_path),
                    "baseline_config_sha256": sha256_file(baseline_path),
                    "model_artifact_hashes": quality["model_artifact_hashes"],
                    "aggregator_sha256": aggregator_hash,
                },
            )
            predictions, budgets, authenticated_quality = authenticate_completed_cfmad(
                c3_path, c3, baseline_path, run_root
            )
            self.assertEqual(
                predictions[CFMAD_STYLE_METHOD][self.question.question_id], "A"
            )
            self.assertEqual(budgets[CFMAD_STYLE_METHOD], 40.0)
            self.assertEqual(len(authenticated_quality["models"]), 4)


if __name__ == "__main__":
    unittest.main()
