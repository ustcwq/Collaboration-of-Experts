from __future__ import annotations

import unittest

from bench_coe.innovation.gpqa_long_reasoning import (
    apply_semantic_reasoning_overrides,
    extract_explicit_answer,
    semantic_answer_by_query,
    source_best_model,
)
from bench_coe.innovation.run_gpqa_label_free_inference import _finalizer_prompt
from bench_coe.innovation.schema import Selection


class GPQALongReasoningTests(unittest.TestCase):
    def test_explicit_final_answer_only(self) -> None:
        self.assertEqual(extract_explicit_answer("Reasoning about (B). Final answer: (C)"), "C")
        self.assertEqual(extract_explicit_answer("Answer: (A)\nLater correction: answer is D"), "D")
        self.assertEqual(extract_explicit_answer("Therefore \\boxed{B}"), "B")
        self.assertIsNone(extract_explicit_answer("Reasoning mentions (B) but is unfinished"))

    def test_source_best_is_deterministic(self) -> None:
        self.assertEqual(
            source_best_model(
                [{"model": "b", "accuracy": 0.7}, {"model": "a", "accuracy": 0.7}]
            ),
            ("a", 0.7),
        )

    def test_semantic_answer_maps_across_permutations(self) -> None:
        semantic, audit = semantic_answer_by_query(
            [
                {
                    "config": "diamond",
                    "base_question_id": 7,
                    "record_id": "r1",
                    "prediction": "B",
                    "options": ["x", "right", "y", "z"],
                }
            ]
        )
        base = [Selection("q1", 0, "old", "a", {"0": 1.0}, {"old": 1.0}, None, {})]
        rows, counts = apply_semantic_reasoning_overrides(
            base,
            {
                "q1": {
                    "config": "diamond",
                    "base_question_id": 7,
                    "record_id": "r1",
                    "options": ["z", "y", "x", "right"],
                }
            },
            semantic,
            model_id="reasoner",
            inference_manifest_sha256="abc",
        )
        self.assertEqual(rows[0].normalized_answer, "d")
        self.assertEqual(counts["overridden"], 1)
        self.assertEqual(audit["semantic_queries"], 1)
        self.assertFalse(rows[0].observable_features["gpqa_long_reasoning_uses_target_labels"])
        self.assertTrue(rows[0].observable_features["gpqa_query_local_answer_identity"])

    def test_same_record_in_different_configs_remains_query_local(self) -> None:
        inference = [
            {
                "config": "diamond",
                "base_question_id": 7,
                "record_id": "r1",
                "prediction": "A",
                "options": ["diamond answer", "x", "y", "z"],
            },
            {
                "config": "main",
                "base_question_id": 11,
                "record_id": "r1",
                "prediction": "B",
                "options": ["x", "main answer", "y", "z"],
            },
        ]
        semantic, audit = semantic_answer_by_query(inference)
        base = [
            Selection("q1", 0, "old", "a", {"0": 1.0}, {"old": 1.0}, None, {}),
            Selection("q2", 0, "old", "a", {"0": 1.0}, {"old": 1.0}, None, {}),
        ]
        metadata = {
            "q1": {
                "config": "diamond",
                "base_question_id": 7,
                "record_id": "r1",
                "options": ["x", "diamond answer", "y", "z"],
            },
            "q2": {
                "config": "main",
                "base_question_id": 11,
                "record_id": "r1",
                "options": ["x", "y", "main answer", "z"],
            },
        }
        rows, counts = apply_semantic_reasoning_overrides(
            base,
            metadata,
            semantic,
            model_id="reasoner",
            inference_manifest_sha256="abc",
        )
        self.assertEqual([row.normalized_answer for row in rows], ["b", "c"])
        self.assertEqual(counts["overridden"], 2)
        self.assertEqual(audit["semantic_queries"], 2)

    def test_duplicate_query_identity_is_rejected(self) -> None:
        inference = [
            {
                "config": "diamond",
                "base_question_id": 7,
                "prediction": "A",
                "options": ["a", "b", "c", "d"],
            },
            {
                "config": "diamond",
                "base_question_id": 7,
                "prediction": "B",
                "options": ["a", "b", "c", "d"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicate query identity"):
            semantic_answer_by_query(inference)

    def test_invalid_prediction_falls_back(self) -> None:
        base = [Selection("q1", 0, "old", "a", {"0": 1.0}, {"old": 1.0}, None, {})]
        rows, counts = apply_semantic_reasoning_overrides(
            base,
            {
                "q1": {
                    "config": "diamond",
                    "base_question_id": 7,
                    "record_id": "r1",
                    "options": ["x", "y", "z", "w"],
                }
            },
            {},
            model_id="reasoner",
            inference_manifest_sha256="abc",
        )
        self.assertEqual(rows, base)
        self.assertEqual(counts["fallback_missing_prediction"], 1)

    def test_finalizer_fallback_prompt_preserves_reasoning(self) -> None:
        class Tokenizer:
            chat_template = None

        prompt = _finalizer_prompt(Tokenizer(), "question", "worked solution")
        self.assertIn("question", prompt)
        self.assertIn("worked solution", prompt)
        self.assertIn("exactly one", prompt)


if __name__ == "__main__":
    unittest.main()
