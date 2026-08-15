from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench_coe.innovation.data import CacheAdapter
from bench_coe.innovation.prior_art_overlap import (
    CLASSIC_BASELINE_METHODS,
    EMBEDDING_RESPONSE_METHODS,
    FCRG_METHODS,
    RESPONSE_SELECTION_METHODS,
    cascade_selections,
    depth_weights,
    fcrg_ablation_selections,
    fcrg_score_matrix,
    fit_predict_prior_art_baselines,
    graph_variant,
    output_profile_matrix,
    repair_hop_sequence,
    response_embedding_observables,
)
from bench_coe.innovation.prior_art_targets import (
    authenticate_prediction_package,
    project_observable_pool,
)
from bench_coe.innovation.repair_simplification import RepairComponents
from bench_coe.innovation.run_prior_art_overlap import legacy_equivalence_diagnostics
from bench_coe.innovation.schema import EvaluationLabels, ObservableQueryBatch
from registry_fixture import write_registry


class PriorArtOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        gold = ["a", "b", "c", "a", "b", "c", "a", "b"]
        predictions = {
            "e1": ["a", "a", "c", "b", "b", "c", "a", "c"],
            "e2": ["a", "b", "a", "a", "c", "c", "b", "b"],
            "e3": ["b", "b", "c", "a", "b", "a", "a", "b"],
        }
        for expert, answers in predictions.items():
            model_dir = root / expert
            model_dir.mkdir()
            rows = []
            for index, (answer, prediction) in enumerate(zip(gold, answers, strict=True)):
                rows.append(
                    {
                        "id": str(index),
                        "subject": f"s{index % 4}",
                        "domain": f"d{index % 2}",
                        "difficulty": "hard" if index % 2 else "easy",
                        "question_type": "multiple-choice",
                        "answer": answer,
                        "prediction": prediction,
                        "response": ("maybe " if index == 3 and expert == "e1" else "") + prediction,
                        "is_correct": prediction == answer,
                        "model_latency_seconds": 0.1 + index / 100.0,
                    }
                )
            (model_dir / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")
        self.registry, self.registry_hash = write_registry(root, root, "toy", "validation", "multimodal")
        self.adapter = CacheAdapter.from_source_registry(
            root,
            "toy",
            "validation",
            "multimodal",
            {"e1": "f1", "e2": "f2", "e3": "f3"},
            ("e1", "e2", "e3"),
            self.registry,
            self.registry_hash,
        )
        self.batch = self.adapter.load_observables()
        self.labels = self.adapter.load_source_labels()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_output_profile_uses_query_local_relations_not_cluster_ids(self) -> None:
        relabeled = ObservableQueryBatch(
            dataset=self.batch.dataset,
            split=self.batch.split,
            modality=self.batch.modality,
            pool=self.batch.pool,
            records=tuple(
                replace(
                    record,
                    per_query_cluster_id=(
                        None
                        if record.per_query_cluster_id is None
                        else 100 + 7 * int(record.per_query_cluster_id)
                    ),
                )
                for record in self.batch.records
            ),
        )
        np.testing.assert_array_equal(output_profile_matrix(self.batch), output_profile_matrix(relabeled))

    def test_all_registered_prior_art_adaptations_predict_with_masks(self) -> None:
        train_ids = self.batch.question_ids[:6]
        target_ids = self.batch.question_ids[6:]
        bundle = fit_predict_prior_art_baselines(
            self.batch.subset(train_ids),
            self.labels.subset(train_ids),
            self.batch.subset(target_ids),
            neighbors=3,
            seed=17,
        )
        expected = set(CLASSIC_BASELINE_METHODS).union(
            set(RESPONSE_SELECTION_METHODS).difference(EMBEDDING_RESPONSE_METHODS)
        )
        expected.remove("oprs_robust_output_profile")
        expected.add("fast_global_best_single_call")
        self.assertEqual(set(bundle.selections), expected)
        for selections in bundle.selections.values():
            self.assertEqual({row.question_id for row in selections}, set(target_ids))
            for row in selections:
                self.assertEqual(set(row.observable_features["valid_mask"]), set(self.batch.pool.expert_ids))
                self.assertEqual(set(row.observable_features["missing_mask"]), set(self.batch.pool.expert_ids))
        for row in bundle.selections["knora_u_output_profile"]:
            self.assertTrue(set(row.expert_scores.values()).issubset({0.0, 1.0}))

    def test_minilm_adaptations_run_with_response_encoder(self) -> None:
        class StubEncoder:
            dimension = 4

            def encode_batch(self, batch: ObservableQueryBatch) -> np.ndarray:
                values = np.zeros(
                    (len(batch.question_ids), len(batch.pool.expert_ids), self.dimension),
                    dtype=np.float32,
                )
                for row_index, question_id in enumerate(batch.question_ids):
                    for col, record in enumerate(batch.for_question(question_id)):
                        raw = record.raw_output or ""
                        values[row_index, col] = np.asarray(
                            [
                                1.0,
                                float(len(raw)),
                                float(sum(ord(char) for char in raw) % 17),
                                float(col + 1),
                            ],
                            dtype=np.float32,
                        )
                norms = np.linalg.norm(values, axis=2, keepdims=True)
                return values / np.maximum(norms, 1e-12)

            def diagnostics(self) -> dict[str, object]:
                return {"model_id": "deterministic-test-encoder", "dimension": self.dimension}

        train_ids = self.batch.question_ids[:6]
        target_ids = self.batch.question_ids[6:]
        encoder = StubEncoder()
        bundle = fit_predict_prior_art_baselines(
            self.batch.subset(train_ids),
            self.labels.subset(train_ids),
            self.batch.subset(target_ids),
            neighbors=3,
            seed=17,
            response_encoder=encoder,
        )
        self.assertTrue(set(EMBEDDING_RESPONSE_METHODS).issubset(bundle.selections))
        for method in EMBEDDING_RESPONSE_METHODS:
            self.assertEqual(len(bundle.selections[method]), len(target_ids))
        embeddings = encoder.encode_batch(self.batch.subset(target_ids))
        features, profile, agreement = response_embedding_observables(
            self.batch.subset(target_ids), embeddings
        )
        self.assertEqual(features.shape, (2, 3, 4))
        self.assertEqual(profile.shape, (2, 6))
        self.assertEqual(agreement.shape, (3, 3))
        self.assertTrue(np.isfinite(features).all())

    def test_target_pool_projection_and_prediction_authentication(self) -> None:
        projected = project_observable_pool(self.batch, ("e1", "e3"))
        self.assertEqual(projected.pool.expert_ids, ("e1", "e3"))
        self.assertEqual(len(projected.records), 2 * len(self.batch.question_ids))
        with self.assertRaises(ValueError):
            project_observable_pool(self.batch, ("e1", "missing"))

        seed_dir = Path(self.temp.name) / "seed"
        prediction = seed_dir / "predictions" / "target" / "method.jsonl"
        prediction.parent.mkdir(parents=True)
        prediction.write_text('{"question_id":"q"}\n', encoding="utf-8")
        import hashlib

        digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
        manifest = {
            "prediction_paths": {"method": "predictions/target/method.jsonl"},
            "prediction_hashes_before_evaluation": {"method": digest},
        }
        authenticate_prediction_package(seed_dir, manifest)
        prediction.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            authenticate_prediction_package(seed_dir, manifest)

    def test_prior_art_fit_rejects_evaluation_labels(self) -> None:
        evaluation = EvaluationLabels(self.batch.dataset, self.batch.split, {})
        with self.assertRaises(TypeError):
            fit_predict_prior_art_baselines(
                self.batch.subset(self.batch.question_ids[:6]),
                evaluation,  # type: ignore[arg-type]
                self.batch.subset(self.batch.question_ids[6:]),
                neighbors=2,
                seed=7,
            )

    def test_graph_controls_have_declared_invariants(self) -> None:
        graph = np.asarray([[0.1, 0.2, 0.4], [0.5, 0.3, 0.7], [0.9, 0.8, 0.6]], dtype=float)
        no_self = graph_variant(graph, "no_self", 3)
        np.testing.assert_array_equal(np.diag(no_self), np.zeros(3))
        row = graph_variant(graph, "row_normalized", 3)
        col = graph_variant(graph, "column_normalized", 3)
        softmax = graph_variant(graph, "row_softmax", 3)
        np.testing.assert_allclose(row.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(col.sum(axis=0), np.ones(3))
        np.testing.assert_allclose(softmax.sum(axis=1), np.ones(3))
        randomized = graph_variant(graph, "random_edges", 11)
        for index in range(3):
            offdiag = [col for col in range(3) if col != index]
            np.testing.assert_allclose(
                np.sort(randomized[index, offdiag]), np.sort(graph[index, offdiag])
            )
        relabeled = graph_variant(graph, "degree_relabel", 19)
        np.testing.assert_allclose(np.sort(relabeled.sum(axis=1)), np.sort(graph.sum(axis=1)))
        np.testing.assert_allclose(np.sort(relabeled.sum(axis=0)), np.sort(graph.sum(axis=0)))

    def test_two_hop_weights_exactly_recover_legacy_graph_mass(self) -> None:
        np.testing.assert_allclose(depth_weights(2), np.asarray([0.25, 0.18]), atol=1e-15)
        support = np.asarray([[2 / 3, 1 / 3, 2 / 3]], dtype=float)
        uncertainty = np.asarray([[0.0, 0.2, 0.1]], dtype=float)
        graph = np.asarray([[0.2, 0.7, 0.4], [0.6, 0.1, 0.8], [0.5, 0.9, 0.3]], dtype=float)
        weights, hops = repair_hop_sequence(support, uncertainty, graph, max_hops=5)
        self.assertEqual(len(hops), 5)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(1))

    def test_all_fcrg_ablations_are_finite_and_depth2_equals_full(self) -> None:
        target = self.batch.subset(self.batch.question_ids[6:])
        rows = len(target.question_ids)
        cols = len(target.pool.expert_ids)
        support = np.asarray([[2 / 3, 1 / 3, 2 / 3], [1 / 3, 2 / 3, 2 / 3]], dtype=float)
        uncertainty = np.zeros((rows, cols), dtype=float)
        valid = np.ones((rows, cols), dtype=bool)
        cluster_ids = np.asarray(
            [
                [int(record.per_query_cluster_id) for record in target.for_question(question_id)]
                for question_id in target.question_ids
            ],
            dtype=int,
        )
        graph = np.asarray([[0.2, 0.7, 0.4], [0.6, 0.1, 0.8], [0.5, 0.9, 0.3]], dtype=float)
        failure, hops = repair_hop_sequence(support, uncertainty, graph, max_hops=2)
        components = RepairComponents(
            question_ids=target.question_ids,
            expert_ids=target.pool.expert_ids,
            local=np.asarray([[0.4, 0.5, 0.6], [0.6, 0.3, 0.5]], dtype=float),
            support=support,
            uncertainty=uncertainty,
            global_accuracy=np.asarray([0.5, 0.6, 0.7], dtype=float),
            repair_graph=graph,
            failure_weights=failure,
            hop1=hops[0],
            hop2=hops[1],
            valid_mask=valid,
            cluster_ids=cluster_ids,
            graph_mode="raw",
        )
        full, _ = fcrg_score_matrix("fcrg_full", components)
        depth2, _ = fcrg_score_matrix("fcrg_depth_2", components)
        np.testing.assert_allclose(full, depth2)
        selections, diagnostics = fcrg_ablation_selections(target, components, seed=23, device=None)
        self.assertEqual(set(selections), set(FCRG_METHODS))
        self.assertEqual(set(diagnostics), set(FCRG_METHODS))
        for values in selections.values():
            self.assertEqual(len(values), rows)
            self.assertTrue(all(np.isfinite(list(row.expert_scores.values())).all() for row in values))

    def test_cascade_trigger_depends_only_on_fast_output(self) -> None:
        train_ids = self.batch.question_ids[:6]
        target_ids = self.batch.question_ids[6:]
        bundle = fit_predict_prior_art_baselines(
            self.batch.subset(train_ids),
            self.labels.subset(train_ids),
            self.batch.subset(target_ids),
            neighbors=2,
            seed=5,
            include_mlp=False,
        )
        fast = bundle.selections["fast_global_best_single_call"]
        full = bundle.selections["knop_output_profile"]
        cascade, diagnostics = cascade_selections(self.batch.subset(target_ids), fast, full, threshold=1.0)
        self.assertEqual(len(cascade), len(target_ids))
        self.assertEqual(diagnostics["mean_nominal_calls"], 1.0)

    def test_legacy_equivalence_excludes_rows_with_missing_outputs(self) -> None:
        bundle = fit_predict_prior_art_baselines(
            self.batch.subset(self.batch.question_ids[:6]),
            self.labels.subset(self.batch.question_ids[:6]),
            self.batch,
            neighbors=2,
            seed=5,
            include_mlp=False,
        )
        modern = list(bundle.selections["global_best_posthoc"])
        legacy = list(modern)
        incomplete_id = self.batch.question_ids[1]
        incomplete_batch = ObservableQueryBatch(
            dataset=self.batch.dataset,
            split=self.batch.split,
            modality=self.batch.modality,
            pool=self.batch.pool,
            records=tuple(
                replace(
                    record,
                    valid_output=False,
                    normalized_answer=None,
                    per_query_cluster_id=None,
                    missing_reason="test_missing_output",
                )
                if record.question_id == incomplete_id and record.expert_id == "e3"
                else record
                for record in self.batch.records
            ),
        )
        index = next(i for i, row in enumerate(legacy) if row.question_id == incomplete_id)
        legacy[index] = replace(legacy[index], normalized_answer="legacy-missing-cluster")
        diagnostics = legacy_equivalence_diagnostics(incomplete_batch, modern, legacy)
        self.assertEqual(diagnostics["complete_answer_mismatch_count"], 0)
        self.assertEqual(diagnostics["incomplete_answer_difference_count"], 1)
        self.assertEqual(
            diagnostics["incomplete_answer_difference_ids"], [incomplete_id]
        )

        complete_id = self.batch.question_ids[0]
        index = next(i for i, row in enumerate(legacy) if row.question_id == complete_id)
        legacy[index] = replace(legacy[index], normalized_answer="complete-mismatch")
        diagnostics = legacy_equivalence_diagnostics(incomplete_batch, modern, legacy)
        self.assertEqual(diagnostics["complete_answer_mismatch_ids"], [complete_id])


if __name__ == "__main__":
    unittest.main()
