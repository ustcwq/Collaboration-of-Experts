from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bench_coe.innovation.artifacts import seed_gpu_map
from bench_coe.innovation.cpi import (
    InvariantClusterScorer,
    PoolExample,
    apply_known_swap,
    apply_intervention,
    fit_source_fingerprints,
    make_pool_example,
    max_probability_difference,
    relabel_clusters,
    remove_expert,
    score_examples,
    selections_from_scores,
    train_cluster_scorer,
)
from bench_coe.innovation.conservative_cpi import (
    apply_conservative_gate,
    calibrate_threshold,
    grouped_environment_folds,
    proposal_margin,
)
from bench_coe.innovation.cpi_ce import (
    CategoricalClusterScorer,
    CategoricalScores,
    apply_categorical_gate,
    categorical_target,
    max_categorical_probability_difference,
    none_aware_margin,
    scaled_probabilities,
    score_categorical_logits,
    temperature_nll,
)
from bench_coe.innovation.cpi_remaining import (
    METHODS as REMAINING_METHODS,
    PRIMARY_METHOD as REMAINING_PRIMARY,
    RICH_CLUSTER_FEATURE_NAMES,
    RemainingCategoricalScorer,
    collate_rich_pool_examples,
    fit_masked_source_fingerprints,
    max_remaining_invariance_difference,
    smooth_subject_dro_loss,
)
from bench_coe.innovation.data import CacheAdapter, EvaluationLabelAdapter
from bench_coe.innovation.schema import Selection
from registry_fixture import write_registry


class CPIPropertiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        experts = ("e1", "e2", "e3", "e4")
        for expert_index, expert in enumerate(experts):
            model_dir = root / expert
            model_dir.mkdir()
            rows = []
            for index in range(12):
                gold = "a" if index % 2 == 0 else "b"
                prediction = gold if (index + expert_index) % 3 else ("b" if gold == "a" else "a")
                rows.append(
                    {
                        "id": str(index),
                        "subject": f"s{index % 3}",
                        "answer": gold,
                        "prediction": prediction,
                        "response": prediction,
                        "is_correct": prediction == gold,
                    }
                )
            (model_dir / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")
        self.registry, self.registry_hash = write_registry(root, root, "toy", "source", "language")
        self.adapter = CacheAdapter.from_source_registry(
            root,
            "toy",
            "source",
            "language",
            {expert: f"f{index % 2}" for index, expert in enumerate(experts)},
            experts,
            self.registry,
            self.registry_hash,
        )
        self.batch = self.adapter.load_observables()
        self.labels = self.adapter.load_source_labels()
        self.fingerprints = fit_source_fingerprints(self.batch, self.labels, rank=2)
        self.example = make_pool_example(self.batch, self.batch.question_ids[0], self.fingerprints, self.labels)
        torch.manual_seed(3)
        self.model = InvariantClusterScorer(self.fingerprints.dimension + 2, hidden_dim=12)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_permutation_and_clone_invariance(self) -> None:
        original = score_examples(self.model, [self.example], "cpu")
        permutation = apply_intervention(self.example, "permutation", np.random.default_rng(4))
        clone = apply_intervention(self.example, "exact_clone", np.random.default_rng(5))
        self.assertLess(max_probability_difference(original, score_examples(self.model, [permutation], "cpu")), 1e-6)
        self.assertEqual(max_probability_difference(original, score_examples(self.model, [clone], "cpu")), 0.0)

    def test_cluster_relabeling_is_equivariant(self) -> None:
        clusters = sorted(set(int(value) for value in self.example.cluster_ids if value >= 0))
        mapping = {cluster: cluster + 17 for cluster in clusters}
        relabeled = relabel_clusters(self.example, mapping)
        original = score_examples(self.model, [self.example], "cpu")[0]
        changed = score_examples(self.model, [relabeled], "cpu")[0]
        for cluster in clusters:
            self.assertAlmostEqual(original[cluster], changed[mapping[cluster]], places=6)

    def test_variable_pool_missing_and_unseen_fingerprint(self) -> None:
        reduced = remove_expert(self.example, self.example.expert_ids[0])
        missing = apply_intervention(self.example, "missing_output", np.random.default_rng(9))
        pseudo = apply_intervention(self.example, "pseudo_clone", np.random.default_rng(10))
        results = score_examples(self.model, [reduced, missing, pseudo], "cpu")
        self.assertEqual(len(results), 3)
        self.assertTrue(all(np.isfinite(list(result.values())).all() for result in results))

    def test_real_swap_preserves_replacement_token_and_pool_legality(self) -> None:
        replacement = PoolExample(
            question_id=self.example.question_id,
            expert_ids=("external",),
            family_ids=("external-family",),
            fingerprints=self.example.fingerprints[:1].copy(),
            cluster_ids=np.asarray([99]),
            uncertainties=np.asarray([0.37], dtype=np.float32),
            valid=np.asarray([True]),
            cluster_labels={99: 1.0},
            normalized_answers={99: "external-answer"},
        )
        swapped = apply_known_swap(
            self.example,
            replacement,
            {self.example.expert_ids[0]: "external"},
            np.random.default_rng(4),
        )
        self.assertNotIn(self.example.expert_ids[0], swapped.expert_ids)
        self.assertIn("external", swapped.expert_ids)
        external_cluster = int(swapped.cluster_ids[swapped.expert_ids.index("external")])
        scores = [{cluster: float(cluster == external_cluster) for cluster in set(swapped.cluster_ids) if cluster >= 0}]
        selection = selections_from_scores(self.batch, [swapped], scores, "swap")[0]
        self.assertEqual(selection.selected_expert_id, "external")
        self.assertIn(selection.selected_expert_id, selection.observable_features["available_expert_ids"])

    def test_invariance_alignment_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            max_probability_difference([{0: 1.0}], [])
        with self.assertRaises(ValueError):
            max_probability_difference([{0: 1.0}], [{1: 1.0}])

    def test_paired_variants_share_initialization_hash(self) -> None:
        examples = [make_pool_example(self.batch, qid, self.fingerprints, self.labels) for qid in self.batch.question_ids[:4]]
        none, _ = train_cluster_scorer(examples, self.fingerprints.dimension + 2, "cpu", 77, "none", epochs=1, hidden_dim=8)
        clone, _ = train_cluster_scorer(examples, self.fingerprints.dimension + 2, "cpu", 77, "exact_clone", epochs=1, hidden_dim=8)
        self.assertEqual(none.initialization_sha256, clone.initialization_sha256)

    def test_unanimous_and_all_singleton_probabilities(self) -> None:
        unanimous = replace(self.example, cluster_ids=np.zeros_like(self.example.cluster_ids), valid=np.ones_like(self.example.valid))
        unanimous_scores = score_examples(self.model, [unanimous], "cpu")[0]
        self.assertEqual(unanimous_scores, {0: 1.0})
        singleton = replace(
            self.example,
            cluster_ids=np.arange(len(self.example.expert_ids), dtype=np.int64),
            valid=np.ones_like(self.example.valid),
        )
        singleton_scores = score_examples(self.model, [singleton], "cpu")[0]
        self.assertEqual(len(singleton_scores), len(self.example.expert_ids))
        self.assertAlmostEqual(sum(singleton_scores.values()), 1.0, places=6)

    def test_evaluation_labels_cannot_build_training_examples(self) -> None:
        with self.assertRaises(TypeError):
            make_pool_example(
                self.batch,
                self.batch.question_ids[0],
                self.fingerprints,
                EvaluationLabelAdapter.from_registry(
                    self.adapter.cache_path,
                    "toy",
                    "source",
                    "language",
                    self.adapter.expert_ids,
                    self.registry,
                    self.registry_hash,
                ).load(),  # type: ignore[arg-type]
            )

    def test_conservative_gate_switches_only_above_margin(self) -> None:
        question_id = self.batch.question_ids[0]
        baseline = Selection(question_id, 0, "e1", "a", {}, {"e1": 0.5}, None, {})
        candidate = Selection(question_id, 1, "e2", "b", {"0": 0.2, "1": 0.8}, {}, None, {})
        self.assertAlmostEqual(proposal_margin(candidate, baseline), 0.6)
        accepted = apply_conservative_gate([candidate], [baseline], 0.5)[0]
        rejected = apply_conservative_gate([candidate], [baseline], 0.7)[0]
        self.assertEqual(accepted.selected_expert_id, "e2")
        self.assertEqual(rejected.selected_expert_id, "e1")
        self.assertEqual(rejected.fallback_reason, "conservative_margin_below_threshold")

    def test_nested_threshold_calibration_has_safe_fallback_and_oof_coverage(self) -> None:
        baseline: list[Selection] = []
        candidate: list[Selection] = []
        for index, question_id in enumerate(self.batch.question_ids):
            baseline.append(Selection(question_id, 0, "e1", "a", {}, {"e1": 0.5}, None, {}))
            candidate.append(
                Selection(
                    question_id,
                    1,
                    "e2",
                    "b",
                    {"0": 0.2 + 0.01 * index, "1": 0.8 - 0.01 * index},
                    {},
                    None,
                    {},
                )
            )
        threshold, diagnostics = calibrate_threshold(
            candidate,
            baseline,
            self.labels,
            self.labels.environment_by_question,
            [0.0, 0.5, 1.01],
            min_worst_delta=-0.005,
            min_micro_delta=0.0,
            worst_weight=0.5,
        )
        self.assertIn(threshold, {0.0, 0.5, 1.01})
        self.assertTrue(next(row for row in diagnostics if row.threshold == 1.01).feasible)
        folds = grouped_environment_folds(self.labels, 2)
        heldout = [question_id for _, _, question_ids in folds for question_id in question_ids]
        self.assertEqual(sorted(heldout), sorted(self.batch.question_ids))
        for _, train_ids, test_ids in folds:
            self.assertTrue(set(train_ids).isdisjoint(test_ids))

    def test_run_and_base_gpu_maps_are_independent_and_validated(self) -> None:
        config = {
            "seeds": [11, 12, 13, 14],
            "physical_gpus": [4, 5, 6, 7],
            "base_physical_gpus": [0, 1, 2, 3],
        }
        self.assertEqual(seed_gpu_map(config, "physical_gpus"), {11: 4, 12: 5, 13: 6, 14: 7})
        self.assertEqual(seed_gpu_map(config, "base_physical_gpus"), {11: 0, 12: 1, 13: 2, 14: 3})
        with self.assertRaises(ValueError):
            seed_gpu_map({"seeds": [11, 12], "physical_gpus": [4, 4]}, "physical_gpus")

    def test_categorical_target_and_none_probability_are_well_formed(self) -> None:
        clusters = sorted(set(int(value) for value in self.example.cluster_ids if value >= 0))
        no_correct = replace(self.example, cluster_labels={cluster: 0.0 for cluster in clusters})
        one_correct = replace(
            self.example,
            cluster_labels={cluster: float(cluster == clusters[0]) for cluster in clusters},
        )
        self.assertIsNone(categorical_target(no_correct))
        self.assertEqual(categorical_target(one_correct), clusters[0])
        if len(clusters) > 1:
            multiple = replace(self.example, cluster_labels={cluster: 1.0 for cluster in clusters})
            with self.assertRaises(ValueError):
                categorical_target(multiple)
        scores = CategoricalScores("q", {0: 0.2, 1: -0.1}, 0.4)
        probabilities, none_probability = scaled_probabilities(scores, 1.25)
        self.assertAlmostEqual(sum(probabilities.values()) + none_probability, 1.0, places=7)

    def test_categorical_exact_clone_is_invariant(self) -> None:
        model = CategoricalClusterScorer(self.fingerprints.dimension + 2, hidden_dim=12)
        clone = apply_intervention(self.example, "exact_clone", np.random.default_rng(91))
        original_scores = score_categorical_logits(model, [self.example], "cpu")
        clone_scores = score_categorical_logits(model, [clone], "cpu")
        self.assertEqual(max_categorical_probability_difference(original_scores, clone_scores), 0.0)

    def test_none_aware_gate_requires_advantage_over_none_and_source_best(self) -> None:
        baseline = Selection("q", 0, "e1", "a", {}, {"e1": 0.5}, None, {})
        candidate = Selection(
            "q",
            1,
            "e2",
            "b",
            {"0": 0.2, "1": 0.65},
            {},
            None,
            {"none_correct_probability": 0.15},
        )
        self.assertAlmostEqual(none_aware_margin(candidate, baseline), 0.45)
        self.assertEqual(apply_categorical_gate([candidate], [baseline], 0.4)[0].selected_expert_id, "e2")
        blocked = replace(candidate, observable_features={"none_correct_probability": 0.7})
        self.assertEqual(apply_categorical_gate([blocked], [baseline], 0.0)[0].selected_expert_id, "e1")

    def test_categorical_temperature_rejects_evaluation_labels(self) -> None:
        output = CategoricalScores(self.example.question_id, {0: 0.1}, 0.2)
        labeled = replace(self.example, cluster_ids=np.zeros_like(self.example.cluster_ids), cluster_labels={0: 1.0})
        evaluation_labels = EvaluationLabelAdapter.from_registry(
            self.adapter.cache_path,
            "toy",
            "source",
            "language",
            self.adapter.expert_ids,
            self.registry,
            self.registry_hash,
        ).load()
        with self.assertRaises(TypeError):
            temperature_nll([output], [labeled], evaluation_labels, 1.0)  # type: ignore[arg-type]

    def test_masked_fingerprint_excludes_invalid_outputs_from_accuracy(self) -> None:
        masked = fit_masked_source_fingerprints(self.batch, self.labels, rank=2)
        expert = self.batch.pool.expert_ids[0]
        records = self.batch.for_question
        observed = [
            question_id
            for question_id in self.batch.question_ids
            if next(row for row in records(question_id) if row.expert_id == expert).valid_output
        ]
        expected = (sum(bool(self.labels.get(question_id, expert)) for question_id in observed) + 1.0) / (
            len(observed) + 2.0
        )
        self.assertAlmostEqual(masked.values[expert][0], expected, places=7)
        self.assertEqual(masked.feature_names[0], "source_observed_accuracy")

    def test_rich_features_are_finite_and_clone_invariant(self) -> None:
        batch = collate_rich_pool_examples([self.example], "cpu")
        self.assertEqual(batch.cluster_extra.shape[-1], len(RICH_CLUSTER_FEATURE_NAMES))
        self.assertTrue(torch.isfinite(batch.cluster_extra).all())
        model = RemainingCategoricalScorer(
            self.fingerprints.dimension + 2,
            len(RICH_CLUSTER_FEATURE_NAMES),
            hidden_dim=12,
        )
        difference = max_remaining_invariance_difference(
            model,
            [self.example],
            "cpu",
            "rich",
            "exact_clone",
            17,
        )
        self.assertEqual(difference, 0.0)

    def test_smooth_subject_dro_matches_equal_group_loss_and_tracks_tail(self) -> None:
        equal = torch.tensor([0.7, 0.7], requires_grad=True)
        equal_loss = smooth_subject_dro_loss(equal, ["a", "b"], alpha=0.5, tau=0.1)
        self.assertAlmostEqual(float(equal_loss.detach()), 0.7, places=6)
        unequal = torch.tensor([0.1, 1.0], requires_grad=True)
        dro_loss = smooth_subject_dro_loss(unequal, ["a", "b"], alpha=0.5, tau=0.1)
        self.assertGreater(float(dro_loss.detach()), float(unequal.mean().detach()))
        dro_loss.backward()
        self.assertGreater(float(unequal.grad[1]), float(unequal.grad[0]))

    def test_remaining_method_family_has_one_frozen_primary(self) -> None:
        self.assertEqual(len(REMAINING_METHODS), len(set(REMAINING_METHODS)))
        self.assertIn(REMAINING_PRIMARY, REMAINING_METHODS)
        self.assertTrue(REMAINING_PRIMARY.endswith("__none_fallback"))


if __name__ == "__main__":
    unittest.main()
