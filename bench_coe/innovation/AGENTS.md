# Bench-CoE innovation experiment rules

These rules apply to every file under `bench_coe/innovation/`.

1. Target gold labels and correctness may be read only by the evaluator after a prediction artifact has been written and hashed. They must never enter fitting, hyperparameter selection, threshold selection, method selection, seed selection, or prediction.
2. Source-local correctness statistics, fingerprints, graph edges, and training features used to score a source validation row must be out-of-fold or computed from a strictly disjoint split.
3. Never overwrite existing Bench-CoE, Improve5, Improve6, or historical output directories. New runs belong under `outputs/bench_coe/innovation/<method>/<run_id>/`.
4. Every run must save its configuration, explicit seed, command, environment, repository state, input-cache hashes, and prediction hash. If Git metadata is unavailable, record `UNKNOWN` rather than inventing a commit.
5. Randomized code must accept an explicit seed and produce deterministic CPU results for the same seed.
6. Per-query output must include prediction, selected answer cluster, selected expert, cluster/expert scores, fallback reason, observable features, valid/missing masks, and tie-breaking metadata.
7. Negative results and failed GO criteria must be reported.
8. Defaults may be selected only from source validation or source leave-one-environment-out results, never from development OOD or target results.
9. Run `python -m unittest discover -s tests/innovation -v` and a bounded smoke test before any full cached experiment.
10. Do not hard-code expert names, family names, dataset sizes, or machine-specific absolute paths in algorithms. Expert families live in reviewable configuration.
11. Raw answer identity must never be compared across queries or benchmarks. Answer clusters are query-local and missing outputs are not answer clusters.
12. Selectors consume `ObservableQueryBatch`; only source training consumes `SourceTrainingLabels`; only the evaluator consumes `EvaluationLabels`.
13. GPU experiments are restricted to physical GPUs 0-3. Do not use GPUs 4-7. Do not preempt or terminate unrelated GPU processes; wait or queue when GPUs 0-3 are occupied.
