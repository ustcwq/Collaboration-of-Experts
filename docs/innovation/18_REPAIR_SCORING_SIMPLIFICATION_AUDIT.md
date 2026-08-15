# Adversarial audit: Improve6 scoring simplification

Audit date: 2026-08-14. Scope: the new simplification implementation, frozen
config, four GPU runs, and aggregate package. This audit is read-only with
respect to authenticated code and predictions.

## Findings

No BLOCKER or HIGH issue was found.

### MEDIUM-1: evidence remains single-source

The only genuine compatible multimodal source is MMMU-Pro validation-id. The
implementation correctly uses nested subject LOSO, but subject shift is not the
same as cross-benchmark invariance. The source NO-GO prevents development OOD,
and there is no locked final test. This limits claims; it does not invalidate
the reported source result.

### MEDIUM-2: seeds are reproducibility repeats

All deterministic methods have identical prediction hashes across the four
GPUs. Treating the seeds as independent experiments would understate
uncertainty. The aggregate code explicitly labels this limitation and uses one
shared query resample in its crossed bootstrap; reports do not multiply the
effective query count.

### LOW-1: graph randomization has weak structural content

The smoothed correction graph is dense, so row-wise off-diagonal permutation
preserves a trivial full degree. It still preserves the registered row weight
multiset and is a valid score-centrality diagnostic, but it is not evidence for
sparse motif robustness. This limitation strengthens, rather than weakens, the
conservative H2 DELETE decision because the static column-centrality control is
already better than the real two-hop graph.

### LOW-2: workload is mostly CPU-bound

KNN feature construction runs on CPU; the assigned A6000 executes float64 H1/H2
matrix propagation. Peak CUDA allocation is about 32 MiB and utilization is
bursty. Device identity and physical GPU restrictions are authentic, but this
experiment should not be described as GPU-intensive.

## Leakage and statistical checks

- `fit_repair_components` accepts only `SourceTrainingLabels` and checks source
  provenance at `bench_coe/innovation/repair_simplification.py:179`.
- Every scored source subject is removed at the outer fold boundary, while
  beta/alpha selection runs another LOSO over only the outer training batch at
  `bench_coe/innovation/run_repair_simplification.py:106`.
- Predictions, component tables, graph tables, and SHA-256 values are written
  before the evaluation adapter is opened at
  `bench_coe/innovation/run_repair_simplification.py:423` and line 452.
- Aggregation authenticates all four completion packages before opening labels
  at `bench_coe/innovation/aggregate_repair_simplification.py:291`.
- Missing outputs have support zero and no cluster; query-local cluster scores
  use mean H1 and configured pool-size support at
  `bench_coe/innovation/repair_simplification.py:353`.
- M0-M8 equations are explicit at
  `bench_coe/innovation/repair_simplification.py:258`; M0 has zero answer-choice
  mismatch against the untouched legacy selector in every seed.
- Gate selection is source-only and hierarchical: M4 first, cluster M3 only as
  fallback, plus matched original/replacement pool checks at
  `bench_coe/innovation/aggregate_repair_simplification.py:406`.

## Engineering checks

- The test receipt records 49 passing tests and authenticates both code and the
  frozen config.
- CUDA visibility, deterministic algorithms, and the physical GPU mapping are
  asserted before computation.
- Each seed covers exactly 577 questions and 30 environments and writes a
  completion manifest. Independent re-hashing found 0/444 seed-artifact
  mismatches and 0/10 aggregate-artifact mismatches.
- Existing Bench-CoE/Improve5/Improve6 code and historical output directories
  were not modified or overwritten.
- GPU 4-7 were never exposed to these processes.

## Audit conclusion

The authenticated implementation supports the reported NO-GO and H2 DELETE
decisions. Residual risks concern external validity and limited observables, not
an implementation or evaluation defect. No rerun or code remediation is
required for the stated source-only conclusion.
