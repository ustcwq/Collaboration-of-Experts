# Conservative-CPI source-only protocol

Frozen: 2026-08-09, before inspecting any margin-threshold result.

## Motivation

Full CPI improved over its same-capacity comparator but remained below Source
Best Single because it harmed more queries than it rescued. Conservative-CPI
tests the paper's intended mechanism more directly: all cached experts produce
outputs, CPI proposes an answer cluster, and the proposal replaces Source Best
only when an internally calibrated score margin is sufficiently large.

This is a new source-development iteration informed by the v5 source result. It
is not an untouched confirmation experiment. MathVista and all other development
OOD labels are prohibited from fitting, threshold selection and method choice.

## Nested validation

For every outer leave-one-subject-out fold:

1. Remove the outer held-out subject.
2. Partition the remaining source subjects into two deterministic groups by
   sorted subject index modulo two.
3. For each inner group, fit fingerprints, Source Best and a full-intervention
   CPI model on the other group, then predict the inner held-out group.
4. Pool the two sets of strictly OOF inner predictions and choose one margin
   threshold from the frozen grid below.
5. Apply that threshold to the already authenticated v5 outer-fold CPI and
   Source-Best predictions. Outer labels are unavailable until predictions are
   written and hashed.

Margin is `CPI_score(proposed_cluster) - CPI_score(source_best_cluster)`. If CPI
and Source Best select the same cluster, the answer is unchanged. If the Source
Best cluster is absent from the CPI score map, the proposal is rejected.

Frozen threshold grid:

`[0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]`

An inner threshold is feasible only when its worst-subject delta is at least
`-0.005` and its micro delta is non-negative. Among feasible thresholds, maximize
`macro_delta + 0.5 * worst_subject_delta`, then macro delta, micro delta and the
higher threshold. The `1.01` threshold guarantees a no-switch fallback.

## Immutable outer inputs

The outer CPI and Source-Best selections are the v5 prediction JSONLs. Their
per-seed SHA-256 values and prediction-manifest SHA-256 values are frozen in
`configs/innovation/cpi_conservative_source_loso.yaml`. The new runner refuses a
hash mismatch and never regenerates or overwrites v5.

## Gate

Conservative-CPI is GO only if, against the paired outer Source Best predictions:

- mean macro subject delta is at least `+0.0025`;
- worst seed-by-subject delta is at least `-0.005`;
- at least two thirds of seed-by-subject deltas are non-negative;
- the crossed seed-by-shared-query 95% interval and exact McNemar result are
  reported, without requiring significance in this development iteration;
- all leakage, alignment, determinism and manifest tests pass.

All four seeds are retained. A failed gate is reported and stops this branch;
development OOD cannot reverse it.

## Execution hardware note

After the protocol and scientific hyperparameters were frozen, physical GPUs
0-3 remained occupied by unrelated work. The user explicitly requested a
temporary run on GPUs 4-7. The dedicated execution config records
`physical_gpus: [4, 5, 6, 7]` and independently records the frozen v5 input
directories as `base_physical_gpus: [0, 1, 2, 3]`. No threshold, seed, fold,
input prediction or acceptance criterion changed. Results are reported in
`docs/innovation/11_CONSERVATIVE_CPI_RESULTS.md`.
