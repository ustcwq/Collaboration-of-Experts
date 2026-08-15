# CPI-CE source-only protocol

Frozen: 2026-08-09, before running any CPI-CE source-validation prediction.

Status: prospective source-development iteration, not an untouched confirmation
experiment. This iteration is motivated by the audited mismatch between the v5
independent cluster BCE loss and its softmax probability interpretation, and by
the subsequent Conservative-CPI NO-GO result.

## Hypothesis

CPI-CE retains the clone-canonical DeepSets representation and full frozen pool
intervention schedule. It changes the supervised output semantics to one
categorical distribution over the query-local answer clusters plus an explicit
`none-correct` class. This tests whether coherent competition between clusters
and the ability to represent "all available answers are wrong" improve safe
replacement of Source Best.

The source cache contains 577 questions: 367 have exactly one correct answer
cluster and 210 have no correct answer cluster. No source question has multiple
correct clusters, no cluster has mixed correctness, and the same invariant holds
after adding the two configured real replacement experts. This inspection fixes
the one-hot target semantics; it does not select a performance hyperparameter.

## Model and loss

The expert encoder, answer-cluster mean/max pooling, global pool mean/max
pooling, width, optimizer and intervention schedule match full CPI. Cluster
logits use the existing shared cluster head. A symmetric `none-correct` head
uses only gold-free global pool summaries and symmetric mean/max summaries of
the existing cluster-extra features.

For source training query q, the target is its sole correct cluster when one
exists, otherwise `none-correct`. Training minimizes categorical cross entropy.
More than one correct cluster is a fatal data-contract violation rather than an
arbitrary tie break.

## Nested source-only calibration

For every outer leave-one-subject-out fold:

1. Remove the outer held-out subject.
2. Split the remaining subjects into two deterministic groups by sorted subject
   index modulo two.
3. Fit one CPI-CE model on the complement of each inner group and predict the
   strictly held-out inner group.
4. Select a scalar softmax temperature by minimum pooled inner-OOF categorical
   NLL from the frozen grid `[0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]`.
5. At that temperature, select a Source-Best replacement margin from
   `[0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]`.
6. Refit CPI-CE on the full outer training fold and predict the untouched outer
   subject. Outer labels remain unavailable until all predictions are written
   and hashed.

For a proposed cluster c and Source-Best cluster b, define

`margin = p(c) - max(p(b), p(none-correct))`.

The gate preserves Source Best when the proposal equals b, `none-correct` wins,
the Source-Best cluster is unavailable, or the margin is below the calibrated
threshold. Threshold `1.01` guarantees no replacement.

A threshold is feasible only when inner-OOF micro delta versus Source Best is
non-negative and its worst inner subject delta is at least -0.005. Among
feasible thresholds, maximize `macro_delta + 0.5 * worst_subject_delta`, then
macro delta, micro delta, worst delta and the higher threshold.

## Frozen comparisons and gate

Three CPI-CE outputs are written and hashed before evaluation:

- `cpi_ce_raw`: highest-probability real answer cluster, ignoring `none`;
- `cpi_ce_none_fallback`: Source Best only when `none` outranks every cluster;
- `cpi_ce_calibrated`: Source Best unless the nested calibrated margin permits a
  replacement.

Frozen v5 Source Best and full BCE CPI predictions are authenticated comparison
inputs. The primary method is `cpi_ce_calibrated`. It is GO only if, over all
four predeclared seeds:

- mean subject-macro delta versus Source Best is at least +0.0025;
- worst seed-subject delta is at least -0.005;
- at least two thirds of seed-subject deltas are non-negative;
- all label-boundary, categorical-target, invariance, deterministic GPU and
  artifact-hash checks pass.

Crossed seed-by-shared-query bootstrap and exact McNemar are descriptive. All
seeds and negative results are retained. Development OOD cannot reverse a
source NO-GO.

## Execution

The user explicitly authorized temporary execution on physical GPUs 4-7 while
GPUs 0-3 are occupied. The run config must separately retain the frozen v5 base
artifact mapping on GPUs 0-3. Each seed records the actual physical device,
CUDA visibility, deterministic settings, runtime and peak memory. These cached
selector measurements exclude expert-model forward-pass cost.
