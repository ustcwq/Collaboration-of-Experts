# Bench-CoE innovation experiment protocol

Protocol version: 1.0, frozen 2026-08-08 before new-method experiments.

## Data roles

- **Development source:** labeled caches used to fit methods. Initial primary source is MMMU-Pro validation-id (577 rows, fixed common 11-expert cohort). Language portfolios are secondary sources.
- **Source validation:** deterministic folds or leave-one-subject/environment-out rows from a development source. Every scored row is excluded from its source statistics and learned parameters.
- **Development OOD:** MMMU-Pro test-id, MathVista testmini, CMMMU val, BBH, GPQA, MMStar, and Gaokao caches. They may describe transfer after a source-only GO decision, but cannot select defaults.
- **Locked final test:** unavailable. No final confirmatory claim is permitted until a genuinely untouched benchmark or blind split is supplied.

Post-freeze status update (2026-08-15): the statement above records the
2026-08-08 inventory. A separate additive protocol subsequently preregistered
MuSR test as an experimenter-untouched secondary test. It was consumed exactly
once after prediction sealing; the frozen FCRG-H1 superiority claim was not
confirmed. See `28_LOCKED_MUSR_PROTOCOL.md` and
`29_LOCKED_MUSR_RESULTS.md`. MuSR is no longer available for future locked-test
claims, and no claim is made that it was absent from model pretraining.

## Baselines

Required baselines are source-only Best Single, Majority Vote, source-accuracy Weighted Vote, family-balanced vote, output-profile KNN, global+local competence, DARE, RepairChain, random expert, random answer cluster, and oracle-any-expert as an evaluator-only upper bound. Target-best single is descriptive only and must never be called a deployable baseline.

## Metrics

Let the source-only Best Single prediction be the primary reference. Report accuracy, delta versus that reference, rescue and harm counts/rates, unchanged-correct, unchanged-wrong, switch count/rate, switch precision, net uplift, oracle accuracy, and oracle-gap closed. Also report per-subject accuracy, per-family selection rate, expert/cluster selection entropy, missing-output impact, paired bootstrap 95% confidence intervals, exact McNemar p-values, and Holm-adjusted multi-method decisions.

`oracle_gap_closed = (method_accuracy - source_best_accuracy) / (oracle_accuracy - source_best_accuracy)`. If the denominator is zero, report null rather than infinity.

## Statistics

- Paired bootstrap resamples query indices, not aggregate cells; default 10,000 resamples for final reports and 1,000 for smoke runs.
- McNemar uses paired correctness on exactly aligned query IDs and the exact binomial test on discordant pairs.
- Holm correction is applied across all pre-registered method comparisons within an experiment family.
- Seeds are fixed before target evaluation. Results from every attempted seed are retained.

## Pool-shift stress tests

Evaluate deterministic expert permutation, random 20% dropout, leave-one-family-out, exact clone duplication, same-family pseudo-clone, missing outputs, and configured pool swaps. Missing outputs are masks, not wrong answer clusters. Exact clones must not change existing cluster probabilities beyond the configured numerical tolerance.

## Pre-registered GO/NO-GO gates

### DCRG

GO only if source leave-one-environment-out macro accuracy exceeds out-of-fold RepairChain by at least 0.25 percentage points, worst-environment delta versus RepairChain is no worse than -0.50 points, at least two thirds of environments are non-negative, and all leakage/invariance tests pass. Otherwise stop graph expansion and report NO-GO; development OOD cannot reverse this decision.

### CPI-Selector

GO only if intervention-trained DeepSets exceeds the same-capacity no-intervention model by at least 0.25 points on source validation, does not reduce the worst leave-one-family-out environment by more than 0.50 points, exact-clone probability sensitivity is below `1e-4`, and permutation/cluster-relabeling tests pass. Attention pooling is allowed only after the small DeepSets gate passes.

### TopoMix

Implementation requires at least two genuine source environments with compatible or mask-aware fingerprints, at least 200 examples per environment, and unlabeled target outputs. GO only if topology weighting exceeds uniform weighting on source-held-out pseudo-targets without worse support-fallback behavior. Target labels cannot choose weighting.

### Probe and sequential extensions

Probe Coreset requires a frozen selector that passed its source gate. Sequential acquisition requires a frozen full-pool selector and is compared at matched average calls. When real costs are unavailable, one call equals one cost unit.

## GPU protocol

CPI seed experiments use physical GPUs 0, 1, 2, and 3 only, one pre-registered seed per GPU. CPU-only cached statistics remain on CPU. Before launch, record `nvidia-smi`; if those GPUs are occupied by unrelated work, do not preempt it. Save the physical GPU, visible device, seed, start/end timestamps, peak allocated memory, and runtime in each run manifest.

## Prediction firewall

Prediction and evaluation are separate CLI phases. Prediction receives observable target records only, writes JSONL, and records SHA-256. Evaluation then loads the immutable prediction file and a separate label object. Any target label access in fit/predict is a hard failure.
