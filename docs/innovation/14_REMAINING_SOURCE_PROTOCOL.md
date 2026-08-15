# Remaining source-development experiment protocol

Status: **frozen before implementation and result generation on 2026-08-09**

## Scope and exclusions

This batch executes every remaining experiment from the final development
report that can be answered with the registered MMMU-Pro source cache. It does
not open any development-OOD or target labels.

TopoMix remains infeasible because there is only one genuine compatible
multimodal source environment. Repair-Probe Coreset and Sequential Rescue remain
unauthorized because no proposed selector passed its source gate. A locked final
test remains blocked because no untouched dataset or blind split exists. These
preconditions cannot be repaired by relabeling a dataset whose results have
already influenced development.

The executable batch covers:

1. full four-seed, 30-subject LOSO training ablations for every CPI pool
   intervention;
2. a worst-subject distributionally robust training objective;
3. richer, gold-free answer-cluster representations; and
4. an explicit observation mask in source fingerprints so invalid or missing
   outputs are not converted into incorrect observations.

## Data and firewall

- Source: registered `mmmu_pro/validation_id`, 577 questions, 30 subjects and
  the frozen common 11-expert pool.
- Real replacement episodes use the registered Qwen 4B and Qwen 9B cached
  source outputs.
- Every scored question is held out by subject from fingerprints, model fitting
  and all source statistics.
- Predictions are written and SHA-256 hashed before `EvaluationLabels` are
  constructed.
- No development-OOD result selects a method, seed, feature, threshold or
  hyperparameter.

The user explicitly authorized temporary scheduling on physical GPUs 4-7 while
GPUs 0-3 are occupied. This is a scheduling exception to the local default and
does not change the statistical protocol.

## Models and controlled variants

All variants use the categorical answer-cluster objective with an explicit
`none-correct` class, clone-canonical DeepSets pooling, hidden size 48, 24
epochs, Adam learning rate 0.003, deterministic initialization and the same
outer-fold seed. Temperature is fixed at 1.0; no result-dependent threshold is
fitted in this factorial experiment.

### Full-LOSO intervention training ablations

The frozen variants are:

- no intervention;
- permutation;
- random expert dropout;
- leave-one-expert-out;
- leave-one-family-out;
- missing output;
- exact clone;
- pseudo clone;
- real known expert swap; and
- the existing deterministic full-intervention schedule.

These variants use the legacy fingerprint and four legacy cluster features.
They isolate training interventions; test-time pools are unchanged.

### Feature/mask/DRO factorial

The full-intervention schedule is evaluated under every combination of:

- cluster features: legacy or rich;
- source fingerprint: legacy or observation-mask-aware; and
- objective: mean CE or subject-DRO CE.

The legacy/legacy/mean cell is shared with the full-intervention ablation. The
remaining seven cells are independently fitted with the same seed and budget.

The mask-aware fingerprint computes expert accuracy and centered correctness
components only over valid, observed outputs. Missing entries are neutral in
the centered matrix, while validity remains an explicit fingerprint dimension.

Rich cluster features are query-local and permutation/clone invariant. In
addition to agreement share, family breadth and uncertainty, they contain
source-fingerprint summaries, normalized family entropy, fingerprint
dispersion, total-pool share, valid/missing fractions, singleton status and the
plurality margin. They never compare answer identity across queries.

Subject-DRO uses the frozen loss

`0.5 * mean_CE + 0.5 * (tau * logmeanexp(subject_CE / tau))`

with `tau=0.1`, computed only across subjects present in the outer training
fold. Augmented copies inherit their source question's subject.

## Predictions and primary hypothesis

Every fitted cell emits:

- `raw`: the highest-scoring real answer cluster; and
- `none_fallback`: Source Best is retained when the explicit `none-correct`
  probability is at least the proposed cluster probability.

The sole predeclared primary method is:

`factor_rich_mask_dro__none_fallback`.

It is compared with source-only Best Single. GO requires all of:

- subject-macro accuracy delta at least +0.25 percentage points;
- worst seed-subject delta at least -0.50 percentage points;
- at least two thirds of seed-subject cells non-negative; and
- all firewall, determinism, invariance and artifact-authentication tests pass.

All other cells are diagnostic. Pairwise intervals and exact McNemar tests are
reported, with Holm correction across the registered candidate family. A
post-hoc best cell cannot replace the primary gate.

## Frozen execution

- Seeds: 20260808, 20260809, 20260810 and 20260811.
- Physical GPUs: 4, 5, 6 and 7 respectively.
- Outer validation: complete 30-subject LOSO.
- Bootstrap: crossed seed/shared-query bootstrap, 10,000 aggregate samples.
- Existing source prediction artifacts are never overwritten.
- Smoke validation uses a separate output root and at most two environments.
- Independent variants may run as four concurrent shards per physical GPU.
  A label-free merger accepts the run only when the authenticated shard methods
  form the exact frozen partition; it byte-copies predictions and verifies that
  their pre-evaluation SHA-256 values are unchanged.
