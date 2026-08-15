# CPI-Selector design

Protocol status: frozen before implementation and source-validation runs on
2026-08-08. CPI is evaluated independently because DCRG failed its source gate.

## Scope and prediction object

CPI is a post-hoc answer-cluster selector for a variable expert pool. For one
query it receives a set of expert tokens, a query-local set of answer-cluster
tokens, expert-to-cluster incidence, valid/missing masks, and source-only expert
fingerprints. It emits one correctness logit per non-empty answer cluster. It
does not have an expert classification head and never treats a query-local
cluster identifier or raw answer value as a cross-query feature.

Target correctness is absent from the model interface. Source correctness is
used only to construct fingerprints and supervised cluster labels inside a
source fold. Held-out labels are loaded only by the evaluator after predictions
have been serialized.

## Expert tokens and fingerprints

The default expert fingerprint is fitted on the training portion of the source
fold and contains:

- smoothed source global accuracy and valid-output rate;
- a four-dimensional truncated-SVD embedding of the source correctness profile;
- configured family one-hot encoding plus an unknown-family slot;
- log cost, when available, with a missing-cost indicator.

The query-dependent suffix contains valid/missing status, lexical uncertainty,
membership support, family breadth, and the cluster's mean uncertainty. Expert
identity is deliberately excluded. Source subject-accuracy vectors are an
ablation because subject vocabularies do not align reliably across datasets.
DCRG in/out profiles are disabled by default because DCRG received NO-GO. NMF,
larger PCA embeddings, and learned identity embeddings are also ablations, not
defaults.

An unseen expert is supported when the caller provides the same source-derived
fingerprint schema. Missing components use explicit masks and training-fold
means; they are never silently interpreted as zero competence.

## Clone-canonical set representation

Before neural encoding, expert tokens are canonicalized as a mathematical set.
An exact-clone key contains the query-local answer cluster, source fingerprint,
family, valid mask, uncertainty, and cost mask/value. Repeated exact keys collapse
to one token and multiplicity is not exposed. Thus exact-clone duplication leaves
the model input and all aligned cluster logits bitwise identical, apart from
backend floating-point tolerance. Cluster alignment uses normalized answer only
inside the current query; tests relabel cluster integers arbitrarily.

A symmetric KL clone loss is retained as an assertion/diagnostic and is zero by
construction after canonicalization. Same-family pseudo-clones are not collapsed
when their fingerprints or uncertainty differ. Expert dropout is not assigned an
invariance loss: each subpool receives its own correct cluster supervision.

## Models

### Linear pooled baseline

For every cluster, compute invariant mean/max expert features, global mean/max
pool features, support fraction, family breadth, and uncertainty summaries. A
single linear layer scores each cluster. This checks whether gains come from the
representation rather than nonlinear capacity.

### Small DeepSets (default)

An MLP `phi` independently embeds canonical expert tokens. Masked mean and max
pooling construct global and per-cluster summaries. A shared MLP `rho` maps each
cluster summary to a logit. Sharing `phi` and `rho`, and using only symmetric
pooling, makes expert permutation and arbitrary cluster relabeling invariant.
The default hidden width is 48 with two-layer MLPs and no attention library.

### Attention pooling (gated)

Attention pooling is not implemented unless small DeepSets passes its source
gate. This prevents a capacity increase from being mistaken for the contribution
of pool interventions.

## Supervision and interventions

Cluster targets are binary correctness labels derived only from source labels.
Members of an exact answer cluster should agree; inconsistencies are counted and
resolved by majority solely for robustness. Masked BCE supports examples where
none of the proposed clusters is correct. The optional calibration term is off
in the primary test.

Every random choice is derived from the run seed, fold, epoch, query, and
intervention name. Training variants include original-only, each single
intervention, and the full mixture:

1. Random subset: sample a non-empty expert subpool.
2. Leave-one-expert-out: remove one valid expert.
3. Leave-one-family-out: remove all experts in one present family.
4. Missing output: keep an expert token and fingerprint but mask its answer.
5. Exact clone: duplicate an identical token; canonicalization must erase it.
6. Same-family pseudo-clone: add a same-answer token with a perturbed fingerprint.
7. Known pool swap: replace one token by another configured expert token while
   preserving the replacement expert's answer, family, mask, and fingerprint.

## Validation sequence

1. Unit properties: permutation, clone, variable pool, missing expert,
   unanimous/singleton cases, unseen fingerprint, and cluster relabeling.
2. A 100-query overfit sanity run.
3. Source-only grouped validation and leave-one-subject-out prediction.
4. Source stress tests for random dropout, family removal, clones, pseudo-clones,
   missing output, and known swaps.
5. Leave-one-expert/family removal sensitivity and pool-size curves.
6. Development-OOD prediction only after a source GO decision.

Comparisons retain source Best Single, Majority, family-balanced vote, KNN,
DARE/Improve5, RepairChain/Improve6, DCRG, the linear selector, same-capacity
DeepSets without interventions, every single-intervention variant, and full CPI.

## GO/NO-GO rule

The frozen protocol applies: full CPI must exceed same-capacity no-intervention
DeepSets by at least 0.25 percentage points on source validation, its worst
leave-one-family-out delta may not be below -0.50 points, exact-clone probability
sensitivity must be below `1e-4`, and all permutation/relabeling tests must pass.
All seeds are retained; the decision uses their pooled out-of-fold predictions,
not the best seed.

Expected memory is below 1 GiB per process for 11 experts and fewer than 11
clusters. Four independent pre-registered seeds run concurrently on physical
GPUs 0-3. Each manifest records the logical/physical device, peak CUDA memory,
runtime, software environment, and prediction hashes.

## Method boundary

CPI differs from a general GNN or structured message-passing aggregator because
its central object is a clone-canonical intervention-trained set function, with
no iterative graph propagation. It differs from family-weighted voting because
family is one token attribute rather than a fixed vote multiplier. It is not a
universal pre-inference router: all currently available cached responses define
the answer clusters it scores. Cost reduction is reserved for the separately
gated sequential-acquisition extension.
