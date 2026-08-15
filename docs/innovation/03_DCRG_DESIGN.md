# DCRG design: difficulty-adjusted cross-environment rescue graph

Design frozen: 2026-08-08, before DCRG result generation.

## Research question

The historical edge `C_ij = P(Y_j=1 | Y_i=0)` confounds: (1) expert j's global strength, (2) item difficulty among i's failures, and (3) source-environment composition. DCRG tests whether a directed rescue effect remains after predicting j's expected correctness from source-only difficulty and environment information.

## Expected-correctness models

### A. Default logistic/Rasch-style model

For every ordered edge i->j and source item r, define leave-i-j-out difficulty:

`d_r^(-ij) = mean_{m not in {i,j}} Y_rm`.

For each j, edge-specific logistic regression predicts `Y_rj` from `d_r^(-ij)` plus a one-hot environment/subject vector. The intercept represents expert strength. Predictions are generated out of fold: a row's label is never used to fit its expected-correctness prediction. If a training fold has one class, use the Laplace-smoothed training prevalence. The difficulty feature never contains i or j correctness.

### B. Low-rank logistic matrix factorization (designed ablation)

Fit `logit p_rj = alpha_j + beta_r + u_r^T v_j` on training rows only, with L2 shrinkage and a held-out-item encoder that estimates `beta_r,u_r` from the other experts' observed source labels. This model can represent latent item/expert interactions but is less stable with 577 rows and complicates held-out item inference. It is not the default and will be implemented only if the logistic model leaves systematic source-validation residuals and the GO gate is otherwise close.

## Residual rescue edges

For OOF expected correctness `p_hat_rj^(-i)`, define `E_rij = Y_rj - p_hat_rj^(-i)`. In environment e:

`R_ij^e = mean(E_rij | Y_ri=0, environment=e)`.

Every edge record stores failure support, raw conditional correctness, expected baseline, residual mean, standard error, lower confidence bound, per-environment signs, stable flag, and final weight. Self-loops are zero.

Environment estimates use empirical-Bayes shrinkage toward the pooled edge residual with `support/(support+16)`. The lower bound is `shrunk_mean - z*SE`, default `z=1.0`. An environment contributes only with at least 8 failures. A stable positive edge requires at least 3 eligible environments, positive pooled LCB, at least 80% non-negative environment residual signs, and positive 10th percentile of environment LCBs. The final weight is that positive 10th percentile. Negative edges are stored diagnostically but dropped from the default graph.

MMMU-Pro validation has one benchmark and 30 subjects (9-28 rows each). Results are therefore **cross-subject stable**, not cross-dataset invariant.

## Current-query failure model

For each expert, fit source-only logistic regression for `P(Y_i=0 | observable features)` using:

- current answer support;
- answer-partition entropy;
- top1-top2 cluster margin;
- number of families supporting its cluster;
- valid/missing indicator;
- lexical uncertainty when available;
- source global accuracy as the expert fingerprint.

The model uses scaled numeric features. One-class experts fall back to a Laplace-smoothed constant. OOF probabilities are produced for diagnostics; deployment fitting uses all allowed source training rows. Target query embeddings and target labels are excluded.

## Scoring and answer-cluster selection

For target failure probabilities `f_i(q)`, one-hop rescue evidence is `H_j(q)=sum_i f_i(q) R_tilde_ij`. Expert score is `clip(1-f_j(q)+H_j(q),0,1)`. A query-local answer cluster score is the maximum member-expert score; this is invariant to exact clone duplication. Select the highest cluster score, then the highest expert score in that cluster, breaking all ties lexicographically.

Default DCRG does not use a second hop or a manually tuned weighted sum of unrelated score terms.

## Shapes and complexity

- correctness: `[N,M]`;
- edge-specific expected correctness: `[M,M,N]` conceptually, streamed as `[N]` per edge;
- graph/edge tables: `[M,M]`;
- query expert features: `[N,M,7]`;
- target failure probabilities and expert scores: `[N_target,M]`.

Logistic expected-correctness fitting costs `O(F*M*(M-1)*N)` for fixed low-dimensional features and F folds. Edge aggregation is `O(N*M^2)`; target propagation is `O(N_target*M^2)`. With N=577 and M=11 this is small and CPU-suitable.

## Pre-registered ablations

- raw C;
- C minus global expert accuracy;
- pooled residual R;
- stable residual R (default);
- no difficulty adjustment;
- no expert-strength adjustment;
- pooled versus environment-stable edges;
- self-loop on/off;
- real versus degree/weight-distribution-preserving randomized graph;
- one-hop versus two-hop (ablation only).

Randomized graphs permute nonzero destination endpoints within source rows while preserving each source node's out-degree and multiset of weights. A seed is mandatory.

## Validation order and GO gate

1. Unit tests and synthetic edge sanity cases.
2. Fifty-row smoke.
3. Source leave-one-subject-out over all eligible MMMU-Pro validation subjects.
4. Development OOD only if the frozen source gate passes.

GO requires macro source LODO accuracy at least 0.25 percentage points above an identically split RepairChain, worst-subject delta no worse than -0.50 points, at least two thirds of subjects non-negative, and all leakage/invariance tests passing. Target results cannot reverse GO/NO-GO.

## Failure conditions

Fail closed on overlapping folds, duplicate query/expert records, target evaluation labels passed to fit, fewer than two experts after masking, invalid family mapping, or non-finite scores. Extreme experts that never fail or are never correct use smoothed constants and must not produce NaN. Sparse edges shrink to zero rather than being extrapolated.
