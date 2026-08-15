# Prior-art overlap execution status

Status: **complete on physical GPUs 0-3 as of 2026-08-14**.

## Completed stages

1. Implemented and tested 23 classic/response-selection baselines, 25 FCRG
   variants, four K values, five cascade thresholds, and two 12-method
   multimodal pool-shift panels.
2. Materialized six physically label-free target caches. Counts are
   1,153/1,000/900 multimodal and 6,511/4,768/1,500 language questions; no
   forbidden label field was found.
3. Passed 59 tests for the source snapshot and 60 tests for the final OOD
   snapshot. GPU smoke tests covered source, both OOD modalities, all method
   counts, and the real missing-output regression case.
4. Ran MMMU-Pro source LOSO seeds 20260822-20260825 on GPUs 0-3 and aggregated
   81 methods: 57 full-pool plus 12 in each four-expert pool.
5. Ran CMMMU, MathVista, and MMMU-Pro test seeds 20260826-20260829 on GPUs 0-3;
   each target has 81 methods.
6. Ran BBH, GPQA, and MMStar seeds 20260830-20260833 on GPUs 0-3; each target
   has 57 methods.
7. Independently rehashed 2,165 completion-bound files totaling
   17,774,364,619 bytes (16.554 GiB) with zero mismatches.

## Preserved failed attempt

The first full multimodal launch is retained at
`outputs/bench_coe/innovation/prior_art_overlap/multimodal_ood_gpu0_3_v2_20260814`.
All four seeds stopped on the same MMMU-Pro row because a legacy-equivalence
assertion was incorrectly applied to incomplete expert rows. The legacy code
treated missing strings as an answer cluster, while canonical FCRG correctly
excludes missing outputs.

The fix scopes strict equivalence to complete rows and records incomplete-row
differences separately. A 150-question GPU regression smoke covered the exact
failing row; both four-expert complete pools had zero mismatches. The successful
retry uses a new directory and never overwrites the failed attempt.

## Formal output roots

- Source:
  `outputs/bench_coe/innovation/prior_art_overlap/source_loso_gpu0_3_v2_20260814`
- Multimodal diagnostic OOD:
  `outputs/bench_coe/innovation/prior_art_overlap/multimodal_ood_gpu0_3_v2_retry1_20260814`
- Language diagnostic OOD:
  `outputs/bench_coe/innovation/prior_art_overlap/language_ood_gpu0_3_v2_20260814`

## Decision state

The source gate remains **NO-GO**: FCRG is 27.9029% versus 28.5962% for
Global Best, delta -0.6932 points with 95% CI [-2.9463, +1.5598]. The completed
OOD results are diagnostic only. They neither override this decision nor
authorize a locked evaluation or a main-method novelty claim.
