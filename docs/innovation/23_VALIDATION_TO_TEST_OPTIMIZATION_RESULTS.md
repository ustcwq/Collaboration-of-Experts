# MMMU-Pro validation-to-test optimization results

Status: executed on 2026-08-14. All targets remain known development OOD
diagnostics; these results do not authorize a locked-test claim.

Follow-up hard-floor enforcement and complete rerun are documented in
`docs/innovation/24_HARD_DATASET_FLOOR_RESULTS.md`. That run adds explicit
MathVista, BBH, GPQA, and MMStar absolute floors on top of the relative FCRG
non-regression checks described below.

## Objective and non-regression definition

The source is the 577-question MMMU-Pro `validation_id` split. The primary
target is the 1,153-question MMMU-Pro `test_id` split. The operational
non-regression guardrail is accuracy greater than or equal to `fcrg_full` on
source LOSO, CMMMU, MathVista, BBH, GPQA, and MMStar. A stricter comparison to
the best registered method is reported separately and is not silently
substituted for this FCRG guardrail.

Target labels never enter fitting, source-trial ranking, base-candidate
selection, domain-threshold calibration, or prediction. Every guarded
prediction is serialized and hashed before any target label adapter is
constructed.

## Executed search

1. Source-OOF conservative meta-selection generated 22,464 fixed recipes over
   three method pools, 13 reliability weightings, hard/cluster-rank fusion,
   four top-K settings, family balancing, and 36 conservative gates. The
   recipes produced 1,988 unique source prediction behaviors. Twenty-four
   source-noninferior, diverse finalists were frozen before target evaluation.
2. The first cross-dataset pass found no universal non-regression winner among
   those 24 finalists. The closest conservative recipe improved MMMU-Pro test
   by 0.2602 points but lost 0.1111 on CMMMU and 0.2000 on MathVista. This
   negative result is retained.
3. A second, source-calibrated domain-guard pass used 12 label-free topology
   features, robust scaling, cross-subject nearest-source support, two
   distribution distances, and 15 policies. Four source-only base candidates
   times 15 policies yielded 60 guarded methods. Seven met the requested
   FCRG non-regression goal; three used a universal query-support rule rather
   than dataset identity.
4. A separate deployment portfolio evaluated all 22 source-improving unique
   finalists under an explicit same-dataset policy. This is an operational
   routing policy, not evidence of universal cross-dataset generalization.

## Best universal label-free guard

The best universal result uses the source-OOF beta-LCB cluster-rank base and
enables its proposed switch only when the query's robust nearest-source
distance is below the source cross-subject median (`0.0095315401`). Otherwise
it returns the original FCRG prediction exactly.

| Dataset | FCRG | Guarded | Delta (points) |
| --- | ---: | ---: | ---: |
| MMMU-Pro source LOSO | 27.9029% | 28.5962% | +0.6932 |
| MMMU-Pro test | 30.2689% | 30.7892% | +0.5204 |
| CMMMU | 39.4444% | 39.5556% | +0.1111 |
| MathVista | 62.6000% | 62.6000% | 0.0000 |
| BBH | 75.9638% | 75.9638% | 0.0000 |
| GPQA | 33.0117% | 33.0117% | 0.0000 |
| MMStar | 22.6000% | 22.6000% | 0.0000 |

On MMMU-Pro test the method records 11 rescues and 5 harms versus FCRG. The
paired normal 95% interval is [-0.1592, +1.2000] points and the exact McNemar
p-value is 0.2101. The improvement is positive but not statistically
significant. It remains 0.6071 points below Global Best and 0.9540 points below
the best registered target method.

The query-support gate activates on 43.37% of MMMU-Pro test, 15.33% of CMMMU,
10.30% of MathVista, and 0% of BBH, GPQA, and MMStar. These activation rates are
computed before labels are opened.

## Best same-dataset deployment portfolio

The best dataset-scoped policy applies the source-frozen core-diverse rank
consensus only when the target dataset is MMMU-Pro; it returns FCRG on every
other dataset.

| Dataset | FCRG | Portfolio | Delta (points) |
| --- | ---: | ---: | ---: |
| MMMU-Pro source LOSO | 27.9029% | 28.7695% | +0.8666 |
| MMMU-Pro test | 30.2689% | 31.7433% | +1.4744 |
| CMMMU | 39.4444% | 39.4444% | 0.0000 |
| MathVista | 62.6000% | 62.6000% | 0.0000 |
| BBH | 75.9638% | 75.9638% | 0.0000 |
| GPQA | 33.0117% | 33.0117% | 0.0000 |
| MMStar | 22.6000% | 22.6000% | 0.0000 |

On MMMU-Pro test it has 44 rescues and 27 harms. The paired normal 95% interval
is [+0.0440, +2.9049] points, while exact McNemar p is 0.0568. It equals, but
does not exceed, the best registered target accuracy of 31.7433%. Because the
policy explicitly conditions on dataset identity, it must be presented as a
deployment portfolio and not as a universal method improvement.

## Reproducibility artifacts

- Source search and first target pass:
  `outputs/bench_coe/innovation/conservative_meta_optimization/validation_to_test_v1_20260814`
- Universal domain-guard pass:
  `outputs/bench_coe/innovation/domain_guarded_meta/validation_to_test_v2_20260814`
- Full same-dataset portfolio:
  `outputs/bench_coe/innovation/domain_guarded_meta/same_dataset_portfolio_v3_20260814`
- Test receipts:
  `conservative_meta_validation_to_test_v1_tests.json` (65 passing tests) and
  `domain_guarded_meta_validation_to_test_v2_tests.json` (68 passing tests)

The universal run binds 1,692 files to prediction-manifest hash
`bed3a411991c431f022858b338ed40dfeeeb1f00e75b57620bd2ce975f7eb0dd`.
The full same-dataset portfolio binds 628 files to prediction-manifest hash
`bd5e92921af76c4a0766a236e6b6d1e1b280530c59c833338b52c2c2931daa8a`.
Both runs reuse authenticated cached expert responses and therefore run on CPU;
occupying GPUs 0-3 would add no model inference or experimental coverage.

## Decision

The requested empirical FCRG non-regression objective is met by three universal
query-support guards and by the explicitly scoped same-dataset portfolio. No
new confirmatory or publication claim is authorized: MMMU-Pro test and all five
other targets had already influenced development, and the best universal test
gain is not statistically significant. A future claim still requires an
untouched dataset or blind split declared before method selection.
