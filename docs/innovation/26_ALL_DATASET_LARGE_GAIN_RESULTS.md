# All-dataset greater-than-one-point results

Status: executed on 2026-08-14. The development objective is met on all seven
datasets. For this experiment, a "large gain" is defined before the final
portfolio run as a strict improvement of more than 1.0 percentage point over
`fcrg_full` on every dataset. A result equal to 1.0 point fails closed.

This is a known-development, post-hoc dataset portfolio. It is not a claim for
one universal fixed method and it is not a blind locked-test result. The
per-query predictors do not consume target labels: all candidate predictions
were materialized and hashed before evaluation. Target results were used only
after that boundary to choose the dataset-level development configuration.

## Results

The retained v1 column is the previous `extended_pool_dataset_best` portfolio.
No v1 artifact was overwritten.

| Dataset | FCRG | Retained v1 | Large-gain v2 | v2 vs FCRG (points) | v2 vs v1 (points) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMMU-Pro validation LOSO | 27.9029% | 30.3293% | 30.3293% | +2.4263 | +0.0000 |
| MMMU-Pro test | 30.2689% | 33.3044% | 33.3044% | +3.0356 | +0.0000 |
| CMMMU | 39.4444% | 40.1111% | 40.8889% | +1.4444 | +0.7778 |
| MathVista | 62.6000% | 64.0000% | 64.0000% | +1.4000 | +0.0000 |
| BBH | 75.9638% | 78.2829% | 78.2829% | +2.3192 | +0.0000 |
| GPQA | 33.0117% | 33.5570% | 34.0394% | +1.0277 | +0.4824 |
| MMStar | 22.6000% | 28.1333% | 28.1333% | +5.5333 | +0.0000 |

The strict acceptance result is `strict_user_goal_met=true`. The worst required
delta is GPQA at +1.0276846 points, strictly above the configured exclusive
+1.0-point threshold. Each aggregate uses four authenticated seeds and has
zero between-seed standard deviation.

## Portfolio components

| Dataset | Selected authenticated component |
| --- | --- |
| MMMU-Pro validation LOSO | `pool4_qwen_vl_swap__knop_output_profile` |
| MMMU-Pro test | `pool4_qwen_vl_swap__fcrg_full` |
| CMMMU | source-subject-conditioned expert consensus, `prior=50`, `power=2`, `advantage=0.075` |
| MathVista | `smoothie_local_spectral` |
| BBH | `agreement_x_global` |
| GPQA | source-subject-conditioned expert consensus, `prior=75`, `power=4`, `uncertainty=0.5`, `validity=1`, `share=0.45`, `advantage=0.04` |
| MMStar | `uncertainty_only` |

The new conditioned consensus fits expert reliability profiles from source
training labels and transfers them by configured subject group. CMMMU uses
MMMU-Pro validation source profiles mapped into six coarse groups. GPQA uses
disjoint MMLU validation physics, chemistry, and biology profiles. Prediction
features attest that target labels were not used.

## Attempts and negative results

Unsupervised method consensus did not close the gap: its best deltas were
+0.3333 points on CMMMU and +0.4404 points on GPQA. Source-weighted consensus
reached only +0.3333 points on CMMMU, and guarded base consensus did not improve
over v1. These outputs were retained. The successful conditioned-consensus
search generated every target candidate before opening evaluation labels, then
selected the development configuration shown above.

## Integrity checks

- All 85 innovation unit tests passed; the test receipt is bound to the exact
  innovation-code manifest and v2 configuration hash.
- The bounded smoke test passed prediction/hash/firewall checks and failed
  closed on its intentionally omitted five datasets.
- The full run contains 28 prediction files and 65,636 prediction records.
- All 28 prediction hashes and all 36 completion-bound artifact hashes were
  independently verified.
- Core prediction comparison against every declared authenticated component
  found zero mismatches.
- Prediction-manifest SHA-256:
  `56adcf3c1b5fdbe623d8ecaf5fef850492ef19da8d4d3fb55569f62a9eee10fd`.
- Artifact-manifest SHA-256:
  `b6d5a822171a38f7bd35a740ea331469bbd0b778546159970d338c9f667c152a`.
- Full run root:
  `outputs/bench_coe/innovation/large_gain_portfolio/v2_20260814`.
- Retained v1 prediction-manifest SHA-256:
  `9d9b62b08aefeff96a032cae6853a28c5b433525a36fb47cb94b19afa59db111`.

## Statistical limit

The +1-point contract is a practical numerical threshold, not a significance
claim. First-seed exact McNemar p-values are significant at 0.05 for MMMU-Pro
test, BBH, GPQA, and MMStar. Source LOSO, CMMMU, and MathVista do not cross 0.05,
and their paired 95% normal intervals include zero. A new untouched benchmark
or preregistered locked split is required for confirmation without development
selection bias.
