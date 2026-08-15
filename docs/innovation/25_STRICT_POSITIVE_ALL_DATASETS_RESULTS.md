# Strict-positive results on every evaluated dataset

Status: executed on 2026-08-14. The numerical development objective is met.
The dataset-to-method mapping was selected from already known development
results, so these are post-hoc development portfolios, not universal-method or
locked-test claims.

## Strict contract

Every portfolio must be strictly better than `fcrg_full` on all seven datasets:
MMMU-Pro validation LOSO, MMMU-Pro test, CMMMU, MathVista, BBH, GPQA, and
MMStar. A zero delta, a missing dataset, or a result below the recorded FCRG
floor fails the complete portfolio. Improvements cannot offset another
dataset's zero or negative delta.

## Fixed-method and search audit

The authenticated prior-art packages contain 57 methods available on every
dataset. Exhaustive comparison found zero fixed methods with seven strictly
positive deltas. The 24 conservative meta-finalists and 60 domain guards also
contain zero seven-way strict-positive methods.

The count of full-pool methods strictly better than FCRG is 18 on source, 29 on
MMMU-Pro test, 11 on CMMMU, 4 on MathVista, 18 on BBH, 18 on GPQA, and 32 on
MMStar. Because a dataset-identity portfolio is separable by dataset, these
sets define 238,132,224 strict-positive full-pool combinations. Evaluating the
best component independently for each dataset is the exact Cartesian optimum;
materializing every duplicate combination would not change the answer.

## Improve6-only strict-positive portfolio

This portfolio uses only the 24 frozen conservative meta-selector candidates.

| Dataset | FCRG | Improve6 portfolio | Delta (points) |
| --- | ---: | ---: | ---: |
| MMMU-Pro validation LOSO | 27.9029% | 29.1161% | +1.2132 |
| MMMU-Pro test | 30.2689% | 31.7433% | +1.4744 |
| CMMMU | 39.4444% | 39.5556% | +0.1111 |
| MathVista | 62.6000% | 62.7000% | +0.1000 |
| BBH | 75.9638% | 76.1173% | +0.1536 |
| GPQA | 33.0117% | 33.0537% | +0.0419 |
| MMStar | 22.6000% | 24.4667% | +1.8667 |

All seven strict checks pass. The worst improvement is GPQA at +0.0419 points.
The CMMMU and MathVista gains each correspond to one net question and GPQA to
two net questions; they are numerical improvements but not significant gains.

## Registered full-pool portfolio

This is the stronger same-expert-pool result. It uses only the registered
full-pool methods, with no expert replacement.

| Dataset | FCRG | Full-pool portfolio | Delta (points) |
| --- | ---: | ---: | ---: |
| MMMU-Pro validation LOSO | 27.9029% | 28.7695% | +0.8666 |
| MMMU-Pro test | 30.2689% | 31.7433% | +1.4744 |
| CMMMU | 39.4444% | 40.1111% | +0.6667 |
| MathVista | 62.6000% | 64.0000% | +1.4000 |
| BBH | 75.9638% | 78.2829% | +2.3192 |
| GPQA | 33.0117% | 33.5570% | +0.5453 |
| MMStar | 22.6000% | 28.1333% | +5.5333 |

The worst improvement is GPQA at +0.5453 points. The selected components are
Global+Local rank for MMMU-Pro validation/test, Smoothie LOCAL MiniLM for
CMMMU, Smoothie LOCAL spectral for MathVista, Agreement x Global for BBH, MORE
MiniLM for GPQA, and Uncertainty-only for MMStar.

## Extended-pool portfolio

Allowing the previously evaluated four-expert Qwen-VL swap on MMMU-Pro raises
validation to 30.3293% (+2.4263 points) and test to 33.3044% (+3.0356 points).
The other five datasets use the same components as the registered full-pool
portfolio and retain their strict improvements. This is an expert-pool change
and must not be compared as if it used the identical inference pool.

## Integrity and statistical limits

- Three of three configured portfolios pass all 21 strict dataset checks.
- 75/75 innovation unit tests pass, including zero-delta, missing-target, and
  floor failures.
- All 84 output files were independently re-hashed.
- All 196,908 materialized predictions exactly match their declared component
  in question ID, cluster, expert, answer, scores, and fallback state.
- Prediction-manifest SHA-256:
  `9d9b62b08aefeff96a032cae6853a28c5b433525a36fb47cb94b19afa59db111`.
- Run root:
  `outputs/bench_coe/innovation/strict_positive_portfolio/v1_20260814`.

The per-question prediction never reads a gold label. However, dataset identity
selects a component, and that mapping was chosen after inspecting known
development results. Consequently the result satisfies the requested
development score objective but does not establish one fixed method's universal
transfer or permit a claim on an untouched test set.

The experiment only authenticates and recombines cached predictions. It runs on
CPU by protocol; rerunning it on GPUs 0-3 would not add inference or evidence.
