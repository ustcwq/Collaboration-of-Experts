# Conservative-CPI source-validation results

Status: **NO-GO on source development; not promoted to locked evaluation**

## Scope

Conservative-CPI is a source-only follow-up informed by the negative v5 CPI
result. For each outer MMMU-Pro subject fold, its replacement threshold is
selected from strictly grouped inner OOF predictions. The outer predictions are
the immutable, hash-authenticated v5 CPI and Source-Best artifacts. No target or
development-OOD label enters training, threshold calibration or method choice.

The four formal seeds cover 577 questions and 30 subjects each. At the user's
explicit request, this run used idle physical GPUs 4, 5, 6 and 7 because GPUs
0-3 were occupied. The run manifest separately records the original v5 artifact
mapping on GPUs 0-3, so changing execution hardware did not change or relocate
the frozen outer inputs.

## Aggregate result

| Measure | Conservative-CPI | Source Best / requirement | Result |
| --- | ---: | ---: | --- |
| Mean accuracy | 28.5095% | 28.5962% | -0.0867 pp |
| Mean subject-macro delta | -0.1530 pp | at least +0.2500 pp | Fail |
| Worst seed-subject delta | -11.1111 pp | at least -0.5000 pp | Fail |
| Non-negative seed-subject fraction | 96.6667% | at least 66.6667% | Pass |
| Crossed seed/query 95% CI | [-0.5199, +0.2600] pp | descriptive | Includes zero |
| Pooled rescues / harms | 3 / 5 | descriptive | More harms |
| Exact McNemar p-value | 0.7266 | descriptive | No evidence of a difference |

The predeclared aggregate decision is **NO-GO**. It fails both the minimum
macro improvement and worst-subject protection criteria. Passing the
non-negative-fraction criterion cannot override either failure.

## Seed results

| Seed / physical GPU | Accuracy | Micro delta | Macro delta | Worst subject | Rescues / harms | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20260808 / GPU 4 | 28.7695% | +0.1733 pp | +0.1852 pp | 0.0000 pp | 1 / 0 | NO-GO |
| 20260809 / GPU 5 | 28.5962% | 0.0000 pp | 0.0000 pp | 0.0000 pp | 0 / 0 | NO-GO |
| 20260810 / GPU 6 | 28.4229% | -0.1733 pp | -0.3820 pp | -11.1111 pp | 2 / 3 | NO-GO |
| 20260811 / GPU 7 | 28.2496% | -0.3466 pp | -0.4151 pp | -7.6923 pp | 0 / 2 | NO-GO |

The four negative seed-subject cells are Electronics (-11.1111 pp) and
Diagnostics and Laboratory Medicine (-4.3478 pp) for seed 20260810, plus Manage
(-7.6923 pp) and Math (-4.7619 pp) for seed 20260811.

## Gate behavior

- The no-switch threshold `1.01` was selected in 88 of 120 outer folds
  (73.33%). The other 32 folds selected a threshold between 0.025 and 0.30.
- Only 21 of 2308 seed-query decisions (0.91%) replaced the Source-Best answer
  cluster. Of those switches, 3 rescued an error, 5 introduced an error and 13
  did not change correctness.
- The gate therefore removes almost all CPI switches, but the remaining rare
  switches are not reliably beneficial and retain a severe small-subject tail.

This result does not support promoting Conservative-CPI. It does support a
narrower diagnosis: source-only margin calibration can learn to abstain, but
the available score margin does not rank the residual replacement opportunities
well enough to beat Source Best robustly.

## Compute and reproducibility

The four seeds ran in parallel on NVIDIA RTX A6000 GPUs 4-7. Per-seed runtime
was 158.64-165.96 seconds, peak allocated memory was 76,337,152 bytes and peak
reserved memory was 94,371,840 bytes. These measurements cover the cached
selector experiment, not end-to-end forward passes through all expert models.

Authenticated anchors:

- Config SHA-256: `c4ecb93c970240f717aae4e07857a1f27ee054ec2c4cc64eb544b5b772f6bd2a`.
- Test receipt SHA-256: `89ef492928b2830b75bbf16355dc5fe86f0d0bf443aaadf9b86ea9d19a0b53c0`
  with 32 passing tests.
- Aggregate gate SHA-256: `071bb3c0939723bd990e7d2aa21ccd9adf998ee7322ce5bac2785a3413e3d274`.
- Aggregate completion-manifest SHA-256:
  `265230a5203c0336658658a9725370cb2e9030437094b60bb74fc3be56703478`.
- Output root:
  `outputs/bench_coe/innovation/cpi_conservative/source_loso_gpu4_7_v1_20260809`.

