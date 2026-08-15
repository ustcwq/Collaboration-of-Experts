# CPI-CE source-validation results

Status: **NO-GO on source development; not promoted to development OOD or a
locked evaluation**

## Scope

CPI-CE is a source-only follow-up to the audited BCE/softmax semantic mismatch
and the Conservative-CPI NO-GO. It retains the same clone-canonical DeepSets,
full pool-intervention schedule, fingerprints, optimizer and four seeds. It
replaces independent cluster BCE with categorical cross entropy over the
query-local answer clusters plus an explicit `none-correct` class.

Every outer MMMU-Pro subject fold uses two grouped inner OOF models to choose a
temperature and a Source-Best replacement margin. A third model is fitted on
the complete outer training fold. The 577 outer predictions are written and
hashed before source evaluation labels are constructed. Frozen v5 Source-Best
and BCE predictions are hash-authenticated comparison inputs.

## Aggregate result

| Measure | CPI-CE calibrated | Source Best / requirement | Result |
| --- | ---: | ---: | --- |
| Mean accuracy | 28.4229% | 28.5962% | -0.1733 pp |
| Mean subject-macro delta | -0.1713 pp | at least +0.2500 pp | Fail |
| Worst seed-subject delta | -5.2632 pp | at least -0.5000 pp | Fail |
| Non-negative seed-subject fraction | 96.6667% | at least 66.6667% | Pass |
| Crossed seed/query 95% CI | [-0.5633, 0.0000] pp | descriptive | Non-positive |
| Pooled rescues / harms | 0 / 4 | descriptive | All accepted changes harm |
| Exact McNemar p-value | 0.1250 | descriptive | Not significant |

The predeclared aggregate decision is **NO-GO**. Both the macro-improvement and
worst-subject requirements fail.

## Objective and fallback ablations

| Method | Accuracy | Delta vs Source Best | Rescues / harms | Crossed 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Raw CPI-CE | 27.2097% | -1.3865 pp | 66 / 98 | [-3.4673, +0.6932] pp |
| CPI-CE + none fallback | 28.3795% | -0.2166 pp | 2 / 7 | [-0.7366, +0.1733] pp |
| CPI-CE + nested calibration | 28.4229% | -0.1733 pp | 0 / 4 | [-0.5633, 0.0000] pp |

Raw CPI-CE reaches 27.2097% versus 27.7730% for the frozen full BCE CPI, a
-0.5633 pp difference with crossed 95% CI [-1.9497, +0.7799] pp. Therefore the
categorical objective repairs probability semantics but does not improve the
answer-cluster ranking.

## Seed results for the primary method

| Seed / physical GPU | Accuracy | Micro delta | Macro delta | Worst subject | Rescues / harms | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20260808 / GPU 4 | 28.2496% | -0.3466 pp | -0.3509 pp | -5.2632 pp | 0 / 2 | NO-GO |
| 20260809 / GPU 5 | 28.2496% | -0.3466 pp | -0.3342 pp | -5.2632 pp | 0 / 2 | NO-GO |
| 20260810 / GPU 6 | 28.5962% | 0.0000 pp | 0.0000 pp | 0.0000 pp | 0 / 0 | NO-GO |
| 20260811 / GPU 7 | 28.5962% | 0.0000 pp | 0.0000 pp | 0.0000 pp | 0 / 0 | NO-GO |

Negative cells occur in Agriculture and Computer Science for seed 20260808,
and Computer Science and Math for seed 20260809. Each accepted replacement
changes a correct InternVL answer to an incorrect LFM or MiniCPM answer.

## Calibration behavior

- `none-correct` outranks every real cluster on 1848 of 2308 seed-query rows
  (80.07%).
- The calibrated gate accepts only 4 replacements (0.17%), and all four are
  harmful. The raw selector changes 432 rows (18.72%) and produces 66 rescues
  versus 98 harms.
- Threshold `1.01` is selected in 72 of 120 outer folds (60.00%).
- Temperature 1.25 is selected in 98 folds, temperature 1.0 in 14 and
  temperature 1.5 in 8. No other frozen temperature is selected.

The result isolates the remaining limitation: the model can represent and
calibrate `none-correct`, but the gold-free expert/cluster features do not rank
rare beneficial replacements reliably. More calibration or a different loss
on the same representation is unlikely to solve that information bottleneck.

## Compute and reproducibility

The four seeds ran in parallel on physical NVIDIA RTX A6000 GPUs 4-7, as
explicitly authorized while GPUs 0-3 were occupied. Per-seed runtime was
327.68-333.74 seconds. Peak allocated/reserved CUDA memory was
76,422,656 / 96,468,992 bytes. These are cached-selector measurements and do not
include expert-model forward passes.

Authenticated anchors:

- Config SHA-256: `33d7d7622bede21de92069420845aebef087900cae508d70b119c7cfd8615774`.
- Frozen protocol SHA-256: `90061c5299aec1ede494d42a1dec304646baa568968a72e3386192c17a8bde10`.
- Test receipt SHA-256: `7bfb9355f25c55686be03bc25f85c047f3f7c2c3050a0d1b29af226640483da8`
  with 36 passing tests.
- Aggregate gate SHA-256: `3efafb034e3ee38c3017659859249fe1c90dfb30be0acb265f597b92cbd11565`.
- Aggregate completion-manifest SHA-256:
  `9940ead0cd583785d19d0a047239a10135952581d4b38a58fb6e5fcd91ab100e`.
- Output root:
  `outputs/bench_coe/innovation/cpi_ce/source_loso_gpu4_7_v1_20260809`.

