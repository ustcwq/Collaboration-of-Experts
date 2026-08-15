# CPI-Selector source-validation results

Final authenticated run root:
`outputs/bench_coe/innovation/cpi_selector/source_loso_remediation_v5_20260808`

Four pre-registered seeds each performed 30-subject LOSO prediction on the same
577 MMMU-Pro validation-id questions. Full and no-intervention variants used the
same initialization and equal optimization budget in every fold. The aggregate
evaluator verifies every prediction and completion-manifest hash, opens labels
through the frozen dataset registry, and recomputes correctness rather than
trusting `per_query` or `seed_gate` values.

## Gate result

| Quantity | Observed | Required |
| --- | ---: | ---: |
| Full CPI accuracy, mean over seeds | 27.773% | descriptive |
| No-intervention DeepSets accuracy | 27.383% | descriptive |
| Paired seed-query delta | +0.390 pp | >= +0.250 pp |
| Crossed seed x shared-query bootstrap 95% CI | [-0.477, +1.256] pp | descriptive |
| Worst seed x family-removal delta | -1.386 pp | >= -0.500 pp |
| Maximum exact-clone sensitivity | 0 | < 1e-4 |
| Maximum permutation sensitivity | 1.79e-7 | < 1e-4 |
| Property tests | 29/29 pass | all pass |

**Decision: NO-GO.** Mean improvement passes its arithmetic threshold, but the
worst-family robustness gate fails and the crossed-bootstrap interval includes
zero. Pooled seed-query McNemar counts are 25 rescues and 16 harms
(`p=0.2110`, Holm-adjusted `p=1.0`); this pooled p-value is descriptive because
the same questions repeat across seeds.

## Main comparisons

| Method | Source LOSO accuracy | Delta vs Best Single |
| --- | ---: | ---: |
| Source Best Single | 28.596% | 0.000 pp |
| RepairChain / Improve6 | 27.903% | -0.693 pp |
| Full CPI | 27.773% | -0.823 pp |
| No-intervention DeepSets | 27.383% | -1.213 pp |
| DARE / Improve5 | 27.383% | -1.213 pp |
| Majority vote | 25.823% | -2.773 pp |
| Output-profile KNN | 25.303% | -3.293 pp |
| Family-balanced vote | 24.957% | -3.640 pp |

Full CPI improves over its controlled no-intervention comparator by 0.390 pp,
but remains 0.823 pp below source Best Single and 0.130 pp below RepairChain.
This supports the implementation of the intervention mechanism, not a claim of
superior selection accuracy.

## Seed stability

| Seed / physical GPU | Full CPI | No intervention | Delta | Worst family | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 20260808 / 0 | 28.423% | 27.730% | +0.693 pp | -0.520 pp | NO-GO |
| 20260809 / 1 | 27.730% | 27.036% | +0.693 pp | -0.520 pp | NO-GO |
| 20260810 / 2 | 27.383% | 27.730% | -0.347 pp | -1.386 pp | NO-GO |
| 20260811 / 3 | 27.556% | 27.036% | +0.520 pp | -0.520 pp | NO-GO |

The corrected uncertainty calculation treats seeds and the 577 shared questions
as crossed factors: each bootstrap draw samples seeds and one common set of
question indices, then takes their Cartesian submatrix. The older nested-query
interval `[-0.303, +1.083] pp` is superseded by
`[-0.477, +1.256] pp`.

## Pool-shift stress

| Condition, full CPI | Accuracy |
| --- | ---: |
| Original / exact clone / permutation | 27.773% |
| Real known swap to Qwen 4B/9B experts | 28.466% |
| Leave one family out | 28.033% |
| Leave one expert out | 27.730% |
| Same-family pseudo-clone | 27.556% |
| Random 20% dropout | 26.430% |
| Missing output | 27.210% |
| Random pool size 9 / 7 / 5 / 3 | 26.690 / 27.036 / 25.217 / 23.050% |

The known-swap condition uses the real cached Qwen3-VL-4B-Instruct and
Qwen3.5-9B output, family, validity, uncertainty, answer and source fingerprint;
it is a stress result involving stronger external experts, not an isolated
training-effect comparison. Removing InternVL causes the largest absolute drop,
to 24.610%.

The fixed six-subject ablation uses 110 questions per seed and is intentionally
not presented as complete LOSO. Linear-full reaches 34.545%, full intervention
33.636%, and no intervention 33.864%; it cannot override the complete-LOSO gate.

## Compute and reproducibility

- GPU 0-3 were used concurrently, one seed per physical NVIDIA RTX A6000.
- Per-seed runtime was 337.8-346.5 seconds; peak allocated/reserved memory was
  76,444,672 / 96,468,992 bytes on every card.
- Each seed contains 62 prediction files and a 123-file completion hash list.
- CUDA determinism evidence records `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the seed-
  specific `PYTHONHASHSEED`, deterministic PyTorch algorithms, deterministic
  cuDNN, and disabled cuDNN benchmark mode.
- An independent v4 execution and the final v5 execution produced identical
  SHA-256 values for all 248 paired prediction files (`mismatches=0`).
- Gate SHA-256:
  `114225820ba5e7aa86f8d32030edb53a2d829a47459f2e7ed699225cfd0696df`.

The v4 training outputs are retained as deterministic-repeat evidence; its
aggregate attempt failed before writing results because the first strict
aggregator incorrectly required the fixed six-subject ablation to contain all
577 LOSO questions. That contract bug was fixed, a new receipt was generated,
and all v5 runs were repeated. No v4 gate is used.
