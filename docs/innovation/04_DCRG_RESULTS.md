# DCRG source-validation results

Final authenticated run: `outputs/bench_coe/innovation/dcrg/source_loso_remediation_v5_20260808`

The experiment held out each of the 30 MMMU-Pro source subjects in turn. Every
fitted quantity excluded the held-out subject. The OOF nuisance model uses only
observable topology features fitted inside each training fold; changing an OOF
row's correctness cannot change that row's nuisance prediction.

## Gate result

| Quantity | Observed | Required |
| --- | ---: | ---: |
| Macro accuracy delta vs RepairChain | -3.605 pp | >= +0.250 pp |
| Micro accuracy delta vs RepairChain | -2.600 pp | descriptive |
| Worst-subject delta vs RepairChain | -22.222 pp | >= -0.500 pp |
| Subjects with non-negative delta | 46.7% | >= 66.7% |
| Paired query bootstrap 95% CI | [-5.373, +0.173] pp | descriptive |
| Exact McNemar p | 0.0912 | descriptive |
| Holm-adjusted p | 0.6383 | descriptive |

**Decision: NO-GO.** DCRG does not pass any of the three pre-registered
performance gates. Development-OOD labels cannot reverse this source-only
decision, so DCRG is not promoted or used as a CPI prior.

## Aggregate source LOSO accuracy

| Method | Accuracy | Delta vs source Best Single |
| --- | ---: | ---: |
| Source Best Single | 28.596% | 0.000 pp |
| RepairChain | 27.903% | -0.693 pp |
| DCRG residual | 25.477% | -3.120 pp |
| DCRG stable | 25.303% | -3.293 pp |
| Raw conditional correctness | 15.945% | -12.652 pp |

The stable graph had no edges passing the cross-environment stability rule, so
stable, randomized, two-hop, no-difficulty, and self-loop variants were
degenerate and made identical selections. Raw and residual variants switched on
almost every query and harmed more often than they rescued.

## Reproducibility

- Source: 577 questions, 30 subjects, 11 experts; runtime 121.64 seconds on CPU.
- Predictions were written and hashed before aggregate evaluation metrics.
- The environment manifest recursively hashes the consumed source cache,
  dataset registry, code tree, config, family map, and passing test receipt.
- The direct DCRG-vs-RepairChain comparison uses aligned question IDs, paired
  bootstrap, exact McNemar, and Holm correction.
- Gate SHA-256:
  `621d8901691aa6bc2bdf062527117b5cb9c40bfaf9c4e6c5d6bb49546546129e`.

Earlier runs are retained as audit history. Only the v5 run is bound to the
current source and the 29-test v5 receipt.
