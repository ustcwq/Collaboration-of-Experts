# Improve6 scoring-simplification results

Status: **NO-GO on source validation; development OOD not run**

Authenticated run:
`outputs/bench_coe/innovation/repair_simplification/source_loso_gpu0_3_v1_20260814`

The run applies outer leave-one-subject-out validation to all 577 MMMU-Pro
validation-id questions across 30 subjects. `beta` and `alpha` are chosen only
by a nested leave-one-subject-out loop inside each outer training split. M0
reproduces the untouched legacy RepairChain answer on every question for all
four seeds.

## Full M0-M8 result

| Method | Source LOSO accuracy | Delta vs M0 | Delta vs Source Best |
| --- | ---: | ---: | ---: |
| Source Best Single | 28.5962% | +0.6932 pp | 0.0000 pp |
| M0: full five-term RepairChain | 27.9029% | 0.0000 pp | -0.6932 pp |
| M1: no L | 27.3830% | -0.5199 pp | -1.2132 pp |
| M2: no L, no direct G | 26.8631% | -1.0399 pp | -1.7331 pp |
| M3: expert H1+A, nested beta | 28.4229% | +0.5199 pp | -0.1733 pp |
| M3: cluster-mean H1+A, nested beta | 26.3432% | -1.5598 pp | -2.2530 pp |
| M4: H1 only | 28.4229% | +0.5199 pp | -0.1733 pp |
| M4: cluster-mean H1 only | 25.6499% | -2.2530 pp | -2.9463 pp |
| M5: H1+H2, nested alpha | 28.4229% | +0.5199 pp | -0.1733 pp |
| M6: answer support only | 26.3432% | -1.5598 pp | -2.2530 pp |
| M7: local KNN only | 25.4766% | -2.4263 pp | -3.1196 pp |
| M8: source global only | 28.5962% | +0.6932 pp | 0.0000 pp |

The expert-level M3 selector chose `beta=1.0` in 104/120 seed-folds and
`beta=0.9` in 16/120. It therefore collapses almost entirely to M4. The
answer-cluster M3 selector chose `beta=0.9` in 88/120 and `beta=0.8` in 32/120,
but averaging H1 within a cluster reduced accuracy.

## Registered gate

Pure H1 improves micro accuracy over M0 by **+0.5199 pp** with 24 rescues and
21 harms. That point estimate is not stable enough to promote:

| Gate item | Observed | Required |
| --- | ---: | ---: |
| Micro delta vs M0 | +0.5199 pp | >= 0.0000 pp |
| Paired hierarchical 95% CI | [-1.7331, +2.7730] pp | lower >= -0.5000 pp |
| Worst-subject delta | -8.6957 pp | >= -0.5000 pp |
| Non-negative subjects | 80.0% | >= 66.7% |

The cluster H1+A fallback is also NO-GO: its micro delta is -1.5598 pp, CI is
[-4.5061, +1.3865] pp, worst-subject delta is -35.7143 pp, and only 53.3% of
subjects are non-negative.

## Component decisions

No optional component meets the registered requirement of positive delta,
strictly positive paired CI, and more rescues than harms:

| Added evidence | Paired comparison | Delta | 95% CI | Decision |
| --- | --- | ---: | ---: | --- |
| L | M0 vs M1 | +0.5199 pp | [-1.0399, +2.0797] pp | Not justified |
| direct G | M1 vs M2 | +0.5199 pp | [-0.3466, +1.3865] pp | Not justified |
| A | nested M3 vs M4 | 0.0000 pp | [-0.5199, +0.5199] pp | Not justified |
| H2 | M5 vs M4 | 0.0000 pp | [0.0000, 0.0000] pp | Delete |
| H1 over A-only | M4 vs M6 | +2.0797 pp | [-0.6932, +4.8527] pp | Directional only |

H2 produces exactly the same full-pool choices as H1. Its graph controls score
21.4471% (randomized), 28.2496% (symmetric), 28.4229% (no self-loops), and
28.5962% (column centrality), versus 28.4229% for the real two-hop graph. Since
the static column-centrality control is slightly stronger, there is no evidence
for a repair-chain mechanism. The registered H2 decision is **DELETE**.

## Expert-pool replacement

The frozen full-pool beta/alpha values were applied without retuning to the
original judged4 pool and to the pool replacing Qwen3.5-2B with
Qwen3-VL-2B-Instruct.

| Formula | Original pool | Delta vs original M0 | Replacement pool | Delta vs replacement M0 |
| --- | ---: | ---: | ---: | ---: |
| M0 | 26.3432% | 0.0000 pp | 28.2496% | 0.0000 pp |
| M3 expert H1+A | 28.4229% | +2.0797 pp | 27.5563% | -0.6932 pp |
| M3 cluster H1+A | 27.3830% | +1.0399 pp | 27.0364% | -1.2132 pp |
| M4 H1 | 28.4229% | +2.0797 pp | 27.7296% | -0.5199 pp |
| M5 H1+H2 | 28.5962% | +2.2530 pp | 27.9029% | -0.3466 pp |

M4 misses the frozen replacement threshold by 0.0199 pp: -0.5199 pp observed
versus -0.5000 pp required. The cluster fallback misses it by more. Thus neither
registered simplified formula passes both the normal-pool and pool-shift gates.

## Reproducibility and scope

- Physical GPUs 0, 1, 2, and 3 ran seeds 20260814-20260817 respectively.
- Per-seed runtime was 145.93-148.18 seconds; peak CUDA allocation was
  33,565,696 bytes on every card.
- Each seed contains 577 questions, 30 held-out environments, 50 prediction
  methods, and 111 completion-bound artifacts.
- All deterministic prediction hashes are identical across GPUs. Independent
  re-hashing found zero mismatches in all seed and aggregate manifests.
- The passing receipt contains 49 tests and code hash
  `1d7bed87d214b2634685e87bc14142607dbafdcd90f8167157965bfc684c8853`.
- The four seeds are deterministic hardware repeats, not four independent
  datasets; the crossed CI does not manufacture extra query evidence.
- This is source-only evidence from one genuine multimodal source benchmark.
  The source NO-GO forbids MathVista/CMMMU/MMMU-Pro-test development OOD under
  the frozen protocol, and no locked final benchmark is available.

## Conclusion

The five-term formula is not empirically defensible term by term, and H2 should
be removed. H1 alone is the strongest nontrivial simplified formula and nearly
matches Source Best, but its worst-subject failure, crossed CI, and exact
Qwen-pool replacement check prevent promotion. The defensible conclusion is a
negative one: do not claim RepairChain, do not promote the answer-cluster
formula, and retain Source Best as the source-validated deployable baseline.
