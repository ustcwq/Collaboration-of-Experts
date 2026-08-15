# Remaining source-development experiment results

Status: **NO-GO; all currently executable source-only experiments are complete**

## Scope

This batch completed the experiments left actionable by the final development
report: full four-seed LOSO training ablations for every pool intervention, a
worst-subject DRO objective, richer gold-free cluster features and an explicit
observation mask for source fingerprints. It used 577 registered MMMU-Pro
validation questions, 30 held-out subjects, four frozen seeds and the common
11-expert pool. No development-OOD or target result selected a method.

TopoMix, Repair-Probe, Sequential Rescue and a locked final test were not run:
their registered preconditions still fail. They respectively require a second
genuine compatible source, a selector that passed its source gate, and a new
untouched benchmark or blind split.

Latent Failure Archetypes and Higher-order Rescue Motifs are concept-survey
analysis extensions rather than separate steps in the registered execution
sequence. The source document conditions those extensions on a useful CPI or
DCRG information module. Running them as post-selected policies after both
source gates failed would violate the stopping rule, so they remain explicitly
not promoted rather than silently omitted.

## Primary result

The predeclared primary method was categorical CPI with rich features,
mask-aware fingerprints, subject-DRO and explicit `none-correct` fallback.

| Measure | Primary | Source Best / requirement | Result |
| --- | ---: | ---: | --- |
| Mean accuracy | 28.5962% | 28.5962% | +0.0000 pp |
| Subject-macro delta | 0.0000 pp | at least +0.2500 pp | Fail |
| Worst seed-subject delta | 0.0000 pp | at least -0.5000 pp | Pass |
| Non-negative seed-subject fraction | 100.0000% | at least 66.6667% | Pass |
| Crossed seed/query 95% CI | [0.0000, 0.0000] pp | descriptive | No gain |
| Pooled rescues / harms | 0 / 0 | descriptive | No corrected answer |

The aggregate decision is **NO-GO**. The primary selector falls back on 2032 of
2308 seed-query rows (88.04%). It changes three answer clusters, all in
Sociology for seed 20260809; every change maps an incorrect InternVL answer to
an incorrect LFM answer. Thus it is accuracy-equivalent to Source Best but does
not provide the required improvement.

## Factorial findings

Subject-DRO materially improves the un-gated selector in all four controlled
feature cells:

| Cell | Mean CE raw | Subject-DRO raw | DRO change | DRO delta vs Source Best |
| --- | ---: | ---: | ---: | ---: |
| Legacy feature / legacy fingerprint | 26.5598% | 28.1629% | +1.6031 pp | -0.4333 pp |
| Legacy feature / mask fingerprint | 26.5165% | 27.9029% | +1.3865 pp | -0.6932 pp |
| Rich feature / legacy fingerprint | 26.6031% | 27.4697% | +0.8666 pp | -1.1265 pp |
| Rich feature / mask fingerprint | 26.6031% | 27.6430% | +1.0399 pp | -0.9532 pp |

The best raw cell is legacy-feature DRO at 28.1629%. Its difference from Source
Best is -0.4333 pp with crossed 95% CI [-1.9064, +1.0409] pp, 40 rescues and 50
harms. DRO improves the learned ranking but does not make it reliably better
than the strongest source expert.

For `none-correct` fallback predictions:

- mask-aware fingerprints add +0.0433 pp to the legacy mean cell, crossed CI
  [-0.3466, +0.4333] pp;
- rich features add +0.0867 pp, crossed CI [-0.2166, +0.4333] pp;
- legacy DRO adds +0.2600 pp to legacy mean fallback, crossed CI
  [-0.2166, +0.8666] pp; and
- the full rich+mask+DRO combination adds +0.2600 pp to legacy mean fallback,
  with the same interval, but reaches that value by reverting to Source Best.

None of these mechanism contrasts establishes a positive selector claim.

## Full-LOSO intervention ablations

| Training intervention | Raw accuracy | None fallback accuracy | Fallback delta vs Source Best |
| --- | ---: | ---: | ---: |
| None | 26.6898% | 28.2496% | -0.3466 pp |
| Permutation | 26.6464% | 28.2496% | -0.3466 pp |
| Random dropout | 27.0364% | 28.3362% | -0.2600 pp |
| Leave expert out | 27.0364% | 28.2929% | -0.3033 pp |
| Leave family out | 27.2097% | 28.3362% | -0.2600 pp |
| Missing output | 26.9497% | 28.4229% | -0.1733 pp |
| Exact clone | 26.6898% | 28.2496% | -0.3466 pp |
| Pseudo clone | 26.8631% | 28.2496% | -0.3466 pp |
| Known real swap | 26.6031% | 28.2062% | -0.3899 pp |
| Full schedule | 26.5598% | 28.3362% | -0.2600 pp |

Missing-output training gives the best intervention-only fallback result, but
it remains 0.1733 pp below Source Best. No intervention training ablation passes
the source gate. Exact-clone training is identical to no intervention after
clone canonicalization, as expected.

## Interpretation

The batch resolves the outstanding source-only questions:

1. Mean-only training contributed to the negative tail: DRO recovers
   0.87-1.60 pp in raw selector accuracy.
2. Observation masking and richer topology summaries have small, uncertain
   effects and do not solve answer ranking.
3. The safe fallback can eliminate degradation, but only by nearly always
   retaining Source Best. It does not create rescues.
4. All ten intervention schedules now have complete 30-subject, four-seed LOSO
   results; the earlier six-subject limitation no longer applies to these
   categorical experiments.

The remaining bottleneck is information, not training capacity or loss design.
A further selector needs genuinely new gold-free evidence, such as richer model
confidence/logit traces, judge signals that are source-cross-fitted, or a second
predeclared source environment. Recombining the current cached answer letters,
fingerprints and topology is not supported as a high-value next experiment.

## Compute and reproducibility

Four independent variant shards ran concurrently on each physical NVIDIA RTX
A6000 GPU 4-7. Per-seed wall time was 1315.63-1333.72 seconds. Sum-of-shard time
was 4046.26-4062.97 seconds, corresponding to a 3.03-3.08x scheduling speedup.
Peak allocated memory summed across each card's concurrent shards was 291.52
MiB. The output directory occupies approximately 307 MiB.

All 16 shard manifests, four merged seed manifests and the aggregate manifest
were independently checked: 573 registered artifact hashes match. Maximum
exact-clone logit sensitivity is 0 and maximum permutation sensitivity is
2.38e-7. The bound test receipt records 40/40 passing tests.

Authenticated anchors:

- Protocol SHA-256:
  `63f2c920c9904dd4a2089b350e8f41416e72a9d942197d706a453274be534cc7`.
- Config SHA-256:
  `79211aac928017a7b49020d0e3c68f758a2a0fee867af158a679341ccf508ca8`.
- Test receipt SHA-256:
  `8830e72c34a8766e2ca4adfe281106c79da65d85e33bf6cb45494212773df611`.
- Tested innovation code-manifest SHA-256:
  `58a0560e0d65de77f34d7a1ada11b5bc32144b4620d87382e5ecfe46a7895a39`.
- Aggregate gate SHA-256:
  `a09d68168893fae49fd63c7cba8580d53d7181ae67b8d26d38a9a80d1e7308b9`.
- Aggregate completion-manifest SHA-256:
  `b874af071e0e80fbc398dcebaf8bc200cde047601d5704a64026a5ca9f077f83`.
- Output root:
  `outputs/bench_coe/innovation/cpi_remaining/source_loso_gpu4_7_v1_20260809`.
