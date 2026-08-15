# Improve5/6 prior-art overlap results

Status: **all requested source and diagnostic OOD experiments complete; source
decision remains NO-GO**.

Formal roots:

- Source: `outputs/bench_coe/innovation/prior_art_overlap/source_loso_gpu0_3_v2_20260814`
- Multimodal OOD: `outputs/bench_coe/innovation/prior_art_overlap/multimodal_ood_gpu0_3_v2_retry1_20260814`
- Language OOD: `outputs/bench_coe/innovation/prior_art_overlap/language_ood_gpu0_3_v2_20260814`

The OOD datasets were previously used during development. Their results are
reported as diagnostics, not as blind confirmation or permission to reverse
the source gate.

## Source decision

MMMU-Pro source evaluation uses 577 questions, 30 outer subject folds, four
GPU/seed repeats, 57 full-pool methods, and two 12-method pool panels.

| Item | Result |
| --- | ---: |
| FCRG full | 27.9029% |
| Global Best post-hoc | 28.5962% |
| KNOP | 26.8631% |
| FCRG - Global Best | -0.6932 pp |
| Hierarchical paired 95% CI | [-2.9463, +1.5598] pp |
| Exact McNemar p, maximum across seeds | 0.6587 |
| Holm-adjusted p | 1.0000 |
| Rescues / harms versus Global Best | 21 / 25 |
| Worst-subject delta | -7.6923 pp |
| Non-negative subjects | 63.33% |

Only two of nine frozen gate criteria pass (`FCRG > KNOP` and a positive
isolated H2 increment). The aggregate decision is **NO-GO**.

## Strong baseline comparison

| Method | Source | Multimodal macro | Language macro |
| --- | ---: | ---: | ---: |
| Global Best post-hoc | 28.596% | 43.354% | 41.070% |
| Majority / support | 25.823% | 42.306% | 44.516% |
| MCB-DCS adaptation | 26.170% | 41.721% | 40.955% |
| KNOP | 26.863% | 41.966% | 42.179% |
| OPRS / Improve5 | 27.383% | 43.867% | 44.011% |
| MORE structured | 26.690% | 34.109% | 44.300% |
| MORE MiniLM | 27.036% | 34.356% | **44.781%** |
| Smoothie LOCAL spectral | 24.957% | 44.221% | 43.695% |
| Smoothie LOCAL MiniLM | 26.516% | **44.236%** | 41.840% |
| Global + Local rank | **28.769%** | 42.422% | 41.814% |
| Learned logistic | 26.690% | 34.163% | 44.474% |
| FCRG full | 27.903% | 44.104% | 43.858% |

FCRG is not the strongest registered method in any of these three aggregate
views. This directly supports the review's recommendation to keep Improve5 as
a baseline and prevents a broad superiority claim for Improve6.

## Per-target OOD results

| Target | Global Best | FCRG | Delta | Stronger registered result |
| --- | ---: | ---: | ---: | --- |
| CMMMU val | 37.667% | 39.444% | +1.778 pp | Smoothie LOCAL MiniLM 40.111% |
| MathVista testmini | 61.000% | 62.600% | +1.600 pp | Smoothie LOCAL spectral 64.000% |
| MMMU-Pro test-id | 31.396% | 30.269% | -1.127 pp | Global + Local rank 31.743% |
| BBH | 67.808% | 75.964% | +8.155 pp | Agreement x Global 78.283% |
| GPQA | 30.935% | 33.012% | +2.076 pp | MORE MiniLM 33.557% |
| MMStar text-only | 24.467% | 22.600% | -1.867 pp | Uncertainty-only 28.133% |

Multimodal FCRG macro/micro deltas are +0.750/+0.622 points, with only two of
three targets non-negative and a -1.127-point worst target. Language
macro/micro deltas are +2.788/+4.711 points, again with only two of three
targets non-negative and a -1.867-point worst target.

The per-target paired analysis is mixed:

- CMMMU and MathVista FCRG gains over Global Best have CIs crossing zero and
  Holm-adjusted p = 1.0.
- MMMU-Pro test has a negative FCRG delta with CI crossing zero.
- BBH gains +8.155 points over Global Best, CI [+7.326, +9.000], but FCRG is
  significantly **below** MORE MiniLM by 1.490 points, CI
  [-2.027, -0.983], Holm-adjusted p = 1.84e-6.
- GPQA gains +2.076 points over Global Best, CI [+1.133, +2.999],
  Holm-adjusted p = 7.38e-4; its -0.545-point delta versus MORE MiniLM is not
  significant.
- MMStar loses 1.867 points versus Global Best, CI
  [-3.267, -0.467]; after correction the comparison is not significant
  (Holm-adjusted p = 0.438).

## FCRG decisive ablations

All values are accuracy; OOD columns are unweighted three-dataset macros.

| Variant | Source | Multimodal | Language |
| --- | ---: | ---: | ---: |
| Full / depth 2 | 27.903% | 44.104% | 43.858% |
| G only | 28.596% | 43.354% | 41.070% |
| A only | 25.823% | 42.306% | 44.516% |
| L only | 25.477% | 42.505% | 41.984% |
| Column mean only | 28.596% | 43.354% | 41.124% |
| H1 only | 28.423% | 43.487% | 41.741% |
| H2 only | 28.596% | 43.354% | 41.124% |
| H1 + H2 only | 28.596% | 43.354% | 41.385% |
| No failure conditioning | 28.076% | 44.070% | 43.807% |
| No A/U | 27.210% | 42.388% | 42.079% |
| No L | 27.383% | **44.135%** | **44.272%** |
| No G | 27.383% | 44.058% | 43.800% |
| No self-loop | 27.903% | 44.141% | 43.816% |
| Row normalized | 27.730% | 44.091% | 44.060% |
| Column normalized | 26.170% | 43.724% | 43.999% |
| Row softmax | 26.343% | 43.881% | 43.983% |
| Symmetric | 26.516% | 43.884% | 43.870% |
| Random edges | 26.343% | 43.880% | 44.065% |
| Degree relabel | 27.036% | 42.981% | 43.426% |
| Depth 1 | 28.076% | 44.125% | 43.912% |
| Depth 3 | 27.903% | 44.199% | 43.809% |
| Depth 4 | 28.076% | 44.199% | 43.787% |
| Depth 5 | 28.076% | 44.199% | 43.780% |
| Nested learned weights | 26.690% | 43.911% | 44.519% |

The mechanism is not isolated consistently. Static column mean beats full
FCRG on source; no-L beats full FCRG on both OOD macros; random edges and row
normalization beat full FCRG on language; and depths 3-5 beat depth 2 on
multimodal while depth 1 is better on source/language. Thus the evidence does
not establish that the exact directed two-hop graph is the cause of transfer.

## K sensitivity and expert-pool replacement

| KNOP K | Source | Multimodal macro | Language macro |
| ---: | ---: | ---: | ---: |
| 8 | 23.917% | 39.349% | 38.688% |
| 16 | 26.863% | 40.798% | 38.880% |
| 32 | 26.863% | 41.966% | **42.179%** |
| 64 | 26.516% | **42.762%** | 41.442% |

The target-optimal K differs by modality, so no target result is used to
change the frozen K = 32 default.

| Dataset | Original pool FCRG - Global | Qwen-swap FCRG - Global |
| --- | ---: | ---: |
| MMMU-Pro source | -2.253 pp | -0.347 pp |
| CMMMU | +1.222 pp | +0.778 pp |
| MathVista | -0.600 pp | -0.600 pp |
| MMMU-Pro test | -0.434 pp | +1.908 pp |

The sign changes across datasets and pools. Pool replacement therefore does
not demonstrate stable repair-graph transfer.

## Cost and fast/full modes

- Source and multimodal full selectors make 11 nominal calls per query;
  language full selectors make 14.
- The true fast path makes one call. Source cached serial latency is 1.322 s
  versus 6.618 s for full FCRG. Multimodal ranges are 1.030-1.308 s fast and
  6.549-11.107 s full.
- Language caches do not contain usable latency, so only call counts are
  interpretable.
- Source cascades use 1.000-1.035 calls and exactly match the fast baseline.
  Multimodal cascades use 1.000-1.167 calls and give at most +0.087 points on
  MMMU-Pro test. Language cascades use 1.000-4.446 calls; they improve BBH and
  GPQA over the fixed fast expert but remain separate exploratory trade-offs.
- `global_best_posthoc` is not the one-call baseline: it inspects output
  validity and is charged the full pool. This matters most for language, where
  invalid-output fallback substantially improves over the fixed fast expert.

## Reproducibility and integrity

| Stage | Seeds on GPUs 0-3 | Runtime per seed | Peak CUDA allocation |
| --- | --- | ---: | ---: |
| Source | 20260822-20260825 | 207.5-214.3 s | 243,533,824 B |
| Multimodal OOD | 20260826-20260829 | 88.5-92.6 s | 734,031,360 B |
| Language OOD | 20260830-20260833 | 271.0-288.1 s | 734,031,360 B |

Every required method is present. Deterministic methods have identical hashes
across seeds; stochastic methods retain all four outcomes. An independent
post-run pass rehashed 2,165 completion-bound files totaling 16.554 GiB with
zero mismatches. All prediction manifests say `labels_opened=false`; target
aggregation authenticated every prediction package before opening labels.

## Conclusion

Improve5 is fully repositioned as OPRS, a strong prior-art-derived baseline.
FCRG shows useful diagnostic gains on CMMMU, MathVista, BBH, and GPQA, but it
fails source, loses on MMMU-Pro test and MMStar, is beaten by adapted
MORE/Smoothie baselines, and does not consistently beat graph null controls or
simpler ablations. The complete evidence therefore does **not** promote FCRG
to a validated main contribution. No locked test is authorized.
