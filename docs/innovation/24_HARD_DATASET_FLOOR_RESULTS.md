# Validation-to-test optimization with hard dataset floors

Status: executed on 2026-08-14. All evaluated targets are known development
diagnostics; the run does not authorize a locked-test or confirmatory claim.

The later strict-positive extension, which requires every dataset to improve
rather than merely meet a floor, is reported in
`docs/innovation/25_STRICT_POSITIVE_ALL_DATASETS_RESULTS.md`.

## Acceptance contract

A candidate passes only when every condition below is true. Missing a required
dataset fails closed.

1. MMMU-Pro `test_id` accuracy is strictly greater than `fcrg_full`.
2. Source LOSO, CMMMU, MathVista, BBH, GPQA, and MMStar are each no worse than
   `fcrg_full`; improvements on one dataset cannot offset a loss on another.
3. The following absolute accuracy floors are also met:

| Dataset | Hard floor |
| --- | ---: |
| MathVista testmini | 62.6000% |
| BBH | 75.9638% |
| GPQA | 33.0117% |
| MMStar text-only | 22.6000% |

The contract is declared in configuration, validated before prediction, and
reported per candidate with actual accuracy, floor, margin, and pass/fail state.
Target labels remain evaluator-only and are opened after all prediction files
and the prediction manifest have been written and hashed.

## Best universal label-free guard

The universal run evaluates four source-only base candidates under 15
source-calibrated domain policies. Seven of 60 candidates pass the complete
contract; three are universal query-support guards.

| Dataset | FCRG | Universal guard | Delta (points) | Floor met |
| --- | ---: | ---: | ---: | :---: |
| MMMU-Pro source LOSO | 27.9029% | 28.5962% | +0.6932 | n/a |
| MMMU-Pro test | 30.2689% | 30.7892% | +0.5204 | primary gain |
| CMMMU | 39.4444% | 39.5556% | +0.1111 | n/a |
| MathVista | 62.6000% | 62.6000% | 0.0000 | yes |
| BBH | 75.9638% | 75.9638% | 0.0000 | yes |
| GPQA | 33.0117% | 33.0117% | 0.0000 | yes |
| MMStar | 22.6000% | 22.6000% | 0.0000 | yes |

The best universal method is
`dguard__query_support__nearest__q0p5__cmeta__repair_safe__fcrg_full__beta_lcb_t0p02_r1__cluster_rank__k0__raw__share0__margin0__families1`.

## Best same-dataset deployment portfolio

The portfolio run evaluates all 22 source-improving frozen candidates. It
enables a candidate only when the target dataset identity is MMMU-Pro and
returns the exact FCRG decision on every other dataset. All 22 candidates pass
the hard-floor contract.

| Dataset | FCRG | Best portfolio | Delta (points) | Floor met |
| --- | ---: | ---: | ---: | :---: |
| MMMU-Pro source LOSO | 27.9029% | 28.7695% | +0.8666 | n/a |
| MMMU-Pro test | 30.2689% | 31.7433% | +1.4744 | primary gain |
| CMMMU | 39.4444% | 39.4444% | 0.0000 | n/a |
| MathVista | 62.6000% | 62.6000% | 0.0000 | yes |
| BBH | 75.9638% | 75.9638% | 0.0000 | yes |
| GPQA | 33.0117% | 33.0117% | 0.0000 | yes |
| MMStar | 22.6000% | 22.6000% | 0.0000 | yes |

The best portfolio method is
`dguard__same_dataset__identity__q1p0__cmeta__core_diverse__fcrg_full__rank_t0p02_r1__hard__k5__family__share0__margin0p05__families1`.
On MMMU-Pro test it has 44 rescues and 27 harms, exact McNemar p = 0.0568,
and paired normal 95% delta interval [+0.0440, +2.9049] points. Its 31.7433%
equals the best registered development result; it does not exceed it.

The identity policy has 100% activation on the four MMMU-Pro test seeds and 0%
activation in all 20 seed-target rows for CMMMU, MathVista, BBH, GPQA, and
MMStar. It is therefore a dataset-scoped deployment portfolio, not evidence of
universal cross-dataset generalization.

An independent row-level audit compared question ID, selected cluster, selected
expert, normalized answer, cluster/expert scores, and fallback reason for the
best portfolio against FCRG. All 58,716 predictions across those five datasets
and four seeds matched exactly, and every row recorded the guarded fallback.

## Verification and artifacts

- Unit tests: 72/72 passed, including absolute-floor, missing-target, and
  relative-regression failure cases.
- Universal hard-floor run: 60 candidates, 7 strict passes, 3 universal passes,
  1,692 files in the completion manifest.
- Portfolio hard-floor run: 22 candidates, 22 strict passes, 628 files in the
  completion manifest.
- Universal prediction-manifest SHA-256:
  `059978ce651f69d7ba78b6b0ff676a06eba9de15e2795c66d80b3884ed479d8a`.
- Portfolio prediction-manifest SHA-256:
  `c594fc45cda04cc92001386ee049b213b00d2d584fe127bd08daacbd770aa34c`.
- Universal run root:
  `outputs/bench_coe/innovation/domain_guarded_meta/validation_to_test_v3_hard_floors_20260814`.
- Portfolio run root:
  `outputs/bench_coe/innovation/domain_guarded_meta/validation_to_test_v4_hard_floor_portfolio_20260814`.

Both experiments reuse authenticated cached expert responses and perform no
model inference. Per repository protocol, cached statistics and recombination
run on CPU; GPUs 0-3 are reserved for experiments that actually require GPU
inference.
