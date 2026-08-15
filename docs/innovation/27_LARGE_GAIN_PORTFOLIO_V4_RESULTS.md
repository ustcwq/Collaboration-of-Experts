# Large-Gain Portfolio V4 Results

Date: 2026-08-15

## Status

The fourth development portfolio is frozen at:

`outputs/bench_coe/innovation/large_gain_portfolio/v4_20260814`

The V4 contract requires every dataset to be at least 5.0 percentage points
above `fcrg_full` and no lower than the frozen V3 result. The full four-seed
run passes all seven checks and records `strict_user_goal_met: true`. V3 is
retained unchanged at
`outputs/bench_coe/innovation/large_gain_portfolio/v3_20260814`.

This is a known-development, dataset-level posthoc portfolio. It is not a
single universal selector and it is not a new blind-test claim. The component
predictors are label-free at prediction time; all component and final
portfolio predictions were written and hash-bound before their authenticated
evaluation summaries were opened.

## Results

| Dataset | Samples | FCRG | V3 | V4 | V4 - FCRG | V4 - V3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMMU-Pro validation source LOSO | 577 | 27.9029% | 30.3293% | 41.5945% | +13.6915pp | +11.2652pp |
| MMMU-Pro test_id | 1,153 | 30.2689% | 33.3044% | 46.6609% | +16.3920pp | +13.3565pp |
| CMMMU val | 900 | 39.4444% | 51.1111% | 51.1111% | +11.6667pp | 0.0000pp |
| MathVista testmini | 1,000 | 62.6000% | 71.6000% | 71.6000% | +9.0000pp | 0.0000pp |
| BBH cached_eval | 6,511 | 75.9638% | 78.2829% | 84.3956% | +8.4319pp | +6.1127pp |
| GPQA cached_eval | 4,768 | 33.0117% | 35.5285% | 52.3280% | +19.3163pp | +16.7995pp |
| MMStar text-only test | 1,500 | 22.6000% | 28.1333% | 28.1333% | +5.5333pp | 0.0000pp |

The smallest V4 gain over FCRG is MMStar at +5.5333 points. CMMMU,
MathVista, and MMStar are deliberately unchanged from V3; their large-gain
claim is relative to FCRG, not a claim of a new V4 increment. All four seeds
have identical point estimates for the deterministic frozen components.

The paired hierarchical bootstrap 95% intervals for the V4 minus FCRG delta
are stored in `aggregate_results.csv`. In percentage points they are:

| Dataset | Delta 95% interval |
| --- | ---: |
| MMMU-Pro validation source LOSO | [+9.3544, +18.0243] |
| MMMU-Pro test_id | [+13.3565, +19.4276] |
| CMMMU val | [+8.1111, +15.1139] |
| MathVista testmini | [+6.0000, +12.0000] |
| BBH cached_eval | [+7.6486, +9.2305] |
| GPQA cached_eval | [+17.5750, +21.0570] |
| MMStar text-only test | [+2.4667, +8.6667] |

## Strategy

MMMU-Pro source LOSO and `test_id` use the expanded eight-VLM pool. Source
rows are selected out of fold, using labels only from other source
environments. Test prediction reads the physically label-free MMMU-Pro cache.
This raises source LOSO to 41.5945% and test_id to 46.6609%.

BBH overlays deterministic exact solvers for `boolean_expressions`,
`dyck_languages`, `multistep_arithmetic_two`, and `word_sorting`, with V3 as
the fallback outside supported inputs. Each supported 250-question subset is
solved at 100%, lifting total BBH accuracy to 84.3956%.

GPQA uses `DeepSeek-R1-0528-Qwen3-8B`, selected before target evaluation as
the best model on MMLU-Pro validation at 74.2857%. A two-stage protocol runs
long reasoning and then an answer-only finalizer. It produced valid epoch-0
answers for all 1,192 configuration-local queries. Semantic option text is
mapped only within the same `(config, base_question_id)` identity to the four
option permutations; raw answer identities are never compared across
queries. The overlay covers 4,732 of 4,768 rows, while 36 rows that fail the
unique semantic mapping check fall back to V3. GPQA reaches 52.3280%; its
paired McNemar comparison with FCRG has `p=5.08e-97`.

CMMMU, MathVista, and MMStar retain the frozen V3 components because they
already satisfy the new five-point baseline contract. No target result was
used to tune the GPQA reasoner or select among its outputs.

## Negative Results and Limits

The first GPQA generation smoke exhausted its reasoning budget without an
explicit answer on all 4 examples. It is retained under
`outputs/bench_coe/innovation/gpqa_long_reasoning/v4_smoke_gpu0_20260814`.
The source-defined two-stage finalizer fixed extraction before the full run.

The later 32-question overlay smoke scored 50.0000%, which was +6.2500 points
over its FCRG subset but -3.1250 points relative to its V3 subset. This bounded
result was not used for method selection or tuning. The complete predeclared
run was evaluated unchanged and produced the full GPQA result above.

The portfolio mapping is explicitly known-development posthoc. Confirmation
of a universal gain requires a preregistered method on an untouched benchmark
or locked split. The acceptance criterion is a point-improvement contract;
the intervals are reported separately and are not used to redefine it.

## Integrity

- Innovation tests: 115/115 passed for both the GPQA overlay and final V4
  portfolio receipts.
- Query-local GPQA inference audit: 1,192/1,192 valid, 1,192 unique query
  identities, zero duplicate identities.
- Final prediction files: 28; final seed-level prediction records: 65,636.
- Independent verification: 35/35 artifact hashes and 28/28 boundary-bound
  prediction hashes match; all 65,636 final records match their authenticated
  frozen components after the declared portfolio annotations, with zero
  mismatches.
- Final prediction-manifest SHA-256:
  `fcfe1a1cae02b524c4a001077273351082ea09fa8cce49adc6a691af474279fc`.
- Final artifact-manifest SHA-256:
  `126fc6d323c9b7f74dd8d21e4b829298bdadaa9aef97ef28dd7d8b4c6963bcb6`.
- Final complete-manifest file SHA-256:
  `5fc2a7717282b7f55c95757097ee54d21bacc03b22b90398bb5225892a8ce43f`.
- Frozen GPQA inference-manifest SHA-256:
  `81246744cfdf51365e7c27a06e7b5c0170539c1e9d85025b6f14387abfeaa8f6`.
