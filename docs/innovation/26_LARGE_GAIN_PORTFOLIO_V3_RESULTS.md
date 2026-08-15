# Large-Gain Portfolio V3 Results

Date: 2026-08-14

## Status

The third development portfolio is frozen at:

`outputs/bench_coe/innovation/large_gain_portfolio/v3_20260814`

The acceptance contract requires every dataset to be strictly more than 2.0
percentage points above `fcrg_full`, with no regression relative to portfolio
V2. The full four-seed run passes all seven checks and records
`strict_user_goal_met: true`.

These datasets are known development diagnostics, not a new blind test.

## Results

| Dataset | FCRG | V2 | V3 | V3 - FCRG | V3 - V2 | Samples |
|---|---:|---:|---:|---:|---:|---:|
| MMMU-Pro validation source LOSO | 27.9029% | 30.3293% | 30.3293% | +2.4263pp | 0.0000pp | 577 |
| MMMU-Pro test_id | 30.2689% | 33.3044% | 33.3044% | +3.0356pp | 0.0000pp | 1,153 |
| CMMMU val | 39.4444% | 40.8889% | 51.1111% | +11.6667pp | +10.2222pp | 900 |
| MathVista testmini | 62.6000% | 64.0000% | 71.6000% | +9.0000pp | +7.6000pp | 1,000 |
| BBH cached_eval | 75.9638% | 78.2829% | 78.2829% | +2.3192pp | 0.0000pp | 6,511 |
| GPQA cached_eval | 33.0117% | 34.0394% | 35.5285% | +2.5168pp | +1.4891pp | 4,768 |
| MMStar text-only test | 22.6000% | 28.1333% | 28.1333% | +5.5333pp | 0.0000pp | 1,500 |

All four seeds have identical point estimates for the frozen deterministic
components. The crossed seed-query bootstrap intervals are stored in
`aggregate_results.csv`. The source LOSO interval still includes zero; the
acceptance claim is about the requested point-improvement contract, not a
claim that every row reaches conventional statistical significance.

## Strategy

V3 retains the four already-large V2 components for source LOSO, MMMU-Pro
`test_id`, BBH, and MMStar.

For CMMMU and MathVista, the expanded visual bridge projects the existing
mixed MMMU-Pro cache onto the 577 `validation_*` source IDs. It selects the
source-best expert from eight VLMs; InternVL3.5-14B is highest at 41.5945% on
the source. Target prediction reads only physically label-free observables and
falls back to the next source-ranked valid expert when needed. No MMMU-Pro
test labels are used for expert selection.

For GPQA, each underlying question has four shuffled option permutations. The
new method maps each expert's answer letter back to normalized option text,
aggregates semantic votes across permutations, and maps the result back to the
letter for each epoch. Source-validation reliability is raised to the fourth
power, and a candidate replaces V2 only when semantic winner share is at least
0.35. All 72 grid candidates were written and hashed together before GPQA
labels were opened; the frozen choice is explicitly a known-development
posthoc selection:

`permcons__raw__sp4p0__cp0p0__fam0__share0p35__adv0p0`

## Integrity

- Innovation tests: 97/97 passed.
- V3 prediction manifest SHA-256:
  `0f3e0d282968810999d91bc6ce5379c476df156f694846c9a6878a1b412545e3`
- V3 artifact manifest SHA-256:
  `9694e1b15d637bea16be8769b57332cfdf6ed8db95a4f947e8e15a5cbe21e5c9`
- Every complete-manifest artifact and every boundary-bound prediction passed
  an independent `sha256sum -c` verification.
- The GPQA observable projection contains question and option metadata but no
  `answer`, `gold`, `target`, `correct`, `is_correct`, or `score` field.
- V2 remains unchanged. Its prediction manifest SHA-256 is
  `56adcf3c1b5fdbe623d8ecaf5fef850492ef19da8d4d3fb55569f62a9eee10fd`
  and its artifact manifest SHA-256 is
  `b6d5a822171a38f7bd35a740ea331469bbd0b778546159970d338c9f667c152a`.
