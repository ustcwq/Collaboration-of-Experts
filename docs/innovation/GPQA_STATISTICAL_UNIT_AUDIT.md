# GPQA statistical-unit audit

Artifact:
`outputs/bench_coe/innovation/audits/gpqa_statistical_units_v1.json`

The existing cached GPQA result has 4,768 rows per expert, but those rows are
not 4,768 independent questions.

| Configuration | Raw rows | Unique Record IDs |
| --- | ---: | ---: |
| Diamond | 198 | 198 |
| Main | 448 | 448 |
| Extended | 546 | 546 |
| Union | 1,192 config rows | 546 |

Diamond is fully contained in Main and Extended; Main is fully contained in
Extended. The cache then repeats all 1,192 config rows for four shuffled-choice
epochs, producing 4 x 1,192 = 4,768 rows while retaining only 546 unique
`Record ID` values.

For a standards-aligned GPQA headline result, this project will use the
Diamond configuration only and one prediction-level result per each of its 198
`Record ID` units. Choice permutations may be analyzed as repeated measures or
used for a pre-registered within-question aggregation, but may not inflate the
sample size. Pooling Diamond, Main, and Extended as independent rows is also
invalid because the configurations overlap.

All existing 4,768-row GPQA portfolio numbers are therefore retained as
development diagnostics, not independent-question confirmatory estimates.

