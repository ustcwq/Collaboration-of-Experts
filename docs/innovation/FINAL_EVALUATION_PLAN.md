# Frozen final-evaluation plan and additive MuSR outcome

The original 2026-08-08 final evaluation remains **blocked** under its frozen
source-gate protocol:

1. DCRG failed its source gate.
2. CPI failed its source and family-removal gates.
3. Improve6 simplification failed its worst-subject and Qwen pool-replacement
   gates; H2 was registered for deletion.
4. The additive prior-art overlap comparison completed on GPU0-3 and failed its
   authenticated source gate: FCRG is -0.6932 points versus Global Best with a
   paired 95% CI of [-2.9463, +1.5598] points. Six previously known OOD datasets
   were subsequently executed as diagnostics by explicit request, but every
   artifact records that this does not override the source gate.
5. No genuinely untouched locked dataset existed in the 2026-08-08 audited
   cache inventory.

An additive, separately preregistered secondary test was later authorized on
MuSR before any workspace MuSR predictions or scores existed. It used fixed
MMLU-Pro validation source statistics, a fixed 14-model pool, and 756 MuSR test
questions. Predictions were sealed before the evaluator opened labels. The
frozen FCRG-H1 primary gained +0.6614 points over the equal-budget source vote,
with stratified 95% CI [-0.2646, +1.7196] and exact McNemar p=0.3018. Primary
superiority was not confirmed. This negative result does not retroactively pass
the old source gates or promote FCRG.

The conservative deployable reference remains source Best Single, but it is not
presented as a new method. All known OOD caches are explicitly excluded from a
future confirmatory test because they have already influenced development.
The completed diagnostic results cannot be used to choose FCRG variants, K,
weights, thresholds, or expert pools for that future test.

Before any future confirmatory run, another untouched dataset or blind split
must be declared,
hashed, and placed in `locked_target_datasets`. A new innovation must first pass
the frozen source criteria without changing them. Predictions for every seed
must then be serialized and hashed before the locked evaluator can access labels.
The final report must use a 10,000-draw crossed bootstrap over seeds and the
shared query set, exact McNemar tests, Holm correction, worst-domain results,
and all attempted seeds.

MuSR test is now consumed and must be treated like the existing development
benchmarks for future method selection. The complete one-time result and seals
are in `docs/innovation/29_LOCKED_MUSR_RESULTS.md` and
`outputs/bench_coe/innovation/locked_musr/paper_v1_20260815`.

`configs/innovation/final_frozen.yaml` remains the immutable 2026-08-09 decision
snapshot. The additive 2026-08-14 simplification decision is frozen separately
in `configs/innovation/repair_simplification_source_loso_gpu0_3.yaml` and its
authenticated aggregate gate. The source prior-art comparison is frozen in
`configs/innovation/prior_art_overlap_source_loso_gpu0_3_v2.yaml`; the two
diagnostic configurations are `prior_art_overlap_multimodal_ood_gpu0_3_v2.yaml`
and `prior_art_overlap_language_ood_gpu0_3_v2.yaml`. Their scope remains
`development_ood_diagnostic_only`. Git metadata is unavailable to the runner,
so manifests record `UNKNOWN`; this remains a reproducibility limitation rather
than being replaced with a fabricated commit.
