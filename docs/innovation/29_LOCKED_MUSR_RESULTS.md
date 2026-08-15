# Locked MuSR confirmatory results

Status: **completed once; primary superiority not confirmed**

Protocol: `configs/innovation/locked_musr_paper_v1.yaml`

Output root:
`outputs/bench_coe/innovation/locked_musr/paper_v1_20260815`

## Confirmatory conclusion

The frozen primary method, one-hop FCRG (`fcrg_h1_primary`), reaches 50.7937%
on 756 unique MuSR test questions. The equal-budget, 14-call source-weighted
vote reaches 50.1323%. The paired gain is +0.6614 percentage points, with 10
rescues and 5 harms, but the within-task-stratified paired 95% bootstrap
interval is [-0.2646, +1.7196] points and the two-sided exact McNemar p-value
is 0.3018. The preregistered superiority rule is therefore not met.

Majority vote is the strongest preregistered method at 58.4656%, followed by
local Smoothie MiniLM at 58.2011%. These are retained as secondary findings;
neither may replace the frozen primary method after label access.

## Data, pool, and statistical unit

- Source calibration: 70 labeled MMLU-Pro validation questions.
- Locked target: MuSR test, 756 questions: 250 murder mysteries, 256 object
  placements, and 250 team-allocation questions.
- Statistical unit: one unique question. No prompt replica, model, task, or
  retry is counted as an independent observation.
- Expert pool: 14 fixed language models for every equal-budget method.
- Decoding: temperature 0, top-p 1, 4,096-token context, 512-token output cap,
  seed 20260815, and the preregistered prompt and parser.
- Hardware: physical GPUs 0-3 only. Unrelated jobs were not preempted.

## Accuracy

| Method | Calls | Correct | Accuracy | Wilson 95% CI | Task macro | Delta vs source vote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source Best Single | 1 | 378 | 50.0000% | [46.4449, 53.5551]% | 50.0000% | -0.1323 pp |
| Source-weighted vote | 14 | 379 | 50.1323% | [46.5765, 53.6867]% | 50.1333% | 0.0000 pp |
| Majority vote | 14 | 442 | 58.4656% | [54.9187, 61.9269]% | 58.4896% | +8.3333 pp |
| KNOP output profile | 14 | 398 | 52.6455% | [49.0819, 56.1823]% | 52.6417% | +2.5132 pp |
| OPRS output profile | 14 | 423 | 55.9524% | [52.3923, 59.4523]% | 55.9688% | +5.8201 pp |
| MORE MiniLM | 14 | 430 | 56.8783% | [53.3220, 60.3650]% | 56.8958% | +6.7460 pp |
| Smoothie local MiniLM | 14 | 440 | 58.2011% | [54.6524, 61.6668]% | 58.2354% | +8.0688 pp |
| Full FCRG | 14 | 416 | 55.0265% | [51.4638, 58.5383]% | 55.0385% | +4.8942 pp |
| One-hop FCRG, primary | 14 | 384 | 50.7937% | [47.2349, 54.3443]% | 50.7938% | +0.6614 pp |
| FCRG column centrality | 14 | 380 | 50.2646% | [46.7081, 53.8183]% | 50.2667% | +0.1323 pp |
| FCRG random graph | 14 | 424 | 56.0847% | [52.5250, 59.5827]% | 56.1052% | +5.9524 pp |

Source Best Single is a one-call efficiency reference. Every other row uses
the same 14 generated expert outputs, so the primary and secondary full-pool
comparisons are compute matched.

## Paired comparisons

| Candidate versus reference | Delta | Stratified 95% CI | Rescue / harm | Exact p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 vs source-weighted vote, primary | +0.6614 pp | [-0.2646, +1.7196] | 10 / 5 | 0.3018 | N/A |
| H1 vs full FCRG | -4.2328 pp | [-6.6138, -1.9841] | 25 / 57 | 0.0005347 | 0.001604 |
| H1 vs majority vote | -7.6720 pp | [-10.8466, -4.4974] | 47 / 105 | 2.900e-6 | 2.320e-5 |
| H1 vs KNOP | -1.8519 pp | [-4.1005, +0.3968] | 31 / 45 | 0.1354 | 0.2708 |
| H1 vs OPRS | -5.1587 pp | [-7.8042, -2.5132] | 34 / 73 | 0.0002064 | 0.001238 |
| H1 vs MORE MiniLM | -6.0847 pp | [-8.8624, -3.3069] | 38 / 84 | 3.792e-5 | 0.0002654 |
| H1 vs Smoothie local MiniLM | -7.4074 pp | [-11.2434, -3.4392] | 90 / 146 | 0.0003239 | 0.001295 |
| H1 vs column centrality | +0.5291 pp | [-0.3968, +1.4550] | 9 / 5 | 0.4240 | 0.4240 |
| H1 vs random graph | -5.2910 pp | [-8.0688, -2.6455] | 38 / 78 | 0.0002583 | 0.001291 |

The primary comparison alone controls the confirmatory claim. Holm adjustment
is applied to the preregistered secondary family. H1's small positive primary
delta is consistent across the three task point estimates, but it is too small
and uncertain to establish superiority.

## Prediction coverage

Every expert artifact contains all 756 question IDs and zero truncated prompts.
The frozen parser accepts between 286 and 756 outputs per model. In particular,
MAmmoTH2 and Aya have 756 valid predictions; Ministral and Yi-1.5 have 754;
Gemma, GLM, and Qwen have 750; Llama has 714; General-Reasoner has 627;
Granite has 595; Baichuan has 532; Yi-9B has 530; Nemotron has 488; and
InternLM3 has 286. Unparseable outputs are retained and handled by the same
preregistered missing-output rule for every selector. Neither the parser nor
the prompts were changed after seeing target outputs or labels.

Yi-1.5 attempt 1 and Llama attempt 1 failed before generation because unrelated
processes acquired GPU memory during engine startup. Their failure logs remain
in `adaptive_queue`. Successful attempt 2 runs used the identical frozen model,
prompt, decoding settings, seed, and question order. No retry was triggered by
accuracy or target correctness.

## Leakage and claim boundary

The question-only artifact and all 14 expert-output files were generated and
hashed before selection. Eleven method predictions were then written and sealed.
Only after prediction authentication did the evaluator create
`label_access_started.json` and read MuSR answers. The persistent ledger records
this experiment ID as consumed and refuses a second evaluation.

This satisfies the stated experimenter-level rule: target question text and
model outputs may be used, while target answers and correctness do not affect
fitting, hyperparameters, method choice, model pool, prompt, parser, seed,
subset, stopping, or prediction. It does not prove that MuSR was absent from
the pretrained models' training corpora. The allowed claim is one
experimenter-blind cross-dataset test with a fixed local model pool, not a
universally best selector or improvement on every dataset.

All MMMU-Pro, MathVista, CMMMU, BBH, GPQA, MMStar, Gaokao, and V1-V4 portfolio
results remain development-only. The strong majority-vote result on MuSR is now
test evidence and cannot be used to tune a replacement method on this split.

## Reproducibility anchors

- Protocol SHA-256:
  `793134eee6106f2e322a80ba911cfef5a74a66f70953a0bafac1982fa1ca0694`.
- Innovation code manifest SHA-256:
  `f8f8e1acc4b282bf5290a77fadb43fc150722ae3fedc4520ef89648717511e46`.
- Preregistration SHA-256:
  `2b6f625b4da00ca04a4f51e456079c5fdcecfe6d85171fffa57c8712ecb5aa2d`.
- Target observable manifest SHA-256:
  `a0c0f62c31625593d92a77d8ab1719ace99bde4500c4c5212d2251c837d395bf`.
- Prediction seal SHA-256:
  `814972827acb4e06bf4109d9c4a54db713136ce590980b8ec2e9fcaff5a4e760`.
- Evaluation seal SHA-256:
  `9dc5b5abdd06f70418b1d32cc64cb4650677182e0077d2c906627d2ba1c47c64`.
- Machine-readable results: `evaluation/method_results.json` and
  `evaluation/comparisons.json` under the output root.
- Per-query audit records: `evaluation/per_query` under the output root.
- Figure: `evaluation/locked_musr_results.png` and the matching PDF.
- Single-use registry: `outputs/bench_coe/innovation/locked_registry.json`.

The runtime could not attach a Git commit because the workspace is not a Git
work tree. Exact code, configuration, data, model identity, prediction, and
evaluation hashes are recorded instead; the absence of Git metadata remains an
explicit reproducibility limitation.
