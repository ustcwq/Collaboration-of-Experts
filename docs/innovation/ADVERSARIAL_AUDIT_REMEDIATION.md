# Adversarial audit remediation

Remediation date: 2026-08-08

This document records the authorized remediation of all BLOCKER and HIGH
findings in `ADVERSARIAL_AUDIT.md`, followed by independent read-only re-audits
of data leakage, statistics, and engineering. The final authenticated source
runs are v5; earlier runs remain immutable audit history.

## Resolution matrix

| Finding | Resolution |
| --- | --- |
| BLOCKER-1 target labels in prediction process | Resolved. Source access requires a hash-pinned registry role and exact dataset/split/modality/path match. Target access requires a hash-pinned label-free manifest and verifies all expert-file hashes. Prediction and evaluation use separate configs and processes. |
| HIGH-1 confounded CPI comparison | Resolved. Paired variants share initialization hashes, RNG contract, training rows and optimizer-update budget for every fold. |
| HIGH-2 unavailable representative | Resolved. Selection is restricted to valid experts present in the intervened pool; all removed/missing artifacts have zero illegal selections. |
| HIGH-3 DCRG OOF correctness leakage | Resolved. Difficulty was replaced by fold-trained observable topology nuisance features; the OOF-row mutation test passes. |
| HIGH-4 synthetic known swap | Resolved. Qwen 2B-to-4B and 2B-to-9B mappings use real cached replacement outputs and fingerprints without mixing. |
| HIGH-5 fail-open invariance alignment | Resolved. Different sample counts or cluster-key sets are fatal. |
| HIGH-6 missing direct statistics | Resolved. Direct DCRG and CPI contrasts use aligned paired bootstrap, exact McNemar and Holm correction. |
| HIGH-7 unbound tests | Resolved. Runners require a passing receipt bound to current code and every frozen config; the latest remaining-source follow-up has 40 passing tests. |
| HIGH-8 unauthenticated four runs | Resolved. Aggregation requires exact seeds, GPU mapping, 577 questions, 30 environments, common source/code/query hashes, completion manifests and all prediction hashes. |
| HIGH-9 missing cache/code hashes | Resolved. Recursive source-cache, registry, code, config, receipt and prediction manifests are recorded and checked. |
| HIGH-10 parser/judge inconsistency | Resolved. A present-but-null cached prediction remains invalid and is not recovered from response text. |
| Re-audit HIGH: unverified evaluator artifacts | Resolved. Final gate ignores `per_query` and `seed_gate` correctness and recomputes from authenticated selections plus independent labels. |
| Re-audit HIGH: nested bootstrap for crossed data | Resolved. Seeds and shared queries are resampled as crossed factors with one shared query resample per draw. |

## Firewall evidence

- Registry SHA-256:
  `3cc1bf93adc5e7d0b7277447659317184377739d592c68b831dd5768b8cedb79`.
- Sanitized MathVista manifest SHA-256:
  `f8ca7374597dd3019c32a6c6a84aab8c59b4a5e7a5b65c689cea30195f6990ce`.
- The prediction config contains only the sanitized target path. The raw label
  path exists only in `baseline_smoke_evaluation.yaml`.
- Prediction/evaluator phases are independently recorded as
  `prediction_only_process` and `evaluation_only_process`.
- Tests reject raw target-as-source role forgery, direct capability forgery,
  target training-label export and modified observable files.

## Final authenticated artifacts

| Artifact | SHA-256 |
| --- | --- |
| `outputs/bench_coe/innovation/receipts/remediation_v5_tests.json` | `0d688f25384ca4247cb0e6ff2e932de98f31ccfdb02345a46b25363d5c03f790` |
| `outputs/bench_coe/innovation/dcrg/source_loso_remediation_v5_20260808/gate.json` | `621d8901691aa6bc2bdf062527117b5cb9c40bfaf9c4e6c5d6bb49546546129e` |
| `outputs/bench_coe/innovation/cpi_selector/source_loso_remediation_v5_20260808/aggregate/gate.json` | `114225820ba5e7aa86f8d32030edb53a2d829a47459f2e7ed699225cfd0696df` |
| `outputs/bench_coe/innovation/cpi_selector/source_loso_remediation_v5_20260808/aggregate/authenticated_inputs.json` | `597fdd9fefdcea689ff6ba8ceff8a68cd7c46ab71b09d98afcd9a3c86aa8eb1a` |

## Final decisions

- DCRG: **NO-GO**, macro delta -3.605 pp vs RepairChain.
- CPI: **NO-GO**, mean full-vs-none delta +0.390 pp, crossed 95% CI
  [-0.477, +1.256] pp, worst seed-family delta -1.386 pp.
- Conservative-CPI follow-up: **NO-GO**, macro delta -0.153 pp versus
  Source Best, crossed 95% CI [-0.520, +0.260] pp and worst seed-subject delta
  -11.111 pp. Its GPU4-7 execution was an explicitly requested scheduling
  exception and retained the frozen GPU0-3 v5 input hashes.
- CPI-CE follow-up: **NO-GO**, macro delta -0.171 pp versus Source Best,
  crossed 95% CI [-0.563, 0.000] pp and 0 rescues versus 4 harms. The explicit
  none-correct class resolves the BCE/softmax semantic defect but does not
  improve the selector.
- Locked final evaluation was not run because neither innovation passed source
  gates and no genuinely untouched locked dataset exists.

## Residual limitations

The independent re-audits found no remaining BLOCKER in the remediated path.
Residual MEDIUM issues remain: incomplete-label masking is not generalized,
48-character normalization can merge long answers, fixed single-intervention
ablations are not full LOSO, the launcher has a small GPU preflight race, and clean-environment dependency
locking is incomplete. Source labels are loaded into the source runner before
fold iteration, although only training-fold subsets enter each fitted model.
These limitations do not change the conservative NO-GO decisions, but they
prevent a publication claim that the proposed innovations improve performance.
