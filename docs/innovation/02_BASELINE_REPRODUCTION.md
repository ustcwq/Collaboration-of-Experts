# Baseline reproduction and isolated evaluator

Final firewall run:
`outputs/bench_coe/innovation/baseline_reproduction/firewall_smoke_remediation_v5_20260808`

The prediction and evaluation phases must be separate processes:

```bash
python -m bench_coe.innovation.run \
  --config configs/innovation/baseline_smoke.yaml \
  --phase predict

python -m bench_coe.innovation.run \
  --config configs/innovation/baseline_smoke.yaml \
  --evaluation-config configs/innovation/baseline_smoke_evaluation.yaml \
  --phase evaluate
```

The prediction config contains the sanitized MathVista observable path and its
manifest hash, but no raw-label path. The evaluator config contains the raw
cache path and is read only by the evaluation process. All ten method
predictions are serialized and hashed before evaluation starts.

## MathVista development result

Source fitting uses all 577 MMMU-Pro validation-id questions; evaluation uses
all 1,000 MathVista testmini questions.

| Method | Accuracy | Delta vs source best | Rescue | Harm | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source Best | 61.00% | 0.00 pp | 0 | 0 | 1.0000 |
| Majority | 61.30% | +0.30 pp | 86 | 83 | 0.8778 |
| Source-weighted vote | 63.70% | +2.70 pp | 81 | 54 | 0.0249 |
| Family-balanced vote | 61.50% | +0.50 pp | 84 | 79 | 0.7542 |
| Output-profile KNN | 54.10% | -6.90 pp | 61 | 130 | 0.00000066 |
| Global+local | 55.90% | -5.10 pp | 47 | 98 | 0.0000276 |
| DARE / Improve5 | 62.10% | +1.10 pp | 52 | 41 | 0.2997 |
| RepairChain / Improve6 | 62.60% | +1.60 pp | 51 | 35 | 0.1052 |

These are development-OOD results and cannot select the final innovation. The
source-weighted vote remains the strongest result in this comparison, but it was
already observed during development.

## Historical choice comparison

The audited parser keeps a present-but-null cached prediction invalid instead of
recovering a potentially inconsistent answer from response text. Consequently,
the final DARE choices differ from the historical cache on 4/1,000 questions and
RepairChain differs on 3/1,000; the first difference is MathVista question 211.
The older 0-mismatch reproduction is retained as historical evidence but is
superseded because it used the inconsistent parser behavior identified by the
adversarial audit.

## Limitations

- MathVista was repeatedly inspected during development and is not a locked test.
- Family assignments remain manually reviewed configuration, not learned truth.
- Exact cached correctness is used as the evaluator judge; no cross-dataset
  semantic judge is claimed.
- The firewall proves process/config separation for this workflow, not operating-
  system isolation against an actor who can modify source code and all hashes.
