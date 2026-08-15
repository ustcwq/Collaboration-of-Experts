# Canonical cache layer and target-label firewall

Implemented: 2026-08-08

## Components

- `CanonicalPredictionRecord`: observable expert output only; its dataclass has no gold or correctness field.
- `ObservableQueryBatch`: deterministic query/expert records, query-local clusters, validity and missing masks.
- `SourceTrainingLabels`: source-only correctness with an enforced `role=source` contract.
- `EvaluationLabels`: separate target-evaluation object rejected by selector fitting.
- `ExpertPool` and `AnswerCluster`: variable pool metadata and query-local answer equivalence.
- `CacheAdapter`: role-capability adapter. Source access requires a hash-pinned
  registry match on role, dataset, split, modality and resolved cache path;
  target access verifies a label-free manifest and all observable-file hashes.
- `DatasetManifest`: typed manifest record for future adapters.

Implementation: `bench_coe/innovation/schema.py`, `bench_coe/innovation/data.py`, and `bench_coe/innovation/features.py`.

## Firewall

Raw historical caches physically contain `answer` and `is_correct`. A trusted
exporter projects targets into separate `observables.jsonl` files containing no
label-like keys. The prediction process can open only those files, while source
and evaluator access are separately authorized through the frozen registry.
Selectors require a real `SourceTrainingLabels`; `EvaluationLabels` is rejected.

The prediction CLI receives no raw target-label path. It writes every prediction
and SHA-256 before a separate evaluator process loads its label-only config.

## IDs, normalization, and clusters

- Canonical ID: `<dataset>::<split>::<raw_id>`.
- Duplicate raw IDs within one expert cache are fatal.
- Source/target overlap within the same dataset is fatal even when split names differ.
- Normalization preserves the historical Improve5/6 rule: select prediction or final output line, lowercase, collapse whitespace, strip surrounding punctuation, and truncate at 48 characters.
- Cluster IDs are constructed from sorted unique normalized answers within exactly one query. Raw answer identity is never compared across queries or benchmarks.
- Invalid, failed, empty, or absent outputs have `valid_output=false`, a `missing_reason`, and no normalized answer or cluster ID.

## Primary-cache field coverage

| Cache | Rows per expert | ID | Subject proxy | Prediction/output | Correctness | Explicit uncertainty | Judge | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MMMU-Pro validation-id | 577 | 100% | 100% | 100% | 100% | 0% | 0% | 100% |
| MMMU-Pro test-id | 1,153 | 100% | 100% | 100% | 100% | 0% | 0% | 100% |
| MathVista testmini | 1,000 | 100% | 100% | 100% | 100% | 0% | 0% | 100% |
| CMMMU val | 900 | 100% | 100% | 100% | 100% | 0% | 0% | 100% |

Uncertainty is a derived lexical observable. It is nonzero for only 6/7,501 MMMU-Pro-validation expert-query pairs, 3/14,989 MMMU-Pro-test pairs, 0/11,700 CMMMU pairs, and 175/13,000 MathVista pairs. Every method must operate when it is absent or zero.

## Tests

Command:

```bash
python -m unittest discover -s tests/innovation -v
```

Twenty-nine tests pass in the v5 receipt. They include target-label rejection,
target-as-source and direct-capability forgery, sanitized-file tampering, schema
field exclusion, query-local clustering, missing outputs, source/target overlap,
OOF row isolation, real replacement experts, paired initialization, CPI
invariances and crossed seed-by-query bootstrap behavior.

## Fields not yet constructible

- A verified semantic judge score shared across datasets.
- Real monetary inference cost and comparable deployment latency.
- Reviewed fingerprints beyond source correctness/topology.
- A locked-test designation backed by an untouched cache.
- A complete adapter for GAOKAO-MM's multi-file, multi-slot format; the existing historical loader remains available but is not yet behind this firewall.
