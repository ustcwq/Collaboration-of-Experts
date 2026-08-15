# Bench-CoE innovation repository and leakage audit

Audit date: 2026-08-08

## 1. Repository state and applicable instructions

- Workspace root: `/home/sm5/ys/FCS`.
- `git status` fails because this directory is not a Git work tree. Current commit and dirty-tree state are therefore `UNKNOWN`.
- No root or `bench_coe/` `AGENTS.md` existed before this audit. The only discovered instruction files are under unrelated vendored `transformers/` and `vllm/` trees and do not apply to `bench_coe/`.
- Existing Bench-CoE, Improve5, Improve6, and output files were not modified by Prompt 0.

## 2. Relevant code entry points

| Responsibility | Path and lines |
| --- | --- |
| Experiment case registry | `bench_coe/improve2_capability_routing_experiments.py:42-247` |
| JSON/JSONL and benchmark cache loaders | `bench_coe/improve2_capability_routing_experiments.py:267-480` |
| Correctness materialization | `bench_coe/improve2_capability_routing_experiments.py:483-504`; `bench_coe/offline_router_innovation_experiments.py:195-207` |
| Answer normalization and output statistics | `bench_coe/improve4_failure_modeling_experiments.py:41-133` |
| Output-partition features and KNN | `bench_coe/improve4_failure_modeling_experiments.py:136-190` |
| DARE Reliability | `bench_coe/improve5_failure_ecology_experiments.py:231-279` |
| Raw correction graph | `bench_coe/improve5_failure_ecology_experiments.py:170-181` |
| RepairChain | `bench_coe/improve6_adaptive_failure_ecology_experiments.py:219-252` |
| Improve5 target loading/evaluation | `bench_coe/improve5_failure_ecology_experiments.py:710-815` |
| Improve6 target loading/evaluation | `bench_coe/improve6_adaptive_failure_ecology_experiments.py:677-789` |
| Expert-pool configuration | `bench_coe/configs/expert_pools.json:1-83` |

The Improve5/6 call chain is `main -> select_cases -> run_case -> load_full_predictions -> bool_matrix/complete_models -> method builder -> choices_from_scores -> evaluate_choices -> choices/results/manifest`. Improve6 imports DARE and the raw correction graph from Improve5 and adds `repair_chain_choices`.

## 3. Cache inventory

The following are the primary compatible caches for the first implementation. Counts are from the cached prediction lists and existing run manifests, not inferred benchmark sizes.

| Dataset/split | Cache | Samples | Complete cached experts | Format |
| --- | --- | ---: | ---: | --- |
| MMMU-Pro validation-id | `outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/validation_id/<model>/predictions.json` | 577 | 13 raw; 11 in published Improve5/6 cohort | JSON list |
| MMMU-Pro test-id | `outputs/multimodal_babyvision_models/mmmu_pro/standard_10_options/test_id/<model>/predictions.json` | 1,153 | 13 raw; 11 common cohort | JSON list |
| MathVista testmini | `outputs/multimodal_babyvision_models/mathvista/testmini/<model>/predictions.json` | 1,000 | 13 raw; 11 common cohort | JSON list |
| CMMMU val | `outputs/multimodal_babyvision_models/cmmmu/val/<model>/predictions.json` | 900 | 13 raw; 11 common cohort | JSON list |
| MMLU-Pro validation language | `outputs/bench_coe/mmlu_pro_validation_single_models/<model>/CoT/validation/*.json` | 70 | 14 in the fixed language cohort | JSON lists by subject |
| BBH | `outputs/model_benchmarks/official_code_local_models/bbh/<model>/predictions.jsonl` | 6,511 | 14 fixed language cohort | JSONL |
| GPQA cached target | `outputs/model_benchmarks/official_code_local_models/gpqa/<model>/predictions.jsonl` | 4,768 in existing manifest | 14 fixed language cohort | JSONL |
| MMStar text-only | `outputs/model_benchmarks/official_code_local_models/mmstar_text_only/<model>/predictions.jsonl` | 1,500 | 14 fixed language cohort | JSONL |
| GAOKAO-MM | `outputs/model_benchmarks/autonomous_remaining_full_20260802/vision/gaokao_mm/<model>/*_2010-2023_*.json` | per-question slot expansion; exact common count `UNKNOWN` until canonical load | model-specific JSON |

Derived/collated caches include:

- `outputs/bench_coe/mmmu_pro_source_matrices/mmmu_pro_source_correctness_matrix.csv`: 104,617 bytes, with `id`, `subject`, gold `answer`, and 11 expert correctness columns.
- `outputs/bench_coe/mmmu_pro_val_source_improve5/*/choices_by_method.json`: 340-651 KB per target case.
- `outputs/bench_coe/mmmu_pro_val_source_improve6/*/choices_by_method.json`: 489-926 KB per target case.
- `outputs/bench_coe/ALL_EXPERIMENTS_RESULTS.csv`: consolidated historical results.
- `outputs/bench_coe/improve5_improve6_all_experiment_results.md`: expanded Improve5/6 report.

## 4. Field availability

For the four primary multimodal caches, every row has `id`, a dataset-specific subject/category field, `answer`, `prediction`, `response`, `is_correct`, and `model_latency_seconds`. Explicit `uncertainty` and `judge` fields are absent.

| Canonical field | Status | Evidence |
| --- | --- | --- |
| question_id | AVAILABLE as `id`/`question_id` | ID resolution: `bench_coe/offline_router_innovation_experiments.py:195-199` |
| dataset/split | AVAILABLE but some cached split metadata is inconsistent with directory naming | Example validation-id row reports `split=test`; path is authoritative pending review |
| subject | AVAILABLE through `subject`, `domain`, `category`, or `task` | Case group keys: `bench_coe/improve2_capability_routing_experiments.py:42-247` |
| model | AVAILABLE from parent model directory | Loader: `bench_coe/improve2_capability_routing_experiments.py:290-328` |
| raw_answer | AVAILABLE as `response` or `model_outputs` | `bench_coe/improve4_failure_modeling_experiments.py:112-115` |
| normalized_answer | DERIVED, not stored | `bench_coe/improve4_failure_modeling_experiments.py:86-100` |
| correctness | AVAILABLE as `is_correct`, otherwise derived from prediction and gold | `bench_coe/offline_router_innovation_experiments.py:202-207` |
| uncertainty | DERIVED lexical count; explicit field absent | `bench_coe/improve4_failure_modeling_experiments.py:41-52,116-128` |
| judge | ABSENT in the primary exact-match caches | BabyVision judge caches are separate and not selected for the initial exact-answer protocol |
| cost | PARTIAL as per-row latency; standardized monetary/call cost absent | `model_latency_seconds` is present in the four primary caches |

Lexical uncertainty is almost degenerate: MMMU-Pro validation has 6 nonzero expert-query values out of 7,501; MMMU-Pro test-id 3/14,989; CMMMU 0/11,700; MathVista 175/13,000. It must be optional and cannot carry the method.

## 5. Answer normalization and equivalence

The current normalization prefers `pred`/`prediction`, otherwise extracts an option letter from output text, lowercases, collapses whitespace, strips punctuation, maps empty output to `<empty>`, and truncates to 48 characters (`bench_coe/improve4_failure_modeling_experiments.py:86-100`). Equality is exact equality of these normalized strings within a query. GAOKAO-MM has a separate uppercase-list normalizer (`bench_coe/improve2_capability_routing_experiments.py:380-419`).

This is not semantic equivalence. Numeric formatting, units, multi-answer ordering, and judge-based equivalence remain dataset-specific. Cross-query and cross-benchmark answer comparison is invalid.

## 6. Expert pools and intersections

The initial multimodal common cohort is the 11 experts recorded in `outputs/bench_coe/mmmu_pro_val_source_improve5/mmmu_pro_val_to_mathvista/manifest.json`: InternVL3_5-2B, Kimi-VL-A3B-Instruct, LFM2.5-VL-1.6B, MiniCPM-V-4.6, Qwen3-VL-2B-Instruct, Qwen3-VL-2B-Thinking, Qwen3.5-2B, SmolVLM2-2.2B-Base, SmolVLM2-2.2B-Instruct, gemma-4-E2B-it, and glm-edge-v-2b. Qwen3-VL-4B-Instruct and Qwen3.5-9B are present in raw caches but excluded from that fixed cohort.

The fixed language cohort contains 14 experts and is recorded in `outputs/bench_coe/improve5_failure_ecology_exclude_qwen35_deepseek_qwen3/portfolio_to_bbh/manifest.json`. Family labels are not fully configured and require an auditable mapping before family interventions.

## 7. Source/target status

- Development source: MMMU-Pro validation-id and language source portfolios assembled from already evaluated MMLU-Pro validation/BBH/GPQA/MMStar caches.
- Source validation: pre-registered source folds or leave-one-subject/environment-out partitions constructed only from development source.
- Development OOD: MMMU-Pro test-id, MathVista testmini, CMMMU val, BBH, GPQA, MMStar, and GAOKAO targets. These results have already been inspected repeatedly and are not blind tests.
- Locked final test: `UNKNOWN`. No untouched benchmark/split with the required complete expert cache is identified.

## 8. Leakage and validity risks

1. **Interface-level target-label exposure (BLOCKER for new methods).** `run_case` builds `target_bool_raw`, `target_matrix`, target-best and oracle before invoking methods, while passing full target records containing `answer` and `is_correct` (`improve5...py:711-743`; `improve6...py:678-710`). Existing method bodies appear to use only target outputs, but the interface does not enforce this claim.
2. **Evaluator and selector share one process/object graph.** Target-best and oracle are available before predictions are written (`improve5...py:734-775`; `improve6...py:701-748`). A prediction-before-label firewall is absent.
3. **Possible self-neighbor leakage.** Any source-on-source use of the existing KNN helpers must explicitly exclude the row itself. Current generic local scoring does not encode an OOF row-ownership contract.
4. **Source statistics are commonly in-sample.** Raw global accuracy, correction edges, and fingerprints are calculated on all source rows. They are valid for a disjoint target but not for source validation without cross-fitting.
5. **No source/target ID-overlap guard.** Existing loaders do not namespace all IDs and do not reject overlap. Portfolio merging does prefix IDs (`improve2...py:443-465`), but ordinary cases do not.
6. **Missing output can be conflated with an answer.** The existing normalizer emits `<empty>`, which can form a shared answer group instead of a missing mask.
7. **Tie-breaking is model-order dependent.** Models are sorted, then `argmax`-style selection implicitly favors the first expert. This is deterministic but must be reported.
8. **Unlabeled transductive preprocessing exists.** FATE normalizes stacked source and target features before clustering (`improve5...py:149-157`). It does not use target labels, but it is transductive and must not be presented as online evaluation.
9. **Judge leakage is `UNKNOWN` outside the primary caches.** BabyVision swap outputs mention conservative missing judgments in `bench_coe/configs/expert_pools.json`; they must not be mixed with exact-match gold without a separate protocol.
10. **Development-set reuse.** MathVista, CMMMU, MMMU-Pro test, BBH, GPQA, MMStar, and Gaokao results informed method design. They cannot select final hyperparameters or serve as untouched confirmation.

## 9. What can run from cache

Canonicalization, source cross-fitting, all requested voting/KNN/Improve5/Improve6 baselines, DCRG, CPI pool interventions, TopoMix feasibility/weighting, probe-coreset simulation, and sequential-acquisition simulation can operate on existing prediction caches. No model forward pass is required for those offline experiments.

New expert inference, a truly untouched final benchmark, semantic judging where exact-match equivalence is inadequate, and calibrated real deployment cost require new data or execution. Monetary cost is unavailable; cached latency is hardware/run dependent.

## 10. Available validation commands

- Import/compile: `python -m compileall bench_coe`
- Built-in tests to be added under `tests/innovation/` and run with `python -m unittest discover -s tests/innovation -v`.
- `pytest` is not installed as of this audit.
- Small cached run entry points: `python -m bench_coe.improve5_failure_ecology_experiments --help` and the corresponding Improve6 module.

## 11. Blocking unknowns

- No Git metadata at the workspace root.
- No untouched locked final test is identified.
- Complete, verified expert-family mapping is absent.
- Dataset-specific semantic answer equivalence is incomplete.
- Standardized inference cost, GPU memory, and monetary cost are unavailable.
- The exact provenance of all historical coefficient choices is not machine-readable.

These unknowns do not block cached development experiments, but the first two block a compliant final confirmatory claim.
