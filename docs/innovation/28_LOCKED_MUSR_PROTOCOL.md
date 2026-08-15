# Locked MuSR confirmatory protocol

Status: **completed once; primary superiority not confirmed**

Protocol: `configs/innovation/locked_musr_paper_v1.yaml`

## Purpose and claim boundary

All MMMU-Pro test, MathVista, CMMMU, BBH, GPQA, MMStar, and V1-V4 portfolio
results have already influenced development. They remain useful development
evidence but cannot support a new confirmatory claim. This protocol therefore
uses the locally available MuSR test set as an experimenter-untouched secondary
test. A workspace-wide output search found no prior MuSR prediction or score
artifact before preregistration.

This is not a claim that MuSR was absent from any model's pretraining corpus.
The allowed claim is narrower: one experimenter-blind cross-dataset test with a
fixed local model pool and no use of target correctness before prediction
sealing.

## Frozen data and model pool

- Source: MMLU-Pro validation, 70 labeled source-calibration questions.
- Target: MuSR test, 756 unique questions: 250 murder mysteries, 256 object
  placements, and 250 team-allocation problems.
- Statistical unit: one unique MuSR question. No prompt replicas, seeds, or
  option permutations are treated as independent samples.
- Expert pool: the same 14 language models used by the completed language OOD
  development protocol.
- Decoding: temperature 0, top-p 1, fixed prompt/parser, 4,096-token context,
  at most 512 generated tokens, and seed 20260815.
- Hardware: physical GPUs 0, 1, 2, and 3 only.

The three raw CSV hashes, source registry hash, family map hash, exact model
names, prompt/parser versions, and generation settings are frozen in the YAML
protocol. Question text and model outputs may be used; target answers and
correctness may not influence fitting, hyperparameters, method choice, model
pool, prompt, parser, seed, subset, stopping, or reporting.

## Frozen methods

The primary method is one-hop FCRG (`fcrg_h1_only`), selected from source-only
simplification evidence before MuSR prediction. Its primary reference is the
14-call source-weighted answer vote (`global_best_posthoc` in historical code;
renamed to avoid implying target-label selection). Both consume the same 14
frozen expert outputs.

Pre-registered secondary methods are majority vote, KNOP, OPRS, MORE with
MiniLM features, local Smoothie with MiniLM features, full FCRG, static FCRG
column centrality, and a randomized-graph negative control. Source Best Single
is a one-call efficiency reference and is not presented as an equal-compute
primary comparison.

## Prediction/evaluation firewall

The experiment has four irreversible stages:

1. A trusted sanitizer emits question-only JSONL. Unit tests verify that
   changing `answer_index` and `answer_choice` leaves this observable artifact
   unchanged.
2. Four GPU workers read only the sanitized JSONL and write label-free expert
   responses. The completed cache is hashed.
3. The selector reads the source labels and label-free target cache, writes all
   pre-registered method predictions, and creates a prediction seal.
4. A separate evaluator verifies every hash, atomically writes a single-use
   label-access marker, then and only then reads MuSR answers. A persistent
   ledger refuses a second evaluation of the experiment ID.

The prediction CLIs do not import the evaluator-only label module. Any code,
config, source registry, observable, expert-output, or selector-output hash
mismatch fails closed.

## Statistical analysis

The sole confirmatory test is one-hop FCRG versus the equal-budget 14-call
source-weighted vote. Superiority requires all three conditions:

- positive micro-accuracy delta;
- two-sided exact McNemar p-value below 0.05;
- positive lower endpoint of a 10,000-draw paired, within-task-stratified
  bootstrap 95% interval.

Secondary comparisons use Holm correction. The report must include all 756
questions, all frozen methods, task-level accuracy, Wilson intervals, rescues,
harms, exact p-values, paired intervals, nominal calls, negative results, and
the pretraining-contamination limitation. No post-hoc winning method may
replace the frozen primary method.

## Execution outcome

The prediction boundary was sealed before target-label access. The one-time
evaluation then consumed all 756 MuSR questions and recorded the experiment in
`outputs/bench_coe/innovation/locked_registry.json`. One-hop FCRG improves the
equal-budget source-weighted vote by +0.6614 percentage points, but its
within-task-stratified 95% interval is [-0.2646, +1.7196] points and exact
McNemar p=0.3018. The frozen superiority rule is not met.

The complete negative result, all secondary comparisons, artifact hashes, and
figure are reported in `docs/innovation/29_LOCKED_MUSR_RESULTS.md`. MuSR test is
now consumed and cannot be reused as a fresh locked test.
