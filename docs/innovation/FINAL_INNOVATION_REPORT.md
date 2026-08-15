# Bench-CoE innovation development report

Status: **prior-art source comparison NO-GO; development diagnostics complete;
one-time MuSR secondary test completed with primary superiority not confirmed**

This report consolidates the stepwise innovation implementation after the
adversarial-audit remediation. No innovation passed every original source gate.
An additive experimenter-blind MuSR secondary test was subsequently
preregistered and consumed once; it produced a negative confirmatory decision
and does not establish absence from model pretraining.

## Decision summary

| Branch | Decision | Main evidence |
| --- | --- | --- |
| Canonical data firewall | PASS | Hash-pinned source registry, six physically label-free target caches, separate prediction/evaluation phases, 60 current tests |
| Baseline reproduction | PASS with audited parser differences | Historical MathVista: source-weighted vote 63.70%, RepairChain/FCRG 62.60%, DARE/OPRS 62.10%; development-only |
| DCRG | NO-GO | -3.605 pp macro vs RepairChain; worst subject -22.222 pp |
| CPI-Selector | NO-GO | +0.390 pp vs paired no-intervention, crossed CI [-0.477, +1.256] pp, worst seed-family -1.386 pp |
| Conservative-CPI | NO-GO | -0.153 pp macro vs Source Best; crossed CI [-0.520, +0.260] pp; worst seed-subject -11.111 pp |
| CPI-CE | NO-GO | -0.171 pp macro vs Source Best; crossed CI [-0.563, 0.000] pp; 0 rescues / 4 harms |
| Remaining source factorial | NO-GO | Primary equals Source Best with 0 rescues / 0 harms; raw DRO improves up to +1.603 pp vs mean CE but remains below Source Best |
| Improve6 score simplification | NO-GO; H2 DELETE | H1 is +0.520 pp vs M0 but CI [-1.733, +2.773] pp, worst subject -8.696 pp; cluster H1+A is -1.560 pp; replacement gate fails |
| Prior-art overlap comparison | NO-GO; OOD diagnostic complete | Source FCRG 27.9029% vs Global Best 28.5962%; multimodal diagnostic macro +0.750 pp and language macro +2.788 pp, but FCRG loses on two targets and is beaten by MORE/Smoothie adaptations |
| Latent Failure Archetypes | NOT PROMOTED | Registered as a CPI analysis tool, not an independent claim; CPI source gate failed |
| Higher-order Rescue Motifs | NOT AUTHORIZED | Registered as a DCRG/CPI extension; graph/selector expansion stopped after source NO-GO |
| TopoMix | NOT FEASIBLE | Only one genuine compatible multimodal source environment |
| Repair-Probe Coreset | NOT AUTHORIZED | Requires a selector that passed its source gate |
| Sequential Rescue | NOT AUTHORIZED | Requires a valid frozen full-pool innovation |
| Original locked final test | BLOCKED | No method passed the original source gates |
| Additive MuSR secondary test | COMPLETED; PRIMARY NOT CONFIRMED | H1 50.7937% vs equal-budget source vote 50.1323%; +0.6614 pp, stratified CI [-0.2646, +1.7196], McNemar p=0.3018 |

## What the experiments support

1. A variable-pool DeepSets selector can be exactly invariant to duplicate
   experts and numerically invariant to expert permutation. Exact-clone
   sensitivity is zero and maximum permutation sensitivity is 1.79e-7.
2. Real external-expert replacement is implemented: Qwen 2B experts are swapped
   for cached Qwen 4B/9B experts with their real outputs and fingerprints.
3. Counterfactual interventions improve the controlled mean over no-intervention
   training by 0.390 pp in this source experiment, but the uncertainty and
   worst-family failures prevent a positive performance conclusion.
4. Source Best Single remains stronger than DCRG, CPI, historical Improve5/OPRS
   and historical Improve6/FCRG on
   MMMU-Pro source LOSO. Using all expert forwards does not by itself solve the
   selection problem.
5. A nested source-only abstention threshold permits a replacement on only
   0.91% of seed-query rows, but the 21 retained switches still produce more
   harms (5) than rescues (3). Margin abstention alone does not repair CPI.
6. Categorical CE with an explicit none-correct class fixes the probability
   semantics, but raw accuracy falls to 27.210% and nested calibration yields
   28.423% versus 28.596% for Source Best. The remaining issue is ranking
   information, not merely probability normalization.
7. Worst-subject DRO consistently improves raw categorical selector accuracy by
   0.867-1.603 pp over paired mean-CE cells. The best raw cell reaches 28.163%
   but remains 0.433 pp below Source Best. The predeclared rich+mask+DRO method
   reaches 28.596% only by falling back to Source Best and produces no rescues.
8. Pure one-hop H1 reaches 28.423%, improving the five-term M0/RepairChain by
   0.520 pp and nearly matching Source Best at 28.596%. The gain is not stable:
   its paired CI crosses zero, worst-subject delta is -8.696 pp, and the exact
   Qwen pool-replacement check misses its threshold.
9. The complete source prior-art comparison covers 81 authenticated methods. Full
   FCRG reaches 27.9029% versus 28.5962% for Global Best and 26.8631% for KNOP.
   Its -0.6932 pp primary delta has a paired 95% CI of
   [-2.9463, +1.5598] pp, so the registered source gate is NO-GO.
10. The expanded diagnostic run covers 81 methods on each of three multimodal
    targets and 57 methods on each of three language targets. FCRG has macro
    deltas of +0.750 pp multimodal and +2.788 pp language, but Smoothie LOCAL
    MiniLM has higher multimodal macro accuracy (44.236% vs 44.104%) and MORE
    MiniLM has higher language macro accuracy (44.781% vs 43.858%). MMMU-Pro
    test and MMStar deltas are negative.
11. The later one-time MuSR test preserves the source-only method decision and
    prediction firewall: 14 expert outputs and 11 method predictions were
    hashed before labels were opened. H1 has a small positive primary delta but
    fails both uncertainty criteria, so superiority is not confirmed.
12. Majority vote is the strongest preregistered MuSR method at 58.4656%, with
    local Smoothie MiniLM at 58.2011%. These are secondary test findings, not
    post-hoc replacements for the frozen primary and not reusable development
    evidence on MuSR.

## What did not work

- DCRG's stability filter selected no robust rescue edges. Less conservative
  graphs switched almost everywhere and caused more harms than rescues.
- CPI is seed- and family-sensitive. Seed 20260810 has -0.347 pp full-vs-none
  delta and -1.386 pp worst-family degradation.
- Smaller pools degrade sharply: pool sizes 9/7/5/3 reach
  26.690/27.036/25.217/23.050% compared with 27.773% at size 11.
- Fixed six-subject ablations do not isolate a benefit: linear-full reaches
  34.545%, no-intervention 33.864%, and full interventions 33.636%.
- Conservative-CPI reaches 28.5095% versus 28.5962% for Source Best. Its mean
  macro delta is -0.1530 pp and its worst seed-subject delta is -11.1111 pp.
- CPI-CE predicts none-correct on 80.07% of seed-query rows. Its four calibrated
  replacements are all harmful, despite strict nested source-only calibration.
- The remaining-source primary fallback changes only three answer clusters,
  all incorrect-to-incorrect Sociology changes in one seed. Full four-seed LOSO
  training ablations find no intervention schedule that exceeds Source Best.
- No optional RepairChain score term has a strictly positive paired CI. H2
  changes no full-pool choice relative to H1, while a static column-centrality
  control reaches 28.596% versus 28.423% for the real two-hop graph. The
  answer-cluster H1+A selector reaches only 26.343%.
- In the broader prior-art comparison, full FCRG remains below Global Best,
  static column centrality again reaches 28.596%, the worst-subject delta is
  -7.6923 pp, and the original four-expert pool delta is -2.2530 pp. These
  failures prevent promotion. The subsequently executed known-OOD diagnostic
  matrix is also inconsistent: no-L, learned-weight, random-graph, and deeper
  variants each beat full FCRG in at least one aggregate view, while adapted
  MORE/Smoothie baselines are stronger overall.
- On the one-time MuSR test, H1 is significantly worse after Holm correction
  than full FCRG, majority vote, OPRS, MORE MiniLM, local Smoothie MiniLM, and
  the random-graph control. Its +0.6614-point primary delta is based on only 10
  rescues and 5 harms and is not statistically confirmed.

## Experiments requiring new external conditions

These are prospective directions, not conclusions from the current source data.
The prior-art overlap source comparison completed on physical GPUs 0-3 and
failed its authenticated gate. The six already-known OOD datasets were then run
only as explicitly requested diagnostics; their artifacts state that they do
not override the source gate or authorize a locked test. The remaining
directions require new data or observables:

1. Collect or predeclare at least one additional genuine multimodal source
   environment with the same expert pool. This unlocks source-mixture validation
   and makes robust cross-domain training possible without relabeling a seen
   target as source.
2. Acquire genuinely richer gold-free observables, such as confidence/logit
   traces or source-cross-fitted judge signals. Richer summaries of the current
   answer-letter topology have now been tested and are insufficient.
3. Pre-register another blind benchmark/split before generating any results.
   Do not reuse MuSR, MathVista, CMMMU, MMMU-Pro test, BBH, GPQA or MMStar as
   fresh confirmation.

## Reproducibility anchors

- Final test receipt: `outputs/bench_coe/innovation/receipts/remediation_v5_tests.json`.
- DCRG result: `outputs/bench_coe/innovation/dcrg/source_loso_remediation_v5_20260808`.
- CPI result: `outputs/bench_coe/innovation/cpi_selector/source_loso_remediation_v5_20260808`.
- Conservative-CPI result:
  `outputs/bench_coe/innovation/cpi_conservative/source_loso_gpu4_7_v1_20260809`.
- Conservative-CPI report: `docs/innovation/11_CONSERVATIVE_CPI_RESULTS.md`.
- CPI-CE result: `outputs/bench_coe/innovation/cpi_ce/source_loso_gpu4_7_v1_20260809`.
- CPI-CE report: `docs/innovation/13_CPI_CE_RESULTS.md`.
- Remaining source-factorial result:
  `outputs/bench_coe/innovation/cpi_remaining/source_loso_gpu4_7_v1_20260809`.
- Remaining source-factorial report:
  `docs/innovation/15_REMAINING_SOURCE_RESULTS.md`.
- Audit remediation: `docs/innovation/ADVERSARIAL_AUDIT_REMEDIATION.md`.
- Improve6 simplification result:
  `outputs/bench_coe/innovation/repair_simplification/source_loso_gpu0_3_v1_20260814`.
- Improve6 simplification report:
  `docs/innovation/17_REPAIR_SCORING_SIMPLIFICATION_RESULTS.md`.
- Improve6 simplification audit:
  `docs/innovation/18_REPAIR_SCORING_SIMPLIFICATION_AUDIT.md`.
- Prior-art overlap protocol:
  `docs/innovation/19_PRIOR_ART_OVERLAP_PROTOCOL.md`.
- Prior-art overlap test receipt:
  `outputs/bench_coe/innovation/receipts/prior_art_overlap_complete_v3_tests.json`
  (source) and `prior_art_overlap_complete_v4_tests.json` (OOD).
- Prior-art overlap authenticated output root:
  `outputs/bench_coe/innovation/prior_art_overlap/source_loso_gpu0_3_v2_20260814`.
- Prior-art multimodal OOD root:
  `outputs/bench_coe/innovation/prior_art_overlap/multimodal_ood_gpu0_3_v2_retry1_20260814`.
- Prior-art language OOD root:
  `outputs/bench_coe/innovation/prior_art_overlap/language_ood_gpu0_3_v2_20260814`.
- Prior-art overlap result report:
  `docs/innovation/21_PRIOR_ART_OVERLAP_RESULTS.md`.
- Prior-art overlap requirement matrix:
  `docs/innovation/22_PRIOR_ART_OVERLAP_REQUIREMENTS_MATRIX.md`.
- Locked MuSR protocol: `docs/innovation/28_LOCKED_MUSR_PROTOCOL.md`.
- Locked MuSR result: `docs/innovation/29_LOCKED_MUSR_RESULTS.md`.
- Locked MuSR output root:
  `outputs/bench_coe/innovation/locked_musr/paper_v1_20260815`.
- Locked MuSR evaluation-seal SHA-256:
  `9dc5b5abdd06f70418b1d32cc64cb4650677182e0077d2c906627d2ba1c47c64`.
- Frozen decision config: `configs/innovation/final_frozen.yaml`.
- v4 and v5 produced identical SHA-256 values for all 248 CPI prediction files,
  providing an independent same-seed GPU repeat.

## Method category and submission terminology

Historical artifact names remain unchanged for traceability. New submission
text uses **OPRS** (Output-Profile Reliability Selection) for Improve5/DARE and
positions it as a strong MCB/KNOP-derived baseline. It uses **FCRG**
(Failure-Conditioned Repair Graph) for Improve6/RepairChain. FCRG is bounded-hop
linear rescue-evidence propagation over already generated outputs, not a
transition probability, sequential correction chain, or pre-inference router.

The system therefore has separate cost modes: a true fixed-expert one-call fast
path and full post-hoc response selection that invokes the expert pool. Full
OPRS/FCRG results must not inherit Bench-CoE's one-call cost claim.
