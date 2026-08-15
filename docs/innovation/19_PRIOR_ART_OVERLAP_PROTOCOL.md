# Improve5/6 prior-art overlap protocol

Status: executed on 2026-08-14. Source decisions remain frozen; known OOD
datasets are diagnostic only.

This protocol implements the actionable requirements in
`BENCH_COE_IMPROVE5_6_PRIOR_ART_OVERLAP_REVIEW.md`. Improve5 and Improve6 are
treated as post-hoc response selectors over already generated expert outputs.
They are not described as pre-inference single-expert routers and do not inherit
the original Bench-CoE one-call cost claim.

## Names and admissible claims

- Improve5/DARE is reported as **OPRS: Output-Profile Reliability Selection**.
  It is an MCB/KNOP-derived output-profile baseline augmented with source-only
  cross-partition reliability and a lexical uncertainty proxy.
- Improve6/RepairChain is reported as **FCRG: Failure-Conditioned Repair
  Graph**. Its operation is bounded-depth linear rescue-evidence propagation,
  not a stochastic transition, random walk, sequential model call, or literal
  correction chain.
- The only potentially distinct FCRG claim is the narrow combination of a
  directed source-domain failure-conditioned rescue graph, query-conditioned
  failure mass, finite-hop propagation, and ranking of already generated
  responses. Conditional complementarity, answer agreement, uncertainty,
  local competence, and graph modeling are acknowledged as prior art.

## Frozen datasets and roles

| Role | Dataset/split | Questions | Experts | Use |
| --- | --- | ---: | ---: | --- |
| Source | MMMU-Pro `validation_id` | 577 | 11 | 30-subject outer LOSO and nested source-only fitting |
| Diagnostic OOD | MMMU-Pro `test_id` | 1,153 | 11 | Known development diagnostic |
| Diagnostic OOD | MathVista `testmini` | 1,000 | 11 | Known development diagnostic |
| Diagnostic OOD | CMMMU `val` | 900 | 11 | Known development diagnostic |
| Language source | MMLU-Pro `validation` | 70 | 14 | Source-only fitting for language diagnostics |
| Diagnostic OOD | BBH `cached_eval` | 6,511 | 14 | Known development diagnostic |
| Diagnostic OOD | GPQA `cached_eval` | 4,768 | 14 | Known development diagnostic |
| Diagnostic OOD | MMStar text-only `test` | 1,500 | 14 | Known development diagnostic |

The six OOD caches had already influenced development. Running them completes
the requested matrix but cannot reverse the source NO-GO, select defaults, or
authorize a locked test.

## Label firewall and provenance

All six targets are physically materialized without `answer`, `gold`,
`target`, `correct`, `correctness`, `is_correct`, or `score`. Their observable
manifests and every exported expert file are SHA-256 bound. For each seed, all
configured target/method predictions are serialized and hashed before an
`EvaluationLabelAdapter` is constructed. Aggregation authenticates every
seed-target prediction package before opening any target labels.

Source-local correctness, graph edges, and competence features are computed
from disjoint folds. The MMMU-Pro source run holds out one of 30 subjects. The
learned FCRG weight model adds an inner subject-LOSO layer. The language source
loader supports the real per-expert `CoT/validation/*.json` layout and hashes
every input shard.

The observable uncertainty is exactly
`log1p(count of fixed uncertainty phrases in raw output)`, then divided by four
and clipped to `[0, 1]` by selectors. It is a reproducible lexical proxy, not
token entropy, sequence likelihood, semantic entropy, or calibrated model
confidence.

## Prior-art baseline coverage

| Review requirement | Registered implementation |
| --- | --- |
| Global Best | `global_best_posthoc`; all-output validity-aware reference |
| True one-call Global Best | `fast_global_best_single_call`; expert fixed before the query |
| Majority / Answer Support | `majority_answer_support` |
| OLA / LCA | `ola_metadata`, `lca_support_class` |
| MCB-DCS / KNOP | `mcb_dcs_structured`, `knop_output_profile` |
| KNORA-U / KNORA-E | `knora_u_output_profile`, `knora_e_output_profile` |
| META-DES | `meta_des_logistic` |
| MORE-style | `more_style_structured`, `more_style_minilm` |
| Smoothie GLOBAL / LOCAL | `smoothie_global_spectral`, `smoothie_local_spectral`, `smoothie_global_minilm`, `smoothie_local_minilm` |
| Uncertainty only | `uncertainty_only` |
| Agreement x Global | `agreement_x_global` |
| Local KNN only | `local_knn_only` |
| Global + Local rank | `global_local_rank` |
| Learned logistic / MLP | `learned_logistic_selector`, `learned_mlp_selector` |
| Improve5 adaptation | `oprs_robust_output_profile` |

The MiniLM variants use the locally cached
`sentence-transformers/all-MiniLM-L6-v2` transformer directly, masked-mean
pool its hidden states, L2-normalize 384-dimensional response embeddings, and
bind the model snapshot hash. Similarities are computed only among responses
to the same query; cross-query KNN uses relation profiles, never raw answer
identity.

These are transparent adaptations to the available caches, not claims of
bit-for-bit official-author-code reproduction. In particular, cached token
logits and calibrated confidence are unavailable.

## FCRG ablations and null controls

The 25 registered variants cover `G`, `A`, `L`, static column mean, `H1`, `H2`,
`H1+H2`, no query failure conditioning, no `A/U`, no `L`, no `G`, no self-loop,
row normalization, column normalization, row softmax, symmetric graph,
row-edge permutation, degree-preserving node relabeling, depths 1-5, and
inner-OOF learned weights.

Depth weights decay by 0.72 and sum to graph mass 0.43; depth 2 is exactly
`(0.25, 0.18)`. Equality to the legacy RepairChain answer is enforced on rows
where every expert output is valid. Incomplete rows are reported separately
because legacy code groups missing strings as answers, which the canonical
protocol explicitly forbids.

## Sensitivity, pool shift, and cost

- KNOP is evaluated at K = 8, 16, 32, and 64 without target-based selection.
- Multimodal runs include the registered original four-expert pool and the
  exact Qwen-VL replacement pool for 12 key methods.
- Full response selectors are charged 11 or 14 model calls. The true fast path
  is charged one call. `global_best_posthoc` is also a full-call method because
  it observes validity and can fall back among generated outputs.
- Five fixed lexical-uncertainty cascade thresholds report the accuracy/call
  trade-off. Triggers inspect only the fast expert output.
- Cached serial latency is reported only where caches contain it; language
  latency is marked unavailable rather than interpreted from stored zeros.

## Source gate and diagnostic exception

The source gate requires the full FCRG to exceed Global Best by 0.25 points,
obey `FCRG > KNOP > Global Best`, beat all graph controls, show a positive H2
increment, meet worst-subject/coverage/CI criteria, and remain within -0.50
points in both replacement pools. The source run failed this gate.

The subsequent OOD runs were executed only because the user explicitly
requested the complete experimental matrix despite that NO-GO. Every OOD
artifact records `development_ood_diagnostic_only`,
`source_gate_overridden=false`, and `can_authorize_locked_test=false`.

## Frozen snapshots

- Source: `configs/innovation/prior_art_overlap_source_loso_gpu0_3_v2.yaml`,
  receipt `prior_art_overlap_complete_v3_tests.json`, code hash
  `ea1ec566fe6aca8e2384ed221451c1c2649fb3e5a72219a21ab1bb14279159a5`.
- OOD: `configs/innovation/prior_art_overlap_{multimodal,language}_ood_gpu0_3_v2.yaml`,
  receipt `prior_art_overlap_complete_v4_tests.json`, code hash
  `8af719bd78ae91aa4fd98eecb71ae820d29bff89ffba775f69b25b08c9d0f065`.
