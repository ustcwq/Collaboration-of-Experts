# Prior-art overlap requirement traceability

This matrix maps every actionable point in
`BENCH_COE_IMPROVE5_6_PRIOR_ART_OVERLAP_REVIEW.md` to the implementation and
executed evidence. “Complete” means implemented and run; it does not mean the
method passed its scientific gate.

| Review item | Implementation / evidence | Status |
| --- | --- | --- |
| Reclassify Improve5/6 as post-hoc response selection | Protocol, result report, full-call accounting in `run_prior_art_overlap.py` | Complete |
| Do not claim one-call Bench-CoE cost | `fast_global_best_single_call` is separated from all-output `global_best_posthoc`; 11/14-call reports | Complete |
| Reposition Improve5 as prior-art baseline | Renamed OPRS in docs and `oprs_robust_output_profile` artifacts | Complete |
| Global Best | Post-hoc and true fixed one-call variants | Complete, all source/OOD targets |
| Majority / Answer Support | `majority_answer_support` | Complete, all source/OOD targets |
| OLA / LCA | `ola_metadata`, `lca_support_class` | Complete, all source/OOD targets |
| MCB-DCS | `mcb_dcs_structured` | Complete, all source/OOD targets |
| KNOP | `knop_output_profile`; K = 8/16/32/64 panel | Complete, all source/OOD targets |
| KNORA-U / KNORA-E | `knora_u_output_profile`, `knora_e_output_profile` | Complete, all source/OOD targets |
| META-DES | `meta_des_logistic` | Complete, all source/OOD targets |
| MORE-style selector | Structured and local MiniLM-response variants | Complete, all source/OOD targets |
| Smoothie GLOBAL / LOCAL | Spectral and MiniLM-response GLOBAL/LOCAL variants | Complete, all source/OOD targets |
| Uncertainty-only | `uncertainty_only`; exact lexical proxy documented | Complete, all source/OOD targets |
| Agreement x Global | `agreement_x_global` | Complete, all source/OOD targets |
| Local KNN only | `local_knn_only` | Complete, all source/OOD targets |
| Global + Local rank | `global_local_rank` | Complete, all source/OOD targets |
| Learned logistic / MLP selector | `learned_logistic_selector`, `learned_mlp_selector` | Complete, all source/OOD targets |
| Clarify uncertainty definition | `data.lexical_uncertainty`: `log1p` phrase count; selector scale/clip stated | Complete |
| Do not interpret C as a transition probability | FCRG terminology plus row/column-sum diagnostics | Complete |
| Do not call propagation a real correction chain | Renamed FCRG; bounded-depth linear evidence language throughout | Complete |
| Self-loop ambiguity | `fcrg_no_self` and diagonal diagnostics | Complete, all source/OOD targets |
| Static centrality explanation | `fcrg_column_mean_only` | Complete, all source/OOD targets |
| Row/column normalization and row softmax | `fcrg_row_normalized`, `fcrg_column_normalized`, `fcrg_row_softmax` | Complete, all source/OOD targets |
| Directionality | `fcrg_symmetric` | Complete, all source/OOD targets |
| Random graph null | `fcrg_random_edges`, preserving each row’s off-diagonal weights | Complete, four seeds |
| Degree-preserving/node-relabel null | `fcrg_degree_relabel` | Complete, four seeds |
| G/A/L/column/H1/H2/H1+H2 | Seven explicit score ablations | Complete, all source/OOD targets |
| No A/U, no L, no G | Three explicit score ablations | Complete, all source/OOD targets |
| Query failure conditioning necessity | `fcrg_no_failure_conditioning` and `fcrg_no_a_no_u` | Complete, all source/OOD targets |
| Hops 1-5 / over-smoothing | `fcrg_depth_1` through `fcrg_depth_5` | Complete, all source/OOD targets |
| Fixed versus learned weights | Inner-environment OOF logistic `fcrg_learned_weights` | Complete, no target fitting |
| Multi-seed uncertainty | Four seeds per stage, paired bootstrap, exact McNemar, Holm correction | Complete |
| Per-dataset ID/OOD totals | Source plus six OOD datasets in aggregate CSV/JSON | Complete |
| Different expert pools | Original four-expert pool and exact Qwen-VL swap | Complete on source and three multimodal OOD targets |
| Inference cost and latency | Nominal calls, cached serial latency, explicit unavailable-language latency | Complete |
| Fast versus full working modes | Fixed one-call fast path and full post-hoc selectors | Complete |
| Low-confidence rescue path | Five frozen lexical-uncertainty cascade thresholds | Complete, exploratory |
| Target labels excluded from fitting/selection | Physical label-free exports; all target predictions hashed before label adapter creation | Complete |
| Missing outputs are not answer clusters | Canonical schema, pool masks, regression test, complete-row-only legacy equivalence | Complete |
| Raw answer identity not compared across queries | Query-local clusters; MiniLM KNN uses pairwise relation profiles | Complete |
| Honest claim language and names | OPRS/FCRG wording in protocol, results, and final report | Complete |
| New untouched OOD after method freeze | No such compatible cached dataset exists; all six known OOD datasets are marked diagnostic | External condition unavailable |
| Final locked test | Blocked by source NO-GO and absence of a genuinely untouched target | Correctly not run |

## Coverage anchors

- Source `coverage.json`: 81 authenticated methods, all 23 baseline and 25
  FCRG requirements present.
- Each multimodal target: 81 methods, no required method missing.
- Each language target: 57 methods, no required method missing.
- Test receipts: 59 passing tests for the source snapshot and 60 for the final
  OOD snapshot.
- Independent integrity pass: 2,165 files / 16.554 GiB / zero SHA-256
  mismatches.

## Scientific disposition

Implementation coverage is complete, but the scientific decision is negative.
The full FCRG fails the source gate, does not consistently beat adapted
MORE/Smoothie or simple score variants, and is unstable across targets and
expert pools. The matrix therefore supports a complete negative result, not a
claim that every requested experiment favored the proposed method.
