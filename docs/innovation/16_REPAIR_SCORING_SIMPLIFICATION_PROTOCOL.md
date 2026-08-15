# Improve6 scoring-simplification protocol

Protocol frozen on 2026-08-14 before inspecting any new source-LOSO result.
This is an additive experiment: historical Bench-CoE, Improve5, Improve6,
DCRG, and CPI outputs are immutable.

## Question and stage gate

The experiment asks whether the five-term RepairChain score can be replaced by
one-hop repair evidence, or by one-hop evidence plus query-local answer-cluster
support. It follows the staged rules in
`BENCH_COE_INNOVATION_IDEAS_AND_CODEX_PROMPTS.md`: source validation comes
first, and development OOD is forbidden after a source NO-GO.

MMMU-Pro validation-id is the only genuine compatible multimodal source. Every
one of its 30 subjects is held out once. All graph statistics, global accuracy,
KNN neighbors, and score parameters exclude the outer held-out subject.
`beta` and `alpha` are selected separately inside every outer fold by a full
inner leave-one-subject-out loop over the remaining subjects. The grids are
frozen in the run config, and ties prefer larger one-hop weight.

## Complete method family

| ID | Implemented score |
| --- | --- |
| M0 | `0.30 L + 0.25 H1 + 0.18 H2 + 0.16 A + 0.11 G` |
| M1 | `0.25 H1 + 0.18 H2 + 0.16 A + 0.11 G` |
| M2 | `0.25 H1 + 0.18 H2 + 0.16 A` |
| M3 | nested-source `beta H1 + (1-beta) A` |
| M4 | `H1`, with `A`, then `G`, then expert ID as deterministic ties |
| M5 | nested-source `alpha H1 + (1-alpha) H2` |
| M6 | answer support only |
| M7 | output-profile KNN local competence only |
| M8 | source global accuracy only |

The recommended answer-cluster version is separately implemented as
`m3_cluster_h1_support`: H1 is averaged over the experts in a query-local
cluster and A is cluster size divided by configured pool size. A pure
cluster-mean-H1 diagnostic is also included. Missing outputs never form a
cluster and have support zero.

M0 is checked against the untouched legacy RepairChain selector. The common
baseline set also includes source Best Single, majority, source-weighted vote,
family-balanced vote, Improve5/DARE, random expert, and random answer cluster.

Every M0-M8 formula is also evaluated without retuning its full-pool nested
parameter on two four-expert pools. The original pool is InternVL, LFM,
Qwen3.5-2B, and Gemma; the replacement pool substitutes
Qwen3-VL-2B-Instruct for Qwen3.5-2B. A simplified formula must remain within
0.5 percentage points of the matched M0 in both pools to pass the pool-shift
part of the source gate.

## H2 mechanism controls

The real two-hop graph is compared with:

- row-wise randomized off-diagonal edge weights;
- a symmetric graph;
- a graph with zero self-loops;
- a static graph whose rows all equal the original column means.

H2 is retained only if M5 beats M4, its paired CI is strictly positive, no
source environment degrades, and the real graph beats every registered graph
control. Otherwise the registered decision is DELETE and the method must not be
called RepairChain.

## Simplification gate

M4 is tested first against M0. If it fails, the answer-cluster M3 version is the
only registered fallback. A formula receives GO only when all conditions hold:

- micro delta versus M0 is non-negative;
- worst-subject delta is at least -0.5 percentage points;
- at least two thirds of subjects are non-negative;
- the paired 95% CI lower bound is at least -0.5 percentage points;
- all leakage, determinism, and artifact-authentication tests pass.
- the same formula passes the registered original/replacement judged4 check.

A source NO-GO forbids development-OOD execution. A source GO authorizes a
separate prediction-hash-first development-OOD run; it does not authorize a
confirmatory claim because the repository has no untouched locked test.

## GPU and artifacts

Seeds `20260814..20260817` are mapped one-to-one to physical GPUs 0, 1, 2, and
3. CUDA visibility is asserted inside each process. H1/H2 matrix propagation is
executed in float64 on the assigned GPU; KNN and artifact evaluation remain on
CPU. The four seeds are hardware/reproducibility repeats, not four independent
datasets. Only the randomized-graph and random-choice controls may differ.

Each run writes predictions and SHA-256 hashes before the evaluation-label
adapter is opened. It also saves all per-expert L/H1/H2/A/G components, nested
parameter searches, graph edges, per-query selections, environment metrics,
paired tests, GPU resource use, and a completion manifest. Aggregation rejects
wrong GPU mappings, incomplete query/environment sets, stale test receipts,
code/input hash changes, or non-identical deterministic predictions.
