# Independent adversarial audit

Audit date: 2026-08-08. Three independent read-only agents inspected data
leakage, statistical validity, and engineering/reproducibility. No implementation
was changed in response to these findings, as required by Prompt 12.

This file is the immutable pre-remediation finding record. The authorized fixes,
independent re-audit status, v5 artifact hashes and final NO-GO decisions are in
`docs/innovation/ADVERSARIAL_AUDIT_REMEDIATION.md`.

## Scope-level conclusion

The current DCRG and CPI numbers are source-only negative results. The auditors
did not find the outer CPI/DCRG held-out subject entering its fold's fingerprint,
KNN library, or supervised training rows. Nevertheless, the strict target-label
firewall is not physically enforced, the CPI causal comparison is confounded,
and several pool-shift and statistical contracts are incomplete. No positive
method, OOD, or final claim is publication-ready. The existing NO-GO decisions
remain the conservative action, but their mechanisms should not be over-
interpreted until HIGH findings are fixed and the source experiments rerun.

## Findings

### BLOCKER-1: target labels are present in the prediction process and can be exported as source labels

- **Files:** `bench_coe/innovation/data.py:121`, `data.py:149`, `data.py:176`,
  `data.py:265`, `schema.py:125`, `run.py:101`, `run.py:134`.
- **Trigger:** call `load_observables()` on a target adapter, then inspect
  `adapter._rows_cache`; raw rows still contain `answer` and `is_correct`. Calling
  `load_source_labels()` on the MathVista target adapter also succeeds because
  the adapter has no immutable source/target role.
- **Impact:** current selector objects do not expose these fields and no active
  target tuning was found, but the claimed process-level firewall is false. A new
  runner can fit target labels while satisfying the present type checks.
- **Minimum fix:** generate a physically label-free observable cache; run
  prediction and evaluation in separate processes; bind each adapter to a frozen
  manifest role and make target adapters unable to construct training labels;
  verify dataset/split provenance in every `fit` call.

### HIGH-1: CPI full versus no-intervention is not a controlled comparison

- **Files:** `bench_coe/innovation/run_cpi.py:192`, `run_cpi.py:198`,
  `run_cpi.py:281`, `run_cpi.py:288`, `cpi.py:465`.
- **Trigger:** reconstruct seeds and optimizer steps. Within a fold, `none` and
  `full` use seeds differing by one; full also trains on original plus augmented
  episodes, approximately doubling updates. Each fixed-split ablation receives a
  different initialization.
- **Impact:** the -0.347 pp full-minus-none result mixes intervention effects,
  initialization noise, and training budget. Individual-intervention causal
  interpretations are invalid.
- **Minimum fix:** load an identical initial `state_dict` for paired variants;
  separate initialization/order/intervention RNG streams; equalize optimizer
  updates and batch exposure; save initialization hashes and rerun all seeds.

### HIGH-2: pool-shift prediction can select an unavailable representative expert

- **Files:** `bench_coe/innovation/run_cpi.py:96`, `cpi.py:527`.
- **Trigger:** remove an expert/family or mask an output, then inspect
  `selected_expert_id`. Cluster scoring uses the intervened `PoolExample`, but
  representative selection searches the original full `ObservableQueryBatch`.
- **Impact:** a reported selection may call an expert absent from the simulated
  pool. Accuracy happens to survive when all members of a cluster have consistent
  labels, but availability, family rates, cost, and inconsistent-cluster results
  are wrong. The pool-shift gate is not adequately validated.
- **Minimum fix:** restrict representatives to valid IDs in the intervened
  example and assert membership, or evaluate the selected normalized answer with
  a cluster-level judge.

### HIGH-3: DCRG's internal OOF nuisance feature uses held-out-row correctness

- **Files:** `bench_coe/innovation/dcrg.py:66`, `dcrg.py:94`, `dcrg.py:100`.
- **Trigger:** change another expert's correctness on an internal OOF test row;
  that row's leave-pair-out difficulty and expected correctness change.
- **Impact:** the outer held-out subject remains excluded, but the claimed OOF
  item-difficulty adjustment is not based solely on train-fold parameters and
  deployable observables. Residual-edge mechanism interpretations are invalid.
- **Minimum fix:** use gold-free question/output features, or a nuisance estimate
  learned entirely from the training fold; add a test that changing any OOF test-
  row correctness cannot change its nuisance input/prediction.

### HIGH-4: the implemented known swap is synthetic fingerprint mixing, not a known pool replacement

- **File:** `bench_coe/innovation/cpi.py:280`.
- **Trigger:** inspect `known_swap`: it removes one current token, copies another
  current donor, and averages the first two fingerprint components with those of
  the removed token.
- **Impact:** it does not test the configured Qwen replacement or a real new
  expert, and the training perturbation does not match the claimed stress case.
- **Minimum fix:** freeze explicit source-derived external expert fingerprints
  and swap mappings; preserve the replacement expert's full answer, family,
  validity, uncertainty, cost, and fingerprint without mixing.

### HIGH-5: invariance comparison silently ignores missing samples or clusters

- **File:** `bench_coe/innovation/cpi.py:571`.
- **Trigger:** give `max_probability_difference()` unequal-length prediction
  lists or different cluster-key sets. `zip` truncates and key intersection drops
  mismatches; an empty intersection returns zero.
- **Impact:** a structural alignment failure can pass the `1e-4` gate. The
  current exact-clone construction appears aligned, but the gate implementation
  is not fail-closed.
- **Minimum fix:** require identical question IDs, list lengths, and cluster-key
  sets; treat every mismatch as a hard failure before comparing probabilities.

### HIGH-6: direct gate comparisons lack their registered paired statistics

- **Files:** `bench_coe/innovation/run_dcrg.py:149`, `run_cpi.py:361`,
  `aggregate_cpi.py:55`, `docs/innovation/EXPERIMENT_PROTOCOL.md:18`.
- **Trigger:** inspect generated summaries. `evaluate()` always compares against
  Source Best; DCRG-vs-RepairChain and CPI-full-vs-none have no direct paired
  bootstrap interval, McNemar test, or Holm-adjusted result.
- **Impact:** arithmetic threshold decisions are available, but uncertainty for
  the actual gate contrast and the frozen multiple-comparison protocol are absent.
- **Minimum fix:** align direct comparator predictions by question ID, compute
  paired query/seed bootstrap and exact McNemar, and apply Holm across each
  pre-registered comparison family.

### HIGH-7: test success is not machine-bound to gate decisions

- **Files:** `bench_coe/innovation/run_dcrg.py:172`,
  `aggregate_cpi.py:69`, `aggregate_cpi.py:71`.
- **Trigger:** skip or break unit tests and rerun the aggregator. CPI still writes
  the hard-coded string `21/21 passed`; DCRG does not read a test receipt.
- **Impact:** a future numerically positive result could declare GO despite failed
  leakage/invariance tests.
- **Minimum fix:** emit a machine-readable test receipt containing test IDs,
  exit code, timestamp, and code/config hashes; require and verify it in each gate.

### HIGH-8: CPI aggregation does not authenticate the registered four runs

- **Files:** `bench_coe/innovation/aggregate_cpi.py:34`,
  `aggregate_cpi.py:44`, `aggregate_cpi.py:47`.
- **Trigger:** place any four directories with distinct seed values under the run
  root. The aggregator does not compare exact seeds to frozen config, verify GPUs
  0-3, 30 folds/577 IDs, config/input hashes, prediction hashes, or failed attempts.
- **Impact:** cherry-picked, old, mixed-config, or modified runs can produce an
  apparently formal gate. The current directory happens to contain the intended
  four seeds, but the workflow does not prove it.
- **Minimum fix:** require the frozen config and exact seed/GPU/fold/query sets;
  recompute all prediction hashes and verify common config/input hashes; inventory
  every attempted run.

### HIGH-9: experiment manifests omit the actual prediction-cache hashes

- **Files:** `bench_coe/innovation/artifacts.py:77`, `run.py:110`,
  `run_dcrg.py:138`, `run_cpi.py:346`.
- **Trigger:** inspect any `prediction_manifest.json`; `input_hashes` contains the
  config and family map but not the expert `predictions.json[l]` files. The
  workspace also has no Git commit.
- **Impact:** results cannot be tied to an immutable cache or source snapshot.
- **Minimum fix:** hash every consumed cache file, row count, and relative path;
  store a Merkle/aggregate hash plus an innovation source-tree hash and dependency
  lock when Git is unavailable.

### HIGH-10: observable parsing and evaluation correctness use inconsistent judges

- **Files:** `bench_coe/innovation/data.py:59`, `data.py:105`, `data.py:204`.
- **Trigger:** on MathVista, when cached prediction is null, observable loading
  may recover an answer from response text while evaluation retains cached
  `is_correct=false`. The audit found 1,079 such valid-recovered/false rows among
  11,000; some recovered clusters contain both true and false members.
- **Impact:** prior MathVista development baselines, cluster support, and
  rescue/harm can be biased. The MMMU-Pro source used for formal DCRG/CPI did not
  exhibit this pattern, so their source gates are not directly changed.
- **Minimum fix:** share one benchmark-aware parser/judge. Either keep a null
  cached prediction invalid, or rejudge the recovered answer against gold only in
  the isolated evaluator and record the judge version.

### MEDIUM-1: missing correctness is silently converted to incorrect

- **Files:** `bench_coe/innovation/selectors.py:27`, `selectors.py:163`,
  `cpi.py:81`.
- **Trigger:** remove one correctness entry; `bool(labels.get(...))` converts
  `None` to false in matrices/fingerprints.
- **Impact:** sparse judging or onboarding data confounds coverage with ability.
  The formal 577x11 source cohort has complete labels, so current gates are not
  directly affected.
- **Minimum fix:** propagate an observed-label mask through means, SVD, KNN,
  graph estimation, and loss, or reject incomplete inputs explicitly.

### MEDIUM-2: long-answer truncation can merge different answers

- **File:** `bench_coe/innovation/data.py:66`.
- **Trigger:** two answers sharing the first 48 normalized characters receive the
  same cluster. The audit found a MathVista example with a mixed-label cluster.
- **Impact:** query-local topology and selection can be wrong even though cluster
  IDs do not leak across queries.
- **Minimum fix:** retain full normalized answers or cluster by a full-content
  hash; add a long-common-prefix regression test.

### MEDIUM-3: DCRG reads fold correctness before all predictions are frozen

- **Files:** `bench_coe/innovation/run_dcrg.py:120`, `run_dcrg.py:128`,
  `run_dcrg.py:134`.
- **Trigger:** per-environment accuracy is computed immediately after each fold,
  before later folds predict and before aggregate prediction hashes are written.
- **Impact:** current code does not feed those metrics into later folds, so no
  observed numeric adaptation occurred; however the documented process-level
  order is false and future edits can create feedback leakage.
- **Minimum fix:** save fold predictions only, then compute all environment
  metrics in a separate post-hash evaluation phase.

### MEDIUM-4: worst-family gate averages away worse seed/environment cases

- **Files:** `bench_coe/innovation/run_cpi.py:397`,
  `aggregate_cpi.py:50`, `aggregate_cpi.py:62`.
- **Trigger:** the aggregator averages each family over seeds, then takes the
  minimum. Individual seed minima are -1.21 to -1.91 pp, while the reported gate
  value is -0.737 pp; held-out-subject minima are not computed.
- **Impact:** the definition of “worst environment” is ambiguous and less
  conservative than its wording. This run remains NO-GO either way.
- **Minimum fix:** freeze the dimensions and gate the minimum over
  seed/family/held-out subject, while retaining pooled means as descriptive.

### MEDIUM-5: single-intervention ablations are not complete LOSO experiments

- **Files:** `bench_coe/innovation/run_cpi.py:268`, `run_cpi.py:271`,
  `run_cpi.py:279`, `docs/innovation/05_CPI_SELECTOR_DESIGN.md:106`.
- **Trigger:** only none/full run all 30 held-out subjects. Each single
  intervention and linear baseline uses one alphabetic-stride six-subject split,
  with the actual subject IDs generated at runtime rather than frozen in config.
- **Impact:** the opposite fixed-split versus LOSO direction cannot identify
  which intervention works, and the requested independent ablations are not
  complete.
- **Minimum fix:** freeze and hash explicit IDs, then run every intervention with
  paired initialization over all four-seed LOSO folds.

### MEDIUM-6: BCE training and softmax calibration have incompatible semantics

- **Files:** `bench_coe/innovation/cpi.py:462`, `cpi.py:485`, `cpi.py:507`,
  `run_cpi.py:120`.
- **Trigger:** train independent per-cluster BCE, then normalize logits with
  softmax and report the maximum as correctness confidence. All-wrong queries are
  forced to total probability one.
- **Impact:** ranking remains usable, but the calibration curve and probability
  interpretation are invalid.
- **Minimum fix:** use mutually exclusive CE with an explicit none-correct class,
  or retain/calibrate sigmoid probabilities on separate source validation data.

### MEDIUM-7: deterministic CUDA execution is not requested

- **File:** `bench_coe/innovation/cpi.py:457`.
- **Trigger:** only RNG seeds are set; deterministic algorithms, cuDNN flags, and
  cuBLAS workspace settings are absent.
- **Impact:** exact reruns can vary across CUDA stacks, which matters near small
  accuracy/invariance thresholds.
- **Minimum fix:** enable deterministic PyTorch mode and required environment
  settings, record them, and add same-seed GPU rerun checks.

### MEDIUM-8: launcher can truncate logs and does not guard GPU occupancy

- **Files:** `scripts/launch_cpi_gpu0_3.sh:6`,
  `scripts/launch_cpi_gpu0_3.sh:17`.
- **Trigger:** rerun with the same root. Shell redirection truncates existing logs
  before the runner rejects an existing output directory. GPU 0-3 processes are
  launched without checking for unrelated compute processes.
- **Impact:** provenance can be destroyed and unrelated jobs can be contended.
- **Minimum fix:** atomically require a nonexistent run root and empty GPUs before
  any redirection/process launch; use a unique run ID.

### MEDIUM-9: CLI paths and environment are not clean-environment reproducible

- **Files:** `scripts/launch_cpi_gpu0_3.sh:4`,
  `configs/innovation/cpi_source_loso.yaml:3`,
  `configs/innovation/cpi_source_loso.yaml:32`.
- **Trigger:** invoke the launcher outside the repository root or without the
  current pre-populated Python environment.
- **Impact:** recorded commands depend on CWD and an undocumented environment.
- **Minimum fix:** resolve repository/config-relative paths, add an install entry
  point and minimal lock file, and document/run a clean-environment smoke test.

### MEDIUM-10: unseen/missing property test does not test a truly unseen expert

- **File:** `tests/innovation/test_cpi.py:75`.
- **Trigger:** inspect the test: it deletes/masks existing experts and constructs
  a pseudo-clone from an existing token; no new expert or unknown family is used.
- **Impact:** unseen-expert support and “missing experts cannot be selected” lack
  regression coverage.
- **Minimum fix:** construct a new expert ID/unknown family with a valid frozen-
  schema fingerprint and assert both forward compatibility and available-pool
  selection, including all-missing edge cases.

### MEDIUM-11: overfit sanity has no pass criterion

- **Files:** `bench_coe/innovation/run_cpi.py:312`, `run_cpi.py:443`.
- **Trigger:** the same-data 100-query runs reach only 33-36% selection accuracy;
  the runner records this but never stops.
- **Impact:** optimization failure, label/selection conversion errors, and an
  information-limited representation are not distinguished before formal runs.
- **Minimum fix:** pre-register a separable synthetic near-100% test and a real-
  data loss/ranking threshold; stop formal evaluation when either fails.

### LOW-1: per-query artifacts omit intervention availability masks

- **File:** `bench_coe/innovation/cpi.py:542`.
- **Trigger:** inspect a CPI prediction JSONL; observable features contain method
  and canonical pool size, not ordered expert IDs, masks, or intervention details.
- **Impact:** frozen outputs cannot independently verify availability legality or
  fully replay pool-shift selections.
- **Minimum fix:** serialize ordered expert/family IDs, valid/missing masks, and
  exact intervention metadata for each selection.

## Required next action

Per Prompt 12, stop after this report. Resolve BLOCKER/HIGH findings only after
explicit authorization, regenerate machine-bound test/provenance receipts, and
rerun source validation before reconsidering any GO or target evaluation.
