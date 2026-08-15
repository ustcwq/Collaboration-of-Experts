# Prompt For ChatGPT-Pro

Please review this Bench-CoE routing improvement package from the perspective of a strong machine learning / NLP systems reviewer.

Focus on the proposed method:

```text
Probe-Adaptive Bench-CoE with Paired Local Lower-Confidence Complementarity Gating
```

The method uses cached expert predictions and a small probe/calibration split to estimate paired counterfactual gain for switching from a default strong expert to a candidate expert. It switches only when the local lower-confidence bound of the paired gain is positive.

Please evaluate:

1. Whether the method is sufficiently novel compared with LLM routing, cascade routing, kNN routing, correctness prediction, RouteLLM, FrugalGPT, Routoo/Zooter-like routing, and conformal/risk-aware routing.
2. Whether the contribution should be framed as a new method, a strong empirical protocol, or an analysis tool.
3. Whether the current evaluation protocol has test leakage risk, especially when probe samples come from target test splits.
4. How to redesign the evaluation into a clean setting using validation/dev/source data only.
5. What ablations are necessary:
   - no LCB vs LCB;
   - paired gain vs direct accuracy;
   - local kNN vs group mapping;
   - different probe budgets;
   - target-calibrated vs clean transfer;
   - default expert selection;
   - harm weight lambda;
   - confidence bound z.
6. What theoretical or diagnostic analysis would make the method more convincing:
   - oracle gap;
   - expert complementarity matrix;
   - switch precision;
   - harmful switch rate;
   - probe budget sensitivity;
   - dataset routability.
7. How to rename the method if the current name is too long or too close to prior work.
8. What claims should be avoided to prevent overclaiming novelty.
9. What concrete additional experiments would most improve the chance of acceptance.

Please provide:

- a skeptical reviewer-style critique;
- a revised novelty statement;
- a recommended method name;
- a clean experimental protocol;
- 5 to 10 concrete next experiments in priority order;
- suggested wording for the paper's method and limitations sections.
