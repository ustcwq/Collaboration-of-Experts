# Bench-CoE Innovation Review Package

This package is a compact, sanitized review bundle for discussing Bench-CoE routing improvements with an external assistant or reviewer.

It intentionally excludes model weights, raw datasets, raw prediction caches, server logs, and large benchmark outputs.

## Contents

- `docs/innovation_method_summary.md`: Method summary, examples, and recommended paper framing.
- `results/materialized_summary.md`: Final heldout comparison table.
- `results/materialized_summary.csv`: Machine-readable version of the final heldout comparison table.
- `scripts/offline_router_innovation_experiments.py`: Offline routing experiment script.
- `scripts/materialize_innovation_strategies.py`: Script for materializing selected strategies into heldout prediction summaries.
- `prompts/chatgpt_pro_review_prompt.md`: Suggested prompt for asking ChatGPT-Pro to critique and improve the method.

## Main Method To Review

The primary proposed contribution is:

```text
Probe-Adaptive Bench-CoE with Paired Local Lower-Confidence Complementarity Gating
```

The key idea is to choose an expert only when local probe evidence indicates that switching from a default strong expert has positive paired counterfactual gain under a lower-confidence bound.

The method should not be framed as a generic kNN router or correctness predictor. The paper claim should focus on:

- paired counterfactual complementarity relative to a default strong expert;
- local target-probe adaptation;
- lower-confidence gating to reduce harmful expert switches;
- clear separation between clean evaluation and target-calibrated evaluation.

## Evaluation Caveat

If probe samples are drawn from a target benchmark test split and labels/correctness are used for routing adaptation, the result should be described as target-calibrated heldout evaluation, not as strict zero-shot leaderboard evaluation.

For clean standard evaluation, routing rules should be learned from a validation/dev/source split, then evaluated once on the untouched target test split.
