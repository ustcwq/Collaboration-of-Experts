from __future__ import annotations

import random
import re
from typing import Any


def infer_option_count(row: dict[str, Any]) -> int | None:
    options = row.get("options", row.get("choices"))
    if isinstance(options, list) and options:
        return len(options)
    question = str(row.get("question", row.get("prompt", row.get("input", ""))) or "")
    labels = set()
    for match in re.finditer(r"\b([A-J])\s*[:.)]", question):
        labels.add(match.group(1))
    if labels:
        return max(2, len(labels))
    return None


def paired_bootstrap_delta(
    choices: dict[str, str],
    baseline_model: str,
    target_matrix: dict[str, dict[str, bool]],
    target_ids: list[str],
    iters: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    deltas = []
    per_item = []
    repair = harm = 0
    for rid in target_ids:
        model = choices[rid]
        routed = 1.0 if target_matrix[model].get(rid, False) else 0.0
        base = 1.0 if target_matrix[baseline_model].get(rid, False) else 0.0
        delta = routed - base
        per_item.append(delta)
        repair += int(delta > 0)
        harm += int(delta < 0)
    n = len(per_item)
    if n == 0:
        return {}
    mean = sum(per_item) / n
    if iters <= 0:
        lo = hi = mean
    else:
        for _ in range(iters):
            total = 0.0
            for _j in range(n):
                total += per_item[rng.randrange(n)]
            deltas.append(total / n)
        deltas.sort()
        lo = deltas[int(0.025 * (len(deltas) - 1))]
        hi = deltas[int(0.975 * (len(deltas) - 1))]
    return {
        "paired_delta": mean,
        "paired_ci_low": lo,
        "paired_ci_high": hi,
        "repair_count": repair,
        "harm_count": harm,
        "net_repair": repair - harm,
        "switch_count": sum(1 for rid in target_ids if choices[rid] != baseline_model),
    }
