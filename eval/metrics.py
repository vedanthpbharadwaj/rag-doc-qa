"""Standard IR metrics over ranked document ids and qrels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Mean of every metric@k over the queries in `run`."""
    totals: dict[str, float] = {}
    n = 0
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        for name, fn in METRICS.items():
            for k in ks:
                key = f"{name}@{k}"
                totals[key] = totals.get(key, 0.0) + fn(ranked, rel, k)
    return {key: round(v / n, 4) for key, v in totals.items()} if n else {}
