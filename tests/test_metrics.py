import math

import pytest

from eval.metrics import evaluate_run, hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

REL = {"a": 1, "c": 1}
RANKED = ["a", "b", "c", "d"]


def test_precision_recall():
    assert precision_at_k(RANKED, REL, 1) == 1.0
    assert precision_at_k(RANKED, REL, 4) == 0.5
    assert recall_at_k(RANKED, REL, 1) == 0.5
    assert recall_at_k(RANKED, REL, 3) == 1.0


def test_mrr_and_hit_rate():
    assert mrr_at_k(["x", "a"], REL, 10) == 0.5
    assert mrr_at_k(["x", "y"], REL, 10) == 0.0
    assert hit_rate_at_k(["x", "c"], REL, 1) == 0.0
    assert hit_rate_at_k(["x", "c"], REL, 2) == 1.0


def test_ndcg():
    assert ndcg_at_k(["a", "c"], REL, 2) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], REL, 2) == pytest.approx(expected)
    assert ndcg_at_k(["x"], {}, 1) == 0.0


def test_evaluate_run_averages_and_skips_unjudged():
    run = {"q1": ["a", "b"], "q2": ["z", "y"], "q3": ["a"]}
    qrels = {"q1": {"a": 1}, "q2": {"y": 1}}
    result = evaluate_run(run, qrels, ks=[1, 2])
    assert result["hit_rate@1"] == 0.5
    assert result["mrr@2"] == 0.75
