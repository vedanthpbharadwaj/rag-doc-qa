"""The RAGAS evaluation flow and quality gate, with a fake judge and fake LLM."""

import json

from langchain_core.language_models import FakeListChatModel

from eval import ragas_eval
from eval.gates import Rule, check_against_baseline
from ragqa.pipeline import NO_ANSWER, RAGPipeline
from ragqa.retrieval import build_retriever


class FakeScorer:
    def __init__(self, faithfulness):
        self.values = list(faithfulness)

    async def score(self, question, answer, contexts):
        return {"faithfulness": self.values.pop(0), "answer_relevancy": 0.9}


def test_gate_detects_regressions():
    rules = [Rule("faithfulness", True, 0.03), Rule("hallucination_rate", False, 0.03)]
    base = {"faithfulness": 0.90, "hallucination_rate": 0.05}
    assert check_against_baseline({"faithfulness": 0.88, "hallucination_rate": 0.07}, base, rules) == []
    failures = check_against_baseline({"faithfulness": 0.80, "hallucination_rate": 0.12}, base, rules)
    assert len(failures) == 2
    assert "dropped" in failures[0] and "rose" in failures[1]


def test_claim_to_question():
    q = ragas_eval.claim_to_question("Statins reduce LDL cholesterol.")
    assert '"Statins reduce LDL cholesterol"' in q


def _pipeline(index, settings, answers):
    return RAGPipeline(build_retriever(index, settings, mode="hybrid", k=3), FakeListChatModel(responses=answers))


def _run(tmp_path, index, settings, monkeypatch, answers, faithfulness, extra_args=()):
    monkeypatch.setattr(ragas_eval, "get_settings", lambda: settings)
    out = tmp_path / "results"
    code = ragas_eval.main(
        ["--dataset", "beir_mini", "--out-dir", str(out), "--baseline", str(tmp_path / "baseline.json"), "--concurrency", "1", *extra_args],
        scorer=FakeScorer(faithfulness),
        pipeline=_pipeline(index, settings, answers),
    )
    summary = json.loads(next(out.glob("*_summary.json")).read_text())
    for f in out.iterdir():
        f.unlink()
    return code, summary


def test_end_to_end_summary_and_gate(tmp_path, index, settings, monkeypatch):
    # 4 test queries: 3 answered (one hallucinated), 1 abstained.
    answers = ["Supported [101].", "Supported [103].", "Supported [106].", NO_ANSWER]
    code, s = _run(tmp_path, index, settings, monkeypatch, answers, [1.0, 0.9, 0.2],
                   ["--update-baseline"])
    assert code == 0
    assert s["questions"] == 4
    assert s["abstention_rate"] == 0.25
    assert s["faithfulness"] == round((1.0 + 0.9 + 0.2) / 3, 4)
    assert s["hallucination_rate"] == round(1 / 3, 4)
    assert (tmp_path / "baseline.json").exists()

    # Same quality -> gate passes.
    code, _ = _run(tmp_path, index, settings, monkeypatch, answers, [1.0, 0.9, 0.2], ["--gate", "--min-faithfulness", "0"])
    assert code == 0

    # Faithfulness collapses -> gate fails.
    code, s = _run(tmp_path, index, settings, monkeypatch, answers, [0.3, 0.2, 0.1], ["--gate", "--min-faithfulness", "0"])
    assert code == 1
    assert s["hallucination_rate"] == 1.0
