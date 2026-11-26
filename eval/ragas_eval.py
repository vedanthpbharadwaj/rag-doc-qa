"""End-to-end answer quality with RAGAS, gated against a committed baseline.

For each of the 300 SciFact test claims the pipeline retrieves sources and
writes a cited answer; RAGAS then scores it:

- faithfulness: share of the answer's statements supported by the retrieved
  contexts (LLM-as-judge, claim decomposition + NLI)
- answer relevancy: how directly the answer addresses the question

Derived metrics:
- hallucination rate: share of non-abstained answers with faithfulness below
  --hallucination-threshold (default 0.5)
- abstention rate: share of answers that said the sources were insufficient
- citation rate / gold citation rate: answers citing at least one source, and
  at least one source the dataset marks as relevant (no LLM needed)

    python -m eval.ragas_eval --limit 50                 # quick check
    python -m eval.ragas_eval --gate                     # CI: fail on regression
    python -m eval.ragas_eval --update-baseline          # accept current numbers
    python -m eval.ragas_eval --mode vector              # ablation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from ragqa.config import get_settings
from ragqa.data.beir import download_beir, load_beir
from ragqa.embeddings import get_embeddings
from ragqa.index import DocumentIndex
from ragqa.pipeline import RAGPipeline, RAGResponse, build_llm
from ragqa.retrieval import build_retriever

from .gates import Rule, check_against_baseline, load_baseline, write_baseline, write_step_summary

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "eval" / "baselines" / "ragas.json"

GATE_RULES = [
    Rule("faithfulness", higher_is_better=True, tolerance=0.03),
    Rule("hallucination_rate", higher_is_better=False, tolerance=0.03),
    Rule("answer_relevancy", higher_is_better=True, tolerance=0.05),
    Rule("gold_citation_rate", higher_is_better=True, tolerance=0.05),
]


def claim_to_question(claim: str) -> str:
    claim = claim.strip().rstrip(".")
    return f'Does the research support or contradict the claim that "{claim}"? Summarise the evidence.'


class Scorer(Protocol):
    async def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float | None]: ...


class RagasScorer:
    """RAGAS 0.4 metric collections with an OpenAI judge."""

    def __init__(self, judge_model: str, embedding_model: str = "text-embedding-3-small",
                 relevancy: bool = True) -> None:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        client = AsyncOpenAI()
        llm = llm_factory(judge_model, client=client)
        self.faithfulness = Faithfulness(llm=llm)
        self.relevancy = (
            AnswerRelevancy(llm=llm, embeddings=RagasOpenAIEmbeddings(client=client, model=embedding_model))
            if relevancy else None
        )

    async def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        f = await self.faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
        out["faithfulness"] = _clean(f.value)
        if self.relevancy is not None:
            r = await self.relevancy.ascore(user_input=question, response=answer)
            out["answer_relevancy"] = _clean(r.value)
        return out


def _clean(v: Any) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


async def score_all(scorer: Scorer, responses: list[RAGResponse], concurrency: int) -> list[dict[str, float | None]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(r: RAGResponse) -> dict[str, float | None]:
        if r.abstained:
            return {}
        async with sem:
            try:
                return await scorer.score(r.question, r.answer, r.context_texts)
            except Exception as exc:  # a single judge failure shouldn't sink the run
                print(f"  scoring failed: {exc}")
                return {}

    return await asyncio.gather(*(one(r) for r in responses))


def summarise(records: list[dict[str, Any]], hallucination_threshold: float) -> dict[str, Any]:
    n = len(records)
    answered = [r for r in records if not r["abstained"]]
    faith = [r["faithfulness"] for r in answered if r.get("faithfulness") is not None]
    relev = [r["answer_relevancy"] for r in answered if r.get("answer_relevancy") is not None]
    latency = [r["retrieval_ms"] + r["generation_ms"] for r in records]

    def mean(xs):
        return round(statistics.mean(xs), 4) if xs else None

    return {
        "questions": n,
        "faithfulness": mean(faith),
        "answer_relevancy": mean(relev),
        "hallucination_rate": round(sum(f < hallucination_threshold for f in faith) / len(faith), 4) if faith else None,
        "abstention_rate": round((n - len(answered)) / n, 4) if n else None,
        "citation_rate": round(sum(bool(r["citations"]) for r in records) / n, 4) if n else None,
        "gold_citation_rate": round(sum(r["cites_gold"] for r in records) / n, 4) if n else None,
        "median_latency_ms": round(statistics.median(latency), 1) if latency else None,
        "scored": len(faith),
    }


def to_markdown(s: dict[str, Any], meta: dict[str, Any]) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    def num(v):
        return "n/a" if v is None else f"{v:.4f}"

    return "\n".join([
        f"## RAGAS evaluation ({s['questions']} questions, retrieval: {meta['mode']})",
        "",
        f"Generator `{meta['model']}`, judge `{meta['judge_model']}`.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Faithfulness | {num(s['faithfulness'])} |",
        f"| Hallucination rate (faithfulness < {meta['hallucination_threshold']}) | {pct(s['hallucination_rate'])} |",
        f"| Answer relevancy | {num(s['answer_relevancy'])} |",
        f"| Abstention rate | {pct(s['abstention_rate'])} |",
        f"| Answers with citations | {pct(s['citation_rate'])} |",
        f"| Answers citing a gold document | {pct(s['gold_citation_rate'])} |",
        f"| Median latency | {s['median_latency_ms']} ms |",
    ]) + "\n"


def main(argv: list[str] | None = None, scorer: Scorer | None = None, pipeline: RAGPipeline | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--model", help="Generator model (default: LLM_MODEL)")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--no-relevancy", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--hallucination-threshold", type=float, default=0.5)
    parser.add_argument("--min-faithfulness", type=float, default=0.70)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    mode = args.mode or settings.retrieval_mode
    model = args.model or settings.llm_model

    ds = load_beir(download_beir(args.dataset, settings.data_dir))
    items = sorted(ds.test_queries.items(), key=lambda kv: kv[0])
    if args.limit:
        items = items[: args.limit]

    if pipeline is None:
        index = DocumentIndex.load(settings.index_dir, get_embeddings(settings), settings.embedding_model)
        pipeline = RAGPipeline(build_retriever(index, settings, mode=mode),
                               build_llm(model, settings.openai_api_key, settings.llm_temperature))
    if scorer is None:
        scorer = RagasScorer(args.judge_model, relevancy=not args.no_relevancy)

    print(f"Generating answers for {len(items)} questions ({mode} retrieval, {model})...")
    started = time.time()
    questions = [claim_to_question(claim) for _, claim in items]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        responses = list(pool.map(pipeline.answer, questions))

    print("Scoring with RAGAS...")
    scores = asyncio.run(score_all(scorer, responses, args.concurrency))

    records = []
    for (qid, claim), resp, sc in zip(items, responses, scores, strict=True):
        gold = set(ds.qrels.get(qid, {}))
        records.append({
            "id": qid, "claim": claim, **asdict(resp),
            "faithfulness": sc.get("faithfulness"), "answer_relevancy": sc.get("answer_relevancy"),
            "gold_doc_ids": sorted(gold), "cites_gold": bool(gold & set(resp.citations)),
        })

    summary = summarise(records, args.hallucination_threshold)
    meta = {"mode": mode, "model": model, "judge_model": args.judge_model,
            "hallucination_threshold": args.hallucination_threshold,
            "elapsed_s": round(time.time() - started, 1)}
    md = to_markdown(summary, meta)
    print("\n" + md)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out_dir / f"ragas_{mode}_{stamp}.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in records) + "\n")
    (args.out_dir / f"ragas_{mode}_{stamp}_summary.json").write_text(json.dumps({**summary, **meta}, indent=2))

    failures: list[str] = []
    if summary["faithfulness"] is not None and summary["faithfulness"] < args.min_faithfulness:
        failures.append(f"faithfulness {summary['faithfulness']:.4f} is below the minimum {args.min_faithfulness}")

    baseline = load_baseline(args.baseline)
    if args.update_baseline:
        write_baseline(args.baseline, {k: summary[k] for k in
                                       ("faithfulness", "hallucination_rate", "answer_relevancy", "gold_citation_rate")})
        print(f"Baseline updated: {args.baseline}")
    elif baseline:
        failures += check_against_baseline(summary, baseline, GATE_RULES)
    else:
        md += "\n_No RAGAS baseline committed yet; run with `--update-baseline` to create one._\n"

    if failures:
        md += "\n### ❌ Quality gate failed\n" + "\n".join(f"- {f}" for f in failures) + "\n"
        print("Quality gate failures:\n  " + "\n  ".join(failures))
    elif baseline and not args.update_baseline:
        md += "\n✅ Quality gate passed.\n"
    write_step_summary(md)
    return 1 if (failures and args.gate) else 0


if __name__ == "__main__":
    sys.exit(main())
