"""Compare BM25, vector-only and hybrid retrieval on the BEIR test queries.

No LLM calls: relevance comes from the dataset's human judgements (qrels),
so this runs on every push for free.

    python -m eval.retrieval_eval                          # report
    python -m eval.retrieval_eval --gate                   # fail on regression vs baseline
    python -m eval.retrieval_eval --update-baseline        # accept current numbers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ragqa.config import get_settings
from ragqa.data.beir import download_beir, load_beir
from ragqa.embeddings import get_embeddings
from ragqa.index import DocumentIndex
from ragqa.retrieval import build_retriever

from .gates import Rule, check_against_baseline, load_baseline, write_baseline, write_step_summary
from .metrics import evaluate_run

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "eval" / "baselines" / "retrieval.json"
GATED = ["ndcg@10", "precision@5", "recall@10", "mrr@10"]


def run_mode(retriever, queries: dict[str, str]) -> tuple[dict[str, list[str]], float]:
    run: dict[str, list[str]] = {}
    start = time.perf_counter()
    for qid, text in queries.items():
        run[qid] = [d.metadata["doc_id"] for d in retriever.invoke(text)]
    ms_per_query = (time.perf_counter() - start) * 1000 / max(len(queries), 1)
    return run, round(ms_per_query, 1)


def to_markdown(results: dict[str, dict[str, float]], ks: list[int], n_queries: int, n_docs: int) -> str:
    modes = list(results)
    lines = [
        f"## Retrieval evaluation ({n_queries} queries, {n_docs:,} documents)",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for name in ("precision", "recall", "mrr", "ndcg", "hit_rate"):
        for k in ks:
            key = f"{name}@{k}"
            lines.append(f"| {key} | " + " | ".join(f"{results[m].get(key, 0):.4f}" for m in modes) + " |")
    lines.append("| ms / query | " + " | ".join(f"{results[m]['ms_per_query']}" for m in modes) + " |")

    if "hybrid" in results and "vector" in results:
        lines += ["", "**Hybrid vs vector-only:**"]
        for key in ("precision@5", "ndcg@10", "recall@10"):
            v, h = results["vector"].get(key, 0), results["hybrid"].get(key, 0)
            rel = f"{(h - v) / v * 100:+.1f}%" if v else "n/a"
            lines.append(f"- {key}: {v:.4f} -> {h:.4f} ({rel})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--modes", nargs="+", default=["bm25", "vector", "hybrid"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--gate", action="store_true", help="Exit 1 if a gated metric regresses")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ds = load_beir(download_beir(args.dataset, settings.data_dir))
    queries = dict(sorted(ds.test_queries.items(), key=lambda kv: kv[0]))
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])

    index = DocumentIndex.load(settings.index_dir, get_embeddings(settings), settings.embedding_model)
    max_k = max(args.ks)

    results: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        retriever = build_retriever(index, settings, mode=mode, k=max_k)
        run, ms = run_mode(retriever, queries)
        results[mode] = {**evaluate_run(run, ds.qrels, args.ks), "ms_per_query": ms}
        print(f"{mode:>7}: ndcg@10={results[mode].get('ndcg@10')} precision@5={results[mode].get('precision@5')}")

    md = to_markdown(results, args.ks, len(queries), index.num_documents)
    print("\n" + md)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out_dir / f"retrieval_{stamp}.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / f"retrieval_{stamp}.md").write_text(md)

    exit_code = 0
    baseline = load_baseline(args.baseline)
    if args.update_baseline:
        write_baseline(args.baseline, {m: {k: r[k] for k in GATED if k in r} for m, r in results.items()})
        print(f"Baseline updated: {args.baseline}")
    elif baseline:
        rules = [Rule(m, True, args.tolerance) for m in GATED]
        failures = []
        for mode, cur in results.items():
            failures += [f"[{mode}] {f}" for f in check_against_baseline(cur, baseline.get(mode, {}), rules)]
        if failures:
            md += "\n### ❌ Retrieval regressions\n" + "\n".join(f"- {f}" for f in failures) + "\n"
            print("Regressions:\n  " + "\n  ".join(failures))
            exit_code = 1 if args.gate else 0
        else:
            md += "\n✅ No retrieval regressions against the baseline.\n"
    else:
        md += "\n_No baseline committed yet; run with `--update-baseline` to create one._\n"

    write_step_summary(md)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
