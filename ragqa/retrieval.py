"""Hybrid retrieval: FAISS dense search + BM25 keyword search, fused with RRF.

Dense embeddings capture meaning ("heart attack" ~ "myocardial infarction");
BM25 catches exact terms embeddings blur (gene names, drug names, numbers).
Reciprocal Rank Fusion merges the two ranked lists using ranks only, so the
very different score scales of cosine similarity and BM25 never need to be
calibrated against each other:

    score(d) = sum_i  w_i / (rrf_k + rank_i(d))
"""

from __future__ import annotations

import re
from collections.abc import Callable

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .config import RetrievalMode, Settings
from .index import DocumentIndex

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in into is it its of on or that the their "
    "there these this to was were which with what when where who why how does do did can".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stopwords; keeps hyphenated terms like il-6."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    rrf_k: int = 60,
    key: Callable[[Document], str] = lambda d: d.metadata["chunk_id"],
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc in enumerate(ranked, start=1):
            k = key(doc)
            scores[k] = scores.get(k, 0.0) + weight / (rrf_k + rank)
            docs.setdefault(k, doc)
    order = sorted(scores, key=scores.get, reverse=True)
    return [(docs[k], scores[k]) for k in order]


def dedupe_by_document(chunks: list[Document], limit: int) -> list[Document]:
    """Keep the best-ranked chunk per source document, up to `limit` documents."""
    seen: set[str] = set()
    out: list[Document] = []
    for c in chunks:
        did = c.metadata["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        out.append(c)
        if len(out) == limit:
            break
    return out


class HybridRetriever(BaseRetriever):
    """LangChain retriever supporting `hybrid`, `vector` and `bm25` modes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: FAISS
    bm25: BM25Retriever
    mode: RetrievalMode = "hybrid"
    k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0

    def _vector_search(self, query: str) -> list[Document]:
        hits = self.vector_store.similarity_search_with_score(query, k=self.candidate_k)
        out = []
        for doc, distance in hits:
            doc = Document(page_content=doc.page_content, metadata={**doc.metadata, "vector_distance": float(distance)})
            out.append(doc)
        return out

    def _bm25_search(self, query: str) -> list[Document]:
        # bm25.k is fixed to candidate_k when the retriever is built (see build_retriever).
        return self.bm25.invoke(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        if self.mode == "vector":
            ranked = self._vector_search(query)
        elif self.mode == "bm25":
            ranked = self._bm25_search(query)
        else:
            fused = reciprocal_rank_fusion(
                [self._vector_search(query), self._bm25_search(query)],
                [self.vector_weight, self.bm25_weight],
                self.rrf_k,
            )
            ranked = []
            for doc, score in fused:
                ranked.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "rrf_score": round(score, 6)}))
        return dedupe_by_document(ranked, self.k)


def build_retriever(index: DocumentIndex, settings: Settings, mode: RetrievalMode | None = None,
                    k: int | None = None) -> HybridRetriever:
    bm25 = BM25Retriever.from_documents(index.chunks, preprocess_func=tokenize, k=settings.candidate_k)
    return HybridRetriever(
        vector_store=index.vector_store,
        bm25=bm25,
        mode=mode or settings.retrieval_mode,
        k=k or settings.top_k,
        candidate_k=settings.candidate_k,
        rrf_k=settings.rrf_k,
        vector_weight=settings.vector_weight,
        bm25_weight=settings.bm25_weight,
    )
