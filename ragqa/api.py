"""FastAPI service: /ask (RAG answer with citations), /search (retrieval only)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .config import RetrievalMode, Settings, get_settings
from .embeddings import get_embeddings
from .index import DocumentIndex
from .pipeline import RAGPipeline, build_llm
from .retrieval import HybridRetriever, build_retriever

logger = logging.getLogger("ragqa.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class Engine:
    """Holds the loaded index and one retriever / pipeline per retrieval mode."""

    def __init__(self, index: DocumentIndex, settings: Settings, llm=None) -> None:
        self.index = index
        self.settings = settings
        self.llm = llm
        self._retrievers: dict[tuple[str, int], HybridRetriever] = {}

    def retriever(self, mode: RetrievalMode, k: int) -> HybridRetriever:
        key = (mode, k)
        if key not in self._retrievers:
            self._retrievers[key] = build_retriever(self.index, self.settings, mode=mode, k=k)
        return self._retrievers[key]

    def pipeline(self, mode: RetrievalMode, k: int) -> RAGPipeline:
        if self.llm is None:
            self.llm = build_llm(self.settings.llm_model, self.settings.openai_api_key, self.settings.llm_temperature)
        return RAGPipeline(self.retriever(mode, k), self.llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "engine", None) is None:  # tests inject their own engine
        settings = get_settings()
        index = DocumentIndex.load(settings.index_dir, get_embeddings(settings), settings.embedding_model)
        app.state.engine = Engine(index, settings)
        logger.info("Loaded index: %s", index.manifest)
    yield


app = FastAPI(title="RAG Document QA", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: RetrievalMode | None = None
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float | None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    abstained: bool
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: RetrievalMode | None = None
    top_k: int | None = Field(None, ge=1, le=50)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats(request: Request) -> dict[str, Any]:
    engine: Engine = request.app.state.engine
    return {**engine.index.manifest, "retrieval_mode": engine.settings.retrieval_mode, "top_k": engine.settings.top_k}


@app.post("/search")
async def search(body: SearchRequest, request: Request) -> dict[str, Any]:
    engine: Engine = request.app.state.engine
    mode = body.mode or engine.settings.retrieval_mode
    retriever = engine.retriever(mode, body.top_k or engine.settings.top_k)
    docs = await run_in_threadpool(retriever.invoke, body.query)
    return {
        "query": body.query,
        "mode": mode,
        "results": [
            {"doc_id": d.metadata["doc_id"], "title": d.metadata.get("title", ""), "text": d.page_content,
             "score": d.metadata.get("rrf_score", d.metadata.get("vector_distance"))}
            for d in docs
        ],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    engine: Engine = request.app.state.engine
    pipeline = engine.pipeline(body.mode or engine.settings.retrieval_mode, body.top_k or engine.settings.top_k)
    try:
        resp = await run_in_threadpool(pipeline.answer, body.question)
    except Exception as exc:
        logger.exception("generation failed")
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc
    return AskResponse(
        question=resp.question,
        answer=resp.answer,
        citations=resp.citations,
        abstained=resp.abstained,
        sources=[Source(**c) for c in resp.contexts],
        retrieval_ms=resp.retrieval_ms,
        generation_ms=resp.generation_ms,
    )
