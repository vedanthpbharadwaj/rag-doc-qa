"""Retrieval-augmented answering with source citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .retrieval import HybridRetriever

NO_ANSWER = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered source documents provided.

Rules:
- Every factual sentence must end with the id of the supporting source in square brackets, e.g. [4983].
- Do not use outside knowledge. Do not speculate beyond what the sources state.
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- If sources disagree, say so and cite both.
- Be concise: 2-5 sentences."""

USER_PROMPT = """Sources:
{context}

Question: {question}"""

_CITATION = re.compile(r"\[([^\[\]]+)\]")


def format_context(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        title = d.metadata.get("title") or ""
        header = f"[{d.metadata['doc_id']}] {title}".strip()
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def extract_citations(answer: str, allowed_ids: set[str]) -> list[str]:
    """Doc ids cited in the answer, in order of first appearance, limited to retrieved sources."""
    found: list[str] = []
    for group in _CITATION.findall(answer):
        for part in re.split(r"[,;]\s*", group):
            part = part.strip()
            if part in allowed_ids and part not in found:
                found.append(part)
    return found


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: list[str]
    contexts: list[dict[str, Any]]
    abstained: bool
    retrieval_ms: float
    generation_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c["text"] for c in self.contexts]


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel) -> None:
        self.retriever = retriever
        self.chain = (
            ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
            | llm
            | StrOutputParser()
        )

    def answer(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        docs = self.retriever.invoke(question)
        t1 = time.perf_counter()
        if docs:
            answer = self.chain.invoke({"context": format_context(docs), "question": question}).strip()
        else:
            answer = NO_ANSWER
        t2 = time.perf_counter()

        ids = {d.metadata["doc_id"] for d in docs}
        return RAGResponse(
            question=question,
            answer=answer,
            citations=extract_citations(answer, ids),
            contexts=[
                {
                    "doc_id": d.metadata["doc_id"],
                    "title": d.metadata.get("title", ""),
                    "text": d.page_content,
                    "score": d.metadata.get("rrf_score", d.metadata.get("vector_distance")),
                }
                for d in docs
            ],
            abstained=answer.strip().startswith(NO_ANSWER[:40]),
            retrieval_ms=round((t1 - t0) * 1000, 1),
            generation_ms=round((t2 - t1) * 1000, 1),
        )


def build_llm(model: str, api_key: str = "", temperature: float = 0.0) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key or None, max_retries=3)
