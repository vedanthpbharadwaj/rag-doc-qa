from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["hybrid", "vector", "bm25"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embeddings: "huggingface" (local sentence-transformers), "openai", or "fake" (tests)
    embedding_provider: Literal["huggingface", "openai", "fake"] = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Generation
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0

    # Storage
    data_dir: str = "data"
    index_dir: str = "data/index"

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Retrieval
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 5  # documents passed to the LLM
    candidate_k: int = 30  # candidates fetched from each retriever before fusion
    rrf_k: int = 60  # reciprocal rank fusion constant
    vector_weight: float = 1.0
    bm25_weight: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
