from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from .config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    """Embedding model factory. Imports are lazy so tests need no heavy deps."""
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key or None)
    return DeterministicFakeEmbedding(size=64)
