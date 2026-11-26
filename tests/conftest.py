from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragqa.config import Settings
from ragqa.data.beir import load_beir
from ragqa.index import DocumentIndex, chunk_documents
from ragqa.ingest import documents_from_beir

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        embedding_provider="fake", embedding_model="fake", openai_api_key="test",
        data_dir=str(FIXTURES), index_dir=str(tmp_path / "index"),
        top_k=3, candidate_k=10, chunk_size=500, chunk_overlap=50,
    )


@pytest.fixture
def dataset():
    return load_beir(FIXTURES / "beir_mini")


@pytest.fixture
def index(dataset, settings) -> DocumentIndex:
    docs = documents_from_beir(dataset.corpus, source="beir_mini")
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    return DocumentIndex.build(chunks, DeterministicFakeEmbedding(size=64), "fake")
