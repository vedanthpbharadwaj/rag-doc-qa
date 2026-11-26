"""Chunking and the on-disk document index (FAISS vectors + chunk store for BM25)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents and give every chunk a stable id: <doc_id>::<n>."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            meta = {**doc.metadata, "chunk_id": f"{doc.metadata['doc_id']}::{i}", "chunk_index": i}
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


class DocumentIndex:
    """FAISS index plus the raw chunks (BM25 is rebuilt from them on load)."""

    MANIFEST = "manifest.json"
    CHUNKS = "chunks.jsonl"

    def __init__(self, vector_store: FAISS, chunks: list[Document], manifest: dict | None = None) -> None:
        self.vector_store = vector_store
        self.chunks = chunks
        self.manifest = manifest or {}

    @property
    def num_documents(self) -> int:
        return len({c.metadata["doc_id"] for c in self.chunks})

    @classmethod
    def build(
        cls,
        chunks: list[Document],
        embeddings: Embeddings,
        embedding_model: str,
        batch_size: int = 512,
    ) -> DocumentIndex:
        if not chunks:
            raise ValueError("No chunks to index")
        ids = [c.metadata["chunk_id"] for c in chunks]
        store = FAISS.from_documents(chunks[:batch_size], embeddings, ids=ids[:batch_size])
        for start in range(batch_size, len(chunks), batch_size):
            store.add_documents(chunks[start:start + batch_size], ids=ids[start:start + batch_size])
            print(f"  embedded {min(start + batch_size, len(chunks)):,}/{len(chunks):,} chunks")
        manifest = {
            "embedding_model": embedding_model,
            "num_chunks": len(chunks),
            "num_documents": len({c.metadata["doc_id"] for c in chunks}),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return cls(store, chunks, manifest)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))
        with open(path / self.CHUNKS, "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps({"text": c.page_content, "metadata": c.metadata}) + "\n")
        (path / self.MANIFEST).write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path, embeddings: Embeddings, embedding_model: str | None = None) -> DocumentIndex:
        path = Path(path)
        if not (path / cls.MANIFEST).exists():
            raise FileNotFoundError(f"No index at {path}. Run `python -m ragqa.ingest` first.")
        manifest = json.loads((path / cls.MANIFEST).read_text())
        if embedding_model and manifest.get("embedding_model") != embedding_model:
            raise ValueError(
                f"Index was built with {manifest.get('embedding_model')!r} but settings use "
                f"{embedding_model!r}. Rebuild the index or change EMBEDDING_MODEL."
            )
        # The index is produced by this project's own ingest step, so pickle loading is trusted.
        store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        chunks = []
        with open(path / cls.CHUNKS, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                chunks.append(Document(page_content=row["text"], metadata=row["metadata"]))
        return cls(store, chunks, manifest)
