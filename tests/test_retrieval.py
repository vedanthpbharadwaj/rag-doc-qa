from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragqa.index import DocumentIndex, chunk_documents
from ragqa.retrieval import build_retriever, dedupe_by_document, reciprocal_rank_fusion, tokenize


def _doc(cid, did=None):
    return Document(page_content=cid, metadata={"chunk_id": cid, "doc_id": did or cid.split("::")[0]})


def test_tokenize_keeps_scientific_terms():
    assert tokenize("Does IL-6 predict mortality in BRCA1 carriers?") == ["il-6", "predict", "mortality", "brca1", "carriers"]


def test_rrf_rewards_agreement():
    vector = [_doc("a"), _doc("b"), _doc("c")]
    bm25 = [_doc("c"), _doc("a"), _doc("d")]
    fused = reciprocal_rank_fusion([vector, bm25], [1.0, 1.0], rrf_k=60)
    order = [d.metadata["chunk_id"] for d, _ in fused]
    assert order[:2] == ["a", "c"]  # both lists rank them highly
    assert set(order) == {"a", "b", "c", "d"}
    assert fused[0][1] == 1 / 61 + 1 / 62


def test_rrf_weights():
    fused = reciprocal_rank_fusion([[_doc("a")], [_doc("b")]], [1.0, 2.0])
    assert fused[0][0].metadata["chunk_id"] == "b"


def test_dedupe_by_document():
    chunks = [_doc("1::0"), _doc("1::1"), _doc("2::0"), _doc("3::0")]
    out = dedupe_by_document(chunks, 2)
    assert [c.metadata["doc_id"] for c in out] == ["1", "2"]


def test_chunk_ids_are_stable():
    long = Document(page_content="word " * 400, metadata={"doc_id": "d1"})
    chunks = chunk_documents([long], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert [c.metadata["chunk_id"] for c in chunks[:2]] == ["d1::0", "d1::1"]


def test_bm25_mode_finds_keyword_match(index, settings):
    retriever = build_retriever(index, settings, mode="bm25", k=3)
    docs = retriever.invoke("interleukin-6 septic shock mortality")
    assert docs[0].metadata["doc_id"] == "103"


def test_hybrid_mode_returns_unique_docs_with_scores(index, settings):
    retriever = build_retriever(index, settings, mode="hybrid", k=3)
    docs = retriever.invoke("caffeine Parkinson disease risk")
    ids = [d.metadata["doc_id"] for d in docs]
    assert len(ids) == 3 == len(set(ids))
    assert "110" in ids  # BM25 signal survives fusion even with random fake embeddings
    assert all("rrf_score" in d.metadata for d in docs)


def test_vector_mode(index, settings):
    docs = build_retriever(index, settings, mode="vector", k=2).invoke("statins")
    assert len(docs) == 2 and "vector_distance" in docs[0].metadata


def test_index_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = DocumentIndex.load(tmp_path / "idx", DeterministicFakeEmbedding(size=64), "fake")
    assert loaded.num_documents == 12
    assert len(loaded.chunks) == len(index.chunks)


def test_index_rejects_mismatched_embedding_model(index, tmp_path):
    import pytest

    index.save(tmp_path / "idx")
    with pytest.raises(ValueError, match="Rebuild the index"):
        DocumentIndex.load(tmp_path / "idx", DeterministicFakeEmbedding(size=64), "other-model")
