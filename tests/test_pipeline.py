from langchain_core.language_models import FakeListChatModel

from ragqa.pipeline import NO_ANSWER, RAGPipeline, extract_citations, format_context
from ragqa.retrieval import build_retriever


def test_extract_citations_filters_to_retrieved_ids():
    answer = "Aspirin helps [101]. Also see [103, 999] and [101]; unrelated [note]."
    assert extract_citations(answer, {"101", "103"}) == ["101", "103"]


def test_format_context_includes_ids(index):
    text = format_context(index.chunks[:2])
    assert text.count("---") == 1
    assert text.startswith("[")


def test_pipeline_answer_with_citations(index, settings):
    llm = FakeListChatModel(responses=["Elevated IL-6 predicted mortality in septic shock [103]."])
    pipeline = RAGPipeline(build_retriever(index, settings, mode="bm25", k=3), llm)
    resp = pipeline.answer("Does IL-6 predict mortality in septic shock?")

    assert resp.citations == ["103"]
    assert not resp.abstained
    assert resp.contexts[0]["doc_id"] == "103"
    assert len(resp.context_texts) == 3


def test_pipeline_abstention(index, settings):
    llm = FakeListChatModel(responses=[NO_ANSWER])
    resp = RAGPipeline(build_retriever(index, settings, mode="hybrid", k=2), llm).answer("Who won the 1998 World Cup?")
    assert resp.abstained
    assert resp.citations == []
