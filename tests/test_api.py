from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel

from ragqa.api import Engine, app


def test_api_endpoints(index, settings):
    llm = FakeListChatModel(responses=["Caffeine intake was linked to lower Parkinson risk [110]."])
    app.state.engine = Engine(index, settings, llm=llm)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/stats").json()["num_documents"] == 12

        r = client.post("/search", json={"query": "caffeine Parkinson", "mode": "bm25", "top_k": 2})
        assert r.status_code == 200
        assert r.json()["results"][0]["doc_id"] == "110"

        r = client.post("/ask", json={"question": "Does coffee lower Parkinson risk?", "mode": "bm25"})
        body = r.json()
        assert r.status_code == 200
        assert body["citations"] == ["110"]
        assert len(body["sources"]) == 3

        assert client.post("/ask", json={"question": "hi", "mode": "nope"}).status_code == 422
    app.state.engine = None
