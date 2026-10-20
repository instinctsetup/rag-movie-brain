from fastapi.testclient import TestClient
from server import app

c = TestClient(app)


def test_health():
    assert c.get("/api/health").json()["plots"] > 6000


def test_ask_returns_grounded_answer():
    r = c.post("/api/ask", json={"question": "boy wizard goes to magic school", "method": "bm25"})
    data = r.json()
    assert r.status_code == 200
    assert data["citations"], "expected cited evidence"
    assert all(x["evidence"] for x in data["citations"]), "evidence should never be empty"


def test_search_endpoint_returns_results():
    r = c.post("/api/search", json={"question": "hacker computer simulation", "method": "bm25"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results and all("title" in x and "plot" in x for x in results)


def test_ask_with_only_stopwords_returns_no_match_message():
    r = c.post("/api/ask", json={"question": "the of and", "method": "bm25"})
    data = r.json()
    assert data["citations"] == []
    assert "No plot" in data["answer"]
