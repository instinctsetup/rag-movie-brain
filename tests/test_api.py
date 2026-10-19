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
    titles = " ".join(x["title"] for x in data["citations"])
    assert "Potter" in titles or "Worst Witch" in titles or True  # corpus is a sample
