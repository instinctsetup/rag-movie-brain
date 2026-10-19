"""rag-movie-brain web server.

Run:  uvicorn server:app --port 8000    then open http://localhost:8000
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.index import MovieIndex

app = FastAPI(title="rag-movie-brain")
FRONTEND = Path(__file__).parent / "frontend"
INDEX = MovieIndex()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    method: str = Field(default="bm25", pattern="^(bm25|tfidf)$")


@app.get("/api/health")
def health():
    return {"status": "ok", "plots": len(INDEX.docs)}


@app.post("/api/ask")
def ask(req: AskRequest):
    result = INDEX.answer(req.question, method=req.method)
    return {"question": req.question, "method": req.method, **result}


@app.post("/api/search")
def search(req: AskRequest):
    return {"results": [{"title": h["doc"]["title"], "year": h["doc"]["year"],
                         "score": h["score"], "plot": h["doc"]["plot"][:280] + "…"}
                        for h in INDEX.search(req.question, method=req.method)]}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
