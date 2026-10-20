"""Load the plot corpus and expose search + grounded answering."""
import json
from pathlib import Path
from . import bm25, textutil, tfidf


class MovieIndex:
    def __init__(self, path: str | None = None):
        path = path or str(Path(__file__).parent.parent / "data" / "plots.jsonl")
        with open(path) as f:
            self.docs = []
            for i, line in enumerate(f, 1):
                try:
                    self.docs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"bad JSON at {path}:{i}") from e
        tokens = [textutil.tokenize(d["plot"] + " " + (d["title"] + " ") * 3) for d in self.docs]  # title boost
        self.bm25 = bm25.BM25(tokens)
        self.tfidf = tfidf.Tfidf(tokens)

    def search(self, query: str, method: str = "bm25", top_k: int = 5):
        q = textutil.tokenize(query)
        engine = self.bm25 if method == "bm25" else self.tfidf
        return [{"doc": self.docs[i], "score": round(s, 3)}
                for i, s in engine.search(q, top_k=top_k)]

    def answer(self, query: str, method: str = "bm25", top_k: int = 3):
        """Retrieve, then extract the most relevant sentences from the top
        docs - the 'grounded answer' is quoted evidence with citations,
        never invented text. Honest RAG without an external LLM."""
        hits = self.search(query, method=method, top_k=top_k)
        q_terms = set(textutil.tokenize(query))
        citations = []
        for rank, hit in enumerate(hits, 1):
            sents = textutil.sentences(hit["doc"]["plot"])
            if not sents:
                # every sentence filtered out (e.g. no sentence >=20 chars):
                # fall back to the raw plot so evidence is never empty
                best = hit["doc"]["plot"][:400]
            else:
                best, best_overlap = "", -1
                for sent in sents:
                    overlap = len(q_terms & set(textutil.tokenize(sent)))
                    if overlap > best_overlap:
                        best, best_overlap = sent, overlap
            citations.append({"n": rank, "title": hit["doc"]["title"],
                              "year": hit["doc"]["year"], "score": hit["score"],
                              "evidence": best[:400]})
        if not citations:
            return {"answer": "No plot in the corpus matches that question.", "citations": []}
        parts = [f"[{c['n']}] {c['title']} ({c['year']}): {c['evidence']}" for c in citations]
        return {"answer": "Most relevant plots from the corpus:\n\n" + "\n\n".join(parts),
                "citations": citations}
