"""BM25 from scratch - the ranking function inside Elasticsearch and most
real search engines. Every constant is explained in docs/MATH.md.

Score(query, doc) = sum over shared terms:
    idf(t) * f(t,doc) * (k1+1) / (f(t,doc) + k1 * (1 - b + b * dl/avgdl))

- idf(t)        : rare terms count more
- f(t,doc)      : term frequency, but with *saturation* (k1)
- dl/avgdl      : long documents are normalized (b)
"""
import math
from collections import Counter


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            for term in set(d):
                df[term] += 1
        # Robertson's idf, floored at 0 so ultra-common terms can't go negative.
        self.idf = {t: max(0.0, math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)))
                    for t, n in df.items()}

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        dl = self.doc_len[doc_id] or 1
        avgdl = self.avgdl or 1.0  # all-empty corpus would make avgdl 0
        total = 0.0
        for t in query_tokens:
            if t not in self.idf:
                continue
            f = self.tf[doc_id].get(t, 0)
            total += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
        return total

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        scored = [s for s in scored if s[1] > 0]
        return sorted(scored, key=lambda s: -s[1])[:top_k]
