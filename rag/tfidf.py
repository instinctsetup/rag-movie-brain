"""TF-IDF + cosine similarity: the other classic, kept for comparison.

Each document is a vector with one dimension per vocabulary term;
the query is a vector in the same space; ranking is cosine similarity -
"how parallel are the two vectors". Derivation in docs/MATH.md.
"""
import math
from collections import Counter


class Tfidf:
    def __init__(self, docs_tokens: list[list[str]]):
        self.n_docs = len(docs_tokens)
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            for term in set(d):
                df[term] += 1
        self.idf = {t: math.log(self.n_docs / n) for t, n in df.items()}
        # Precompute each document's vector length for cosine.
        self.doc_norm = [self._norm(tf) for tf in self.tf]

    def _weights(self, counts: Counter) -> dict[str, float]:
        return {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in counts.items() if c > 0}

    def _norm(self, counts: Counter) -> float:
        return math.sqrt(sum(w * w for w in self._weights(counts).values())) or 1.0

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        qw = self._weights(Counter(query_tokens))
        qnorm = math.sqrt(sum(w * w for w in qw.values())) or 1.0
        out = []
        for i in range(self.n_docs):
            dot = 0.0
            for t, w in qw.items():
                c = self.tf[i].get(t, 0)
                if c:
                    dot += w * (1 + math.log(c)) * self.idf.get(t, 0.0)
            cos = dot / (qnorm * self.doc_norm[i])
            if cos > 0:
                out.append((i, cos))
        return sorted(out, key=lambda s: -s[1])[:top_k]
