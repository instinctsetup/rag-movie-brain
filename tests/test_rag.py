"""Retrieval correctness on toy corpora + real-corpus sanity."""
from rag import bm25, textutil, tfidf


TOY = [["the", "cat", "sat", "on", "the", "mat"],
       ["a", "dog", "barks", "loudly"],
       ["cats", "and", "dogs", "are", "friends"]]
TOY = [[t for t in doc] for doc in TOY]


def test_tokenize_strips_stopwords_and_case():
    assert textutil.tokenize("The Cat, AND a Dog!") == ["cat", "dog"]


def test_bm25_ranks_relevant_first():
    engine = bm25.BM25(TOY)
    hits = engine.search(["cat", "mat"])
    assert hits[0][0] == 0


def test_bm25_idf_prefers_rare_terms():
    docs = [["cat", "sat"], ["cat", "ran"], ["barks"]]  # cat df=2, barks df=1
    engine = bm25.BM25(docs)
    assert engine.idf["barks"] > engine.idf["cat"]


def test_tfidf_cosine_bounds_and_ranking():
    engine = tfidf.Tfidf(TOY)
    hits = engine.search(["dog"])
    assert hits[0][0] == 1
    assert all(0 < s <= 1.0001 for _, s in hits)


def test_real_corpus_finds_matrix_like_plot():
    from rag.index import MovieIndex
    idx = MovieIndex()
    hits = idx.search("computer simulation reality hacker", top_k=5)
    titles = [h["doc"]["title"] for h in hits]
    assert any("Mindwarp" in t or "Matrix" in t or "eXistenZ" in t or "13th" in t for t in titles), titles


def test_answer_has_citations():
    from rag.index import MovieIndex
    idx = MovieIndex()
    out = idx.answer("space ship crew fights alien monster")
    assert out["citations"] and all(c["evidence"] for c in out["citations"])
