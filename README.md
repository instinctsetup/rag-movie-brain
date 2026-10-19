# rag-movie-brain

**Retrieval-augmented Q&A over 7,000 real Wikipedia movie plots - BM25 and TF-IDF written from scratch, answers quoted from evidence with citations.**

- **Demo video**: https://drive.google.com/file/d/10b-FqJWCy7LV-aWWEtAnQWl5XZ4KbNk5/view
- **Project report (PDF)**: https://drive.google.com/file/d/1jWLlmGNoTgHiVeBgfDn9NYOPad2B85Ht/view


Ask "a hacker discovers reality is a computer simulation" and it retrieves the most relevant plots, extracts the most relevant sentence from each, and answers *only* with cited evidence. The retrieval math is the point of the project: both classic ranking functions are implemented line-by-line, with derivations in `docs/MATH.md`.

## Why this design (read: honest label)

Modern RAG = retrieve relevant documents, then have an LLM write an answer grounded in them. This project implements the retrieval and grounding pipeline itself - tokenization, inverted scoring, sentence-level evidence extraction, citation - and deliberately uses **no external LLM API**: answers are extractive (quoted sentences) so everything is reproducible offline and nothing can be hallucinated. Swapping in an LLM for the final prose step is a 10-line change once you have an API key; the retrieval underneath would be exactly this.

## Quick start

```bash
pip install -r requirements.txt
uvicorn server:app --port 8000    # open http://localhost:8000
```

The index builds from `data/plots.jsonl` in ~2 seconds at startup.

```bash
python -m pytest tests -q         # 8 tests: tokenizer, BM25, TF-IDF, corpus sanity, API
python eval.py                    # hit-rate@5 on 12 hand-labeled questions
```

## The two rankers, side by side

| | TF-IDF + cosine | BM25 |
|---|---|---|
| idea | documents are vectors; rank by angle to the query vector | probabilistic term weighting with saturation + length normalization |
| term frequency | log-scaled | saturates (k1 = 1.5) |
| document length | ignored (cosine normalizes the vector) | explicitly normalized (b = 0.75) |
| in the UI | pick "TF-IDF" | pick "BM25" (default) |

The frontend toggle lets you ask the same question both ways and compare.

## Data

`data/plots.jsonl`: 7,000 plots sampled from the CMU MovieSummaries corpus (Wikipedia plot summaries + metadata). Sampled only to keep the repo light; the build script would scale to all 42k.

## Layout

```
rag/textutil.py   tokenizer + sentence splitter
rag/bm25.py       BM25 from scratch
rag/tfidf.py      TF-IDF + cosine from scratch
rag/index.py      corpus loading, search, extractive grounded answering
server.py         FastAPI: /api/ask, /api/search
frontend/         ask UI with highlighted evidence (vanilla JS)
eval.py           12 labeled questions -> hit-rate@5 for both methods
docs/             MATH.md, VIVA.md, eval-results.md
```
