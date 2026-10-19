# The math inside rag-movie-brain, in plain words

## TF-IDF + cosine

**Term frequency**: how often term t appears in document d. We log-scale it: `1 + log(f)` - the 10th occurrence of a word is not 10x as meaningful as the 1st.

**Inverse document frequency**: `idf(t) = log(N / df(t))` - a term in 1 of 7000 documents is precious (log 7000 ≈ 8.9); a term in every document is noise (log 1 = 0).

**Document vector**: one number per vocabulary term, `tf * idf`. The query becomes a vector in the same space.

**Cosine similarity**: `cos(a, b) = (a . b) / (|a| |b|)` - the cosine of the angle between the vectors. Identical vocabulary mix -> 1; nothing shared -> 0. Dividing by lengths makes a 200-word plot comparable to a 2000-word one.

## BM25 (what real search engines run)

    score(q, d) = sum over query terms t:
        idf(t) * f(t,d) * (k1 + 1) / ( f(t,d) + k1 * (1 - b + b * dl/avgdl) )

- **idf(t)** = `ln(1 + (N - df + 0.5) / (df + 0.5))` - Robertson's form; smoother than plain log(N/df), never negative.
- **Saturation**: the fraction `f * (k1+1) / (f + k1 * ...)` rises quickly for the first few occurrences of t, then flattens towards k1+1. k1 = 1.5 controls where it flattens. This models diminishing returns of repeated terms.
- **Length normalization**: `dl / avgdl` is the document's length relative to average; with b = 0.75, long documents are penalized (a long plot mentions everything, so mentions mean less).

## Why extractive answers are "grounded"

The answer string is assembled only from sentences that (a) come from the top-ranked plots and (b) share the most content terms with the query. Every claim is followed by [n], and citation n names the movie. There is no generative step, so the system literally cannot state anything that is not in the corpus - the standard trick for trustworthy RAG.

## Evaluation

`eval.py` asks 12 questions with a known target movie ("boy wizard goes to magic school" -> Harry Potter) and reports hit-rate@5: how often the right movie is in the top 5. Results in `docs/eval-results.md`.
