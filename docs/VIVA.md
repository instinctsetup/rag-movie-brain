# Viva questions you should expect, with answers

**Q: What is RAG?**
Retrieval-augmented generation: instead of asking a model to answer from memory, you first *retrieve* relevant documents, then compose the answer *from them*. Benefits: citations, freshness, and far fewer hallucinations.

**Q: Why no LLM in this project?**
Deliberately. The retrieval and grounding pipeline is the part I built and am being examined on; the final prose step is a 10-line API call once you have a key. Answers here are extractive - quoted sentences with citations - which also means the system cannot invent facts.

**Q: BM25 vs TF-IDF - the two-sentence version?**
Both weight terms by tf and idf. BM25 adds (a) term-frequency *saturation* - the 10th repetition adds almost nothing - and (b) explicit document-length normalization, which is why it beats raw TF-IDF on most benchmarks.

**Q: What does the idf formula do to a word like "the"?**
df ≈ N, so (N - df + 0.5)/(df + 0.5) ≈ small, log ≈ 0 - common words get nearly zero weight. (Our tokenizer drops stopwords anyway; idf is the second line of defence.)

**Q: Your eval is 6/10 - why is that honest, and what is the fix?**
The misses are the *lexical gap*: the query says "clown terrorizes children", the plot says "an evil entity preys on Derry". Keyword retrieval cannot bridge vocabulary mismatch. The industry fix is dense retrieval (embedding vectors from a neural model), often hybrid BM25 + embeddings. That is the stated next step, and seeing the failure mode clearly is worth more than a tuned number.

**Q: How would you scale this to 40M documents?**
Inverted index (we build one implicitly per query; Lucene builds it once), sharding, and approximate-nearest-neighbor indices (HNSW) for the dense half.

**Q: Why include the title three times in each document?**
A title match is much stronger evidence than a body mention - field boosting. Standard in BM25F (the multi-field extension); here it is approximated by repetition, which BM25's saturation keeps sane.
