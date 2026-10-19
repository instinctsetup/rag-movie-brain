"""Hit-rate@5 on 12 hand-labeled questions, for both rankers."""
from rag.index import MovieIndex

QA = [
    ("boy wizard goes to magic school", "Harry Potter and the Philosopher's Stone"),
    ("a hacker discovers reality is a computer simulation", "Mindwarp|The Matrix|eXistenZ"),
    ("space ship crew fights alien monster", "Alien"),
    ("shark attacks beach town", "Jaws"),
    ("archaeologist hunts for lost ark", "Raiders of the Lost Ark"),
    ("mafia family crime saga", "The Godfather"),
    ("farm boy joins rebellion against space empire", "Star Wars"),
    ("clown terrorizes small town children", "It"),
    ("robot from the future protects a boy", "Terminator 2: Judgment Day"),
    ("hobbit must destroy a magic ring", "The Lord of the Rings: The Fellowship of the Ring"),
    ("ship sinks on maiden voyage love story", "Titanic"),
    ("toys come alive when humans leave", "Toy Story"),
]

def main():
    idx = MovieIndex()
    corpus_titles = {d["title"].lower() for d in idx.docs}
    answerable = [(q, t) for q, t in QA
                  if any(x.lower() in corpus_titles for x in t.split("|"))]
    lines = ["# eval-results\n",
             f"Hit-rate@5 over {len(QA)} hand-labeled questions; "
             f"{len(answerable)} targets exist in the 7,000-plot sample "
             "(scored on answerable questions only, exact title match).\n",
             "| question | BM25 hit | TF-IDF hit |", "|---|---|---|"]
    marks = {}
    for method in ("bm25", "tfidf"):
        hits = 0
        for q, target in answerable:
            titles = {h["doc"]["title"].lower() for h in idx.search(q, method=method, top_k=5)}
            ok = any(t.lower() in titles for t in target.split("|"))
            hits += ok
            marks.setdefault(q, {})[method] = ok
        print(f"{method}: hit-rate@5 = {hits}/{len(answerable)} (answerable)")
        lines[2] = lines[2]
    for q, _ in answerable:
        lines.append(f"| {q} | {'Y' if marks[q]['bm25'] else 'n'} | {'Y' if marks[q]['tfidf'] else 'n'} |")
    open("docs/eval-results.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
