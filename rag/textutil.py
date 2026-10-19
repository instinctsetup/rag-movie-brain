"""Tokenization: the unglamorous first 10% of every search system."""
import re

STOPWORDS = set("""a an and are as at be but by for from has have he her his i in is it its
of on or she that the their they this to was were what which who will with you your about
after all also been before between can could did do does had him how if into just like me
more my no not now only other our out over s so some such t than them then there these
through too under up very we when where while why would""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumerics, drop stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Naive sentence splitter - good enough for extracting answer snippets."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
