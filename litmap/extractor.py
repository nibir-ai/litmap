"""
Concept extraction from research paper text.
TF-based, zero external NLP dependencies.
"""
import re
from collections import Counter
from typing import Optional

_STOPWORDS: frozenset[str] = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","that","which","who","this","these","those","it","its","they",
    "them","we","our","i","my","you","your","he","she","not","no","so",
    "yet","each","more","most","other","some","than","too","very","just",
    "also","only","then","now","here","there","all","any","about","into",
    "paper","work","study","approach","method","methods","propose","proposed",
    "present","show","shows","demonstrate","use","uses","using","based","new",
    "novel","achieve","achieves","improve","improves","state","art","recent",
    "problem","task","tasks","existing","previous","prior","model","models",
    "result","results","performance","experiments","dataset","datasets",
    "training","evaluate","evaluation","benchmark","benchmarks","outperforms",
    "large","small","high","low","across","via","et","al","i.e","e.g","etc",
})

_PUNCT_RE = re.compile(r"[^\w\s\-]")
_WHITESPACE_RE = re.compile(r"\s+")


def extract_concepts(text: str, title: Optional[str] = None,
                     top_n: int = 20, min_term_length: int = 3,
                     include_bigrams: bool = True) -> list[tuple[str, float]]:
    tokens = _tokenize(text)
    filtered = [t for t in tokens if _keep(t, min_term_length)]
    counts: Counter = Counter(filtered)
    if include_bigrams:
        counts.update(_build_bigrams(filtered))
    if not counts:
        return []
    title_tokens: set[str] = set()
    if title:
        tt = set(_tokenize(title.lower()))
        title_tokens = tt | set(_build_bigrams(list(tt)))
    total = sum(counts.values())
    raw = []
    for term, count in counts.most_common(top_n * 3):
        tf = count / total
        boost = 1.5 if term in title_tokens else 1.0
        raw.append((term, tf * boost))
    if not raw:
        return []
    max_score = max(s for _, s in raw)
    normalized = [(t, round(s / max_score, 4)) for t, s in raw]
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized[:top_n]


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.split()


def _keep(token: str, min_len: int) -> bool:
    if len(token) < min_len:
        return False
    if token in _STOPWORDS:
        return False
    if not any(c.isalpha() for c in token):
        return False
    return True


def _build_bigrams(tokens: list[str]) -> list[str]:
    return [
        f"{tokens[i]} {tokens[i+1]}"
        for i in range(len(tokens) - 1)
        if _keep(tokens[i], 3) and _keep(tokens[i+1], 3)
    ]


def batch_extract(papers: list[dict], top_n: int = 15) -> dict[str, list[tuple[str, float]]]:
    return {
        p["id"]: extract_concepts(p.get("abstract", ""), title=p.get("title"), top_n=top_n)
        for p in papers
    }
