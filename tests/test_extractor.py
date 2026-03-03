"""Tests for litmap.extractor"""
import pytest
from litmap.extractor import extract_concepts, batch_extract, _tokenize, _keep

ABSTRACT = (
    "We propose a new transformer architecture for natural language understanding. "
    "The self-attention mechanism allows the model to capture long-range dependencies. "
    "Our experiments demonstrate state-of-the-art performance on multiple benchmarks."
)

def test_tokenize_lowercases():
    assert "transformer" in _tokenize("TRANSFORMER Architecture")

def test_tokenize_strips_punctuation():
    tokens = _tokenize("attention-mechanism, transformer.")
    assert all("," not in t and "." not in t for t in tokens)

def test_keep_rejects_stopwords():
    assert _keep("the", 3) is False
    assert _keep("paper", 3) is False

def test_keep_rejects_short():
    assert _keep("ml", 3) is False

def test_keep_rejects_numeric():
    assert _keep("2023", 3) is False

def test_keep_accepts_valid():
    assert _keep("transformer", 3) is True

def test_extract_returns_list():
    assert isinstance(extract_concepts(ABSTRACT), list)
    assert len(extract_concepts(ABSTRACT)) > 0

def test_extract_top_score_is_one():
    result = extract_concepts(ABSTRACT)
    assert result[0][1] == 1.0

def test_extract_sorted_descending():
    result = extract_concepts(ABSTRACT)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)

def test_extract_top_n():
    assert len(extract_concepts(ABSTRACT, top_n=5)) <= 5

def test_extract_no_stopwords():
    result = extract_concepts(ABSTRACT)
    for term, _ in result:
        if " " not in term:
            assert term not in {"the", "and", "of", "paper"}

def test_bigrams_included():
    text = "attention mechanism attention mechanism attention mechanism"
    terms = [t for t, _ in extract_concepts(text, include_bigrams=True)]
    assert "attention mechanism" in terms

def test_no_bigrams():
    result = extract_concepts(ABSTRACT, include_bigrams=False)
    assert all(" " not in t for t, _ in result)

def test_empty_text():
    assert extract_concepts("") == []

def test_batch_extract():
    papers = [
        {"id": "A001", "abstract": ABSTRACT, "title": "Transformer"},
        {"id": "A002", "abstract": "Graph neural networks for molecular prediction.", "title": "GNN"},
    ]
    result = batch_extract(papers)
    assert "A001" in result and "A002" in result
    assert len(result["A001"]) > 0

def test_batch_extract_empty():
    assert batch_extract([]) == {}
