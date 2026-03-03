"""Tests for litmap.graph"""
import pytest
from litmap.graph import LitmapGraph


@pytest.fixture
def g(tmp_path):
    db = tmp_path / "test.db"
    with LitmapGraph(db_path=db) as graph:
        yield graph


@pytest.fixture
def populated(g):
    g.upsert_paper("A001", "Transformer Paper", "Deep learning transformer architecture.",
                   published="2023-01-01", updated="2023-01-02", url="https://arxiv.org/abs/A001")
    g.add_authors("A001", ["Alice Smith", "Bob Jones"])
    g.add_categories("A001", ["cs.LG", "cs.CL"])
    g.add_concepts("A001", [("transformer", 1.0), ("attention", 0.85)])
    g.upsert_paper("A002", "GNN Chemistry", "Graph neural networks for molecules.",
                   published="2023-01-02", updated="2023-01-02", url="https://arxiv.org/abs/A002")
    g.add_authors("A002", ["Alice Smith", "Carol Lee"])
    g.add_categories("A002", ["cs.LG"])
    g.add_concepts("A002", [("graph neural", 1.0), ("attention", 0.3)])
    return g


def test_upsert_and_get(g):
    g.upsert_paper("X001", "Test", "Abstract.", published="2023-01-01", url="x", updated="2023-01-01")
    p = g.get_paper("X001")
    assert p is not None and p.title == "Test"

def test_upsert_updates(g):
    g.upsert_paper("X001", "Old", "Abstract.", published="2023-01-01", url="x", updated="2023-01-01")
    g.upsert_paper("X001", "New", "Abstract.", published="2023-01-01", url="x", updated="2023-01-01")
    assert g.get_paper("X001").title == "New"

def test_get_missing_returns_none(g):
    assert g.get_paper("nope") is None

def test_authors_order(g):
    g.upsert_paper("X001", "T", "A", published="2023-01-01", url="x", updated="2023-01-01")
    g.add_authors("X001", ["Zara", "Ben", "Maria"])
    assert g.get_paper_authors("X001") == ["Zara", "Ben", "Maria"]

def test_categories(g):
    g.upsert_paper("X001", "T", "A", published="2023-01-01", url="x", updated="2023-01-01")
    g.add_categories("X001", ["cs.LG", "stat.ML"])
    assert set(g.get_paper_categories("X001")) == {"cs.LG", "stat.ML"}

def test_concepts_sorted(g):
    g.upsert_paper("X001", "T", "A", published="2023-01-01", url="x", updated="2023-01-01")
    g.add_concepts("X001", [("low", 0.1), ("high", 1.0), ("mid", 0.5)])
    scores = [s for _, s in g.get_paper_concepts("X001")]
    assert scores == sorted(scores, reverse=True)

def test_search_by_title(populated):
    assert any(p.id == "A001" for p in populated.search_papers("Transformer"))

def test_search_by_abstract(populated):
    assert any(p.id == "A002" for p in populated.search_papers("molecules"))

def test_search_no_results(populated):
    assert populated.search_papers("zzz_no_match") == []

def test_top_concepts_cross_paper(populated):
    top = populated.top_concepts(10)
    attn = next(r for r in top if r.term == "attention")
    assert attn.paper_count == 2

def test_top_authors_count(populated):
    alice = next(r for r in populated.top_authors(10) if r.name == "Alice Smith")
    assert alice.paper_count == 2

def test_papers_by_author(populated):
    assert len(populated.papers_by_author("Alice")) == 2

def test_papers_by_concept(populated):
    assert len(populated.papers_by_concept("attention")) == 2

def test_stats(populated):
    s = populated.stats()
    assert s["papers"] == 2 and s["authors"] == 3

def test_delete_cascades(populated):
    populated.delete_paper("A001")
    assert populated.get_paper("A001") is None
    assert populated.papers_by_author("Bob Jones") == []

def test_delete_not_found(populated):
    assert populated.delete_paper("nonexistent") is False

def test_category_filter(populated):
    papers = populated.get_papers(category="cs.CL")
    assert all(p.id == "A001" for p in papers)
