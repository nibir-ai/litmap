"""
Ingest pipeline: arXiv → extraction → graph.
"""
from pathlib import Path
from typing import Optional

from .arxiv import Paper, fetch_papers, fetch_by_id, fetch_bulk
from .extractor import extract_concepts
from .graph import LitmapGraph


def ingest_query(query, max_results=20, db_path=None, top_concepts=15, verbose=False):
    papers = fetch_papers(query, max_results=max_results)
    if verbose:
        print(f"[ingest] fetched {len(papers)} papers for query: {query!r}")
    return _ingest_papers(papers, db_path=db_path, top_concepts=top_concepts, verbose=verbose)


def ingest_id(arxiv_id, db_path=None, top_concepts=15, verbose=False):
    paper = fetch_by_id(arxiv_id)
    if not paper:
        if verbose:
            print(f"[ingest] paper not found: {arxiv_id}")
        return None
    ids = _ingest_papers([paper], db_path=db_path, top_concepts=top_concepts, verbose=verbose)
    return ids[0] if ids else None


def ingest_bulk(query, total=100, db_path=None, top_concepts=15, verbose=False):
    papers = fetch_bulk(query, total=total)
    if verbose:
        print(f"[ingest] bulk fetched {len(papers)} papers")
    return _ingest_papers(papers, db_path=db_path, top_concepts=top_concepts, verbose=verbose)


def _ingest_papers(papers, db_path, top_concepts, verbose):
    ingested_ids = []
    with LitmapGraph(db_path=db_path) as graph:
        for paper in papers:
            graph.upsert_paper(
                paper_id=paper.arxiv_id, title=paper.title, abstract=paper.abstract,
                published=paper.published, updated=paper.updated, url=paper.url,
                doi=paper.doi, journal_ref=paper.journal_ref,
            )
            graph.add_authors(paper.arxiv_id, paper.authors)
            graph.add_categories(paper.arxiv_id, paper.categories)
            concepts = extract_concepts(paper.abstract, title=paper.title, top_n=top_concepts)
            if concepts:
                graph.add_concepts(paper.arxiv_id, concepts)
            ingested_ids.append(paper.arxiv_id)
            if verbose:
                print(f"[ingest] ✓ {paper.arxiv_id}  {paper.title[:60]}...")
    return ingested_ids

