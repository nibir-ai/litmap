"""
FastAPI REST API for litmap.
"""
from pathlib import Path
from typing import Optional
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graph import LitmapGraph
from .ingest import ingest_query, ingest_id

DB_PATH = Path(os.environ.get("LITMAP_DB", str(Path.home() / ".litmap" / "graph.db")))

app = FastAPI(title="litmap API",
              description="Research literature knowledge graph — arXiv ingest + query",
              version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IngestQueryRequest(BaseModel):
    query: str
    max_results: int = 20
    top_concepts: int = 15


class IngestIdRequest(BaseModel):
    arxiv_id: str
    top_concepts: int = 15


class IngestResponse(BaseModel):
    ingested: list[str]
    count: int


class PaperResponse(BaseModel):
    id: str
    title: str
    abstract: str
    published: str
    updated: str
    url: str
    doi: Optional[str]
    journal_ref: Optional[str]
    authors: list[str]
    categories: list[str]
    concepts: list[dict]


class ConceptResponse(BaseModel):
    term: str
    paper_count: int
    avg_score: float


class AuthorResponse(BaseModel):
    name: str
    paper_count: int


def _get_graph():
    return LitmapGraph(db_path=DB_PATH)


def _enrich_paper(paper_id, graph):
    paper = graph.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id!r} not found")
    return PaperResponse(
        id=paper.id, title=paper.title, abstract=paper.abstract,
        published=paper.published, updated=paper.updated, url=paper.url,
        doi=paper.doi, journal_ref=paper.journal_ref,
        authors=graph.get_paper_authors(paper_id),
        categories=graph.get_paper_categories(paper_id),
        concepts=[{"term": t, "score": s} for t, s in graph.get_paper_concepts(paper_id)],
    )


@app.get("/health")
def health():
    with _get_graph() as graph:
        return {"status": "ok", **graph.stats()}


@app.get("/papers", response_model=list[PaperResponse])
def list_papers(limit: int = Query(default=20, ge=1, le=100),
                offset: int = Query(default=0, ge=0),
                category: Optional[str] = None):
    with _get_graph() as graph:
        papers = graph.get_papers(limit=limit, offset=offset, category=category)
        return [_enrich_paper(p.id, graph) for p in papers]


@app.get("/papers/search", response_model=list[PaperResponse])
def search_papers(q: str = Query(..., min_length=2),
                  limit: int = Query(default=20, ge=1, le=100)):
    with _get_graph() as graph:
        papers = graph.search_papers(q, limit=limit)
        return [_enrich_paper(p.id, graph) for p in papers]


@app.get("/papers/{paper_id:path}", response_model=PaperResponse)
def get_paper(paper_id: str):
    with _get_graph() as graph:
        return _enrich_paper(paper_id, graph)


@app.post("/ingest/query", response_model=IngestResponse)
def ingest_from_query(req: IngestQueryRequest):
    ids = ingest_query(query=req.query, max_results=req.max_results,
                       db_path=DB_PATH, top_concepts=req.top_concepts)
    return IngestResponse(ingested=ids, count=len(ids))


@app.post("/ingest/id", response_model=IngestResponse)
def ingest_from_id(req: IngestIdRequest):
    paper_id = ingest_id(arxiv_id=req.arxiv_id, db_path=DB_PATH, top_concepts=req.top_concepts)
    if not paper_id:
        raise HTTPException(status_code=404, detail=f"arXiv ID {req.arxiv_id!r} not found")
    return IngestResponse(ingested=[paper_id], count=1)


@app.get("/concepts", response_model=list[ConceptResponse])
def top_concepts(limit: int = Query(default=30, ge=1, le=200)):
    with _get_graph() as graph:
        return [ConceptResponse(term=r.term, paper_count=r.paper_count, avg_score=r.avg_score)
                for r in graph.top_concepts(limit=limit)]


@app.get("/concepts/{term}/papers", response_model=list[PaperResponse])
def papers_by_concept(term: str, limit: int = Query(default=20, ge=1, le=100)):
    with _get_graph() as graph:
        papers = graph.papers_by_concept(term, limit=limit)
        return [_enrich_paper(p.id, graph) for p in papers]


@app.get("/authors", response_model=list[AuthorResponse])
def top_authors(limit: int = Query(default=20, ge=1, le=200)):
    with _get_graph() as graph:
        return [AuthorResponse(name=r.name, paper_count=r.paper_count)
                for r in graph.top_authors(limit=limit)]


@app.get("/authors/{name}/papers", response_model=list[PaperResponse])
def papers_by_author(name: str, limit: int = Query(default=20, ge=1, le=100)):
    with _get_graph() as graph:
        papers = graph.papers_by_author(name, limit=limit)
        return [_enrich_paper(p.id, graph) for p in papers]
