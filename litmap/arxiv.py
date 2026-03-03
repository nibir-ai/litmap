"""
arXiv API client.
Fetches paper metadata via the arXiv Atom feed API.
Rate-limit: 1 request per 3 seconds as per arXiv policy.
"""
import xml.etree.ElementTree as ET
import time
from dataclasses import dataclass, field
from typing import Optional
import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_last_request_time: float = 0.0
_MIN_INTERVAL: float = 3.0


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    url: str = ""
    doi: Optional[str] = None
    journal_ref: Optional[str] = None


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def fetch_papers(
    query: str,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "relevance",
) -> list[Paper]:
    if max_results > 100:
        raise ValueError("arXiv API max_results cannot exceed 100 per request.")
    _rate_limit()
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_feed(resp.text)


def fetch_by_id(arxiv_id: str) -> Optional[Paper]:
    _rate_limit()
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    params = {"id_list": clean_id, "max_results": 1}
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    papers = _parse_feed(resp.text)
    return papers[0] if papers else None


def fetch_bulk(
    query: str,
    total: int = 50,
    batch_size: int = 25,
) -> list[Paper]:
    papers: list[Paper] = []
    fetched = 0
    while fetched < total:
        batch = min(batch_size, total - fetched)
        results = fetch_papers(query, max_results=batch, start=fetched)
        if not results:
            break
        papers.extend(results)
        fetched += len(results)
        if len(results) < batch:
            break
    return papers


def _parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        id_elem = entry.find("atom:id", NS)
        title_elem = entry.find("atom:title", NS)
        summary_elem = entry.find("atom:summary", NS)
        published_elem = entry.find("atom:published", NS)
        updated_elem = entry.find("atom:updated", NS)
        if id_elem is None or title_elem is None:
            continue
        raw_id = id_elem.text or ""
        arxiv_id = raw_id.split("/abs/")[-1].strip()
        title = (title_elem.text or "").strip().replace("\n", " ")
        abstract = (summary_elem.text if summary_elem is not None else "").strip().replace("\n", " ")
        published = (published_elem.text or "")[:10]
        updated = (updated_elem.text or "")[:10]
        url = raw_id.strip()
        authors = [
            (a.find("atom:name", NS).text or "").strip()
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]
        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", NS)
            if c.get("term")
        ]
        doi_elem = entry.find("arxiv:doi", NS)
        doi = doi_elem.text.strip() if doi_elem is not None else None
        jref_elem = entry.find("arxiv:journal_ref", NS)
        journal_ref = jref_elem.text.strip() if jref_elem is not None else None
        papers.append(Paper(
            arxiv_id=arxiv_id, title=title, abstract=abstract,
            authors=authors, categories=categories,
            published=published, updated=updated, url=url,
            doi=doi, journal_ref=journal_ref,
        ))
    return papers
