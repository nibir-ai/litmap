# litmap

> Research literature knowledge graph — arXiv ingest, concept extraction, and local query.

`litmap` builds a local SQLite knowledge graph from arXiv papers. Fetch paper metadata, extract concepts from abstracts, link authors and categories — query everything via CLI or REST API. Entirely offline, no cloud, no subscriptions.

---

## Problem Statement

Researchers accumulate hundreds of papers with no local, queryable record of the concept space they cover. Zotero and Mendeley are reference managers, not knowledge graphs. `litmap` gives you a local graph you can query by concept, author, or category.

---

## Architecture

```
arXiv API
    │
    ▼
litmap/arxiv.py        ← rate-limited HTTP + XML parsing → Paper dataclass
    │
    ▼
litmap/extractor.py    ← TF-based concept extraction (zero ML deps)
    │
    ▼
litmap/graph.py        ← SQLite knowledge graph (WAL mode, cascading deletes)
    │
    ├── litmap/cli.py  ← Typer CLI
    └── litmap/api.py  ← FastAPI REST API
```

Data lives at `~/.litmap/graph.db` by default.

---

## Installation

```bash
git clone https://github.com/nibir-ai/litmap.git
cd litmap
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
litmap --help
```

---

## CLI Usage

```bash
# Ingest papers
litmap ingest query "attention mechanism transformer" --max 20
litmap ingest id 2310.06825

# Search and browse
litmap search "diffusion models"
litmap show 2310.06825
litmap concepts --limit 30
litmap authors
litmap stats

# Start REST API
litmap serve
# → http://127.0.0.1:8000/docs
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | DB stats |
| GET | `/papers` | List papers |
| GET | `/papers/search?q=...` | Keyword search |
| GET | `/papers/{id}` | Single paper |
| POST | `/ingest/query` | Ingest from arXiv query |
| POST | `/ingest/id` | Ingest by arXiv ID |
| GET | `/concepts` | Top concepts |
| GET | `/authors` | Top authors |

```bash
# Example
curl -X POST http://localhost:8000/ingest/query \
  -H "Content-Type: application/json" \
  -d '{"query": "diffusion models", "max_results": 15}'

curl "http://localhost:8000/concepts?limit=20"
```

---

## Running Tests

```bash
pytest
pytest --cov=litmap --cov-report=term-missing
```

---

## Version History

### v0.1.0
- arXiv API client with rate limiting and XML parsing
- SQLite knowledge graph (papers, authors, concepts, categories)
- TF-based concept extraction — zero NLP dependencies
- Typer CLI: 8 commands
- FastAPI REST API: 10 endpoints with Pydantic v2
- 45 unit tests

### v0.2.0 (planned)
- SQLite FTS5 full-text search
- Optional spaCy noun-phrase extraction
- Citation graph (paper → paper edges)
- Semantic similarity via sentence-transformers (CPU)
- GEXF export for Gephi visualization

---

## License

MIT — nibir-ai
