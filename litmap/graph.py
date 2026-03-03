"""
SQLite-backed knowledge graph for research literature.
"""
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

DEFAULT_DB_PATH = Path.home() / ".litmap" / "graph.db"

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL DEFAULT '',
    published    TEXT NOT NULL DEFAULT '',
    updated      TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    doi          TEXT,
    journal_ref  TEXT,
    inserted_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS authors (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id   TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id  INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE IF NOT EXISTS concepts (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    term  TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS paper_concepts (
    paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id  INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    score       REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE TABLE IF NOT EXISTS categories (
    paper_id  TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    category  TEXT NOT NULL,
    PRIMARY KEY (paper_id, category)
);

CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX IF NOT EXISTS idx_paper_concepts_concept ON paper_concepts(concept_id);
CREATE INDEX IF NOT EXISTS idx_categories_category ON categories(category);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
"""


@dataclass
class PaperRow:
    id: str
    title: str
    abstract: str
    published: str
    updated: str
    url: str
    doi: Optional[str]
    journal_ref: Optional[str]
    inserted_at: str


@dataclass
class ConceptResult:
    term: str
    paper_count: int
    avg_score: float


@dataclass
class AuthorResult:
    name: str
    paper_count: int


class LitmapGraph:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def __enter__(self) -> "LitmapGraph":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def upsert_paper(self, paper_id, title, abstract, published="",
                     updated="", url="", doi=None, journal_ref=None):
        with self._tx() as conn:
            conn.execute("""
                INSERT INTO papers (id, title, abstract, published, updated, url, doi, journal_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, abstract=excluded.abstract,
                    updated=excluded.updated, doi=excluded.doi,
                    journal_ref=excluded.journal_ref
            """, (paper_id, title, abstract, published, updated, url, doi, journal_ref))

    def add_authors(self, paper_id: str, authors: list[str]) -> None:
        with self._tx() as conn:
            for position, name in enumerate(authors):
                conn.execute("INSERT OR IGNORE INTO authors (name) VALUES (?)", (name,))
                row = conn.execute(
                    "SELECT id FROM authors WHERE name = ? COLLATE NOCASE", (name,)
                ).fetchone()
                conn.execute(
                    "INSERT OR IGNORE INTO paper_authors (paper_id, author_id, position) VALUES (?, ?, ?)",
                    (paper_id, row["id"], position),
                )

    def add_categories(self, paper_id: str, categories: list[str]) -> None:
        with self._tx() as conn:
            for cat in categories:
                conn.execute(
                    "INSERT OR IGNORE INTO categories (paper_id, category) VALUES (?, ?)",
                    (paper_id, cat),
                )

    def add_concepts(self, paper_id: str, concepts: list[tuple[str, float]]) -> None:
        with self._tx() as conn:
            for term, score in concepts:
                conn.execute("INSERT OR IGNORE INTO concepts (term) VALUES (?)", (term,))
                row = conn.execute(
                    "SELECT id FROM concepts WHERE term = ? COLLATE NOCASE", (term,)
                ).fetchone()
                conn.execute("""
                    INSERT INTO paper_concepts (paper_id, concept_id, score)
                    VALUES (?, ?, ?)
                    ON CONFLICT(paper_id, concept_id) DO UPDATE SET score=excluded.score
                """, (paper_id, row["id"], score))

    def delete_paper(self, paper_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            return cur.rowcount > 0

    def get_paper(self, paper_id: str) -> Optional[PaperRow]:
        row = self._conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return _row_to_paper(row) if row else None

    def get_papers(self, limit=20, offset=0, category=None) -> list[PaperRow]:
        if category:
            rows = self._conn.execute("""
                SELECT p.* FROM papers p JOIN categories c ON c.paper_id = p.id
                WHERE c.category = ? ORDER BY p.published DESC LIMIT ? OFFSET ?
            """, (category, limit, offset)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM papers ORDER BY published DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        return [_row_to_paper(r) for r in rows]

    def search_papers(self, query: str, limit: int = 20) -> list[PaperRow]:
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM papers WHERE title LIKE ? OR abstract LIKE ? ORDER BY published DESC LIMIT ?",
            (pattern, pattern, limit)
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    def get_paper_authors(self, paper_id: str) -> list[str]:
        rows = self._conn.execute("""
            SELECT a.name FROM authors a JOIN paper_authors pa ON pa.author_id = a.id
            WHERE pa.paper_id = ? ORDER BY pa.position
        """, (paper_id,)).fetchall()
        return [r["name"] for r in rows]

    def get_paper_concepts(self, paper_id: str) -> list[tuple[str, float]]:
        rows = self._conn.execute("""
            SELECT c.term, pc.score FROM concepts c JOIN paper_concepts pc ON pc.concept_id = c.id
            WHERE pc.paper_id = ? ORDER BY pc.score DESC
        """, (paper_id,)).fetchall()
        return [(r["term"], r["score"]) for r in rows]

    def get_paper_categories(self, paper_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT category FROM categories WHERE paper_id = ? ORDER BY category", (paper_id,)
        ).fetchall()
        return [r["category"] for r in rows]

    def top_concepts(self, limit: int = 20) -> list[ConceptResult]:
        rows = self._conn.execute("""
            SELECT c.term, COUNT(pc.paper_id) AS paper_count, AVG(pc.score) AS avg_score
            FROM concepts c JOIN paper_concepts pc ON pc.concept_id = c.id
            GROUP BY c.id ORDER BY paper_count DESC, avg_score DESC LIMIT ?
        """, (limit,)).fetchall()
        return [ConceptResult(term=r["term"], paper_count=r["paper_count"], avg_score=r["avg_score"]) for r in rows]

    def top_authors(self, limit: int = 20) -> list[AuthorResult]:
        rows = self._conn.execute("""
            SELECT a.name, COUNT(pa.paper_id) AS paper_count
            FROM authors a JOIN paper_authors pa ON pa.author_id = a.id
            GROUP BY a.id ORDER BY paper_count DESC LIMIT ?
        """, (limit,)).fetchall()
        return [AuthorResult(name=r["name"], paper_count=r["paper_count"]) for r in rows]

    def papers_by_author(self, author_name: str, limit: int = 50) -> list[PaperRow]:
        rows = self._conn.execute("""
            SELECT p.* FROM papers p JOIN paper_authors pa ON pa.paper_id = p.id
            JOIN authors a ON a.id = pa.author_id
            WHERE a.name LIKE ? COLLATE NOCASE ORDER BY p.published DESC LIMIT ?
        """, (f"%{author_name}%", limit)).fetchall()
        return [_row_to_paper(r) for r in rows]

    def papers_by_concept(self, term: str, limit: int = 50) -> list[PaperRow]:
        rows = self._conn.execute("""
            SELECT p.* FROM papers p JOIN paper_concepts pc ON pc.paper_id = p.id
            JOIN concepts c ON c.id = pc.concept_id
            WHERE c.term LIKE ? COLLATE NOCASE ORDER BY pc.score DESC LIMIT ?
        """, (f"%{term}%", limit)).fetchall()
        return [_row_to_paper(r) for r in rows]

    def stats(self) -> dict:
        return {
            "papers": self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "authors": self._conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0],
            "concepts": self._conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
            "categories": self._conn.execute("SELECT COUNT(DISTINCT category) FROM categories").fetchone()[0],
            "db_path": str(self.db_path),
        }


def _row_to_paper(row: sqlite3.Row) -> PaperRow:
    return PaperRow(
        id=row["id"], title=row["title"], abstract=row["abstract"],
        published=row["published"], updated=row["updated"], url=row["url"],
        doi=row["doi"], journal_ref=row["journal_ref"], inserted_at=row["inserted_at"],
    )
