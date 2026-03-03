"""
litmap CLI — research literature knowledge graph tool.
"""
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

app = typer.Typer(name="litmap", help="Research literature knowledge graph.", add_completion=False)
ingest_app = typer.Typer(help="Ingest papers from arXiv.")
app.add_typer(ingest_app, name="ingest")

console = Console()


def _db_path(db):
    if db:
        return db
    env = os.environ.get("LITMAP_DB")
    return Path(env) if env else None


@ingest_app.command("query")
def ingest_query_cmd(
    query: str = typer.Argument(..., help="arXiv search query"),
    max_results: int = typer.Option(20, "--max", "-n"),
    top_concepts: int = typer.Option(15, "--concepts", "-c"),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """Fetch papers matching QUERY from arXiv and ingest into the local graph."""
    from .ingest import ingest_query
    console.print(f"[bold cyan]Fetching arXiv papers:[/] {query!r}  (max={max_results})")
    with console.status("[bold green]Ingesting..."):
        ids = ingest_query(query=query, max_results=max_results,
                           db_path=_db_path(db), top_concepts=top_concepts, verbose=False)
    console.print(f"[bold green]✓ Ingested {len(ids)} papers[/]")
    for pid in ids:
        console.print(f"  [dim]{pid}[/]")


@ingest_app.command("id")
def ingest_id_cmd(
    arxiv_id: str = typer.Argument(...),
    top_concepts: int = typer.Option(15, "--concepts", "-c"),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """Ingest a single paper by its arXiv ID."""
    from .ingest import ingest_id
    console.print(f"[bold cyan]Fetching paper:[/] {arxiv_id}")
    with console.status("[bold green]Ingesting..."):
        pid = ingest_id(arxiv_id=arxiv_id, db_path=_db_path(db), top_concepts=top_concepts)
    if pid:
        console.print(f"[bold green]✓ Ingested:[/] {pid}")
    else:
        console.print(f"[bold red]✗ Paper not found:[/] {arxiv_id}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit", "-n"),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """Search locally ingested papers by keyword."""
    from .graph import LitmapGraph
    with LitmapGraph(db_path=_db_path(db)) as graph:
        papers = graph.search_papers(query, limit=limit)
    if not papers:
        console.print("[yellow]No results found.[/]")
        return
    table = Table(title=f'Search: "{query}"', box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("arXiv ID", style="cyan", no_wrap=True)
    table.add_column("Published", style="dim", width=10)
    table.add_column("Title", style="white")
    for p in papers:
        table.add_row(p.id, p.published, p.title[:80])
    console.print(table)


@app.command()
def show(
    arxiv_id: str = typer.Argument(...),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """Display full details for a paper."""
    from .graph import LitmapGraph
    with LitmapGraph(db_path=_db_path(db)) as graph:
        paper = graph.get_paper(arxiv_id)
        if not paper:
            console.print(f"[bold red]Paper not found:[/] {arxiv_id}")
            raise typer.Exit(1)
        authors = graph.get_paper_authors(arxiv_id)
        categories = graph.get_paper_categories(arxiv_id)
        concepts = graph.get_paper_concepts(arxiv_id)
    meta_lines = [
        f"[cyan]arXiv ID:[/]    {paper.id}",
        f"[cyan]Published:[/]   {paper.published}",
        f"[cyan]Authors:[/]     {', '.join(authors[:5])}{'...' if len(authors) > 5 else ''}",
        f"[cyan]Categories:[/]  {', '.join(categories)}",
        f"[cyan]DOI:[/]         {paper.doi or '—'}",
        "",
        f"[bold]Abstract:[/]\n{paper.abstract[:600]}{'...' if len(paper.abstract) > 600 else ''}",
        "",
        f"[bold]Top Concepts:[/] {', '.join(t for t, _ in concepts[:10])}",
    ]
    console.print(Panel("\n".join(meta_lines), title=paper.title[:70], border_style="cyan", padding=(1, 2)))


@app.command()
def concepts(
    limit: int = typer.Option(20, "--limit", "-n"),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """List top concepts across all ingested papers."""
    from .graph import LitmapGraph
    with LitmapGraph(db_path=_db_path(db)) as graph:
        results = graph.top_concepts(limit=limit)
    table = Table(title="Top Concepts", box=box.SIMPLE_HEAVY)
    table.add_column("Term", style="green")
    table.add_column("Papers", justify="right", style="cyan")
    table.add_column("Avg Score", justify="right", style="dim")
    for r in results:
        table.add_row(r.term, str(r.paper_count), f"{r.avg_score:.3f}")
    console.print(table)


@app.command()
def authors(
    limit: int = typer.Option(20, "--limit", "-n"),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """List top authors by paper count."""
    from .graph import LitmapGraph
    with LitmapGraph(db_path=_db_path(db)) as graph:
        results = graph.top_authors(limit=limit)
    table = Table(title="Top Authors", box=box.SIMPLE_HEAVY)
    table.add_column("Author", style="green")
    table.add_column("Papers", justify="right", style="cyan")
    for r in results:
        table.add_row(r.name, str(r.paper_count))
    console.print(table)


@app.command()
def stats(db: Optional[Path] = typer.Option(None, "--db")):
    """Show database statistics."""
    from .graph import LitmapGraph
    with LitmapGraph(db_path=_db_path(db)) as graph:
        s = graph.stats()
    console.print(Panel(
        "\n".join([
            f"[cyan]Database:[/]   {s['db_path']}",
            f"[cyan]Papers:[/]     {s['papers']}",
            f"[cyan]Authors:[/]    {s['authors']}",
            f"[cyan]Concepts:[/]   {s['concepts']}",
            f"[cyan]Categories:[/] {s['categories']}",
        ]),
        title="[bold]litmap stats[/]", border_style="green",
    ))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
    db: Optional[Path] = typer.Option(None, "--db"),
):
    """Start the litmap FastAPI server."""
    import uvicorn
    if db:
        os.environ["LITMAP_DB"] = str(db)
    console.print(f"[bold green]Starting litmap API on http://{host}:{port}[/]")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/]")
    uvicorn.run("litmap.api:app", host=host, port=port, reload=reload)


def main():
    app()


if __name__ == "__main__":
    main()
