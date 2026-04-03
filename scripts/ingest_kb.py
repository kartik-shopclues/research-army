"""
scripts/ingest_kb.py — Bulk document ingestion for domain knowledge bases
Supports: local files, directories, arXiv papers, URLs

Usage:
    # Ingest local files
    python scripts/ingest_kb.py --domain space --dir ./data/space/

    # Ingest arXiv papers by topic
    python scripts/ingest_kb.py --domain quantum --arxiv "quantum key distribution" --limit 50

    # Ingest a specific URL
    python scripts/ingest_kb.py --domain defence --url https://example.com/report.pdf

    # Ingest all domains from default data dirs
    python scripts/ingest_kb.py --all
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import (
    ingest_document, ingest_file, ingest_directory,
    init_collections, get_weaviate_client,
)

console = Console()


# ── arXiv ingestion ────────────────────────────────────────────────────────

ARXIV_CATEGORIES = {
    "space":   ["astro-ph.GA", "astro-ph.EP", "astro-ph.IM", "physics.space-ph"],
    "defence": ["cs.CR", "cs.AI", "eess.SP", "physics.ins-det"],
    "quantum": ["quant-ph", "cond-mat.supr-con", "cs.CC"],
}

SEED_QUERIES = {
    "space": [
        "orbital mechanics satellite navigation",
        "astrophysics dark matter cosmology",
        "space mission design propulsion",
        "exoplanet detection spectroscopy",
        "space debris mitigation LEO",
    ],
    "defence": [
        "cyber warfare threat intelligence",
        "military strategy autonomous systems",
        "electronic warfare signal jamming",
        "defence satellite communications",
        "geopolitics national security policy",
    ],
    "quantum": [
        "quantum key distribution QKD protocol",
        "quantum error correction surface code",
        "quantum computing algorithm optimization",
        "quantum cryptography post-quantum",
        "superconducting qubit decoherence",
    ],
}


async def ingest_arxiv(domain: str, query: str, limit: int = 20) -> int:
    """Search arXiv and ingest paper abstracts + full text."""
    console.print(f"[cyan]arXiv search:[/cyan] '{query}' → {domain}")

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
    except Exception as e:
        console.print(f"[red]arXiv request failed: {e}[/red]")
        return 0

    # Parse Atom XML
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)

    total = 0
    for entry in entries:
        title    = (entry.findtext("atom:title", "", ns) or "").strip()
        abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
        paper_id = (entry.findtext("atom:id", "", ns) or "").strip()
        authors  = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ]

        if not abstract:
            continue

        content = f"Title: {title}\n\nAuthors: {', '.join(authors[:5])}\n\nAbstract:\n{abstract}"
        n = await ingest_document(domain, content, source=f"arxiv:{paper_id.split('/')[-1]}")
        total += n
        await asyncio.sleep(0.1)  # rate limit

    console.print(f"  [green]{total} chunks from {len(entries)} arXiv papers[/green]")
    return total


async def ingest_url(domain: str, url: str) -> int:
    """Fetch a URL and ingest its text content."""
    console.print(f"[cyan]Fetching URL:[/cyan] {url}")
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "ResearchArmy/1.0"})
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            if "pdf" in content_type:
                # Save and process as PDF
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(resp.content)
                    tmp = f.name
                n = await ingest_file(domain, tmp)
                os.unlink(tmp)
                return n
            else:
                # Extract text from HTML
                text = resp.text
                # Simple HTML strip
                import re
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return await ingest_document(domain, text, source=url[:80])

    except Exception as e:
        console.print(f"[red]URL fetch failed: {e}[/red]")
        return 0


async def ingest_seed_queries(domain: str, limit_per_query: int = 15) -> int:
    """Ingest arXiv papers for all seed queries of a domain."""
    queries = SEED_QUERIES.get(domain, [])
    total = 0
    for q in queries:
        n = await ingest_arxiv(domain, q, limit=limit_per_query)
        total += n
        await asyncio.sleep(1)
    return total


async def ingest_all_domains(data_base_dir: str = "./data"):
    """Ingest local files for all three domains."""
    for domain in ["space", "defence", "quantum"]:
        domain_dir = os.path.join(data_base_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        console.rule(f"[bold]{domain.upper()} KB[/bold]")

        if any(Path(domain_dir).rglob("*.*")):
            n = await ingest_directory(domain, domain_dir)
            console.print(f"[green]Local: {n} chunks[/green]")
        else:
            console.print(f"[yellow]No local files in {domain_dir} — seeding from arXiv[/yellow]")
            n = await ingest_seed_queries(domain, limit_per_query=10)
            console.print(f"[green]arXiv seed: {n} chunks for {domain}[/green]")


async def show_kb_stats():
    """Print stats for all domain KBs."""
    from config.settings import DOMAIN_COLLECTIONS
    import weaviate

    client = get_weaviate_client()
    try:
        for domain, col_name in DOMAIN_COLLECTIONS.items():
            try:
                col = client.collections.get(col_name)
                agg = col.aggregate.over_all(total_count=True)
                count = agg.total_count or 0
                console.print(f"  [bold]{domain:10}[/bold] {count:6} chunks  ({col_name})")
            except Exception:
                console.print(f"  [dim]{domain:10} not found[/dim]")
    finally:
        client.close()


async def main_async(args):
    # Ensure Weaviate collections exist
    client = get_weaviate_client()
    init_collections(client)
    client.close()

    if args.stats:
        console.print("\n[bold]KB Statistics:[/bold]")
        await show_kb_stats()
        return

    if args.all:
        await ingest_all_domains(args.dir or "./data")
        console.print("\n[bold]Final KB stats:[/bold]")
        await show_kb_stats()
        return

    if not args.domain:
        console.print("[red]--domain required unless using --all or --stats[/red]")
        return

    total = 0

    if args.dir:
        total += await ingest_directory(args.domain, args.dir)

    if args.file:
        total += await ingest_file(args.domain, args.file)

    if args.url:
        total += await ingest_url(args.domain, args.url)

    if args.arxiv:
        total += await ingest_arxiv(args.domain, args.arxiv, limit=args.limit)

    if args.seed:
        total += await ingest_seed_queries(args.domain, limit_per_query=args.limit)

    console.print(f"\n[bold green]Total: {total} chunks ingested into {args.domain} KB[/bold green]")
    console.print("\n[bold]Updated KB stats:[/bold]")
    await show_kb_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest documents into domain KBs")
    parser.add_argument("--domain",  choices=["space", "defence", "quantum"], help="Target domain")
    parser.add_argument("--dir",     help="Directory of files to ingest")
    parser.add_argument("--file",    help="Single file to ingest")
    parser.add_argument("--url",     help="URL to fetch and ingest")
    parser.add_argument("--arxiv",   help="arXiv search query")
    parser.add_argument("--seed",    action="store_true", help="Ingest seed arXiv papers for domain")
    parser.add_argument("--limit",   type=int, default=20, help="Papers per arXiv query")
    parser.add_argument("--all",     action="store_true", help="Ingest all domains")
    parser.add_argument("--stats",   action="store_true", help="Show KB statistics")
    args = parser.parse_args()

    asyncio.run(main_async(args))
