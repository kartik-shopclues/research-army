"""
rag/pipeline.py — In-memory RAG: ingestion + retrieval
Uses numpy cosine similarity — no Weaviate / Docker required.
"""
import asyncio
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import httpx
import numpy as np
from rich.console import Console

from config.settings import settings, DOMAIN_COLLECTIONS

console = Console()


# ── In-memory vector store ─────────────────────────────────────────────────
# { domain: [ { content, source, domain, source_domain,
#               is_cross_domain, confidence, ingested_at,
#               chunk_id, vector } ] }
_store: Dict[str, List[Dict]] = {d: [] for d in DOMAIN_COLLECTIONS}


# ── Weaviate stubs (kept so imports don't break) ───────────────────────────

def get_weaviate_client():
    return None


def init_collections(client=None):
    """No-op — collections are in-memory."""
    for domain in DOMAIN_COLLECTIONS:
        if domain not in _store:
            _store[domain] = []
    console.print("[green]In-memory KB ready (no Weaviate needed)[/green]")


# ── Embedding via Ollama ───────────────────────────────────────────────────

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Ollama nomic-embed-text."""
    async with httpx.AsyncClient(timeout=120) as client:
        vectors = []
        for text in texts:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return vectors


async def embed_single(text: str) -> List[float]:
    vectors = await embed_texts([text])
    return vectors[0]


# ── Text chunking ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or settings.rag_chunk_size
    overlap    = overlap    or settings.rag_chunk_overlap

    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def doc_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Cosine similarity ──────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ── Ingestion ─────────────────────────────────────────────────────────────

async def ingest_document(
    domain: str,
    content: str,
    source: str,
    source_domain: str = None,
    is_cross_domain: bool = False,
    confidence: float = 1.0,
) -> int:
    """Chunk a document and store in the appropriate domain KB."""
    if domain not in _store:
        _store[domain] = []

    chunks = chunk_text(content)
    if not chunks:
        return 0

    console.print(f"[cyan]Ingesting {len(chunks)} chunks → {domain}KB[/cyan] ({source[:60]})")

    vectors = await embed_texts(chunks)
    d_hash  = doc_hash(content)
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        _store[domain].append({
            "content":         chunk,
            "source":          source,
            "domain":          domain,
            "chunk_id":        f"{d_hash}_{i}",
            "doc_hash":        d_hash,
            "ingested_at":     now_str,
            "source_domain":   source_domain or domain,
            "is_cross_domain": is_cross_domain,
            "confidence":      confidence,
            "vector":          vector,
        })

    console.print(f"[green]KB {domain}: {len(_store[domain])} total chunks[/green]")
    return len(chunks)


async def ingest_file(domain: str, file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        return 0

    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader  = PdfReader(str(path))
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")

    return await ingest_document(domain, content, source=path.name)


async def ingest_directory(domain: str, directory: str) -> int:
    total = 0
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith((".txt", ".pdf", ".md")):
                total += await ingest_file(domain, os.path.join(root, fname))
    return total


# ── Retrieval ─────────────────────────────────────────────────────────────

async def retrieve(
    domain: str,
    query: str,
    top_k: int = None,
    include_cross_domain: bool = True,
) -> List[Dict]:
    """Retrieve top-k relevant chunks using cosine similarity."""
    top_k = top_k or settings.rag_top_k
    docs  = _store.get(domain, [])

    if not docs:
        return []

    query_vector = await embed_single(query)

    scored = []
    for doc in docs:
        if not include_cross_domain and doc.get("is_cross_domain"):
            continue
        score = _cosine_similarity(query_vector, doc["vector"])
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "content":         doc["content"],
            "source":          doc.get("source", "unknown"),
            "domain":          doc.get("domain", domain),
            "source_domain":   doc.get("source_domain", domain),
            "is_cross_domain": doc.get("is_cross_domain", False),
            "confidence":      doc.get("confidence", 1.0),
            "score":           score,
        })
    return results


def format_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant documents found in knowledge base."
    lines = []
    for i, c in enumerate(chunks, 1):
        tag = " [CROSS-DOMAIN]" if c.get("is_cross_domain") else ""
        lines.append(
            f"[{i}] Source: {c['source']} | Domain: {c['source_domain']}{tag}\n"
            f"{c['content']}\n"
        )
    return "\n".join(lines)


# ── Delta extraction (for sync bus) ───────────────────────────────────────

async def get_delta_since(domain: str, since_iso: str) -> List[Dict]:
    """Return chunks ingested after since_iso."""
    docs = _store.get(domain, [])
    return [
        {k: v for k, v in d.items() if k != "vector"}  # exclude vector for sync payload
        for d in docs
        if d.get("ingested_at", "") > since_iso
    ]
