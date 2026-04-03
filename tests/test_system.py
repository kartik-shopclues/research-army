"""
tests/test_system.py — End-to-end system tests
Run: python tests/test_system.py

Tests every layer: Ollama, Weaviate, RAG, Commander, Sync, Debate
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()

PASS = "[bold green]PASS[/bold green]"
FAIL = "[bold red]FAIL[/bold red]"
SKIP = "[bold yellow]SKIP[/bold yellow]"

results = []


def record(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    icon = "✓" if "PASS" in status else ("✗" if "FAIL" in status else "–")
    console.print(f"  {icon} {name}: {status} {detail}")


# ── 1. Ollama connectivity ─────────────────────────────────────────────────
async def test_ollama():
    console.rule("[cyan]1. Ollama[/cyan]")
    import httpx
    from config.settings import settings

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            record("Ollama API", PASS, f"{len(models)} models")

            # Check each required model
            required = [
                settings.commander_model,
                settings.space_model,
                settings.defence_model,
                settings.quantum_model,
                settings.synthesis_model,
                settings.embed_model,
            ]
            for m in required:
                short = m.split(":")[0]
                found = any(short in ml for ml in models)
                record(f"Model: {m[:40]}", PASS if found else FAIL)
    except Exception as e:
        record("Ollama API", FAIL, str(e))


# ── 2. In-memory KB ───────────────────────────────────────────────────────
async def test_weaviate():
    console.rule("[cyan]2. In-memory KB[/cyan]")
    try:
        from rag.pipeline import init_collections, _store
        init_collections()
        domains = list(_store.keys())
        assert len(domains) == 3
        record("In-memory KB init", PASS, f"domains: {domains}")
    except Exception as e:
        record("In-memory KB", FAIL, str(e))


# ── 3. In-memory session store ────────────────────────────────────────────
async def test_redis():
    console.rule("[cyan]3. In-memory Session Store[/cyan]")
    try:
        from memory.store import MemoryStore
        store = MemoryStore()
        await store.add_message("test_session", "user", "hello")
        history = await store.get_history("test_session")
        assert len(history) == 1
        assert history[0]["content"] == "hello"

        await store.cache_result("test query", {"answer": 42})
        cached = await store.get_cached_result("test query")
        assert cached == {"answer": 42}

        await store.close()
        record("MemoryStore read/write + cache", PASS)
    except Exception as e:
        record("MemoryStore", FAIL, str(e))


# ── 4. Embedding ───────────────────────────────────────────────────────────
async def test_embedding():
    console.rule("[cyan]4. Embedding[/cyan]")
    try:
        from rag.pipeline import embed_single
        vec = await embed_single("quantum key distribution satellite")
        assert len(vec) > 100
        record("nomic-embed-text", PASS, f"dim={len(vec)}")
    except Exception as e:
        record("nomic-embed-text", FAIL, str(e))


# ── 5. RAG ingestion + retrieval ───────────────────────────────────────────
async def test_rag():
    console.rule("[cyan]5. RAG pipeline[/cyan]")
    try:
        from rag.pipeline import ingest_document, retrieve

        test_text = (
            "Quantum key distribution (QKD) uses quantum mechanical properties "
            "to enable two parties to produce a shared random secret key. "
            "The BB84 protocol, proposed by Bennett and Brassard in 1984, "
            "is the first and most well-known QKD protocol. "
            "It uses four quantum states to encode bits in two conjugate bases."
        )

        n = await ingest_document("quantum", test_text, source="test_doc.txt")
        record("Ingest document", PASS, f"{n} chunks")

        results_r = await retrieve("quantum", "BB84 quantum protocol")
        assert len(results_r) > 0
        record("Retrieve from KB", PASS, f"{len(results_r)} chunks returned")
        record("Top chunk preview", PASS, results_r[0]["content"][:60] + "...")

    except Exception as e:
        record("RAG pipeline", FAIL, str(e))


# ── 6. Commander LLM ───────────────────────────────────────────────────────
async def test_commander():
    console.rule("[cyan]6. Commander LLM[/cyan]")
    try:
        from agents.commander import CommanderAgent
        cmd = CommanderAgent()
        plan = await cmd.analyze_query(
            "How does quantum key distribution work for securing satellite links?"
        )
        assert "mode" in plan
        assert "domains" in plan
        record("Commander routing", PASS, f"mode={plan['mode']} domains={plan['domains']}")
        record("Sub-tasks generated", PASS, str(list(plan.get("sub_tasks", {}).keys())))
    except Exception as e:
        record("Commander LLM", FAIL, str(e))


# ── 7. Specialist LLM (quick) ──────────────────────────────────────────────
async def test_specialist():
    console.rule("[cyan]7. Specialist LLM[/cyan]")
    try:
        from agents.specialist import SpecialistAgent
        agent = SpecialistAgent("quantum")
        out = await agent.respond("What is quantum entanglement?")
        assert len(out["response"]) > 50
        record("Quantum specialist", PASS, f"{len(out['response'])} chars")
    except Exception as e:
        record("Specialist LLM", FAIL, str(e))


# ── 8. Knowledge sync ──────────────────────────────────────────────────────
async def test_sync():
    console.rule("[cyan]8. Knowledge sync bus[/cyan]")
    try:
        from sync.bus import SyncBus
        bus = SyncBus()
        ts = await bus.get_last_sync_time("space")
        record("Sync state read", PASS, f"last sync: {ts[:19]}")
        await bus.close()
    except Exception as e:
        record("Sync bus", FAIL, str(e))


# ── 9. Full pipeline (Mode B) ──────────────────────────────────────────────
async def test_full_pipeline():
    console.rule("[cyan]9. Full pipeline - Mode B[/cyan]")
    try:
        from agents.orchestrator import ResearchOrchestrator
        orc = ResearchOrchestrator()
        result = await orc.research(
            "What is quantum entanglement?",
            force_mode="mode_b",
        )
        assert "synthesis" in result
        assert len(result["synthesis"]) > 100
        record("Mode B pipeline", PASS, f"synthesis={len(result['synthesis'])} chars")
        await orc.close()
    except Exception as e:
        record("Mode B pipeline", FAIL, str(e))


# ── Summary ────────────────────────────────────────────────────────────────
async def run_all():
    console.rule("[bold magenta]Research Army — System Tests[/bold magenta]")

    await test_ollama()
    await test_weaviate()
    await test_redis()
    await test_embedding()
    await test_rag()
    await test_commander()
    await test_specialist()
    await test_sync()
    # Full pipeline test is slow — comment out for quick checks
    # await test_full_pipeline()

    # Summary table
    console.rule("[bold]Results[/bold]")
    table = Table(show_header=True)
    table.add_column("Test", style="bold")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    passes = fails = 0
    for name, status, detail in results:
        table.add_row(name, status, detail)
        if "PASS" in status: passes += 1
        elif "FAIL" in status: fails += 1

    console.print(table)
    console.print(f"\n[green]{passes} passed[/green]  [red]{fails} failed[/red]  "
                  f"out of {len(results)} checks\n")

    if fails > 0:
        console.print("[yellow]Check setup.sh was run and all services are running.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all())
