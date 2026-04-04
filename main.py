"""
main.py — Research Army entry point
Run with: python main.py
"""
import asyncio
import sys
import uvicorn
from rich.console import Console
from rich.panel import Panel

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold green]Research Army[/bold green]\n"
        "[dim]Space · Defence · Quantum — Multi-LLM RAG System[/dim]\n\n"
        "[cyan]Space LLM:[/cyan]    mistral:7b\n"
        "[yellow]Defence LLM:[/yellow]  llama3.1:8b\n"
        "[magenta]Quantum LLM:[/magenta]  qwen3:8b\n"
        "[green]Commander:[/green]   qwen3:8b\n"
        "[blue]Synthesis:[/blue]   gemma2:27b\n"
        "[white]Embedder:[/white]    nomic-embed-text\n\n"
        "[dim]UI → http://localhost:8000[/dim]",
        title="[bold]v1.0.0[/bold]",
        border_style="green",
    ))


def main():
    print_banner()

    # CLI subcommands
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "ingest":
            # python3 main.py ingest space ./data/space_docs/
            from rag.pipeline import ingest_directory, init_collections

            domain    = sys.argv[2] if len(sys.argv) > 2 else "space"
            directory = sys.argv[3] if len(sys.argv) > 3 else "./data"

            async def _run():
                init_collections()
                n = await ingest_directory(domain, directory)
                console.print(f"[green]Ingested {n} chunks into {domain} KB[/green]")

            asyncio.run(_run())
            return

        if cmd == "sync":
            from sync.bus import SyncBus
            async def _sync():
                bus = SyncBus()
                result = await bus.run_sync(force=True)
                console.print(result)
                await bus.close()
            asyncio.run(_sync())
            return

        if cmd == "query":
            # python main.py query "How does QKD work in space?"
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Tell me about quantum communication"
            from agents.orchestrator import ResearchOrchestrator

            async def _query():
                orc = ResearchOrchestrator()
                result = await orc.research(query)
                console.rule("Result")
                console.print(result.get("synthesis", result))
                await orc.close()

            asyncio.run(_query())
            return

    # Default: start API server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=60,
    )


if __name__ == "__main__":
    main()
