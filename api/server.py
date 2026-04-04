"""
api/server.py — FastAPI backend
REST endpoints + WebSocket streaming
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from rich.console import Console

from agents.orchestrator import ResearchOrchestrator
from rag.pipeline import init_collections, ingest_document, ingest_file
from sync.bus import SyncBus, run_sync_scheduler
from memory.store import MemoryStore
from config.settings import settings
from api.finetune import router as finetune_router

console = Console()

# Globals (initialised in lifespan)
orchestrator: ResearchOrchestrator = None
sync_bus: MemoryStore = None
memory: MemoryStore = None
_sync_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, sync_bus, memory, _sync_task

    console.rule("[bold green]Research Army — Starting Up[/bold green]")

    # Init in-memory KB collections
    init_collections()

    orchestrator = ResearchOrchestrator()
    sync_bus     = SyncBus()
    memory       = MemoryStore()

    # Start sync scheduler in background
    _sync_task = asyncio.create_task(run_sync_scheduler())

    console.print("[green]Server ready[/green]")
    yield

    # Cleanup
    _sync_task.cancel()
    await orchestrator.close()
    await sync_bus.close()
    await memory.close()


app = FastAPI(
    title="Research Army API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(finetune_router)


# ── Request / Response models ──────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query:         str
    session_id:    Optional[str]  = None
    force_mode:    Optional[str]  = None   # mode_a | mode_b | mode_b_plus
    skip_critique: Optional[bool] = None   # None = use settings default


class IngestRequest(BaseModel):
    domain:  str   # space | defence | quantum
    content: str
    source:  str = "manual"


class SyncRequest(BaseModel):
    force: bool = False


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/research")
async def research(req: ResearchRequest):
    """Main research endpoint. Runs full pipeline and returns result."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    try:
        result = await orchestrator.research(
            query         = req.query,
            session_id    = req.session_id,
            force_mode    = req.force_mode,
            skip_critique = req.skip_critique,
        )
        return result
    except Exception as e:
        console.print_exception()
        raise HTTPException(500, str(e))


@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a document into a domain KB."""
    if req.domain not in ["space", "defence", "quantum"]:
        raise HTTPException(400, "domain must be space | defence | quantum")

    async def _ingest():
        n = await ingest_document(req.domain, req.content, req.source)
        console.print(f"[green]Ingested {n} chunks → {req.domain}[/green]")

    background_tasks.add_task(_ingest)
    return {"status": "ingestion_queued", "domain": req.domain}


@app.post("/sync")
async def trigger_sync(req: SyncRequest):
    """Manually trigger knowledge sync across domains."""
    result = await sync_bus.run_sync(force=req.force)
    return result


@app.get("/sync/history")
async def sync_history():
    return await sync_bus.get_sync_history()


@app.get("/sync/conflicts")
async def get_conflicts():
    return await sync_bus.get_conflicts()


@app.get("/sessions")
async def list_sessions():
    return await memory.list_sessions()


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    return await memory.get_history(session_id)


@app.get("/models")
async def list_models():
    """List all configured models."""
    from config.settings import DOMAIN_MODELS
    return {
        "commander":  settings.commander_model,
        "specialists": DOMAIN_MODELS,
        "synthesis":  settings.synthesis_model,
        "critique":   settings.critique_model,
        "embedding":  settings.embed_model,
    }


# ── WebSocket streaming endpoint ───────────────────────────────────────────

@app.websocket("/ws/research")
async def ws_research(websocket: WebSocket):
    """
    WebSocket endpoint for streaming research.
    Client sends: {"query": "...", "session_id": "...", "mode": "..."}
    Server streams: {"type": "token"|"progress"|"done"|"error", "data": "..."}
    """
    await websocket.accept()
    console.print("[cyan]WebSocket connected[/cyan]")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            query         = msg.get("query", "").strip()
            session_id    = msg.get("session_id") or str(uuid.uuid4())
            mode          = msg.get("mode")
            skip_critique = msg.get("skip_critique")  # optional per-message override

            if not query:
                await websocket.send_json({"type": "error", "data": "empty query"})
                continue

            async def send_progress(text: str):
                await websocket.send_json({"type": "progress", "data": text})

            try:
                if mode == "mode_a" or mode is None:
                    # Stream Mode A token by token
                    await websocket.send_json({"type": "progress", "data": "Routing query..."})
                    async for token in orchestrator.stream_mode_a(query, session_id):
                        await websocket.send_json({"type": "token", "data": token})
                    await websocket.send_json({"type": "done", "data": ""})
                else:
                    # Non-streaming for Mode B / B+
                    await websocket.send_json({"type": "progress", "data": "Starting research pipeline..."})
                    result = await orchestrator.research(
                        query         = query,
                        session_id    = session_id,
                        force_mode    = mode,
                        skip_critique = skip_critique,
                        progress_callback = send_progress,
                    )
                    await websocket.send_json({"type": "result", "data": result})
                    await websocket.send_json({"type": "done", "data": ""})

            except Exception as e:
                console.print_exception()
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        console.print("[dim]WebSocket disconnected[/dim]")


# ── Serve UI ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("ui/index.html", "r") as f:
        return HTMLResponse(content=f.read())
