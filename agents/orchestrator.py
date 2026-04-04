"""
agents/orchestrator.py — Main research pipeline
Ties Commander → Specialist(s) → Debate/Broadcast → Synthesis → Output
"""
import asyncio
import uuid
from typing import Dict, Optional, AsyncGenerator

from rich.console import Console

from agents.commander import CommanderAgent, QueryMode
from agents.specialist import SpecialistAgent
from debate.engine import DebateEngine, BroadcastEngine
from memory.store import MemoryStore
from config.settings import settings

console = Console()


class ResearchOrchestrator:
    def __init__(self):
        self.commander   = CommanderAgent()
        self.specialists = {
            "space":   SpecialistAgent("space"),
            "defence": SpecialistAgent("defence"),
            "quantum": SpecialistAgent("quantum"),
        }
        self.debate_engine    = DebateEngine()
        self.broadcast_engine = BroadcastEngine()
        self.memory           = MemoryStore()

    async def research(
        self,
        query:         str,
        session_id:    str  = None,
        force_mode:    Optional[str]  = None,
        skip_critique: Optional[bool] = None,   # None = use settings default
        progress_callback = None,
    ) -> Dict:
        """
        Main entry point. Run the full pipeline for a query.

        force_mode: override commander's mode choice
            "mode_a" | "mode_b" | "mode_b_plus"
        """
        session_id = session_id or str(uuid.uuid4())

        # ── Save user message to memory ────────────────────────────────────
        await self.memory.add_message(session_id, "user", query)

        # ── Check cache ────────────────────────────────────────────────────
        cached = await self.memory.get_cached_result(query)
        if cached:
            console.print("[dim]Cache hit[/dim]")
            return cached

        # ── Commander analyzes query ───────────────────────────────────────
        if progress_callback:
            await progress_callback("Commander analyzing query...")

        plan = await self.commander.analyze_query(query)

        mode      = force_mode or plan.get("mode", QueryMode.MODE_B)
        domains   = plan.get("domains", ["space", "defence", "quantum"])
        sub_tasks = plan.get("sub_tasks", {})

        console.rule(f"[bold]Mode: {mode.upper()} | Domains: {domains}[/bold]")

        # ── Route to correct engine ────────────────────────────────────────
        if mode == QueryMode.MODE_A:
            result = await self._run_mode_a(query, plan, progress_callback)

        elif mode == QueryMode.MODE_B:
            result = await self.broadcast_engine.run(
                query, domains, sub_tasks, progress_callback
            )

        elif mode == QueryMode.MODE_B_PLUS:
            result = await self.debate_engine.run(
                query         = query,
                domains       = domains,
                sub_tasks     = sub_tasks,
                debate_topic  = plan.get("debate_topic", query),
                max_rounds    = settings.max_debate_rounds,
                skip_critique = skip_critique,
                progress_callback = progress_callback,
            )
        else:
            result = await self.broadcast_engine.run(
                query, domains, sub_tasks, progress_callback
            )

        result["session_id"] = session_id
        result["plan"]       = plan

        # ── Save result to memory + cache ──────────────────────────────────
        await self.memory.add_message(
            session_id, "assistant", result.get("synthesis", "")
        )
        await self.memory.cache_result(query, result, ttl=3600)

        return result

    async def _run_mode_a(self, query: str, plan: Dict, progress_callback) -> Dict:
        """Mode A: single targeted specialist."""
        domain = plan.get("primary_domain") or plan.get("domains", ["space"])[0]
        sub_task = plan.get("sub_tasks", {}).get(domain)

        if progress_callback:
            await progress_callback(f"Querying {domain} specialist...")

        specialist = self.specialists[domain]
        output = await specialist.respond(query, sub_task)

        return {
            "query":     query,
            "mode":      "mode_a",
            "domain":    domain,
            "synthesis": output["response"],
            "chunks":    output["chunks"],
        }

    async def stream_mode_a(
        self,
        query: str,
        session_id: str = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming Mode A — commander routes, then streams specialist output.
        Used by the WebSocket endpoint for real-time token delivery.
        """
        session_id = session_id or str(uuid.uuid4())
        plan   = await self.commander.analyze_query(query)
        domain = plan.get("primary_domain") or "space"

        specialist = self.specialists[domain]
        sub_task   = plan.get("sub_tasks", {}).get(domain)

        yield f"[Routing to {domain.upper()} specialist]\n\n"

        async for token in specialist.stream_respond(query, sub_task):
            yield token

    async def close(self):
        await self.memory.close()
