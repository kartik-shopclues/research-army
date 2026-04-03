"""
sync/bus.py — Knowledge sync bus (in-memory, no Redis)
Extracts deltas from each domain KB and cross-injects into others.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from rich.console import Console

from rag.pipeline import get_delta_since, ingest_document
from config.settings import settings, DOMAIN_COLLECTIONS

console = Console()


class SyncBus:
    def __init__(self):
        self.domains        = list(DOMAIN_COLLECTIONS.keys())  # [space, defence, quantum]
        self._last_sync: Dict[str, str] = {}
        self._sync_history: List[Dict]  = []
        self._conflicts:    List[Dict]  = []

    # ── Sync state ─────────────────────────────────────────────────────────

    async def get_last_sync_time(self, domain: str) -> str:
        if domain not in self._last_sync:
            return (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self._last_sync[domain]

    async def set_last_sync_time(self, domain: str, ts: str):
        self._last_sync[domain] = ts

    async def log_conflict(self, conflict: Dict):
        self._conflicts.insert(0, conflict)
        self._conflicts = self._conflicts[:1000]   # keep last 1000

    async def get_conflicts(self, limit: int = 50) -> List[Dict]:
        return self._conflicts[:limit]

    # ── Conflict resolution ────────────────────────────────────────────────

    def _resolve_conflict(
        self,
        chunk: Dict,
        source_domain: str,
        target_domain: str,
    ) -> Optional[Dict]:
        return {
            "content":       chunk["content"],
            "source":        chunk.get("source", "cross-domain-sync"),
            "source_domain": source_domain,
            "confidence":    chunk.get("confidence", 0.85),
        }

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        seen = set()
        result = []
        for c in chunks:
            cid = c.get("chunk_id", c.get("content", "")[:64])
            if cid not in seen:
                seen.add(cid)
                result.append(c)
        return result

    # ── Core sync ──────────────────────────────────────────────────────────

    async def sync_domain_pair(
        self,
        source_domain: str,
        target_domain: str,
        since_iso: str,
    ) -> int:
        console.print(
            f"[cyan]Syncing:[/cyan] {source_domain} → {target_domain} "
            f"(since {since_iso[:19]})"
        )

        delta_chunks = await get_delta_since(source_domain, since_iso)
        if not delta_chunks:
            console.print(f"  [dim]No delta for {source_domain}[/dim]")
            return 0

        delta_chunks = self._deduplicate(delta_chunks)
        console.print(f"  [green]{len(delta_chunks)} delta chunks found[/green]")

        injected = 0
        for chunk in delta_chunks:
            resolved = self._resolve_conflict(chunk, source_domain, target_domain)
            if resolved is None:
                continue
            n = await ingest_document(
                domain=target_domain,
                content=resolved["content"],
                source=resolved["source"],
                source_domain=source_domain,
                is_cross_domain=True,
                confidence=resolved["confidence"],
            )
            injected += n

        console.print(f"  [green]Injected {injected} cross-domain chunks → {target_domain}[/green]")
        return injected

    async def run_sync(self, force: bool = False) -> Dict:
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        results = {}

        console.rule("[bold cyan]Knowledge Sync Cycle[/bold cyan]")
        console.print(f"Sync started at {now}")

        for source in self.domains:
            since = await self.get_last_sync_time(source)
            results[source] = {}
            for target in self.domains:
                if source == target:
                    continue
                count = await self.sync_domain_pair(source, target, since)
                results[source][target] = count
            await self.set_last_sync_time(source, now)

        total = sum(c for src in results.values() for c in src.values())
        console.print(f"[green]Sync complete. {total} total cross-domain chunks injected.[/green]")

        record = {"timestamp": now, "results": results, "total": total}
        self._sync_history.insert(0, record)
        self._sync_history = self._sync_history[:100]

        return {"timestamp": now, "results": results, "total_injected": total}

    async def get_sync_history(self, limit: int = 10) -> List[Dict]:
        return self._sync_history[:limit]

    async def close(self):
        pass  # nothing to close


# ── Scheduler ─────────────────────────────────────────────────────────────

async def run_sync_scheduler():
    """Background task: sync every N hours."""
    bus      = SyncBus()
    interval = settings.sync_interval_hours * 3600
    console.print(f"[cyan]Sync scheduler started (every {settings.sync_interval_hours}h)[/cyan]")

    while True:
        try:
            await bus.run_sync()
        except Exception as e:
            console.print(f"[red]Sync error:[/red] {e}")
        await asyncio.sleep(interval)
