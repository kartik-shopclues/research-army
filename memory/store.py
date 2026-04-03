"""
memory/store.py — Shared memory: session history + episodic cache
Pure in-memory implementation — no Redis required.
"""
import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

SESSION_TTL = 86400 * 7  # 7 days (not enforced in-memory, kept for API compat)
_CACHE_MAX  = 512        # max cached results


class MemoryStore:
    def __init__(self):
        # session_id → deque of message dicts
        self._sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        # session_id → metadata dict
        self._session_meta: Dict[str, Dict] = {}
        # md5(query) → result dict
        self._cache: Dict[str, Dict] = {}
        # insertion-order list for LRU eviction
        self._cache_keys: deque = deque(maxlen=_CACHE_MAX)

    # ── Session history ────────────────────────────────────────────────────

    async def add_message(self, session_id: str, role: str, content: str):
        msg = {
            "role":      role,
            "content":   content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._sessions[session_id].append(msg)

    async def get_history(self, session_id: str, last_n: int = 20) -> List[Dict]:
        msgs = list(self._sessions.get(session_id, []))
        return msgs[-last_n:]

    async def get_llm_history(self, session_id: str, last_n: int = 10) -> List[Dict]:
        """Return history in Ollama message format."""
        history = await self.get_history(session_id, last_n)
        return [{"role": h["role"], "content": h["content"]} for h in history]

    # ── Research results cache ─────────────────────────────────────────────

    async def cache_result(self, query: str, result: Dict, ttl: int = 3600):
        """Cache a research result (LRU, max 512 entries)."""
        key = hashlib.md5(query.encode()).hexdigest()
        if key not in self._cache:
            self._cache_keys.append(key)
        self._cache[key] = result

    async def get_cached_result(self, query: str) -> Optional[Dict]:
        key = hashlib.md5(query.encode()).hexdigest()
        return self._cache.get(key)

    # ── Session metadata ───────────────────────────────────────────────────

    async def set_session_meta(self, session_id: str, meta: Dict):
        self._session_meta[session_id] = meta

    async def get_session_meta(self, session_id: str) -> Dict:
        return self._session_meta.get(session_id, {})

    async def list_sessions(self, limit: int = 20) -> List[str]:
        return list(self._sessions.keys())[:limit]

    async def close(self):
        pass  # nothing to close
