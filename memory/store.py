"""
memory/store.py — Shared memory: session history + episodic cache
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

import redis.asyncio as aioredis

from config.settings import settings

SESSION_TTL = 86400 * 7  # 7 days


class MemoryStore:
    def __init__(self):
        self.redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    # ── Session history ────────────────────────────────────────────────────

    async def add_message(self, session_id: str, role: str, content: str):
        key = f"session:{session_id}:messages"
        msg = json.dumps({
            "role":      role,
            "content":   content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        await self.redis.rpush(key, msg)
        await self.redis.expire(key, SESSION_TTL)

    async def get_history(self, session_id: str, last_n: int = 20) -> List[Dict]:
        key = f"session:{session_id}:messages"
        items = await self.redis.lrange(key, -last_n, -1)
        return [json.loads(i) for i in items]

    async def get_llm_history(self, session_id: str, last_n: int = 10) -> List[Dict]:
        """Return history in Ollama message format."""
        history = await self.get_history(session_id, last_n)
        return [{"role": h["role"], "content": h["content"]} for h in history]

    # ── Research results cache ─────────────────────────────────────────────

    async def cache_result(self, query: str, result: Dict, ttl: int = 3600):
        """Cache a research result for 1 hour to avoid redundant LLM calls."""
        import hashlib
        key = "cache:" + hashlib.md5(query.encode()).hexdigest()
        await self.redis.setex(key, ttl, json.dumps(result))

    async def get_cached_result(self, query: str) -> Optional[Dict]:
        import hashlib
        key = "cache:" + hashlib.md5(query.encode()).hexdigest()
        val = await self.redis.get(key)
        return json.loads(val) if val else None

    # ── Session metadata ───────────────────────────────────────────────────

    async def set_session_meta(self, session_id: str, meta: Dict):
        key = f"session:{session_id}:meta"
        await self.redis.setex(key, SESSION_TTL, json.dumps(meta))

    async def get_session_meta(self, session_id: str) -> Dict:
        key = f"session:{session_id}:meta"
        val = await self.redis.get(key)
        return json.loads(val) if val else {}

    async def list_sessions(self, limit: int = 20) -> List[str]:
        keys = await self.redis.keys("session:*:meta")
        return [k.split(":")[1] for k in keys[:limit]]

    async def close(self):
        await self.redis.aclose()
