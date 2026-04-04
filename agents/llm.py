"""
agents/llm.py — Async Ollama LLM wrapper with streaming support

Inference optimisations vs original:
  - keep_alive injected into every request → model never unloads from VRAM
  - num_ctx capped via settings → faster KV-cache allocation + attention
  - Per-role max_tokens (commander=256, specialist=800, synthesis=1500, critique=600)
  - Exponential-backoff retry for Ollama timeouts under GPU load
"""
import asyncio
import json
from typing import AsyncGenerator, Optional, List, Dict

import httpx
from rich.console import Console

from config.settings import settings

console = Console()

_RETRY_ATTEMPTS  = 3
_RETRY_BASE_WAIT = 2   # seconds


class OllamaLLM:
    """
    Async wrapper around Ollama /api/chat.
    Injects keep_alive + num_ctx into every request for single-GPU efficiency.
    """

    def __init__(
        self,
        model:         str,
        system_prompt: str = "",
        timeout:       int = 300,
        max_tokens:    int = 800,
    ):
        self.model         = model
        self.system_prompt = system_prompt
        self.base_url      = settings.ollama_base_url
        self.timeout       = timeout
        self.max_tokens    = max_tokens   # per-role budget, used as default

    def _build_messages(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _options(self, temperature: float, max_tokens: int) -> Dict:
        return {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     settings.ollama_num_ctx,   # cap context window
        }

    async def generate(
        self,
        prompt:      str,
        history:     Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens:  int   = None,
    ) -> str:
        """Non-streaming generation with exponential-backoff retry."""
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        messages   = self._build_messages(prompt, history)
        payload    = {
            "model":      self.model,
            "messages":   messages,
            "stream":     False,
            "keep_alive": settings.ollama_keep_alive,
            "options":    self._options(temperature, max_tokens),
        }

        for attempt in range(_RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/api/chat", json=payload
                    )
                    resp.raise_for_status()
                    return resp.json()["message"]["content"]
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                wait = _RETRY_BASE_WAIT ** attempt
                console.print(
                    f"[yellow]Ollama timeout ({self.model}), "
                    f"retry {attempt + 1}/{_RETRY_ATTEMPTS - 1} in {wait}s…[/yellow]"
                )
                await asyncio.sleep(wait)
            except Exception:
                raise

    async def stream(
        self,
        prompt:      str,
        history:     Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens:  int   = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation. Yields text chunks as they arrive."""
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        messages   = self._build_messages(prompt, history)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model":      self.model,
                    "messages":   messages,
                    "stream":     True,
                    "keep_alive": settings.ollama_keep_alive,
                    "options":    self._options(temperature, max_tokens),
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if not chunk.get("done"):
                                yield chunk["message"]["content"]
                        except json.JSONDecodeError:
                            continue

    async def generate_json(
        self,
        prompt:  str,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate and parse a JSON response."""
        json_prompt = prompt + "\n\nRespond ONLY with valid JSON. No markdown, no preamble."
        response    = await self.generate(json_prompt, history, temperature=0.1)

        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("`")

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            console.print(f"[red]JSON parse error:[/red] {e}")
            console.print(f"[dim]Raw response:[/dim] {response[:200]}")
            return {"error": str(e), "raw": response}


# ── Factory functions — each sets the correct per-role token budget ───────────

def make_commander() -> OllamaLLM:
    from config.settings import COMMANDER_SYSTEM_PROMPT
    return OllamaLLM(
        settings.commander_model,
        COMMANDER_SYSTEM_PROMPT,
        timeout    = 300,
        max_tokens = settings.max_tokens_commander,   # 256 — just routing JSON
    )


def make_specialist(domain: str) -> OllamaLLM:
    from config.settings import DOMAIN_MODELS, DOMAIN_SYSTEM_PROMPTS
    return OllamaLLM(
        DOMAIN_MODELS[domain],
        DOMAIN_SYSTEM_PROMPTS[domain],
        timeout    = 300,
        max_tokens = settings.max_tokens_specialist,  # 800 — focused domain answer
    )


def make_synthesis() -> OllamaLLM:
    from config.settings import SYNTHESIS_SYSTEM_PROMPT
    return OllamaLLM(
        settings.synthesis_model,
        SYNTHESIS_SYSTEM_PROMPT,
        timeout    = 600,
        max_tokens = settings.max_tokens_synthesis,   # 1500 — cross-domain narrative
    )


def make_critique() -> OllamaLLM:
    from config.settings import CRITIQUE_SYSTEM_PROMPT
    return OllamaLLM(
        settings.critique_model,
        CRITIQUE_SYSTEM_PROMPT,
        timeout    = 300,
        max_tokens = settings.max_tokens_critique,    # 600 — fact-check bullets
    )
