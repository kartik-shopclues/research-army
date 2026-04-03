"""
agents/llm.py — Async Ollama LLM wrapper with streaming support
Handles model loading/unloading awareness for 24GB VRAM constraint
"""
import asyncio
import json
from typing import AsyncGenerator, Optional, List, Dict

import httpx
from rich.console import Console

from config.settings import settings

console = Console()


class OllamaLLM:
    """
    Thin async wrapper around Ollama /api/chat endpoint.
    Supports streaming and non-streaming modes.
    """

    def __init__(self, model: str, system_prompt: str = "", timeout: int = 300):
        self.model         = model
        self.system_prompt = system_prompt
        self.base_url      = settings.ollama_base_url
        self.timeout       = timeout

    async def _ensure_model_loaded(self):
        """Ping Ollama to pre-load model into VRAM."""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": "", "stream": False},
                )
            except Exception:
                pass  # Model may already be loaded

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

    async def generate(
        self,
        prompt: str,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming generation. Returns full response string."""
        messages = self._build_messages(prompt, history)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def stream(
        self,
        prompt: str,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation. Yields text chunks as they arrive."""
        messages = self._build_messages(prompt, history)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
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
        prompt: str,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate and parse JSON response.
        Appends instruction to respond only in JSON.
        """
        json_prompt = prompt + "\n\nRespond ONLY with valid JSON. No markdown, no preamble."
        response = await self.generate(json_prompt, history, temperature=0.1)

        # Strip markdown fences if present
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


def make_commander() -> OllamaLLM:
    from config.settings import COMMANDER_SYSTEM_PROMPT
    return OllamaLLM(settings.commander_model, COMMANDER_SYSTEM_PROMPT, timeout=600)


def make_specialist(domain: str) -> OllamaLLM:
    from config.settings import DOMAIN_MODELS, DOMAIN_SYSTEM_PROMPTS
    return OllamaLLM(DOMAIN_MODELS[domain], DOMAIN_SYSTEM_PROMPTS[domain], timeout=300)


def make_synthesis() -> OllamaLLM:
    from config.settings import SYNTHESIS_SYSTEM_PROMPT
    return OllamaLLM(settings.synthesis_model, SYNTHESIS_SYSTEM_PROMPT, timeout=600)


def make_critique() -> OllamaLLM:
    from config.settings import CRITIQUE_SYSTEM_PROMPT
    return OllamaLLM(settings.critique_model, CRITIQUE_SYSTEM_PROMPT, timeout=300)
