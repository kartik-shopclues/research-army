"""
agents/specialist.py — Domain specialist agent with RAG
One instance per domain: space, defence, quantum
"""
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from rich.console import Console

from agents.llm import make_specialist
from rag.pipeline import retrieve, format_context
from config.settings import settings

console = Console()


class SpecialistAgent:
    def __init__(self, domain: str):
        self.domain = domain
        self.llm    = make_specialist(domain)

    async def _retrieve_context(self, query: str) -> Tuple[List[Dict], str]:
        chunks = await retrieve(self.domain, query, top_k=settings.rag_top_k)
        context_str = format_context(chunks)
        return chunks, context_str

    async def respond(
        self,
        query: str,
        sub_task: Optional[str] = None,
        peer_context: Optional[str] = None,
        debate_role: Optional[Dict] = None,
        round_num: int = 1,
    ) -> Dict:
        """
        Generate a domain response with RAG context.

        peer_context: combined responses from other agents (debate mode)
        debate_role:  assigned role + instruction from commander
        """
        chunks, rag_context = await self._retrieve_context(sub_task or query)

        # Build prompt
        task_text = sub_task if sub_task else query

        sections = [
            f"## Research Task\n{task_text}",
            f"## Retrieved Knowledge (your domain KB)\n{rag_context}",
        ]

        if peer_context and round_num > 1:
            sections.append(
                f"## Other Specialists' Responses (Round {round_num - 1})\n{peer_context}\n\n"
                f"Read the above carefully. Then in your response:\n"
                f"1. Acknowledge what you agree with from peer responses\n"
                f"2. Challenge anything that conflicts with your domain knowledge\n"
                f"3. Add cross-domain connections you can uniquely contribute"
            )

        if debate_role:
            sections.append(
                f"## Your Debate Role: {debate_role.get('role', '').upper()}\n"
                f"{debate_role.get('instruction', '')}"
            )

        prompt = "\n\n".join(sections)
        prompt += "\n\nProvide a detailed, evidence-based response. Reference your sources."

        console.print(
            f"[blue]  {self.domain.upper()} LLM[/blue] generating "
            f"(round {round_num}, {len(chunks)} chunks retrieved)..."
        )

        response = await self.llm.generate(prompt, temperature=0.7)
        # max_tokens comes from OllamaLLM.max_tokens (set via make_specialist factory)

        return {
            "domain":    self.domain,
            "round":     round_num,
            "response":  response,
            "chunks":    chunks,
            "sub_task":  task_text,
        }

    async def stream_respond(
        self,
        query: str,
        sub_task: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming version for Mode A — direct single domain."""
        chunks, rag_context = await self._retrieve_context(sub_task or query)

        task_text = sub_task if sub_task else query
        prompt = (
            f"## Research Task\n{task_text}\n\n"
            f"## Retrieved Knowledge\n{rag_context}\n\n"
            "Provide a detailed, evidence-based response. Reference your sources."
        )

        async for token in self.llm.stream(prompt):
            yield token

