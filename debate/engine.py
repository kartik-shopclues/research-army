"""
debate/engine.py — Multi-round debate engine (Mode B+) and BroadcastEngine (Mode B)

Inference optimisations vs original:
  - Specialists run SEQUENTIALLY — Ollama serialises GPU calls anyway; gather adds overhead
  - QUANTUM FIRST — qwen3:8b (quantum) shares model with the commander → zero VRAM swap
  - Critique is OPTIONAL — skip via skip_critique flag for ~20-40s saving
  - peer_context TRUNCATED before passing to specialists and synthesis
  - Token budgets taken from settings (not hardcoded)
"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from rich.console import Console

from agents.commander import CommanderAgent
from agents.specialist import SpecialistAgent
from agents.llm import make_synthesis, make_critique
from config.settings import settings, SYNTHESIS_SYSTEM_PROMPT

console = Console()

# Quantum ( qwen3:8b ) shares the same model as the commander — put it first
# so Ollama makes NO VRAM swap between commander routing and quantum specialist.
_PREFERRED_DOMAIN_ORDER = ["quantum", "space", "defence"]


def _order_domains(domains: List[str]) -> List[str]:
    """Return domains in VRAM-efficient order: quantum first, then others."""
    ordered   = [d for d in _PREFERRED_DOMAIN_ORDER if d in domains]
    remaining = [d for d in domains if d not in _PREFERRED_DOMAIN_ORDER]
    return ordered + remaining


class DebateEngine:
    def __init__(self):
        self.commander = CommanderAgent()
        self.specialists = {
            "space":   SpecialistAgent("space"),
            "defence": SpecialistAgent("defence"),
            "quantum": SpecialistAgent("quantum"),
        }
        self.synthesis_llm = make_synthesis()
        self.critique_llm  = make_critique()

    async def run(
        self,
        query:            str,
        domains:          List[str],
        sub_tasks:        Dict[str, Optional[str]],
        debate_topic:     str,
        max_rounds:       int  = None,
        skip_critique:    bool = None,
        progress_callback       = None,
    ) -> Dict:
        """
        Full debate pipeline:
        1. Assign roles
        2. Round 1: independent drafts (sequential, quantum first)
        3. Rounds 2+: read peers, challenge + refine
        4. Convergence check after each round
        5. Synthesis + optional critique
        """
        max_rounds    = max_rounds    if max_rounds    is not None else settings.max_debate_rounds
        skip_critique = skip_critique if skip_critique is not None else settings.skip_critique
        start_time    = datetime.utcnow()
        transcript    = []

        # Reorder: quantum first → zero VRAM swap after commander (both use qwen3:8b)
        ordered_domains = _order_domains(domains)

        if progress_callback:
            await progress_callback("Assigning debate roles...")

        roles = await self.commander.assign_debate_roles(query, ordered_domains)
        console.print(f"[magenta]Debate roles:[/magenta] {roles}")

        all_round_outputs: List[List[Dict]] = []
        peer_context: Optional[str] = None

        for round_num in range(1, max_rounds + 1):
            console.rule(f"[bold magenta]Debate Round {round_num}[/bold magenta]")

            if progress_callback:
                await progress_callback(
                    f"Running debate round {round_num}/{max_rounds} "
                    f"({' → '.join(ordered_domains)})..."
                )

            # ── Sequential specialist calls (quantum → space → defence) ────────
            # asyncio.gather is NOT used — Ollama queues GPU requests anyway.
            # Sequential await: simpler, equal speed, zero connection-pool thrash.
            round_outputs: List[Dict] = []
            for domain in ordered_domains:
                specialist = self.specialists[domain]
                role       = roles.get(domain)

                # Truncate peer_context to keep prompts lean
                trimmed_peer = (
                    peer_context[:settings.max_peer_context_chars]
                    if peer_context else None
                )

                output = await specialist.respond(
                    query      = query,
                    sub_task   = sub_tasks.get(domain),
                    peer_context = trimmed_peer,
                    debate_role  = role,
                    round_num    = round_num,
                )
                round_outputs.append(output)

            all_round_outputs.append(round_outputs)

            for output in round_outputs:
                transcript.append({
                    "round":    round_num,
                    "domain":   output["domain"],
                    "role":     roles.get(output["domain"], {}).get("role", ""),
                    "response": output["response"],
                })

            # Build peer context for next round
            peer_context = "\n\n".join([
                f"--- {o['domain'].upper()} SPECIALIST (Round {round_num}) ---\n{o['response']}"
                for o in round_outputs
            ])

            # ── Convergence check ──────────────────────────────────────────────
            if round_num < max_rounds:
                if progress_callback:
                    await progress_callback("Checking convergence...")

                convergence = await self.commander.check_convergence(
                    round_outputs, debate_topic, round_num
                )
                transcript.append({
                    "round":       round_num,
                    "domain":      "MODERATOR",
                    "response":    (
                        f"Convergence score: {convergence.get('score', 0):.2f}\n"
                        f"Agreements: {convergence.get('key_agreements', [])}\n"
                        f"Conflicts:  {convergence.get('key_conflicts', [])}"
                    ),
                    "convergence": convergence,
                })

                if convergence.get("converged"):
                    console.print("[green]Converged! Moving to synthesis.[/green]")
                    break
                else:
                    console.print(
                        f"[yellow]Not converged (score={convergence.get('score', 0):.2f}). "
                        f"Next focus: {convergence.get('next_round_focus', '')}[/yellow]"
                    )

        # ── Synthesis ──────────────────────────────────────────────────────────
        if progress_callback:
            await progress_callback("Synthesising cross-domain insights...")

        console.rule("[bold teal]Synthesis[/bold teal]")

        # Truncate peer_context for synthesis — keep prompt tight
        synthesis_context = (peer_context or "")[:4000]
        synthesis_prompt  = (
            f'Original research question: "{query}"\n\n'
            f"Final specialist positions after debate:\n\n{synthesis_context}\n\n"
            "Synthesise:\n"
            "1. Core findings all specialists agree on\n"
            "2. Cross-domain insights that emerged\n"
            "3. Most surprising connection across space, defence, and quantum\n"
            "4. What this means for the original question\n\n"
            "Produce a coherent, well-structured research response."
        )
        synthesis = await self.synthesis_llm.generate(synthesis_prompt)

        # ── Critique (optional) ────────────────────────────────────────────────
        critique = ""
        if not skip_critique:
            if progress_callback:
                await progress_callback("Running critique and fact-check...")

            console.rule("[bold coral]Critique[/bold coral]")
            critique_prompt = (
                f'Original question: "{query}"\n\n'
                f"Synthesised response:\n{synthesis}\n\n"
                f"Specialist responses (excerpt):\n{synthesis_context[:2000]}\n\n"
                "Critique: fact-check claims, flag conflicts, "
                "rate confidence HIGH/MEDIUM/LOW per claim."
            )
            critique = await self.critique_llm.generate(critique_prompt)
        else:
            console.print("[dim]Critique skipped (skip_critique=True)[/dim]")

        elapsed = (datetime.utcnow() - start_time).seconds

        return {
            "query":      query,
            "mode":       "mode_b_plus",
            "domains":    ordered_domains,
            "rounds":     len(all_round_outputs),
            "transcript": transcript,
            "synthesis":  synthesis,
            "critique":   critique,
            "elapsed_s":  elapsed,
        }


class BroadcastEngine:
    """Mode B — sequential broadcast with quantum-first ordering, then synthesis."""

    def __init__(self):
        self.specialists = {
            "space":   SpecialistAgent("space"),
            "defence": SpecialistAgent("defence"),
            "quantum": SpecialistAgent("quantum"),
        }
        self.synthesis_llm = make_synthesis()

    async def run(
        self,
        query:            str,
        domains:          List[str],
        sub_tasks:        Dict[str, Optional[str]],
        progress_callback       = None,
    ) -> Dict:
        # Quantum first → zero VRAM swap after commander
        ordered_domains = _order_domains(domains)

        if progress_callback:
            await progress_callback(
                f"Querying specialists: {' → '.join(ordered_domains)}..."
            )

        # Sequential: Ollama serialises GPU calls anyway
        outputs: List[Dict] = []
        for domain in ordered_domains:
            output = await self.specialists[domain].respond(
                query, sub_tasks.get(domain), round_num=1
            )
            outputs.append(output)

        # Truncate combined context before synthesis
        combined = "\n\n".join([
            f"--- {o['domain'].upper()} SPECIALIST ---\n{o['response']}"
            for o in outputs
        ])[:4000]

        if progress_callback:
            await progress_callback("Synthesising responses...")

        synthesis_prompt = (
            f'Research question: "{query}"\n\n'
            f"Specialist responses:\n{combined}\n\n"
            "Synthesise into a comprehensive response with cross-domain insights."
        )
        synthesis = await self.synthesis_llm.generate(synthesis_prompt)

        return {
            "query":     query,
            "mode":      "mode_b",
            "domains":   ordered_domains,
            "outputs":   [{"domain": o["domain"], "response": o["response"]} for o in outputs],
            "synthesis": synthesis,
        }
