"""
debate/engine.py — Multi-round debate engine (Mode B+)
Agents read each other's outputs and iteratively refine positions
"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.commander import CommanderAgent
from agents.specialist import SpecialistAgent
from agents.llm import make_synthesis, make_critique
from config.settings import settings, SYNTHESIS_SYSTEM_PROMPT

console = Console()


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
        query: str,
        domains: List[str],
        sub_tasks: Dict[str, Optional[str]],
        debate_topic: str,
        max_rounds: int = None,
        progress_callback=None,
    ) -> Dict:
        """
        Full debate pipeline:
        1. Assign roles
        2. Round 1: independent drafts
        3. Rounds 2+: read peers, challenge + refine
        4. Convergence check after each round
        5. Synthesis + critique
        """
        max_rounds = max_rounds or settings.max_debate_rounds
        start_time = datetime.utcnow()
        transcript = []

        if progress_callback:
            await progress_callback("Assigning debate roles...")

        # ── Role assignment ────────────────────────────────────────────────
        roles = await self.commander.assign_debate_roles(query, domains)
        console.print(f"[magenta]Debate roles:[/magenta] {roles}")

        all_round_outputs: List[List[Dict]] = []
        peer_context: Optional[str] = None

        for round_num in range(1, max_rounds + 1):
            console.rule(f"[bold magenta]Debate Round {round_num}[/bold magenta]")

            if progress_callback:
                await progress_callback(f"Running debate round {round_num}/{max_rounds}...")

            # ── Parallel specialist generation ─────────────────────────────
            tasks = []
            for domain in domains:
                specialist = self.specialists[domain]
                role = roles.get(domain)
                task = specialist.respond(
                    query=query,
                    sub_task=sub_tasks.get(domain),
                    peer_context=peer_context,
                    debate_role=role,
                    round_num=round_num,
                )
                tasks.append(task)

            round_outputs = await asyncio.gather(*tasks)
            all_round_outputs.append(list(round_outputs))

            # Record transcript
            for output in round_outputs:
                transcript.append({
                    "round":    round_num,
                    "domain":   output["domain"],
                    "role":     roles.get(output["domain"], {}).get("role", ""),
                    "response": output["response"],
                })

            # ── Build peer context for next round ──────────────────────────
            peer_context = "\n\n".join([
                f"--- {o['domain'].upper()} SPECIALIST (Round {round_num}) ---\n{o['response']}"
                for o in round_outputs
            ])

            # ── Convergence check ──────────────────────────────────────────
            if round_num < max_rounds:
                if progress_callback:
                    await progress_callback("Checking convergence...")

                convergence = await self.commander.check_convergence(
                    list(round_outputs), debate_topic, round_num
                )
                transcript.append({
                    "round":       round_num,
                    "domain":      "MODERATOR",
                    "response":    f"Convergence score: {convergence.get('score', 0):.2f}\n"
                                   f"Agreements: {convergence.get('key_agreements', [])}\n"
                                   f"Conflicts:  {convergence.get('key_conflicts', [])}",
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

        # ── Synthesis ──────────────────────────────────────────────────────
        if progress_callback:
            await progress_callback("Synthesising cross-domain insights...")

        console.rule("[bold teal]Synthesis[/bold teal]")
        final_round = all_round_outputs[-1]

        synthesis_prompt = f"""Original research question: "{query}"

Here are the final positions of all specialist LLMs after debate:

{peer_context}

Now synthesise:
1. What do all specialists agree on? (core findings)
2. What cross-domain insights emerged from the debate?
3. What is the most surprising or non-obvious connection across space, defence, and quantum?
4. What does this mean for the original question?

Produce a coherent, well-structured research response."""

        synthesis = await self.synthesis_llm.generate(synthesis_prompt, max_tokens=3000)

        # ── Critique ───────────────────────────────────────────────────────
        if progress_callback:
            await progress_callback("Running critique and fact-check...")

        console.rule("[bold coral]Critique[/bold coral]")
        critique_prompt = f"""Original question: "{query}"

Synthesised response:
{synthesis}

Specialist responses that contributed:
{peer_context[:3000]}

Provide your critique: fact-check, flag conflicts, rate confidence per claim."""

        critique = await self.critique_llm.generate(critique_prompt, max_tokens=1500)

        elapsed = (datetime.utcnow() - start_time).seconds

        return {
            "query":      query,
            "mode":       "mode_b_plus",
            "domains":    domains,
            "rounds":     len(all_round_outputs),
            "transcript": transcript,
            "synthesis":  synthesis,
            "critique":   critique,
            "elapsed_s":  elapsed,
        }


class BroadcastEngine:
    """Mode B — parallel broadcast, no debate, direct synthesis."""

    def __init__(self):
        self.specialists = {
            "space":   SpecialistAgent("space"),
            "defence": SpecialistAgent("defence"),
            "quantum": SpecialistAgent("quantum"),
        }
        self.synthesis_llm = make_synthesis()

    async def run(
        self,
        query: str,
        domains: List[str],
        sub_tasks: Dict[str, Optional[str]],
        progress_callback=None,
    ) -> Dict:
        if progress_callback:
            await progress_callback("Broadcasting to all specialists...")

        tasks = [
            self.specialists[d].respond(query, sub_tasks.get(d), round_num=1)
            for d in domains
        ]
        outputs = await asyncio.gather(*tasks)

        combined = "\n\n".join([
            f"--- {o['domain'].upper()} SPECIALIST ---\n{o['response']}"
            for o in outputs
        ])

        if progress_callback:
            await progress_callback("Synthesising responses...")

        synthesis_prompt = (
            f'Research question: "{query}"\n\n'
            f"Specialist responses:\n{combined}\n\n"
            "Synthesise into a comprehensive response with cross-domain insights."
        )
        synthesis = await self.synthesis_llm.generate(synthesis_prompt, max_tokens=2500)

        return {
            "query":      query,
            "mode":       "mode_b",
            "domains":    domains,
            "outputs":    [{"domain": o["domain"], "response": o["response"]} for o in outputs],
            "synthesis":  synthesis,
        }
