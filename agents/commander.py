"""
agents/commander.py — Commander LLM: orchestrator, router, mode selector
"""
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum

from rich.console import Console

from agents.llm import make_commander
from config.settings import DOMAIN_MODELS

console = Console()


class QueryMode(str, Enum):
    MODE_A  = "mode_a"   # single targeted LLM
    MODE_B  = "mode_b"   # broadcast to all
    MODE_B_PLUS = "mode_b_plus"  # full debate engine


class CommanderAgent:
    def __init__(self):
        self.llm = make_commander()

    async def analyze_query(self, query: str) -> Dict:
        """
        Decompose the query and decide:
        - Which mode to use
        - Which domains are relevant
        - Sub-tasks per domain
        """
        prompt = f"""Analyze this research query and output a routing plan.

Query: "{query}"

Available domains: space, defence, quantum

Output a JSON object with:
{{
  "mode": "mode_a" | "mode_b" | "mode_b_plus",
  "primary_domain": "space" | "defence" | "quantum" | null,
  "domains": ["space", "defence", "quantum"],  // all relevant domains
  "sub_tasks": {{
    "space": "specific sub-question for space LLM or null",
    "defence": "specific sub-question for defence LLM or null",
    "quantum": "specific sub-question for quantum LLM or null"
  }},
  "debate_topic": "central thesis for debate if mode_b_plus, else null",
  "reasoning": "one sentence explaining mode choice"
}}

Rules:
- mode_a: query clearly belongs to ONE domain only
- mode_b: query touches multiple domains, parallel answers needed
- mode_b_plus: query needs cross-domain synthesis and debate — use for complex, multi-domain research questions
"""
        result = await self.llm.generate_json(prompt)
        console.print(f"[magenta]Commander routing:[/magenta] {result.get('mode')} → {result.get('domains')}")
        return result

    async def check_convergence(
        self,
        round_outputs: List[Dict],
        debate_topic: str,
        round_num: int,
    ) -> Dict:
        """
        After each debate round, assess if agents have converged.
        Returns {converged: bool, score: float, summary: str}
        """
        outputs_text = "\n\n".join([
            f"[{o['domain'].upper()} LLM - Round {round_num}]:\n{o['response']}"
            for o in round_outputs
        ])

        prompt = f"""You are moderating a research debate on: "{debate_topic}"

Here are the agent responses from round {round_num}:

{outputs_text}

Assess convergence. Output JSON:
{{
  "converged": true | false,
  "score": 0.0-1.0,  // 1.0 = full agreement
  "key_agreements": ["list of points all agents agree on"],
  "key_conflicts": ["list of unresolved disagreements"],
  "next_round_focus": "what should agents address in the next round if not converged"
}}

Converged = score >= 0.75 OR all key scientific facts are agreed upon."""

        result = await self.llm.generate_json(prompt)
        console.print(
            f"[magenta]Convergence check round {round_num}:[/magenta] "
            f"score={result.get('score', 0):.2f} converged={result.get('converged')}"
        )
        return result

    async def assign_debate_roles(self, query: str, domains: List[str]) -> Dict:
        """Assign adversarial roles to each agent for richer debate."""
        prompt = f"""For this research debate: "{query}"

Assign a debate role to each domain specialist to maximize intellectual diversity:
- One should be the "proponent" (argue for the most optimistic interpretation)
- One should be the "skeptic" (challenge assumptions, ask hard questions)
- One should be the "synthesizer" (bridge gaps, find common ground)

Domains: {domains}

Output JSON:
{{
  "space":    {{"role": "proponent|skeptic|synthesizer", "instruction": "specific debate angle"}},
  "defence":  {{"role": "...", "instruction": "..."}},
  "quantum":  {{"role": "...", "instruction": "..."}}
}}"""

        return await self.llm.generate_json(prompt)
