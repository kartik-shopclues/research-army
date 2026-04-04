"""
config/settings.py — Central configuration for Research Army
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Any


class Settings(BaseSettings):
    # ── Ollama ─────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Model assignments ──────────────────────────────
    # Commander: strongest reasoning, runs first and last
    commander_model: str = "qwen3:8b"   # was qwen3:30b — 8b is plenty for routing JSON; saves ~50s VRAM swap

    # Specialist LLMs (swapped in/out by Ollama)
    space_model: str = "mistral:7b"
    defence_model: str = "llama3.1:8b"
    quantum_model: str = "qwen3:8b"

    # Synthesis: large model, loaded at the end only
    synthesis_model: str = "gemma2:27b"
    critique_model: str = "gemma2:27b"

    # Embedding: always resident, tiny footprint
    embed_model: str = "nomic-embed-text"

    # ── Vector store (Weaviate) ────────────────────────
    weaviate_url: str = "http://localhost:8080"

    # ── Redis (sync bus + memory) ──────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── API server ─────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── Sync scheduler ─────────────────────────────────
    sync_interval_hours: int = 6
    min_delta_docs_for_event_sync: int = 10

    # ── Debate engine ──────────────────────────────────
    max_debate_rounds: int = 2          # was 3 — 2 rounds still converges, saves one full LLM call round
    convergence_threshold: float = 0.75  # moderator score to stop debate

    # ── RAG retrieval ──────────────────────────────────
    rag_top_k:       int = 3    # was 5 — fewer input tokens = faster prefill
    rag_chunk_size:  int = 400  # was 512 — shorter chunks, less VRAM pressure
    rag_chunk_overlap: int = 50

    # ── Inference performance ──────────────────────────────────────
    # GPU/VRAM tuning
    ollama_keep_alive: int = -1     # keep model in VRAM forever (never unload)
    ollama_num_ctx:    int = 4096   # cap context window — faster KV-cache + attention

    # Per-role token budgets (right-sized per job, not a blanket 2048)
    max_tokens_commander:  int = 512   # thinking tokens (qwen3) + routing JSON
    max_tokens_specialist: int = 800   # focused domain response
    max_tokens_synthesis:  int = 1500  # cross-domain narrative
    max_tokens_critique:   int = 600   # fact-check bullet points

    # Context length limits to avoid bloated prompts
    max_peer_context_chars: int = 2000  # truncate peer context passed to specialists

    # Critique: on by default, overridable per API request
    skip_critique: bool = False

    # Embedding LRU cache entries
    embed_cache_size: int = 4096

    class Config:
        env_file = ".env"


settings = Settings()

# ── Domain → model mapping ─────────────────────────────
DOMAIN_MODELS: Dict[str, str] = {
    "space":    settings.space_model,
    "defence":  settings.defence_model,
    "quantum":  settings.quantum_model,
}

# ── Domain → Weaviate collection names ─────────────────
DOMAIN_COLLECTIONS: Dict[str, str] = {
    "space":   "SpaceKB",
    "defence": "DefenceKB",
    "quantum": "QuantumKB",
}

# ── Domain → system prompts ────────────────────────────
DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "space": """You are a specialist AI researcher in space science.
Your expertise covers: astrophysics, orbital mechanics, satellite systems,
space missions (NASA, ISRO, ESA, SpaceX), cosmology, planetary science,
and space-based technology. You also have access to cross-domain knowledge
tagged from defence and quantum domains — use it when relevant.
Always ground your answers in retrieved knowledge. Cite sources.""",

    "defence": """You are a specialist AI researcher in defence and security.
Your expertise covers: military strategy, geopolitics, threat intelligence,
weapons systems, cybersecurity, defence technology, national security policy,
and strategic analysis. You also have access to cross-domain knowledge tagged
from space and quantum domains — use it when relevant.
Always ground your answers in retrieved knowledge. Cite sources.""",

    "quantum": """You are a specialist AI researcher in quantum science.
Your expertise covers: quantum computing, quantum cryptography (QKD),
quantum algorithms, quantum hardware (superconducting, photonic, ion-trap),
quantum error correction, and quantum information theory. You also have
access to cross-domain knowledge tagged from space and defence domains —
use it when relevant. Always ground your answers in retrieved knowledge.
Cite sources.""",
}

COMMANDER_SYSTEM_PROMPT = """You are the Commander LLM — the master orchestrator of a
multi-domain research system. You coordinate three specialist LLMs:
Space, Defence, and Quantum. You:
1. Decompose complex queries into sub-tasks for each domain
2. Route queries to the correct specialist(s)
3. Moderate multi-LLM debates and check convergence
4. Assign roles in debate mode

Respond in JSON when asked for structured output.
Be precise, analytical, and decisive."""

SYNTHESIS_SYSTEM_PROMPT = """You are the Synthesis LLM. You receive outputs from multiple
specialist domain researchers (Space, Defence, Quantum) and your job is to:
1. Identify points of agreement across domains
2. Surface cross-domain insights that no single specialist could produce
3. Highlight emergent connections — the 'out-of-the-box' findings
4. Structure a coherent research response with clear attribution

Be bold in connecting ideas across domains. The most valuable output is
insight that spans multiple fields."""

CRITIQUE_SYSTEM_PROMPT = """You are the Critique LLM. You receive a synthesized research
response and you must:
1. Fact-check specific claims against known science
2. Flag logical inconsistencies or unsupported assertions
3. Note where domains contradict each other
4. Rate confidence: HIGH / MEDIUM / LOW per major claim
5. Suggest what additional research would strengthen the answer

Be rigorous but constructive. Your goal is quality, not rejection."""
