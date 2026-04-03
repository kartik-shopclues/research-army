# Research Army

Multi-domain LLM research system with Space, Defence, and Quantum specialists,
connected via RAG, cross-domain knowledge sync, and a multi-round debate engine.
Built for AWS g5.xlarge (A10G 24GB GPU), single-user deployment.

---

## Architecture

```
User Query
    │
    ▼
Commander LLM (Qwen3-30B-A3B MoE)
    │   ├── Mode A  → one specialist
    │   ├── Mode B  → all in parallel
    │   └── Mode B+ → full debate rounds
    ▼
Smart Router + Shared Memory (Redis)
    │
    ├── Space LLM    ←→ Space KB    (Weaviate)
    ├── Defence LLM  ←→ Defence KB  (Weaviate)
    └── Quantum LLM  ←→ Quantum KB  (Weaviate)
           │
    Knowledge Sync Bus (cross-domain delta injection)
           │
    Debate Engine (multi-round, shared context window)
           │
    Synthesis LLM + Critique LLM (Gemma2-27B)
           │
    Final Response (brief + citation + transcript)
```

---

## Models (24GB VRAM — Ollama hot-swap)

| Role | Model | VRAM |
|------|-------|------|
| Commander | Qwen3 30B-A3B MoE Q4_K_M | ~18 GB |
| Space LLM | Mistral 7B Instruct Q4_K_M | ~3.5 GB |
| Defence LLM | Llama 3.2 8B Instruct Q4_K_M | ~4.5 GB |
| Quantum LLM | Qwen2.5 7B Instruct Q4_K_M | ~4.5 GB |
| Synthesis | Gemma2 27B Q4_K_M | ~14 GB |
| Embedding | nomic-embed-text | ~274 MB |

Models hot-swap: only one LLM loads at a time. Embedder stays resident.

---

## Quick Start

### 1. Install everything
```bash
chmod +x setup.sh
./setup.sh
```
This installs: Python venv, Ollama, all 6 models, Redis, Weaviate (Docker), Nginx.

### 2. Activate venv and start
```bash
source ~/venv/bin/activate
python main.py
```
UI at: **http://localhost:8000**

### 3. Run tests
```bash
python tests/test_system.py
```

---

## CLI Commands

```bash
# Start server
python main.py

# Ingest local documents into a domain KB
python main.py ingest space ./data/space/
python main.py ingest defence ./data/defence/
python main.py ingest quantum ./data/quantum/

# Trigger manual knowledge sync
python main.py sync

# Quick terminal query
python main.py query "How does QKD secure satellite links?"
```

### Advanced ingestion
```bash
# Ingest from arXiv (fetches paper abstracts)
python scripts/ingest_kb.py --domain quantum --arxiv "quantum key distribution" --limit 50

# Ingest from URL
python scripts/ingest_kb.py --domain defence --url https://example.com/report.pdf

# Seed all domains from arXiv automatically
python scripts/ingest_kb.py --all

# Check KB sizes
python scripts/ingest_kb.py --stats
```

---

## Fine-tuning Domain Specialists (Optional)

### Step 1: Prepare data
Put `.txt`, `.pdf`, or `.md` files in:
- `./data/space/`   — NASA papers, astrophysics docs, mission reports
- `./data/defence/` — Strategy docs, threat intelligence, policy papers
- `./data/quantum/` — arXiv quant-ph papers, IBM/Google quantum docs

### Step 2: Fine-tune with QLoRA
```bash
# Install training deps
pip install transformers peft trl datasets bitsandbytes accelerate

# Fine-tune space specialist
python scripts/finetune.py \
  --domain space \
  --data ./data/space/ \
  --base mistralai/Mistral-7B-Instruct-v0.3 \
  --epochs 3

# Fine-tune defence specialist
python scripts/finetune.py \
  --domain defence \
  --data ./data/defence/ \
  --base meta-llama/Llama-3.2-8B-Instruct

# Fine-tune quantum specialist
python scripts/finetune.py \
  --domain quantum \
  --data ./data/quantum/ \
  --base Qwen/Qwen2.5-7B-Instruct
```

### Step 3: Convert and load into Ollama
```bash
python scripts/merge_and_convert.py \
  --domain space \
  --base mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter ./models/space_lora
```

### Step 4: Update config
Edit `config/settings.py`:
```python
space_model   = "space-specialist"
defence_model = "defence-specialist"
quantum_model = "quantum-specialist"
```

---

## Production Deployment

### Supervisor (run all services as daemons)
```bash
sudo mkdir -p /var/log/research_army
sudo cp scripts/supervisord.conf /etc/supervisor/conf.d/research_army.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start research_army:*

# Check status
sudo supervisorctl status
```

### Nginx (reverse proxy)
```bash
sudo cp scripts/nginx.conf /etc/nginx/sites-available/research_army
sudo ln -s /etc/nginx/sites-available/research_army /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/research` | POST | Run research query |
| `/ingest` | POST | Ingest document |
| `/sync` | POST | Trigger knowledge sync |
| `/sync/history` | GET | Sync history |
| `/sync/conflicts` | GET | Conflict log |
| `/sessions` | GET | List sessions |
| `/sessions/{id}/history` | GET | Session messages |
| `/models` | GET | Model config |
| `/ws/research` | WS | Streaming research |
| `/health` | GET | Health check |

### Example API call
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How could quantum computing break satellite encryption?",
    "force_mode": "mode_b_plus"
  }'
```

---

## Knowledge Sync

The sync bus runs every 6 hours by default (set `SYNC_INTERVAL_HOURS` in `.env`).
Each domain emits a delta (new chunks since last sync) which gets cross-injected into
the other two domain KBs, tagged with `source_domain` and `is_cross_domain=true`.

Manual sync:
```bash
curl -X POST http://localhost:8000/sync -H "Content-Type: application/json" -d '{"force": true}'
```

---

## Environment Variables (.env)

```env
OLLAMA_BASE_URL=http://localhost:11434
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379
API_HOST=0.0.0.0
API_PORT=8000
SYNC_INTERVAL_HOURS=6
MAX_DEBATE_ROUNDS=3
EMBED_MODEL=nomic-embed-text
```

---

## File Structure

```
research_army/
├── setup.sh                    ← One-time install
├── requirements.txt
├── main.py                     ← Entry point + CLI
├── .env
├── config/
│   └── settings.py             ← All config + prompts
├── agents/
│   ├── llm.py                  ← Ollama wrapper
│   ├── commander.py            ← Query analysis + routing
│   ├── specialist.py           ← Domain agent + RAG
│   └── orchestrator.py         ← Main pipeline
├── rag/
│   └── pipeline.py             ← Weaviate ingest + retrieval
├── debate/
│   └── engine.py               ← Multi-round debate (Mode B+)
├── sync/
│   └── bus.py                  ← Knowledge sync + scheduler
├── memory/
│   └── store.py                ← Redis session + cache
├── api/
│   └── server.py               ← FastAPI REST + WebSocket
├── ui/
│   └── index.html              ← Dark terminal UI
├── scripts/
│   ├── finetune.py             ← QLoRA fine-tuning
│   ├── merge_and_convert.py    ← Merge + GGUF conversion
│   ├── ingest_kb.py            ← Bulk ingestion + arXiv
│   ├── supervisord.conf        ← Production daemon config
│   └── nginx.conf              ← Reverse proxy config
├── tests/
│   └── test_system.py          ← End-to-end tests
└── data/
    ├── space/                  ← Drop space docs here
    ├── defence/                ← Drop defence docs here
    └── quantum/                ← Drop quantum docs here
```
