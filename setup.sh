#!/bin/bash
set -e

echo "============================================"
echo "  Research Army - Full Setup Script"
echo "  AWS g5.xlarge | A10G 24GB GPU"
echo "============================================"

# ── System deps ──────────────────────────────────
sudo apt-get update -y
sudo apt-get install -y \
    curl wget git build-essential \
    python3-pip python3-venv python3-dev \
    libpq-dev gcc g++ cmake \
    redis-server redis-tools \
    docker.io docker-compose \
    nginx supervisor htop nvtop

# ── Python venv ───────────────────────────────────
echo "[1/7] Creating Python environment..."
python3 -m venv /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate
pip install --upgrade pip wheel setuptools

# ── Python packages ───────────────────────────────
echo "[2/7] Installing Python packages..."
pip install \
    fastapi uvicorn[standard] \
    langchain langchain-community langchain-core \
    langgraph \
    chromadb \
    sentence-transformers \
    ollama \
    httpx aiohttp \
    redis \
    pydantic pydantic-settings \
    python-dotenv \
    PyPDF2 pypdf \
    unstructured \
    tiktoken \
    numpy pandas \
    rich typer \
    schedule \
    websockets \
    python-multipart \
    jinja2 \
    torch --index-url https://download.pytorch.org/whl/cu118

# ── Ollama ────────────────────────────────────────
echo "[3/7] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 5

# ── Pull models ───────────────────────────────────
echo "[4/7] Pulling models (this will take a while)..."
ollama pull qwen3:30b-a3b-instruct-q4_K_M    # Commander ~18GB
ollama pull mistral:7b-instruct-q4_K_M        # Space LLM ~3.5GB
ollama pull llama3.3:8b-instruct-q4_K_M       # Defence LLM ~4.5GB  (using llama3.2 as fallback)
ollama pull qwen3:8b-instruct-q4_K_M          # Quantum LLM ~4.5GB (using qwen2.5 as fallback)
ollama pull gemma2:27b-instruct-q4_K_M        # Synthesis ~14GB
ollama pull nomic-embed-text                   # Embedder ~274MB

echo "Models pulled:"
ollama list

# ── Redis ─────────────────────────────────────────
echo "[5/7] Starting Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

# ── Weaviate via Docker ───────────────────────────
echo "[6/7] Starting Weaviate vector store..."
docker run -d \
  --name weaviate \
  --restart always \
  -p 8080:8080 \
  -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -v weaviate_data:/var/lib/weaviate \
  cr.weaviate.io/semitechnologies/weaviate:1.24.1

sleep 5
echo "Weaviate running at http://localhost:8080"

# ── Project dirs ──────────────────────────────────
echo "[7/7] Setting up project..."
cd /home/ubuntu
cp -r /home/claude/research_army ./research_army
cd research_army

# env file
cat > .env << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
SYNC_INTERVAL_HOURS=6
MAX_DEBATE_ROUNDS=3
EMBED_MODEL=nomic-embed-text
EOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Start server: python main.py"
echo "  UI:           http://localhost:8000"
echo "============================================"
