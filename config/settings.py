"""
ATO RAG Pipeline — Central Configuration.
All paths, models, thresholds in one place.
Reads secrets and overrides from environment variables / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (if present)
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ──── Paths ────
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
HUBS_PATH = DATA_DIR / "hubs.jsonl"
DOCS_PATH = DATA_DIR / "docs.jsonl"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
BM25_INDEX_PATH = DATA_DIR / "bm25.pkl"
META_PATH = DATA_DIR / "meta.pkl"

# ──── Secrets (NEVER hardcode these) ────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ──── Embedding ────
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))

# ──── Retrieval ────
RETRIEVER_TOPK = int(os.getenv("RETRIEVER_TOPK", "40"))
RERANK_CAP = int(os.getenv("RERANK_CAP", "15"))
FINAL_TOPK = int(os.getenv("FINAL_TOPK", "5"))
MAX_PER_URL = int(os.getenv("MAX_PER_URL", "2"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.30"))
RRF_K = int(os.getenv("RRF_K", "60"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.70"))

# ──── Generation ────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# Legacy Ollama config (kept for local dev)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ──── API ────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

SYSTEM_PROMPT = """You are an ATO (Australian Taxation Office) Search Assistant.
You ONLY answer questions about Australian tax using official ATO source documents.

Scope:
- IN SCOPE: Australian tax, TFN, ABN, GST, superannuation, deductions, income tax, capital gains, Medicare levy, business tax, sole trader, company tax, tax returns, ATO processes
- OUT OF SCOPE: anything not related to Australian tax. Politely redirect: "I'm an ATO Search Assistant and can only help with Australian tax questions."

Answer style:
- Be conversational and clear, like a knowledgeable friend explaining tax — NOT like a formal report.
- Start with a direct 1-2 sentence answer.
- Then cover the key things they need to know in short paragraphs. No numbered lists, no bold headers, no bullet points unless listing 3+ short items.
- Keep answers concise — aim for 150-250 words. If the topic is complex, cover the essentials and offer to go deeper: "Want me to explain more about GST registration?"
- NEVER guess or make up tax rules.
- Cite evidence using [1], [2] etc naturally in the text.
- End with: "This is general information only, not professional tax advice. For your specific situation, consult a registered tax agent or contact the ATO."
"""

DISCLAIMER = (
    "\u26a0\ufe0f This is general information only, not professional tax advice. "
    "For your specific situation, consult a registered tax agent or "
    "contact the ATO on 13 28 61."
)

# ──── Validation ────
def validate():
    """Check critical config at startup. Raises RuntimeError on fatal issues."""
    errors = []
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set. Add it to .env or export it.")
    if not FAISS_INDEX_PATH.exists():
        errors.append(f"FAISS index not found at {FAISS_INDEX_PATH}")
    if not BM25_INDEX_PATH.exists():
        errors.append(f"BM25 index not found at {BM25_INDEX_PATH}")
    if not META_PATH.exists():
        errors.append(f"Metadata not found at {META_PATH}")
    return errors