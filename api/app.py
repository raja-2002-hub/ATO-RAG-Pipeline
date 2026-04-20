"""
ATO Tax Assistant — FastAPI endpoints (production-hardened).

Usage:
    cd ATO-RAG_Pipeline
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

Environment:
    Requires OPENAI_API_KEY in .env or environment.
    See config/settings.py for all configuration.
"""

import re
import time
import uuid
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import (
    SYSTEM_PROMPT, DISCLAIMER, FINAL_TOPK,
    OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
    CORS_ORIGINS, MAX_QUERY_LENGTH, RATE_LIMIT_PER_MIN,
    FAISS_INDEX_PATH, BM25_INDEX_PATH, META_PATH,
    validate as validate_config,
)

# ──────────────────── Logging ────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ato-rag")


# ──────────────────── Startup / Shutdown ────────────────────
# Track components that loaded successfully
_state = {
    "reranker": None,
    "retriever_ok": False,
    "llm_ok": False,
    "start_time": None,
    "request_count": 0,
    "error_count": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown. Validates config, loads models."""
    _state["start_time"] = time.time()

    # 1. Validate config
    config_errors = validate_config()
    for err in config_errors:
        log.error(f"CONFIG  │ {err}")
    if any("OPENAI_API_KEY" in e for e in config_errors):
        log.error("CONFIG  │ Cannot start without OPENAI_API_KEY. Exiting.")
        raise RuntimeError("Missing OPENAI_API_KEY")

    # 2. Load reranker
    try:
        from retrieval.reranker import Reranker
        _state["reranker"] = Reranker()
        log.info("STARTUP │ Reranker loaded")
    except Exception as e:
        log.warning(f"STARTUP │ Reranker unavailable: {e}")

    # 3. Verify retriever works
    try:
        from retrieval.retriever import retrieve
        _ = retrieve("test", reranker=None)
        _state["retriever_ok"] = True
        log.info("STARTUP │ Retriever verified (FAISS + BM25)")
    except Exception as e:
        log.error(f"STARTUP │ Retriever failed: {e}")

    # 4. Verify LLM connectivity
    try:
        import openai
        test_client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=10)
        test_client.models.list()
        _state["llm_ok"] = True
        log.info(f"STARTUP │ LLM verified ({LLM_MODEL})")
    except Exception as e:
        log.warning(f"STARTUP │ LLM check failed: {e}")

    index_warnings = [e for e in config_errors if "not found" in e]
    if index_warnings:
        for w in index_warnings:
            log.warning(f"STARTUP │ {w}")

    log.info("STARTUP │ Ready ✓")
    yield
    log.info("SHUTDOWN │ Goodbye")


# ──────────────────── App ────────────────────
app = FastAPI(
    title="ATO Tax Assistant",
    version="1.0.0",
    description="RAG-powered Australian tax information assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────── Middleware: request tracing + timing ────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.time()

    response = await call_next(request)

    elapsed = round((time.time() - t0) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed}ms"

    _state["request_count"] += 1

    # Log all requests except /health (too noisy)
    if request.url.path != "/health":
        log.info(
            f"REQ {request_id} │ {request.method} {request.url.path} │ "
            f"{response.status_code} │ {elapsed}ms"
        )

    return response


# ──────────────────── Request / Response Schemas ────────────────────
class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500, description="Tax question")

class Reference(BaseModel):
    ref_number: int
    title: str
    section: str
    url: str
    breadcrumb: str
    snippet: str

class AskResponse(BaseModel):
    answer: Optional[str]
    status: str
    references: List[Reference]
    disclaimer: str
    request_id: Optional[str] = None
    elapsed_ms: Optional[int] = None

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500)

class HealthResponse(BaseModel):
    ok: bool
    model: str
    reranker: bool
    retriever: bool
    llm: bool
    uptime_seconds: int
    requests_served: int
    errors: int


# ──────────────────── LLM ────────────────────
import openai

_llm_client = None

def _get_llm_client() -> openai.OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=LLM_TIMEOUT,
        )
    return _llm_client


def call_llm(prompt: str, request_id: str = "") -> tuple:
    """Call OpenAI API. Returns (response_text, error_string)."""
    try:
        t0 = time.time()
        response = _get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        elapsed = round((time.time() - t0) * 1000)
        text = response.choices[0].message.content.strip()
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        log.info(
            f"LLM {request_id} │ {LLM_MODEL} │ "
            f"{tokens_in}→{tokens_out} tokens │ {elapsed}ms"
        )
        return text, ""

    except openai.AuthenticationError:
        log.error(f"LLM {request_id} │ Authentication failed — check OPENAI_API_KEY")
        return "", "LLM authentication failed. Check API key configuration."

    except openai.RateLimitError:
        log.warning(f"LLM {request_id} │ Rate limited by OpenAI")
        return "", "LLM rate limited. Please try again in a moment."

    except openai.APITimeoutError:
        log.warning(f"LLM {request_id} │ Timeout after {LLM_TIMEOUT}s")
        return "", "LLM request timed out. Please try again."

    except openai.APIConnectionError:
        log.error(f"LLM {request_id} │ Connection failed")
        return "", "Cannot reach LLM service. Please try again later."

    except Exception as e:
        log.error(f"LLM {request_id} │ Unexpected: {repr(e)}")
        return "", "An unexpected error occurred generating the answer."


# ──────────────────── Helpers ────────────────────
def format_evidence(results: List[Dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        header = r.get("title", "")
        section = r.get("section_heading", "")
        if section and section != header:
            header += f" — {section}"
        block = f"[{i}] {header}\n    URL: {r.get('url', '')}\n    {r.get('text', '')}"
        blocks.append(block)
    return "\n\n".join(blocks)


def clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_references(results: List[Dict]) -> List[dict]:
    refs = []
    for i, r in enumerate(results, 1):
        refs.append({
            "ref_number": i,
            "title": r.get("title", ""),
            "section": r.get("section_heading", ""),
            "url": r.get("url", ""),
            "breadcrumb": r.get("breadcrumb", ""),
            "snippet": r.get("text", "")[:200],
        })
    return refs


# ──────────────────── Endpoints ────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — shows component status and basic metrics."""
    uptime = int(time.time() - _state["start_time"]) if _state["start_time"] else 0
    return {
        "ok": _state["retriever_ok"] and _state["llm_ok"],
        "model": LLM_MODEL,
        "reranker": _state["reranker"] is not None,
        "retriever": _state["retriever_ok"],
        "llm": _state["llm_ok"],
        "uptime_seconds": uptime,
        "requests_served": _state["request_count"],
        "errors": _state["error_count"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    """Main RAG endpoint — retrieves evidence and generates an answer."""
    request_id = getattr(request.state, "request_id", "?")
    t0 = time.time()

    # 1. Retrieve
    try:
        from retrieval.retriever import retrieve
        results = retrieve(body.q, reranker=_state["reranker"])
    except Exception as e:
        _state["error_count"] += 1
        log.error(f"RETRIEVE {request_id} │ {repr(e)}")
        return AskResponse(
            answer="I'm having trouble searching ATO sources right now. Please try again.",
            status="retrieval_error",
            references=[],
            disclaimer=DISCLAIMER,
            request_id=request_id,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

    log.info(f"RETRIEVE {request_id} │ {len(results)} passages for: {body.q[:80]}")

    # 2. Handle no results
    if not results:
        return AskResponse(
            answer="I couldn't find relevant ATO information for that question. Please try rephrasing or check ato.gov.au directly.",
            status="no_results",
            references=[],
            disclaimer=DISCLAIMER,
            request_id=request_id,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

    # 3. Generate answer
    evidence_text = format_evidence(results)
    prompt = (
        f"Question: {body.q}\n\n"
        f"Evidence from ato.gov.au:\n{evidence_text}\n\n"
        f"Please answer using the evidence above. Use [1], [2] etc to cite sources."
    )

    answer, err = call_llm(prompt, request_id)
    references = build_references(results)

    if err:
        _state["error_count"] += 1
        return AskResponse(
            answer=f"I found relevant sources but couldn't generate an answer: {err}",
            status="generation_error",
            references=references,
            disclaimer=DISCLAIMER,
            request_id=request_id,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

    answer = clean_answer(answer)

    # Append disclaimer if model didn't include one
    if "general information" not in answer.lower() and "professional" not in answer.lower():
        answer += f"\n\n{DISCLAIMER}"

    elapsed = round((time.time() - t0) * 1000)
    log.info(f"ANSWER  {request_id} │ {len(answer)} chars │ {elapsed}ms total")

    return AskResponse(
        answer=answer,
        status="answered",
        references=references,
        disclaimer=DISCLAIMER,
        request_id=request_id,
        elapsed_ms=elapsed,
    )


@app.post("/search")
def search_only(body: SearchRequest, request: Request):
    """Retrieval-only endpoint — no LLM call."""
    request_id = getattr(request.state, "request_id", "?")
    try:
        from retrieval.retriever import retrieve
        results = retrieve(body.q, reranker=_state["reranker"])
        return {
            "q": body.q,
            "results": [
                {
                    "title": r.get("title", ""),
                    "section": r.get("section_heading", ""),
                    "url": r.get("url", ""),
                    "breadcrumb": r.get("breadcrumb", ""),
                    "score": round(r.get("rerank_score", r.get("score", 0)), 4),
                    "text": r.get("text", ""),
                }
                for r in results
            ],
            "request_id": request_id,
        }
    except Exception as e:
        _state["error_count"] += 1
        log.error(f"SEARCH {request_id} │ {repr(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Search failed. Please try again.", "request_id": request_id},
        )


@app.get("/diag")
def diag(q: str = "How do I apply for a TFN?"):
    """Debug endpoint — shows retrieval internals."""
    try:
        from retrieval.retriever import retrieve
        results = retrieve(q, reranker=_state["reranker"])
        return {
            "q": q,
            "num_results": len(results),
            "model": LLM_MODEL,
            "reranker_active": _state["reranker"] is not None,
            "results": [
                {
                    "rank": i,
                    "title": r.get("title", ""),
                    "section": r.get("section_heading", ""),
                    "url": r.get("url", ""),
                    "score": round(r.get("rerank_score", r.get("score", 0)), 4),
                    "text_preview": r.get("text", "")[:300],
                }
                for i, r in enumerate(results, 1)
            ],
        }
    except Exception as e:
        log.error(f"DIAG │ {repr(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Diagnostics failed.", "detail": repr(e)},
        )


# ──────────────────── Global Error Handler ────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so users never see raw stack traces."""
    request_id = getattr(request.state, "request_id", "?")
    _state["error_count"] += 1
    log.error(f"UNHANDLED {request_id} │ {repr(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Something went wrong. Please try again.",
            "request_id": request_id,
        },
    )