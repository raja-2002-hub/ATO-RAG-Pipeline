"""
ATO Search Assistant — FastAPI endpoints (production v3).

Intelligence pipeline:
    User query → Preprocess (spell fix, intent, query expansion) → Multi-search → Generate

Features:
    - Query understanding: figures out what the user actually needs
    - Query expansion: generates 2-3 search queries to cover all angles
    - Conversation memory: understands follow-up questions
    - Spell correction + intent routing
    - Structured logging with request tracing
"""

import json
import re
import time
import uuid
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
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


# ──────────────────── State ────────────────────
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
    _state["start_time"] = time.time()

    config_errors = validate_config()
    for err in config_errors:
        log.error(f"CONFIG  │ {err}")
    if any("OPENAI_API_KEY" in e for e in config_errors):
        raise RuntimeError("Missing OPENAI_API_KEY")

    try:
        from retrieval.reranker import Reranker
        _state["reranker"] = Reranker()
        log.info("STARTUP │ Reranker loaded")
    except Exception as e:
        log.warning(f"STARTUP │ Reranker unavailable: {e}")

    try:
        from retrieval.retriever import retrieve
        _ = retrieve("test", reranker=None)
        _state["retriever_ok"] = True
        log.info("STARTUP │ Retriever verified (FAISS + BM25)")
    except Exception as e:
        log.error(f"STARTUP │ Retriever failed: {e}")

    try:
        import openai
        test_client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=10)
        test_client.models.list()
        _state["llm_ok"] = True
        log.info(f"STARTUP │ LLM verified ({LLM_MODEL})")
    except Exception as e:
        log.warning(f"STARTUP │ LLM check failed: {e}")

    log.info("STARTUP │ Ready ✓")
    yield
    log.info("SHUTDOWN │ Goodbye")


# ──────────────────── App ────────────────────
app = FastAPI(
    title="ATO Search Assistant",
    version="3.0.0",
    description="RAG-powered search system with intelligent query expansion",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────── Middleware ────────────────────
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
    if request.url.path not in ("/health", "/"):
        log.info(f"REQ {request_id} │ {request.method} {request.url.path} │ {response.status_code} │ {elapsed}ms")
    return response


# ──────────────────── Schemas ────────────────────
class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500)
    messages: Optional[List[Message]] = Field(default=None)

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
        _llm_client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)
    return _llm_client


def call_llm(prompt: str, request_id: str = "") -> tuple:
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
        log.info(f"LLM {request_id} │ {LLM_MODEL} │ {tokens_in}→{tokens_out} tokens │ {elapsed}ms")
        return text, ""
    except openai.AuthenticationError:
        return "", "LLM authentication failed."
    except openai.RateLimitError:
        return "", "LLM rate limited. Try again."
    except openai.APITimeoutError:
        return "", "LLM request timed out."
    except openai.APIConnectionError:
        return "", "Cannot reach LLM service."
    except Exception as e:
        log.error(f"LLM {request_id} │ {repr(e)}")
        return "", "An unexpected error occurred."


def call_llm_quick(prompt: str) -> str:
    """Lightweight LLM call for preprocessing."""
    try:
        response = _get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except:
        return ""


# ──────────────────── Query Intelligence ────────────────────

def understand_and_expand(query: str, messages: Optional[List[Message]], request_id: str = "") -> dict:
    """
    Single LLM call that:
    1. Fixes spelling
    2. Classifies intent (tax_question vs general_chat)
    3. Understands what the user ACTUALLY needs to know
    4. Generates 1-3 search queries to cover all relevant angles
    5. Handles follow-up questions using conversation history

    Returns: {
        "intent": "tax_question" | "general_chat",
        "understood_need": "what the user actually needs",
        "search_queries": ["query1", "query2", "query3"],
        "response": "direct response for general_chat"
    }
    """
    history_block = ""
    if messages and len(messages) >= 2:
        recent = messages[-6:]
        history_block = "Recent conversation:\n" + "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}"
            for m in recent
        ) + "\n\n"

    prompt = f"""{history_block}The user says: "{query}"

You are a query understanding system for an Australian tax information search engine.

Analyze this query and return a JSON object:
{{
    "intent": "tax_question" or "general_chat",
    "understood_need": "What does the user actually need to know? Think beyond the literal question.",
    "search_queries": ["query1", "query2", "query3"],
    "response": "Only for general_chat - a brief friendly response. Empty string for tax questions."
}}

Rules:
- - "general_chat" has TWO sub-types. Handle them differently in the "response" field:
  a) FRIENDLY: greetings (hi, hello, how are you, good morning), thanks (thank you, cheers), who are you → Give a warm friendly reply. Examples: "Hi! I'm the ATO Search Assistant. Ask me anything about Australian tax!", "You're welcome! Let me know if you have more tax questions.", "I'm doing great! How can I help with your tax question today?"
  b) OFF-TOPIC: questions about non-tax topics (programming, science, cooking, sports, general knowledge like "what is a linked list") → Politely redirect: "I'm the ATO Search Assistant — I can only help with Australian tax questions like TFN, ABN, GST, deductions, super, and more. How can I help with your tax question?"
- Questions about starting a business, freelancing, gig work (Uber, delivery, rideshare, contracting), or "what do I need to do X" are ALWAYS tax_question — the user needs to know about ABN, TFN, GST, sole trader obligations etc.
- "tax_question" = anything about Australian tax, ATO, TFN, ABN, GST, super, deductions, income, business.

CRITICAL for search_queries:
- Think about what the user ACTUALLY needs, not just what they literally asked.
- Example: "can I Uber with a TFN?" → The user wants to start Uber delivery. They need to know about:
  1. "ABN requirements for Uber Eats food delivery driver" (they probably need an ABN, not just TFN)
  2. "sole trader tax obligations gig economy" (how to report the income)
  3. "GST registration threshold ride sharing delivery" (when GST kicks in)
- Generate 1-3 search queries that cover ALL aspects the user should know about.
- Each query should be specific enough to find relevant ATO pages.
- For simple questions ("what is a TFN?"), one query is fine.

Return ONLY valid JSON."""

    raw = call_llm_quick(prompt)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        intent = result.get("intent", "tax_question")
        understood = result.get("understood_need", query)
        queries = result.get("search_queries", [query])
        direct_response = result.get("response", "")

        # Ensure we have at least one query
        if not queries:
            queries = [query]

        # Cap at 3 queries
        queries = queries[:3]

        log.info(
            f"UNDERSTAND {request_id} │ intent={intent} │ "
            f"need='{understood[:60]}' │ queries={queries}"
        )

        return {
            "intent": intent,
            "understood_need": understood,
            "search_queries": queries,
            "response": direct_response,
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning(f"UNDERSTAND {request_id} │ Parse failed: {e}")
        return {
            "intent": "tax_question",
            "understood_need": query,
            "search_queries": [query],
            "response": "",
        }


def multi_retrieve(queries: List[str], reranker) -> List[Dict]:
    """
    Run multiple search queries and merge results.
    Deduplicates by URL+text, keeps the highest-scored version.
    """
    from retrieval.retriever import retrieve

    seen = {}  # key: (url, text[:100]) → value: result dict
    all_results = []

    for q in queries:
        try:
            results = retrieve(q, reranker=reranker)
            for r in results:
                key = (r.get("url", ""), r.get("text", "")[:100])
                if key not in seen:
                    seen[key] = r
                    all_results.append(r)
        except Exception as e:
            log.warning(f"MULTI-RETRIEVE │ Query '{q[:50]}' failed: {e}")
            continue

    # Sort by score (rerank_score if available, else score)
    all_results.sort(
        key=lambda x: x.get("rerank_score", x.get("score", 0)),
        reverse=True,
    )

    # Return top results (more than usual since we merged multiple queries)
    return all_results[:FINAL_TOPK + 3]  # A few extra for richer evidence


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
    """
    Intelligent RAG endpoint.

    Pipeline: Understand → Expand queries → Multi-search → Generate
    """
    request_id = getattr(request.state, "request_id", "?")
    t0 = time.time()

    # ── Step 1: Understand intent + expand queries ──
    understood = understand_and_expand(body.q, body.messages, request_id)

    # Handle general conversation (no retrieval)
    if understood["intent"] == "general_chat" and understood["response"]:
        return AskResponse(
            answer=understood["response"],
            status="general_chat",
            references=[],
            disclaimer="",
            request_id=request_id,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

    # ── Step 2: Multi-query retrieval ──
    try:
        results = multi_retrieve(understood["search_queries"], _state["reranker"])
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

    log.info(
        f"RETRIEVE {request_id} │ {len(results)} passages from "
        f"{len(understood['search_queries'])} queries"
    )

    if not results:
        return AskResponse(
            answer="I couldn't find relevant ATO information. Please try rephrasing or check ato.gov.au.",
            status="no_results",
            references=[],
            disclaimer=DISCLAIMER,
            request_id=request_id,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

    # ── Step 3: Generate answer with full context ──
    evidence_text = format_evidence(results)

    # Conversation history for context
    history_block = ""
    if body.messages and len(body.messages) >= 2:
        recent = body.messages[-6:]
        history_block = (
            "Recent conversation:\n"
            + "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:500]}"
                for m in recent
            )
            + "\n\n"
        )

    # Tell the LLM what we understood so it answers comprehensively
    need_hint = ""
    if understood["understood_need"] != body.q:
        need_hint = (
            f"(System note: The user's underlying need is: {understood['understood_need']}. "
            f"Make sure your answer covers this comprehensively, not just the literal question.)\n\n"
        )

    prompt = (
        f"{history_block}"
        f"{need_hint}"
        f"Question: {body.q}\n\n"
        f"Evidence from ato.gov.au:\n{evidence_text}\n\n"
        f"Answer the user's question using the evidence. Use [1], [2] etc to cite sources. "
        f"Cover all aspects they need to know, even if they didn't explicitly ask."
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
        return JSONResponse(status_code=500, content={"error": "Search failed."})


@app.get("/diag")
def diag(q: str = "How do I apply for a TFN?"):
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
        return JSONResponse(status_code=500, content={"error": "Diagnostics failed."})


# ──────────────────── Error Handler ────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "?")
    _state["error_count"] += 1
    log.error(f"UNHANDLED {request_id} │ {repr(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong.", "request_id": request_id},
    )


# ──────────────────── Serve Frontend ────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "tax-advisor.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "ATO Search Assistant API is running."}