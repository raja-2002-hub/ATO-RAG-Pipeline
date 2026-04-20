# ATO Search Assistant — RAG Pipeline

A production-grade Retrieval-Augmented Generation system that answers Australian tax questions using official ATO (Australian Taxation Office) source documents. Every answer is grounded in real ATO content with clickable citations users can verify.

Built as a complete, end-to-end RAG pipeline: web crawling → content processing → hybrid indexing → intelligent query expansion → retrieval → reranking → generation → evaluated frontend.

**Live Demo:** [https://huggingface.co/spaces/rajaRajaseelan2002/ato-search-assistant](https://huggingface.co/spaces/rajaRajaseelan2002/ato-search-assistant)

**GitHub:** [https://github.com/raja-2002-hub/ATO-RAG-Pipeline](https://github.com/raja-2002-hub/ATO-RAG-Pipeline)

---

## What it does

Ask a tax question in plain English — even with typos or vague phrasing. The system understands what you actually need, searches 30,828 chunks from 7,517 ATO pages using multiple expanded queries, reranks with a cross-encoder, and generates a comprehensive answer with inline citations.

**Example — vague question, intelligent answer:**

> **Q:** i want to start uber eats delivery what do i need
>
> The system expands this into 3 search queries:
> - "ABN requirements for Uber food delivery driver"
> - "sole trader tax obligations gig economy"
> - "GST registration threshold delivery services"
>
> **A:** To do Uber Eats delivery in Australia, you'll need an ABN (Australian Business Number) since you'll be operating as a sole trader **[1]**. You should also register for GST if your annual turnover hits $75,000 — and ride-sourcing services require GST registration regardless of turnover **[2]**. For your tax return, report all delivery income under the business items section using your TFN **[1]**, and you can claim deductions for expenses like fuel and vehicle costs **[3]**.

**Example — follow-up conversation:**

> **Q:** explain it in simple terms
>
> The system uses conversation history to understand "it" refers to the Uber delivery requirements, and gives a simplified explanation without the user repeating the topic.

---

## Intelligence Pipeline

Every query goes through a preprocessing step before retrieval:

```
User input → Understand (spell fix, intent, query expand) → Multi-search → Generate
                    │                                              │
              ┌─────┴──────┐                               ┌──────┴───────┐
              │ "general"  │                               │ "tax"        │
              │ greeting → │                               │ 1-3 expanded │
              │ friendly   │                               │ queries      │
              │ reply      │                               │ → retrieve   │
              │            │                               │ → merge      │
              │ off-topic →│                               │ → rerank     │
              │ redirect   │                               │ → generate   │
              └────────────┘                               └──────────────┘
```

- **Spell correction**: "wat is tfn" → "what is a TFN"
- **Intent routing**: Greetings get friendly replies, off-topic gets redirected, tax questions go through RAG
- **Query expansion**: "can I Uber with a TFN?" generates 3 searches covering ABN, sole trader, and GST
- **Conversation memory**: Follow-ups like "explain it simply" use chat history for context
- **Scope enforcement**: Non-tax questions ("what is a linked list?") are politely redirected

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
│              tax-advisor.html (vanilla JS)                      │
│    Chat UI · Citations · Sources · Conversation memory          │
└────────────────────────┬────────────────────────────────────────┘
                         │ POST /ask {q, messages}
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY INTELLIGENCE                           │
│         Spell fix · Intent classify · Query expansion           │
│              1 LLM call → 1-3 search queries                   │
└────────┬───────────────────────────────────────┬────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────────┐             ┌────────────────────────────┐
│  MULTI-RETRIEVAL     │             │       GENERATION           │
│                      │             │                            │
│  For each query:     │             │  Conversation history      │
│  ┌───────────────┐  │             │  + user need hint          │
│  │  FAISS Dense  │  │             │  + evidence                │
│  │  (30,828 vecs)│──┤  RRF        │         ▼                  │
│  └───────────────┘  │  Fusion     │    GPT-4o-mini             │
│  ┌───────────────┐  │──► Rerank   │         ▼                  │
│  │  BM25 Sparse  │──┤   (cross-   │  Answer with [1][2] cites  │
│  │  (keyword)    │  │   encoder)  │                            │
│  └───────────────┘  │      ▼      │                            │
│                      │    MMR      │                            │
│  Merge + deduplicate │  (diverse)  │                            │
└─────────────────────┘             └────────────────────────────┘
         ▲
         │ Offline indexing pipeline
┌────────┴────────────────────────────────────────────────────────┐
│                     INGESTION                                   │
│  Crawl ATO pages → Extract content → Chunk → Embed → Index     │
│  7,517 pages  →  30,828 chunks  →  FAISS + BM25 indexes        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Results

Evaluated against 100 hand-crafted questions across 4 difficulty tiers and 16 categories.

### Retrieval Metrics

| Metric | Score |
|---|---|
| Hit Rate | **92%** |
| MRR | **0.790** |
| Recall@1 | 70% |
| Recall@3 | 87% |
| Recall@5 | 92% |
| Keyword Recall | 89.5% |

### Performance by Difficulty Tier

| Tier | Description | Hit Rate | MRR |
|---|---|---|---|
| T1 | Simple factual (e.g. "What is the tax-free threshold?") | 85% | 0.717 |
| T2 | Moderate (e.g. "Can I claim self-education expenses?") | **96.7%** | 0.868 |
| T3 | Complex multi-part (e.g. "CGT on inherited property") | **100%** | 0.829 |
| T4 | Edge cases and safety (e.g. "negative gearing", abbreviations) | 90% | 0.775 |

### Category Breakdown

| Category | Hit Rate | Category | Hit Rate |
|---|---|---|---|
| Super | 8/8 (100%) | CGT | 8/8 (100%) |
| Medicare | 7/7 (100%) | Foreign Income | 6/6 (100%) |
| Life Events | 5/5 (100%) | Multi-part | 5/5 (100%) |
| WHM | 6/6 (100%) | Abbreviations | 5/5 (100%) |
| Natural Language | 5/5 (100%) | Negative Tests | 5/5 (100%) |
| Income Tax | 9/10 (90%) | Investments | 4/5 (80%) |
| Deductions | 10/15 (67%) | Accuracy | 6/7 (86%) |

### Test Suite

45 automated tests across 9 categories — all passing:

```
tests/test_pipeline.py  ·  45 passed in 87s

 Retrieval (11)        ✓  Results, field structure, ATO URLs, relevance, edge cases, diversity
 Search Components (5) ✓  Dense, sparse, hybrid, fused scores, overlap validation
 Reranker (5)          ✓  Loading, count preservation, scoring, relevance preference
 API Endpoints (9)     ✓  /health, /search, /diag, /ask — structure, headers, tracing
 Input Validation (5)  ✓  Empty, missing, too-long, whitespace — all rejected
 Performance (3)       ✓  Retrieval <15s, search <15s, health <1s
 Eval Quality Gates (6)✓  Hit rate ≥85%, MRR ≥0.65, R@5 ≥85%, all tiers ≥75%
 Safety (1)            ✓  No stack traces exposed in error responses
 Answer Quality (8)    ✓  Citations, references, disclaimers, ATO URLs (LLM tests)
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Crawling | Requests + Playwright | Playwright retries JS-rendered ATO pages that fail with plain HTTP |
| Chunking | Custom (hub detection + section-aware) | ATO pages have inconsistent structure; generic splitters lose section context |
| Dense Index | FAISS (flat, normalized) | Fast, no external service, good enough for 30k vectors |
| Sparse Index | BM25 (custom implementation) | Catches keyword matches that dense embeddings miss |
| Fusion | Reciprocal Rank Fusion (RRF) | Proven method to combine dense + sparse without score calibration |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Lightweight cross-encoder, runs on CPU, improves precision |
| Diversity | MMR (Maximal Marginal Relevance) | Prevents top-5 from being 5 chunks from the same page |
| Query Intelligence | GPT-4o-mini (preprocessing) | Single LLM call handles spell fix, intent, and query expansion |
| Embeddings | all-MiniLM-L6-v2 | Fast, runs locally, no API cost for indexing |
| Generation | GPT-4o-mini | Best cost/quality ratio for grounded Q&A |
| API | FastAPI + Pydantic | Type-safe, async-ready, auto-generated docs at /docs |
| Frontend | Vanilla HTML/CSS/JS | No build step, single file, works anywhere |
| Deployment | HuggingFace Spaces (Docker) | Free hosting, handles large files, always available |

---

## Project Structure

```
ATO-RAG_Pipeline/
├── api/
│   └── app.py                  # FastAPI endpoints, query intelligence, request tracing
├── config/
│   └── settings.py             # All config from environment variables
├── crawler/
│   ├── fetch_pages.py          # Downloads ATO HTML pages
│   └── retry_failed.py         # Playwright retry for JS-rendered pages
├── processing/
│   ├── process_pages.py        # Hub detection + content extraction + chunking
│   └── prep_docs.py            # Builds docs.jsonl for indexing
├── indexing/
│   ├── bm25.py                 # BM25 keyword search implementation
│   └── build_index.py          # Builds FAISS + BM25 indexes
├── retrieval/
│   ├── retriever.py            # Hybrid search + RRF + MMR + score adjustments
│   └── reranker.py             # Cross-encoder reranker
├── frontend/
│   └── tax-advisor.html        # Chat UI with citations + conversation memory
├── tests/
│   └── test_pipeline.py        # 45 automated tests
├── data/
│   ├── raw_html/               # 7,517 crawled ATO pages
│   ├── chunks.jsonl            # 30,828 processed chunks
│   ├── faiss.index             # Dense vector index
│   ├── bm25.pkl                # Sparse keyword index
│   └── meta.pkl                # Chunk metadata
├── evaluate.py                 # 100-question evaluation suite
├── run_pipeline.py             # One command: prep + index
├── Dockerfile                  # Container build
├── docker-compose.yml          # One-command deployment
├── .env.example                # Environment template
├── pytest.ini                  # Test configuration
└── requirements.txt            # Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Setup

```bash
git clone https://github.com/raja-2002-hub/ATO-RAG-Pipeline.git
cd ATO-RAG-Pipeline

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
# Start the server (serves both API and frontend)
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

### Docker (alternative)

```bash
cp .env.example .env
# Edit .env with your API key

docker compose up --build
# Open http://localhost:8000
```

### Run Tests

```bash
# Fast (no LLM calls, free)
python -m pytest tests/ -v -k "not slow"

# Full (includes LLM-graded tests, ~$0.10)
python -m pytest tests/ -v
```

### Run Evaluation

```bash
# Retrieval only (fast, free)
python evaluate.py --no-llm

# Full eval with GPT grading
python evaluate.py

# Compare dense vs hybrid search
python evaluate.py --compare-search
```

---

## API Reference

### `POST /ask`

Main endpoint — understands intent, expands queries, retrieves evidence, generates an answer.

**Request:**
```json
{
  "q": "wat do i need for uber delivery",
  "messages": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
}
```

**Response:**
```json
{
  "answer": "To do Uber Eats delivery in Australia, you'll need an ABN...",
  "status": "answered",
  "references": [
    {
      "ref_number": 1,
      "title": "Business structures - key tax obligations",
      "section": "Sole trader",
      "url": "https://www.ato.gov.au/...",
      "breadcrumb": "Businesses And Organisations > ...",
      "snippet": "As a sole trader, you use your individual TFN..."
    }
  ],
  "disclaimer": "⚠️ This is general information only...",
  "request_id": "a3f2b1c0",
  "elapsed_ms": 3420
}
```

### `POST /search`

Retrieval only — no LLM call. Returns ranked passages.

### `GET /health`

Component status, uptime, request count, error count.

### `GET /diag?q=...`

Debug view — shows retrieval scores and internals.

---

## Design Decisions

**Why intelligent query expansion?**
Users don't know what to search for. Someone asking "can I Uber with a TFN?" doesn't know they need an ABN — that's why they're asking. The system generates 1-3 targeted search queries that cover all relevant angles, finding evidence the user didn't know to ask about.

**Why conversation memory without a database?**
The frontend holds the conversation in a JavaScript array and sends the last 3 exchanges with each request. The backend uses this to rewrite vague follow-ups ("explain it simply") into standalone queries. No database, no session storage, no complexity — the frontend is the source of truth.

**Why hybrid search (dense + sparse)?**
Dense embeddings are great at semantic matching ("how do I get a TFN" → "tax file number application"), but miss exact keyword matches that matter in tax ("Division 293 tax"). BM25 catches these. RRF fusion combines both without needing score calibration.

**Why a cross-encoder reranker?**
Bi-encoder retrieval (FAISS) is fast but imprecise — it encodes query and document separately. The cross-encoder sees them together, catching subtle relevance signals. On our eval, adding the reranker improved MRR by ~0.08.

**Why MMR diversity?**
Without it, top-5 results are often 5 chunks from the same ATO page. MMR pushes for diverse URLs, giving the LLM a broader evidence base and better answers.

**Why GPT-4o-mini over a local model?**
For a tax domain where wrong answers have real consequences, generation quality matters more than cost. GPT-4o-mini at ~$0.15/1M input tokens is cheap enough for this use case. The system is designed so swapping to a local model (Ollama) requires changing one environment variable.

**Why no caching?**
Tax rules change. Caching answers risks serving stale information about tax rates, thresholds, or deadlines. The retrieval layer (FAISS + BM25) is already fast (<5s). If cost became an issue at scale, I'd add embedding caching and LLM provider prompt caching — not answer caching.

**Why no vector database (Pinecone, Weaviate)?**
With 30,828 chunks, FAISS flat index fits in memory and searches in <10ms. A managed vector DB adds cost, latency, and operational complexity for zero benefit at this scale. If the corpus grew to 1M+ chunks, I'd consider it.

**Why no LangChain / LangGraph?**
The entire intelligence pipeline is two LLM calls — one to understand and expand the query, one to generate the answer. LangChain would wrap this in layers of abstraction with no added value. Raw Python is easier to debug, test, and explain in an interview.

---

## What I'd Improve With More Time

- **Streaming responses** — stream LLM output token-by-token for better UX
- **Incremental re-indexing** — currently full rebuild; would add delta updates when ATO pages change
- **User feedback loop** — thumbs up/down on answers to build a preference dataset
- **A/B testing framework** — compare retrieval configurations on live traffic
- **Prompt caching** — use Anthropic/OpenAI prompt caching to reduce LLM costs by ~60%
- **Multi-turn query planning** — for complex scenarios requiring multiple retrieval steps

---

## License

This project is for educational and portfolio purposes. ATO content is Australian Government material used under the terms of the Creative Commons Attribution 4.0 International licence.