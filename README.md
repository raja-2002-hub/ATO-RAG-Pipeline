# ATO Tax Advisor — RAG Pipeline

A production-grade Retrieval-Augmented Generation system that answers Australian tax questions using official ATO (Australian Taxation Office) source documents. Every answer is grounded in real ATO content with clickable citations users can verify.

Built as a complete, end-to-end RAG pipeline: web crawling → content processing → hybrid indexing → retrieval → reranking → generation → evaluated frontend.

---

## What it does

Ask a tax question in plain English. The system searches 30,828 chunks from 7,517 ATO pages, retrieves the most relevant passages using hybrid search (dense + sparse), reranks them with a cross-encoder, and generates an answer with inline citations pointing to the exact ATO source pages.

**Example:**

> **Q:** What is a tax file number and how do I get one?
>
> **A:** A tax file number (TFN) is your personal reference number in the tax and superannuation systems, and it is free to apply for one. You can apply at any age, and should do so before starting work to avoid paying more tax **[1][2]**. If you apply online using a Digital ID, you must be at least 15 years old **[2]**. It can take up to 28 days to process **[2]**.
>
> *Sources: [1] What is a tax file number — ato.gov.au  [2] Tax in Australia: what you need to know — ato.gov.au*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
│              tax-advisor.html (vanilla JS)                      │
│         Chat UI · Citation links · Source cards                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ POST /ask {q: "..."}
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                 │
│                  FastAPI + Pydantic                              │
│     /ask  ·  /search  ·  /diag  ·  /health                     │
│  Request tracing · Structured logging · Error handling          │
└────────┬───────────────────────────────────────┬────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────────┐             ┌────────────────────────────┐
│    RETRIEVAL         │             │       GENERATION           │
│                      │             │                            │
│  ┌───────────────┐  │             │  System prompt + evidence  │
│  │  FAISS Dense  │  │             │         ▼                  │
│  │  (30,828 vecs)│──┤  RRF        │    GPT-4o-mini             │
│  └───────────────┘  │  Fusion     │         ▼                  │
│  ┌───────────────┐  │──► Rerank   │  Answer with [1][2] cites  │
│  │  BM25 Sparse  │──┤   (cross-   │                            │
│  │  (keyword)    │  │   encoder)  │                            │
│  └───────────────┘  │      ▼      │                            │
│                      │    MMR      │                            │
│                      │  (diverse)  │                            │
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
| Embeddings | all-MiniLM-L6-v2 | Fast, runs locally, no API cost for indexing |
| Generation | GPT-4o-mini | Best cost/quality ratio for grounded Q&A |
| API | FastAPI + Pydantic | Type-safe, async-ready, auto-generated docs at /docs |
| Frontend | Vanilla HTML/CSS/JS | No build step, single file, works anywhere |

---

## Project Structure

```
ATO-RAG_Pipeline/
├── api/
│   └── app.py                  # FastAPI endpoints, error handling, request tracing
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
│   └── tax-advisor.html        # Chat UI with citation links
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
git clone https://github.com/yourusername/ATO-RAG_Pipeline.git
cd ATO-RAG_Pipeline

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
# Start the API
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# In another terminal — serve the frontend
cd frontend
python -m http.server 3000

# Open http://localhost:3000/tax-advisor.html
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

Main endpoint — retrieves evidence and generates an answer.

**Request:**
```json
{"q": "What is a tax file number?"}
```

**Response:**
```json
{
  "answer": "A tax file number (TFN) is your personal reference...",
  "status": "answered",
  "references": [
    {
      "ref_number": 1,
      "title": "What is a tax file number?",
      "section": "What is a tax file number?",
      "url": "https://www.ato.gov.au/individuals-and-families/tax-file-number/...",
      "breadcrumb": "Individuals And Families > Tax File Number > ...",
      "snippet": "Find out why you need a TFN..."
    }
  ],
  "disclaimer": "⚠️ This is general information only...",
  "request_id": "a3f2b1c0",
  "elapsed_ms": 1342
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

---

## What I'd Improve With More Time

- **Streaming responses** — stream LLM output token-by-token for better UX
- **Incremental re-indexing** — currently full rebuild; would add delta updates when ATO pages change
- **Query understanding** — classify query intent (factual, procedural, eligibility) to adjust retrieval strategy
- **User feedback loop** — thumbs up/down on answers to build a preference dataset
- **A/B testing framework** — compare retrieval configurations on live traffic
- **Prompt caching** — use Anthropic/OpenAI prompt caching to reduce LLM costs by ~60%

---

## License

This project is for educational and portfolio purposes. ATO content is Australian Government material used under the terms of the Creative Commons Attribution 4.0 International licence.
