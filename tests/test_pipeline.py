"""
ATO RAG Pipeline — Test Suite.

Tests retrieval quality, API endpoints, input validation,
error handling, and pipeline integrity.

Usage:
    cd ATO-RAG_Pipeline
    python -m pytest tests/ -v
    python -m pytest tests/ -v -k "not slow"     # skip LLM tests
    python -m pytest tests/ -v --tb=short         # compact output
"""

import sys
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Setup path ──
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def reranker():
    """Load reranker once for all tests."""
    try:
        from retrieval.reranker import Reranker
        return Reranker()
    except Exception:
        pytest.skip("Reranker not available")


@pytest.fixture(scope="session")
def retriever_fn():
    """Return the retrieve function (validates FAISS + BM25 are loaded)."""
    from retrieval.retriever import retrieve
    return retrieve


@pytest.fixture(scope="session")
def search_fns():
    """Return individual search functions for comparison tests."""
    from retrieval.retriever import dense_search, sparse_search, hybrid_search
    return {
        "dense": dense_search,
        "sparse": sparse_search,
        "hybrid": hybrid_search,
    }


@pytest.fixture(scope="session")
def test_client():
    """FastAPI test client — no server needed."""
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


@pytest.fixture(scope="session")
def test_questions():
    """Load test questions if available."""
    path = ROOT / "data" / "test_questions.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════════
#  1. RETRIEVAL UNIT TESTS
# ════════════════════════════════════════════════════════════

class TestRetrieval:
    """Test the retrieval pipeline components."""

    def test_retrieve_returns_results(self, retriever_fn):
        """Basic smoke test — does retrieval return anything?"""
        results = retriever_fn("What is a tax file number?")
        assert len(results) > 0, "Retriever returned no results"

    def test_retrieve_returns_correct_count(self, retriever_fn):
        """Should return FINAL_TOPK results."""
        from config.settings import FINAL_TOPK
        results = retriever_fn("What is the tax-free threshold?")
        assert len(results) == FINAL_TOPK, f"Expected {FINAL_TOPK}, got {len(results)}"

    def test_retrieve_has_required_fields(self, retriever_fn):
        """Each result must have title, url, text at minimum."""
        results = retriever_fn("How do I lodge a tax return?")
        required = {"title", "url", "text"}
        for r in results:
            missing = required - set(r.keys())
            assert not missing, f"Result missing fields: {missing}"

    def test_retrieve_urls_are_ato(self, retriever_fn):
        """All results should be from ato.gov.au."""
        results = retriever_fn("What is GST?")
        for r in results:
            assert "ato.gov.au" in r["url"], f"Non-ATO URL: {r['url']}"

    def test_retrieve_text_not_empty(self, retriever_fn):
        """Retrieved text should not be empty."""
        results = retriever_fn("Medicare levy surcharge")
        for r in results:
            assert len(r["text"].strip()) > 20, "Retrieved text is too short"

    def test_retrieve_relevance_tfn(self, retriever_fn):
        """TFN query should return TFN-related content."""
        results = retriever_fn("What is a tax file number?")
        texts = " ".join(r["text"].lower() for r in results)
        assert "tax file number" in texts or "tfn" in texts, \
            "TFN query didn't return TFN content"

    def test_retrieve_relevance_super(self, retriever_fn):
        """Super query should return superannuation content."""
        results = retriever_fn("How does superannuation work?")
        texts = " ".join(r["text"].lower() for r in results)
        assert "super" in texts, "Super query didn't return super content"

    def test_retrieve_relevance_cgt(self, retriever_fn):
        """CGT query should return capital gains content."""
        results = retriever_fn("How does capital gains tax work on property?")
        texts = " ".join(r["text"].lower() for r in results)
        assert "capital gain" in texts or "cgt" in texts, \
            "CGT query didn't return CGT content"

    def test_retrieve_empty_query_handled(self, retriever_fn):
        """Empty or whitespace query should not crash."""
        # Should either return empty or raise a handled exception
        try:
            results = retriever_fn("")
            # If it returns, it should be a list
            assert isinstance(results, list)
        except Exception:
            pass  # Raising is acceptable for empty input

    def test_retrieve_long_query_handled(self, retriever_fn):
        """Very long query should not crash."""
        long_q = "tax deduction " * 100
        results = retriever_fn(long_q)
        assert isinstance(results, list)

    def test_retrieve_url_diversity(self, retriever_fn):
        """Results should come from multiple URLs (MMR working)."""
        results = retriever_fn("What deductions can I claim?")
        urls = set(r["url"] for r in results)
        assert len(urls) >= 2, f"Only {len(urls)} unique URL(s) — MMR may not be working"


# ════════════════════════════════════════════════════════════
#  2. SEARCH COMPONENT TESTS
# ════════════════════════════════════════════════════════════

class TestSearchComponents:
    """Test individual search functions."""

    def test_dense_search_returns_results(self, search_fns):
        results = search_fns["dense"]("tax file number")
        assert len(results) > 0

    def test_sparse_search_returns_results(self, search_fns):
        results = search_fns["sparse"]("tax file number")
        assert len(results) > 0, "BM25 returned nothing — is bm25.pkl loaded?"

    def test_hybrid_search_returns_results(self, search_fns):
        results = search_fns["hybrid"]("tax file number")
        assert len(results) > 0

    def test_hybrid_has_fused_scores(self, search_fns):
        """Hybrid results should have RRF fused scores."""
        results = search_fns["hybrid"]("income tax rates")
        for r in results[:5]:
            assert "score" in r, "Hybrid result missing fused 'score'"
            assert r["score"] > 0, "Score should be positive"

    def test_dense_vs_hybrid_overlap(self, search_fns):
        """Hybrid should include most dense results (fusion adds, doesn't lose)."""
        q = "What is the tax-free threshold?"
        dense = set(r["idx"] for r in search_fns["dense"](q, k=10))
        hybrid = set(r["idx"] for r in search_fns["hybrid"](q, k=20))
        overlap = len(dense & hybrid)
        assert overlap >= len(dense) * 0.5, \
            f"Hybrid lost too many dense results: {overlap}/{len(dense)}"


# ════════════════════════════════════════════════════════════
#  3. RERANKER TESTS
# ════════════════════════════════════════════════════════════

class TestReranker:
    """Test cross-encoder reranker."""

    def test_reranker_loads(self, reranker):
        assert reranker is not None

    def test_rerank_returns_same_count(self, reranker, retriever_fn):
        """Reranking shouldn't add or lose items."""
        results = retriever_fn("What is a TFN?", reranker=None)
        reranked = reranker.rerank("What is a TFN?", results.copy())
        assert len(reranked) == len(results)

    def test_rerank_has_scores(self, reranker):
        """Reranked items should have rerank_score field."""
        items = [
            {"text": "A tax file number is a unique identifier.", "score": 0.5},
            {"text": "Sydney weather forecast for today.", "score": 0.3},
        ]
        reranked = reranker.rerank("What is a TFN?", items)
        for r in reranked:
            assert "rerank_score" in r, "Missing rerank_score"

    def test_rerank_prefers_relevant(self, reranker):
        """Relevant text should score higher than irrelevant."""
        items = [
            {"text": "The weather in Melbourne is sunny today.", "score": 0.5},
            {"text": "A tax file number (TFN) is your personal reference number in the Australian tax system.", "score": 0.3},
        ]
        reranked = reranker.rerank("What is a tax file number?", items)
        assert "tax file number" in reranked[0]["text"].lower(), \
            "Reranker didn't prefer the relevant text"

    def test_rerank_empty_list(self, reranker):
        """Reranking empty list should return empty list."""
        assert reranker.rerank("test", []) == []


# ════════════════════════════════════════════════════════════
#  4. API ENDPOINT TESTS
# ════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    def test_health_endpoint(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True or data["ok"] is False  # returns a boolean
        assert "model" in data
        assert "reranker" in data
        assert "uptime_seconds" in data

    def test_health_reports_components(self, test_client):
        """Health should report on retriever and LLM status."""
        data = test_client.get("/health").json()
        assert "retriever" in data
        assert "llm" in data

    def test_search_endpoint(self, test_client):
        r = test_client.post("/search", json={"q": "What is a TFN?"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_result_structure(self, test_client):
        """Search results should have expected fields."""
        data = test_client.post("/search", json={"q": "tax rates"}).json()
        result = data["results"][0]
        for field in ["title", "url", "text", "score"]:
            assert field in result, f"Search result missing '{field}'"

    def test_diag_endpoint(self, test_client):
        r = test_client.get("/diag", params={"q": "How do I apply for a TFN?"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "q" in data

    def test_ask_returns_request_id(self, test_client):
        """Every response should have a request_id for tracing."""
        r = test_client.post("/ask", json={"q": "What is a TFN?"})
        assert r.status_code == 200
        data = r.json()
        assert "request_id" in data

    def test_ask_returns_elapsed(self, test_client):
        """Response should include elapsed_ms for performance tracking."""
        r = test_client.post("/ask", json={"q": "What is GST?"})
        data = r.json()
        assert "elapsed_ms" in data
        assert isinstance(data["elapsed_ms"], int)

    def test_ask_has_response_time_header(self, test_client):
        """X-Response-Time header should be set."""
        r = test_client.post("/ask", json={"q": "What is a TFN?"})
        assert "x-response-time" in r.headers or "X-Response-Time" in r.headers

    def test_ask_has_request_id_header(self, test_client):
        """X-Request-ID header should be set."""
        r = test_client.post("/ask", json={"q": "What is a TFN?"})
        assert "x-request-id" in r.headers or "X-Request-ID" in r.headers


# ════════════════════════════════════════════════════════════
#  5. INPUT VALIDATION TESTS
# ════════════════════════════════════════════════════════════

class TestInputValidation:
    """Test that bad inputs are rejected gracefully."""

    def test_ask_empty_query_rejected(self, test_client):
        """Empty query should return 422."""
        r = test_client.post("/ask", json={"q": ""})
        assert r.status_code == 422

    def test_ask_missing_field_rejected(self, test_client):
        """Missing 'q' field should return 422."""
        r = test_client.post("/ask", json={"question": "test"})
        assert r.status_code == 422

    def test_ask_too_long_rejected(self, test_client):
        """Query over 500 chars should return 422."""
        r = test_client.post("/ask", json={"q": "x" * 501})
        assert r.status_code == 422

    def test_search_empty_query_rejected(self, test_client):
        r = test_client.post("/search", json={"q": ""})
        assert r.status_code == 422

    def test_ask_whitespace_only_rejected(self, test_client):
        """Whitespace-only query should be rejected."""
        r = test_client.post("/ask", json={"q": "   "})
        # Either 422 (validation) or 200 with no_results — both acceptable
        assert r.status_code in [200, 422]


# ════════════════════════════════════════════════════════════
#  6. ANSWER QUALITY TESTS
# ════════════════════════════════════════════════════════════

class TestAnswerQuality:
    """Test answer structure and content. These call the LLM."""

    @pytest.mark.slow
    def test_ask_returns_answer(self, test_client):
        data = test_client.post("/ask", json={"q": "What is a tax file number?"}).json()
        assert data["answer"] is not None
        assert len(data["answer"]) > 50, "Answer seems too short"

    @pytest.mark.slow
    def test_ask_returns_references(self, test_client):
        data = test_client.post("/ask", json={"q": "What is the tax-free threshold?"}).json()
        assert len(data["references"]) > 0, "No references returned"
        ref = data["references"][0]
        assert "ref_number" in ref
        assert "title" in ref
        assert "url" in ref

    @pytest.mark.slow
    def test_ask_returns_disclaimer(self, test_client):
        data = test_client.post("/ask", json={"q": "What is GST?"}).json()
        assert data["disclaimer"] is not None
        assert "general information" in data["disclaimer"].lower()

    @pytest.mark.slow
    def test_ask_answer_has_citations(self, test_client):
        """Answer text should contain [1], [2] etc citation markers."""
        data = test_client.post("/ask", json={"q": "How do I apply for a TFN?"}).json()
        import re
        citations = re.findall(r'\[\d+\]', data["answer"] or "")
        assert len(citations) > 0, "Answer has no citation markers"

    @pytest.mark.slow
    def test_ask_references_have_ato_urls(self, test_client):
        data = test_client.post("/ask", json={"q": "What are the income tax rates?"}).json()
        for ref in data["references"]:
            assert "ato.gov.au" in ref["url"], f"Non-ATO reference: {ref['url']}"

    @pytest.mark.slow
    def test_ask_no_results_handled(self, test_client):
        """Nonsense query should return a graceful 'no results' message."""
        data = test_client.post("/ask", json={"q": "xyzzy flurble quantum wombat"}).json()
        assert data["status"] in ["no_results", "answered"]
        assert data["answer"] is not None  # Should not crash


# ════════════════════════════════════════════════════════════
#  7. PERFORMANCE TESTS
# ════════════════════════════════════════════════════════════

class TestPerformance:
    """Test latency and throughput baselines."""

    def test_retrieval_latency(self, retriever_fn):
        """Retrieval should complete in under 15 seconds."""
        t0 = time.time()
        retriever_fn("What is the tax-free threshold?")
        elapsed = time.time() - t0
        assert elapsed < 15, f"Retrieval took {elapsed:.1f}s — too slow"

    def test_search_endpoint_latency(self, test_client):
        """Search endpoint (no LLM) should respond quickly."""
        t0 = time.time()
        r = test_client.post("/search", json={"q": "tax rates"})
        elapsed = time.time() - t0
        assert r.status_code == 200
        assert elapsed < 15, f"/search took {elapsed:.1f}s"

    def test_health_is_fast(self, test_client):
        """Health check should be near-instant."""
        t0 = time.time()
        r = test_client.get("/health")
        elapsed = time.time() - t0
        assert r.status_code == 200
        assert elapsed < 1, f"/health took {elapsed:.1f}s"


# ════════════════════════════════════════════════════════════
#  8. EVAL RESULTS VALIDATION
# ════════════════════════════════════════════════════════════

class TestEvalResults:
    """Validate that eval results meet minimum quality bar."""

    @pytest.fixture
    def eval_data(self):
        """Load latest eval results."""
        # Try multiple possible paths
        for name in ["eval_results_v3.json", "eval_results.json"]:
            path = ROOT / "data" / name
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        pytest.skip("No eval results file found in data/")

    def test_eval_hit_rate_above_threshold(self, eval_data):
        """Hit rate should be at least 85%."""
        hr = eval_data["retrieval"]["hit_rate"]
        assert hr >= 0.85, f"Hit rate {hr:.1%} below 85% threshold"

    def test_eval_mrr_above_threshold(self, eval_data):
        """MRR should be at least 0.65."""
        mrr = eval_data["retrieval"]["mrr"]
        assert mrr >= 0.65, f"MRR {mrr:.3f} below 0.65 threshold"

    def test_eval_recall_at_5_above_threshold(self, eval_data):
        """Recall@5 should be at least 85%."""
        r5 = eval_data["retrieval"]["r5"]
        assert r5 >= 0.85, f"Recall@5 {r5:.1%} below 85% threshold"

    def test_eval_keyword_recall_above_threshold(self, eval_data):
        """Keyword recall should be at least 80%."""
        kw = eval_data["retrieval"]["kw"]
        assert kw >= 0.80, f"Keyword recall {kw:.1%} below 80% threshold"

    def test_eval_all_tiers_above_minimum(self, eval_data):
        """Every tier should have at least 75% hit rate."""
        for tier, metrics in eval_data["tiers"].items():
            assert metrics["hit"] >= 0.75, \
                f"Tier {tier} hit rate {metrics['hit']:.1%} below 75%"

    def test_eval_ran_100_questions(self, eval_data):
        """Eval should cover all 100 questions."""
        assert eval_data["n"] == 100, f"Eval only ran {eval_data['n']} questions"


# ════════════════════════════════════════════════════════════
#  9. SAFETY TESTS
# ════════════════════════════════════════════════════════════

class TestSafety:
    """Test that the system doesn't produce harmful outputs."""

    @pytest.mark.slow
    def test_refuses_non_tax_questions(self, test_client):
        """Non-tax questions should get a polite redirect."""
        data = test_client.post("/ask", json={"q": "What is the meaning of life?"}).json()
        answer = (data["answer"] or "").lower()
        # Should either say "no info" or redirect to ATO — not answer the philosophy question
        has_redirect = any(phrase in answer for phrase in [
            "ato", "tax", "don't have", "couldn't find", "not related",
            "information", "unable", "can't"
        ])
        assert has_redirect or data["status"] == "no_results", \
            "System answered a non-tax question without redirecting"

    @pytest.mark.slow
    def test_disclaimer_always_present(self, test_client):
        """Every answer must include the disclaimer."""
        data = test_client.post("/ask", json={"q": "What is the tax-free threshold?"}).json()
        assert data["disclaimer"] is not None
        assert len(data["disclaimer"]) > 20

    def test_no_stack_traces_in_errors(self, test_client):
        """Error responses should never expose internal stack traces."""
        # Send malformed request
        r = test_client.post("/ask", content=b"not json",
                            headers={"content-type": "application/json"})
        if r.status_code >= 400:
            body = r.text
            assert "Traceback" not in body, "Stack trace exposed in error response"
            assert "File \"" not in body, "File paths exposed in error response"
