"""
Hybrid retriever: FAISS dense + BM25 sparse + rerank + MMR.

Usage:
    python retrieval/retriever.py
"""

import pickle, sys
import faiss
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (FAISS_INDEX_PATH, META_PATH, BM25_INDEX_PATH,
                              EMBED_MODEL, EMBED_FALLBACK, RETRIEVER_TOPK,
                              RERANK_CAP, FINAL_TOPK, MAX_PER_URL, MMR_LAMBDA,
                              RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT)
from retrieval.reranker import Reranker


# ──────────────────── Load assets ────────────────────
for p in [FAISS_INDEX_PATH, META_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run build_index.py first.")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
USE_BGE = False

faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
META = pickle.load(open(META_PATH, "rb"))

bm25_index = None
if BM25_INDEX_PATH.exists():
    bm25_index = pickle.load(open(BM25_INDEX_PATH, "rb"))
    print(f"[retriever] Hybrid mode ({len(META)} docs)")
else:
    print(f"[retriever] Dense-only mode ({len(META)} docs)")


# ──────────────────── Search functions ────────────────────
def dense_search(query: str, k: int = 30) -> list:
    q_text = f"Represent this sentence: {query}" if USE_BGE else query
    qv = encoder.encode([q_text], normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(qv, k)
    results = []
    for idx, score in zip(I[0], D[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(META): continue
        m = META[idx]
        results.append({"idx": idx, "dense_score": float(score),
            "title": m.get("title",""), "url": m.get("url",""),
            "section_heading": m.get("section_heading",""),
            "breadcrumb": m.get("breadcrumb",""),
            "keywords": m.get("keywords",""), "text": m.get("text","")})
    return results


def sparse_search(query: str, k: int = 30) -> list:
    if not bm25_index: return []
    results = []
    for idx, score in bm25_index.search(query, top_k=k):
        if idx < 0 or idx >= len(META): continue
        m = META[idx]
        results.append({"idx": idx, "bm25_score": float(score),
            "title": m.get("title",""), "url": m.get("url",""),
            "section_heading": m.get("section_heading",""),
            "breadcrumb": m.get("breadcrumb",""),
            "keywords": m.get("keywords",""), "text": m.get("text","")})
    return results


def reciprocal_rank_fusion(dense_results, sparse_results,
                           k_rrf=RRF_K, dw=DENSE_WEIGHT, sw=SPARSE_WEIGHT):
    scores = defaultdict(float)
    items = {}
    for rank, item in enumerate(dense_results):
        idx = item["idx"]
        scores[idx] += dw * (1.0 / (k_rrf + rank + 1))
        items[idx] = item
    for rank, item in enumerate(sparse_results):
        idx = item["idx"]
        scores[idx] += sw * (1.0 / (k_rrf + rank + 1))
        if idx not in items: items[idx] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for idx, fused in ranked:
        item = items[idx].copy()
        item["score"] = fused
        results.append(item)
    return results


def hybrid_search(query: str, k: int = 30) -> list:
    dense = dense_search(query, k=k)
    sparse = sparse_search(query, k=k)
    if sparse:
        return reciprocal_rank_fusion(dense, sparse)
    for item in dense:
        item["score"] = item["dense_score"]
    return dense


# ──────────────────── MMR + grouping ────────────────────
def mmr_select(query: str, candidates: list, top_k: int = 5, lam: float = MMR_LAMBDA):
    if not candidates: return []
    texts = [c.get("text", "") for c in candidates]
    X = encoder.encode(texts, normalize_embeddings=True)
    rel = np.array([c.get("rerank_score", c.get("score", 0.0)) for c in candidates], dtype="float32")
    selected, remaining = [], list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            i = int(np.argmax(rel[remaining]))
            selected.append(remaining.pop(i))
            continue
        S = X[np.array(selected)]
        cur_max = np.array([float(np.max(S @ X[idx])) for idx in remaining], dtype="float32")
        rem_rel = np.array([rel[i] for i in remaining], dtype="float32")
        score = lam * rem_rel - (1 - lam) * cur_max
        i = int(np.argmax(score))
        selected.append(remaining.pop(i))

    return [candidates[i] for i in selected]


def group_by_url(candidates: list, max_per_url: int = MAX_PER_URL) -> list:
    buckets = defaultdict(list)
    for it in candidates:
        buckets[it["url"]].append(it)
    for u in buckets:
        buckets[u].sort(key=lambda x: x.get("score", 0.0), reverse=True)
    url_order = sorted(buckets.keys(), key=lambda u: buckets[u][0].get("score", 0.0), reverse=True)
    merged = []
    for url in url_order:
        merged.extend(buckets[url][:max_per_url])
    return merged


# ──────────────────── Score adjustments ────────────────────
import re as _re

# Occupation guide URLs — penalise for generic queries
_OCCUPATION_PATTERN = _re.compile(r'/guides-for-occupations-and-industries/', _re.I)

# Common occupation terms — if query mentions these, don't penalise
_OCCUPATION_TERMS = _re.compile(
    r'\b(nurse|doctor|teacher|driver|pilot|engineer|mechanic|builder|'
    r'electrician|plumber|police|army|adf|military|lawyer|accountant|'
    r'mining|farm|chef|hospitality|retail|security|guard|firefighter|'
    r'paramedic|dentist|surgeon|tradesperson|carpenter|painter)\b', _re.I)


def _apply_score_adjustments(query: str, candidates: list) -> list:
    """Boost title matches, penalise occupation guides for generic queries."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    is_occupation_query = bool(_OCCUPATION_TERMS.search(query))

    for c in candidates:
        boost = 0.0
        title = c.get("title", "").lower()
        url = c.get("url", "")

        # 1. TITLE BOOST: If query words appear in page title, boost score
        title_words = set(title.split())
        overlap = query_words & title_words
        if len(overlap) >= 2:
            boost += 0.003  # significant boost for title match
        elif len(overlap) == 1 and len(query_words) <= 4:
            boost += 0.001

        # 2. OCCUPATION PENALTY: Penalise occupation-specific guides for generic queries
        if _OCCUPATION_PATTERN.search(url) and not is_occupation_query:
            boost -= 0.004  # push occupation guides down for generic questions

        # 3. GENERAL PAGE BOOST: Prefer general deduction pages over niche ones
        general_paths = ['/deductions-you-can-claim/', '/work-related-deductions/',
                        '/working-from-home', '/clothing', '/self-education',
                        '/cars-transport', '/phone-internet']
        for gp in general_paths:
            if gp in url.lower():
                boost += 0.001
                break

        c["score"] = c.get("score", 0.0) + boost

    # Re-sort by adjusted score
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return candidates


# ──────────────────── Full pipeline ────────────────────
def retrieve(query: str, reranker=None) -> list:
    candidates = hybrid_search(query, k=RETRIEVER_TOPK)
    if not candidates: return []
    candidates = _apply_score_adjustments(query, candidates)
    candidates = group_by_url(candidates)
    if reranker:
        try:
            candidates = reranker.rerank(query, candidates)[:RERANK_CAP]
        except Exception as e:
            print(f"[WARN] Rerank failed: {e}")
    return mmr_select(query, candidates, top_k=FINAL_TOPK)


# ──────────────────── CLI test ────────────────────
if __name__ == "__main__":
    queries = [
        "What is the tax-free threshold?",
        "Can I claim self-education expenses?",
        "working holiday maker tax rate",
    ]
    reranker = None
    try:
        reranker = Reranker()
    except Exception as e:
        print(f"[WARN] No reranker: {e}\n")

    for q in queries:
        print(f"Q: {q}")
        for i, r in enumerate(retrieve(q, reranker=reranker)):
            print(f"  [{i+1}] {r['title'][:50]} — {r.get('section_heading','')[:30]}")
            print(f"      {r['text'][:100]}...")
        print()