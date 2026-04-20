"""
build_index.py — Build hybrid search index (FAISS dense + BM25 sparse).

Usage:
    python indexing/build_index.py
"""

import json, pickle, sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (DOCS_PATH, FAISS_INDEX_PATH, BM25_INDEX_PATH,
                              META_PATH, EMBED_MODEL, EMBED_FALLBACK, EMBED_BATCH_SIZE)
from indexing.bm25 import BM25Index


def load_docs():
    docs = []
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or ""
            if not text:
                continue
            docs.append({
                "title":           obj.get("title", ""),
                "url":             obj.get("url", ""),
                "section_heading": obj.get("section_heading", ""),
                "breadcrumb":      obj.get("breadcrumb", ""),
                "keywords":        obj.get("keywords", ""),
                "embed_text":      obj.get("embed_text") or text,
                "text":            text,
            })
    return docs


def main():
    docs = load_docs()
    if not docs:
        raise SystemExit(f"No docs loaded from {DOCS_PATH}")

    print(f"Loaded {len(docs)} docs from {DOCS_PATH}")

    # ━━━━ 1. Dense index (FAISS) ━━━━
    # Use MiniLM for speed on CPU — BGE is better but too slow without GPU
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Max sequence length: {model.max_seq_length} tokens")

    texts = [d["embed_text"] for d in docs]
    total = len(texts)
    embs = []
    print(f"  Encoding {total} chunks...")
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        v = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        embs.append(v)
        done = min(i + EMBED_BATCH_SIZE, total)
        pct = done * 100 // total
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        print(f"\r  [{bar}] {done}/{total} ({pct}%)", end="", flush=True)
    print()  # newline after progress bar
    X = np.vstack(embs)
    dim = X.shape[1]

    M, efC = 32, 200
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.add(X)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"\n  FAISS: {FAISS_INDEX_PATH} ({len(docs)} vectors, dim={dim})")

    # ━━━━ 2. BM25 index ━━━━
    print(f"\nBuilding BM25 index...")
    keyword_texts = [f"{d['keywords']} {d['title'].lower()}" for d in docs]
    bm25 = BM25Index()
    bm25.fit(keyword_texts)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25: {BM25_INDEX_PATH} ({len(bm25.vocab)} terms)")

    # ━━━━ 3. Metadata ━━━━
    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)
    print(f"  Meta: {META_PATH}")

    print(f"\nDone! {len(docs)} docs indexed (FAISS + BM25)")


if __name__ == "__main__":
    main()