"""
prep_docs.py — Process chunks.jsonl into docs.jsonl for indexing.

Builds embed_text (heading prefix + content) and keyword string for BM25.

Usage:
    python processing/prep_docs.py
"""

import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CHUNKS_PATH, DOCS_PATH


def validate_chunk(doc: dict) -> bool:
    text = doc.get("text", "")
    if not text or len(text.strip()) < 80:
        return False
    boilerplate = ["Skip to main content", "BEGIN NOINDEX", "Report webpage issue"]
    clean = text
    for bp in boilerplate:
        clean = clean.replace(bp, "")
    if len(clean.strip()) < 60:
        return False
    return True


def build_embed_text(doc: dict) -> str:
    """Prepend heading context so embedding captures what this chunk is about."""
    prefix = doc.get("heading_prefix", doc.get("title", ""))
    text = doc.get("text", "")
    if prefix:
        return f"{prefix}\n\n{text}"
    return text


def build_keyword_string(doc: dict) -> str:
    """Combine hub keywords + URL keywords + section heading for BM25."""
    keywords = doc.get("keywords", [])
    heading = doc.get("section_heading", "")
    if heading:
        stop = {"and", "or", "the", "for", "to", "of", "in", "an", "a", "is",
                "you", "your", "how", "what", "when", "if", "on", "at", "by"}
        words = heading.lower().replace("-", " ").split()
        keywords = list(keywords) + [w for w in words if w not in stop and len(w) > 1]
    keywords = list(dict.fromkeys(keywords))
    return " ".join(keywords)


def run():
    if not CHUNKS_PATH.exists():
        print(f"[ERROR] Not found: {CHUNKS_PATH}")
        sys.exit(1)

    n_in, n_out, n_skipped = 0, 0, 0

    with CHUNKS_PATH.open("r", encoding="utf-8") as f, \
         DOCS_PATH.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            obj = json.loads(line)

            if not validate_chunk(obj):
                n_skipped += 1
                continue

            doc = {
                "id":              obj.get("id", ""),
                "title":           obj.get("title", ""),
                "url":             obj.get("url", ""),
                "section_heading": obj.get("section_heading", ""),
                "breadcrumb":      obj.get("breadcrumb", ""),
                "keywords":        build_keyword_string(obj),
                "embed_text":      build_embed_text(obj),
                "text":            obj.get("text", ""),
                "chunk_index":     obj.get("chunk_index", 0),
                "chunk_total":     obj.get("chunk_total", 1),
            }

            if not doc["id"] or not doc["text"]:
                n_skipped += 1
                continue

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"prep_docs: {n_in} input → {n_out} output ({n_skipped} skipped)")
    print(f"  → {DOCS_PATH}")


if __name__ == "__main__":
    run()
