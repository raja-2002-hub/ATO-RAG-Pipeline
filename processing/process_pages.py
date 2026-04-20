#!/usr/bin/env python3
"""
process_pages.py — Process saved HTML into chunks for RAG.

Reads HTML files saved by fetch_pages.py and produces chunks.jsonl.
All processing is offline — no network needed.

Hub detection: Uses ATO's MasterCardNavigation_card__title class.
  - Has card title elements → HUB (extract headings as keywords, don't index)
  - No card title elements → LEAF (extract content by H2/H3, chunk, index)

Usage:
    python process_pages.py --html-dir raw_html/html --inventory raw_html/inventory.csv --debug
"""

import argparse, csv, hashlib, json, re, unicodedata
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# ──────────────────── Text helpers ────────────────────
BOILERPLATE = [
    "BEGIN NOINDEX", "END NOINDEX",
    "Skip to main content", "Skip To search",
    "Skip To Alex Virtual Assistant",
    "Report webpage issue", "Print or Download",
]
NOISE_HEADINGS = {
    "on this page", "tools", "tax information for", "help and support",
    "our commitment to you", "copyright notice", "log in to ato online services",
    "individuals", "business", "agents", "non-residents", "access manager",
    "foreign investor",
}
QC_PATTERN = re.compile(r"\bQC\s*\d+\b")
LAST_UPDATED = re.compile(r"Last updated\s*\d+\s*\w+\s*\d{4}")
STOP_WORDS = {
    "and", "or", "the", "for", "to", "of", "in", "an", "a", "is",
    "on", "at", "by", "from", "as", "if", "you", "your", "how",
    "what", "when", "it", "are", "be", "do", "has", "have", "this",
    "that", "with", "not", "can", "may", "will", "about", "more",
    "our", "we", "us", "its",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()


def clean_text(text: str) -> str:
    for bp in BOILERPLATE:
        text = text.replace(bp, "")
    text = QC_PATTERN.sub("", text)
    text = LAST_UPDATED.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return norm(text)


def real_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


def extract_keywords(text: str) -> List[str]:
    words = re.sub(r"[^a-zA-Z0-9\s-]", " ", text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


# ──────────────────── URL helpers ────────────────────
def url_to_breadcrumb(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return ""
    segments = path.split("/")
    return " > ".join(seg.replace("-", " ").title() for seg in segments)


def url_to_keywords(url: str) -> List[str]:
    path = urlparse(url).path.strip("/")
    if not path:
        return []
    keywords = []
    for seg in path.split("/"):
        keywords.extend(extract_keywords(seg.replace("-", " ")))
    return list(dict.fromkeys(keywords))


# ──────────────────── Hub detection ────────────────────
def detect_hub_page(soup: BeautifulSoup) -> Tuple[bool, List[str]]:
    """
    Hub detection using ATO's MasterCardNavigation_card__title class.

    IMPORTANT: Do NOT check for MasterCardNavigation_nav-cards — that class
    appears in embedded <style> blocks on ALL pages (including content pages).

    Only MasterCardNavigation_card__title exists as actual DOM elements
    exclusively on real hub/navigation pages.
    """
    card_titles = soup.select("[class*='MasterCardNavigation_card__title']")

    if len(card_titles) < 3:
        return False, []

    card_headings = []
    for title_el in card_titles:
        text = norm(title_el.get_text())
        if text and len(text) < 150:
            card_headings.append(text)

    return True, list(dict.fromkeys(card_headings))


# ──────────────────── Content extraction ────────────────────
def extract_sections(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract content organized by H2/H3 headings.
    Removes: nav cards, "On this page" sidebar, boilerplate, footer widgets.
    Keeps: paragraphs, list items, table cells, definition lists.
    """
    main = (soup.select_one("main")
            or soup.select_one("[role='main']")
            or soup.select_one("article"))
    root = main if main else (soup.body or soup)

    # Remove noise elements BEFORE extraction
    for tag in root.select("script, style, noscript, svg, img, picture, video, iframe"):
        tag.decompose()
    for sel in [
        "header", "footer", "nav", "[role='navigation']",
        "[class*='RclOnThisPage']",
        "[class*='MasterCardNavigation']",
        "[class*='breadcrumb']", "[class*='Breadcrumb']", "[class*='RclBreadcrumb']",
        "[class*='social-share']", "[class*='share']",
        "[class*='toolbar']", "[class*='sidebar']", "[class*='RclSideMenuContainer']",
        "[class*='cookie']", "#cookie",
        "[class*='LoginSlider']", "[class*='RclLoginModal']",  # Login boxes
        "[class*='RclFooter']",                                 # Footer
        "[class*='RclHeader']",                                 # Header
        "[class*='RclSkipToContent']",                          # Skip links
        "[class*='PrintAndDownload']",                          # Print/download buttons
        "[class*='QuickCode']",                                 # QC code elements
        "[class*='commitment']", "[class*='Commitment']",       # "Our commitment" footer
        "[class*='copyright']", "[class*='Copyright']",         # Copyright notice
    ]:
        for t in root.select(sel):
            t.decompose()

    sections = []
    current_heading = ""
    current_parts = []

    def flush():
        if current_parts:
            text = clean_text("\n".join(current_parts))
            if text and len(text) > 50:
                sections.append({"heading": current_heading, "content": text})

    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th",
                              "dt", "dd", "blockquote", "pre", "figcaption"]):
        if el.name in ("h1", "h2", "h3", "h4"):
            heading_text = norm(el.get_text())
            if heading_text.lower() in NOISE_HEADINGS:
                continue
            flush()
            current_heading = heading_text
            current_parts = []
        else:
            text = norm(el.get_text())
            if text and len(text) > 10:
                current_parts.append(text)

    flush()

    if not sections:
        all_text = clean_text(root.get_text(separator="\n"))
        if all_text and len(all_text) > 50:
            sections.append({"heading": "", "content": all_text})

    return sections


# ──────────────────── Chunking ────────────────────
def chunk_sections(
    sections: List[Dict[str, str]],
    max_tokens: int = 384,
    min_tokens: int = 60,
) -> List[Dict[str, str]]:
    """
    Convert sections into right-sized chunks for embedding model.
    Small sections get merged; large sections get split at sentence boundaries.
    """
    chunks = []

    # Pass 1: merge tiny sections
    merged = []
    buf_heading, buf_text, buf_tokens = "", "", 0

    for sec in sections:
        sec_tokens = real_token_count(sec["content"])
        if buf_tokens + sec_tokens <= max_tokens:
            if not buf_heading and sec["heading"]:
                buf_heading = sec["heading"]
            buf_text += ("\n\n" if buf_text else "") + sec["content"]
            buf_tokens += sec_tokens
        else:
            if buf_text:
                merged.append({"heading": buf_heading, "content": buf_text})
            buf_heading = sec["heading"]
            buf_text = sec["content"]
            buf_tokens = sec_tokens

    if buf_text:
        merged.append({"heading": buf_heading, "content": buf_text})

    # Pass 2: split oversized at sentence boundaries
    for sec in merged:
        tokens = real_token_count(sec["content"])
        if tokens <= max_tokens:
            if tokens >= min_tokens:
                chunks.append(sec)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', sec["content"])
            cur, cur_tok = "", 0
            for sent in sentences:
                st = real_token_count(sent)
                if cur_tok + st > max_tokens and cur:
                    if real_token_count(cur) >= min_tokens:
                        chunks.append({"heading": sec["heading"], "content": cur.strip()})
                    cur, cur_tok = sent, st
                else:
                    cur += (" " if cur else "") + sent
                    cur_tok += st
            if cur and real_token_count(cur) >= min_tokens:
                chunks.append({"heading": sec["heading"], "content": cur.strip()})

    return chunks


# ──────────────────── Main processing ────────────────────
def process(args):
    html_dir = Path(args.html_dir)
    inv_path = Path(args.inventory)
    out_chunks = Path(args.out_chunks)
    out_hubs = Path(args.out_hubs)

    if not inv_path.exists():
        print(f"[ERROR] Inventory not found: {inv_path}")
        return

    pages = []
    with inv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename"):
                pages.append(row)

    print(f"[1/4] Loaded {len(pages)} pages from inventory\n")

    # ── PASS 1: Classify all pages ──
    print("[2/4] Classifying pages (hub vs leaf)...")
    classifications = {}

    for i, page in enumerate(pages):
        url = page["url"]
        filepath = html_dir / page["filename"]
        if not filepath.exists():
            continue

        try:
            html = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        is_hub, card_headings = detect_hub_page(soup)

        classifications[url] = {
            "is_hub": is_hub,
            "card_headings": card_headings,
            "title": page.get("title", ""),
            "filename": page.get("filename", ""),
        }

        if args.debug and (i + 1) % 200 == 0:
            hubs = sum(1 for c in classifications.values() if c["is_hub"])
            print(f"  Classified {i+1}/{len(pages)} — {hubs} hubs, {i+1-hubs} leaves")

    hub_count = sum(1 for c in classifications.values() if c["is_hub"])
    leaf_count = len(classifications) - hub_count
    print(f"  Result: {hub_count} hubs, {leaf_count} leaves\n")

    # ── PASS 2: Build keyword map from hub card headings ──
    print("[3/4] Building keyword map from hub card headings...")

    hub_keywords_map = {}

    for url in classifications:
        accumulated = []
        path = urlparse(url).path.rstrip("/")
        parts = path.split("/")

        # Walk up the URL path to find parent hubs
        # But only take the card heading that MATCHES this page's branch
        # not ALL card headings from the parent hub
        for depth in range(len(parts) - 1, 0, -1):  # from deepest parent upward
            parent_path = "/".join(parts[:depth])
            parent_url = f"https://www.ato.gov.au{parent_path}"

            if parent_url not in classifications:
                continue
            if not classifications[parent_url]["is_hub"]:
                continue

            # Find which card heading matches this page's path
            child_segment = parts[depth] if depth < len(parts) else ""
            for heading in classifications[parent_url]["card_headings"]:
                # Check if this card heading relates to the child's URL segment
                heading_words = set(extract_keywords(heading))
                segment_words = set(extract_keywords(child_segment.replace("-", " ")))
                # If there's word overlap, this card is relevant to this branch
                if heading_words & segment_words:
                    accumulated.extend(extract_keywords(heading))

            # Only go up 2 levels max to keep keywords specific
            if len(accumulated) >= 3:
                break

        # Also add URL-based keywords (always specific to this page)
        accumulated.extend(url_to_keywords(url))

        # Dedupe
        hub_keywords_map[url] = list(dict.fromkeys(accumulated))

    print(f"  Keywords mapped for {len(hub_keywords_map)} URLs\n")

    # ── PASS 3: Process leaf pages into chunks ──
    print("[4/4] Extracting content from leaf pages...")

    chunk_f = out_chunks.open("w", encoding="utf-8")
    hub_f = out_hubs.open("w", encoding="utf-8")
    total_chunks = 0
    processed_leaves = 0

    for i, page in enumerate(pages):
        url = page["url"]
        if url not in classifications:
            continue

        cls = classifications[url]

        if cls["is_hub"]:
            hub_f.write(json.dumps({
                "url": url,
                "title": cls["title"],
                "card_headings": cls["card_headings"],
                "keywords_passed_down": hub_keywords_map.get(url, []),
            }, ensure_ascii=False) + "\n")
            continue

        # ── LEAF PAGE ──
        filepath = html_dir / cls["filename"]
        if not filepath.exists():
            continue

        try:
            html = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        sections = extract_sections(soup)
        if not sections:
            continue

        page_title = re.sub(r"\s*\|.*$", "", cls["title"]).strip()
        page_chunks = chunk_sections(sections, max_tokens=args.chunk_max_tokens)

        if not page_chunks:
            continue

        breadcrumb = url_to_breadcrumb(url)
        keywords = hub_keywords_map.get(url, [])
        content_hash = hashlib.sha1(url.encode()).hexdigest()

        for j, chunk in enumerate(page_chunks):
            heading_prefix = page_title
            if chunk["heading"] and chunk["heading"] != page_title:
                heading_prefix += f" — {chunk['heading']}"

            record = {
                "id": f"{content_hash[:10]}_{j+1:03d}",
                "url": url,
                "title": page_title,
                "section_heading": chunk["heading"],
                "breadcrumb": breadcrumb,
                "keywords": keywords,
                "heading_prefix": heading_prefix,
                "text": chunk["content"],
                "chunk_index": j,
                "chunk_total": len(page_chunks),
                "approx_tokens": real_token_count(chunk["content"]),
            }
            chunk_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chunks += 1

        processed_leaves += 1

        if args.debug and processed_leaves % 100 == 0:
            print(f"  Processed {processed_leaves} leaves — {total_chunks} chunks")

    chunk_f.close()
    hub_f.close()

    print(f"\nDone!")
    print(f"  Hub pages (keywords only) : {hub_count}")
    print(f"  Leaf pages processed      : {processed_leaves}")
    print(f"  Chunks written            : {total_chunks}")
    print(f"\n  Chunks file : {out_chunks}")
    print(f"  Hubs file   : {out_hubs}")
    print(f"\n  Next: python prep_docs.py --src {out_chunks} --dst docs.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-dir", default="raw_html/html")
    ap.add_argument("--inventory", default="raw_html/inventory.csv")
    ap.add_argument("--out-chunks", default="chunks.jsonl")
    ap.add_argument("--out-hubs", default="hubs.jsonl")
    ap.add_argument("--chunk-max-tokens", type=int, default=384)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    process(args)