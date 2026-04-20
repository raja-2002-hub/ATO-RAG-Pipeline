#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATO Smart Crawler — hub-aware, heading-structured, keyword-accumulating.

Hub pages  → extract card headings as keywords, don't index content.
Leaf pages → extract content by H2/H3 sections, attach accumulated keywords.

Usage:
    python crawl_ato.py --out kb_out --max-pages 600 --debug
    python crawl_ato.py --out kb_out --max-pages 600 --max-depth 2 --debug
"""

import argparse, asyncio, csv, hashlib, json, random, re, time, unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib import robotparser
from playwright.async_api import async_playwright

# ──────────────────── Config ────────────────────
SITEMAP_URL  = "https://www.ato.gov.au/sitemap.xml"
ALLOWED_HOST = "www.ato.gov.au"
DEFAULT_PREFIX = "/"  # Crawl entire ato.gov.au
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Boilerplate patterns to strip from extracted text
BOILERPLATE = [
    "BEGIN NOINDEX", "END NOINDEX",
    "Skip to main content", "Skip To search",
    "Skip To Alex Virtual Assistant",
    "Report webpage issue", "Print or Download",
]
QC_PATTERN = re.compile(r"\bQC\s*\d+\b")
LAST_UPDATED_PATTERN = re.compile(r"Last updated \d+ \w+ \d{4}")


# ──────────────────── Text helpers ────────────────────
def norm(s: str) -> str:
    """Normalize whitespace and unicode."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()


def clean_text(text: str) -> str:
    """Remove ATO boilerplate from extracted text."""
    for bp in BOILERPLATE:
        text = text.replace(bp, "")
    text = QC_PATTERN.sub("", text)
    text = LAST_UPDATED_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return norm(text)


def real_token_count(text: str) -> int:
    """Approximate token count — ~1.3 tokens per word for English."""
    words = text.split()
    return int(len(words) * 1.3)


# ──────────────────── URL helpers ────────────────────
def to_https(u: str) -> str:
    p = urlparse(u)
    p = p._replace(scheme="https", params="", query="", fragment="")
    return urlunparse(p)


def under_prefix(u: str, prefix: str) -> bool:
    path = urlparse(u).path.rstrip("/")
    return path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")


def should_skip(u: str) -> bool:
    low = u.lower()
    if low.startswith(("mailto:", "tel:")):
        return True
    if any(tok in low for tok in ("/share/", "/print/", "/download/", "/search?")):
        return True
    if re.search(r"[?&](page|sort|order|view|start|from|to|size|limit)=", low):
        return True
    return False


def url_to_breadcrumb(url: str, prefix: str) -> str:
    """Extract breadcrumb path from URL structure."""
    path = urlparse(url).path
    path = path.replace(prefix, "").strip("/")
    if not path:
        return ""
    segments = path.split("/")
    return " > ".join(seg.replace("-", " ").title() for seg in segments)


def url_to_keywords(url: str, prefix: str) -> List[str]:
    """Extract keywords from URL path segments."""
    path = urlparse(url).path
    path = path.replace(prefix, "").strip("/")
    if not path:
        return []
    stop_words = {"and", "or", "the", "for", "to", "of", "in", "an", "a", "is", "on", "at", "by", "from", "as", "if", "you", "your"}
    keywords = []
    for seg in path.split("/"):
        words = seg.replace("-", " ").split()
        keywords.extend(w.lower() for w in words if w.lower() not in stop_words and len(w) > 1)
    return list(dict.fromkeys(keywords))  # dedupe preserving order


def branch_key(u: str) -> str:
    path = urlparse(u).path.rstrip("/")
    parts = path.split("/")
    return "/".join(parts[:4]) if len(parts) >= 4 else path


# ──────────────────── Page classification ────────────────────
def detect_hub_page(soup: BeautifulSoup) -> Tuple[bool, List[str]]:
    """
    Detect if a page is a hub (navigation index) page.
    Returns (is_hub, card_headings).

    Hub detection: ATO hub pages have card grids with headings
    linking to child pages. They contain little to no prose content.
    """
    main = soup.select_one("main") or soup.select_one("[role='main']") or soup.select_one("article") or soup.body or soup

    # Look for ATO card patterns — cards are typically in <div> with heading + short description
    # ATO uses various card patterns: <h2>/<h3> inside card containers with links
    card_headings = []

    # Pattern 1: Cards with h2/h3 headings inside link containers
    for card in main.select(".card, .card-item, [class*='card'], [class*='Card']"):
        heading = card.select_one("h2, h3, h4, [class*='heading'], [class*='title']")
        if heading:
            text = norm(heading.get_text())
            if text and len(text) < 100:
                card_headings.append(text)

    # Pattern 2: Sections with h2/h3 that are link-heavy (more links than paragraphs)
    if not card_headings:
        for section in main.select("section, .content-area, [class*='content']"):
            links = section.select("a")
            paras = section.select("p")
            headings = section.select("h2, h3")
            if len(links) > 3 and len(links) > len(paras) * 2:
                for h in headings:
                    text = norm(h.get_text())
                    if text and len(text) < 100:
                        card_headings.append(text)

    # Pattern 3: Direct h3 links (common ATO pattern — h3 > a)
    if not card_headings:
        h3_links = main.select("h3 > a, h2 > a")
        if len(h3_links) >= 3:
            for hl in h3_links:
                text = norm(hl.get_text())
                if text and len(text) < 100:
                    card_headings.append(text)

    # Fallback: check if page has very little prose relative to links
    if not card_headings:
        all_text = norm(main.get_text())
        all_links = main.select("a[href]")
        all_paras = main.select("p")
        # High link-to-paragraph ratio suggests a hub page
        if len(all_links) > 6 and len(all_paras) <= 3 and len(all_text) < 2000:
            # Extract heading-like text from prominent links
            for a in all_links:
                parent = a.parent
                if parent and parent.name in ("h2", "h3", "h4"):
                    text = norm(a.get_text())
                    if text and len(text) < 100:
                        card_headings.append(text)

    # A page is a hub if it has 3+ card-style headings
    is_hub = len(card_headings) >= 3
    return is_hub, card_headings


# ──────────────────── Content extraction ────────────────────
def extract_sections(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract content organized by H2/H3 headings.
    Returns list of {"heading": ..., "content": ...} dicts.
    """
    main = soup.select_one("main") or soup.select_one("[role='main']") or soup.select_one("article")
    root = main if main else (soup.body or soup)

    # Remove noise elements
    for tag in root.select("script, style, noscript, svg, img, picture, video, iframe"):
        tag.decompose()
    for sel in ["header", "footer", "nav", ".breadcrumbs", ".breadcrumb",
                ".cookie", "#cookie", ".social-share", ".share", ".toolbar",
                ".sidebar", "[role='navigation']"]:
        for t in root.select(sel):
            t.decompose()

    sections = []
    current_heading = ""
    current_parts = []

    def flush():
        if current_parts:
            text = clean_text("\n".join(current_parts))
            if text and len(text) > 50:
                sections.append({
                    "heading": current_heading,
                    "content": text,
                })

    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th",
                              "dt", "dd", "blockquote", "pre", "figcaption"]):
        if el.name in ("h1", "h2", "h3", "h4"):
            flush()
            current_heading = norm(el.get_text())
            current_parts = []
        else:
            text = norm(el.get_text())
            if text and len(text) > 5:
                current_parts.append(text)

    flush()

    # If no headings found at all, treat the whole page as one section
    if not sections:
        all_text = clean_text(root.get_text(separator="\n"))
        if all_text and len(all_text) > 50:
            sections.append({"heading": "", "content": all_text})

    return sections


def chunk_sections(
    sections: List[Dict[str, str]],
    page_title: str,
    max_tokens: int = 384,
    min_tokens: int = 60,
) -> List[Dict[str, str]]:
    """
    Convert sections into right-sized chunks for embedding.
    Small sections get merged; large sections get split at sentence boundaries.
    Each chunk carries its heading context.
    """
    chunks = []

    # First pass: merge tiny sections together
    merged = []
    buffer_heading = ""
    buffer_text = ""
    buffer_tokens = 0

    for sec in sections:
        sec_tokens = real_token_count(sec["content"])

        if buffer_tokens + sec_tokens <= max_tokens:
            # Merge into buffer
            if not buffer_heading and sec["heading"]:
                buffer_heading = sec["heading"]
            elif sec["heading"] and buffer_heading:
                buffer_heading = buffer_heading  # keep first heading
            buffer_text += ("\n\n" if buffer_text else "") + sec["content"]
            buffer_tokens += sec_tokens
        else:
            # Flush buffer
            if buffer_text:
                merged.append({"heading": buffer_heading, "content": buffer_text})
            buffer_heading = sec["heading"]
            buffer_text = sec["content"]
            buffer_tokens = sec_tokens

    if buffer_text:
        merged.append({"heading": buffer_heading, "content": buffer_text})

    # Second pass: split oversized chunks at sentence boundaries
    for sec in merged:
        tokens = real_token_count(sec["content"])
        if tokens <= max_tokens:
            if tokens >= min_tokens:
                chunks.append(sec)
        else:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', sec["content"])
            cur_chunk = ""
            cur_tokens = 0

            for sent in sentences:
                sent_tokens = real_token_count(sent)
                if cur_tokens + sent_tokens > max_tokens and cur_chunk:
                    if real_token_count(cur_chunk) >= min_tokens:
                        chunks.append({"heading": sec["heading"], "content": cur_chunk.strip()})
                    cur_chunk = sent
                    cur_tokens = sent_tokens
                else:
                    cur_chunk += (" " if cur_chunk else "") + sent
                    cur_tokens += sent_tokens

            if cur_chunk and real_token_count(cur_chunk) >= min_tokens:
                chunks.append({"heading": sec["heading"], "content": cur_chunk.strip()})

    return chunks


# ──────────────────── Robots.txt ────────────────────
# Fetch robots.txt ONCE and cache it
_robots_parser = None

def get_robots_parser():
    global _robots_parser
    if _robots_parser is None:
        _robots_parser = robotparser.RobotFileParser()
        _robots_parser.set_url(f"https://{ALLOWED_HOST}/robots.txt")
        try:
            print("[init] Fetching robots.txt...")
            _robots_parser.read()
            print("[init] robots.txt loaded")
        except Exception as e:
            print(f"[init] robots.txt failed ({e}), allowing all URLs")
            _robots_parser = None
    return _robots_parser

def allowed_by_robots(test_url: str) -> bool:
    rp = get_robots_parser()
    if rp is None:
        return True  # If robots.txt failed, allow all
    try:
        return rp.can_fetch(USER_AGENT, test_url)
    except Exception:
        return True


def get_seed_urls(prefix: str) -> List[str]:
    """Get seed URLs from ATO sitemap."""
    print("[init] Fetching sitemap...")
    r = requests.get(SITEMAP_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    print(f"[init] Sitemap fetched ({len(r.text)} bytes)")
    root = ET.fromstring(r.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    for loc in root.findall(".//sm:url/sm:loc", ns):
        if not loc.text:
            continue
        u = to_https(loc.text.strip())
        p = urlparse(u)
        if p.netloc == ALLOWED_HOST and under_prefix(u, prefix) and allowed_by_robots(u):
            urls.append(u)
    print(f"[init] {len(urls)} seed URLs found")
    return sorted(set(urls))


# ──────────────────── Playwright ────────────────────
async def open_browser(play, profile_dir: str = "pw_profile"):
    ctx = await play.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
        viewport={"width": 1366, "height": 900},
        locale="en-AU",
        user_agent=USER_AGENT,
    )
    await ctx.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
    )
    await ctx.set_extra_http_headers({
        "Upgrade-Insecure-Requests": "1",
        "Accept-Language": "en-AU,en;q=0.9",
    })
    return ctx


async def fetch_page(ctx, url: str, timeout_ms=45000):
    page = await ctx.new_page()
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(800 + int(random.random() * 500))
        status = resp.status if resp else 0
        title = await page.title()
        html = await page.content()
        # Retry if blocked
        if "Access Denied" in (title or "") and "edgesuite" in (html or ""):
            await page.wait_for_timeout(2500 + int(random.random() * 1500))
            resp2 = await page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
            status = resp2.status if resp2 else status
            title = await page.title()
            html = await page.content()
        return status, title or "", html or "", page
    except Exception:
        try:
            await page.close()
        except Exception:
            pass
        raise


async def collect_child_links(page, base_url: str, prefix: str, max_children: int) -> List[str]:
    hrefs = await page.eval_on_selector_all(
        "a[href]", "els => els.map(a => a.getAttribute('href'))"
    )
    links = []
    for h in hrefs:
        if not h:
            continue
        absu = to_https(urljoin(base_url, h))
        p = urlparse(absu)
        if p.netloc != ALLOWED_HOST:
            continue
        if not under_prefix(absu, prefix):
            continue
        if should_skip(absu):
            continue
        links.append(absu)
    return list(dict.fromkeys(links))[:max_children]


# ──────────────────── Main crawl ────────────────────
async def crawl(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_csv_path = out_dir / "pages.csv"
    chunks_path = out_dir / "chunks.jsonl"
    hubs_path = out_dir / "hubs.jsonl"

    # CSV writer for page inventory
    csvf = pages_csv_path.open("w", newline="", encoding="utf-8")
    csvw = csv.DictWriter(csvf, fieldnames=["url", "status", "title", "type", "chunks", "hash"])
    csvw.writeheader()

    chunkf = chunks_path.open("w", encoding="utf-8")
    hubf = hubs_path.open("w", encoding="utf-8")

    # Get seeds
    seeds = get_seed_urls(args.allow_prefix)
    if args.debug:
        print(f"[seed] {len(seeds)} URLs from sitemap under {args.allow_prefix}")

    # ── BFS queue: (url, depth, accumulated_keywords) ──
    # This is the key change: we pass keywords DOWN through the tree
    queue = deque()
    for u in seeds[:args.max_pages]:
        queue.append((u, 0, []))  # start with empty keyword list

    visited = set()
    seen_hashes = set()
    branch_counts = defaultdict(int)
    stats = {"pages": 0, "hubs": 0, "leaves": 0, "chunks": 0, "blocked": 0}

    async with async_playwright() as play:
        ctx = await open_browser(play)

        # Warmup visit
        try:
            _, _, _, p0 = await fetch_page(ctx, to_https(f"https://{ALLOWED_HOST}/"))
            await p0.close()
        except Exception:
            pass

        while queue and stats["pages"] < args.max_pages:
            url, depth, parent_keywords = queue.popleft()

            if url in visited:
                continue
            if should_skip(url) or not under_prefix(url, args.allow_prefix):
                continue
            if urlparse(url).netloc != ALLOWED_HOST:
                continue

            # Per-branch cap
            bk = branch_key(url)
            if branch_counts[bk] >= args.max_pages_per_branch:
                continue

            # Polite delay
            await asyncio.sleep(max(args.delay, 0.7) + random.random() * 0.6)

            try:
                status, title, html, page = await fetch_page(ctx, url)
            except Exception as e:
                csvw.writerow({"url": url, "status": 0, "title": "", "type": "fetch-error", "chunks": 0, "hash": ""})
                visited.add(url)
                continue

            try:
                # Block check
                if "Access Denied" in title and "edgesuite" in html:
                    if args.debug:
                        print(f"[blocked] {url}")
                    csvw.writerow({"url": url, "status": status, "title": title, "type": "blocked", "chunks": 0, "hash": ""})
                    stats["blocked"] += 1
                    await asyncio.sleep(2.5 + random.random() * 1.5)
                    continue

                soup = BeautifulSoup(html, "html.parser")

                # ━━━━ DETECT: hub or leaf? ━━━━
                is_hub, card_headings = detect_hub_page(soup)

                if is_hub:
                    # ── HUB PAGE: extract card headings as keywords, don't index ──
                    stats["hubs"] += 1

                    # Card headings become keywords for children
                    hub_keywords = parent_keywords.copy()
                    for heading in card_headings:
                        words = heading.lower().replace("-", " ").split()
                        stop = {"and", "or", "the", "for", "to", "of", "in", "an", "a", "is", "you", "your", "how", "what", "when", "if"}
                        hub_keywords.extend(w for w in words if w not in stop and len(w) > 1)

                    # Also add page title keywords
                    title_clean = re.sub(r"\s*\|.*$", "", title)  # strip "| Australian Taxation Office"
                    title_words = title_clean.lower().replace("-", " ").split()
                    stop = {"and", "or", "the", "for", "to", "of", "in", "an", "a", "is", "you", "your"}
                    hub_keywords.extend(w for w in title_words if w not in stop and len(w) > 1)

                    # Dedupe keywords
                    hub_keywords = list(dict.fromkeys(hub_keywords))

                    # Log hub
                    hubf.write(json.dumps({
                        "url": url,
                        "title": title,
                        "card_headings": card_headings,
                        "keywords_passed_down": hub_keywords,
                    }, ensure_ascii=False) + "\n")

                    csvw.writerow({"url": url, "status": status, "title": title, "type": "hub", "chunks": 0, "hash": ""})

                    if args.debug:
                        print(f"[hub] {url}")
                        print(f"       cards: {card_headings[:5]}")
                        print(f"       keywords → children: {hub_keywords[:15]}")

                    # Enqueue children WITH accumulated keywords (no depth limit)
                    children = await collect_child_links(page, url, args.allow_prefix, args.max_children_per_page)
                    for c in children:
                        if c not in visited:
                            queue.append((c, depth + 1, hub_keywords))

                else:
                    # ── LEAF PAGE: extract content, attach keywords, index ──

                    # Content dedup
                    raw_text = soup.get_text()
                    content_hash = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()
                    if content_hash in seen_hashes:
                        csvw.writerow({"url": url, "status": status, "title": title, "type": "leaf-dup", "chunks": 0, "hash": content_hash})
                        continue
                    seen_hashes.add(content_hash)

                    # Extract sections by headings
                    sections = extract_sections(soup)
                    if not sections:
                        csvw.writerow({"url": url, "status": status, "title": title, "type": "leaf-empty", "chunks": 0, "hash": content_hash})
                        continue

                    # Build right-sized chunks
                    page_title = re.sub(r"\s*\|.*$", "", title).strip()
                    page_chunks = chunk_sections(sections, page_title, max_tokens=args.chunk_max_tokens)

                    if not page_chunks:
                        csvw.writerow({"url": url, "status": status, "title": title, "type": "leaf-empty", "chunks": 0, "hash": content_hash})
                        continue

                    # Build breadcrumb from URL
                    breadcrumb = url_to_breadcrumb(url, args.allow_prefix)
                    url_keywords = url_to_keywords(url, args.allow_prefix)

                    # Merge parent keywords + URL keywords (deduped)
                    all_keywords = list(dict.fromkeys(parent_keywords + url_keywords))

                    # Write chunks
                    for i, chunk in enumerate(page_chunks):
                        # Prefix chunk text with heading context for better embedding
                        heading_prefix = f"{page_title}"
                        if chunk["heading"] and chunk["heading"] != page_title:
                            heading_prefix += f" — {chunk['heading']}"

                        chunk_id = f"{content_hash[:10]}_{i + 1:03d}"
                        record = {
                            "id": chunk_id,
                            "url": url,
                            "title": page_title,
                            "section_heading": chunk["heading"],
                            "breadcrumb": breadcrumb,
                            "keywords": all_keywords,
                            "heading_prefix": heading_prefix,
                            "text": chunk["content"],
                            "chunk_index": i,
                            "chunk_total": len(page_chunks),
                            "approx_tokens": real_token_count(chunk["content"]),
                        }
                        chunkf.write(json.dumps(record, ensure_ascii=False) + "\n")

                    stats["leaves"] += 1
                    stats["chunks"] += len(page_chunks)
                    branch_counts[bk] += 1

                    csvw.writerow({"url": url, "status": status, "title": title, "type": "leaf", "chunks": len(page_chunks), "hash": content_hash})

                    if args.debug and stats["leaves"] % 10 == 0:
                        print(f"[progress] leaves={stats['leaves']}, hubs={stats['hubs']}, chunks={stats['chunks']}, queue={len(queue)}")

                    # Leaf pages can also have child links (no depth limit)
                    children = await collect_child_links(page, url, args.allow_prefix, args.max_children_per_page)
                    for c in children:
                        if c not in visited:
                            queue.append((c, depth + 1, all_keywords))

                stats["pages"] += 1

            finally:
                try:
                    await page.close()
                except Exception:
                    pass
                visited.add(url)

        await ctx.close()

    csvf.close()
    chunkf.close()
    hubf.close()

    print(f"\n{'=' * 50}")
    print(f"Crawl complete!")
    print(f"  Pages visited : {stats['pages']}")
    print(f"  Hub pages     : {stats['hubs']} (keywords extracted, not indexed)")
    print(f"  Leaf pages    : {stats['leaves']} (content indexed)")
    print(f"  Chunks written: {stats['chunks']}")
    print(f"  Blocked       : {stats['blocked']}")
    print(f"\nOutputs:")
    print(f"  {chunks_path}  — chunks for indexing")
    print(f"  {hubs_path}    — hub page metadata")
    print(f"  {pages_csv_path} — page inventory")
    print(f"{'=' * 50}")


def parse_args():
    ap = argparse.ArgumentParser(description="ATO Smart Crawler — hub-aware, heading-structured")
    ap.add_argument("--allow-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--out", default="kb_out")
    ap.add_argument("--max-pages", type=int, default=9999, help="Page cap (default: 9999 = crawl everything)")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-children-per-page", type=int, default=30)
    ap.add_argument("--max-pages-per-branch", type=int, default=50)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--chunk-max-tokens", type=int, default=384, help="Max tokens per chunk (sized for embedding model)")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(crawl(args))