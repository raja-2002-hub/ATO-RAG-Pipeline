#!/usr/bin/env python3
"""
Step 1: fetch_pages.py — Download all ATO pages as raw HTML.

No classification, no chunking, no processing.
Just fetch every page from the sitemap and save the HTML.
Processing happens in step 2 (process_pages.py).

Usage:
    python fetch_pages.py --out raw_html --debug
"""

import argparse, asyncio, csv, hashlib, json, random, re, time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from playwright.async_api import async_playwright

# ──────────────────── Config ────────────────────
SITEMAP_URL  = "https://www.ato.gov.au/sitemap.xml"
ALLOWED_HOST = "www.ato.gov.au"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def to_https(u: str) -> str:
    p = urlparse(u)
    p = p._replace(scheme="https", params="", query="", fragment="")
    return urlunparse(p)


def get_sitemap_urls() -> list:
    """Fetch all URLs from ATO sitemap."""
    print("[1/3] Fetching sitemap...")
    r = requests.get(SITEMAP_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    print(f"      Sitemap size: {len(r.text)} bytes")

    root = ET.fromstring(r.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    for loc in root.findall(".//sm:url/sm:loc", ns):
        if not loc.text:
            continue
        u = to_https(loc.text.strip())
        if urlparse(u).netloc == ALLOWED_HOST:
            urls.append(u)

    urls = sorted(set(urls))
    print(f"      Found {len(urls)} URLs")
    return urls


async def fetch_all(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get URLs from sitemap
    urls = get_sitemap_urls()

    # Filter by prefix if specified
    if args.prefix and args.prefix != "/":
        urls = [u for u in urls if urlparse(u).path.startswith(args.prefix)]
        print(f"      Filtered to {len(urls)} URLs under {args.prefix}")

    # Inventory CSV
    inv_path = out_dir / "inventory.csv"
    inv_f = inv_path.open("w", newline="", encoding="utf-8")
    inv_w = csv.DictWriter(inv_f, fieldnames=["url", "status", "title", "filename", "size"])
    inv_w.writeheader()

    html_dir = out_dir / "html"
    html_dir.mkdir(exist_ok=True)

    print(f"\n[2/3] Fetching {len(urls)} pages with Playwright...")
    print(f"      Saving HTML to {html_dir}/")
    print(f"      Estimated time: {len(urls) * 2 // 60} - {len(urls) * 3 // 60} minutes\n")

    stats = {"ok": 0, "blocked": 0, "error": 0}

    async with async_playwright() as play:
        ctx = await play.chromium.launch_persistent_context(
            user_data_dir=str(out_dir / "pw_profile"),
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

        # Warmup
        try:
            p0 = await ctx.new_page()
            await p0.goto("https://www.ato.gov.au/", wait_until="domcontentloaded", timeout=30000)
            await p0.wait_for_timeout(2000)
            await p0.close()
        except Exception:
            pass

        for i, url in enumerate(urls):
            # Polite delay
            await asyncio.sleep(max(args.delay, 0.8) + random.random() * 0.5)

            # Create filename from URL
            path = urlparse(url).path.strip("/")
            if not path:
                path = "index"
            filename = path.replace("/", "__") + ".html"
            # Truncate long filenames
            if len(filename) > 200:
                filename = filename[:190] + "__" + hashlib.md5(path.encode()).hexdigest()[:8] + ".html"

            page = None
            try:
                page = await ctx.new_page()
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(800 + int(random.random() * 400))

                status = resp.status if resp else 0
                title = await page.title() or ""
                html = await page.content() or ""

                # Check for block
                if "Access Denied" in title and "edgesuite" in html:
                    stats["blocked"] += 1
                    inv_w.writerow({"url": url, "status": status, "title": title, "filename": "", "size": 0})
                    if args.debug:
                        print(f"  [{i+1}/{len(urls)}] BLOCKED {url}")
                    await asyncio.sleep(3 + random.random() * 2)
                    continue

                # Save HTML
                filepath = html_dir / filename
                filepath.write_text(html, encoding="utf-8")

                stats["ok"] += 1
                inv_w.writerow({"url": url, "status": status, "title": title, "filename": filename, "size": len(html)})

                if args.debug and (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(urls)}] ok={stats['ok']} blocked={stats['blocked']} err={stats['error']}")

            except Exception as e:
                stats["error"] += 1
                inv_w.writerow({"url": url, "status": 0, "title": "", "filename": "", "size": 0})
                if args.debug:
                    print(f"  [{i+1}/{len(urls)}] ERROR {url}: {e}")

            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass

        await ctx.close()

    inv_f.close()

    print(f"\n[3/3] Done!")
    print(f"      Pages fetched: {stats['ok']}")
    print(f"      Blocked: {stats['blocked']}")
    print(f"      Errors: {stats['error']}")
    print(f"\n      HTML files: {html_dir}/")
    print(f"      Inventory: {inv_path}")
    print(f"\n      Next step: python process_pages.py --html-dir {html_dir} --inventory {inv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch all ATO pages as raw HTML")
    ap.add_argument("--out", default="raw_html", help="Output directory")
    ap.add_argument("--prefix", default="/", help="URL prefix filter (default: / = all)")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    asyncio.run(fetch_all(args))
