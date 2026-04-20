#!/usr/bin/env python3
"""
retry_failed.py — Re-fetch only the failed pages using Playwright.

Reads inventory.csv, finds URLs with no filename (failed),
and retries them with a real browser.

Usage:
    python retry_failed.py --out raw_html --debug
"""

import argparse, asyncio, csv, hashlib, random, re
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

ALLOWED_HOST = "www.ato.gov.au"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def url_to_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        path = "index"
    filename = path.replace("/", "__") + ".html"
    if len(filename) > 200:
        filename = filename[:190] + "__" + hashlib.md5(path.encode()).hexdigest()[:8] + ".html"
    return filename


async def retry(args):
    out_dir = Path(args.out)
    html_dir = out_dir / "html"
    inv_path = out_dir / "inventory.csv"

    if not inv_path.exists():
        print(f"[ERROR] {inv_path} not found")
        return

    # Find failed URLs
    failed = []
    with inv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("filename"):  # No file saved = failed
                failed.append(row["url"])

    print(f"[retry] {len(failed)} failed URLs to retry with Playwright\n")

    if not failed:
        print("Nothing to retry!")
        return

    # New inventory for retried pages
    retry_inv_path = out_dir / "inventory_retry.csv"
    retry_f = retry_inv_path.open("w", newline="", encoding="utf-8")
    retry_w = csv.DictWriter(retry_f, fieldnames=["url", "status", "title", "filename", "size"])
    retry_w.writeheader()

    stats = {"ok": 0, "blocked": 0, "error": 0}

    async with async_playwright() as play:
        ctx = await play.chromium.launch_persistent_context(
            user_data_dir=str(out_dir / "pw_retry_profile"),
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

        for i, url in enumerate(failed):
            await asyncio.sleep(max(args.delay, 0.8) + random.random() * 0.5)

            filename = url_to_filename(url)
            page = None

            try:
                page = await ctx.new_page()
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(800 + int(random.random() * 400))

                status = resp.status if resp else 0
                title = await page.title() or ""
                html = await page.content() or ""

                if "Access Denied" in title and "edgesuite" in html:
                    stats["blocked"] += 1
                    retry_w.writerow({"url": url, "status": status, "title": title, "filename": "", "size": 0})
                    if args.debug:
                        print(f"  [{i+1}/{len(failed)}] BLOCKED {url}")
                    await asyncio.sleep(3 + random.random() * 2)
                    continue

                # Save HTML
                filepath = html_dir / filename
                filepath.write_text(html, encoding="utf-8")
                stats["ok"] += 1
                retry_w.writerow({"url": url, "status": status, "title": title, "filename": filename, "size": len(html)})

                if args.debug and (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(failed)}] ok={stats['ok']} blocked={stats['blocked']} err={stats['error']}")

            except Exception as e:
                stats["error"] += 1
                retry_w.writerow({"url": url, "status": 0, "title": "", "filename": "", "size": 0})
                if args.debug:
                    print(f"  [{i+1}/{len(failed)}] ERROR {url.split('/')[-1][:50]}: {str(e)[:40]}")

            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass

        await ctx.close()

    retry_f.close()

    # Merge retry results into main inventory
    print(f"\n[merge] Merging retry results into inventory...")

    # Read original inventory
    original = []
    with inv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original.append(row)

    # Read retry results
    retried = {}
    with retry_inv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename"):  # Only replace if retry succeeded
                retried[row["url"]] = row

    # Write merged inventory
    merged_count = 0
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "status", "title", "filename", "size"])
        writer.writeheader()
        for row in original:
            if row["url"] in retried:
                writer.writerow(retried[row["url"]])
                merged_count += 1
            else:
                writer.writerow(row)

    print(f"\nDone!")
    print(f"  Retried   : {len(failed)}")
    print(f"  Recovered : {stats['ok']}")
    print(f"  Blocked   : {stats['blocked']}")
    print(f"  Still failed: {stats['error']}")
    print(f"  Merged into inventory: {merged_count}")
    print(f"\n  Total pages now: {2391 + stats['ok']}")
    print(f"\n  Next: python process_pages.py --html-dir {html_dir} --inventory {inv_path} --debug")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw_html")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    asyncio.run(retry(args))
