#!/usr/bin/env python3
"""
DeepWiki Hover Links Capture
----------------------------
Moves the mouse cursor over each sidebar <a> link (without clicking),
triggers the ?rsc= fetch requests, and captures all text/x-component
responses as Markdown.

Usage:
  pip install playwright
  python3 -m playwright install chromium
  python3 deepwiki_hover_links_capture.py https://deepwiki.com/juspay/hyperswitch/ ./out --verbose --keep-raw
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

# ------------------------- helpers -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s[:200] or "file"

def filename_from_url(u: str) -> str:
    parsed = urlparse(u)
    parts = [p for p in parsed.path.split("/") if p]
    last = parts[-1] if parts else "index"
    last = re.sub(r"\.(json|txt|js|jsx|ts|tsx)$", "", last, flags=re.I)
    return slugify(last)

def try_parse_json(txt: str):
    t = txt.lstrip()
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None

def looks_like_markdown(text: str) -> bool:
    return bool(
        re.search(r"^\s*#{1,6}\s+\S", text, re.M)
        or re.search(r"^\s*[-*]\s+\S", text, re.M)
        or re.search(r"^\s*\d+\.\s+\S", text, re.M)
        or "```" in text
        or re.search(r"\[[^\]]+\]\([^)]+\)", text)
        or "\n\n" in text
    )

def extract_markdown_candidates(body: str):
    results = []

    def add(name, content):
        if not content:
            return
        results.append({"name": name or None, "content": content})

    if looks_like_markdown(body):
        add(None, body)

    parsed = try_parse_json(body)
    if parsed is not None:
        def walk(node):
            if isinstance(node, dict):
                if isinstance(node.get("content"), str):
                    add(node.get("name") or node.get("title"), node["content"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(parsed)

    dedup = []
    seen = set()
    for r in results:
        h = hashlib.sha256(r["content"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            dedup.append(r)
    return dedup


# ------------------------- main -------------------------

async def hover_links_capture(url: str, out_dir: Path, verbose=False, keep_raw=False):
    ensure_dir(out_dir)
    seen_hashes = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(bypass_csp=True)
        page = await context.new_page()

        print(f"Opening {url}")
        await page.goto(url, wait_until="networkidle", timeout=60000)

        # --- Listen for network responses ---
        async def on_response(resp):
            try:
                r_url = resp.url
                ct = (resp.headers.get("content-type") or "").lower()
                if "text/x-component" in ct or "?rsc=" in r_url:
                    body = (await resp.body()).decode("utf-8", errors="replace")
                    cands = extract_markdown_candidates(body)
                    for i, c in enumerate(cands):
                        h = hashlib.sha256(c["content"].encode("utf-8")).hexdigest()
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                        slug = slugify(filename_from_url(r_url))
                        md_path = out_dir / f"{slug}-{i}.md"
                        md_path.write_text(c["content"], encoding="utf-8")
                        print(f"[ok] {md_path.name} <- {r_url}")
                    if keep_raw:
                        raw = out_dir / f"{slug}.raw.txt"
                        raw.write_text(body, encoding="utf-8")
                        if verbose:
                            print(f"[raw] {raw.name}")
            except Exception as e:
                if verbose:
                    print("response error:", e)

        page.on("response", on_response)

        # --- Find sidebar links ---
        sidebar = await page.query_selector("div[class*='md:sticky']")
        if not sidebar:
            sidebar = await page.query_selector("aside, nav")
        if not sidebar:
            print("❌ Sidebar not found.")
            await browser.close()
            return

        links = await sidebar.query_selector_all("a")
        print(f"✅ Found {len(links)} sidebar links. Hovering to trigger RSC fetches...")

        # --- Move mouse to each link sequentially ---
        for i, link in enumerate(links):
            try:
                box = await link.bounding_box()
                if not box:
                    continue
                x = box["x"] + box["width"] / 2
                y = box["y"] + box["height"] / 2
                await page.mouse.move(x, y)
                if verbose:
                    text = await link.inner_text()
                    print(f"[{i+1}/{len(links)}] Hovering {text.strip()[:60]!r}")
                await page.wait_for_timeout(300)  # small pause for request to fire
            except Exception as e:
                if verbose:
                    print(f"hover error on link {i}: {e}")

        # --- Wait for final pending requests ---
        await page.wait_for_timeout(5000)

        await browser.close()
        print("✅ Done. Files saved to:", out_dir.resolve())


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Hover over DeepWiki sidebar links and capture ?rsc= responses")
    ap.add_argument("url", help="DeepWiki page URL")
    ap.add_argument("out", nargs="?", default="./out", help="Output directory")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    asyncio.run(hover_links_capture(args.url, Path(args.out), verbose=args.verbose, keep_raw=args.keep_raw))

if __name__ == "__main__":
    main()

