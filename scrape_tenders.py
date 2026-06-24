"""
Scrapes live GeM bids matching KEYWORD and downloads the official bid PDFs.
Output:
  data/tenders/<BID_NO>.pdf
  data/tenders/metadata.json
"""
import asyncio, json, re, sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

KEYWORD       = "Laptop"            # change to your vertical
MAX_TENDERS   = 5
MAX_PAGES     = 3
BASE          = "https://bidplus.gem.gov.in"
LISTING_URL   = f"{BASE}/all-bids"
OUT           = Path("data/tenders"); OUT.mkdir(parents=True, exist_ok=True)


def cookies_from_context(ctx_cookies):
    """Convert Playwright cookies → requests-compatible dict."""
    return {c["name"]: c["value"] for c in ctx_cookies}


def parse_cards(html):
    """Pull bid_no, pdf_url, and surrounding metadata from a listing page."""
    soup = BeautifulSoup(html, "html.parser")
    bids = []
    for card in soup.select("div.card"):
        a = card.select_one("a.bid_no_hover[href*='/showbidDocument/']")
        if not a:
            continue
        bid_no  = a.get_text(strip=True)              # e.g. GEM/2026/B/7472371
        pdf_url = BASE + a["href"]                    # /showbidDocument/9254096
        text    = card.get_text(" ", strip=True)

        def grab(pattern, default=""):
            m = re.search(pattern, text, flags=re.I)
            return m.group(1).strip() if m else default

        bids.append({
            "bid_no":      bid_no,
            "pdf_url":     pdf_url,
            "items":       grab(r"Items?\s*[:\-]\s*(.+?)(?:Quantity|Department|Ministry|$)"),
            "quantity":    grab(r"Quantity\s*[:\-]\s*([\d,]+)"),
            "department":  grab(r"Department Name And Address\s*[:\-]\s*(.+?)(?:Start|End|Ministry|$)"),
            "ministry":    grab(r"Ministry\s*[:\-]\s*(.+?)(?:Department|Start|End|$)"),
            "start_date":  grab(r"Start Date\s*[:\-]\s*([\d\-\:\s]+)"),
            "end_date":    grab(r"End Date\s*[:\-]\s*([\d\-\:\s]+)"),
            "raw_text":    text,
        })
    return bids


async def collect_listings():
    """Use a real browser to render the listing page(s) and return parsed bids + cookies."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=400)
        ctx = await browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"))
        page = await ctx.new_page()

        await page.goto(LISTING_URL, wait_until="domcontentloaded")
        await page.wait_for_selector("div.card", timeout=20000)

        # Fire the keyword search
        await page.fill('input[name="searchBid"]', KEYWORD)
        await page.keyboard.press("Enter")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_selector("div.card", timeout=20000)

        all_bids = []
        for page_idx in range(MAX_PAGES):
            html = await page.content()
            bids = parse_cards(html)
            print(f"  page {page_idx+1}: +{len(bids)} cards")
            all_bids.extend(bids)
            if len(all_bids) >= MAX_TENDERS:
                break

            # Pagination link: look for "Next" anchor; if none, stop
            nxt = await page.query_selector("a.page-link[rel='next'], li.next a, a[aria-label='Next']")
            if not nxt:
                break
            await nxt.click()
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(1500)

        cookies = cookies_from_context(await ctx.cookies())
        ua      = await page.evaluate("navigator.userAgent")
        await browser.close()
        return all_bids, cookies, ua


def download_pdf(url, cookies, ua, out_path):
    headers = {"User-Agent": ua, "Referer": LISTING_URL, "Accept": "application/pdf,*/*"}
    r = requests.get(url, cookies=cookies, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"Not a PDF (got {r.headers.get('content-type')}, {len(r.content)} bytes)")
    out_path.write_bytes(r.content)
    return len(r.content)


async def main():
    print(f"🔎 Searching GeM for: {KEYWORD}")
    bids, cookies, ua = await collect_listings()

    # Dedupe + cap
    seen, unique = set(), []
    for b in bids:
        if b["bid_no"] in seen: continue
        seen.add(b["bid_no"]); unique.append(b)
    unique = unique[:MAX_TENDERS]
    print(f"📄 Downloading {len(unique)} PDFs ...")

    saved = []
    for b in unique:
        safe = b["bid_no"].replace("/", "_")
        path = OUT / f"{safe}.pdf"
        try:
            size = download_pdf(b["pdf_url"], cookies, ua, path)
            b["pdf_path"] = str(path)
            b["pdf_bytes"] = size
            saved.append(b)
            print(f"  ✓ {b['bid_no']}  ({size//1024} KB)")
        except Exception as e:
            print(f"  ✗ {b['bid_no']}  — {e}")

    (OUT / "metadata.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(saved)} tenders → {OUT}")
    print(f"   metadata: {OUT/'metadata.json'}")


if __name__ == "__main__":
    asyncio.run(main())
