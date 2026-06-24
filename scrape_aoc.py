"""
Scrapes historical Award of Contract (AOC) rows from GeM for KEYWORD.
Output:
  data/aoc/awards.csv
  data/aoc/awards_raw.json
"""
import asyncio, json, re, csv
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

KEYWORD     = "Laptop"
MAX_PAGES   = 25                 # ~10 rows/page → up to ~250 awards
BASE        = "https://bidplus.gem.gov.in"
AOC_URL     = f"{BASE}/bidresultlists"
OUT         = Path("data/aoc");  OUT.mkdir(parents=True, exist_ok=True)


def parse_awards(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select("div.card"):
        text = card.get_text(" ", strip=True)
        if not re.search(r"GEM/\d{4}/[A-Z]/\d+", text):
            continue

        def grab(pat, default=""):
            m = re.search(pat, text, flags=re.I)
            return m.group(1).strip() if m else default

        bid_no   = (re.search(r"GEM/\d{4}/[A-Z]/\d+", text) or [""])[0]
        rows.append({
            "bid_no":        bid_no,
            "item":          grab(r"Item[s]?\s*[:\-]\s*(.+?)(?:Quantity|Organisation|Department|$)"),
            "quantity":      grab(r"Quantity\s*[:\-]\s*([\d,]+)"),
            "contract_value":grab(r"Contract\s*Value\s*[:\-]?\s*₹?\s*([\d,\.]+)"),
            "unit_price":    grab(r"Unit\s*Price\s*[:\-]?\s*₹?\s*([\d,\.]+)"),
            "seller":        grab(r"Seller\s*Name\s*[:\-]\s*(.+?)(?:Contract|Order|$)"),
            "buyer_org":     grab(r"Organisation\s*Name\s*[:\-]\s*(.+?)(?:Office|Department|Ministry|Buyer|$)"),
            "ministry":      grab(r"Ministry\s*[:\-]\s*(.+?)(?:Department|Organisation|$)"),
            "contract_date": grab(r"Contract\s*Date\s*[:\-]\s*([\d\-\/\.\s]+)"),
            "raw_text":      text,
        })
    return rows


async def main():
    print(f"🔎 Fetching AOC for: {KEYWORD}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=400)
        ctx  = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        page = await ctx.new_page()

        await page.goto(AOC_URL, wait_until="domcontentloaded")
        await page.wait_for_selector("div.card", timeout=20000)

        # Most AOC pages also expose the same search box; if not, comment these 3 lines
        try:
            await page.fill('input[name="searchBid"]', KEYWORD)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_selector("div.card", timeout=20000)
        except Exception:
            print("  (no search box on AOC page — scraping unfiltered list)")

        all_rows = []
        for i in range(MAX_PAGES):
            html = await page.content()
            rows = parse_awards(html)
            new = [r for r in rows if r["bid_no"] not in {x["bid_no"] for x in all_rows}]
            all_rows.extend(new)
            print(f"  page {i+1}: +{len(new)} rows (total {len(all_rows)})")

            nxt = await page.query_selector("a.page-link[rel='next'], li.next a, a[aria-label='Next']")
            if not nxt:
                break
            try:
                await nxt.click()
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(1500)
            except Exception:
                break

        await browser.close()

    # Save raw + CSV
    (OUT / "awards_raw.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))

    if all_rows:
        keys = [k for k in all_rows[0].keys() if k != "raw_text"]
        with open(OUT / "awards.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in all_rows:
                w.writerow({k: r[k] for k in keys})

    print(f"\n✅ Saved {len(all_rows)} awards → {OUT/'awards.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
