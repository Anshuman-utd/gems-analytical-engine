# scraper.py
import asyncio
import re
import logging
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def collect_bid_result_urls(keyword: str, max_bids: int = 20) -> list[dict]:
    bid_links = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=600)
        page = await browser.new_page()

        # ── Go directly to the bid listing page on bidplus subdomain ──
        logger.info("Navigating to bid listing...")
        await page.goto("https://bidplus.gem.gov.in/all-bids", timeout=60000)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)

        # ── Apply filter: Bid/RA Status ──
        # We use CSS selectors now instead of get_by_label — more reliable
        logger.info("Checking Bid/RA Status checkbox...")
        try:
            # Find checkbox by its sibling label text
            await page.locator("label", has_text="Bid/RA Status").click()
            await page.wait_for_timeout(1500)
        except Exception as e:
            logger.warning(f"Bid/RA Status checkbox failed: {e}")

        # ── Apply filter: Bid/RA Awarded ──
        logger.info("Checking Bid/RA Awarded checkbox...")
        try:
            await page.locator("label", has_text="Bid /RA Awarded").click()
            await page.wait_for_timeout(1500)
        except Exception as e:
            logger.warning(f"Bid/RA Awarded checkbox failed: {e}")

        # ── Type keyword in the CORRECT search bar ──
        # From your screenshot, the placeholder is "Enter Keyword" — not "Looking for something"
        logger.info(f"Typing keyword: {keyword}")
        try:
            # Target the keyword input specifically — it's near the "Contains" dropdown
            keyword_input = page.locator("input[placeholder='Enter Keyword']")
            await keyword_input.wait_for(timeout=10000)
            await keyword_input.click()
            await keyword_input.fill(keyword)
            await page.wait_for_timeout(500)

            # Click the search button (the blue magnifier next to the input)
            search_btn = page.locator("button.btn-primary").last
            await search_btn.click()
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(3000)
            logger.info("Keyword search submitted successfully")
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            logger.info("Printing all input placeholders on page for debugging...")
            # This helps us see what inputs actually exist on the page
            inputs = await page.locator("input").all()
            for inp in inputs:
                ph = await inp.get_attribute("placeholder")
                logger.info(f"  Found input with placeholder: '{ph}'")

        # ── Collect bid result links across pages ──
        while len(bid_links) < max_bids:
            logger.info(f"Scraping current page... ({len(bid_links)} collected so far)")
            await page.wait_for_timeout(1500)

            soup = BeautifulSoup(await page.content(), "html.parser")

            # Find all anchor tags whose text matches "View BID Results" variants
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                href = link.get("href", "")
                text = link.get_text(strip=True)

                if "getBidResultView" in href:
                    match = re.search(r"getBidResultView/(\d+)", href)
                    if not match:
                        continue

                    numeric_id = match.group(1)

                    # Build full URL
                    if href.startswith("http"):
                        full_url = href
                    else:
                        full_url = "https://bidplus.gem.gov.in" + href

                    # Try to find the GEM bid ID from the surrounding card
                    # It's an anchor tag with text like "GEM/2026/B/7662514"
                    card = link.find_parent()
                    gem_id = "UNKNOWN"
                    for _ in range(6):  # walk up max 6 levels to find GEM ID
                        if card is None:
                            break
                        gem_anchor = card.find(
                            "a", string=re.compile(r"GEM/\d{4}/B/\d+")
                        )
                        if gem_anchor:
                            gem_id = gem_anchor.get_text(strip=True)
                            break
                        card = card.parent

                    bid_links.append({
                        "numeric_id": numeric_id,
                        "gem_bid_id": gem_id,
                        "result_url": full_url
                    })
                    logger.info(f"  Found: {gem_id} → {full_url}")

                    if len(bid_links) >= max_bids:
                        break

            # ── Pagination — go to next page ──
            try:
                # Next button is usually ">" or "Next" in pagination
                next_btn = page.locator("a", has_text=re.compile(r"Next|»|>"))
                if await next_btn.count() > 0:
                    await next_btn.first.click()
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(2000)
                else:
                    logger.info("No next page button found — done paginating")
                    break
            except Exception:
                logger.info("Pagination ended")
                break

        await browser.close()

    logger.info(f"Total bid links collected: {len(bid_links)}")
    return bid_links


async def scrape_all_results(bid_links: list[dict]) -> pd.DataFrame:
    records = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for i, bid in enumerate(bid_links):
            gem_bid_id = bid["gem_bid_id"]
            url        = bid["result_url"]
            logger.info(f"[{i+1}/{len(bid_links)}] Scraping {gem_bid_id}")

            try:
                await page.goto(url, timeout=30000)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(1500)

                html   = await page.content()
                record = parse_financial_table(html, gem_bid_id)

                if record:
                    records.append(record)
                    logger.info(f"  ✓ L1=₹{record['l1_price']:,.0f}  Bidders={record['num_bidders']}")
                else:
                    logger.info(f"  ✗ No financial table — skipped")

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")

            await page.wait_for_timeout(2000)

        await browser.close()

    return pd.DataFrame(records)


def parse_financial_table(html: str, gem_bid_id: str) -> dict | None:
    """
    Parses the Financial Evaluation table from a bid result page.
    Returns None if no table found — caller skips this bid cleanly.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find any element containing "FINANCIAL EVALUATION" text
    fin_section = soup.find(
        lambda tag: tag.name and
        "FINANCIAL EVALUATION" in tag.get_text(strip=True).upper()
    )
    if not fin_section:
        return None

    table = fin_section.find_next("table")
    if not table:
        return None

    record = {
        "gem_bid_id":       gem_bid_id,
        "l1_price":         None,
        "l2_price":         None,
        "l3_price":         None,
        "l1_seller":        None,
        "num_bidders":      0,
        "price_spread_pct": None,
    }

    bidders = []
    for row in table.find_all("tr")[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        seller     = cells[1].get_text(strip=True)
        price_text = cells[3].get_text(strip=True)
        rank       = cells[4].get_text(strip=True).upper().strip()

        # Strip Indian currency formatting: ₹ 3,53,300.00 → 353300.0
        clean = re.sub(r"[^\d.]", "", price_text)
        if not clean:
            continue

        bidders.append({"rank": rank, "seller": seller, "price": float(clean)})

    record["num_bidders"] = len(bidders)

    for b in bidders:
        if b["rank"] == "L1":
            record["l1_price"]  = b["price"]
            record["l1_seller"] = b["seller"]
        elif b["rank"] == "L2":
            record["l2_price"]  = b["price"]
        elif b["rank"] == "L3":
            record["l3_price"]  = b["price"]

    # Must have L1 to be useful for ML model
    if record["l1_price"] is None:
        return None

    if record["l1_price"] and record["l3_price"]:
        record["price_spread_pct"] = round(
            (record["l3_price"] - record["l1_price"]) / record["l1_price"] * 100, 2
        )

    return record


def save(df: pd.DataFrame, path: str = "data/awards/laptop_aoc.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(df)} records → {path}")
    print("\n── FINAL DATA ──")
    print(df[["gem_bid_id", "l1_price", "l2_price",
              "num_bidders", "price_spread_pct"]].to_string())


async def main():
    bid_links = await collect_bid_result_urls(keyword="laptop", max_bids=20)

    if not bid_links:
        logger.error("No bid links found. Scraper needs selector fixes.")
        return

    df = await scrape_all_results(bid_links)

    if not df.empty:
        save(df)
    else:
        logger.warning("No financial data collected.")


if __name__ == "__main__":
    asyncio.run(main())