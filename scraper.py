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


async def collect_bid_result_urls(keyword: str, max_bids: int = 50) -> list[dict]:
    seen_ids = set()
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
            keyword_input = page.locator("input[placeholder='Enter Keyword']")
            await keyword_input.wait_for(timeout=10000)

            await keyword_input.fill("")
            await keyword_input.fill(keyword)

            await page.wait_for_timeout(1000)

            logger.info("Clicking search button...")

            # Correct search button
            search_btn = page.locator("#searchBidRA")
            await search_btn.wait_for(timeout=10000)
            await search_btn.click()

            # Wait for search results
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(3000)

            # Save screenshot for debugging
            await page.screenshot(
                path=f"search_{keyword}.png",
                full_page=True
            )

            logger.info(f"Search completed. URL: {page.url}")

            cards = await page.locator("a[href*='getBidResultView']").count()

            logger.info(f"Bid Result links visible after search: {cards}")

        except Exception as e:
            logger.error(f"Keyword search FAILED: {e}")
            await browser.close()
            return []
        # ── Collect bid result links across pages ──
        while len(bid_links) < max_bids:
            logger.info(f"Scraping current page... ({len(bid_links)} collected so far)")
            await page.wait_for_timeout(1500)

            soup = BeautifulSoup(await page.content(), "html.parser")
            page_links = [
                a for a in soup.find_all("a", href=True)
                if "getBidResultView" in a["href"]
            ]

            logger.info(f"Page URL : {page.url}")
    
            logger.info(f"Bid Result links on page : {len(page_links)}")
            gem_ids = []

            for link in page_links:
                href = link.get("href", "")
                m = re.search(r"getBidResultView/(\d+)", href)
                if m:
                    gem_ids.append(m.group(1))

            logger.info(f"Unique result links on page: {len(set(gem_ids))}")

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

                    if numeric_id in seen_ids:
                        continue

                    seen_ids.add(numeric_id)

                    bid_links.append(
                        {
                            "numeric_id": numeric_id,
                            "gem_bid_id": gem_id,
                            "result_url": full_url,
                        }
                    )
                    logger.info(f"  Found: {gem_id} → {full_url}")

                    if len(bid_links) >= max_bids:
                        break

            # ── Pagination — go to next page ──
            try:
                # Scroll to pagination
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)

                next_btn = page.get_by_role("link", name="Next")

                if await next_btn.count() == 0:
                    logger.info("No Next button found")
                    break

                old_url = page.url

                await next_btn.click()

                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(3000)

                logger.info(f"Moved from {old_url} to {page.url}")

            except Exception as e:
                logger.info(f"Pagination finished: {e}")
                break

        await browser.close()

    logger.info(f"Total bid links collected: {len(bid_links)}")
    return bid_links


async def scrape_all_results(bid_links: list[dict]) -> pd.DataFrame:
    records = []

    pages_visited = 0
    financial_pages = 0
    parsed = 0
    saved_failed_pages = 0

    async with async_playwright() as p:

        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for i, bid in enumerate(bid_links):

            gem_bid_id = bid["gem_bid_id"]
            url = bid["result_url"]

            logger.info("=" * 80)
            logger.info(f"[{i+1}/{len(bid_links)}] {gem_bid_id}")

            try:

                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=60000,
                )

                await page.wait_for_timeout(3000)

                html = await page.content()

                pages_visited += 1

                has_financial = "FINANCIAL EVALUATION" in html.upper()

                if has_financial:
                    financial_pages += 1

                try:

                    record = parse_financial_table(html, gem_bid_id)

                    if record:

                        parsed += 1

                        records.append(record)

                        logger.info(
                            f"SUCCESS | "
                            f"L1={record['l1_price']} | "
                            f"Bidders={record['num_bidders']}"
                        )

                    else:

                        logger.info("Parser returned None")

                        if has_financial:

                            filename = (
                                f"failed_{gem_bid_id.replace('/', '_')}.html"
                            )

                            with open(
                                filename,
                                "w",
                                encoding="utf-8",
                            ) as f:
                                f.write(html)

                            saved_failed_pages += 1

                            logger.warning(
                                f"Saved failed HTML -> {filename}"
                            )

                except Exception as parser_error:

                    logger.error(
                        f"Parser exception: {parser_error}"
                    )

                    if has_financial:

                        filename = (
                            f"failed_{gem_bid_id.replace('/', '_')}.html"
                        )

                        with open(
                            filename,
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(html)

                        saved_failed_pages += 1

                        logger.warning(
                            f"Saved parser failure -> {filename}"
                        )

            except Exception as e:

                logger.error(f"Navigation failed: {e}")

        await browser.close()

    logger.info("=" * 80)
    logger.info("SCRAPING SUMMARY")
    logger.info(f"Visited pages        : {pages_visited}")
    logger.info(f"Financial sections   : {financial_pages}")
    logger.info(f"Parsed records       : {parsed}")
    logger.info(f"Failed HTML saved    : {saved_failed_pages}")
    logger.info("=" * 80)

    return pd.DataFrame(records)

def parse_financial_table(html: str, gem_bid_id: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    # Quick exit
    if "FINANCIAL EVALUATION" not in html.upper():
        logger.warning(f"[{gem_bid_id}] Financial Evaluation section missing")
        return None

    # --------------------------------------------------
    # Find the correct financial table by looking at
    # the TABLE HEADERS instead of using find_next()
    # --------------------------------------------------
    financial_table = None

    for table in soup.find_all("table"):

        headers = [
            th.get_text(" ", strip=True).upper()
            for th in table.find_all("th")
        ]

        logger.info(f"[{gem_bid_id}] Checking table headers -> {headers}")

        if (
            "SELLER NAME" in headers
            and "TOTAL PRICE" in headers
            and "RANK" in headers
        ):
            financial_table = table
            logger.info(f"[{gem_bid_id}] Correct financial table found.")
            break

    if financial_table is None:
        logger.warning(f"[{gem_bid_id}] Financial table not found.")
        return None

    # --------------------------------------------------
    # Detect column positions automatically
    # --------------------------------------------------
    headers = [
        th.get_text(" ", strip=True).upper()
        for th in financial_table.find_all("th")
    ]

    seller_idx = headers.index("SELLER NAME")
    price_idx = headers.index("TOTAL PRICE")
    rank_idx = headers.index("RANK")

    logger.info(
        f"[{gem_bid_id}] seller={seller_idx}, "
        f"price={price_idx}, "
        f"rank={rank_idx}"
    )

    bidders = []

    rows = financial_table.find("tbody").find_all("tr")

    for row in rows:

        cells = row.find_all("td")

        values = [
            td.get_text(" ", strip=True)
            for td in cells
        ]

        logger.info(f"[{gem_bid_id}] ROW -> {values}")

        if len(cells) <= max(seller_idx, price_idx, rank_idx):
            continue

        seller = cells[seller_idx].get_text(" ", strip=True)

        price_text = cells[price_idx].get_text(" ", strip=True)

        rank = (
            cells[rank_idx]
            .get_text(" ", strip=True)
            .upper()
        )

        price_match = re.search(r"\d[\d,.]*\.?\d*", price_text)

        if not price_match:
            continue

        price = float(
            price_match.group().replace(",", "")
        )

        bidders.append(
            {
                "seller": seller,
                "price": price,
                "rank": rank,
            }
        )

    if not bidders:
        logger.warning(f"[{gem_bid_id}] No bidder rows parsed.")
        return None

    record = {
        "gem_bid_id": gem_bid_id,
        "l1_price": None,
        "l2_price": None,
        "l3_price": None,
        "l1_seller": None,
        "num_bidders": len(bidders),
        "price_spread_pct": None,
    }

    for bidder in bidders:

        if "L1" in bidder["rank"]:
            record["l1_price"] = bidder["price"]
            record["l1_seller"] = bidder["seller"]

        elif "L2" in bidder["rank"]:
            record["l2_price"] = bidder["price"]

        elif "L3" in bidder["rank"]:
            record["l3_price"] = bidder["price"]

    if record["l1_price"] is None:
        logger.warning(f"[{gem_bid_id}] L1 bidder not found.")
        return None

    if record["l3_price"] is not None:
        record["price_spread_pct"] = round(
            (
                record["l3_price"] - record["l1_price"]
            )
            / record["l1_price"]
            * 100,
            2,
        )

    logger.info(
        f"[{gem_bid_id}] SUCCESS -> "
        f"L1={record['l1_price']} "
        f"L2={record['l2_price']} "
        f"L3={record['l3_price']}"
    )

    return record


def save(df: pd.DataFrame, path: str = "data/awards/it_hardware.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # If previous data exists, merge with it
    if Path(path).exists():
        existing_df = pd.read_csv(path)

        # Combine old + new
        df = pd.concat([existing_df, df], ignore_index=True)

        # Remove duplicates based on the GeM Bid ID
        df = df.drop_duplicates(subset=["gem_bid_id"], keep="first")

    # Save merged data
    df.to_csv(path, index=False, encoding="utf-8")

    logger.info(f"Saved {len(df)} total records → {path}")
    print("\n── FINAL DATA ──")
    print(
        df[
            [
                "gem_bid_id",
                "l1_price",
                "l2_price",
                "num_bidders",
                "price_spread_pct",
            ]
        ].to_string(index=False)
    )


async def main():
    

    keywords = [
    "laptop"
    ]

    for keyword in keywords:
        logger.info(f"Processing keyword: {keyword}")
        bid_links = await collect_bid_result_urls(keyword, 20)

        if not bid_links:
            continue

        df = await scrape_all_results(bid_links)

        if not df.empty:
            save(df)
        else:
            logger.warning("No financial data collected.")


if __name__ == "__main__":
    asyncio.run(main())