"""
Download FULL NQ E-mini 1-minute data from Barchart.
Each contract needs 3-4 downloads of ~20 days each to cover the full quarter.

Barchart limit: ~20,000 rows per download ≈ 19 trading days of 1-min data.
Each quarterly contract active period ≈ 63 trading days → need ~4 downloads.

Uses Playwright's own Chromium + injected cookies. Does NOT touch Chrome.
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from pycookiecheat import chrome_cookies
from playwright.async_api import async_playwright

DATA_DIR = Path("data/barchart_nq")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Each contract: (symbol, active_from, active_to)
# Active period spans from prev expiry to this expiry
CONTRACTS = [
    ("NQH22", "12/17/2021", "03/18/2022"),
    ("NQM22", "03/18/2022", "06/17/2022"),
    ("NQU22", "06/17/2022", "09/16/2022"),
    ("NQZ22", "09/16/2022", "12/16/2022"),
    ("NQH23", "12/16/2022", "03/17/2023"),
    ("NQM23", "03/17/2023", "06/16/2023"),
    ("NQU23", "06/16/2023", "09/15/2023"),
    ("NQZ23", "09/15/2023", "12/15/2023"),
    ("NQH24", "12/15/2023", "03/15/2024"),
    ("NQM24", "03/15/2024", "06/21/2024"),
    ("NQU24", "06/21/2024", "09/20/2024"),
    ("NQZ24", "09/20/2024", "12/20/2024"),
    ("NQH25", "12/20/2024", "03/21/2025"),
    ("NQM25", "03/21/2025", "06/20/2025"),
    ("NQU25", "06/20/2025", "09/19/2025"),
    ("NQZ25", "09/19/2025", "12/19/2025"),
    ("NQH26", "12/19/2025", "03/20/2026"),
    ("NQM26", "03/20/2026", "03/26/2026"),
]


def split_into_chunks(start_str, end_str, chunk_days=20):
    """Split a date range into ~20 day chunks."""
    start = datetime.strptime(start_str, "%m/%d/%Y")
    end = datetime.strptime(end_str, "%m/%d/%Y")
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current.strftime("%m/%d/%Y"), chunk_end.strftime("%m/%d/%Y")))
        current = chunk_end + timedelta(days=1)
    return chunks


def get_barchart_cookies():
    raw = chrome_cookies("https://www.barchart.com")
    pw_cookies = []
    for name, value in raw.items():
        pw_cookies.append({
            "name": name,
            "value": value,
            "domain": ".barchart.com",
            "path": "/",
        })
    print(f"Extracted {len(pw_cookies)} cookies (read-only)")
    return pw_cookies


async def download_chunk(page, symbol, date_from, date_to, chunk_idx):
    """Download one chunk of a contract."""
    from_input = page.locator('input[name="dateFrom"]')
    to_input = page.locator('input[name="dateTo"]')

    await from_input.click(click_count=3)
    await from_input.fill(date_from)
    await from_input.press("Tab")
    await asyncio.sleep(0.5)

    await to_input.click(click_count=3)
    await to_input.fill(date_to)
    await to_input.press("Tab")
    await asyncio.sleep(4)

    try:
        async with page.expect_download(timeout=30000) as download_info:
            await page.locator('a.download-btn').click()
        download = await download_info.value
    except Exception:
        async with page.expect_download(timeout=30000) as download_info:
            await page.evaluate("document.querySelector('a.download-btn').click()")
        download = await download_info.value

    filename = f"{symbol}_chunk{chunk_idx}_{date_from.replace('/', '-')}_{date_to.replace('/', '-')}.csv"
    save_path = DATA_DIR / filename
    await download.save_as(str(save_path))
    return save_path


async def download_contract(page, symbol, active_from, active_to, contract_idx, total):
    """Navigate to contract page and download all chunks."""
    url = f"https://www.barchart.com/futures/quotes/{symbol}/historical-download"
    print(f"\n{'='*50}")
    print(f"[{contract_idx}/{total}] {symbol}: {active_from} -> {active_to}")

    await page.goto(url, wait_until="domcontentloaded", timeout=60000)

    try:
        await page.wait_for_selector('input[name="dateFrom"]', timeout=20000)
    except Exception:
        await asyncio.sleep(5)
        if await page.locator('input[name="dateFrom"]').count() == 0:
            print(f"  SKIP: Form not found for {symbol}")
            return []

    # Split into 20-day chunks
    chunks = split_into_chunks(active_from, active_to, chunk_days=20)
    print(f"  {len(chunks)} chunks to download")

    paths = []
    for ci, (cf, ct) in enumerate(chunks, 1):
        print(f"  chunk {ci}/{len(chunks)}: {cf} -> {ct}", end="")
        try:
            path = await download_chunk(page, symbol, cf, ct, ci)
            size = path.stat().st_size
            if size <= 150:
                print(f" EMPTY ({size}b)")
            else:
                print(f" OK ({size:,}b)")
                paths.append(path)
        except Exception as e:
            print(f" FAILED: {e}")
        await asyncio.sleep(2)

    return paths


async def main():
    # Count total chunks
    total_chunks = 0
    for _, af, at in CONTRACTS:
        total_chunks += len(split_into_chunks(af, at, 20))

    print(f"NQ Full Download | {len(CONTRACTS)} contracts | ~{total_chunks} chunks")
    print(f"Output: {DATA_DIR.resolve()}\n")

    pw_cookies = get_barchart_cookies()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        await context.add_cookies(pw_cookies)
        page = await context.new_page()

        all_files = []
        for idx, (symbol, af, at) in enumerate(CONTRACTS, 1):
            paths = await download_contract(page, symbol, af, at, idx, len(CONTRACTS))
            all_files.extend(paths)
            await asyncio.sleep(1)

        await browser.close()

    print(f"\n{'='*60}")
    print(f"Done! {len(all_files)} chunk files with data")
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Saved to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
