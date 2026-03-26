"""
Download NQ E-mini Nasdaq 100 individual contract 1-minute data from Barchart.
Downloads each quarterly contract separately, then stitches into continuous series.

Contracts: H(Mar), M(Jun), U(Sep), Z(Dec) for 2022-2026.
Each contract's active period: ~3 months before expiry.

Uses Playwright's own Chromium + injected cookies. Does NOT touch Chrome.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from pycookiecheat import chrome_cookies
from playwright.async_api import async_playwright

DATA_DIR = Path("data/barchart_nq")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NQ contract months and their approximate active trading periods
# Format: (symbol_suffix, active_from, active_to)
# Active period: from previous contract's expiry to this contract's expiry
# Expiry: 3rd Friday of the month
CONTRACTS = [
    # 2022
    ("NQH22", "12/17/2021", "03/18/2022"),
    ("NQM22", "03/18/2022", "06/17/2022"),
    ("NQU22", "06/17/2022", "09/16/2022"),
    ("NQZ22", "09/16/2022", "12/16/2022"),
    # 2023
    ("NQH23", "12/16/2022", "03/17/2023"),
    ("NQM23", "03/17/2023", "06/16/2023"),
    ("NQU23", "06/16/2023", "09/15/2023"),
    ("NQZ23", "09/15/2023", "12/15/2023"),
    # 2024
    ("NQH24", "12/15/2023", "03/15/2024"),
    ("NQM24", "03/15/2024", "06/21/2024"),
    ("NQU24", "06/21/2024", "09/20/2024"),
    ("NQZ24", "09/20/2024", "12/20/2024"),
    # 2025
    ("NQH25", "12/20/2024", "03/21/2025"),
    ("NQM25", "03/21/2025", "06/20/2025"),
    ("NQU25", "06/20/2025", "09/19/2025"),
    ("NQZ25", "09/19/2025", "12/19/2025"),
    # 2026
    ("NQH26", "12/19/2025", "03/20/2026"),
    ("NQM26", "03/20/2026", "06/19/2026"),
]


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


async def download_contract(page, symbol, date_from, date_to, idx, total):
    """Navigate to a specific contract page and download its data."""
    url = f"https://www.barchart.com/futures/quotes/{symbol}/historical-download"
    print(f"\n[{idx}/{total}] {symbol}: {date_from} -> {date_to}")
    print(f"  Loading {url}")

    await page.goto(url, wait_until="domcontentloaded", timeout=60000)

    try:
        await page.wait_for_selector('input[name="dateFrom"]', timeout=20000)
    except Exception:
        await asyncio.sleep(5)
        if await page.locator('input[name="dateFrom"]').count() == 0:
            print(f"  SKIP: Form not found for {symbol}")
            return None

    # Set dates
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

    # Download
    try:
        async with page.expect_download(timeout=30000) as download_info:
            await page.locator('a.download-btn').click()
        download = await download_info.value
    except Exception:
        print("  Retrying with JS click...")
        try:
            async with page.expect_download(timeout=30000) as download_info:
                await page.evaluate("document.querySelector('a.download-btn').click()")
            download = await download_info.value
        except Exception as e:
            print(f"  FAILED: {e}")
            return None

    filename = f"{symbol}_1min.csv"
    save_path = DATA_DIR / filename
    await download.save_as(str(save_path))

    size = save_path.stat().st_size
    if size <= 150:
        print(f"  EMPTY: {filename} ({size} bytes)")
    else:
        print(f"  OK: {filename} ({size:,} bytes)")
    return save_path


async def main():
    print(f"NQ Individual Contracts | {len(CONTRACTS)} contracts")
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

        downloaded = []
        for idx, (symbol, dfrom, dto) in enumerate(CONTRACTS, 1):
            path = await download_contract(page, symbol, dfrom, dto, idx, len(CONTRACTS))
            if path and path.stat().st_size > 150:
                downloaded.append(path)
            await asyncio.sleep(2)

        await browser.close()

    print(f"\n{'='*60}")
    print(f"Done! {len(downloaded)}/{len(CONTRACTS)} contracts downloaded")
    print(f"Saved to: {DATA_DIR.resolve()}")

    if downloaded:
        print("\nFiles with data:")
        for p in downloaded:
            print(f"  {p.name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    asyncio.run(main())
