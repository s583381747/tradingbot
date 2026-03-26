"""
Download QQQ 1-minute data from Barchart using Playwright + Chrome cookies.
Extracts cookies from running Chrome, uses fresh Playwright browser.
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from pycookiecheat import chrome_cookies
from playwright.async_api import async_playwright

DATA_DIR = Path("data/barchart")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://www.barchart.com/stocks/quotes/QQQ/historical-download"


def generate_date_ranges():
    """Generate ~1 month date ranges from 2022-03-22 to 2024-03-22."""
    ranges = []
    start = datetime(2022, 3, 22)
    end = datetime(2024, 3, 22)
    current = start
    while current < end:
        next_date = current + timedelta(days=31)
        if next_date > end:
            next_date = end
        ranges.append((current.strftime("%m/%d/%Y"), next_date.strftime("%m/%d/%Y")))
        current = next_date
    return ranges


def get_barchart_cookies():
    """Extract Barchart cookies from running Chrome."""
    cookies = chrome_cookies("https://www.barchart.com")
    # Convert to Playwright cookie format
    pw_cookies = []
    for name, value in cookies.items():
        pw_cookies.append({
            "name": name,
            "value": value,
            "domain": ".barchart.com",
            "path": "/",
        })
    print(f"Extracted {len(pw_cookies)} cookies from Chrome")
    return pw_cookies


async def download_one_range(page, date_from, date_to, idx, total):
    """Set date range and click download."""
    print(f"\n[{idx}/{total}] {date_from} -> {date_to}")

    # Clear and set dateFrom
    await page.evaluate("""
        (dates) => {
            const fromInput = document.querySelector('input[name="dateFrom"]');
            const toInput = document.querySelector('input[name="dateTo"]');
            // Use native setter to trigger React/Vue change events
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(fromInput, dates.from);
            fromInput.dispatchEvent(new Event('input', { bubbles: true }));
            fromInput.dispatchEvent(new Event('change', { bubbles: true }));
            nativeInputValueSetter.call(toInput, dates.to);
            toInput.dispatchEvent(new Event('input', { bubbles: true }));
            toInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
    """, {"from": date_from, "to": date_to})

    # Wait for table to reload
    await asyncio.sleep(4)

    # Click download button and capture file
    try:
        async with page.expect_download(timeout=30000) as download_info:
            await page.locator('a.download-btn').click()
        download = await download_info.value
    except Exception:
        # Fallback: try clicking with JS
        print("  Retrying download click via JS...")
        async with page.expect_download(timeout=30000) as download_info:
            await page.evaluate("document.querySelector('a.download-btn').click()")
        download = await download_info.value

    filename = f"QQQ_1min_{date_from.replace('/', '-')}_{date_to.replace('/', '-')}.csv"
    save_path = DATA_DIR / filename
    await download.save_as(str(save_path))

    size = save_path.stat().st_size
    print(f"  OK: {filename} ({size:,} bytes)")
    return save_path


async def main():
    ranges = generate_date_ranges()
    print(f"Date ranges: {len(ranges)}")
    print(f"Output: {DATA_DIR.resolve()}\n")

    # Get cookies from Chrome
    pw_cookies = get_barchart_cookies()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )

        # Inject cookies
        await context.add_cookies(pw_cookies)

        page = await context.new_page()

        # Navigate
        print("Loading Barchart page...")
        await page.goto(URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)

        # Verify we're logged in by checking page content
        title = await page.title()
        print(f"Page title: {title}")

        # Take a screenshot to verify
        await page.screenshot(path=str(DATA_DIR / "initial_page.png"))
        print("Screenshot saved to data/barchart/initial_page.png")

        # Check if time frame is set to Intraday and aggregation to 1
        agg = await page.locator('input[name="aggregation"]').input_value()
        print(f"Current aggregation: {agg} min")

        # Download each range
        downloaded = []
        for idx, (df, dt) in enumerate(ranges, 1):
            try:
                path = await download_one_range(page, df, dt, idx, len(ranges))
                downloaded.append(path)
            except Exception as e:
                print(f"  FAILED: {e}")
                await page.screenshot(path=str(DATA_DIR / f"error_{idx}.png"))

            # Polite delay
            await asyncio.sleep(2)

        await browser.close()

    print(f"\n{'='*60}")
    print(f"Done! {len(downloaded)}/{len(ranges)} files downloaded")
    print(f"Saved to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
