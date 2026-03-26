"""
Download NQ (E-mini Nasdaq 100 Futures) 1-minute data from Barchart.
4 years: 2022-03-26 to 2026-03-26.

Uses Playwright's own Chromium + injected cookies. Does NOT touch Chrome.
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from pycookiecheat import chrome_cookies
from playwright.async_api import async_playwright

DATA_DIR = Path("data/barchart_nq")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NQ*0 = E-mini Nasdaq 100 continuous front-month
URL = "https://www.barchart.com/futures/quotes/NQ*0/historical-download"


def generate_date_ranges():
    ranges = []
    start = datetime(2022, 3, 26)
    end = datetime(2026, 3, 26)
    current = start
    while current < end:
        next_date = current + timedelta(days=31)
        if next_date > end:
            next_date = end
        ranges.append((current.strftime("%m/%d/%Y"), next_date.strftime("%m/%d/%Y")))
        current = next_date
    return ranges


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


async def download_one_range(page, date_from, date_to, idx, total):
    print(f"\n[{idx}/{total}] {date_from} -> {date_to}")

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
        print("  Retrying with JS click...")
        async with page.expect_download(timeout=30000) as download_info:
            await page.evaluate("document.querySelector('a.download-btn').click()")
        download = await download_info.value

    filename = f"NQ_1min_{date_from.replace('/', '-')}_{date_to.replace('/', '-')}.csv"
    save_path = DATA_DIR / filename
    await download.save_as(str(save_path))

    size = save_path.stat().st_size
    if size <= 150:
        print(f"  EMPTY: {filename} ({size} bytes)")
    else:
        print(f"  OK: {filename} ({size:,} bytes)")
    return save_path


async def main():
    ranges = generate_date_ranges()
    print(f"NQ E-mini Nasdaq 100 Futures | 1-min data | {len(ranges)} ranges")
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

        print("Loading NQ futures page...")
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)

        print("Waiting for download form...")
        try:
            await page.wait_for_selector('input[name="dateFrom"]', timeout=30000)
            print("Form ready!")
        except Exception:
            await asyncio.sleep(8)
            has_form = await page.locator('input[name="dateFrom"]').count()
            if has_form == 0:
                await page.screenshot(path=str(DATA_DIR / "debug.png"))
                text = await page.inner_text('body')
                lines = [l.strip() for l in text.split('\n') if l.strip()][:15]
                print("Form NOT found. Page:")
                for l in lines:
                    print(f"  {l[:100]}")
                await browser.close()
                return

        title = await page.title()
        print(f"Page: {title}")

        # Probe: download first range to check data availability
        print("\n--- Probing data availability ---")
        await page.screenshot(path=str(DATA_DIR / "ready.png"))

        downloaded = []
        empty_count = 0
        for idx, (df, dt) in enumerate(ranges, 1):
            try:
                path = await download_one_range(page, df, dt, idx, len(ranges))
                size = path.stat().st_size
                if size > 150:
                    downloaded.append(path)
                else:
                    empty_count += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                await page.screenshot(path=str(DATA_DIR / f"error_{idx}.png"))
            await asyncio.sleep(2)

        await browser.close()

    print(f"\n{'='*60}")
    print(f"Done! {len(downloaded)} files with data, {empty_count} empty")
    print(f"Saved to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
