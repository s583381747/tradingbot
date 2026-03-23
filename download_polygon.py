"""
Download 2 years of QQQ 1-minute data from Polygon.io (SIP quality).
Uses multi-day ranges to minimize API calls.

Free tier: 5 calls/min. Strategy: request 5-day chunks → ~100 calls for 2 years.
"""
from __future__ import annotations
import os, sys, time, functools
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# Force flush on every print
print = functools.partial(print, flush=True)

API_KEY = os.environ.get("POLYGON_KEY", "")
BASE_URL = "https://api.polygon.io/v2/aggs/ticker/QQQ/range/1/minute"
CACHE_PATH = Path("data/QQQ_1Min_Polygon_2y_raw.csv")
CLEAN_PATH = Path("data/QQQ_1Min_Polygon_2y_clean.csv")

# Free tier: 5 calls/min → be conservative
RATE_LIMIT_DELAY = 15


def download_range(start: str, end: str) -> list[dict]:
    """Download 1-min bars for a date range. Handles pagination + retries."""
    all_bars = []
    url = f"{BASE_URL}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    while True:
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=30)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                wait = 30 * (attempt + 1)
                print(f"    Connection error, retry {attempt+1}/3 in {wait}s...")
                time.sleep(wait)
        else:
            print(f"    Failed after 3 retries for {start}→{end}")
            break

        if r.status_code == 429:
            print("    Rate limited, waiting 90s...")
            time.sleep(90)
            continue
        if r.status_code != 200:
            print(f"    API error {r.status_code}: {r.text[:100]}")
            break

        data = r.json()
        results = data.get("results", [])
        all_bars.extend(results)

        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": API_KEY}
            time.sleep(RATE_LIMIT_DELAY)
        else:
            break

    return all_bars


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Convert Polygon bar dicts to DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df.set_index("timestamp")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _save_partial(bars, partial_path, existing_df=None):
    """Save incremental progress."""
    df = _bars_to_df(bars)
    if existing_df is not None and len(existing_df) > 0:
        df = pd.concat([existing_df, df])
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(partial_path)
    print(f"    [saved partial: {len(df)} bars to {partial_path}]")


def download_all(start: str, end: str) -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"Loading cached raw data from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, index_col="timestamp", parse_dates=True)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    all_bars = []
    current = start_dt
    chunk = 0
    partial_path = Path("data/_polygon_partial.csv")
    existing_partial = None

    # Resume from partial download if available
    if partial_path.exists():
        existing_partial = pd.read_csv(partial_path, index_col="timestamp", parse_dates=True)
        last_ts = existing_partial.index[-1]
        print(f"  Resuming from partial: {len(existing_partial)} bars, last={last_ts}")
        resume_date = last_ts.date() + timedelta(days=1)
        current = max(start_dt, datetime(resume_date.year, resume_date.month, resume_date.day))

    # Download in 7-day chunks to reduce API calls
    while current < end_dt:
        chunk_end = min(current + timedelta(days=6), end_dt)
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        bars = download_range(s, e)
        all_bars.extend(bars)
        chunk += 1

        print(f"  Chunk {chunk}: {s}→{e}, +{len(bars)} bars, total {len(all_bars)}")

        # Incremental save every 10 chunks
        if chunk % 10 == 0 and all_bars:
            _save_partial(all_bars, partial_path, existing_partial)

        current = chunk_end + timedelta(days=1)
        time.sleep(RATE_LIMIT_DELAY)

    print(f"Downloaded {len(all_bars)} new bars in {chunk} chunks")

    if not all_bars and not partial_path.exists():
        raise ValueError("No data. Check POLYGON_KEY.")

    df = _bars_to_df(all_bars)

    # Merge with partial if we resumed
    if existing_partial is not None and len(existing_partial) > 0:
        df = pd.concat([existing_partial, df])
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

    # Clean up partial
    if partial_path.exists():
        partial_path.unlink()
    df = df[~df.index.duplicated(keep="first")]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"Cached raw to {CACHE_PATH}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    import datetime as dt
    t = df.index.time
    mask = (t >= dt.time(9, 30)) & (t < dt.time(16, 0))
    df = df[mask].copy()
    df.to_csv(CLEAN_PATH)
    return df


def main():
    if not API_KEY:
        print("ERROR: Set POLYGON_KEY. Get free key at https://polygon.io/dashboard/signup")
        sys.exit(1)

    print(f"Polygon API Key: {API_KEY[:8]}...")
    print("Downloading QQQ 1m: 2024-03-22 → 2026-03-22")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY}s between calls\n")

    df = download_all("2024-03-22", "2026-03-22")
    print(f"\nRaw: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    df_clean = clean(df)
    days = df_clean.index.normalize().nunique()
    print(f"Clean: {len(df_clean)} bars, {days} trading days")

    # Quality report
    zero_range = (df_clean["High"] == df_clean["Low"]).sum()
    avg_vol = df_clean["Volume"].mean()
    bars_per_day = len(df_clean) / days if days > 0 else 0

    print(f"\n{'='*60}")
    print("DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Bars/day avg: {bars_per_day:.0f} (expect ~390 for 1m)")
    print(f"  Zero-range (H=L): {zero_range} ({zero_range/len(df_clean)*100:.2f}%)")
    print(f"  Avg volume/bar: {avg_vol:,.0f}")

    # Compare with IEX
    iex_path = Path("data/QQQ_1Min_2y_clean.csv")
    if iex_path.exists():
        iex = pd.read_csv(iex_path, index_col="timestamp", parse_dates=True)
        iex_zr = (iex["High"] == iex["Low"]).sum()
        iex_vol = iex["Volume"].mean()
        print(f"\n  vs IEX data:")
        print(f"    IEX bars: {len(iex)}, Polygon: {len(df_clean)}")
        print(f"    IEX zero-range: {iex_zr} ({iex_zr/len(iex)*100:.1f}%) vs Polygon: {zero_range} ({zero_range/len(df_clean)*100:.2f}%)")
        print(f"    IEX avg vol: {iex_vol:,.0f} vs Polygon: {avg_vol:,.0f}")

    print(f"\nSaved to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
