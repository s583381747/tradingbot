"""
Download 2 years of QQQ 1-minute data via yfinance in 7-day chunks.

yfinance allows max 7 days per request for 1m data, but we can loop
back in time to accumulate a larger dataset.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

SYMBOL = "QQQ"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INTERVAL = "1m"
CHUNK_DAYS = 6  # yfinance max is 7 days for 1m, use 6 for safety
TARGET_YEARS = 2


def download_chunks(symbol: str, end_date: datetime, years: int) -> pd.DataFrame:
    """Download 1-min data in 7-day chunks going back `years` years."""
    start_target = end_date - timedelta(days=365 * years)
    all_chunks = []
    current_end = end_date
    attempt = 0
    max_attempts = 365 * years // CHUNK_DAYS + 10
    consecutive_empty = 0

    print(f"Downloading {symbol} 1m data from {start_target.date()} to {end_date.date()}")
    print(f"Chunk size: {CHUNK_DAYS} days, est. {max_attempts} chunks")
    print()

    while current_end > start_target and attempt < max_attempts:
        current_start = current_end - timedelta(days=CHUNK_DAYS)
        if current_start < start_target:
            current_start = start_target

        attempt += 1
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval=INTERVAL,
                prepost=False,
            )

            if df is not None and len(df) > 0:
                all_chunks.append(df)
                consecutive_empty = 0
                bars = len(df)
                first = df.index[0].strftime("%Y-%m-%d")
                last = df.index[-1].strftime("%Y-%m-%d")
                print(f"  [{attempt}] {first} → {last}: {bars} bars", flush=True)
            else:
                consecutive_empty += 1
                print(f"  [{attempt}] {current_start.date()} → {current_end.date()}: empty", flush=True)

                # yfinance 1m data only goes back ~30 days
                if consecutive_empty >= 5:
                    print(f"\n  Hit data limit after {consecutive_empty} consecutive empty chunks.")
                    print(f"  yfinance 1m data likely only available for last ~30 days.")
                    break

        except Exception as e:
            print(f"  [{attempt}] Error: {e}", flush=True)
            consecutive_empty += 1
            if consecutive_empty >= 5:
                break

        current_end = current_start
        time.sleep(0.3)  # rate limit

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def try_longer_interval(symbol: str, end_date: datetime, years: int) -> pd.DataFrame:
    """Fallback: download 5-min data which has longer history."""
    start = end_date - timedelta(days=60)  # yfinance 5m max ~60 days
    print(f"\nFallback: downloading {symbol} 5m data (last 60 days)...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval="5m", prepost=False)
    if df is not None and len(df) > 0:
        print(f"  Got {len(df)} bars of 5m data")
    return df if df is not None else pd.DataFrame()


def try_hourly(symbol: str, end_date: datetime, years: int) -> pd.DataFrame:
    """Fallback: download 1h data which goes back 2 years."""
    start = end_date - timedelta(days=365 * years)
    print(f"\nFallback: downloading {symbol} 1h data ({years} years)...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval="1h", prepost=False)
    if df is not None and len(df) > 0:
        print(f"  Got {len(df)} bars of 1h data")
    return df if df is not None else pd.DataFrame()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to match existing data format."""
    df = df.copy()
    # Remove timezone info
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "timestamp"
    # Keep only needed columns
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = df[cols]
    # Add placeholder columns to match existing format
    df["trade_count"] = 0
    df["vwap"] = (df["High"] + df["Low"] + df["Close"]) / 3
    return df


def main():
    end_date = datetime.now()

    # Try 1m first
    df_1m = download_chunks(SYMBOL, end_date, TARGET_YEARS)

    if len(df_1m) > 0:
        df_1m = normalize_df(df_1m)
        first = df_1m.index[0].strftime("%Y-%m-%d")
        last = df_1m.index[-1].strftime("%Y-%m-%d")
        fname = f"{SYMBOL}_1Min_{first}_{last}.csv"
        path = os.path.join(OUTPUT_DIR, fname)
        df_1m.to_csv(path)
        print(f"\n1m data saved: {path}")
        print(f"  {len(df_1m)} bars, {first} to {last}")
    else:
        print("\nNo 1m data retrieved.")

    # Also try 5m (60 days)
    df_5m = try_longer_interval(SYMBOL, end_date, TARGET_YEARS)
    if len(df_5m) > 0:
        df_5m = normalize_df(df_5m)
        first = df_5m.index[0].strftime("%Y-%m-%d")
        last = df_5m.index[-1].strftime("%Y-%m-%d")
        fname = f"{SYMBOL}_5Min_{first}_{last}.csv"
        path = os.path.join(OUTPUT_DIR, fname)
        df_5m.to_csv(path)
        print(f"\n5m data saved: {path}")
        print(f"  {len(df_5m)} bars, {first} to {last}")

    # Also get 1h for 2 years
    df_1h = try_hourly(SYMBOL, end_date, TARGET_YEARS)
    if len(df_1h) > 0:
        df_1h = normalize_df(df_1h)
        first = df_1h.index[0].strftime("%Y-%m-%d")
        last = df_1h.index[-1].strftime("%Y-%m-%d")
        fname = f"{SYMBOL}_1Hour_{first}_{last}.csv"
        path = os.path.join(OUTPUT_DIR, fname)
        df_1h.to_csv(path)
        print(f"\n1h data saved: {path}")
        print(f"  {len(df_1h)} bars, {first} to {last}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if len(df_1m) > 0:
        print(f"  1m: {len(df_1m)} bars ({df_1m.index[0].date()} to {df_1m.index[-1].date()})")
    if len(df_5m) > 0:
        print(f"  5m: {len(df_5m)} bars ({df_5m.index[0].date()} to {df_5m.index[-1].date()})")
    if len(df_1h) > 0:
        print(f"  1h: {len(df_1h)} bars ({df_1h.index[0].date()} to {df_1h.index[-1].date()})")

    existing = os.path.join(OUTPUT_DIR, "QQQ_1Min_2025-09-21_2026-03-21.csv")
    if os.path.exists(existing):
        existing_df = pd.read_csv(existing)
        print(f"\n  Existing 1m data: {len(existing_df)} bars (2025-09-21 to 2026-03-21)")

    print("\nNote: yfinance 1m data is limited to ~30 days.")
    print("For 2-year 1m data, you need Alpaca API or Polygon.io.")
    print("The 1h data covers 2 years and can validate the strategy at hourly level.")


if __name__ == "__main__":
    main()
