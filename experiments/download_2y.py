"""Download 2 years of QQQ 1-minute data from Alpaca and run backtest."""

from __future__ import annotations
import sys, time, requests
from pathlib import Path
from datetime import datetime

import pandas as pd

API_KEY = "PKR6NHBWU34DU2FRSWBRD3JJIV"
SECRET  = "EuBBJmqJtyEvBMWgrohSaASfJEHxnx28YNWwt41sqjeZ"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET}
DATA_URL = "https://data.alpaca.markets/v2/stocks/QQQ/bars"

CACHE_PATH = Path("data/QQQ_1Min_2024-03-22_2026-03-22.csv")


def download_1m(start_date: str, end_date: str) -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} bars")
        return df

    print(f"Downloading QQQ 1m bars: {start_date} → {end_date}")
    all_bars = []
    page_token = None
    page = 0

    while True:
        params = {
            "timeframe": "1Min",
            "start": start_date,
            "end": end_date,
            "limit": 10000,
            "feed": "iex",
            "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token

        r = requests.get(DATA_URL, headers=HEADERS, params=params)
        if r.status_code != 200:
            print(f"API error {r.status_code}: {r.text[:200]}")
            break

        data = r.json()
        bars = data.get("bars", [])
        all_bars.extend(bars)
        page += 1

        if page % 10 == 0:
            print(f"  Page {page}: {len(all_bars)} bars...")

        page_token = data.get("next_page_token")
        if not page_token or not bars:
            break

        time.sleep(0.2)

    print(f"Downloaded {len(all_bars)} bars in {page} pages")

    if not all_bars:
        raise ValueError("No data downloaded")

    df = pd.DataFrame(all_bars)
    df = df.rename(columns={"t": "timestamp", "o": "Open", "h": "High",
                             "l": "Low", "c": "Close", "v": "Volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return df


if __name__ == "__main__":
    df = download_1m("2024-03-22", "2026-03-22")
    print(f"\nTotal bars: {len(df)}")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print(f"Trading days: {df.index.normalize().nunique()}")
