"""
Merge NQ contract chunks into a continuous 1-minute series.

Strategy:
1. Load all chunks per contract, merge & deduplicate
2. For rollover: use data from the MORE LIQUID contract
   - Roll 1 day before expiry (Thursday before expiry Friday)
   - At roll point, check overlap and use new contract from roll date forward
3. Apply Panama Canal back-adjustment (add price diff at roll to all prior data)
4. Filter to RTH (09:30-16:00 ET = 08:30-15:00 CT) for clean output
5. Also save full session (Globex) version

Output:
  - NQ_1min_continuous_RTH.csv (Regular Trading Hours only)
  - NQ_1min_continuous_full.csv (Full Globex session)
  - NQ_rollover_report.txt (audit of each roll)
"""
import pandas as pd
from pathlib import Path
from datetime import time as dtime
import re

DATA_DIR = Path("data/barchart_nq")

# Contract order and approximate expiry dates
CONTRACTS = [
    ("NQH22", "2022-03-18"),
    ("NQM22", "2022-06-17"),
    ("NQU22", "2022-09-16"),
    ("NQZ22", "2022-12-16"),
    ("NQH23", "2023-03-17"),
    ("NQM23", "2023-06-16"),
    ("NQU23", "2023-09-15"),
    ("NQZ23", "2023-12-15"),
    ("NQH24", "2024-03-15"),
    ("NQM24", "2024-06-21"),
    ("NQU24", "2024-09-20"),
    ("NQZ24", "2024-12-20"),
    ("NQH25", "2025-03-21"),
    ("NQM25", "2025-06-20"),
    ("NQU25", "2025-09-19"),
    ("NQZ25", "2025-12-19"),
    ("NQH26", "2026-03-20"),
    ("NQM26", "2026-06-19"),
]


def load_contract(symbol):
    """Load all chunks for a contract and merge."""
    chunks = sorted(DATA_DIR.glob(f"{symbol}_chunk*.csv"))
    if not chunks:
        # Try the single-file version
        single = DATA_DIR / f"{symbol}_1min.csv"
        if single.exists():
            chunks = [single]
        else:
            print(f"  WARNING: No files found for {symbol}")
            return pd.DataFrame()

    dfs = []
    for chunk in chunks:
        df = pd.read_csv(chunk, skipfooter=1, engine="python")
        # Clean up: remove footer row if present
        df = df[~df["Time"].str.contains("Downloaded", na=False)]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first")
    df = df.set_index("Time")

    # Rename columns
    df = df.rename(columns={"Latest": "Close"})
    # Keep only OHLCV
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    print(f"  {symbol}: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")
    return df


def audit_contract(df, symbol):
    """Quick quality check."""
    if df.empty:
        return {}

    stats = {
        "symbol": symbol,
        "bars": len(df),
        "first": str(df.index[0]),
        "last": str(df.index[-1]),
        "trading_days": df.index.normalize().nunique(),
        "zero_range": int((df["High"] == df["Low"]).sum()),
        "zero_range_pct": round((df["High"] == df["Low"]).mean() * 100, 2),
        "avg_volume": round(df["Volume"].mean(), 1),
        "price_range": f"{df['Close'].min():.2f} - {df['Close'].max():.2f}",
    }
    return stats


def main():
    print("=" * 60)
    print("NQ CONTINUOUS SERIES BUILDER")
    print("=" * 60)

    # Step 1: Load all contracts
    print("\n--- Loading contracts ---")
    contracts = {}
    all_stats = []
    for symbol, expiry in CONTRACTS:
        df = load_contract(symbol)
        if not df.empty:
            contracts[symbol] = df
            stats = audit_contract(df, symbol)
            all_stats.append(stats)

    # Step 2: Audit report
    print(f"\n--- Contract Audit ({len(contracts)} contracts) ---")
    print(f"{'Symbol':<10} {'Bars':>8} {'Days':>5} {'ZR%':>6} {'AvgVol':>10} {'Price Range':<25} {'First':<20} {'Last':<20}")
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['bars']:>8,} {s['trading_days']:>5} {s['zero_range_pct']:>5.1f}% {s['avg_volume']:>10,.0f} {s['price_range']:<25} {s['first'][:19]:<20} {s['last'][:19]:<20}")

    # Step 3: Rollover analysis
    print("\n--- Rollover Analysis ---")
    print("Roll strategy: 1 business day before expiry (Thursday before expiry Friday)")
    print(f"{'Roll':<20} {'Expiring':>12} {'New':>12} {'Gap':>8} {'Gap%':>8} {'Overlap bars':>12}")

    roll_report = []
    contract_list = list(contracts.items())

    for i in range(len(contract_list) - 1):
        old_sym, old_df = contract_list[i]
        new_sym, new_df = contract_list[i + 1]

        # Find expiry date
        expiry_str = [e for s, e in CONTRACTS if s == old_sym][0]
        expiry = pd.Timestamp(expiry_str)

        # Roll date = 1 business day before expiry
        roll_date = expiry - pd.tseries.offsets.BDay(1)

        # Find overlap: bars that exist in both contracts
        overlap_idx = old_df.index.intersection(new_df.index)

        # Get prices at roll point (closest bar to roll date 15:00 CT / 16:00 ET)
        roll_ts = pd.Timestamp(f"{roll_date.date()} 15:00")  # 15:00 CT = market close

        # Find nearest bar before roll timestamp
        old_near_roll = old_df.index[old_df.index <= roll_ts]
        new_near_roll = new_df.index[new_df.index <= roll_ts]

        if len(old_near_roll) > 0 and len(new_near_roll) > 0:
            old_price = old_df.loc[old_near_roll[-1], "Close"]
            new_price = new_df.loc[new_near_roll[-1], "Close"]
            gap = new_price - old_price
            gap_pct = gap / old_price * 100
        else:
            # Fallback: use last/first available
            old_price = old_df["Close"].iloc[-1]
            new_price = new_df["Close"].iloc[0]
            gap = new_price - old_price
            gap_pct = gap / old_price * 100

        roll_info = {
            "transition": f"{old_sym}->{new_sym}",
            "roll_date": str(roll_date.date()),
            "old_price": old_price,
            "new_price": new_price,
            "gap": gap,
            "gap_pct": gap_pct,
            "overlap_bars": len(overlap_idx),
        }
        roll_report.append(roll_info)
        print(f"{roll_info['transition']:<20} {old_price:>12.2f} {new_price:>12.2f} {gap:>+8.2f} {gap_pct:>+7.2f}% {len(overlap_idx):>12,}")

    # Step 4: Build continuous series with back-adjustment
    print("\n--- Building continuous series (Panama Canal back-adjustment) ---")

    # Work backwards: start from most recent contract, adjust older ones
    # Cumulative adjustment starts at 0 for most recent
    cum_adjustment = 0.0
    adjusted_dfs = []

    for i in range(len(contract_list) - 1, -1, -1):
        sym, df = contract_list[i]
        expiry_str = [e for s, e in CONTRACTS if s == sym][0]
        expiry = pd.Timestamp(expiry_str)

        if i < len(contract_list) - 1:
            # Determine roll boundary: use data up to roll date
            roll_date = expiry - pd.tseries.offsets.BDay(1)
            # Keep bars up to end of roll date
            roll_end = pd.Timestamp(f"{roll_date.date()} 23:59:59")
            df_slice = df[df.index <= roll_end].copy()
        else:
            # Last contract: use all data
            df_slice = df.copy()

        if i > 0:
            # Not the first contract: trim start to after previous contract's roll
            prev_sym = contract_list[i - 1][0]
            prev_expiry_str = [e for s, e in CONTRACTS if s == prev_sym][0]
            prev_expiry = pd.Timestamp(prev_expiry_str)
            prev_roll = prev_expiry - pd.tseries.offsets.BDay(1)
            roll_start = pd.Timestamp(f"{prev_roll.date()} 15:01")
            df_slice = df_slice[df_slice.index > roll_start]

        # Apply back-adjustment
        df_slice = df_slice.copy()
        df_slice[["Open", "High", "Low", "Close"]] += cum_adjustment

        print(f"  {sym}: {len(df_slice):,} bars, adj={cum_adjustment:+.2f}")
        adjusted_dfs.append(df_slice)

        # Accumulate adjustment for older contracts
        if i > 0:
            gap = roll_report[i - 1]["gap"]
            cum_adjustment -= gap

    # Reverse (oldest first) and concat
    adjusted_dfs.reverse()
    continuous = pd.concat(adjusted_dfs)
    continuous = continuous.sort_index()
    continuous = continuous[~continuous.index.duplicated(keep="first")]

    print(f"\n  Total continuous: {len(continuous):,} bars")
    print(f"  Range: {continuous.index[0]} -> {continuous.index[-1]}")
    print(f"  Trading days: {continuous.index.normalize().nunique()}")

    # Step 5: Save full session
    full_path = DATA_DIR / "NQ_1min_continuous_full.csv"
    continuous.to_csv(full_path)
    print(f"\n  Saved full session: {full_path} ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 6: Filter to RTH (08:30-15:00 CT = 09:30-16:00 ET)
    rth = continuous.between_time(dtime(8, 30), dtime(15, 0))
    rth_path = DATA_DIR / "NQ_1min_continuous_RTH.csv"
    rth.to_csv(rth_path)
    print(f"  Saved RTH only: {rth_path} ({rth_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  RTH bars: {len(rth):,}, trading days: {rth.index.normalize().nunique()}")

    # Step 7: Save rollover report
    report_path = DATA_DIR / "NQ_rollover_report.txt"
    with open(report_path, "w") as f:
        f.write("NQ Continuous Series - Rollover Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Strategy: Panama Canal back-adjustment\n")
        f.write("Roll: 1 business day before expiry (Thursday before Friday expiry)\n\n")
        for r in roll_report:
            f.write(f"{r['transition']}: roll={r['roll_date']}, "
                    f"old={r['old_price']:.2f}, new={r['new_price']:.2f}, "
                    f"gap={r['gap']:+.2f} ({r['gap_pct']:+.2f}%), "
                    f"overlap={r['overlap_bars']} bars\n")
        f.write(f"\nTotal back-adjustment (oldest): {cum_adjustment:+.2f}\n")
        f.write(f"Continuous bars: {len(continuous):,}\n")
        f.write(f"RTH bars: {len(rth):,}\n")
    print(f"  Saved report: {report_path}")

    # Final quality summary
    print(f"\n{'='*60}")
    print("FINAL QUALITY SUMMARY")
    print(f"{'='*60}")
    zr = (rth["High"] == rth["Low"]).sum()
    print(f"  RTH bars: {len(rth):,}")
    print(f"  Trading days: {rth.index.normalize().nunique()}")
    print(f"  Zero-range bars: {zr} ({zr/len(rth)*100:.2f}%)")
    print(f"  Avg volume/bar: {rth['Volume'].mean():,.0f}")
    print(f"  Price range: {rth['Close'].min():.2f} -> {rth['Close'].max():.2f}")


if __name__ == "__main__":
    main()
