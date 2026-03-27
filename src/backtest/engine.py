"""
Shared backtest engine — CORRECT cost model for all agents.
All costs PER CONTRACT. Gap-through stop fill. No shortcuts.

Usage:
    from src.backtest_engine import load_nq, resample, backtest, compute_stats

Cost model (per contract):
    MNQ: comm=$2.46 RT, spread=$0.50, stop_slip=$1.00, be_slip=$1.00
    NQ:  comm=$2.46 RT, spread=$5.00, stop_slip=$1.25, be_slip=$1.25
"""
from __future__ import annotations
import datetime as dt, math
import numpy as np, pandas as pd

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"

COSTS = {
    "MNQ": {"pt_val": 2.0,  "comm_rt": 2.46, "spread": 0.50, "slip_stop": 1.00, "slip_be": 1.00},
    "NQ":  {"pt_val": 20.0, "comm_rt": 2.46, "spread": 5.00, "slip_stop": 1.25, "slip_be": 1.25},
}


def load_nq(path=NQ_PATH, start="2022-01-01"):
    """Load NQ continuous RTH data. Returns DataFrame indexed by ET timestamp."""
    nq = pd.read_csv(path, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)  # CT → ET
    return nq[nq.index >= start]


def resample(df, minutes):
    if minutes <= 1:
        return df
    return df.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_indicators(df, ema_fast=20, ema_slow=50, atr_period=14):
    """Add EMA and ATR. Returns copy."""
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
        np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                   (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(atr_period).mean()
    return df


def compute_trade_cost(nc, exit_type, instrument="MNQ"):
    """Compute total round-trip cost for a trade. All per contract."""
    c = COSTS[instrument]
    entry = c["comm_rt"] * nc / 2 + c["spread"] * nc
    exit_comm = c["comm_rt"] * nc / 2
    if exit_type in ("stop", "trail"):
        exit_slip = c["slip_stop"] * nc
    elif exit_type == "be":
        exit_slip = c["slip_be"] * nc
    else:
        exit_slip = 0
    return entry + exit_comm + exit_slip


def gap_through_fill(stop_price, bar_open, trend):
    """Model gap-through: fill at worst of (stop, open)."""
    if trend == 1:
        return min(stop_price, bar_open) if bar_open < stop_price else stop_price
    else:
        return max(stop_price, bar_open) if bar_open > stop_price else stop_price


def compute_stats(trades_df, starting_equity=50000):
    """Compute comprehensive stats from a trades DataFrame.

    Expected columns: pnl, r, exit, date (str), cost, risk
    """
    tdf = trades_df
    if len(tdf) == 0:
        return {"pf": 0, "n": 0}

    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    cum = tdf["pnl"].cumsum()
    dd = (cum.cummax() - cum).max()

    # Daily PnL
    daily = tdf.groupby("date")["pnl"].sum()
    days_arr = daily.values
    n_days = len(days_arr)
    years = n_days / 252

    # Sharpe, Sortino
    sharpe = (days_arr.mean() / days_arr.std()) * 252**0.5 if days_arr.std() > 0 else 0
    down = days_arr[days_arr < 0]
    sortino = (days_arr.mean() / down.std()) * 252**0.5 if len(down) > 0 and down.std() > 0 else 0

    apr = cum.iloc[-1] / starting_equity / max(years, 0.1) * 100
    calmar = apr / (dd / starting_equity * 100) if dd > 0 else 0

    # Consecutive losses
    is_loss = (tdf["r"] <= 0).values
    mcl = cur = 0
    for x in is_loss:
        if x: cur += 1; mcl = max(mcl, cur)
        else: cur = 0

    # DD in R
    cum_r = tdf["r"].cumsum()
    dd_r = (cum_r.cummax() - cum_r).max()

    return {
        "pf": round(gw / gl, 3) if gl > 0 else 0,
        "wr": round((tdf["r"] > 0).mean() * 100, 1),
        "dd": round(dd, 0),
        "dd_r": round(dd_r, 2),
        "pnl": round(cum.iloc[-1], 0),
        "n": len(tdf),
        "dpnl": round(cum.iloc[-1] / max(n_days, 1), 1),
        "b5": int((tdf["r"] >= 5).sum()),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "apr": round(apr, 1),
        "calmar": round(calmar, 2),
        "mcl": mcl,
        "win_days_pct": round((days_arr > 0).mean() * 100, 0),
        "worst_day": round(days_arr.min(), 0),
        "best_day": round(days_arr.max(), 0),
        "avg_win_r": round(tdf.loc[tdf["r"] > 0, "r"].mean(), 3) if (tdf["r"] > 0).any() else 0,
        "avg_loss_r": round(tdf.loc[tdf["r"] <= 0, "r"].mean(), 3) if (tdf["r"] <= 0).any() else 0,
        "cost_pct": round(tdf["cost"].mean() / tdf["risk"].mean() * 100, 1) if tdf["risk"].mean() > 0 else 0,
    }
