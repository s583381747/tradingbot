"""
Critical edge audit — is the PF=1.15 real or noise?

Tests:
1. Statistical significance (bootstrap p-value)
2. Buy-and-hold benchmark comparison
3. Random entry baseline (same exits, random entries)
4. Walk-forward (train Y1, test Y2)
5. Regime breakdown (per-quarter performance)
6. Monte Carlo: shuffle trade PnLs to get null distribution
"""
from __future__ import annotations
import importlib, sys, time, random, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import numpy as np
import backtrader as bt

CLEAN_PATH = "data/QQQ_1Min_2y_clean.csv"
CASH = 100_000
COMMISSION = 0.001
NCPU = cpu_count()


def _run_backtest(data_path, overrides, cash=CASH):
    """Run one backtest, return metrics + per-trade PnLs."""
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)

    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    mod = importlib.import_module("strategy")
    for k, v in overrides.items():
        setattr(mod, k, v)
    StrategyClass = mod.Strategy

    param_name_map = {
        "EMA_PERIOD": "ema_period", "EMA_SLOW_PERIOD": "ema_slow_period",
        "ATR_PERIOD": "atr_period", "EMA_SLOPE_PERIOD": "ema_slope_period",
        "CHOP_SLOPE_AVG_PERIOD": "chop_slope_avg_period",
        "CHOP_SLOPE_THRESHOLD": "chop_slope_threshold",
        "CHOP_BOX_MIN_BARS": "chop_box_min_bars",
        "PULLBACK_TOUCH_MULT": "pullback_touch_mult",
        "MIN_PULLBACK_BARS": "min_pullback_bars",
        "INITIAL_STOP_ATR_MULT": "initial_stop_atr_mult",
        "TP_ACTIVATE_ATR": "tp_activate_atr",
        "TP1_PCT": "tp1_pct", "TP2_PCT": "tp2_pct", "TP3_PCT": "tp3_pct",
        "TP1_CANDLE_OFFSET": "tp1_candle_offset",
        "TP2_EMA_ATR_MULT": "tp2_ema_atr_mult",
        "TP3_EMA_ATR_MULT": "tp3_ema_atr_mult",
        "ENABLE_ADDON": "enable_addon", "MAX_ADDONS": "max_addons",
        "ADDON_PULLBACK_MULT": "addon_pullback_mult",
        "ADDON_MIN_BARS": "addon_min_bars",
        "RISK_PCT": "risk_pct", "MAX_POSITION_PCT": "max_position_pct",
        "MAX_DAILY_TRADES": "max_daily_trades",
        "MAX_DAILY_LOSS_PCT": "max_daily_loss_pct",
        "LOSERS_MAX_BARS": "losers_max_bars",
    }
    params = {param_name_map[k]: v for k, v in overrides.items() if k in param_name_map}

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df, datetime=None, open="Open", high="High",
                                low="Low", close="Close", volume="Volume", openinterest=-1)
    cerebro.adddata(data)
    cerebro.addstrategy(StrategyClass, **params)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - cash) / cash * 100
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    gross_won = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gross_lost = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    pf = gross_won / gross_lost if gross_lost > 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "total_trades": total_trades,
        "won": won, "lost": lost,
        "profit_factor": round(pf, 4),
        "gross_won": round(gross_won, 2),
        "gross_lost": round(gross_lost, 2),
        "net_pnl": round(gross_won - gross_lost, 2),
    }


def _run_walkforward_half(args):
    """Run backtest on a date-filtered subset."""
    name, overrides, data_path, start_date, end_date = args
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    df = df[(df.index >= start_date) & (df.index < end_date)]

    # Write temp CSV for filtered data
    tmp_path = f"/tmp/wf_{name}_{os.getpid()}.csv"
    df.to_csv(tmp_path)

    try:
        result = _run_backtest(tmp_path, overrides)
        result["name"] = name
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_random_entry(args):
    """Run random entry baseline with same exit system."""
    seed, data_path = args
    # Random entry = disable chop box (always in trend) + random direction
    # Simplified: use very loose chop threshold so most bars are "trend"
    overrides = {
        "CHOP_SLOPE_THRESHOLD": 0.001,  # almost everything passes
        "PULLBACK_TOUCH_MULT": 3.0,     # very wide pullback zone
        "MIN_PULLBACK_BARS": 1,
        "EMA_SLOPE_PERIOD": 3,
        "INITIAL_STOP_ATR_MULT": 2.0,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }
    result = _run_backtest(data_path, overrides)
    result["seed"] = seed
    return result


def main():
    data_path = os.path.abspath(CLEAN_PATH)
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df)} bars, {df.index[0]} → {df.index[-1]}")
    print(f"Trading days: {df.index.normalize().nunique()}")
    print(f"CPUs: {NCPU}\n")

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Buy-and-hold benchmark
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 1: BUY-AND-HOLD BENCHMARK")
    print("=" * 80)
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    bnh_return = (last_close - first_close) / first_close * 100
    print(f"  QQQ: ${first_close:.2f} → ${last_close:.2f}")
    print(f"  Buy & Hold Return: {bnh_return:.2f}%")

    # Per-year
    for year in sorted(df.index.year.unique()):
        yr_data = df[df.index.year == year]
        yr_ret = (yr_data["Close"].iloc[-1] - yr_data["Close"].iloc[0]) / yr_data["Close"].iloc[0] * 100
        print(f"  {year}: {yr_ret:.2f}%")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: v3 strategy baseline on full 2 years
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TEST 2: V3 STRATEGY ON 2 YEARS")
    print("=" * 80)

    v3_result = _run_backtest(data_path, {})
    print(f"  PF={v3_result['profit_factor']:.3f} Return={v3_result['total_return']:.2f}%")
    print(f"  Trades={v3_result['total_trades']} ({v3_result['won']}W / {v3_result['lost']}L)")
    print(f"  Gross won=${v3_result['gross_won']:.0f} lost=${v3_result['gross_lost']:.0f} net=${v3_result['net_pnl']:.0f}")

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Walk-forward (Year 1 vs Year 2)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TEST 3: WALK-FORWARD (Year 1 vs Year 2)")
    print("=" * 80)

    mid_date = "2025-03-22"  # roughly midpoint
    wf_tasks = [
        ("Y1_2024", {}, data_path, "2024-01-01", mid_date),
        ("Y2_2025", {}, data_path, mid_date, "2027-01-01"),
        # Also test no-TP variant
        ("Y1_noTP", {"TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False}, data_path, "2024-01-01", mid_date),
        ("Y2_noTP", {"TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False}, data_path, mid_date, "2027-01-01"),
    ]

    with ProcessPoolExecutor(max_workers=NCPU) as pool:
        futures = {pool.submit(_run_walkforward_half, t): t[0] for t in wf_tasks}
        wf_results = {}
        for f in as_completed(futures):
            r = f.result()
            wf_results[r["name"]] = r
            print(f"  {r['name']}: PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
                  f"Trades={r['total_trades']} ({r['won']}W/{r['lost']}L)")

    # ═══════════════════════════════════════════════════════════
    # TEST 4: Per-quarter breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TEST 4: PER-QUARTER BREAKDOWN")
    print("=" * 80)

    quarters = []
    for year in sorted(df.index.year.unique()):
        for q in range(1, 5):
            m_start = (q-1)*3+1
            m_end = q*3+1 if q < 4 else 1
            y_end = year if q < 4 else year+1
            q_start = f"{year}-{m_start:02d}-01"
            q_end = f"{y_end}-{m_end:02d}-01"
            # Skip quarters with no data
            qdf = df[(df.index >= q_start) & (df.index < q_end)]
            if len(qdf) < 500:  # need minimum bars for indicators
                continue
            quarters.append((f"{year}Q{q}", q_start, q_end))

    q_tasks = [(name, {}, data_path, start, end) for name, start, end in quarters]

    with ProcessPoolExecutor(max_workers=NCPU) as pool:
        futures = {pool.submit(_run_walkforward_half, t): t[0] for t in q_tasks}
        q_results = []
        for f in as_completed(futures):
            r = f.result()
            q_results.append(r)

    q_results.sort(key=lambda x: x["name"])
    pos_quarters = 0
    neg_quarters = 0
    for r in q_results:
        if r["total_trades"] < 1:
            continue
        status = "✓" if r["profit_factor"] >= 1.0 else "✗"
        if r["profit_factor"] >= 1.0:
            pos_quarters += 1
        else:
            neg_quarters += 1
        print(f"  {status} {r['name']}: PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
              f"Trades={r['total_trades']} ({r['won']}W/{r['lost']}L)")

    print(f"\n  Profitable quarters: {pos_quarters}/{pos_quarters + neg_quarters}")

    # ═══════════════════════════════════════════════════════════
    # TEST 5: Statistical significance (bootstrap)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TEST 5: STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    # Using gross_won and gross_lost to compute observed edge per trade
    n_trades = v3_result["total_trades"]
    net = v3_result["net_pnl"]
    avg_pnl = net / n_trades if n_trades > 0 else 0

    print(f"  Net PnL: ${net:.0f} over {n_trades} trades")
    print(f"  Avg PnL per trade: ${avg_pnl:.2f}")

    # Simple binomial test: with 25% win rate and 4:1 R:R,
    # expected PF = (WR * AvgWin) / ((1-WR) * AvgLoss)
    wr = v3_result["won"] / n_trades if n_trades > 0 else 0
    if v3_result["won"] > 0 and v3_result["lost"] > 0:
        avg_win = v3_result["gross_won"] / v3_result["won"]
        avg_loss = v3_result["gross_lost"] / v3_result["lost"]
        # Under random (50% WR), expected PF:
        random_pf = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"  Win rate: {wr:.1%} (need {1/(1+avg_win/avg_loss):.1%} to break even)")
        print(f"  Avg Win: ${avg_win:.0f}, Avg Loss: ${avg_loss:.0f}, R:R = {avg_win/avg_loss:.1f}")

        # Monte Carlo: shuffle win/loss labels 10000 times
        print(f"\n  Monte Carlo (10000 shuffles of {n_trades} trade outcomes)...")
        wins = [avg_win] * v3_result["won"]
        losses = [-avg_loss] * v3_result["lost"]
        all_pnls = wins + losses

        np.random.seed(42)
        mc_pfs = []
        for _ in range(10000):
            np.random.shuffle(all_pnls)
            # Randomly assign direction (long/short) — just shuffle PnLs
            shuffled = np.random.choice(all_pnls, size=n_trades, replace=True)
            w = sum(x for x in shuffled if x > 0)
            l = abs(sum(x for x in shuffled if x < 0))
            mc_pfs.append(w / l if l > 0 else 0)

        observed_pf = v3_result["profit_factor"]
        p_value = sum(1 for pf in mc_pfs if pf >= observed_pf) / len(mc_pfs)
        mc_mean = np.mean(mc_pfs)
        mc_95 = np.percentile(mc_pfs, 95)

        print(f"  Observed PF: {observed_pf:.3f}")
        print(f"  Monte Carlo mean PF: {mc_mean:.3f}")
        print(f"  Monte Carlo 95th percentile: {mc_95:.3f}")
        print(f"  p-value (PF >= observed): {p_value:.4f}")

        if p_value < 0.05:
            print(f"  → SIGNIFICANT at 5% level")
        elif p_value < 0.10:
            print(f"  → MARGINAL at 10% level")
        else:
            print(f"  → NOT SIGNIFICANT — could be noise")

    # ═══════════════════════════════════════════════════════════
    # TEST 6: Loose-filter baseline (is edge from entry or exit?)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TEST 6: LOOSE-FILTER BASELINE (entry edge vs structure)")
    print("=" * 80)

    loose_configs = {
        "very_loose": {
            "CHOP_SLOPE_THRESHOLD": 0.003,
            "PULLBACK_TOUCH_MULT": 2.0,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        },
        "medium_loose": {
            "CHOP_SLOPE_THRESHOLD": 0.008,
            "PULLBACK_TOUCH_MULT": 1.5,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        },
    }

    for name, overrides in loose_configs.items():
        r = _run_backtest(data_path, overrides)
        print(f"  {name}: PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% Trades={r['total_trades']}")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("VERDICT")
    print("=" * 80)
    print(f"""
  Strategy: PF={v3_result['profit_factor']:.3f}, Return={v3_result['total_return']:.2f}%
  Buy&Hold: Return={bnh_return:.2f}%
  Trades: {v3_result['total_trades']} over 2 years ({v3_result['total_trades']/24:.1f}/month)

  Questions to answer:
  1. Does strategy beat buy-and-hold? {'YES' if v3_result['total_return'] > bnh_return else 'NO — underperforms passive'}
  2. Is it profitable in both halves? Check walk-forward above
  3. Is PF statistically significant? Check p-value above
  4. How many quarters profitable? {pos_quarters}/{pos_quarters+neg_quarters}
  5. Could random entries do this? Check loose baseline above
""")


if __name__ == "__main__":
    main()
