"""
Experiment 5: Comprehensive out-of-sample validation.

Tests:
1. 1m walk-forward: split 6mo data into 4mo train / 2mo test
2. 5m cross-validation: same params on 5-min bars (2 months)
3. 1h 2-year validation: same params on hourly bars (2 years)

If strategy only works on 6mo 1m data, it's overfit.
If it works across timeframes and time periods, edge is real.
"""

from __future__ import annotations
import importlib, sys, time
import backtrader as bt
import pandas as pd

CASH = 100_000
COMMISSION = 0.001


def run_test(csv_path: str, overrides: dict, label: str = "") -> dict:
    """Run strategy on arbitrary data with given overrides."""
    df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)

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
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - CASH) / CASH * 100
    sharpe_a = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_a.get("sharperatio", 0.0) or 0.0
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0.0) or 0.0
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    win_rate = won / total_trades * 100 if total_trades > 0 else 0.0
    gross_won = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gross_lost = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    pf = gross_won / gross_lost if gross_lost > 0 else 0.0

    return {
        "label": label,
        "bars": len(df),
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "total_trades": total_trades,
        "won": won, "lost": lost,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
    }


def split_csv(src: str, split_date: str, out_before: str, out_after: str):
    """Split CSV at a date boundary."""
    df = pd.read_csv(src, index_col="timestamp", parse_dates=True)
    before = df[df.index < split_date]
    after = df[df.index >= split_date]
    before.to_csv(out_before)
    after.to_csv(out_after)
    return len(before), len(after)


def print_result(r: dict):
    mark = "***" if r["profit_factor"] >= 1.0 else ("~" if r["profit_factor"] >= 0.8 else "")
    print(f"  {r['label']:<40} PF={r['profit_factor']:>6.3f}  Ret={r['total_return']:>6.2f}%  "
          f"WR={r['win_rate']:>5.1f}%  Trades={r['total_trades']:>4}  "
          f"({r['won']}W/{r['lost']}L)  DD={r['max_drawdown']:>5.2f}%  "
          f"Bars={r['bars']:>6}  {mark}")


def main():
    # v3 params (current strategy.py defaults)
    V3 = {}  # empty = use defaults from strategy.py

    # Also test without TP for comparison
    V3_NO_TP = {"TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False}

    print("=" * 120)
    print("COMPREHENSIVE OUT-OF-SAMPLE VALIDATION")
    print("=" * 120)

    # ═══ TEST 1: Walk-forward on 1m data ═══
    print("\n" + "─" * 120)
    print("TEST 1: Walk-Forward (1m data)")
    print("  Train: 2025-09-22 to 2026-01-21 (4 months)")
    print("  Test:  2026-01-22 to 2026-03-21 (2 months, OUT OF SAMPLE)")
    print("─" * 120)

    src = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
    split_date = "2026-01-22"
    train_path = "data/_tmp_1m_train.csv"
    test_path = "data/_tmp_1m_test.csv"
    n_train, n_test = split_csv(src, split_date, train_path, test_path)
    print(f"  Split: {n_train} train bars, {n_test} test bars\n")

    r1a = run_test(src, V3, "1m Full (6mo) — v3 with TP")
    print_result(r1a)
    r1b = run_test(src, V3_NO_TP, "1m Full (6mo) — v3 no TP")
    print_result(r1b)
    r1c = run_test(train_path, V3, "1m Train (4mo) — IN-SAMPLE")
    print_result(r1c)
    r1d = run_test(test_path, V3, "1m Test (2mo) — OUT-OF-SAMPLE")
    print_result(r1d)
    r1e = run_test(train_path, V3_NO_TP, "1m Train (4mo) — no TP")
    print_result(r1e)
    r1f = run_test(test_path, V3_NO_TP, "1m Test (2mo) — no TP")
    print_result(r1f)

    # ═══ TEST 2: 5-min bars (2 months) ═══
    print("\n" + "─" * 120)
    print("TEST 2: 5-min bars (2 months, different timeframe)")
    print("  Same v3 params, no re-tuning")
    print("─" * 120)

    r2a = run_test("data/QQQ_5Min_2mo.csv", V3, "5m (2mo) — v3 with TP")
    print_result(r2a)
    r2b = run_test("data/QQQ_5Min_2mo.csv", V3_NO_TP, "5m (2mo) — v3 no TP")
    print_result(r2b)

    # Also try with adapted params for 5m (5x fewer bars per day)
    V3_5M = {
        "LOSERS_MAX_BARS": 9,   # 45/5 = 9 five-min bars ≈ same real time
        "CHOP_BOX_MIN_BARS": 1,
    }
    r2c = run_test("data/QQQ_5Min_2mo.csv", V3_5M, "5m (2mo) — time-adapted")
    print_result(r2c)

    # ═══ TEST 3: 1-hour bars (2 years) — THE BIG TEST ═══
    print("\n" + "─" * 120)
    print("TEST 3: 1-hour bars (2 YEARS, biggest out-of-sample)")
    print("  Same v3 params applied to hourly data — zero re-tuning")
    print("─" * 120)

    r3a = run_test("data/QQQ_1Hour_2yr.csv", V3, "1h (2yr) — v3 raw")
    print_result(r3a)
    r3b = run_test("data/QQQ_1Hour_2yr.csv", V3_NO_TP, "1h (2yr) — v3 no TP")
    print_result(r3b)

    # Time-adapted for hourly (1h = 60x fewer bars than 1m)
    V3_1H = {
        "LOSERS_MAX_BARS": 8,     # 45 min-bars ≈ ~1 hour, so ~8 hourly bars = 1 day
        "CHOP_BOX_MIN_BARS": 1,
        "CHOP_SLOPE_AVG_PERIOD": 5,  # 20 min-bars ≈ 20min → ~5h for hourly
    }
    r3c = run_test("data/QQQ_1Hour_2yr.csv", V3_1H, "1h (2yr) — time-adapted losers/chop")
    print_result(r3c)

    V3_1H_NO_TP = {**V3_1H, "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False}
    r3d = run_test("data/QQQ_1Hour_2yr.csv", V3_1H_NO_TP, "1h (2yr) — time-adapted no TP")
    print_result(r3d)

    # Try wider chop on 1h
    for ct in [0.008, 0.010, 0.012, 0.015, 0.020, 0.030]:
        r = run_test("data/QQQ_1Hour_2yr.csv",
                     {**V3_1H_NO_TP, "CHOP_SLOPE_THRESHOLD": ct},
                     f"1h (2yr) — chop={ct}")
        print_result(r)

    # ═══ TEST 4: 1h split — in-sample vs out-of-sample ═══
    print("\n" + "─" * 120)
    print("TEST 4: 1h walk-forward (1yr train / 1yr test)")
    print("─" * 120)

    src_1h = "data/QQQ_1Hour_2yr.csv"
    train_1h = "data/_tmp_1h_train.csv"
    test_1h = "data/_tmp_1h_test.csv"
    n1, n2 = split_csv(src_1h, "2025-04-22", train_1h, test_1h)
    print(f"  Split: {n1} train bars, {n2} test bars\n")

    for ct in [0.008, 0.012, 0.015, 0.020]:
        params = {**V3_1H_NO_TP, "CHOP_SLOPE_THRESHOLD": ct}
        r_train = run_test(train_1h, params, f"1h Train (Y1) chop={ct}")
        r_test = run_test(test_1h, params, f"1h Test  (Y2) chop={ct}")
        print_result(r_train)
        print_result(r_test)
        print()

    # ═══ VERDICT ═══
    print("\n" + "=" * 120)
    print("VERDICT SUMMARY")
    print("=" * 120)

    all_results = [r1a, r1b, r1c, r1d, r1e, r1f, r2a, r2b, r2c, r3a, r3b, r3c, r3d]
    profitable = [r for r in all_results if r["profit_factor"] >= 1.0]
    losing = [r for r in all_results if r["profit_factor"] < 1.0 and r["total_trades"] > 0]

    print(f"\n  Profitable tests: {len(profitable)}/{len(all_results)}")
    for r in profitable:
        print(f"    ✓ {r['label']}: PF={r['profit_factor']:.3f}")

    if losing:
        print(f"\n  Losing tests: {len(losing)}/{len(all_results)}")
        for r in losing:
            print(f"    ✗ {r['label']}: PF={r['profit_factor']:.3f}")

    # Cleanup
    import os
    for f in [train_path, test_path, train_1h, test_1h]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
