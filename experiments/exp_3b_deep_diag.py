"""
Experiment 3B: Deep diagnosis — isolate WHY the new framework loses money.

Key questions:
1. Is the chop box hurting? (Test: disable it)
2. Is the exit system the problem? (Test: hold to close)
3. Is the entry itself broken? (Test: wider slopes, match old params)
4. What if we use the old strategy's slope=0.012 with the new framework?
"""

from __future__ import annotations
import importlib, sys, time
import backtrader as bt
import pandas as pd

DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001


def load_data():
    return pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)


def run_one(df, overrides: dict) -> dict:
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
        "total_return": round(total_return, 2), "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2), "total_trades": total_trades,
        "won": won, "lost": lost, "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
    }


def score(m):
    trades = m["total_trades"]
    if trades < 5: return -10.0
    sharpe_norm = max(0, min(1, (m["sharpe"] + 2) / 5))
    pf_norm = max(0, min(1, m["profit_factor"] / 3))
    ret_norm = max(0, min(1, (m["total_return"] + 20) / 70))
    wr_norm = max(0, min(1, m["win_rate"] / 100))
    dd_norm = max(0, min(1, 1 - m["max_drawdown"] / 30))
    raw = 0.30*sharpe_norm + 0.25*pf_norm + 0.20*ret_norm + 0.10*wr_norm + 0.15*dd_norm
    if trades < 20: mult = 0.2
    elif trades < 50: mult = 0.6
    elif trades <= 500: mult = 1.0
    else: mult = 0.8
    return round(raw * mult, 4)


def main():
    df = load_data()
    print(f"Data: {len(df)} bars\n")

    experiments = {}

    # ═══ DIAGNOSIS 1: Chop threshold (match old strategy's 0.012) ═══
    # Hypothesis: 0.005 too low → too many bars marked as "trend" → bad entries
    for t in [0.012, 0.015, 0.018, 0.020, 0.025, 0.030]:
        experiments[f"chop_{t}"] = {
            "CHOP_SLOPE_THRESHOLD": t,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,  # disable TP
            "ENABLE_ADDON": False,
        }

    # ═══ DIAGNOSIS 2: Disable chop box (just use slope for trend like old version) ═══
    # Trick: set chop_box_min_bars very high so breakout always happens immediately
    experiments["no_chopbox_slope012"] = {
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "CHOP_BOX_MIN_BARS": 0,  # immediate breakout
        "CHOP_SLOPE_AVG_PERIOD": 5,  # very short avg = responsive
        "INITIAL_STOP_ATR_MULT": 5.0,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    # ═══ DIAGNOSIS 3: Slope avg period sensitivity ═══
    for p in [5, 10, 15, 30]:
        experiments[f"slopeavg_{p}"] = {
            "CHOP_SLOPE_AVG_PERIOD": p,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══ DIAGNOSIS 4: EMA slope period ═══
    for sp in [3, 5, 8, 10]:
        experiments[f"slopeperiod_{sp}"] = {
            "EMA_SLOPE_PERIOD": sp,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══ DIAGNOSIS 5: EMA period itself ═══
    for ema in [10, 15, 20, 30]:
        experiments[f"ema_{ema}"] = {
            "EMA_PERIOD": ema,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══ DIAGNOSIS 6: Kill losers faster vs slower ═══
    for lb in [20, 30, 45, 60, 120, 999]:
        experiments[f"losers_{lb}"] = {
            "LOSERS_MAX_BARS": lb,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══ DIAGNOSIS 7: Best chop + try adding TP back ═══
    for act in [2.0, 3.0, 4.0, 5.0]:
        for tp1off in [0.20, 0.50, 1.00]:
            experiments[f"tp_act{act}_off{tp1off}"] = {
                "CHOP_SLOPE_THRESHOLD": 0.012,
                "INITIAL_STOP_ATR_MULT": 5.0,
                "TP_ACTIVATE_ATR": act,
                "TP1_CANDLE_OFFSET": tp1off,
                "TP2_EMA_ATR_MULT": 4.0,
                "TP3_EMA_ATR_MULT": 8.0,
                "ENABLE_ADDON": False,
            }

    # ═══ DIAGNOSIS 8: Stop size sweep with chop 0.012 ═══
    for s in [1.5, 2.5, 3.0, 5.0, 7.0, 10.0, 15.0]:
        experiments[f"stop_{s}_chop012"] = {
            "INITIAL_STOP_ATR_MULT": s,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══ DIAGNOSIS 9: Combined optimal ═══
    experiments["combo_A"] = {
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "INITIAL_STOP_ATR_MULT": 7.0,
        "PULLBACK_TOUCH_MULT": 1.2,
        "LOSERS_MAX_BARS": 45,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    experiments["combo_B"] = {
        "CHOP_SLOPE_THRESHOLD": 0.015,
        "INITIAL_STOP_ATR_MULT": 10.0,
        "PULLBACK_TOUCH_MULT": 1.5,
        "LOSERS_MAX_BARS": 60,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    experiments["combo_C"] = {
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "CHOP_SLOPE_AVG_PERIOD": 10,
        "INITIAL_STOP_ATR_MULT": 7.0,
        "PULLBACK_TOUCH_MULT": 1.5,
        "LOSERS_MAX_BARS": 90,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    # ═══ DIAGNOSIS 10: Addon with best base ═══
    experiments["combo_A_addon"] = {
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "INITIAL_STOP_ATR_MULT": 7.0,
        "PULLBACK_TOUCH_MULT": 1.2,
        "LOSERS_MAX_BARS": 45,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": True,
        "MAX_ADDONS": 1,
    }

    experiments["combo_D_tp_careful"] = {
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "INITIAL_STOP_ATR_MULT": 7.0,
        "PULLBACK_TOUCH_MULT": 1.2,
        "LOSERS_MAX_BARS": 45,
        "TP_ACTIVATE_ATR": 5.0,
        "TP1_CANDLE_OFFSET": 1.0,
        "TP2_EMA_ATR_MULT": 5.0,
        "TP3_EMA_ATR_MULT": 10.0,
        "ENABLE_ADDON": False,
    }

    # RUN
    print(f"Running {len(experiments)} experiments...")
    print("=" * 130)
    print(f"{'Name':<30} {'Score':>8} {'PF':>8} {'Return%':>8} {'Sharpe':>8} {'WR%':>6} {'Trades':>7} {'W/L':>8} {'MaxDD%':>8}")
    print("-" * 130)

    results_list = []
    for name, overrides in experiments.items():
        t0 = time.time()
        try:
            m = run_one(df, overrides)
            s = score(m)
            elapsed = time.time() - t0
            results_list.append({"name": name, "score": s, **m, "overrides": overrides})
            print(f"{name:<30} {s:>8.4f} {m['profit_factor']:>8.3f} {m['total_return']:>7.2f}% "
                  f"{m['sharpe']:>8.3f} {m['win_rate']:>5.1f}% {m['total_trades']:>6} "
                  f"{m['won']}/{m['lost']:>5} {m['max_drawdown']:>7.2f}%  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"{name:<30} {'ERROR':>8}  {str(e)[:60]}")
            results_list.append({"name": name, "score": -10, "error": str(e)})

    # SUMMARY
    print("\n" + "=" * 130)
    print("TOP 15 CONFIGURATIONS")
    print("=" * 130)

    valid = [r for r in results_list if r["score"] > -10]
    valid.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(valid[:15]):
        print(f"\n  #{i+1}: {r['name']}")
        print(f"      Score={r['score']:.4f} | PF={r['profit_factor']:.3f} | Return={r['total_return']:.2f}% | "
              f"WR={r['win_rate']:.1f}% | Trades={r['total_trades']} | MaxDD={r['max_drawdown']:.2f}%")
        params = ", ".join(f"{k}={v}" for k, v in r.get("overrides", {}).items())
        print(f"      Params: {params}")

    # PF > 1.0 check
    profitable = [r for r in valid if r.get("profit_factor", 0) >= 1.0]
    print(f"\n{'='*130}")
    print(f"PROFITABLE CONFIGS (PF >= 1.0): {len(profitable)}")
    if profitable:
        for r in profitable:
            print(f"  {r['name']}: PF={r['profit_factor']:.3f} Return={r['total_return']:.2f}%")
    else:
        print("  NONE — framework is not yet profitable. Root cause analysis needed.")

    # Best by PF
    by_pf = sorted(valid, key=lambda x: x.get("profit_factor", 0), reverse=True)
    print(f"\nTop 5 by Profit Factor:")
    for r in by_pf[:5]:
        print(f"  {r['name']}: PF={r['profit_factor']:.3f} Trades={r['total_trades']} WR={r['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
