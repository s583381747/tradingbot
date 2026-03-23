"""
Experiment 4: Layer TP and addon onto the proven profitable base.

FIXED: EMA_PERIOD=20 (core, immovable)

Profitable base (from Round 2):
  EMA20, slopeperiod=3, chop=0.012, stop=5.0, no TP, no addon
  → PF=1.41, 19 trades, Return=+0.55%

Goal: Add TP (30/40/30) and addon WITHOUT destroying the edge.
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
    avg_win = gross_won / won if won > 0 else 0
    avg_loss = gross_lost / lost if lost > 0 else 0
    return {
        "total_return": round(total_return, 2), "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2), "total_trades": total_trades,
        "won": won, "lost": lost, "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
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

    # ═══ Profitable base (FIXED: EMA20) ═══
    BASE = {
        "EMA_PERIOD": 20,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "INITIAL_STOP_ATR_MULT": 5.0,
    }

    experiments = {}

    # ═══ PHASE 0: Reproduce the profitable base ═══
    experiments["base_noTP_noAddon"] = {
        **BASE,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    # Also test losers variants
    for lb in [30, 45, 60, 90]:
        experiments[f"base_los{lb}"] = {
            **BASE,
            "LOSERS_MAX_BARS": lb,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # Also test higher chop thresholds
    for ct in [0.013, 0.014, 0.015, 0.018]:
        experiments[f"base_chop{ct}"] = {
            **BASE,
            "CHOP_SLOPE_THRESHOLD": ct,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # Stop sensitivity at EMA20
    for st in [1.5, 2.0, 2.5, 3.0, 5.0, 7.0]:
        experiments[f"base_stop{st}"] = {
            **BASE,
            "INITIAL_STOP_ATR_MULT": st,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══ PHASE 1: Add TP only (no addon) — gentle TP ═══
    # Test with high activation thresholds and generous offsets
    for act in [3.0, 4.0, 5.0, 8.0]:
        for tp1off in [0.30, 0.50, 1.00, 2.00]:
            experiments[f"tp_act{act}_off{tp1off}"] = {
                **BASE,
                "TP_ACTIVATE_ATR": act,
                "TP1_CANDLE_OFFSET": tp1off,
                "TP2_EMA_ATR_MULT": 4.0,
                "TP3_EMA_ATR_MULT": 8.0,
                "ENABLE_ADDON": False,
            }

    # TP2/TP3 width variations with best activation
    for tp2 in [2.0, 3.0, 4.0, 5.0]:
        for tp3 in [5.0, 6.0, 8.0, 10.0]:
            if tp3 <= tp2:
                continue
            experiments[f"tp_t2_{tp2}_t3_{tp3}"] = {
                **BASE,
                "TP_ACTIVATE_ATR": 5.0,
                "TP1_CANDLE_OFFSET": 1.00,
                "TP2_EMA_ATR_MULT": tp2,
                "TP3_EMA_ATR_MULT": tp3,
                "ENABLE_ADDON": False,
            }

    # ═══ PHASE 2: Add addon only (no TP) ═══
    for ma in [1, 2]:
        for amb in [1, 2, 3]:
            experiments[f"addon_max{ma}_min{amb}"] = {
                **BASE,
                "TP_ACTIVATE_ATR": 100.0,
                "ENABLE_ADDON": True,
                "MAX_ADDONS": ma,
                "ADDON_MIN_BARS": amb,
            }

    # ═══ PHASE 3: TP + addon together ═══
    # Use generous TP settings
    for act in [5.0, 8.0]:
        experiments[f"full_act{act}_addon1"] = {
            **BASE,
            "TP_ACTIVATE_ATR": act,
            "TP1_CANDLE_OFFSET": 1.00,
            "TP2_EMA_ATR_MULT": 4.0,
            "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": True,
            "MAX_ADDONS": 1,
        }
        experiments[f"full_act{act}_addon2"] = {
            **BASE,
            "TP_ACTIVATE_ATR": act,
            "TP1_CANDLE_OFFSET": 1.00,
            "TP2_EMA_ATR_MULT": 4.0,
            "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": True,
            "MAX_ADDONS": 2,
        }

    # ═══ PHASE 4: Best base + chop tuning + TP ═══
    for ct in [0.012, 0.015]:
        for act in [5.0, 100.0]:
            addon = act < 100
            experiments[f"opt_c{ct}_act{act}_a{addon}"] = {
                **BASE,
                "CHOP_SLOPE_THRESHOLD": ct,
                "TP_ACTIVATE_ATR": act,
                "TP1_CANDLE_OFFSET": 1.00,
                "TP2_EMA_ATR_MULT": 4.0,
                "TP3_EMA_ATR_MULT": 8.0,
                "ENABLE_ADDON": addon,
                "MAX_ADDONS": 1,
                "LOSERS_MAX_BARS": 45,
            }

    # ═══ PHASE 5: TP portion ratios ═══
    for t1, t2, t3 in [(0.20, 0.30, 0.50), (0.30, 0.40, 0.30), (0.10, 0.20, 0.70), (0.50, 0.30, 0.20)]:
        experiments[f"ratio_{int(t1*100)}_{int(t2*100)}_{int(t3*100)}"] = {
            **BASE,
            "TP_ACTIVATE_ATR": 5.0,
            "TP1_PCT": t1,
            "TP2_PCT": t2,
            "TP3_PCT": t3,
            "TP1_CANDLE_OFFSET": 1.00,
            "TP2_EMA_ATR_MULT": 4.0,
            "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": False,
        }

    # RUN
    print(f"Running {len(experiments)} experiments...")
    print("=" * 145)
    print(f"{'Name':<30} {'Score':>7} {'PF':>7} {'Ret%':>7} {'Sharpe':>7} {'WR%':>6} {'#':>5} {'W/L':>7} {'DD%':>6} {'AvgW':>8} {'AvgL':>8}")
    print("-" * 145)

    results_list = []
    for name, overrides in experiments.items():
        t0 = time.time()
        try:
            m = run_one(df, overrides)
            s = score(m)
            elapsed = time.time() - t0
            results_list.append({"name": name, "score": s, **m, "overrides": overrides})
            marker = " ***" if m["profit_factor"] >= 1.0 else (" ~" if m["profit_factor"] >= 0.8 else "")
            print(f"{name:<30} {s:>7.4f} {m['profit_factor']:>7.3f} {m['total_return']:>6.2f}% "
                  f"{m['sharpe']:>7.3f} {m['win_rate']:>5.1f}% {m['total_trades']:>4} "
                  f"{m['won']}/{m['lost']:<4} {m['max_drawdown']:>5.2f}% "
                  f"${m['avg_win']:>7.0f} ${m['avg_loss']:>7.0f}  ({elapsed:.1f}s){marker}")
        except Exception as e:
            print(f"{name:<30} {'ERR':>7}  {str(e)[:50]}")

    # SUMMARY
    valid = [r for r in results_list if r["score"] > -10]
    profitable = [r for r in valid if r.get("profit_factor", 0) >= 1.0]
    near = [r for r in valid if 0.8 <= r.get("profit_factor", 0) < 1.0]

    print(f"\n{'='*145}")
    print(f"PROFITABLE (PF >= 1.0): {len(profitable)}")
    for r in sorted(profitable, key=lambda x: x["profit_factor"], reverse=True):
        print(f"  {r['name']:<30} PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
              f"Trades={r['total_trades']} WR={r['win_rate']:.1f}% AvgW=${r['avg_win']:.0f} AvgL=${r['avg_loss']:.0f}")
        key_params = {k: v for k, v in r["overrides"].items()
                     if k not in ("EMA_PERIOD", "EMA_SLOW_PERIOD", "ATR_PERIOD")}
        print(f"    {key_params}")

    print(f"\nNEAR-PROFITABLE (0.8 <= PF < 1.0): {len(near)}")
    for r in sorted(near, key=lambda x: x["profit_factor"], reverse=True)[:10]:
        print(f"  {r['name']:<30} PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% Trades={r['total_trades']}")

    print(f"\nTOP 10 BY SCORE:")
    for i, r in enumerate(sorted(valid, key=lambda x: x["score"], reverse=True)[:10]):
        pf_mark = " ***" if r["profit_factor"] >= 1.0 else ""
        print(f"  #{i+1}: {r['name']:<28} Score={r['score']:.4f} PF={r['profit_factor']:.3f} "
              f"Trades={r['total_trades']} Ret={r['total_return']:.2f}%{pf_mark}")

    # TP impact analysis
    base_r = next((r for r in valid if r["name"] == "base_noTP_noAddon"), None)
    if base_r:
        print(f"\n{'='*145}")
        print(f"TP IMPACT ANALYSIS (vs base PF={base_r['profit_factor']:.3f})")
        print("-" * 80)
        tp_results = [r for r in valid if r["name"].startswith("tp_act")]
        tp_results.sort(key=lambda x: x["profit_factor"], reverse=True)
        for r in tp_results[:10]:
            delta = r["profit_factor"] - base_r["profit_factor"]
            sign = "+" if delta >= 0 else ""
            print(f"  {r['name']:<30} PF={r['profit_factor']:.3f} ({sign}{delta:.3f}) Trades={r['total_trades']}")


if __name__ == "__main__":
    main()
