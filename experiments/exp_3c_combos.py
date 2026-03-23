"""
Experiment 3C: Combine the best findings from 3a and 3b.

Key discoveries so far:
- slopeperiod=3 is profitable (PF 1.41, 19 trades)
- chop_threshold 0.015-0.018 is profitable (PF 1.56-3.81, 7-10 trades)
- EMA_PERIOD=10 gives most trades (64) with best composite score
- losers_max_bars=30 gets PF to 0.976 (almost breakeven)
- TP system hurts in ALL configs
- TIGHTER stop is better with chop box (opposite of old finding!)

This round: combine these discoveries to find a profitable config with enough trades.
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

    # Common "clean" base: no TP, no addon, tight losers
    CLEAN = {
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    experiments = {}

    # ═══ BLOCK 1: EMA10 + slopeperiod=3 combos (two best individual findings) ═══
    for chop in [0.008, 0.010, 0.012, 0.013, 0.014, 0.015]:
        for losers in [20, 30, 45]:
            experiments[f"e10_sp3_c{chop}_l{losers}"] = {
                **CLEAN,
                "EMA_PERIOD": 10,
                "EMA_SLOPE_PERIOD": 3,
                "CHOP_SLOPE_THRESHOLD": chop,
                "INITIAL_STOP_ATR_MULT": 1.5,
                "LOSERS_MAX_BARS": losers,
            }

    # ═══ BLOCK 2: EMA10 only (best score finder) ═══
    for chop in [0.008, 0.010, 0.012, 0.015]:
        for stop in [1.5, 2.5, 5.0]:
            experiments[f"e10_c{chop}_s{stop}"] = {
                **CLEAN,
                "EMA_PERIOD": 10,
                "CHOP_SLOPE_THRESHOLD": chop,
                "INITIAL_STOP_ATR_MULT": stop,
                "LOSERS_MAX_BARS": 30,
            }

    # ═══ BLOCK 3: EMA15 + slopeperiod=3 ═══
    for chop in [0.010, 0.012, 0.015]:
        experiments[f"e15_sp3_c{chop}"] = {
            **CLEAN,
            "EMA_PERIOD": 15,
            "EMA_SLOPE_PERIOD": 3,
            "CHOP_SLOPE_THRESHOLD": chop,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ═══ BLOCK 4: Pullback zone sweep with EMA10 ═══
    for pb in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        experiments[f"e10_pb{pb}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "PULLBACK_TOUCH_MULT": pb,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ═══ BLOCK 5: Min pullback bars ═══
    for mpb in [1, 2, 3]:
        experiments[f"e10_minpb{mpb}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "MIN_PULLBACK_BARS": mpb,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ═══ BLOCK 6: Chop box min bars ═══
    for cmb in [2, 3, 5, 10, 15]:
        experiments[f"e10_chopmin{cmb}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "CHOP_BOX_MIN_BARS": cmb,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ═══ BLOCK 7: Daily limits ═══
    for dt in [3, 6, 10]:
        experiments[f"e10_daily{dt}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "MAX_DAILY_TRADES": dt,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
        }

    # ═══ BLOCK 8: Best combos — aggressive ═══
    experiments["best_A"] = {
        **CLEAN,
        "EMA_PERIOD": 10,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.012,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 1.0,
        "INITIAL_STOP_ATR_MULT": 1.5,
        "LOSERS_MAX_BARS": 30,
    }

    experiments["best_B"] = {
        **CLEAN,
        "EMA_PERIOD": 10,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.010,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 1.5,
        "INITIAL_STOP_ATR_MULT": 1.5,
        "LOSERS_MAX_BARS": 25,
    }

    experiments["best_C"] = {
        **CLEAN,
        "EMA_PERIOD": 10,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.013,
        "CHOP_BOX_MIN_BARS": 3,
        "PULLBACK_TOUCH_MULT": 1.2,
        "MIN_PULLBACK_BARS": 2,
        "INITIAL_STOP_ATR_MULT": 1.5,
        "LOSERS_MAX_BARS": 30,
    }

    experiments["best_D_wide_net"] = {
        **CLEAN,
        "EMA_PERIOD": 10,
        "EMA_SLOPE_PERIOD": 3,
        "CHOP_SLOPE_THRESHOLD": 0.008,
        "CHOP_BOX_MIN_BARS": 2,
        "PULLBACK_TOUCH_MULT": 2.0,
        "INITIAL_STOP_ATR_MULT": 1.5,
        "LOSERS_MAX_BARS": 20,
        "MAX_DAILY_TRADES": 10,
    }

    # ═══ BLOCK 9: Add TP back to best base (if base is profitable) ═══
    for name_base, base_overrides in [
        ("best_A", experiments["best_A"]),
    ]:
        for act in [3.0, 5.0]:
            override = {**base_overrides}
            override["TP_ACTIVATE_ATR"] = act
            override["TP1_CANDLE_OFFSET"] = 0.50
            override["TP2_EMA_ATR_MULT"] = 3.0
            override["TP3_EMA_ATR_MULT"] = 6.0
            experiments[f"{name_base}_tp{act}"] = override

    # ═══ BLOCK 10: Add addon to best base ═══
    for name_base, base_overrides in [
        ("best_A", experiments["best_A"]),
    ]:
        override = {**base_overrides}
        override["ENABLE_ADDON"] = True
        override["MAX_ADDONS"] = 1
        experiments[f"{name_base}_addon"] = override

    # ═══ BLOCK 11: EMA slow period ═══
    for slow in [30, 40, 50, 80]:
        experiments[f"e10_slow{slow}"] = {
            **CLEAN,
            "EMA_PERIOD": 10,
            "EMA_SLOW_PERIOD": slow,
            "CHOP_SLOPE_THRESHOLD": 0.012,
            "INITIAL_STOP_ATR_MULT": 1.5,
            "LOSERS_MAX_BARS": 30,
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
            # Highlight profitable
            marker = " ***" if m["profit_factor"] >= 1.0 else ""
            print(f"{name:<30} {s:>8.4f} {m['profit_factor']:>8.3f} {m['total_return']:>7.2f}% "
                  f"{m['sharpe']:>8.3f} {m['win_rate']:>5.1f}% {m['total_trades']:>6} "
                  f"{m['won']}/{m['lost']:>5} {m['max_drawdown']:>7.2f}%  ({elapsed:.1f}s){marker}")
        except Exception as e:
            print(f"{name:<30} {'ERROR':>8}  {str(e)[:60]}")

    # SUMMARY
    valid = [r for r in results_list if r.get("score", -10) > -10]
    profitable = [r for r in valid if r.get("profit_factor", 0) >= 1.0]

    print(f"\n{'='*130}")
    print(f"PROFITABLE CONFIGS (PF >= 1.0): {len(profitable)}")
    print("=" * 130)
    if profitable:
        profitable.sort(key=lambda x: x["profit_factor"], reverse=True)
        for r in profitable:
            print(f"\n  {r['name']}")
            print(f"    PF={r['profit_factor']:.3f} | Return={r['total_return']:.2f}% | WR={r['win_rate']:.1f}% | "
                  f"Trades={r['total_trades']} ({r['won']}W/{r['lost']}L) | MaxDD={r['max_drawdown']:.2f}% | Score={r['score']:.4f}")
            params = ", ".join(f"{k}={v}" for k, v in r["overrides"].items()
                             if k not in ("TP_ACTIVATE_ATR", "ENABLE_ADDON") or v != 100.0)
            print(f"    Params: {params}")

    print(f"\n{'='*130}")
    print("TOP 15 BY SCORE")
    print("=" * 130)
    valid.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(valid[:15]):
        marker = " <<<" if r.get("profit_factor", 0) >= 1.0 else ""
        print(f"  #{i+1}: {r['name']}: Score={r['score']:.4f} PF={r['profit_factor']:.3f} "
              f"Ret={r['total_return']:.2f}% WR={r['win_rate']:.1f}% Trades={r['total_trades']} DD={r['max_drawdown']:.2f}%{marker}")

    # Best by PF with >=20 trades (meaningful sample)
    enough = [r for r in valid if r.get("total_trades", 0) >= 20]
    print(f"\n{'='*130}")
    print("TOP 10 BY PF (trades >= 20)")
    print("=" * 130)
    enough.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)
    for r in enough[:10]:
        print(f"  {r['name']}: PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
              f"WR={r['win_rate']:.1f}% Trades={r['total_trades']} Score={r['score']:.4f}")


if __name__ == "__main__":
    main()
